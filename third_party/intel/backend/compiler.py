from triton.backends.compiler import BaseBackend
from triton._C.libtriton import ir, passes, llvm, intel

from dataclasses import dataclass
import functools
from typing import Any, Dict, Tuple
from types import ModuleType
import hashlib
import re
import tempfile
import signal
import os
import shutil
import subprocess
from pathlib import Path


@functools.lru_cache()
def _path_to_binary(binary: str):
    paths = [
        os.environ.get(f"TRITON_{binary.upper().replace('-', '_')}_PATH", ""),
        os.path.join(os.path.dirname(__file__), "bin", binary),
        shutil.which(binary) or "",
    ]

    for p in paths:
        bin = p.split(" ")[0]
        if os.path.exists(bin) and os.path.isfile(bin):
            result = subprocess.check_output([bin, "--version"], stderr=subprocess.STDOUT)
            if result is not None:
                version = re.search(r".*SPIRV-Tools v(\d+\.\d+).*", result.decode("utf-8"), flags=re.MULTILINE)
                if version is not None:
                    return p, version.group(1)
    raise RuntimeError(f"Cannot find {binary}")


@dataclass
class XPUOptions:
    num_warps: int = 4
    num_ctas: int = 1
    num_stages: int = 2
    cluster_dims: tuple = (1, 1, 1)
    threads_per_warp: int = 32
    optimize_epilogue: bool = False
    enable_fp_fusion: bool = True
    supported_fp8_dtypes: Tuple[str] = ("fp8e5", "fp8e4nv", "fp8e4b15")
    deprecated_fp8_dtypes: Tuple[str] = ()
    default_dot_input_precision: str = "tf32"
    allowed_dot_input_precisions: Tuple[str] = ("tf32", "tf32x3", "ieee")
    allow_fp8e4nv: bool = False
    allow_fp8e4b15: bool = True
    grf_mode: tuple = ('small', 'large', 'auto', 'default')
    max_num_imprecise_acc_default: int = 0  # `max_num_imprecise_acc` only applies to fp8 -> fp32 dot on sm_90 for cuda
    extern_libs: dict = None
    debug: bool = False
    generate_native_code: bool = True
    backend_name: str = 'intel'

    def __post_init__(self):
        default_libdir = Path(__file__).parent / 'lib'
        extern_libs = {} if self.extern_libs is None else dict(self.extern_libs)
        if not extern_libs.get('libdevice', None):
            extern_libs['libdevice'] = os.getenv("TRITON_LIBDEVICE_PATH",
                                                 str(default_libdir / 'libsycl-spir64-unknown-unknown.bc'))
        object.__setattr__(self, 'extern_libs', tuple(extern_libs.items()))
        if self.num_warps <= 0 or (self.num_warps & (self.num_warps - 1)) != 0:
            raise AssertionError("num_warps must be a power of 2")
        generate_native_code_env = os.getenv("TRITON_XPU_GEN_NATIVE_CODE")
        if generate_native_code_env:
            self.generate_native_code = bool(generate_native_code_env)
        else:
            os.putenv("TRITON_XPU_GEN_NATIVE_CODE", str(self.generate_native_code))

    def hash(self):
        key = '_'.join([f'{name}-{val}' for name, val in self.__dict__.items()])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


def min_dot_size(device_props: dict):
    # (M, N, K)
    # M: repeatCount. 1,2,4,8
    # N: executionSize. 16 for PVC, 8 for ATS
    # K: systolicDepth x opsPerChan. systolicDepth must be 8

    # default 8 because 1,2,4 is not supported by our backend now.
    repeat_count = 8
    sdepth = 8
    exec_size = min(device_props["sub_group_sizes"])

    def get_ops_per_channel(lhs_type, rhs_type):
        l_bitwidth = lhs_type.scalar.primitive_bitwidth
        r_bitwidth = rhs_type.scalar.primitive_bitwidth
        max_ops_per_chan = 32 / max(l_bitwidth, r_bitwidth)
        return min(8, max_ops_per_chan)

    return lambda lhs_type, rhs_type: (repeat_count, exec_size, sdepth * get_ops_per_channel(lhs_type, rhs_type))


class XPUBackend(BaseBackend):

    # AdvancedPath pass pipeline for kernels using block pointers.
    class AdvancedPath:

        @staticmethod
        def make_ttgir(mod, metadata, opt):
            pm = ir.pass_manager(mod.context)
            pm.enable_debug()

            intel.passes.ttir.add_convert_to_ttgpuir_warp(pm, opt.num_warps)
            inject_split_barriers = False
            intel.passes.ttgpuir.add_prefetch_block(pm, opt.num_stages, inject_split_barriers)
            intel.passes.ttgpuir.add_distribute_to_warps(pm)
            passes.common.add_canonicalizer(pm)
            passes.common.add_cse(pm)
            intel.passes.ttgpuir.add_match_target_size(pm)
            passes.common.add_canonicalizer(pm)
            passes.common.add_cse(pm)
            intel.passes.ttgpuir.add_schedule_load(pm)
            passes.common.add_symbol_dce(pm)
            pm.run(mod)
            return mod

    @staticmethod
    def supports_target(target: tuple):
        return target.backend == 'xpu'

    def __init__(self, target: tuple) -> None:
        super().__init__(target)
        if not isinstance(target.arch, dict):
            raise TypeError("target.arch is not a dict")
        self.properties = self.parse_target(target.arch)
        self.binary_ext = "spv"

    def parse_target(self, tgt_prop) -> dict:
        dev_prop = {}
        dev_prop['name'] = tgt_prop.get('name', 'xpu')
        dev_prop['platform_name'] = tgt_prop.get('platform_name', None)
        dev_prop['vendor'] = tgt_prop.get('vendor', None)
        dev_prop['version'] = tgt_prop.get('version', None)
        dev_prop['gpu_eu_count'] = tgt_prop.get('gpu_eu_count', None)
        dev_prop['gpu_subslice_count'] = tgt_prop.get('gpu_subslice_count', None)
        dev_prop['max_work_group_size'] = tgt_prop.get('max_work_group_size', None)
        dev_prop['max_num_sub_groups'] = tgt_prop.get('max_num_sub_groups', None)
        dev_prop['sub_group_sizes'] = tgt_prop.get('sub_group_sizes', None)
        dev_prop['has_fp64'] = tgt_prop.get('has_fp64', None)
        dev_prop['has_subgroup_matrix_multiply_accumulate'] = tgt_prop.get('has_subgroup_matrix_multiply_accumulate',
                                                                           False)
        dev_prop['has_subgroup_matrix_multiply_accumulate_tensor_float32'] = tgt_prop.get(
            'has_subgroup_matrix_multiply_accumulate_tensor_float32', False)
        dev_prop['has_subgroup_2d_block_io'] = tgt_prop.get('has_subgroup_2d_block_io', False)
        dev_prop['has_bfloat16_conversions'] = tgt_prop.get('has_bfloat16_conversions', True)
        return dev_prop

    def parse_options(self, opts) -> Any:
        args = {k: opts[k] for k in XPUOptions.__dataclass_fields__.keys() if k in opts}
        args["allow_fp8e4nv"] = True
        return XPUOptions(**args)

    def pack_metadata(self, metadata):
        return metadata

    def get_codegen_implementation(self):
        from triton.language.extra.intel import convert_custom_float8
        codegen_fns = {}
        codegen_fns["convert_custom_types"] = convert_custom_float8
        codegen_fns["min_dot_size"] = min_dot_size(self.properties)
        return codegen_fns

    def get_module_map(self) -> Dict[str, ModuleType]:
        from triton.language.extra.intel import libdevice
        return {"triton.language.extra.libdevice": libdevice}

    def load_dialects(self, ctx):
        intel.load_dialects(ctx)

    @staticmethod
    def make_ttir(mod, metadata, opt):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_combine(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_licm(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod)
        return mod

    @staticmethod
    def make_ttgir(mod, metadata, opt, properties):
        cluster_info = intel.ClusterInfo()
        if opt.cluster_dims is not None:
            cluster_info.clusterDimX = opt.cluster_dims[0]
            cluster_info.clusterDimY = opt.cluster_dims[1]
            cluster_info.clusterDimZ = opt.cluster_dims[2]

        # Annotate module with information required by subsequent transformations.
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        intel.passes.ttgpuir.add_triton_annotate_module(pm, min(properties["sub_group_sizes"]),
                                                        properties["has_subgroup_2d_block_io"],
                                                        properties["has_subgroup_matrix_multiply_accumulate"],
                                                        properties["has_bfloat16_conversions"], opt.threads_per_warp)
        pm.run(mod)

        # Overwrite the threads_per_warp option with the module annotation.
        opt.threads_per_warp = ir.ttgpuir.get_threads_per_warp(mod)

        # Run the TTIR -> TTGIR pass pipeline.
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        if (properties["has_subgroup_2d_block_io"] and properties["has_subgroup_matrix_multiply_accumulate"]
                and os.getenv("TRITON_INTEL_ADVANCED_PATH", "0") == "1"):
            return XPUBackend.AdvancedPath.make_ttgir(mod, metadata, opt)

        passes.ttir.add_convert_to_ttgpuir(pm, "xpu", opt.num_warps, opt.threads_per_warp, opt.num_ctas)
        intel.passes.ttgpuir.add_accelerate_matmul(pm)
        intel.passes.ttgpuir.add_remove_layout_conversions(pm)
        intel.passes.ttgpuir.add_materialize_block_pointer(pm)
        intel.passes.ttgpuir.add_rewrite_tensor_pointer(pm)
        intel.passes.ttgpuir.add_pipeline(pm, opt.num_stages, False)

        passes.ttgpuir.add_coalesce(pm)
        intel.passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_thread_locality(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm, True)
        passes.common.add_cse(pm)
        passes.ttgpuir.add_prefetch(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm, True)
        intel.passes.ttgpuir.add_remove_layout_conversions(pm)
        intel.passes.ttgpuir.add_reduce_data_duplication(pm)
        passes.ttgpuir.add_reorder_instructions(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.common.add_canonicalizer(pm)
        pm.run(mod)
        metadata["cluster_dims"] = (cluster_info.clusterDimX, cluster_info.clusterDimY, cluster_info.clusterDimZ)
        return mod

    @staticmethod
    def make_llir(src, metadata, options):
        # warp-specialization mutates num_warps
        num_warp_groups = src.get_int_attr("triton_gpu.num-warp-groups-per-cta")
        if num_warp_groups is not None:
            metadata["num_warps"] *= num_warp_groups
        threads_per_warp = ir.ttgpuir.get_threads_per_warp(src)
        metadata["threads_per_warp"] = threads_per_warp
        mod = src
        # TritonGPU -> LLVM-IR (MLIR)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        # FIXME: Advanced path drops tensor layouts, so this will crash on some
        # operations being used, e.g., convert_layout.
        if os.getenv("TRITON_INTEL_REDUCE_TRANSPOSE", "0") != "1":
            intel.passes.ttgpuir.add_decompose_unsupported_conversions(pm)
        passes.convert.add_scf_to_cf(pm)
        passes.convert.add_index_to_llvmir(pm)
        # FIXME: Advanced path uses custom type conversion and needs hacky
        # solutions for SLM allocation, so this will crash on some operations
        # being used, e.g., convert_layout.
        if os.getenv("TRITON_INTEL_REDUCE_TRANSPOSE", "0") != "1":
            intel.passes.ttgpuir.add_allocate_shared_memory(pm)
        intel.passes.ttgpuir.add_to_llvmir(pm)
        passes.convert.add_arith_to_llvmir(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        if os.environ.get("TRITON_DISABLE_LINE_INFO", "0") == "0":
            passes.llvmir.add_di_scope(pm)
        pm.run(mod)
        # LLVM-IR (MLIR) -> LLVM-IR (LLVM)
        llvm.init_targets()
        context = llvm.context()
        llvm_mod = llvm.to_module(mod, context)
        intel.set_spv_target_triple(llvm_mod)
        if options.extern_libs:
            paths = [path for (name, path) in options.extern_libs]
            llvm.link_extern_libs(llvm_mod, paths)
        intel.optimize_module(llvm_mod, llvm.OPTIMIZE_O3)
        if os.getenv("TRITON_INTEL_ENABLE_POST_PROCESS_LLIR", "1") == "1":
            intel.post_process_llir(llvm_mod)

        # Get some metadata
        metadata["shared"] = src.get_int_attr("triton_gpu.shared")
        ret = str(llvm_mod)
        del llvm_mod
        del context
        return ret

    @staticmethod
    def make_spv(src, metadata, options):
        spirv, name = intel.translate_to_spirv(src)
        metadata["name"] = name
        if options.grf_mode == 'small':
            metadata["build_flags"] = "-cl-intel-128-GRF-per-thread"
        elif options.grf_mode == 'large':
            if options.num_warps > 32:
                raise RuntimeError("grf_mode = large cannot be used with num_warps > 32")
            metadata["build_flags"] = "-cl-intel-256-GRF-per-thread"
        elif options.grf_mode == 'auto':
            metadata["build_flags"] = "-cl-intel-enable-auto-large-GRF-mode"
        else:
            metadata["build_flags"] = ""

        if options.generate_native_code:
            with tempfile.NamedTemporaryFile(delete=False, mode='wb', suffix='.spv') as fsrc, \
                tempfile.NamedTemporaryFile(delete=False, mode='r', suffix='.log') as flog:
                fsrc.write(spirv)
                fsrc.flush()
                fbin = fsrc.name + '.o'

                ocloc_cmd = [
                    'ocloc', 'compile', '-file', fsrc.name, '-o', fbin, '-spirv_input', '-device', 'pvc', '-options',
                    metadata["build_flags"]
                ]

                try:
                    subprocess.run(ocloc_cmd, check=True, close_fds=False, stdout=flog, stderr=subprocess.STDOUT)
                    if os.path.exists(flog.name):
                        with open(flog.name) as log_file:
                            log = log_file.read().strip()
                            if 'spilled' in log:
                                """
                                The exact message is something like:
                                    warning: kernel matmul_kernel  compiled SIMD16 allocated 128 regs and spilled around 217
                                is "spilled" enough for now?
                                """
                                metadata["build_flags"] += " -cl-intel-256-GRF-per-thread"
                                # re-run with new build flags
                                ocloc_cmd[-1] = metadata["build_flags"]
                                subprocess.run(ocloc_cmd, check=True, close_fds=False, stdout=flog,
                                               stderr=subprocess.STDOUT)
                        os.remove(flog.name)
                    if os.path.exists(fsrc.name):
                        os.remove(fsrc.name)
                except subprocess.CalledProcessError as e:
                    with open(flog.name) as log_file:
                        log = log_file.read()
                    if os.path.exists(flog.name):
                        os.remove(flog.name)

                    if e.returncode == 255:
                        error = 'Internal Triton ZEBIN codegen error'
                    elif e.returncode == 128 + signal.SIGSEGV:
                        error = '`ocloc` raised SIGSEGV'
                    else:
                        error = f'`ocloc` failed with error code {e.returncode}'

                    raise RuntimeError(f'{error}\n'
                                       f'`ocloc` stderr:\n{log}\n'
                                       f'Repro command: {ocloc_cmd}\n')

                with open(fbin, 'rb') as f:
                    zebin = f.read()
                if os.path.exists(fbin):
                    os.remove(fbin)
            return zebin
        return spirv

    def add_stages(self, stages, options):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options, self.properties)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options)
        stages["spv"] = lambda src, metadata: self.make_spv(src, metadata, options)

    @functools.lru_cache()
    def hash(self):
        return f'SPIR-V 1.5-{self.properties}'
