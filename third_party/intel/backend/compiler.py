from triton.backends.compiler import BaseBackend, Language
from triton._C.libtriton import ir, passes, llvm, intel
from triton.backends.intel.driver import compile_module_from_src
from triton import knobs

from dataclasses import dataclass
import functools
from typing import Any, Dict, Tuple
from types import ModuleType
import hashlib
import tempfile
import signal
import os
import shutil
import subprocess
from pathlib import Path


@dataclass
class XPUOptions:
    num_warps: int = 4
    num_ctas: int = 1
    num_stages: int = 2
    cluster_dims: tuple = (1, 1, 1)
    threads_per_warp: int = 32
    optimize_epilogue: bool = False
    enable_fp_fusion: bool = True
    launch_cooperative_grid: bool = False
    reduce_variable_liveness: bool = True
    supported_fp8_dtypes: Tuple[str] = ("fp8e5", "fp8e4nv", "fp8e4b15")
    deprecated_fp8_dot_operand_dtypes: Tuple[str] = ()
    default_dot_input_precision: str = "tf32"
    allowed_dot_input_precisions: Tuple[str] = ("tf32", "tf32x3", "ieee")
    allow_fp8e4nv: bool = False
    allow_fp8e4b15: bool = True
    grf_mode: tuple = ('small', 'large', 'auto', 'default')
    split_barriers_scope: str = 'None'
    max_num_imprecise_acc_default: int = 0  # `max_num_imprecise_acc` only applies to fp8 -> fp32 dot on sm_90 for cuda
    extern_libs: dict = None
    debug: bool = False
    backend_name: str = 'intel'
    sanitize_overflow: bool = False
    generate_native_code: bool = False
    advanced_path: bool = False
    one_matrix_per_load_for_bt: bool = False
    enable_tile_load_linear_layout: bool = True

    def __post_init__(self):
        default_libdir = Path(__file__).parent / 'lib'
        extern_libs = {} if self.extern_libs is None else dict(self.extern_libs)
        if not extern_libs.get('libdevice', None):
            extern_libs['libdevice'] = knobs.intel.libdevice_path or str(
                default_libdir / 'libsycl-spir64-unknown-unknown.bc')

        object.__setattr__(self, 'extern_libs', tuple(extern_libs.items()))
        if self.num_warps <= 0 or (self.num_warps & (self.num_warps - 1)) != 0:
            raise AssertionError("num_warps must be a power of 2")
        self.generate_native_code = knobs.intel.gen_native_code or self.generate_native_code

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
    device_props: dict = {}

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
        dirname = os.path.dirname(os.path.realpath(__file__))
        mod = compile_module_from_src(Path(os.path.join(dirname, "arch_parser.c")).read_text(), "arch_utils")
        self.device_arch = mod.parse_device_arch(target.arch.get('architecture', 0))
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

        if self.device_arch and shutil.which('ocloc'):
            if self.device_arch in self.device_props:
                dev_prop.update(self.device_props[self.device_arch])
                return dev_prop
            try:
                ocloc_cmd = ['ocloc', 'query', 'CL_DEVICE_EXTENSIONS', '-device', self.device_arch]
                with tempfile.TemporaryDirectory() as temp_dir:
                    output = subprocess.check_output(ocloc_cmd, text=True, cwd=temp_dir)
                supported_extensions = set()
                for extension in output.split(' '):
                    supported_extensions.add(extension)
                ocloc_dev_prop = {}
                ocloc_dev_prop[
                    'has_subgroup_matrix_multiply_accumulate'] = 'cl_intel_subgroup_matrix_multiply_accumulate' in supported_extensions
                ocloc_dev_prop[
                    'has_subgroup_matrix_multiply_accumulate_tensor_float32'] = 'cl_intel_subgroup_matrix_multiply_accumulate_tensor_float32' in supported_extensions
                ocloc_dev_prop['has_subgroup_2d_block_io'] = 'cl_intel_subgroup_2d_block_io' in supported_extensions
                ocloc_dev_prop['has_bfloat16_conversions'] = 'cl_intel_bfloat16_conversions' in supported_extensions
                self.device_props[self.device_arch] = ocloc_dev_prop
                dev_prop.update(ocloc_dev_prop)
            except subprocess.CalledProcessError:
                # Note: LTS driver does not support ocloc query CL_DEVICE_EXTENSIONS.
                pass
        return dev_prop

    def parse_options(self, opts) -> Any:
        args = {k: opts[k] for k in XPUOptions.__dataclass_fields__.keys() if k in opts}
        args["allow_fp8e4nv"] = True
        args["enable_tile_load_linear_layout"] = knobs.intel.tile_load_ll
        return XPUOptions(**args)

    def pack_metadata(self, metadata):
        return metadata

    def get_codegen_implementation(self, options):
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
    def parse_raise_block_pointer_flags() -> dict:
        str = knobs.intel.raise_block_pointer
        raise_block_ptr_flags = {}
        raise_block_ptr_flags['enabled'] = False
        raise_block_ptr_flags['ignore-masks'] = False
        for flag in str.split(':'):
            if (flag == "1"):
                raise_block_ptr_flags['enabled'] = True
            if (flag == "ignore-masks"):
                raise_block_ptr_flags['enabled'] = True
                raise_block_ptr_flags['ignore-masks'] = True
        return raise_block_ptr_flags

    @staticmethod
    def validate_options(opt, properties):
        # Check threads_per_warp and num_threads are within limits.
        if opt.threads_per_warp not in properties['sub_group_sizes']:
            raise ValueError(
                f"threads_per_warp={opt.threads_per_warp} is unsupported for the target (supported values are {properties['sub_group_sizes']})"
            )
        if opt.num_warps > properties['max_num_sub_groups']:
            raise ValueError(
                f"num_warps={opt.num_warps} is unsupported for the target (limit is {properties['max_num_sub_groups']})"
            )

    @staticmethod
    def annotate_module(mod, properties, opt, target_arch):
        # Annotate module with information required by subsequent transformations.
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        module_opts = intel.passes.ttgpuir.AnnotateModuleOptions()
        module_opts.min_sg_size = min(properties["sub_group_sizes"])
        module_opts.support_sg_2d_block = properties["has_subgroup_2d_block_io"]
        module_opts.support_dpas = properties["has_subgroup_matrix_multiply_accumulate"]
        module_opts.support_bf16_conversion = properties["has_bfloat16_conversions"]
        module_opts.threads_per_warp = opt.threads_per_warp
        module_opts.target_arch = target_arch
        intel.passes.ttgpuir.add_triton_annotate_module(pm, module_opts)
        pm.run(mod)

    @staticmethod
    def get_split_barrier_scope(opt):
        split_barriers_scope = intel.SplitBarrierScope.none
        if opt.split_barriers_scope == 'Workgroup':
            split_barriers_scope = intel.SplitBarrierScope.Workgroup
        elif opt.split_barriers_scope == 'Subgroup':
            split_barriers_scope = intel.SplitBarrierScope.Subgroup
        return split_barriers_scope

    @staticmethod
    def make_ttir(mod, metadata, opt):
        raise_block_ptr_flags = XPUBackend.parse_raise_block_pointer_flags()

        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        intel.passes.ttir.add_convert_tdesc_to_block_pointer(pm)
        passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
        passes.common.add_cse(pm)
        passes.common.add_licm(pm)
        intel.passes.ttir.add_remove_masks(pm)
        if raise_block_ptr_flags['enabled']:
            ignore_masks = True if raise_block_ptr_flags['ignore-masks'] else False
            intel.passes.ttir.add_raise_block_pointer(pm, ignore_masks)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
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
        XPUBackend.annotate_module(mod, properties, opt, "spir64")

        # Overwrite the threads_per_warp option with the module annotation.
        opt.threads_per_warp = intel.get_threads_per_warp(mod)
        XPUBackend.validate_options(opt, properties)

        if (properties["has_subgroup_2d_block_io"] and properties["has_subgroup_matrix_multiply_accumulate"]
                and (knobs.intel.advanced_path or opt.advanced_path)):
            return XPUBackend.AdvancedPath.make_ttgir(mod, metadata, opt)

        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.ttir.add_convert_to_ttgpuir(pm, "xpu", opt.num_warps, opt.threads_per_warp, opt.num_ctas)
        # optimize TTGIR
        intel.passes.ttgpuir.add_coalesce(pm)
        intel.passes.ttgpuir.add_remove_layout_conversions(pm)

        intel.passes.ttgpuir.add_accelerate_matmul(pm)
        intel.passes.ttgpuir.add_materialize_block_pointer(pm)
        intel.passes.ttgpuir.add_optimize_block_load_encoding(pm)
        intel.passes.ttgpuir.add_remove_layout_conversions(pm)
        intel.passes.ttgpuir.add_pipeline(pm, opt.num_stages, XPUBackend.get_split_barrier_scope(opt))

        if (opt.reduce_variable_liveness):
            intel.passes.ttgpuir.add_reduce_variable_liveness(pm)

        passes.ttgpuir.add_fuse_nested_loops(pm)
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
        if knobs.intel.opt_reduction_locality:
            intel.passes.ttgpuir.add_optimize_reduction_locality(pm)
        intel.passes.arith.add_arith_emulate_unsupported_floats(pm, ["bf16"], "f32")
        pm.run(mod)
        metadata["cluster_dims"] = (cluster_info.clusterDimX, cluster_info.clusterDimY, cluster_info.clusterDimZ)
        return mod

    @staticmethod
    def ttgir_opt(src, metadata, options):
        mod = src
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        passes.ttgpuir.add_inliner(pm)
        passes.ttir.add_loop_aware_cse(pm)
        passes.ttgpuir.add_canonicalizer(pm)
        passes.ttgpuir.add_combine_tensor_select_and_if(pm)

        pm.run(mod)
        return mod

    @staticmethod
    def make_llir(src, metadata, options):
        mod = src
        # TritonGPU -> LLVM-IR (MLIR)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        passes.convert.add_scf_to_cf(pm)
        passes.convert.add_index_to_llvmir(pm)
        # FIXME: Advanced path uses custom type conversion and needs hacky
        # solutions for SLM allocation, so this will crash on some operations
        # being used, e.g., convert_layout.
        if not knobs.intel.reduce_transpose:
            intel.passes.ttgpuir.add_allocate_shared_memory(pm)
        passes.ttgpuir.add_allocate_global_scratch_memory(pm)
        intel.passes.ttgpuir.add_to_llvmir(pm, options.advanced_path, options.one_matrix_per_load_for_bt,
                                           options.enable_tile_load_linear_layout)
        intel.passes.ttgpuir.add_gen_to_llvm(pm)
        intel.passes.ttgpuir.add_rewrite_stack_ptr(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.convert.add_arith_to_llvmir(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        if not knobs.compilation.disable_line_info:
            passes.llvmir.add_di_scope(pm)
        pm.run(mod)
        # LLVM-IR (MLIR) -> LLVM-IR (LLVM)
        llvm.init_targets()
        context = llvm.context()
        llvm_mod = llvm.to_module(mod, context)
        intel.set_spv_target_triple(llvm_mod)
        intel.set_fast_math(llvm_mod)
        if options.extern_libs:
            paths = [path for (name, path) in options.extern_libs]
            llvm.link_extern_libs(llvm_mod, paths)

        intel.optimize_module(llvm_mod, llvm.OPTIMIZE_O3)
        intel.post_process_llir(llvm_mod)

        # Get some metadata
        total_num_warps = src.get_int_attr("ttg.total-num-warps")
        if total_num_warps is not None:
            metadata["num_warps"] = total_num_warps
        metadata["threads_per_warp"] = intel.get_threads_per_warp(src)
        metadata["shared"] = src.get_int_attr("ttg.shared")
        metadata["global_scratch_size"] = src.get_int_attr("ttg.global_scratch_memory_size")
        metadata["global_scratch_align"] = src.get_int_attr("ttg.global_scratch_memory_alignment")
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

        if knobs.intel.disable_igc_opt:
            metadata["build_flags"] += " -cl-opt-disable"

        metadata["generate_native_code"] = options.generate_native_code

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
                            if 'spilled' in log and metadata["build_flags"].find("-cl-intel-256-GRF-per-thread") == -1:
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

    def add_stages(self, stages, options, language):
        if language == Language.TRITON:
            stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
            stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options, self.properties)
        elif language == Language.GLUON:
            stages["ttgir"] = lambda src, metadata: self.ttgir_opt(src, metadata, options)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options)
        stages["spv"] = lambda src, metadata: self.make_spv(src, metadata, options)

    @functools.lru_cache()
    def hash(self):
        return f'SPIR-V 1.5-{self.properties}'
