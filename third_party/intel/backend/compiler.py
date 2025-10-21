from triton.backends.compiler import BaseBackend, Language
from triton._C.libtriton import ir, passes, llvm, intel
from triton.backends.intel.driver import compile_module_from_src
from triton.backends.intel.track import track
from triton import knobs

from dataclasses import dataclass
import functools
from typing import Any, Dict, Tuple
from types import ModuleType
import hashlib
import tempfile
import signal
import os
import subprocess
from pathlib import Path


@dataclass
class XPUOptions:
    num_warps: int = 4
    num_ctas: int = 1
    num_stages: int = 2
    cluster_dims: tuple = (1, 1, 1)
    warp_size: int = 32
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
    arch: str = None
    # FIXME: enable for XPU: https://github.com/intel/intel-xpu-backend-for-triton/issues/4954
    instrumentation_mode: str = ""

    def __post_init__(self):
        default_libdir = Path(__file__).parent / 'lib'
        extern_libs = {} if self.extern_libs is None else dict(self.extern_libs)
        if not extern_libs.get('libdevice', None):
            extern_libs['libdevice'] = knobs.intel.libdevice_path or str(
                default_libdir / 'libsycl-spir64-unknown-unknown.bc')

        object.__setattr__(self, 'extern_libs', tuple(extern_libs.items()))
        if self.num_warps <= 0 or (self.num_warps & (self.num_warps - 1)) != 0:
            raise AssertionError("num_warps must be a power of 2")
        self.generate_native_code = (knobs.intel.gen_native_code
                                     or knobs.intel.dump_shader_info) or self.generate_native_code

    def hash(self):
        key = '_'.join([f'{name}-{val}' for name, val in self.__dict__.items()])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


def min_dot_size(device_props: dict):
    # (M, N, K)
    # M: repeatCount. 1,2,4,8
    # N: executionSize. 16 for PVC, 8 for ATS
    # K: systolicDepth x opsPerChan. systolicDepth must be 8
    repeat_count = 1
    sdepth = 8
    exec_size = min(device_props["sub_group_sizes"])

    def get_ops_per_channel(lhs_type, rhs_type):
        l_bitwidth = lhs_type.scalar.primitive_bitwidth
        r_bitwidth = rhs_type.scalar.primitive_bitwidth
        max_ops_per_chan = 32 / max(l_bitwidth, r_bitwidth)
        return min(8, max_ops_per_chan)

    return lambda lhs_type, rhs_type: (repeat_count, exec_size, sdepth * get_ops_per_channel(lhs_type, rhs_type))


class XPUBackend(BaseBackend):
    instrumentation = None

    @staticmethod
    def supports_target(target: tuple):
        return target.backend == 'xpu'

    def __init__(self, target: tuple) -> None:
        super().__init__(target)
        if not isinstance(target.arch, dict):
            raise TypeError("target.arch is not a dict")
        dirname = os.path.dirname(os.path.realpath(__file__))
        mod = compile_module_from_src(src=Path(os.path.join(dirname, "arch_parser.c")).read_text(), name="arch_utils")
        self.device_arch = knobs.intel.device_arch or mod.parse_device_arch(target.arch.get('architecture', 0))
        self.properties = self.parse_target(target.arch)
        self.binary_ext = "spv"

    def get_target_name(self, options) -> str:
        return f"xpu:{self.device_arch}"

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
        if XPUBackend.instrumentation:
            XPUBackend.instrumentation.load_dialects(ctx)

    @staticmethod
    def validate_options(opt, properties):
        # Check warp_size and num_threads are within limits.
        if opt.warp_size not in properties['sub_group_sizes']:
            raise ValueError(
                f"warp_size={opt.warp_size} is unsupported for the target (supported values are {properties['sub_group_sizes']})"
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
        module_opts.threads_per_warp = opt.warp_size
        module_opts.target_arch = target_arch
        intel.passes.ttgpuir.add_triton_annotate_module(pm, module_opts)
        pm.run(mod, 'annotate_module')

    @staticmethod
    def get_split_barrier_scope(opt):
        split_barriers_scope = intel.SplitBarrierScope.none
        if opt.split_barriers_scope == 'Workgroup':
            split_barriers_scope = intel.SplitBarrierScope.Workgroup
        elif opt.split_barriers_scope == 'Subgroup':
            split_barriers_scope = intel.SplitBarrierScope.Subgroup
        return split_barriers_scope

    @staticmethod
    @track
    def make_ttir(mod, metadata, opt):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        intel.passes.ttir.add_convert_tdesc_to_block_pointer(pm)
        passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
        passes.common.add_cse(pm)
        passes.common.add_licm(pm)
        intel.passes.ttir.add_remove_masks(pm)
        intel.passes.ttir.add_fuse_reshape(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod, 'make_ttir')
        return mod

    @staticmethod
    @track
    def make_ttgir(mod, metadata, opt, properties):
        cluster_info = intel.ClusterInfo()
        if opt.cluster_dims is not None:
            cluster_info.clusterDimX = opt.cluster_dims[0]
            cluster_info.clusterDimY = opt.cluster_dims[1]
            cluster_info.clusterDimZ = opt.cluster_dims[2]

        # Annotate module with information required by subsequent transformations.
        XPUBackend.annotate_module(mod, properties, opt, "spir64")

        # Overwrite the warp_size option with the module annotation.
        opt.warp_size = intel.get_threads_per_warp(mod)
        XPUBackend.validate_options(opt, properties)

        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.ttir.add_convert_to_ttgpuir(pm, "xpu", opt.num_warps, opt.warp_size, opt.num_ctas)
        # optimize TTGIR
        intel.passes.ttgpuir.add_coalesce(pm)
        intel.passes.ttgpuir.add_remove_layout_conversions(pm)

        intel.passes.ttgpuir.add_accelerate_matmul(pm)
        intel.passes.ttgpuir.add_materialize_block_pointer(pm)
        intel.passes.ttgpuir.add_remove_layout_conversions(pm)
        intel.passes.ttgpuir.add_optimize_dot_operands(pm)
        intel.passes.ttgpuir.add_pipeline(pm, opt.num_stages, XPUBackend.get_split_barrier_scope(opt))

        if (opt.reduce_variable_liveness):
            intel.passes.ttgpuir.add_reduce_variable_liveness(pm)

        passes.ttgpuir.add_fuse_nested_loops(pm)

        passes.common.add_canonicalizer(pm)
        passes.ttir.add_triton_licm(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttgpuir.add_combine_tensor_select_and_if(pm)

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
        passes.common.add_sccp(pm)
        passes.common.add_canonicalizer(pm)
        if knobs.intel.opt_reduction_locality:
            intel.passes.ttgpuir.add_optimize_reduction_locality(pm)
        intel.passes.arith.add_arith_emulate_unsupported_floats(pm, ["bf16"], "f32")
        pm.run(mod, 'make_ttgir')
        metadata["cluster_dims"] = (cluster_info.clusterDimX, cluster_info.clusterDimY, cluster_info.clusterDimZ)
        return mod

    def gluon_to_ttgir(self, src, metadata, options):
        mod = src
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        passes.gluon.add_inliner(pm)
        passes.gluon.add_resolve_auto_encodings(pm)
        passes.common.add_sccp(pm)
        passes.ttir.add_loop_aware_cse(pm)
        passes.gluon.add_canonicalizer(pm)
        passes.ttgpuir.add_combine_tensor_select_and_if(pm)

        pm.run(mod, 'gluon_to_ttgir')
        metadata["tensordesc_meta"] = mod.get_tensordesc_metadata()
        return mod

    @staticmethod
    @track
    def make_llir(src, metadata, options):
        mod = src
        # TritonGPU -> LLVM-IR (MLIR)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        passes.convert.add_scf_to_cf(pm)
        passes.gluon.add_inliner(pm)
        passes.convert.add_index_to_llvmir(pm)
        intel.passes.ttgpuir.add_allocate_shared_memory(pm)
        passes.ttgpuir.add_allocate_global_scratch_memory(pm)
        # instrumentation point here so we can override IRs above (e.g., ttir and ttgir)
        if XPUBackend.instrumentation:
            XPUBackend.instrumentation.patch("ttgpuir_to_llvmir", pm, mod.context)
        intel.passes.ttgpuir.add_to_llvmir(pm)
        intel.passes.ttgpuir.add_gen_to_llvm(pm)
        passes.common.add_canonicalizer(pm)
        intel.passes.ttgpuir.add_rewrite_stack_ptr(pm)
        passes.common.add_cse(pm)
        passes.convert.add_arith_to_llvmir(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)

        if not knobs.compilation.disable_line_info and not knobs.compilation.dump_ir_extract_di_local_variables:
            passes.llvmir.add_di_scope(pm)

        if XPUBackend.instrumentation:
            XPUBackend.instrumentation.patch("llvmir_to_llvm", pm, mod.context)
        pm.run(mod, 'make_llir')

        if knobs.compilation.dump_ir_extract_di_local_variables:
            # comments below on why separate it
            if not knobs.compilation.disable_line_info:
                pm = ir.pass_manager(mod.context)
                pm.enable_debug()
                passes.llvmir.add_di_scope(pm)
                pm.run(mod, 'make_llir.disable_line_info')

            # insert dbg intrinsic with several DI Attribute including source
            # var name and type info note: unknown reason for now, but this
            # pass and add_di_scope has to be run separately, otherwise if we
            # put them into previous pipline, it trigger a segmentfault without
            # any error message; could be due to a bug in mlir or pybind11
            pm = ir.pass_manager(mod.context)
            pm.enable_debug()
            passes.llvmir.add_di_local_variable(pm)
            pm.run(mod, 'make_llir.dump_ir_extract_di_local_variables')

        # LLVM-IR (MLIR) -> LLVM-IR (LLVM)
        llvm.init_targets()
        context = llvm.context()
        llvm_mod = llvm.to_module(mod, context)
        intel.set_spv_target_triple(llvm_mod)
        intel.set_fast_math(llvm_mod)
        if options.extern_libs:
            paths = [path for (name, path) in options.extern_libs]
            llvm.link_extern_libs(llvm_mod, paths)

        with track("optimize_module") as tr:
            intel.optimize_module(llvm_mod, llvm.OPTIMIZE_O3, tr.callback("passes"))

        intel.post_process_llir(llvm_mod)

        # Get some metadata
        total_num_warps = src.get_int_attr("ttg.total-num-warps")
        if total_num_warps is not None:
            metadata["num_warps"] = total_num_warps
        metadata["threads_per_warp"] = intel.get_threads_per_warp(src)
        metadata["shared"] = src.get_int_attr("ttg.shared")
        metadata["global_scratch_size"] = src.get_int_attr("ttg.global_scratch_memory_size")
        metadata["global_scratch_align"] = src.get_int_attr("ttg.global_scratch_memory_alignment")
        metadata["profile_scratch_size"] = src.get_int_attr("ttg.profile_scratch_memory_size") or 0
        metadata["profile_scratch_align"] = src.get_int_attr("ttg.profile_scratch_memory_alignment") or 1
        ret = str(llvm_mod)
        del llvm_mod
        del context
        return ret

    @staticmethod
    @track
    def make_spv(src, metadata, options, device_arch):
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

        shader_dump_opt = ""
        if knobs.intel.dump_shader_info:
            # The IGC (Intel Graphic Compiler) only parses the options at first time in JIT-ing the binary per process.
            # Have to use the `ocloc` to generate the binary in sub-process to work around the limitation.
            assert options.generate_native_code, "Only support native code generation with shader dump"
            shader_dump_opt = f" -igc_opts ',DumpToCustomDir={metadata['cache_dir']},ShaderDumpEnable=1'"

        metadata["generate_native_code"] = options.generate_native_code

        if options.generate_native_code:
            with track("generate_native_code"), tempfile.TemporaryDirectory() as temp_dir:
                with tempfile.NamedTemporaryFile(mode='wb', suffix='.spv', dir=temp_dir, delete=False) as fsrc:
                    fsrc.write(spirv)
                fbin = fsrc.name + '.o'

                ocloc_cmd = [
                    'ocloc', 'compile', '-file', fsrc.name, '-o', fbin, '-spirv_input', '-device', device_arch,
                    '-options', metadata["build_flags"] + shader_dump_opt
                ]

                try:
                    output = subprocess.check_output(ocloc_cmd, stderr=subprocess.STDOUT, text=True)
                    if 'spilled' in output and metadata["build_flags"].find("-cl-intel-256-GRF-per-thread") == -1:
                        """
                        The exact message is something like:
                            warning: kernel matmul_kernel  compiled SIMD16 allocated 128 regs and spilled around 217
                        is "spilled" enough for now?
                        """
                        metadata["build_flags"] += " -cl-intel-256-GRF-per-thread"
                        # re-run with new build flags
                        ocloc_cmd[-1] = metadata["build_flags"] + shader_dump_opt
                        subprocess.check_output(ocloc_cmd, stderr=subprocess.STDOUT, text=True)
                except subprocess.CalledProcessError as e:
                    if e.returncode == 255:
                        error = 'Internal Triton ZEBIN codegen error'
                    elif e.returncode == 128 + signal.SIGSEGV:
                        error = '`ocloc` raised SIGSEGV'
                    else:
                        error = f'`ocloc` failed with error code {e.returncode}'

                    raise RuntimeError(f'{error}\n'
                                       f'`ocloc` stderr:\n{e.output}\n'
                                       f'Repro command: {ocloc_cmd}\n') from e

                with open(fbin, 'rb') as f:
                    zebin = f.read()
            return zebin
        return spirv

    def add_stages(self, stages, options, language):
        if language == Language.TRITON:
            stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
            stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options, self.properties)
        elif language == Language.GLUON:
            stages["ttgir"] = lambda src, metadata: self.gluon_to_ttgir(src, metadata, options)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options)
        stages["spv"] = lambda src, metadata: self.make_spv(src, metadata, options, self.device_arch)
        if knobs.runtime.add_stages_inspection_hook is not None:
            knobs.runtime.add_stages_inspection_hook(self, stages, options, language, None)

    @functools.lru_cache()
    def hash(self):
        return f'SPIR-V 1.5-{self.properties}'
