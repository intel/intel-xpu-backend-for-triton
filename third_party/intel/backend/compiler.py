from triton.backends.compiler import BaseBackend, GPUTarget, Language
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

try:  # XPUBackend allows metaclasses injection
    from .meta import XPUBackendMeta
except ImportError:
    XPUBackendMeta = type(BaseBackend)


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


class XPUBackend(BaseBackend, metaclass=XPUBackendMeta):
    arch_to_impl = {}  # Architecture id to backend implementation class mapping
    binary_ext = "spv"
    target_arch = "spir64"
    device_props: dict = {}
    instrumentation = None

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'xpu'

    def __new__(cls, target: GPUTarget):
        if not isinstance(target.arch, dict):
            raise TypeError("target.arch is not a dict")
        if cls is not XPUBackend:
            return super().__new__(cls)
        arch = target.arch.get("architecture", 0)
        if (impl := cls.arch_to_impl.get(arch, None)) is None:
            # Try to find an arch-specific implementation in the .arch.<name> submodule.
            if not (dev_arch := knobs.intel.device_arch):
                dirname = os.path.dirname(os.path.realpath(__file__))
                parser = compile_module_from_src(src=Path(os.path.join(dirname, "arch_parser.c")).read_text(),
                                                 name="arch_utils")
                dev_arch = parser.parse_device_arch(target.arch.get('architecture', 0))
            mod_name = f"{__package__}.arch.{dev_arch}"
            try:
                impl = __import__(mod_name, fromlist=["XPUBackendImpl"]).XPUBackendImpl
            except ImportError:
                impl = type(f"{mod_name}.XPUBackendImpl", (cls, ), {})
            impl.device_arch = dev_arch
            cls.arch_to_impl[arch] = impl
        return super().__new__(impl)

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.properties = self.parse_target(target.arch)

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

        if not self.device_arch:
            return dev_prop

        if self.device_arch in self.device_props:
            dev_prop.update(self.device_props[self.device_arch])
            return dev_prop

        return dev_prop

    def parse_options(self, opts) -> Any:
        args = {k: v for k, v in opts.items() if k in XPUOptions.__dataclass_fields__}
        args["allow_fp8e4nv"] = True
        return XPUOptions(**args)

    def pack_metadata(self, metadata):
        return metadata

    @staticmethod
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

    def get_codegen_implementation(self, options):
        from triton.language.extra.intel import convert_custom_float8
        codegen_fns = {}
        codegen_fns["convert_custom_types"] = convert_custom_float8
        codegen_fns["min_dot_size"] = self.min_dot_size(self.properties)
        return codegen_fns

    def get_module_map(self) -> Dict[str, ModuleType]:
        from triton.language.extra.intel import libdevice
        return {"triton.language.extra.libdevice": libdevice}

    def load_dialects(self, ctx):
        intel.load_dialects(ctx)
        if self.instrumentation:
            self.instrumentation.load_dialects(ctx)

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

    @classmethod
    def annotate_module(cls, module_opts, properties, opt):
        # Annotate module with information required by subsequent transformations.
        module_opts.min_sg_size = min(properties["sub_group_sizes"])
        module_opts.support_sg_2d_block = properties["has_subgroup_2d_block_io"]
        module_opts.support_dpas = properties["has_subgroup_matrix_multiply_accumulate"]
        module_opts.support_bf16_conversion = properties["has_bfloat16_conversions"]
        module_opts.threads_per_warp = opt.warp_size
        module_opts.target_arch = cls.target_arch

    @staticmethod
    def get_split_barrier_scope(opt):
        split_barriers_scope = intel.SplitBarrierScope.none
        if opt.split_barriers_scope == 'Workgroup':
            split_barriers_scope = intel.SplitBarrierScope.Workgroup
        elif opt.split_barriers_scope == 'Subgroup':
            split_barriers_scope = intel.SplitBarrierScope.Subgroup
        return split_barriers_scope

    @classmethod
    def create_pass_manager(cls, context, add_passes=[]):
        pm = ir.pass_manager(context)
        pm.enable_debug()
        for p in add_passes:
            if p is None:
                continue
            elif isinstance(p, tuple):
                p[0](pm, *p[1:])
            else:
                p(pm)
        return pm

    @classmethod
    def get_ttir_passes(cls, opt):
        return [
            passes.common.add_inliner,
            intel.passes.ttir.add_convert_tdesc_to_block_pointer,
            passes.ttir.add_rewrite_tensor_descriptor_to_pointer,
            passes.common.add_cse,
            passes.common.add_licm,
            intel.passes.ttir.add_remove_masks,
            intel.passes.ttir.add_fuse_reshape,
            passes.common.add_canonicalizer,
            passes.ttir.add_combine,
            passes.ttir.add_reorder_broadcast,
            passes.common.add_cse,
            passes.common.add_symbol_dce,
            passes.ttir.add_loop_unroll,
        ]

    @classmethod
    @track
    def make_ttir(cls, mod, metadata, opt):
        pm = cls.create_pass_manager(mod.context, cls.get_ttir_passes(opt))
        pm.run(mod, 'make_ttir')
        return mod

    @classmethod
    def get_ttgir_passes(cls, opt):
        # fmt: off
        return [
            (passes.ttir.add_convert_to_ttgpuir, "xpu", opt.num_warps, opt.warp_size, opt.num_ctas),
            # optimize TTGIR
            intel.passes.ttgpuir.add_coalesce,
            intel.passes.ttgpuir.add_remove_layout_conversions,

            intel.passes.ttgpuir.add_accelerate_matmul,
            intel.passes.ttgpuir.add_materialize_block_pointer,
            intel.passes.ttgpuir.add_remove_layout_conversions,
            intel.passes.ttgpuir.add_optimize_dot_operands,
            (intel.passes.ttgpuir.add_pipeline, opt.num_stages, cls.get_split_barrier_scope(opt)),

            intel.passes.ttgpuir.add_reduce_variable_liveness if opt.reduce_variable_liveness else None,

            passes.ttgpuir.add_fuse_nested_loops,

            passes.common.add_canonicalizer,
            passes.ttir.add_triton_licm,
            passes.common.add_canonicalizer,
            passes.ttgpuir.add_combine_tensor_select_and_if,

            passes.ttgpuir.add_optimize_thread_locality,
            (passes.ttgpuir.add_optimize_dot_operands, True),
            passes.common.add_cse,
            passes.ttgpuir.add_prefetch,
            (passes.ttgpuir.add_optimize_dot_operands, True),
            intel.passes.ttgpuir.add_remove_layout_conversions,
            intel.passes.ttgpuir.add_reduce_data_duplication,
            passes.ttgpuir.add_reorder_instructions,
            passes.common.add_cse,
            passes.common.add_symbol_dce,
            passes.common.add_sccp,
            passes.common.add_canonicalizer,
            intel.passes.ttgpuir.add_optimize_reduction_locality if knobs.intel.opt_reduction_locality else None,
            (intel.passes.arith.add_arith_emulate_unsupported_floats, ["bf16"], "f32")
        ]
        # fmt: on

    @classmethod
    @track
    def make_ttgir(cls, mod, metadata, opt, properties):
        cluster_info = intel.ClusterInfo()
        if opt.cluster_dims is not None:
            cluster_info.clusterDimX = opt.cluster_dims[0]
            cluster_info.clusterDimY = opt.cluster_dims[1]
            cluster_info.clusterDimZ = opt.cluster_dims[2]

        # Annotate module with information required by subsequent transformations.
        pm = cls.create_pass_manager(mod.context)
        module_opts = intel.passes.ttgpuir.AnnotateModuleOptions()
        cls.annotate_module(module_opts, properties, opt)
        intel.passes.ttgpuir.add_triton_annotate_module(pm, module_opts)
        pm.run(mod, 'annotate_module')

        # Overwrite the warp_size option with the module annotation.
        opt.warp_size = intel.get_threads_per_warp(mod)
        cls.validate_options(opt, properties)

        pm = cls.create_pass_manager(mod.context, cls.get_ttgir_passes(opt))
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

    @classmethod
    def get_llir_passes(cls, opt, mod):
        # fmt: off
        return [
            passes.convert.add_scf_to_cf,
            passes.gluon.add_inliner,
            passes.convert.add_index_to_llvmir,
            intel.passes.ttgpuir.add_allocate_shared_memory,
            passes.ttgpuir.add_allocate_global_scratch_memory,
            # instrumentation point here so we can override IRs above (e.g., ttir and ttgir)
            lambda pm: cls.instrumentation.patch("ttgpuir_to_llvmir", pm, mod.context) if cls.instrumentation else None,
            intel.passes.ttgpuir.add_to_llvmir,
            intel.passes.ttgpuir.add_gen_to_llvm,
            passes.common.add_canonicalizer,
            intel.passes.ttgpuir.add_rewrite_stack_ptr,
            passes.common.add_cse,
            passes.convert.add_arith_to_llvmir,
            passes.common.add_canonicalizer,
            passes.common.add_cse,
            passes.common.add_symbol_dce,
            None if knobs.compilation.disable_line_info or knobs.compilation.dump_ir_extract_di_local_variables else passes.llvmir.add_di_scope,
            lambda pm: cls.instrumentation.patch("llvmir_to_llvm", pm, mod.context) if cls.instrumentation else None,
        ]
        # fmt: on

    @classmethod
    def optimize_llvm_mod(cls, llvm_mod, options):
        intel.set_spv_target_triple(llvm_mod)
        with track("optimize_module") as tr:
            intel.optimize_module(llvm_mod, llvm.OPTIMIZE_O3, tr.callback("passes"))

    @classmethod
    @track
    def make_llir(cls, src, metadata, options):
        mod = src
        # TritonGPU -> LLVM-IR (MLIR)
        pm = cls.create_pass_manager(mod.context, cls.get_llir_passes(options, mod))
        pm.run(mod, 'make_llir')

        if knobs.compilation.dump_ir_extract_di_local_variables:
            # comments below on why separate it
            if not knobs.compilation.disable_line_info:
                pm = cls.create_pass_manager(mod.context, [passes.llvmir.add_di_scope])
                pm.run(mod, 'make_llir.disable_line_info')

            # insert dbg intrinsic with several DI Attribute including source
            # var name and type info note: unknown reason for now, but this
            # pass and add_di_scope has to be run separately, otherwise if we
            # put them into previous pipline, it trigger a segmentfault without
            # any error message; could be due to a bug in mlir or pybind11
            pm = cls.create_pass_manager(mod.context, [passes.llvmir.add_di_local_variable])
            pm.run(mod, 'make_llir.dump_ir_extract_di_local_variables')

        # LLVM-IR (MLIR) -> LLVM-IR (LLVM)
        llvm.init_targets()
        context = llvm.context()
        llvm_mod = llvm.to_module(mod, context)
        intel.set_fast_math(llvm_mod)
        if options.extern_libs:
            paths = [path for (name, path) in options.extern_libs]
            llvm.link_extern_libs(llvm_mod, paths)

        cls.optimize_llvm_mod(llvm_mod, options)
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

    @classmethod
    @track
    def make_spv(cls, src, metadata, options):
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
                    'ocloc', 'compile', '-file', fsrc.name, '-o', fbin, '-spirv_input', '-device', cls.device_arch,
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
        stages["spv"] = lambda src, metadata: self.make_spv(src, metadata, options)
        if knobs.runtime.add_stages_inspection_hook is not None:
            knobs.runtime.add_stages_inspection_hook(self, stages, options, language, None)

    @functools.lru_cache()
    def hash(self):
        return f'SPIR-V 1.5-{self.properties}'
