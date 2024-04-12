from triton.backends.compiler import BaseBackend
from triton._C.libtriton import ir, passes, llvm, intel
from triton.backends.intel.driver import compile_module_from_src

from dataclasses import dataclass
import functools
from typing import Any, Tuple
import hashlib
import re
import os
import subprocess
from pathlib import Path


def _path_to_binary(binary: str):
    paths = [
        os.environ.get(f"TRITON_{binary.upper()}_PATH", ""),
        os.path.join(os.path.dirname(__file__), "bin", binary),
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


@dataclass(frozen=True)
class XPUOptions:
    num_warps: int = 4
    num_ctas: int = 1
    num_stages: int = 2
    cluster_dims: tuple = (1, 1, 1)
    threads_per_warp: int = 32
    optimize_epilogue: bool = False
    enable_fp_fusion: bool = True
    default_dot_input_precision: str = "tf32"
    allowed_dot_input_precisions: Tuple[str] = ("tf32", "tf32x3", "ieee")
    allow_fp8e4nv: bool = False
    allow_fp8e4b15: bool = False
    max_num_imprecise_acc_default: int = 0  # `max_num_imprecise_acc` only applies to fp8 -> fp32 dot on sm_90 for cuda
    extern_libs: dict = None
    debug: bool = False

    def __post_init__(self):
        default_libdir = Path(__file__).parent / 'lib'
        extern_libs = {} if self.extern_libs is None else dict(self.extern_libs)
        if not extern_libs.get('libdevice', None):
            extern_libs['libdevice'] = str(default_libdir / 'libsycl-spir64-unknown-unknown.bc')
        object.__setattr__(self, 'extern_libs', tuple(extern_libs.items()))
        assert self.num_warps > 0 and (self.num_warps & (self.num_warps - 1)) == 0, \
            "num_warps must be a power of 2"

    def hash(self):
        key = '_'.join([f'{name}-{val}' for name, val in self.__dict__.items()])
        return hashlib.md5(key.encode("utf-8")).hexdigest()


class XPUBackend(BaseBackend):

    @staticmethod
    def supports_target(target: tuple):
        return target[0] == 'xpu'

    def __init__(self, target: tuple) -> None:
        super().__init__(target)
        assert isinstance(target[1], dict)
        dirname = os.path.dirname(os.path.realpath(__file__))
        mod = compile_module_from_src(Path(os.path.join(dirname, "arch_parser.c")).read_text(), "arch_utils")
        self.parse_device_arch = mod.parse_device_arch
        # TODO: Deprecate capability in XPU compilation
        # capability should be < 80, because some features in passes with capability >= 80 are not supported on PVC
        self.capability = intel.passes.ttgpuir.DEVICE_ARCH.PVC
        self.properties = self._parse_target(target[1])
        self.device_arch = self.properties["device_arch"]
        self.binary_ext = "spv"

    def _parse_target(self, tgt_prop) -> dict:
        dev_prop = {}
        dev_prop['name'] = tgt_prop.get('name', 'xpu')
        dev_prop['platform_name'] = tgt_prop.get('platform_name', None)
        dev_prop['vendor'] = tgt_prop.get('vendor', None)
        dev_prop['driver_version'] = tgt_prop.get('driver_version', None)
        dev_prop['version'] = tgt_prop.get('version', None)
        dev_prop['gpu_eu_count'] = tgt_prop.get('gpu_eu_count', None)
        dev_prop['gpu_subslice_count'] = tgt_prop.get('gpu_subslice_count', None)
        dev_prop['max_work_group_size'] = tgt_prop.get('max_work_group_size', None)
        dev_prop['max_num_sub_groups'] = tgt_prop.get('max_num_sub_groups', None)
        dev_prop['sub_group_sizes'] = tgt_prop.get('sub_group_sizes', None)
        dev_prop['has_fp64'] = tgt_prop.get('has_fp64', None)
        dev_prop['device_arch'] = self.parse_device_arch(tgt_prop.get('device_arch', 0))
        return dev_prop

    def parse_options(self, opts) -> Any:
        args = {k: opts[k] for k in XPUOptions.__dataclass_fields__.keys() if k in opts}
        args["allow_fp8e4nv"] = True
        return XPUOptions(**args)

    def pack_metadata(self, metadata):
        return metadata

    def load_dialects(self, ctx):
        intel.load_dialects(ctx)

    @staticmethod
    def make_ttir(mod, metadata, opt):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        # passes.ttir.add_rewrite_tensor_pointer(pm)
        passes.ttir.add_combine(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_licm(pm)
        passes.common.add_symbol_dce(pm)
        pm.run(mod)
        return mod

    @staticmethod
    def make_ttgir(mod, metadata, opt, device_arch):
        cluster_info = intel.ClusterInfo()
        if opt.cluster_dims is not None:
            cluster_info.clusterDimX = opt.cluster_dims[0]
            cluster_info.clusterDimY = opt.cluster_dims[1]
            cluster_info.clusterDimZ = opt.cluster_dims[2]
        # TTIR -> TTGIR
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.ttir.add_convert_to_ttgpuir(pm, opt.num_warps, opt.threads_per_warp, opt.num_ctas, device_arch)
        # optimize TTGIR
        # passes.ttgpuir.add_coalesce(pm)
        # TODO(Qingyi): Move PlanCTAPass to the front of CoalescePass
        intel.passes.ttnvgpuir.add_plan_cta(pm, cluster_info)
        # passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_thread_locality(pm)
        intel.passes.ttgpuir.add_accelerate_matmul(pm, device_arch)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        if opt.optimize_epilogue:
            passes.ttgpuir.add_optimize_epilogue(pm)
        intel.passes.ttgpuir.add_rewrite_tensor_pointer(pm, intel.passes.ttgpuir.DEVICE_ARCH.PVC)
        passes.ttgpuir.add_optimize_dot_operands(pm)
        passes.common.add_cse(pm)
        passes.ttgpuir.add_prefetch(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_reduce_data_duplication(pm)
        passes.ttgpuir.add_reorder_instructions(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.common.add_canonicalizer(pm)
        pm.run(mod)
        metadata["cluster_dims"] = (cluster_info.clusterDimX, cluster_info.clusterDimY, cluster_info.clusterDimZ)
        return mod

    @staticmethod
    def make_llir(src, metadata, options, capability):
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
        intel.passes.ttgpuir.add_decompose_unsupported_conversions(pm)
        passes.convert.add_scf_to_cf(pm)
        passes.convert.add_index_to_llvmir(pm)
        intel.passes.ttgpuir.add_allocate_shared_memory(pm)
        intel.passes.ttgpuir.add_to_llvmir(pm, capability)
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
        llvm.set_spv_target_triple(llvm_mod)
        if options.extern_libs:
            paths = [path for (name, path) in options.extern_libs]
            llvm.link_extern_libs(llvm_mod, paths)
        llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3)
        # Get some metadata
        metadata["shared"] = src.get_int_attr("triton_gpu.shared")
        ret = str(llvm_mod)
        del llvm_mod
        del context
        return ret

    @staticmethod
    def make_spv(src, metadata):
        ret, name = llvm.translate_to_spirv(src)
        metadata["name"] = name
        return ret

    def add_stages(self, stages, options):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options, self.device_arch)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options, self.device_arch)
        stages["spv"] = lambda src, metadata: self.make_spv(src, metadata)

    @functools.lru_cache()
    def hash(self):
        version = subprocess.check_output([_path_to_binary("spirv-dis")[0], "--version"])
        if type(version) is bytes:
            version = version.decode("utf-8")
        return f'{version}-{self.properties}'

    def get_codegen_implementation(self):
        from triton.language.extra.intel import convert_custom_float8
        codegen_fns = {}
        codegen_fns["convert_custom_types"] = convert_custom_float8
        return codegen_fns
