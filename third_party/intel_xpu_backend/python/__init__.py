import functools
import hashlib
import os
import re
import sysconfig
import tempfile
from pathlib import Path
from typing import Any

import setuptools
import torch
import triton._C.libintel_xpu_backend_for_triton.triton as _triton  # noqa:E402
from dataclasses import asdict, dataclass
from triton.common.backend import (TRITON_PATH, TRITON_VERSION,  # noqa:E402
                                   BaseBackend, register_backend)
from triton.compiler.make_launcher import make_so_cache_key  # noqa:E402
from triton.runtime.cache import get_cache_manager  # noqa:E402
from triton.runtime.driver import DriverBase  # noqa:E402

from .extensions import SYCLBuildExtension, SYCLExtension, use_profile  # noqa:E402


@functools.lru_cache()
def version_key():
    import pkgutil
    contents = []
    # frontend
    with open(__file__, "rb") as f:
        contents += [hashlib.md5(f.read()).hexdigest()]
    # compiler
    compiler_path = os.path.join(TRITON_PATH, 'compiler')
    for lib in pkgutil.iter_modules([compiler_path]):
        with open(lib.module_finder.find_spec(lib.name).origin, "rb") as f:
            contents += [hashlib.md5(f.read()).hexdigest()]
    # backend
    with open(os.path.join(TRITON_PATH, "_C/libtriton.so"), "rb") as f:
        contents += [hashlib.md5(f.read()).hexdigest()]
    with open(os.path.join(TRITON_PATH, "_C/libintel_xpu_backend_for_triton.so"), "rb") as f:
        contents += [hashlib.md5(f.read()).hexdigest()]
    # language
    language_path = os.path.join(TRITON_PATH, 'language')
    for lib in pkgutil.iter_modules([language_path]):
        with open(lib.module_finder.find_spec(lib.name).origin, "rb") as f:
            contents += [hashlib.md5(f.read()).hexdigest()]
    return '-'.join(TRITON_VERSION) + '-' + '-'.join(contents)


def _add_external_libs(mod, libs):
    for name, path in libs.items():
        if len(name) == 0 or len(path) == 0:
            return
    _triton.add_external_libs(mod, list(libs.keys()), list(libs.values()))


def spirv_get_kernel_name(spirv: str) -> str:
    '''
    Get kernel name from SPIRV code.
    This Kernel name is required when launching the kernel.
    '''
    assert spirv
    decl_ops = []
    for line in spirv.split('\n'):
        line = line.strip()
        if line.startswith('OpName'):
            decl_ops += [line.split()[-1]]
    def_ops = []
    for line in spirv.split('\n'):
        line = line.strip()
        if re.compile(r'\bOpEntryPoint\b').search(line):
            def_op = line.split()[2][1:]
            if '"{}"'.format(def_op) in decl_ops:
                def_ops += [def_op]
    assert len(def_ops) == 1, "expect only one kernel per spriv"
    return def_ops[0]


def ty_to_cpp(ty):
    if ty[0] == '*':
        return "void*"
    return {
        "i1": "int32_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "float",
        "bf16": "float",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
    }[ty]


def generate_launcher(signature, constants, ids):
    arg_decls = ', '.join(f"{ty_to_cpp(ty)} arg{i}" for i, ty in signature.items())  # noqa: E501

    def _extracted_type_pybind11(ty):
        if ty[0] == '*':
            return "py::object"
        return {
            'i1': 'int32_t',
            'i32': 'int32_t',
            'i64': 'int64_t',
            'u32': 'uint32_t',
            'u64': 'uint64_t',
            'fp32': 'float',
            'f32': 'float',
            'fp64': 'double',
        }[ty]

    folded_without_constexprs = [c for c in ids['ids_of_folded_args'] if c not in ids['ids_of_const_exprs']]
    params = [
        i for i in signature.keys()
        if (i not in constants and i not in folded_without_constexprs)
    ]

    # Ipex available src
    return f"""
#include <pybind11/pybind11.h>
#include <sycl/sycl.hpp>
#include <cstdlib>
#ifdef TRITON_XPU_PROFILE
#include <ipex.h>
#include <ATen/record_function.h>
#endif

namespace py = pybind11;

namespace {{

bool getBoolEnv(const std::string &env) {{
        const char *s = std::getenv(env.c_str());
        std::string str(s ? s : "");
        std::transform(str.begin(), str.end(), str.begin(),
                        [](unsigned char c) {{ return std::tolower(c); }});
        return (str == "on" || str == "true" || str == "1");
}}

}}

static inline void* getPointer(const py::object& _obj, int idx) {{
  PyObject* obj = _obj.ptr();
  if (PyLong_Check(obj)) {{
    auto ptrValue = PyLong_AsVoidPtr(obj);
    if (PyErr_Occurred()) {{
      PyErr_Print();
    }}
    return (void*)ptrValue;
  }}
  if (obj == Py_None) {{
    return (void*)0;
  }}
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  if (ptr) {{
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {{
      PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
    }}
    return (void*)PyLong_AsVoidPtr(ret);
  }}
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  return (void*)0;
}}

static void set_scalar_arg(
        sycl::handler& cgh,
        int index,
        size_t size,
        const void* value) {{
    switch (size) {{
    case sizeof(uint8_t):
    cgh.set_arg(index, *static_cast<const uint8_t*>(value));
    break;
    case sizeof(uint16_t):
    cgh.set_arg(index, *static_cast<const uint16_t*>(value));
    break;
    case sizeof(uint32_t):
    cgh.set_arg(index, *static_cast<const uint32_t*>(value));
    break;
    case sizeof(uint64_t):
    cgh.set_arg(index, *static_cast<const uint64_t*>(value));
    break;
    default:
    assert(false && "wrong scalar size in sycl gen.");
    }}
}}

static void sycl_kernel_launch(int gridX, int gridY, int gridZ, int num_warps, int threads_per_warp, int shared_memory, sycl::queue& stream, sycl::kernel& kernel_ptr{', ' + arg_decls if len(arg_decls) > 0 else ''}) {{
  std::string kernel_name = kernel_ptr.get_info<sycl::info::kernel::function_name>();
#ifdef TRITON_XPU_PROFILE
RECORD_FUNCTION("XPU Triton kernel:" + kernel_name, {{}});
#endif
  void *params[] = {{ {', '.join(f"&arg{i}" for i in params)} }};
  uint32_t num_params = sizeof(params)/sizeof(params[0]);
  uint32_t expected_num_params = kernel_ptr.get_info<sycl::info::kernel::num_args>();

  size_t global_range_x = gridX*threads_per_warp*num_warps;
  size_t global_range_y = gridY;
  size_t global_range_z = gridZ;

  size_t local_range_x = num_warps*threads_per_warp;
  size_t local_range_y = 1;
  size_t local_range_z = 1;

  sycl::range<3> global_range(global_range_z, global_range_y, global_range_x);
  sycl::range<3> local_range(local_range_z, local_range_y, local_range_x);
  sycl::nd_range<3> parallel_work_size(global_range, local_range);

  if (getBoolEnv("MLIR_ENABLE_DUMP")){{
    std::cout << "kernel info name:" << kernel_name << " @" << &kernel_ptr << std::endl;
    std::cout << "kernel info attributes:" << kernel_ptr.get_info<sycl::info::kernel::attributes>() << std::endl;
    std::cout << "kernel info reference_count:" << kernel_ptr.get_info<sycl::info::kernel::reference_count>() << std::endl;
    std::cout << "kernel info num_args:" << kernel_ptr.get_info<sycl::info::kernel::num_args>() << std::endl;

    std::cout << "launch num param:" << num_params << std::endl;
    std::cout << "  gridx: " << gridX << std::endl;
    std::cout << "  gridY: " << gridY << std::endl;
    std::cout << "  gridZ: " << gridZ << std::endl;
    std::cout << "  num_warps: " << num_warps << std::endl;
    std::cout << "  threads_per_warp: " << threads_per_warp << std::endl;
    std::cout << "  global range:[" << "x:"<< global_range_x << ", y:" << global_range_y << ", z:" << global_range_z << "]" << std::endl;
    std::cout << "  local range:[" << "x:"<< local_range_x << ", y:" << local_range_y << ", z:" << local_range_z << "]" << std::endl;
    std::cout << "  shared_memory: " << shared_memory << std::endl;

    // param
    {" ".join(f'std::cout << "  param {idx}:" << *({ty_to_cpp(item)}*)params[{idx}] << std::endl;' for idx, item in enumerate([signature[i] for i in signature if i not in constants]))}
  }}

  if (shared_memory) {{
    expected_num_params -= 1;
  }}
  assert(num_params == expected_num_params && "number of kernel param not matched");

  // Submit the imported kernel.
  auto cgf = [&](sycl::handler &cgh) {{

    {" ".join(f'set_scalar_arg(cgh, {idx}, sizeof({ty_to_cpp(item)}), params[{idx}]);' for idx, item in enumerate([signature[i] for i in signature if i not in constants]))}

    if (shared_memory) {{
        using share_mem_t = sycl::accessor<int8_t, 1, sycl::access::mode::read_write, sycl::access::target::local>;
        share_mem_t local_buffer = share_mem_t(shared_memory, cgh);
        cgh.set_arg(num_params, local_buffer);
        //cgh.parallel_for(sycl::nd_range{{sycl::range{{(uint32_t)gridX*threads_per_warp*num_warps}}, sycl::range{{work_group_size}}}}, kernel_ptr);
        cgh.parallel_for(parallel_work_size, kernel_ptr);
    }} else {{
        cgh.parallel_for(parallel_work_size, kernel_ptr);
    }}

    }};

  auto event = stream.submit(cgf);
#ifdef TRITON_XPU_PROFILE
xpu::profiler_record(kernel_name, event);
#endif
}}

PYBIND11_MODULE(__triton_launcher, m) {{
  m.doc() = "triton bindings to the C++ launcher API";
    m.def("launch", [](int grid_x,
                       int grid_y,
                       int grid_z,
                       int num_warps,
                       int num_ctas,
                       int clusterDimX,
                       int clusterDimY,
                       int clusterDimZ,
                       int shared_memory,
                       void* _stream,
                       void* _kernel,
                       py::object &launch_enter_hook,
                       py::object &launch_exit_hook,
                       py::object &compiled_kernel{', ' if signature.items() else ''}
                       {', '.join([f"{_extracted_type_pybind11(ty)} _arg{i}" for i, ty in signature.items()])}){{
      int threads_per_warp = 32;
      if(py::hasattr(compiled_kernel, "threads_per_warp"))
        threads_per_warp = compiled_kernel.attr("threads_per_warp").cast<int>();
      sycl::queue* stream = static_cast<sycl::queue*>(_stream);
      sycl::kernel* kernel = static_cast<sycl::kernel*>(_kernel);
      sycl_kernel_launch(grid_x, grid_y, grid_z, num_warps, threads_per_warp, shared_memory, *stream, *kernel{', ' if signature.items() else ''}
             {', '.join(f"getPointer(_arg{i},{i})" if ty[0] == "*" else f"_arg{i}" for i, ty in signature.items())});
    }});
}}

"""  # noqa: E501


def _build_xpu_ext(name, src, srcdir):

    TRITON_XPU_BUILD_LOGGING = os.getenv('TRITON_XPU_BUILD_LOGGING')
    if TRITON_XPU_BUILD_LOGGING is None or TRITON_XPU_BUILD_LOGGING == '0' or TRITON_XPU_BUILD_LOGGING.lower() == 'off':
        import logging
        logging.disable(logging.CRITICAL)

    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    so = os.path.join(srcdir, '{name}{suffix}'.format(name=name, suffix=suffix))  # noqa: E501

    # fallback on setuptools
    extra_compile_args = ['-w']
    # library_dirs = [cuda_lib_dir]
    # include_dirs = [srcdir, cu_include_dir]
    # library_dirs = []
    # include_dirs = [srcdir]
    libraries = ['ze_loader']
    # extra arguments
    # extra_link_args = []
    # create extension module
    # build extension module
    define_macros = [('TRITON_XPU_PROFILE', None)] if use_profile() else []

    # create extension module
    ext = SYCLExtension(name,
                        [src],
                        extra_compile_args=extra_compile_args,
                        libraries=libraries,
                        define_macros=define_macros)

    args = ['build_ext']
    args.append('--build-temp=' + srcdir)
    args.append('--build-lib=' + srcdir)
    args.append('-q')
    args = dict(
        name=name,
        ext_modules=[ext],
        cmdclass={
            'build_ext': SYCLBuildExtension},
        script_args=args,
    )
    # with quiet():
    setuptools.setup(**args)
    return so


#
# SYCL
#
class SYCLUtils(object):

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(SYCLUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        dirname = os.path.dirname(os.path.realpath(__file__))
        src = Path(os.path.join(dirname, "utils", "sycl.cpp")).read_text()
        key = hashlib.md5(src.encode("utf-8")).hexdigest()
        cache = get_cache_manager(key)
        fname = "sycl_utils.so"
        cache_path = cache.get_file(fname)
        if cache_path is None:
            with tempfile.TemporaryDirectory() as tmpdir:
                src_path = os.path.join(tmpdir, "main.cpp")
                with open(src_path, "w") as f:
                    f.write(src)
                so = _build_xpu_ext("sycl_utils", src_path, tmpdir)
                with open(so, "rb") as f:
                    cache_path = cache.put(f.read(), fname, binary=True)
        import importlib.util
        spec = importlib.util.spec_from_file_location("sycl_utils", cache_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.load_binary = mod.load_binary
        self.get_device_properties = mod.get_device_properties


class SYCLDriver(DriverBase):

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(SYCLDriver, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.utils = SYCLUtils()
        self.backend = "SYCL"
        self.binary_ext = "spvbin"
        self.get_current_device = torch.xpu.current_device
        self.set_current_device = torch.xpu.set_device
        self.get_current_stream = lambda idx: torch.xpu.current_stream(idx).sycl_queue
        self.get_device_capability = lambda idx: self.utils.get_device_properties(torch.xpu.device(idx).sycl_device)  # noqa: E501
        self.get_device_properties = self.get_device_capability
        self.device_id_mapping = dict()

    def load_binary(self, name, kernel, shared, device):
        return self.utils.load_binary(
            name, kernel, shared, torch.xpu.device(device).sycl_device)

    def get_current_target(self):
        device = self.get_current_device()
        capability = self.get_device_capability(device)
        return ("xpu", capability)

    def assemble_tensormap_to_arg(self, tensormaps_info, args):
        args_with_tma = list(args)
        return args_with_tma


@dataclass
class XPULinkerOptions:
    libs: dict = None

    def __post_init__(self):
        if self.libs is not None:
            self.libs = {k: v for k, v in self.libs.items() if v}


@dataclass(frozen=True)
class XPUOptions:
    num_warps: int = 4
    num_ctas: int = 1
    num_stages: int = 3
    # cluster_dims: tuple = (1, 1, 1)
    spirv_version: int = None
    dev_type: str = 'GPU'
    dev_id: int = None
    threads_per_warp: int = 16
    enable_warp_specialization: bool = False
    enable_persistent: bool = False
    optimize_epilogue: bool = False
    enable_fp_fusion: bool = True
    allow_fp8e4nv: bool = False
    max_num_imprecise_acc_default: bool = None
    extern_libs: dict = None
    debug: bool = False

    def __post_init__(self):
        if isinstance(self.extern_libs, dict):
            extern_libs = tuple([(k, v) for k, v in self.extern_libs.items() if v])
            object.__setattr__(self, 'extern_libs', extern_libs)
        assert self.num_warps > 0 and (self.num_warps & (self.num_warps - 1)) == 0, \
            "num_warps must be a power of 2"

    def hash(self):
        key = '_'.join([f'{name}-{val}' for name, val in self.__dict__.items()])
        return hashlib.md5(key.encode("utf-8")).hexdigest()


class XPUBackend(BaseBackend):
    stub_so_path = ""

    def __init__(self, device_type: str) -> None:
        super(XPUBackend, self).__init__(device_type)
        self.driver = SYCLDriver()

    def parse_options(self, _opts, target) -> Any:
        args = {k: _opts[k] for k in XPUOptions.__dataclass_fields__.keys() if k in _opts}
        assert isinstance(target, tuple), "the target must be tuple of (dev_type, capability)"
        capability = target[1]
        threads_per_warp = args['threads_per_warp'] if 'threads_per_warp' in args else 16
        sub_group_sizes = capability['sub_group_sizes']
        assert threads_per_warp in sub_group_sizes, "Device '{}' does not support threads_per_warp {}".format(
            capability['dev_name'],
            threads_per_warp)  # noqa: E501

        max_work_group_size = capability['max_work_group_size']
        num_warps = args['num_warps'] if 'num_warps' in args else max_work_group_size // threads_per_warp

        max_num_sub_groups = capability['max_num_sub_groups']
        assert num_warps <= max_num_sub_groups, \
            "invalid setting. max_work_group_size {}, max_num_subgroup {}, subgroup_sizes {}".format(  # noqa: E501
                max_work_group_size,
                max_num_sub_groups,
                threads_per_warp)

        args['num_warps'] = num_warps
        args['threads_per_warp'] = threads_per_warp
        args['dev_type'] = 'GPU'
        # args['dev_id'] = capability['dev_id']
        args['dev_id'] = self.get_device_properties(self.get_current_device())['dev_id']
        return XPUOptions(**args)

    def parse_linker_options(self, opts, capability):
        return XPULinkerOptions(**opts)

    @staticmethod
    def make_ttir(mod, metadata, opt):
        context = _triton.context()
        mod = _triton.parse_mlir_module(str(mod), context)
        mod.context = context
        pm = _triton.pass_manager(mod.context)
        pm.enable_debug()
        pm.add_inliner_pass()
        pm.add_triton_combine_pass()
        pm.add_canonicalizer_pass()
        pm.add_reorder_broadcast_pass()
        pm.add_cse_pass()
        pm.add_licm_pass()
        pm.add_symbol_dce_pass()
        pm.run(mod)
        return mod

    @staticmethod
    def make_ttgir(mod, metadata, opt):
        # TTIR -> TTGIR
        pm = _triton.pass_manager(mod.context)
        pm.enable_debug()
        pm.add_convert_triton_to_tritongpu_pass(opt.num_warps, opt.threads_per_warp, opt.num_ctas, asdict(opt))
        # optimize TTGIR
        pm.add_tritongpu_coalesce_pass()
        # TODO: add 2d block load/store optimize passes.
        pm.add_tritongpu_rewrite_tensor_pointer_pass(asdict(opt))
        pm.add_tritongpu_remove_layout_conversions_pass()
        if opt.optimize_epilogue:
            pm.add_tritongpu_optimize_epilogue_pass()
        pm.add_tritongpu_optimize_dot_operands_pass()
        pm.add_cse_pass()
        pm.add_tritongpu_pipeline_pass(opt.num_stages, opt.num_warps, opt.num_ctas, asdict(opt))
        # pm.add_tritongpu_materialize_load_store_pass(opt.num_warps, capability)
        # if capability // 10 <= 8:
        #     pm.add_tritongpu_prefetch_pass()
        pm.add_tritongpu_optimize_dot_operands_pass()
        pm.add_tritongpu_remove_layout_conversions_pass()
        pm.add_tritongpu_decompose_conversions_pass()
        # pm.add_tritongpu_ws_fixup_missing_attrs_pass()
        pm.add_tritongpu_reorder_instructions_pass()
        pm.add_cse_pass()
        pm.add_symbol_dce_pass()
        # if capability // 10 >= 9:
        #     pm.add_tritongpu_fence_insertion_pass()
        # pm.add_tritongpu_optimize_thread_locality_pass()
        pm.add_canonicalizer_pass()
        pm.run(mod)

        return mod

    @staticmethod
    def make_llir(src, metadata, opt):
        metadata["enable_warp_specialization"] = False
        metadata["num_warps"] = _triton.get_num_warps(src)
        # link libraries
        if opt.extern_libs:
            names = [lib[0] for lib in opt.extern_libs]
            paths = [lib[1] for lib in opt.extern_libs]
            _add_external_libs(src, names, paths)
        # TritonGPU -> LLVM-IR
        ret = _triton.translate_triton_gpu_to_spirv(src, asdict(opt))
        metadata["shared"] = _triton.get_shared_memory_size(src)
        metadata["threads_per_warp"] = _triton.get_threads_per_warp(src)
        metadata["name"] = spirv_get_kernel_name(ret)
        metadata["cluster_dims"] = (1, 1, 1)
        return ret

    @staticmethod
    def make_spirv(src, metadata, opt):
        return _triton.translate_llvmir_to_spirv(src, asdict(opt))

    @staticmethod
    def make_spvbin(src, metadata, opt):
        metadata["name"] = spirv_get_kernel_name(src)
        return _triton.compile_spirv_to_spvbin(src, asdict(opt))

    def add_stages(self, stages, options):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options)
        stages["spvbin"] = lambda src, metadata: self.make_spvbin(src, metadata, options)

    def add_meta_info(self, ir, module, next_module, metadata, asm):
        if ir == "spirv":
            metadata["name"] = spirv_get_kernel_name(next_module)
            if "shared" not in metadata:
                metadata["shared"] = module.share_memory_size

        if ir == "spvbin":
            asm[ir] = next_module

    def get_driver(self):
        return self.driver

    def get_stream(self, idx=None):
        if idx is None:
            idx = self.get_current_device()
        return torch.xpu.current_stream(idx).sycl_queue

    @functools.lru_cache(None)
    def get_device_properties(self, device):
        return self.driver.utils.get_device_properties(torch.xpu.device(device).sycl_device)  # noqa: E501

    def get_current_device(self):
        return torch.xpu.current_device()

    def set_current_device(self, device):
        torch.xpu.set_device(device)

    def get_load_binary_fn(self):

        def _load_binary_fn(kernel_name, binary, shared_size, device):
            return self.driver.utils.load_binary(kernel_name, binary, shared_size, torch.xpu.device(device).sycl_device)  # noqa: E501

        return _load_binary_fn

    def get_kernel_bin(self):
        return "spvbin"

    def get_architecture_descriptor(self, **kwargs):
        arch = kwargs.get("cc", None)
        if arch is None:
            arch = self.get_device_properties(self.get_current_device())
        max_work_group_size = arch['max_work_group_size']
        max_num_sub_groups = arch['max_num_sub_groups']
        sub_group_sizes = arch['sub_group_sizes']
        # TODO: chose a reasonable subgroup size
        threads_per_warp = 32
        assert threads_per_warp in sub_group_sizes, "Current platform does not support threads_per_warp to be 32"  # noqa: E501
        num_warps = max_work_group_size // threads_per_warp
        assert num_warps < max_num_sub_groups, \
            "invalid setting. max_work_group_size {}, max_num_subgroup {}, subgroup_sizes {}".format(  # noqa: E501
                max_work_group_size,
                max_num_sub_groups,
                max_num_sub_groups)
        capability = {"num_warps": num_warps, "threads_per_warp": threads_per_warp, "num_stages": 2}  # noqa: E501
        return capability

    def make_launcher_stub(self, src, metadata):
        # name of files that are cached
        name = src.name
        signature = src.signature
        ids = {
            "ids_of_tensormaps": metadata.get("ids_of_tensormaps", tuple()), "ids_of_folded_args":
                metadata.get("ids_of_folded_args",
                             tuple()), "ids_of_const_exprs": src.fn.constexprs if hasattr(src, "fn") else tuple()
        }
        constants = src.constants if hasattr(src, "constants") else dict()
        so_cache_key = make_so_cache_key(version_key(), signature, constants, ids)
        so_cache_manager = get_cache_manager(so_cache_key)
        so_name = f"{name}.so"
        # retrieve stub from cache if it exists
        cache_path = so_cache_manager.get_file(so_name)
        if cache_path is None:
            with tempfile.TemporaryDirectory() as tmpdir:
                src = generate_launcher(signature, constants, ids)
                src_path = os.path.join(tmpdir, "main.cpp")
                with open(src_path, "w") as f:
                    f.write(src)
                so = _build_xpu_ext(name, src_path, tmpdir)
                with open(so, "rb") as f:
                    return so_cache_manager.put(f.read(), so_name, binary=True)
        else:
            return cache_path

    def get_version_key(self):
        return version_key()

    def hash(self):
        # return f'{get_version_key()}-{self.capability}'
        return f'{self.get_version_key()}'


register_backend("xpu", XPUBackend)
