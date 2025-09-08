import importlib.metadata
import os
import re
import hashlib
import shutil
import ctypes
import sysconfig
import tempfile
from pathlib import Path
from functools import cached_property

from triton import knobs
from triton.runtime.build import _build, platform_key, _load_module_from_path
from triton.runtime.cache import get_cache_manager
from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase
from triton.tools.tensor_descriptor import TensorDescriptor

# A hard-coded cache version that can be updated when we know that the cached file is invalid and
# there are no other ways to detect that the runtime environment has changed. For example, a shared
# library has been updated as a result of updated dependencies.
# See https://github.com/intel/intel-xpu-backend-for-triton/issues/3095.
__CACHE_VERSION = "1"


def find_sycl(include_dir: list[str]) -> tuple[list[str], list[str]]:
    """
    Looks for the sycl library in known places.

    Arguments:
      include_dir: list of include directories to pass to compiler.

    Returns:
      enriched include_dir and libsycl.so location.

    Raises:
      AssertionError: if library was not found.
    """
    include_dir = include_dir.copy()
    assertion_message = ("sycl headers not found, please install `icpx` compiler, "
                         "or provide `ONEAPI_ROOT` environment "
                         "or install `intel-sycl-rt>=2025.0.0` wheel")
    icpx_path = shutil.which("icpx")
    if icpx_path:
        # only `icpx` compiler knows where sycl runtime binaries and header files are
        compiler_root = os.path.abspath(f"{icpx_path}/../..")
        include_dir += [os.path.join(compiler_root, "include"), os.path.join(compiler_root, "include/sycl")]
        sycl_dir = os.path.join(compiler_root, "lib")
        return include_dir, [sycl_dir]

    oneapi_root = os.getenv("ONEAPI_ROOT")
    if oneapi_root:
        include_dir += [
            os.path.join(oneapi_root, "compiler/latest/include"),
            os.path.join(oneapi_root, "compiler/latest/include/sycl")
        ]
        sycl_dir = os.path.join(oneapi_root, "compiler/latest/lib")
        return include_dir, [sycl_dir]

    try:
        sycl_rt = importlib.metadata.metadata("intel-sycl-rt")
    except importlib.metadata.PackageNotFoundError:
        raise AssertionError(assertion_message)

    if sycl_rt.get("version", "0.0.0").startswith("2024"):
        raise AssertionError(assertion_message)

    sycl_dirs = []
    for f in importlib.metadata.files("intel-sycl-rt"):
        # sycl/sycl.hpp and sycl/CL/sycl.hpp results in both folders
        # being add: include and include/sycl.
        if f.name == "sycl.hpp":
            include_dir += [str(f.locate().parent.parent.resolve())]
        if f.name in ["libsycl.so", "sycl8.dll", "sycl8.lib"]:
            sycl_dir = str(f.locate().parent.resolve())
            # should we handle `_` somehow?
            if os.name == "nt":
                _ = os.add_dll_directory(sycl_dir)
            sycl_dirs.append(sycl_dir)

    assert len(sycl_dirs) != 0
    return include_dir, sycl_dirs


class CompilationHelper:
    _library_dir: list[str]
    _include_dir: list[str]
    libraries: list[str]

    def __init__(self):
        self._library_dir = None
        self._include_dir = None
        self._libsycl_dir = None
        self.libraries = ['ze_loader']
        if os.name != "nt":
            self.libraries += ["sycl"]
        else:
            self.libraries += ['sycl8']

    @property
    def inject_pytorch_dep(self):
        return os.environ.get("INJECT_PYTORCH", "False") == "True"

    @cached_property
    def _compute_compilation_options_lazy(self):
        ze_root = os.getenv("LEVEL_ZERO_V1_SDK_PATH")
        if ze_root is None:
            ze_root = os.getenv("ZE_PATH", default="/usr/local")
        include_dir = [os.path.join(ze_root, "include")]

        library_dir = []
        include_dir, self._libsycl_dir = find_sycl(include_dir)
        if self._libsycl_dir:
            library_dir += self._libsycl_dir
        if os.name == "nt":
            library_dir += [os.path.join(ze_root, "lib")]

        dirname = os.path.dirname(os.path.realpath(__file__))
        include_dir += [os.path.join(dirname, "include")]
        library_dir += [os.path.join(dirname, "lib")]

        if self.inject_pytorch_dep:
            import torch

            torch_path = torch.utils.cmake_prefix_path
            include_dir += [
                os.path.join(torch_path, "../../include"),
                os.path.join(torch_path, "../../include/torch/csrc/api/include"),
            ]
            library_dir += [os.path.join(torch_path, "../../lib")]
            self.libraries += ['torch']

        self._library_dir = library_dir
        self._include_dir = include_dir

    @cached_property
    def library_dir(self) -> list[str]:
        self._compute_compilation_options_lazy
        return self._library_dir

    @cached_property
    def include_dir(self) -> list[str]:
        self._compute_compilation_options_lazy
        return self._include_dir

    @cached_property
    def libsycl_dir(self) -> list[str]:
        self._compute_compilation_options_lazy
        return self._libsycl_dir


COMPILATION_HELPER = CompilationHelper()


class ArchParser:

    def __init__(self, cache_path: str):
        self.shared_library = ctypes.CDLL(cache_path)
        self.shared_library.parse_device_arch.restype = ctypes.c_char_p
        self.shared_library.parse_device_arch.argtypes = (ctypes.c_uint64, )

    def __getattribute__(self, name):
        if name == "parse_device_arch":
            shared_library = super().__getattribute__("shared_library")
            attr = getattr(shared_library, name)

            def wrapper(*args, **kwargs):
                return attr(*args, **kwargs).decode("utf-8")

            return wrapper

        return super().__getattribute__(name)

    if os.name != 'nt':

        def __del__(self):
            handle = self.shared_library._handle
            self.shared_library.dlclose.argtypes = (ctypes.c_void_p, )
            self.shared_library.dlclose(handle)
    else:

        def __del__(self):
            handle = self.shared_library._handle
            ctypes.windll.kernel32.FreeLibrary.argtypes = (ctypes.c_uint64, )
            ctypes.windll.kernel32.FreeLibrary(handle)


class SpirvUtils:

    def __init__(self, cache_path: str):
        self.shared_library = ctypes.PyDLL(cache_path)
        methods = ("init_devices", "load_binary", "wait_on_sycl_queue", "has_opencl_extension")
        for method in methods:
            getattr(self.shared_library, method).restype = ctypes.py_object
            getattr(self.shared_library, method).argtypes = (ctypes.py_object, )
        self.shared_library.get_device_properties.restype = ctypes.py_object
        self.shared_library.get_device_properties.argtypes = (ctypes.c_int, )
        self.shared_library.has_opencl_extension.restype = ctypes.py_object
        self.shared_library.has_opencl_extension.argtypes = (ctypes.c_int, ctypes.c_char_p)

    def __getattribute__(self, name):
        if name in ("get_device_properties", "init_devices", "wait_on_sycl_queue", "has_opencl_extension"):
            shared_library = super().__getattribute__("shared_library")
            return getattr(shared_library, name)

        return super().__getattribute__(name)

    def load_binary(self, *args):
        # if we don't use parameter passing in this way,
        # we will need to rewrite the line in the general part of the code:
        # driver.active.utils.load_binary(self.name, self.kernel, self.metadata.shared, self.metadata.build_flags, device) ->
        # driver.active.utils.load_binary((self.name, self.kernel, self.metadata.shared, self.metadata.build_flags, device))
        return self.shared_library.load_binary(args)

    if os.name != 'nt':

        def __del__(self):
            handle = self.shared_library._handle
            self.shared_library.dlclose.argtypes = (ctypes.c_void_p, )
            self.shared_library.dlclose(handle)
    else:

        def __del__(self):
            handle = self.shared_library._handle
            ctypes.windll.kernel32.FreeLibrary.argtypes = (ctypes.c_uint64, )
            ctypes.windll.kernel32.FreeLibrary(handle)


class TritonLauncher:

    def __init__(self, cache_path: str):
        self.shared_library = ctypes.PyDLL(cache_path)
        self.shared_library.launch.restype = ctypes.py_object
        self.shared_library.launch.argtypes = (ctypes.py_object, )

    def __getattribute__(self, name):
        if name == "launch":
            shared_library = super().__getattribute__("shared_library")
            return getattr(shared_library, name)

        return super().__getattribute__(name)

    if os.name != 'nt':

        def __del__(self):
            handle = self.shared_library._handle
            self.shared_library.dlclose.argtypes = (ctypes.c_void_p, )
            self.shared_library.dlclose(handle)
    else:

        def __del__(self):
            handle = self.shared_library._handle
            ctypes.windll.kernel32.FreeLibrary.argtypes = (ctypes.c_uint64, )
            ctypes.windll.kernel32.FreeLibrary(handle)


def compile_module_from_src(src: str, name: str):
    hasher = hashlib.sha256(__CACHE_VERSION.encode("utf-8"))
    hasher.update((src + platform_key()).encode("utf-8"))
    key = hasher.hexdigest()
    cache = get_cache_manager(key)
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    cache_path = cache.get_file(f"{name}{suffix}")
    if cache_path is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "main.cpp")
            with open(src_path, "w") as f:
                f.write(src)
            extra_compiler_args = []
            if COMPILATION_HELPER.libsycl_dir:
                if os.name == "nt":
                    extra_compiler_args += ["/LIBPATH:" + dir for dir in COMPILATION_HELPER.libsycl_dir]
                else:
                    extra_compiler_args += ["-Wl,-rpath," + dir for dir in COMPILATION_HELPER.libsycl_dir]

            so = _build(name, src_path, tmpdir, COMPILATION_HELPER.library_dir, COMPILATION_HELPER.include_dir,
                        COMPILATION_HELPER.libraries, ccflags=extra_compiler_args)
            with open(so, "rb") as f:
                cache_path = cache.put(f.read(), f"{name}{suffix}", binary=True)

    if name == 'arch_utils':
        return ArchParser(cache_path)
    if name == 'spirv_utils':
        return SpirvUtils(cache_path)
    if name == '__triton_launcher':
        return TritonLauncher(cache_path)
    if name == 'proton_utils':
        return cache_path

    return _load_module_from_path(name, cache_path)


# ------------------------
# Utils
# ------------------------


class XPUUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(XPUUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        dirname = os.path.dirname(os.path.realpath(__file__))
        # we save `spirv_utils` module so that the destructor is not called prematurely, which will unload the dll
        # and can cause `Fatal Python error: Segmentation fault`
        mod = compile_module_from_src(src=Path(os.path.join(dirname, "driver.c")).read_text(), name="spirv_utils")
        self.load_binary = mod.load_binary
        self.get_device_properties = mod.get_device_properties
        self.device_count = mod.init_devices(self.get_sycl_queue())
        self.wait_on_sycl_queue = mod.wait_on_sycl_queue
        self.has_opencl_extension = mod.has_opencl_extension

    def get_current_device(self):
        import torch
        return torch.xpu.current_device()

    def get_sycl_queue(self):
        import torch
        return torch.xpu.current_stream().sycl_queue

    def wait(self):
        self.wait_on_sycl_queue(self.get_sycl_queue())


# ------------------------
# Launcher
# ------------------------


def ty_to_cpp(ty):
    if ty[0] == '*':
        return "void*"
    return {
        "i1": "int8_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint8_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "double",
        "bf16": "double",
        "fp32": "double",
        "f32": "double",
        "fp64": "double",
    }[ty]


FLOAT_STORAGE_TYPE = {
    "fp16": "uint16_t",
    "bf16": "uint16_t",
    "fp32": "uint32_t",
    "f32": "uint32_t",
    "fp64": "uint64_t",
}
FLOAT_PACK_FUNCTION = {
    "fp16": "pack_fp16",
    "bf16": "pack_bf16",
    "fp32": "pack_fp32",
    "f32": "pack_fp32",
    "fp64": "pack_fp64",
}

_BASE_ARGS_FORMAT = "iiiOOOOOO"


def make_launcher(constants, signature):

    def _expand_signature(signature):
        output = []
        # Expand tensor descriptor arguments into base pointer, shape, and
        # strides
        for sig in signature:
            if isinstance(sig, str) and sig.startswith("tensordesc"):
                match = re.match("tensordesc<([^[>]*)\\[([^]]*)\\]", sig)
                dtype = match.group(1)
                shape = match.group(2)
                ndim = shape.count(",") + 1

                output.append("*" + dtype)
                # Currently the host side tensor descriptors get passed in as a
                # tensor desc, shape, and strides. We have no way to use these
                # shape and strides when processing tensor descriptors which is
                # why we provide our own decomposition above. Sadly this means
                # we have to pass the shape and strides twice.
                for _ in range(2 * ndim):
                    output.append("i64")
                output.append("i1")
                for _ in range(ndim):
                    output.append("i32")
                for _ in range(ndim):
                    output.append("i64")
            else:
                output.append(sig)

        return output

    def _flatten_signature(sig, output):
        if isinstance(sig, tuple):
            for x in sig:
                _flatten_signature(x, output)
        else:
            output.append(sig)

    def _extracted_type(ty):
        if isinstance(ty, tuple):
            val = ','.join(map(_extracted_type, ty))
            return f"[{val}]"
        if ty[0] == '*':
            return "PyObject*"
        if ty in ("constexpr"):
            return "PyObject*"
        return ty_to_cpp(ty)

    def format_of(ty):
        if isinstance(ty, tuple):
            val = ''.join(map(format_of, ty))
            return f"({val})"
        if ty[0] == '*':
            return "O"
        if ty in ("constexpr"):
            return "O"
        if ty == "void*":
            return "O"
        return {
            "double": "d",
            "long": "l",
            "int8_t": "b",
            "int16_t": "h",
            "int32_t": "i",
            "int64_t": "L",
            "uint8_t": "B",
            "uint16_t": "H",
            "uint32_t": "I",
            "uint64_t": "K",
        }[ty_to_cpp(ty)]

    expand_signature = _expand_signature(signature.values())
    signature = {i: s for i, s in enumerate(expand_signature)}

    args_format = ''.join([format_of(ty) for ty in signature.values()])
    format = _BASE_ARGS_FORMAT + args_format

    flat_signature = []
    for sig in signature.values():
        _flatten_signature(sig, flat_signature)
    signature = {i: s for i, s in enumerate(flat_signature)}
    args_list = ', ' + ', '.join(f"&_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''
    # Record the end of regular arguments;
    # subsequent arguments are architecture-specific descriptors.
    arg_decl_list = []
    for i, ty in signature.items():
        if ty == "constexpr":
            continue
        if ty in FLOAT_STORAGE_TYPE:
            arg_decl_list.append(f"{FLOAT_STORAGE_TYPE[ty]} arg{i}")
        else:
            arg_decl_list.append(f"{ty_to_cpp(ty)} arg{i}")
    arg_decls = ', '.join(arg_decl_list)
    internal_args_list = []
    for i, ty in signature.items():
        if ty[0] == "*":
            internal_args_list.append(f"ptr_info{i}.dev_ptr")
        elif ty in FLOAT_STORAGE_TYPE:
            internal_args_list.append(f"_arg{i}_storage")
        elif ty != "constexpr":
            internal_args_list.append(f"_arg{i}")

    # generate glue code
    newline = '\n  '
    ptr_decls = [
        f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}, stream); if (!ptr_info{i}.valid) return NULL;"
        for i, ty in signature.items()
        if ty[0] == "*"
    ]
    float_storage_decls = [
        f"{FLOAT_STORAGE_TYPE[ty]} _arg{i}_storage = {FLOAT_PACK_FUNCTION[ty]}(_arg{i});"
        for i, ty in signature.items()
        if ty in FLOAT_STORAGE_TYPE
    ]
    params = [f"&arg{i}" for i, ty in signature.items() if ty != "constexpr"]
    params.append("&global_scratch")
    params.append("&profile_scratch")
    num_params = len(params)
    params_decl = ""
    if num_params:
        params_decl = f"void *params[] = {{ {', '.join(params)} }};"
    src = f"""
#include <cstddef>
#include <Python.h>
#include <string>
#include <iostream>
#include <iomanip>
#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>
{ "#include <ATen/record_function.h>" if COMPILATION_HELPER.inject_pytorch_dep else "" }

#if defined(_WIN32)
#define EXPORT_FUNC __declspec(dllexport)
#else
#define EXPORT_FUNC __attribute__((visibility("default")))
#endif

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <stdio.h>
#include <numpy/arrayobject.h>

namespace {{

bool getBoolEnv(const std::string &env) {{
            const char *s = std::getenv(env.c_str());
            std::string str(s ? s : "");
            std::transform(str.begin(), str.end(), str.begin(),
                            [](unsigned char c) {{ return std::tolower(c); }});
            return (str == "on" || str == "true" || str == "1");
}}

}}


static inline void gpuAssert(ze_result_t code, const char *file, int line)
{{
  if (code != ZE_RESULT_SUCCESS)
  {{
    const char* prefix = "Triton Error [ZE]: ";
    std::string str = std::to_string(code);
    char err[1024] = {{0}};
    strcat(err, prefix);
    strcat(err, str.c_str());
    PyErr_SetString(PyExc_RuntimeError, err);
  }}
}}

#define ZE_CHECK(ans) {{ gpuAssert((ans), __FILE__, __LINE__); }}

typedef struct _DevicePtrInfo {{
  void* dev_ptr;
  bool valid;
}} DevicePtrInfo;

static inline void checkDevicePointer(DevicePtrInfo *ptr_info, int idx, const sycl::queue &queue) {{
  if (!ptr_info->dev_ptr || !ptr_info->valid) {{
    return;
  }}
  auto context = queue.get_context();
  auto handle = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(context);
  ze_memory_allocation_properties_t prop;
  prop.stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
  prop.pNext = nullptr;
  ze_device_handle_t device;
  auto res = zeMemGetAllocProperties((ze_context_handle_t)handle, ptr_info->dev_ptr, &prop, &device);
  if (res != ZE_RESULT_SUCCESS) {{
    PyErr_Format(PyExc_ValueError,
                 "Cannot get memory properties for pointer argument (at %d, err=%d)", idx, res);
    ptr_info->valid = false;
  }} else if (prop.type != ZE_MEMORY_TYPE_DEVICE) {{
    PyErr_Format(PyExc_ValueError,
                 "Pointer argument (at %d) doesn't reference XPU device memory (cpu tensor?)", idx);
    ptr_info->valid = false;
  }}
}}

static inline DevicePtrInfo getPointer(PyObject *obj, int idx, const sycl::queue &queue) {{
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {{
    ptr_info.dev_ptr = PyLong_AsVoidPtr(obj);
    checkDevicePointer(&ptr_info, idx, queue);
    return ptr_info;
  }}
  if (obj == Py_None) {{
    // valid nullptr
    return ptr_info;
  }}
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  if(ptr){{
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {{
      PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
      ptr_info.valid = false;
      return ptr_info;
    }}
    ptr_info.dev_ptr = PyLong_AsVoidPtr(ret);
    if(!ptr_info.dev_ptr) {{
      return ptr_info;
    }}
    checkDevicePointer(&ptr_info, idx, queue);
    Py_DECREF(ret);  // Thanks ChatGPT!
    return ptr_info;
  }}
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  ptr_info.valid = false;
  return ptr_info;
}}

// start sycl
template <class T>
static inline void set_scalar_arg(sycl::handler &cgh, int index, const void *value) {{
  cgh.set_arg(index, *static_cast<const T *>(value));
}}

static void sycl_kernel_launch(uint32_t gridX, uint32_t gridY, uint32_t gridZ,
                               int num_warps, int threads_per_warp, int shared_memory,
                               sycl::queue& stream, sycl::kernel& kernel_ptr,
                               void* global_scratch, void* profile_scratch{', ' + arg_decls if len(arg_decls) > 0 else ''}) {{

  std::string kernel_name = kernel_ptr.get_info<sycl::info::kernel::function_name>();
  { 'RECORD_FUNCTION("XPU Triton kernel:" + kernel_name, {});' if COMPILATION_HELPER.inject_pytorch_dep else "" }

  {params_decl};
  uint32_t num_params = {num_params};
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
  if (shared_memory) {{
    expected_num_params -= 1;
  }}

  static bool launchDebug = getBoolEnv("TRITON_INTEL_LAUNCH_DEBUG");
  if (launchDebug){{
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
    {" ".join(f'std::cout << "  param {idx}:" << *({ty_to_cpp(item)}*)params[{idx}] << std::endl;' for idx, item in enumerate([signature[i] for i in signature if signature[i] != "constexpr"]))}
  }}
  assert(num_params == expected_num_params && "number of kernel param not matched");
  // Submit the imported kernel.
  auto cgf = [&](sycl::handler &cgh) {{
    {" ".join(f'set_scalar_arg<{ty_to_cpp(item)}>(cgh, {idx}, params[{idx}]);' for idx, item in enumerate([signature[i] for i in signature if signature[i] != "constexpr"]))}
    if (shared_memory) {{
      using share_mem_t = sycl::local_accessor<int8_t, 1>;
      share_mem_t local_buffer = share_mem_t(shared_memory, cgh);
      cgh.set_arg(num_params, local_buffer);
      cgh.parallel_for(parallel_work_size, kernel_ptr);
    }} else {{
      cgh.parallel_for(parallel_work_size, kernel_ptr);
    }}
  }};
  auto event = stream.submit(cgf);
}}
// end sycl

static uint16_t pack_fp16(double f) {{
    uint16_t result;
    // from https://github.com/python/pythoncapi-compat
#if 0x030600B1 <= PY_VERSION_HEX && PY_VERSION_HEX <= 0x030B00A1 && !defined(PYPY_VERSION)
    _PyFloat_Pack2(f, (unsigned char *)&result, 1);
#else
    PyFloat_Pack2(f, (char*)&result, 1);
#endif
    return result;
}}

static uint16_t pack_bf16(double f) {{
    float f32 = (float)f;
    uint32_t u32 = *(uint32_t*)&f32;
    return (uint16_t)(u32 >> 16);
}}

static uint32_t pack_fp32(double f) {{
    float f32 = (float)f;
    return *(uint32_t*)&f32;
}}

static uint64_t pack_fp64(double f) {{
    return *(uint64_t*)&f;
}}

extern "C" EXPORT_FUNC PyObject* launch(PyObject* args) {{
  int gridX, gridY, gridZ;
  void* global_scratch = nullptr;
  void* profile_scratch = nullptr;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *kernel_metadata = NULL;
  PyObject *launch_metadata = NULL;
  PyObject *py_obj_stream;
  PyObject* py_kernel;

  {newline.join([f"{_extracted_type(ty)} _arg{i};" for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ,
                                           &py_obj_stream, &py_kernel,
                                           &kernel_metadata, &launch_metadata,
                                           &launch_enter_hook, &launch_exit_hook{args_list})) {{
    return NULL;
  }}

  // extract kernel metadata
  PyObject *num_warps_attr = PyObject_GetAttrString(kernel_metadata, "num_warps");
  int num_warps = PyLong_AsLong(num_warps_attr);
  Py_DECREF(num_warps_attr);
  PyObject *num_ctas_attr = PyObject_GetAttrString(kernel_metadata, "num_ctas");
  int num_ctas = PyLong_AsLong(num_ctas_attr);
  Py_DECREF(num_ctas_attr);
  PyObject *shared_attr = PyObject_GetAttrString(kernel_metadata, "shared");
  int shared_memory = PyLong_AsLong(shared_attr);
  Py_DECREF(shared_attr);
  PyObject *threads_per_warp_attr = PyObject_GetAttrString(kernel_metadata, "threads_per_warp");
  int threads_per_warp = PyLong_AsLong(threads_per_warp_attr);
  Py_DECREF(threads_per_warp_attr);

  // extract cluster dims
  PyObject *clusterDim =  PyObject_GetAttrString(kernel_metadata, "cluster_dims");
  if (!PyTuple_Check(kernel_metadata)) {{
    PyErr_SetString(PyExc_TypeError, "kernel_metadata.cluster_dims must be a tuple");
    return NULL;
  }}
  int clusterDimX   = PyLong_AsLong(PyTuple_GetItem(clusterDim, 0));
  int clusterDimY   = PyLong_AsLong(PyTuple_GetItem(clusterDim, 1));
  int clusterDimZ   = PyLong_AsLong(PyTuple_GetItem(clusterDim, 2));
  Py_DECREF(clusterDim);
  // extract launch metadata
  if (launch_enter_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_enter_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
    Py_DECREF(ret);
  }}

  void * pStream = PyLong_AsVoidPtr(py_obj_stream);
  //error check
  if(pStream == nullptr || py_kernel == nullptr) return NULL;

  sycl::queue stream = *(static_cast<sycl::queue*>(pStream));
  sycl::kernel* kernel_ptr = reinterpret_cast<sycl::kernel*>(PyCapsule_GetPointer(py_kernel, "kernel"));
  if(kernel_ptr == nullptr) return NULL;
  sycl::kernel kernel = *kernel_ptr;

  {newline.join(ptr_decls)}
  {newline.join(float_storage_decls)}
  sycl_kernel_launch(gridX, gridY, gridZ, num_warps, threads_per_warp, shared_memory, stream, kernel, global_scratch, profile_scratch{',' + ', '.join(internal_args_list) if len(internal_args_list) > 0 else ''});
  if (PyErr_Occurred()) {{
    return NULL;
  }}

  if(launch_exit_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_exit_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
    Py_DECREF(ret);
  }}

  Py_RETURN_NONE;
}}
"""
    return src


def wrap_handle_tensor_descriptor(launcher):
    """
    Replace all tensor descriptors with the base ptr, shape, and strides
    """

    def inner(args):
        meta_args = args[:len(_BASE_ARGS_FORMAT)]
        raw_kernel_args = args[len(_BASE_ARGS_FORMAT):]
        final_args = []
        for arg in raw_kernel_args:
            if isinstance(arg, TensorDescriptor):
                # Currently the host side tensor descriptors get decomposed in
                # the frontend to tensor desc, shape, and strides. We have no
                # way to use these shape and strides when processing tensor
                # descriptors which is why we provide our own decomposition
                # above. Sadly this means we have to pass the shape and strides
                # twice.
                final_args.extend([arg.base, *arg.shape, *arg.strides, arg.padding == "nan", *arg.shape, *arg.strides])
            else:
                final_args.append(arg)

        return launcher(meta_args + tuple(final_args))

    return inner


def serialize_args(args, constants, signature):
    import torch
    import numbers
    dir_path = knobs.intel.dump_spirv_kernel_args
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Path to directory consisting of SPIR-V Runner data: {dir_path}")

    def serialize_kernel_metadata(arg, args_dict):
        args_dict['num_warps'] = arg.num_warps
        args_dict['threads_per_warp'] = arg.threads_per_warp
        args_dict['shared_memory'] = arg.shared
        args_dict['kernel_name'] = arg.name
        args_dict['spv_name'] = f"{arg.name}.spv"
        args_dict['build_flags'] = arg.build_flags

    cnt = 0
    args_dict = {"gridX": int(args[cnt]), "gridY": int(args[cnt + 1]), "gridZ": int(args[cnt + 2])}
    # 3: stream
    # 4: function
    # 5: packed kernel metadata
    assert type(args[cnt + 5]).__name__ == "KernelMetadata"
    serialize_kernel_metadata(args[cnt + 5], args_dict)
    # 6: launch_metadata
    # 7: launch_enter_hook
    # 8: launch_exit_hook
    args_dict['argument_list'] = []
    counts = {"tensors": 0, "scalars": 0, "karg_cnt": 0}
    cnt += 9
    for arg in args[cnt:]:
        sig_name = list(signature.keys())[counts['karg_cnt']]
        if isinstance(arg, torch.Tensor):
            cpu_tensor = arg.cpu()
            tensor_path = os.path.join(dir_path, f"tensor_{counts['tensors']}.pt")
            with open(tensor_path, 'wb') as f:
                torch.save(cpu_tensor, f)
            new_arg = {
                "name": f"tensor_{counts['tensors']}", "type": "tensor", "dtype": str(arg.dtype), "ctype":
                signature[sig_name]
            }
            args_dict['argument_list'].append(new_arg)
            counts['tensors'] += 1
        if isinstance(arg, numbers.Number):
            if (counts['karg_cnt'], ) not in constants.keys():
                new_arg = {
                    "name": f"scalarArg_{counts['scalars']}", "type": "scalar", "value": arg, "ctype":
                    signature[sig_name]
                }
                args_dict['argument_list'].append(new_arg)
            counts['scalars'] += 1
        counts['karg_cnt'] += 1

    # Dump argument info as a JSON file
    json_path = os.path.join(dir_path, 'args_data.json')
    with open(json_path, 'w') as json_file:
        import json
        json.dump(args_dict, json_file, indent=4)


class XPULauncher(object):

    def __init__(self, src, metadata):
        constants = src.constants if hasattr(src, "constants") else dict()
        arg_idx = lambda x: (src.fn.arg_names.index(x), ) if isinstance(x, str) else x
        constants = {arg_idx(idx): value for idx, value in constants.items()}
        signature = {idx: value for idx, value in src.signature.items()}
        src = make_launcher(constants, signature)
        self.mod = compile_module_from_src(src=src, name="__triton_launcher")
        has_tensor_desc_arg = any(isinstance(sig, str) and sig.startswith("tensordesc") for sig in signature.values())

        self.launch = wrap_handle_tensor_descriptor(self.mod.launch) if has_tensor_desc_arg else self.mod.launch

        # Serialize KernelArguments for SPIR-V Runner
        self.serialize_kernel_args = knobs.intel.dump_spirv_kernel_args
        self.constants = constants
        self.signature = signature

    def __call__(self, *args):
        if self.serialize_kernel_args:
            serialize_args(args, self.constants, self.signature)
        self.launch(args)


class XPUDriver(DriverBase):

    def __init__(self):
        self.launcher_cls = XPULauncher
        super().__init__()

    def __getattr__(self, name):
        # Lazily initialize utils to avoid unnecessary XPU runtime invocations.
        # See https://github.com/intel/intel-xpu-backend-for-triton/issues/624
        if name == "utils":
            self.utils = XPUUtils()
            return self.utils
        else:
            raise AttributeError

    def get_current_device(self):
        return self.utils.get_current_device()

    def get_current_stream(self, device):
        import torch
        return torch.xpu.current_stream().sycl_queue

    def get_current_target(self):
        import torch
        device = self.get_current_device()
        dev_property = torch.xpu.get_device_capability(device)

        def update_advanced_features(device, dev_property):
            if knobs.intel.device_extensions:
                # May be useful when using the `TRITON INTEL_DEVICE_ARCH` environment variable
                # to be able to flexibly turn on/off the advanced feature.
                supported_extensions = set()
                supported_extensions.update(knobs.intel.device_extensions.split(" "))
                dev_property[
                    "has_subgroup_matrix_multiply_accumulate"] = "cl_intel_subgroup_matrix_multiply_accumulate" in supported_extensions
                dev_property[
                    "has_subgroup_matrix_multiply_accumulate_tensor_float32"] = "cl_intel_subgroup_matrix_multiply_accumulate_tensor_float32" in supported_extensions
                dev_property["has_subgroup_2d_block_io"] = "cl_intel_subgroup_2d_block_io" in supported_extensions
                dev_property["has_bfloat16_conversions"] = "cl_intel_bfloat16_conversions" in supported_extensions
            else:
                check = self.utils.has_opencl_extension
                dev_property["has_subgroup_matrix_multiply_accumulate"] = check(
                    device, b"cl_intel_subgroup_matrix_multiply_accumulate")
                dev_property["has_subgroup_matrix_multiply_accumulate_tensor_float32"] = check(
                    device, b"cl_intel_subgroup_matrix_multiply_accumulate_tensor_float32")
                dev_property["has_subgroup_2d_block_io"] = check(device, b"cl_intel_subgroup_2d_block_io")
                dev_property["has_bfloat16_conversions"] = check(device, b"cl_intel_bfloat16_conversions")

        update_advanced_features(device, dev_property)
        return GPUTarget("xpu", dev_property, warp_size=32)

    def build_proton_help_lib(self):
        from triton.backends.intel.driver import compile_module_from_src

        dirname = os.path.dirname(os.path.realpath(__file__))
        return compile_module_from_src(src=Path(dirname).joinpath("proton_utils.cpp").read_text(), name="proton_utils")

    def get_active_torch_device(self):
        import torch
        return torch.device("xpu", self.get_current_device())

    def get_device_interface(self):
        import torch
        return torch.xpu

    @staticmethod
    def is_active():
        try:
            import torch
            return torch.xpu.is_available()
        except ImportError:
            return False

    def map_python_to_cpp_type(self, ty: str) -> str:
        return ty_to_cpp(ty)

    def get_benchmarker(self):
        from triton.testing import do_bench
        return do_bench

    def get_empty_cache_for_benchmark(self):
        import torch

        # We maintain a buffer of 256 MB that we clear
        # before each kernel call to make sure that the L2 cache
        # doesn't contain any input data before the run
        cache_size = 256 * 1024 * 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device='xpu')

    def clear_cache(self, cache):
        cache.zero_()
