import os
import hashlib
import importlib.util
import tempfile
from pathlib import Path

from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase
from triton.runtime.cache import get_cache_manager
from triton.runtime.build import _build, quiet

import torch
import intel_extension_for_pytorch

_dirname = os.getenv("ZE_PATH", default="/usr/local")

include_dir = [
    os.path.join(_dirname, "include"),
    os.path.join(torch.utils.cmake_prefix_path, "../../include"),
    os.path.join(torch.utils.cmake_prefix_path, "../../include/torch/csrc/api/include"),
    os.path.join(intel_extension_for_pytorch.cmake_prefix_path, "../../include")
]

oneapi_root = os.getenv("ONEAPI_ROOT")
if oneapi_root:
    include_dir += [
        os.path.join(oneapi_root, "compiler/latest/include"),
        os.path.join(oneapi_root, "compiler/latest/include/sycl")
    ]

library_dir = [
    os.path.join(_dirname, "lib"),
    os.path.join(torch.utils.cmake_prefix_path, "../../lib"),
    os.path.join(intel_extension_for_pytorch.cmake_prefix_path, "../../lib")
]
libraries = ["ze_loader", "sycl", "torch", "intel-ext-pt-gpu"]


def compile_module_from_src(src, name):
    key = hashlib.sha256(src.encode("utf-8")).hexdigest()
    cache = get_cache_manager(key)
    cache_path = cache.get_file(f"{name}.so")
    if cache_path is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "main.cpp")
            with open(src_path, "w", encoding="utf-8") as f:
                f.write(src)
            with quiet():
                so = _build(name, src_path, tmpdir, library_dir, include_dir, libraries)
            with open(so, "rb") as f:
                cache_path = cache.put(f.read(), f"{name}.so", binary=True)
    spec = importlib.util.spec_from_file_location(name, cache_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------
# Utils
# ------------------------


class XPUUtils:

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(XPUUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        dirname = os.path.dirname(os.path.realpath(__file__))
        mod = compile_module_from_src(
            Path(os.path.join(dirname, "driver.c")).read_text(encoding="utf-8"), "spirv_utils")
        self.load_binary = mod.load_binary
        self.get_device_properties = mod.get_device_properties
        self.context = mod.init_context(self.get_sycl_queue())
        self.device_count = mod.init_devices(self.get_sycl_queue())
        self.current_device = 0 if self.device_count[0] > 0 else -1

    def get_current_device(self):
        return self.current_device

    def get_sycl_queue(self):
        return torch.xpu.current_stream().sycl_queue


# ------------------------
# Launcher
# ------------------------


def ty_to_cpp(ty):
    if ty[0] == "*":
        return "void*"
    return {
        "i1": "int32_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint32_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "float",
        "bf16": "float",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
        "nvTmaDesc": "TMADesc",
    }[ty]


def make_launcher(constants, signature, ids):  # pylint: disable=unused-argument
    # Record the end of regular arguments;
    # subsequent arguments are architecture-specific descriptors.
    arg_decls = ", ".join(f"{ty_to_cpp(ty)} arg{i}" for i, ty in signature.items())

    def _extracted_type(ty):
        if ty[0] == '*':
            return "PyObject*"
        if ty == "nvTmaDesc":
            return "PyObject*"

        return ty_to_cpp(ty)

    def format_of(ty):
        return {
            "PyObject*": "O",
            "float": "f",
            "double": "d",
            "long": "l",
            "int8_t": "b",
            "int16_t": "h",
            "int32_t": "i",
            "int64_t": "l",
            "uint8_t": "B",
            "uint16_t": "H",
            "uint32_t": "I",
            "uint64_t": "K",
        }[ty]

    args_format = ''.join([format_of(_extracted_type(ty)) for ty in signature.values()])
    format = "iiiOOOOOO" + args_format
    args_list = ', ' + ', '.join(f"&_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''

    internal_args_list = []
    for i, ty in signature.items():
        if ty[0] == "*":
            internal_args_list.append(f"ptr_info{i}.dev_ptr")
        elif ty == "nvTmaDesc":
            # Note: we have to dereference the pointer
            internal_args_list.append(f"*tma_ptr{i}")
        else:
            internal_args_list.append(f"_arg{i}")

    # generate glue code
    src = f"""
    #include <cstddef>
    #include <string>
    #include <iostream>
    #include <iomanip>
    #include <level_zero/ze_api.h>
    #include <sycl/sycl.hpp>
    #include <torch/extension.h>
    #include <ipex.h>

    #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
    #include <Python.h>
    #include <stdio.h>
    #include <numpy/arrayobject.h>

    typedef struct {{
      void* base;
      uint64_t shape[5];
      uint64_t stride[5];
    }} TMADesc;

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
  static void sycl_kernel_launch(uint32_t gridX, uint32_t gridY, uint32_t gridZ, int num_warps, int threads_per_warp, int shared_memory, sycl::queue& stream, sycl::kernel& kernel_ptr {', ' + arg_decls if len(arg_decls) > 0 else ''}) {{

    std::string kernel_name = kernel_ptr.get_info<sycl::info::kernel::function_name>();
    RECORD_FUNCTION("XPU Triton kernel:" + kernel_name, {{}});
    void *params[] = {{ {", ".join(f"&arg{i}" for i in signature.keys() if i not in constants)} }};
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
    if (shared_memory) {{
      expected_num_params -= 1;
    }}
    assert(num_params == expected_num_params && "number of kernel param not matched");
    // Submit the imported kernel.
    auto cgf = [&](sycl::handler &cgh) {{
      {" ".join(f'set_scalar_arg<{ty_to_cpp(item)}>(cgh, {idx}, params[{idx}]);' for idx, item in enumerate([signature[i] for i in signature if i not in constants]))}
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
    xpu::profiler_record(kernel_name, event);
  }}
// end sycl

    static inline TMADesc* getTmaDesc(PyObject *obj) {{
      if (sizeof(TMADesc*) != 8) {{
        PyErr_SetString(PyExc_SystemError, "getTmaDesc() requires 64-bit compilation");
        return NULL;
      }}

      PyObject *method_handle = PyObject_GetAttrString(obj, "tma_desc_cpu_ptr");
      if (!method_handle) {{
        PyErr_SetString(PyExc_TypeError, "tma_desc_cpu_ptr() method does not exist");
        return NULL;
      }}

      PyObject *empty_tuple = PyTuple_New(0);
      if (!empty_tuple) {{
        Py_DECREF(method_handle);
        PyErr_SetString(PyExc_SystemError, "Internal Python error!");
        return NULL;
      }}
      PyObject *method_ret = PyObject_Call(method_handle, empty_tuple, NULL);
      Py_DECREF(empty_tuple);
      Py_DECREF(method_handle);
      if (!method_ret) {{
        PyErr_SetString(PyExc_SystemError, "Internal Python error!");
        return NULL;
      }}

      if (!PyLong_Check(method_ret)) {{
        PyErr_SetString(PyExc_TypeError, "tma_desc_cpu_ptr() must return 64-bit int");
        Py_DECREF(method_ret);
        return NULL;
      }}

      uint64_t ptr_as_uint = PyLong_AsUnsignedLongLong(method_ret);
      Py_DECREF(method_ret);
      if (!ptr_as_uint) {{
        PyErr_SetString(PyExc_ValueError, "received NULL ptr from tma_desc_cpu_ptr()");
        return NULL;
      }}
      if (ptr_as_uint % 64 != 0) {{
        PyErr_SetString(PyExc_ValueError, "tma_desc_cpu_ptr() must be 64-byte aligned");
        return NULL;
      }}

      return (TMADesc*)(ptr_as_uint);
    }}

    static PyObject* launch(PyObject* self, PyObject* args) {{

      int gridX, gridY, gridZ;
      PyObject *launch_enter_hook = NULL;
      PyObject *launch_exit_hook = NULL;
      PyObject *kernel_metadata = NULL;
      PyObject *launch_metadata = NULL;
      PyObject *py_obj_stream;
      PyObject* py_kernel;

      {' '.join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])}
      if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ, &py_obj_stream, &py_kernel,
                                           &kernel_metadata, &launch_metadata,
                                           &launch_enter_hook, &launch_exit_hook {args_list})) {{
        return NULL;
      }}

      // extract kernel metadata
      int num_warps     = PyLong_AsLong(PyObject_GetAttrString(kernel_metadata, "num_warps"));
      int num_ctas      = PyLong_AsLong(PyObject_GetAttrString(kernel_metadata, "num_ctas"));
      int shared_memory = PyLong_AsLong(PyObject_GetAttrString(kernel_metadata, "shared"));
      int threads_per_warp = PyLong_AsLong(PyObject_GetAttrString(kernel_metadata, "threads_per_warp"));

      // extract cluster dims
      PyObject *clusterDim =  PyObject_GetAttrString(kernel_metadata, "cluster_dims");
      if (!PyTuple_Check(kernel_metadata)) {{
        PyErr_SetString(PyExc_TypeError, "kernel_metadata.cluster_dims must be a tuple");
        return NULL;
      }}
      int clusterDimX   = PyLong_AsLong(PyTuple_GetItem(clusterDim, 0));
      int clusterDimY   = PyLong_AsLong(PyTuple_GetItem(clusterDim, 1));
      int clusterDimZ   = PyLong_AsLong(PyTuple_GetItem(clusterDim, 2));
      // extract launch metadata
      if (launch_enter_hook != Py_None){{
        PyObject* args = Py_BuildValue("(O)", launch_metadata);
        PyObject* ret = PyObject_CallObject(launch_enter_hook, args);
        Py_DECREF(args);
        if (!ret)
          return NULL;
      }}

      void * pStream = PyLong_AsVoidPtr(py_obj_stream);
      //error check
      if(pStream == nullptr || py_kernel == nullptr) return NULL;

      sycl::queue stream = *(static_cast<sycl::queue*>(pStream));
      sycl::kernel* kernel_ptr = reinterpret_cast<sycl::kernel*>(PyCapsule_GetPointer(py_kernel, "kernel"));
      if(kernel_ptr == nullptr) return NULL;
      sycl::kernel kernel = *kernel_ptr;

      {"".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}, stream); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature.items()])};
      {"".join([f"TMADesc* tma_ptr{i} = getTmaDesc(_arg{i}); if (!tma_ptr{i}) return NULL;" if ty == "nvTmaDesc" else "" for i, ty in signature.items()])};
      sycl_kernel_launch(gridX, gridY, gridZ, num_warps, threads_per_warp, shared_memory, stream, kernel {', ' + ', '.join(internal_args_list) if len(internal_args_list) > 0 else ''});

      if(launch_exit_hook != Py_None){{
        PyObject* args = Py_BuildValue("(O)", launch_metadata);
        PyObject* ret = PyObject_CallObject(launch_exit_hook, args);
        Py_DECREF(args);
        if (!ret)
          return NULL;
      }}
      if (PyErr_Occurred()) {{
        return NULL;
      }}

      // return None
      Py_INCREF(Py_None);
      return Py_None;
    }}

    static PyMethodDef ModuleMethods[] = {{
      {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
      {{NULL, NULL, 0, NULL}} // sentinel
    }};

    static struct PyModuleDef ModuleDef = {{
      PyModuleDef_HEAD_INIT,
      \"__triton_launcher\",
      NULL, //documentation
      -1, //size
      ModuleMethods
    }};

    PyMODINIT_FUNC PyInit___triton_launcher(void) {{
      PyObject *m = PyModule_Create(&ModuleDef);
      if(m == NULL) {{
        return NULL;
      }}
      PyModule_AddFunctions(m, ModuleMethods);
      return m;
    }}
    """
    return src


class XPULauncher:

    def __init__(self, src, metadata):  # pylint: disable=unused-argument
        ids = {"ids_of_const_exprs": src.fn.constexprs if hasattr(src, "fn") else tuple()}
        constants = src.constants if hasattr(src, "constants") else {}
        cst_key = lambda i: src.fn.arg_names.index(i) if isinstance(i, str) else i
        constants = {cst_key(key): value for key, value in constants.items()}
        signature = {cst_key(key): value for key, value in src.signature.items()}
        src = make_launcher(constants, signature, ids)
        mod = compile_module_from_src(src, "__triton_launcher")
        self.launch = mod.launch

    def __call__(self, *args, **kwargs):
        self.launch(*args, **kwargs)


class XPUDriver(DriverBase):

    def __init__(self):
        self.launcher_cls = XPULauncher

    def __getattr__(self, name):
        # Lazily initialize utils to avoid unnecessary XPU runtime invocations.
        # See https://github.com/intel/intel-xpu-backend-for-triton/issues/624
        if name == "utils":
            self.utils = XPUUtils()  # pylint: disable=attribute-defined-outside-init
            return self.utils
        raise AttributeError

    def get_current_device(self):
        return self.utils.get_current_device()

    def get_current_stream(self, device):  # pylint: disable=unused-argument
        return torch.xpu.current_stream().sycl_queue

    def get_current_target(self):
        device = self.get_current_device()
        dev_property = torch.xpu.get_device_capability(device)
        warp_size = 32
        return GPUTarget("xpu", dev_property, warp_size)

    @staticmethod
    def is_active():
        return torch.xpu.is_available()
