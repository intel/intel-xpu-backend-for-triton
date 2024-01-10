import hashlib
import os
import tempfile

from ..common import _build
from ..common.backend import get_cuda_version_key
from ..common.build import is_hip
from ..runtime.cache import get_cache_manager
from .utils import generate_cu_signature


def is_spirv():
    return os.environ.get("TRITON_TARGET_NVVM", "0") != "1"


# ----- stub --------


def make_so_cache_key(version_hash, signature, constants, ids, **kwargs):
    # Get unique key for the compiled code
    signature = {k: 'ptr' if v[0] == '*' else v for k, v in signature.items()}
    key = f"{version_hash}-{''.join(signature.values())}-{constants}-{ids}"
    for kw in kwargs:
        key = f"{key}-{kwargs.get(kw)}"
    key = hashlib.md5(key.encode("utf-8")).hexdigest()
    return key


def make_stub(name, signature, constants, ids, **kwargs):
    # name of files that are cached
    so_cache_key = make_so_cache_key(get_cuda_version_key(), signature, constants, ids, **kwargs)
    so_cache_manager = get_cache_manager(so_cache_key)
    so_name = f"{name}.so"
    # retrieve stub from cache if it exists
    cache_path = so_cache_manager.get_file(so_name)
    if cache_path is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            src = generate_launcher(constants, signature, ids)
            src_path = os.path.join(tmpdir, "main.cpp" if is_spirv() else "main.c")
            with open(src_path, "w") as f:
                f.write(src)
            so = _build(name, src_path, tmpdir)
            with open(so, "rb") as f:
                return so_cache_manager.put(f.read(), so_name, binary=True)
    else:
        return cache_path


# ----- source code generation --------


def ty_to_cpp(ty):
    if ty[0] == '*':
        return "void*" if is_spirv() else "hipDeviceptr_t" if is_hip() else "CUdeviceptr"
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


def generate_launcher(constants, signature, ids):
    # Record the end of regular arguments;
    # subsequent arguments are architecture-specific descriptors, such as tensor descriptors for CUDA.
    signature, desc_start_idx = generate_cu_signature(constants, signature, ids)
    arg_decls = ', '.join(f"{ty_to_cpp(ty)} arg{i}" for i, ty in signature.items())

    def _extracted_type(ty):
        if ty[0] == '*':
            return "void*" if is_spirv() else "PyObject*"
        return {
            'i1': 'int32_t',
            'i32': 'int32_t',
            'i64': 'int64_t',
            'u32': 'uint32_t',
            'u64': 'uint64_t',
            'fp16': 'float',
            'bf16': 'float',
            'fp32': 'float',
            'f32': 'float',
            'fp64': 'double',
        }[ty]

    def format_of(ty):
        return {
            "PyObject*": "O",
            "void*": "K",
            "float": "f",
            "double": "d",
            "long": "l",
            "uint32_t": "I",
            "int32_t": "i",
            "uint64_t": "K",
            "int64_t": "L",
        }[ty]

    format = "iiiiiiiiiKKOOOK" + ''.join([format_of(_extracted_type(ty)) for ty in signature.values()])
    if is_spirv():
        reg_launch_format = "iiiiiiiiiiKKKKKOOOK" + ''.join(
            [format_of(_extracted_type(ty)) for ty in signature.values()])

    # generate glue code
    if is_spirv():
        src = f"""
    #include <cstddef>
    #include <string>
    #include <iostream>
    #include <iomanip>
    #include <level_zero/ze_api.h>

    #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
    #include <Python.h>
    #include <stdio.h>
    #include <numpy/arrayobject.h>

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

    static void _regular_launch(uint32_t gridX, uint32_t gridY, uint32_t gridZ, int num_warps, int shared_memory,
                                ze_command_queue_handle_t queue, ze_device_handle_t _dev, ze_context_handle_t _ctxt,
                                ze_kernel_handle_t function, ze_event_pool_handle_t event_pool
                                {', ' + arg_decls if len(arg_decls) > 0 else ''}) {{
      void *params[] = {{ {', '.join(f"&arg{i}" for i in signature.keys() if i not in constants)} }};

      if (gridX*gridY*gridZ > 0) {{
        {" ".join(f'zeKernelSetArgumentValue(function, {idx}, sizeof({ty_to_cpp(item)}), params[{idx}]);' for idx, item in enumerate([signature[i] for i in signature if i not in constants]))}
        if (shared_memory) {{
          uint32_t num_params = sizeof(params)/sizeof(params[0]);
          zeKernelSetArgumentValue(function, num_params, shared_memory, NULL);
        }}
        zeKernelSetGroupSize(function, 32*num_warps, 1, 1);

        ze_group_count_t grpCount = {{gridX, gridY, gridZ}};

        // Create command list
        ze_command_list_handle_t CmdList;
        ze_command_list_desc_t CommandListDesc_ = {{
            ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
            nullptr,
            0,
            0,
        }};

        ZE_CHECK(zeCommandListCreate(_ctxt, _dev, &CommandListDesc_, &CmdList));

        ze_event_desc_t eventDesc = {{
            ZE_STRUCTURE_TYPE_EVENT_DESC,
            nullptr,
            0,
            0,
            ZE_EVENT_SCOPE_FLAG_HOST
        }};
        ze_event_handle_t hEvent;
        ZE_CHECK(zeEventCreate(event_pool, &eventDesc, &hEvent));

        // Append a signal of an event into the command list after the kernel executes
        ZE_CHECK(zeCommandListAppendLaunchKernel(CmdList, function, &grpCount, hEvent, 0, nullptr));

        // close command list
        ZE_CHECK(zeCommandListClose(CmdList));

        // FIXME: The following statement currently doesn't synchronize all IPEX SYCL queues.
        //        Needs to find all IPEX SYCL queues
        // Synchronize the command queue to ensure previous IPEX SYCL commands complete before Triton kernel starts
        // ZE_CHECK(zeCommandQueueSynchronize(queue, std::numeric_limits<uint64_t>::max()));

        // execute command list
        ZE_CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &CmdList, nullptr));

        // Wait on event to complete
        ZE_CHECK(zeEventHostSynchronize(hEvent, std::numeric_limits<uint64_t>::max()));
      }}
    }}

    static void _launch(uint32_t gridX, uint32_t gridY, uint32_t gridZ, int num_warps, int shared_memory,
                        ze_command_list_handle_t queue, ze_kernel_handle_t function, ze_event_pool_handle_t event_pool
                        {', ' + arg_decls if len(arg_decls) > 0 else ''}) {{
      void *params[] = {{ {', '.join(f"&arg{i}" for i in signature.keys() if i not in constants)} }};

      if (gridX*gridY*gridZ > 0) {{
        {" ".join(f'zeKernelSetArgumentValue(function, {idx}, sizeof({ty_to_cpp(item)}), params[{idx}]);' for idx, item in enumerate([signature[i] for i in signature if i not in constants]))}
        if (shared_memory) {{
          uint32_t num_params = sizeof(params)/sizeof(params[0]);
          zeKernelSetArgumentValue(function, num_params, shared_memory, NULL);
        }}
        zeKernelSetGroupSize(function, 32*num_warps, 1, 1);
        ze_group_count_t grpCount = {{gridX, gridY, gridZ}};

        ze_event_desc_t eventDesc = {{
            ZE_STRUCTURE_TYPE_EVENT_DESC,
            nullptr,
            0,
            0,
            ZE_EVENT_SCOPE_FLAG_HOST
        }};
        ze_event_handle_t hEvent;
        ZE_CHECK(zeEventCreate(event_pool, &eventDesc, &hEvent));

        // FIXME: The following statement currently doesn't synchronize all IPEX SYCL queues.
        //        Needs to find all IPEX SYCL queues
        // Synchronize to ensure previous IPEX SYCL commands complete before Triton kernel starts
        ZE_CHECK(zeCommandListHostSynchronize(queue, std::numeric_limits<uint64_t>::max()));

        // Append a signal of an event into the command list after the kernel executes
        ZE_CHECK(zeCommandListAppendLaunchKernel(queue, function, &grpCount, hEvent, 0, nullptr));
        // Wait on event to complete
        ZE_CHECK(zeEventHostSynchronize(hEvent, std::numeric_limits<uint64_t>::max()));
      }}
    }}

    typedef struct _DevicePtrInfo {{
      void* dev_ptr;
      bool valid;
    }} DevicePtrInfo;

    static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
      DevicePtrInfo ptr_info;
      ptr_info.dev_ptr = 0;
      ptr_info.valid = true;
      PyTypeObject* obj_type = Py_TYPE(obj);

      if (PyLong_Check(obj)) {{
        ptr_info.dev_ptr = (void*) PyLong_AsUnsignedLongLong(obj);
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
        ptr_info.dev_ptr = (void*) PyLong_AsUnsignedLongLong(ret);
        if(!ptr_info.dev_ptr) {{
          return ptr_info;
        }}
        Py_DECREF(ret);  // Thanks ChatGPT!
        return ptr_info;
      }}
      PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
      return ptr_info;
    }}

    static PyObject* launch(PyObject* self, PyObject* args) {{

      int gridX, gridY, gridZ;
      uint64_t _queue;
      uint64_t _stream;
      uint64_t _function;
      uint64_t _event_pool;
      uint64_t _dev;
      uint64_t _ctxt;
      int num_warps;
      int num_ctas;
      int clusterDimX;
      int clusterDimY;
      int clusterDimZ;
      int _is_icl;
      int shared_memory;
      PyObject *launch_enter_hook = NULL;
      PyObject *launch_exit_hook = NULL;
      PyObject *compiled_kernel = NULL;

      {' '.join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])}
      if (!PyArg_ParseTuple(args, \"{reg_launch_format}\", &gridX, &gridY, &gridZ, &num_warps, &num_ctas,
                            &clusterDimX, &clusterDimY, &clusterDimZ, &shared_memory, &_is_icl, &_stream,
                            &_queue, &_dev, &_ctxt, &_function, &launch_enter_hook, &launch_exit_hook,
                            &compiled_kernel, &_event_pool
                            {', ' + ', '.join(f"&_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''})) {{
        return NULL;
      }}

      if (launch_enter_hook != Py_None) {{
        PyObject_CallObject(launch_enter_hook, args);
      }}

      // raise exception asap
      // {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature.items()])};
      if (_is_icl == 0) {{
        _regular_launch(gridX, gridY, gridZ, num_warps, shared_memory, (ze_command_queue_handle_t)_queue,
                        (ze_device_handle_t)_dev, (ze_context_handle_t)_ctxt, (ze_kernel_handle_t)_function,
                        (ze_event_pool_handle_t)_event_pool
                        {', ' + ', '.join(f"(void *) _arg{i}" if ty[0]=="*" else f"_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''});
      }} else {{
        _launch(gridX, gridY, gridZ, num_warps, shared_memory, (ze_command_list_handle_t)_stream,
                (ze_kernel_handle_t)_function, (ze_event_pool_handle_t)_event_pool
                {', ' + ', '.join(f"(void *) _arg{i}" if ty[0]=="*" else f"_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''});
      }}

      if (launch_exit_hook != Py_None) {{
        PyObject_CallObject(launch_exit_hook, args);
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
    else:
        folded_without_constexprs = [c for c in ids['ids_of_folded_args'] if c not in ids['ids_of_const_exprs']]
        params = [
            i for i in signature.keys()
            if i >= desc_start_idx or (i not in constants and i not in folded_without_constexprs)
        ]
        src = f"""
#include \"cuda.h\"
#include <stdbool.h>
#include <Python.h>
#include <dlfcn.h>

static inline void gpuAssert(CUresult code, const char *file, int line)
{{
   if (code != CUDA_SUCCESS)
   {{
      const char* prefix = "Triton Error [CUDA]: ";
      const char* str;
      cuGetErrorString(code, &str);
      char err[1024] = {{0}};
      strcat(err, prefix);
      strcat(err, str);
      PyGILState_STATE gil_state;
      gil_state = PyGILState_Ensure();
      PyErr_SetString(PyExc_RuntimeError, err);
      PyGILState_Release(gil_state);
   }}
}}

#define CUDA_CHECK(ans) {{ gpuAssert((ans), __FILE__, __LINE__); }}

typedef CUresult (*cuLaunchKernelEx_t)(const CUlaunchConfig* config, CUfunction f, void** kernelParams, void** extra);

static cuLaunchKernelEx_t getLaunchKernelExHandle() {{
  // Open the shared library
  void* handle = dlopen("libcuda.so", RTLD_LAZY);
  if (!handle) {{
    PyErr_SetString(PyExc_RuntimeError, "Failed to open libcuda.so");
    return NULL;
  }}
  // Clear any existing error
  dlerror();
  cuLaunchKernelEx_t cuLaunchKernelExHandle = (cuLaunchKernelEx_t)dlsym(handle, "cuLaunchKernelEx");
  // Check for errors
  const char *dlsym_error = dlerror();
  if (dlsym_error) {{
    PyErr_SetString(PyExc_RuntimeError, "Failed to retrieve cuLaunchKernelEx from libcuda.so");
    return NULL;
  }}
  return cuLaunchKernelExHandle;
}}

static void _launch(int gridX, int gridY, int gridZ, int num_warps, int num_ctas, int clusterDimX, int clusterDimY, int clusterDimZ, int shared_memory, CUstream stream, CUfunction function{', ' + arg_decls if len(arg_decls) > 0 else ''}) {{
  void *params[] = {{ {', '.join(f"&arg{i}" for i in params)} }};
  if (gridX*gridY*gridZ > 0) {{
    if (num_ctas == 1) {{
      CUDA_CHECK(cuLaunchKernel(function, gridX, gridY, gridZ, 32*num_warps, 1, 1, shared_memory, stream, params, 0));
    }} else {{
      CUlaunchAttribute launchAttr[2];
      launchAttr[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
      launchAttr[0].value.clusterDim.x = clusterDimX;
      launchAttr[0].value.clusterDim.y = clusterDimY;
      launchAttr[0].value.clusterDim.z = clusterDimZ;
      launchAttr[1].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
      launchAttr[1].value.clusterSchedulingPolicyPreference = CU_CLUSTER_SCHEDULING_POLICY_SPREAD;
      CUlaunchConfig config;
      config.gridDimX = gridX * clusterDimX;
      config.gridDimY = gridY * clusterDimY;
      config.gridDimZ = gridZ * clusterDimZ;
      config.blockDimX = 32 * num_warps;
      config.blockDimY = 1;
      config.blockDimZ = 1;
      config.sharedMemBytes = shared_memory;
      config.hStream = stream;
      config.attrs = launchAttr;
      config.numAttrs = 2;
      static cuLaunchKernelEx_t cuLaunchKernelExHandle = NULL;
      if (cuLaunchKernelExHandle == NULL) {{
        cuLaunchKernelExHandle = getLaunchKernelExHandle();
      }}
      CUDA_CHECK(cuLaunchKernelExHandle(&config, function, params, 0));
    }}
  }}
}}

typedef struct _DevicePtrInfo {{
    CUdeviceptr dev_ptr;
    bool valid;
}} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {{
    ptr_info.dev_ptr = PyLong_AsUnsignedLongLong(obj);
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
    ptr_info.dev_ptr = PyLong_AsUnsignedLongLong(ret);
    if(!ptr_info.dev_ptr)
      return ptr_info;
    uint64_t dev_ptr;
    int status = cuPointerGetAttribute(&dev_ptr, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, ptr_info.dev_ptr);
    if (status == CUDA_ERROR_INVALID_VALUE) {{
        PyErr_Format(PyExc_ValueError,
                     "Pointer argument (at %d) cannot be accessed from Triton (cpu tensor?)", idx);
        ptr_info.valid = false;
    }}
    ptr_info.dev_ptr = dev_ptr;
    Py_DECREF(ret);  // Thanks ChatGPT!
    return ptr_info;
  }}
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  ptr_info.valid = false;
  return ptr_info;
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  uint64_t _stream;
  uint64_t _function;
  int num_warps;
  int num_ctas;
  int clusterDimX;
  int clusterDimY;
  int clusterDimZ;
  int shared_memory;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *compiled_kernel = NULL;
  {' '.join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ, &num_warps, &num_ctas,
                       &clusterDimX, &clusterDimY, &clusterDimZ, &shared_memory, &_stream,
                       &_function, &launch_enter_hook, &launch_exit_hook, &compiled_kernel
                       {', ' + ', '.join(f"&_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''})) {{
    return NULL;
  }}

  if (launch_enter_hook != Py_None && !PyObject_CallObject(launch_enter_hook, args)) {{
    return NULL;
  }}


  // raise exception asap
  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature.items()])};
  Py_BEGIN_ALLOW_THREADS;
  _launch(gridX, gridY, gridZ, num_warps, num_ctas, clusterDimX, clusterDimY, clusterDimZ, shared_memory, (CUstream)_stream, (CUfunction)_function{', ' + ', '.join(f"ptr_info{i}.dev_ptr" if ty[0]=="*" else f"_arg{i}"for i, ty in signature.items()) if len(signature) > 0 else ''});
  Py_END_ALLOW_THREADS;
  if (PyErr_Occurred()) {{
    return NULL;
  }}

  if (launch_exit_hook != Py_None && !PyObject_CallObject(launch_exit_hook, args)) {{
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
