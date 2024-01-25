import os
import hashlib
import tempfile
from pathlib import Path
from triton.runtime.build import _build
from triton.runtime.cache import get_cache_manager
from triton.backends.driver import DriverBase


import intel_extension_for_pytorch as ipex


dirname = os.getenv("ZE_PATH", default="/usr/local")
include_dir = [os.path.join(dirname, "include/level_zero")]
library_dir = [os.path.join(dirname, "lib")]
libraries = ['ze_loader']


def compile_module_from_src(src, name):
    key = hashlib.md5(src.encode("utf-8")).hexdigest()
    cache = get_cache_manager(key)
    cache_path = cache.get_file(f"{name}.so")
    if cache_path is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "main.cpp")
            with open(src_path, "w") as f:
                f.write(src)
            so = _build(name, src_path, tmpdir, library_dir, include_dir, libraries)
            with open(so, "rb") as f:
                cache_path = cache.put(f.read(), f"{name}.so", binary=True)
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, cache_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
        mod = compile_module_from_src(Path(os.path.join(dirname, "driver.c")).read_text(), "spirv_utils")
        self.load_binary = mod.load_binary
        self.load_sycl_binary = mod.load_sycl_binary
        self.get_device_properties = mod.get_device_properties
        self.get_l0_queue = mod.get_l0_queue
        self.get_l0_imm_cmd_list = mod.get_l0_imm_cmd_list
        self.get_l0_dev_ptr = mod.get_l0_dev_ptr
        self.get_l0_ctxt_ptr = mod.get_l0_ctxt_ptr
        import torch
        self.context = mod.init_context(torch.xpu.current_stream().sycl_queue)
        self.device_count = mod.init_devices(torch.xpu.current_stream().sycl_queue)
        self.event_pool = mod.init_event_pool()[0]
        self.current_device = 0 if self.device_count[0] > 0 else -1

    def get_current_device(self):
        import torch
        return torch.xpu.device(self.current_device).sycl_device

    def get_event_pool(self):
        return self.event_pool

    def get_sycl_queue(self):
        return ipex.xpu.current_stream().sycl_queue

    def get_dev_ctxt_queue_objs(self):
        #context = self.get_l0_ctxt_ptr(self.get_sycl_queue())[0]
        #device = self.get_l0_dev_ptr(self.get_sycl_queue())[0]
        #queue = self.get_l0_queue(self.get_sycl_queue())[0]
        return 0, 0, 0

    def use_icl(self):
        return self.get_l0_queue(self.get_sycl_queue())[0] == 0

# ------------------------
# Launcher
# ------------------------


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


def generate_cu_signature(constants, signature, ids):
    # CUtensorMap*s are always the last arguments
    num_regular_signatures = max(signature.keys()) + 1 if len(signature) > 0 else 0
    if ids["ids_of_tensormaps"] is not None:
        for i, _ in enumerate(ids["ids_of_tensormaps"]):
            signature[num_regular_signatures + i] = '*CUtensorMap'
    return signature, num_regular_signatures


def make_launcher(constants, signature, ids):
    # Record the end of regular arguments;
    # subsequent arguments are architecture-specific descriptors, such as tensor descriptors for CUDA.
    signature, desc_start_idx = generate_cu_signature(constants, signature, ids)
    arg_decls = ', '.join(f"{ty_to_cpp(ty)} arg{i}" for i, ty in signature.items())

    def _extracted_type(ty):
        if ty[0] == '*':
            return "void*"
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

    format = "iiiiiiiiiiOKKKKOOOK" + ''.join(
        [format_of(_extracted_type(ty)) for ty in signature.values()])

    # generate glue code
    src = f"""
    #include <cstddef>
    #include <string>
    #include <iostream>
    #include <iomanip>
    #include <level_zero/ze_api.h>
    #include <sycl/sycl.hpp>

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
// start sycl
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
  static void sycl_kernel_launch(uint32_t gridX, uint32_t gridY, uint32_t gridZ, int num_warps, int threads_per_warp, int shared_memory, sycl::queue& stream, sycl::kernel& kernel_ptr {', ' + arg_decls if len(arg_decls) > 0 else ''}) {{
    //std::cout<<"sycl_kernel_launch entry"<<std::endl;
    std::string kernel_name = kernel_ptr.get_info<sycl::info::kernel::function_name>();
    //std::cout<<"Kernel name :"<<kernel_name<<std::endl;
    void *params[] = {{ {', '.join(f"&arg{i}" for i in signature.keys() if i not in constants)} }};
    uint32_t num_params = sizeof(params)/sizeof(params[0]);
    uint32_t expected_num_params = kernel_ptr.get_info<sycl::info::kernel::num_args>();
    //std::cout<<"num_params          :"<<num_params<<std::endl;
    //std::cout<<"expected_num_params :"<<expected_num_params<<std::endl;
    //std::cout<<"shared_memory       :"<<shared_memory<<std::endl;
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
  }}
// end sycl
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
      PyObject *py_obj_stream;
      void* pKrnl;

      {' '.join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])}
      if (!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ, &num_warps, &num_ctas,
                            &clusterDimX, &clusterDimY, &clusterDimZ, &shared_memory, &_is_icl, &py_obj_stream,
                            &_queue, &_dev, &_ctxt, &pKrnl, &launch_enter_hook, &launch_exit_hook,
                            &compiled_kernel, &_event_pool
                            {', ' + ', '.join(f"&_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''})) {{
        return NULL;
      }}

      if (launch_enter_hook != Py_None) {{
        PyObject_CallObject(launch_enter_hook, args);
      }}
      
      void * pStream = PyCapsule_GetPointer(py_obj_stream, PyCapsule_GetName(py_obj_stream));
      //error;
      if(pStream == nullptr || pKrnl == nullptr) return NULL;

      sycl::queue stream = *(static_cast<sycl::queue*>(pStream));
      sycl::kernel kernel = *(static_cast<sycl::kernel*>(pKrnl));
      auto threads_per_warp = 32;
      //std::cout<<"_launch : going to call sycl_kernel_launch"<<std::endl;
      sycl_kernel_launch(gridX, gridY, gridZ, num_warps, threads_per_warp, shared_memory, stream, kernel {',' + ', '.join(f"(void *) _arg{i}" if ty[0]=="*" else f"_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''});

      // Freeing the memory allocated during kernel load
      sycl::kernel* krnlPtr = static_cast<sycl::kernel*>(pKrnl);
      delete krnlPtr;

/*
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
*/
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
    return src


class XPULauncher(object):

    def __init__(self, src, metadata):
        ids = {
            "ids_of_tensormaps": metadata.ids_of_tensormaps, 
            "ids_of_folded_args": metadata.ids_of_folded_args,
            "ids_of_const_exprs": src.fn.constexprs if hasattr(src, "fn") else tuple()
        }
        constants = src.constants if hasattr(src, "constants") else dict()
        enable_warp_specialization = False
        src = make_launcher(constants, src.signature, ids)
        mod = compile_module_from_src(src, "__triton_launcher")
        self.launch = mod.launch

    def __call__(self, *args, **kwargs):
        self.launch(*args, **kwargs)


class XPUDriver(DriverBase):

    def __init__(self):
        self.utils = XPUUtils()
        self.binary_ext = "spv"
        self.launcher_cls = XPULauncher
        self.get_current_stream = self.get_current_stream
        self.get_current_device = self.utils.get_current_device

    def get_current_stream(self, device):
        import torch
        return torch.xpu.current_stream().sycl_queue

    def get_current_target(self):
        return ("xpu", 0)

    @staticmethod
    def is_active():
        import torch
        return torch.xpu.is_available()

    def assemble_tensormap_to_arg(self, tensormaps_info, args):
        args_ptr = tuple([arg.data_ptr() if hasattr(arg, 'data_ptr') else arg for arg in args])
        return args_ptr
