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

    @staticmethod
    def _generate_src():
        return """
        #include <cstddef>
        #include <string>
        #include <vector>
        #include <unordered_map>
        #include <variant>
        #include <iostream>
        #include <level_zero/ze_api.h>
        #include <sycl/sycl.hpp>

        #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
        #include <Python.h>
        #include <numpy/arrayobject.h>

        typedef struct l0_resc_handles {
            ze_context_handle_t context;
            ze_device_handle_t device;
            ze_command_queue_handle_t queue;
            ze_command_list_handle_t cmd_list;
        }l0_resc_handles;

        std::unordered_map<sycl::queue, l0_resc_handles> sycl_queue_map;
        static ze_context_handle_t context = {nullptr};
        static ze_driver_handle_t driverHandle = {nullptr};
        static ze_event_pool_handle_t eventPoolHandle = {nullptr};

        static std::vector<ze_device_handle_t> devices;

        static inline void gpuAssert(ze_result_t code, const char *file, int line)
        {
           if (code != ZE_RESULT_SUCCESS)
           {
              const char* prefix = "Triton Error [ZE]: ";
              std::string str = std::to_string(code);
              char err[1024] = {0};
              strcat(err, prefix);
              strcat(err, str.c_str());
              PyErr_SetString(PyExc_RuntimeError, err);
           }
        }

        #define ZE_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); if(PyErr_Occurred()) return NULL; }

        static PyObject* getDeviceProperties(PyObject* self, PyObject* args){
            int device_id;
            if(!PyArg_ParseTuple(args, "i", &device_id))
                return NULL;

            if (device_id > devices.size()) {
                std::cout << "Device ID not found: " << device_id << std::endl;
                return NULL;
            }

            // Get device handle
            ze_device_handle_t phDevice = devices[device_id];

            // create a struct to hold device properties
            ze_device_properties_t device_properties = {};
            device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            zeDeviceGetProperties(phDevice, &device_properties);

            int multiprocessor_count = device_properties.numSlices * device_properties.numSubslicesPerSlice;
            int sm_clock_rate = device_properties.coreClockRate;

            ze_device_compute_properties_t compute_properties = {};
            compute_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES;
            zeDeviceGetComputeProperties(phDevice, &compute_properties);
            int max_shared_mem = compute_properties.maxSharedLocalMemory;

            uint32_t memoryCount = 0;
            zeDeviceGetMemoryProperties(phDevice, &memoryCount, nullptr);
            auto pMemoryProperties = new ze_device_memory_properties_t[memoryCount];
            for( uint32_t mem = 0; mem < memoryCount; ++mem )
            {
                pMemoryProperties[mem].stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
                pMemoryProperties[mem].pNext = nullptr;
            }
            zeDeviceGetMemoryProperties(phDevice, &memoryCount, pMemoryProperties);
            // for( uint32_t mem = 0; mem < memoryCount; ++mem )
            // {
            //    std::cout << to_string( pMemoryProperties[ mem ] ) << std::endl;
            // }

            int mem_clock_rate = pMemoryProperties[0].maxClockRate;
            int mem_bus_width = pMemoryProperties[0].maxBusWidth;

            delete[] pMemoryProperties;

            return Py_BuildValue("{s:i, s:i, s:i, s:i, s:i}", "max_shared_mem", max_shared_mem,
                                       "multiprocessor_count", multiprocessor_count,
                                       "sm_clock_rate", sm_clock_rate,
                                       "mem_clock_rate", mem_clock_rate,
                                       "mem_bus_width", mem_bus_width);
        }

        static PyObject* loadBinary(PyObject* self, PyObject* args) {
            const char* name;
            int shared;
            PyObject *py_bytes;
            int device_id;
            if(!PyArg_ParseTuple(args, "sSii", &name, &py_bytes, &shared, &device_id)) {
                std::cout << "loadBinary arg parse failed" << std::endl;
                return NULL;
            }

            // uint8_t* data = (uint8_t*) PyBytes_AsString(py_bytes);
            // int data_size = PyBytes_Size(py_bytes);

            if (device_id > devices.size()) {
                std::cout << "Device ID not found: " << device_id << std::endl;
                return NULL;
            }

            ze_device_handle_t device = devices[device_id];

            int32_t n_regs = 0;
            int32_t n_spills = 0;

            ze_module_desc_t module_desc = {};
            module_desc.format = ZE_MODULE_FORMAT_IL_SPIRV;
            module_desc.inputSize = PyBytes_Size(py_bytes);
            module_desc.pInputModule = (uint8_t*) PyBytes_AsString(py_bytes);
            ze_module_handle_t module;
            // std::cout << "SPIRV binary size: " << module_desc.inputSize << std::endl;
            ZE_CHECK(zeModuleCreate(context, device, &module_desc, &module, nullptr));

            // std::cout << "loadBinary zeModuleCreated" << std::endl;
            ze_kernel_desc_t kernel_desc = {};
            kernel_desc.pKernelName = name;
            ze_kernel_handle_t fun;
            ZE_CHECK(zeKernelCreate(module, &kernel_desc, &fun));

            // std::cout << "loadBinary zeKernelCreated" << std::endl;

            if(PyErr_Occurred()) {
              std::cout << "loadBinary error occurred" << std::endl;
              return NULL;
            }

            return Py_BuildValue("(KKii)", (uint64_t)module, (uint64_t)fun, n_regs, n_spills);
        }

        bool update(sycl::queue sycl_queue) {
            // Get l0-context
            auto sycl_context = sycl_queue.get_context();
            ze_context_handle_t hCtxt = get_native<sycl::backend::level_zero>(sycl_context);
	        // Get l0-device
            std::vector<sycl::device> sycl_devices = sycl_context.get_devices();
            ze_device_handle_t hDev = get_native<sycl::backend::level_zero>(sycl_devices[0]);
            // Get l0-queue
            bool immediate_cmd_list = false;
            std::variant<ze_command_queue_handle_t, ze_command_list_handle_t> queue_var = get_native<sycl::backend::level_zero>(sycl_queue);
            auto l0_queue = std::get_if<ze_command_queue_handle_t>(&queue_var);
            if (l0_queue == nullptr) {
                auto imm_cmd_list = std::get_if<ze_command_list_handle_t>(&queue_var);
                if (imm_cmd_list == nullptr) {
                    return false;
                }
                immediate_cmd_list = true;
                sycl_queue_map[sycl_queue].cmd_list = *imm_cmd_list;
            }
            sycl_queue_map[sycl_queue].context = hCtxt;
            sycl_queue_map[sycl_queue].device = hDev;
            sycl_queue_map[sycl_queue].queue = immediate_cmd_list ? 0 : *l0_queue;

            // Update global data
            context = sycl_queue_map[sycl_queue].context;
            uint32_t deviceCount = std::min(sycl_devices.size(), devices.size());
            for (uint32_t i = 0; i < deviceCount; ++i) {
                devices[i] = sycl::get_native<sycl::backend::level_zero>(sycl_devices[i]);
            }

            return true;
        }

        static PyObject* initContext(PyObject* self, PyObject* args) {
            void* queue;
            if(!PyArg_ParseTuple(args, "K", &queue))
                return NULL;
            sycl::queue* sycl_queue = static_cast<sycl::queue*>(queue);
            if(sycl_queue_map.find(*sycl_queue) == sycl_queue_map.end()) {
                update(*sycl_queue);
            }
            context = sycl_queue_map[*sycl_queue].context;
            return Py_BuildValue("(K)", (uint64_t)context);
        }

        static PyObject* initEventPool(PyObject* self, PyObject* args) {
            // Create event pool
            ze_event_pool_desc_t tsEventPoolDesc = {
                ZE_STRUCTURE_TYPE_EVENT_POOL_DESC,
                nullptr,
                ZE_EVENT_POOL_FLAG_HOST_VISIBLE, // all events in pool are visible to Host
                1 // count
            };
            ZE_CHECK(zeEventPoolCreate(context, &tsEventPoolDesc, 0, nullptr, &eventPoolHandle));

            return Py_BuildValue("(K)", (uint64_t)eventPoolHandle);
            // Py_RETURN_NONE;
        }

        static PyObject* initDevices(PyObject* self, PyObject *args) {
            void* queue;
            if(!PyArg_ParseTuple(args, "K", &queue))
                return NULL;
            sycl::queue* sycl_queue = static_cast<sycl::queue*>(queue);

            auto sycl_context = sycl_queue->get_context();

	        // Get l0-device
            std::vector<sycl::device> sycl_devices = sycl_context.get_devices();

            // Retrieve devices
            uint32_t deviceCount = sycl_devices.size();
            for (uint32_t i = 0; i < deviceCount; ++i) {
                devices.push_back(sycl::get_native<sycl::backend::level_zero>(sycl_devices[i]));
            }

            // npy_intp dims[1];
            // dims[0] = deviceCount;
            // std::cout << "Before PyArray_SimpleNewFromData: " << devices.size() << " " << devices.data()[0] << std::endl;
            // PyObject* arr = PyArray_SimpleNewFromData(1, dims, NPY_UINT64, reinterpret_cast<void*>(devices.data()));
            // std::cout << "After PyArray_SimpleNewFromData: " << devices.data()[0] << std::endl;
            // PyObject* ret = Py_BuildValue("(O)", arr);
            // std::cout << "After Py_BuildValue" << std::endl;
            // return ret;
            return Py_BuildValue("(i)", deviceCount);
            // Py_RETURN_NONE;
        }

        static PyObject* getL0ImmCommandList(PyObject* self, PyObject* args) {
            void* queue;
            if(!PyArg_ParseTuple(args, "K", &queue))
                return NULL;
            sycl::queue* sycl_queue = static_cast<sycl::queue*>(queue);

            if(sycl_queue_map.find(*sycl_queue) == sycl_queue_map.end()) {
                update(*sycl_queue);
            }
            return Py_BuildValue("(K)", (uint64_t)(sycl_queue_map[*sycl_queue].cmd_list));
        }
        static PyObject* getL0Queue(PyObject* self, PyObject* args) {
            void* queue;
            if(!PyArg_ParseTuple(args, "K", &queue))
                return NULL;
            sycl::queue* sycl_queue = static_cast<sycl::queue*>(queue);
            if(sycl_queue_map.find(*sycl_queue) == sycl_queue_map.end()) {
                update(*sycl_queue);
            }
            return Py_BuildValue("(K)", (uint64_t)(sycl_queue_map[*sycl_queue].queue));
        }
        static PyObject* getL0DevPtr(PyObject* self, PyObject* args) {
            void* queue;
            if(!PyArg_ParseTuple(args, "K", &queue))
                return NULL;
            sycl::queue* sycl_queue = static_cast<sycl::queue*>(queue);
            if(sycl_queue_map.find(*sycl_queue) == sycl_queue_map.end()) {
                update(*sycl_queue);
            }
            return Py_BuildValue("(K)", (uint64_t)(sycl_queue_map[*sycl_queue].device));
        }
        static PyObject* getL0CtxtPtr(PyObject* self, PyObject* args) {
            void* queue;
            if(!PyArg_ParseTuple(args, "K", &queue))
                return NULL;
            sycl::queue* sycl_queue = static_cast<sycl::queue*>(queue);
            if(sycl_queue_map.find(*sycl_queue) == sycl_queue_map.end()) {
                update(*sycl_queue);
            }
            return Py_BuildValue("(K)", (uint64_t)(sycl_queue_map[*sycl_queue].context));
        }
        static PyObject* isUsingICL(PyObject* self, PyObject* args) {
            void* queue;
            if(!PyArg_ParseTuple(args, "K", &queue))
                return NULL;
            sycl::queue* sycl_queue = static_cast<sycl::queue*>(queue);
            if(sycl_queue_map.find(*sycl_queue) == sycl_queue_map.end()) {
                update(*sycl_queue);
            }
            uint32_t using_icl = sycl_queue_map[*sycl_queue].cmd_list != 0 ? 1 : 0;
            return Py_BuildValue("(i)", using_icl);
        }

        static PyMethodDef ModuleMethods[] = {
          {"load_binary", loadBinary, METH_VARARGS, "Load provided SPV into ZE driver"},
          {"get_device_properties", getDeviceProperties, METH_VARARGS, "Get the properties for a given device"},
          {"init_context", initContext, METH_VARARGS, "Initialize the ZE GPU context"},
          {"init_devices", initDevices, METH_VARARGS, "Initialize the ZE GPU devices and return device count"},
          {"init_event_pool", initEventPool, METH_VARARGS, "Initialize ZE event pool"},
          {"get_l0_imm_cmd_list", getL0ImmCommandList, METH_VARARGS, "Get l0 command list in case of immediate command list"},
          {"get_l0_queue", getL0Queue, METH_VARARGS, "Get l0 queue from sycl queue"},
          {"get_l0_dev_ptr", getL0DevPtr, METH_VARARGS, "Extract l0 device pointer from sycl queue"},
          {"get_l0_ctxt_ptr", getL0CtxtPtr, METH_VARARGS, "Extract l0 context pointer from sycl queue"},
          {"is_using_icl", isUsingICL, METH_VARARGS, "Extract sycl queue info, if it is using ICL"},
          {NULL, NULL, 0, NULL} // sentinel
        };

        static struct PyModuleDef ModuleDef = {
          PyModuleDef_HEAD_INIT,
          "spirv_utils",
          NULL, //documentation
          -1, //size
          ModuleMethods
        };

        PyMODINIT_FUNC PyInit_spirv_utils(void) {
          PyObject *m = PyModule_Create(&ModuleDef);
          if(m == NULL) {
            return NULL;
          }
          PyModule_AddFunctions(m, ModuleMethods);
          return m;
        }
        """

    def __init__(self):
        mod = compile_module_from_src(self._generate_src(), "spirv_utils")
        self.load_binary = mod.load_binary
        self.get_device_properties = mod.get_device_properties
        self.get_l0_queue = mod.get_l0_queue
        self.get_l0_imm_cmd_list = mod.get_l0_imm_cmd_list
        self.get_l0_dev_ptr = mod.get_l0_dev_ptr
        self.get_l0_ctxt_ptr = mod.get_l0_ctxt_ptr
        self.is_using_icl = mod.is_using_icl
        self.context = mod.init_context(ipex.xpu.current_stream().sycl_queue)
        self.device_count = mod.init_devices(ipex.xpu.current_stream().sycl_queue)
        self.event_pool = mod.init_event_pool()[0]
        self.current_device = 0 if self.device_count[0] > 0 else -1

    def get_current_device(instance):
        return instance.current_device

    def get_event_pool(instance):
        return instance.event_pool

    def set_current_device(instance, idx):
        assert instance.device_count[0] > idx, "Device id not found"
        instance.current_device = idx

    def get_device_capability(instance, idx):
        return (0, 0)


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

    format = "iiiiiiiiiiKKKKKOOOK" + ''.join(
        [format_of(_extracted_type(ty)) for ty in signature.values()])

    # generate glue code
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
      if (!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ, &num_warps, &num_ctas,
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
    return src


class XPULauncher(object):

    def __init__(self, src, metadata):
        ids = {
            "ids_of_tensormaps": metadata.get("ids_of_tensormaps", tuple()), "ids_of_folded_args":
            metadata.get("ids_of_folded_args",
                         tuple()), "ids_of_const_exprs": src.fn.constexprs if hasattr(src, "fn") else tuple()
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
        # FIXME
        return 0

    def get_current_target(self):
        return ("xpu", 0)

    @staticmethod
    def is_active():
        import torch
        return torch.xpu.is_available()

    def assemble_tensormap_to_arg(self, tensormaps_info, args):
        args_ptr = tuple([arg.data_ptr() if hasattr(arg, 'data_ptr') else arg for arg in args])
        return args_ptr
