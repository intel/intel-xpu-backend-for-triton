//===- driver.c -----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstddef>
#include <iostream>
#include <level_zero/ze_api.h>
#include <string>
#include <sycl/sycl.hpp>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

typedef struct l0_resc_handles {
  ze_context_handle_t context;
  ze_device_handle_t device;
  ze_command_queue_handle_t queue;
  ze_command_list_handle_t cmd_list;
} l0_resc_handles;

std::unordered_map<sycl::queue, l0_resc_handles> sycl_queue_map;
static ze_context_handle_t context = {nullptr};

static std::vector<ze_device_handle_t> devices;
static std::vector<std::pair<sycl::device, ze_device_handle_t>>
    sycl_l0_device_list;

static inline void gpuAssert(ze_result_t code, const char *file, int line) {
  if (code != ZE_RESULT_SUCCESS) {
    const char *prefix = "Triton Error [ZE]: ";
    std::string str = std::to_string(code);
    char err[1024] = {0};
    strcat(err, prefix);
    strcat(err, str.c_str());
    PyGILState_STATE gil_state;
    gil_state = PyGILState_Ensure();
    PyErr_SetString(PyExc_RuntimeError, err);
    PyGILState_Release(gil_state);
  }
}

#define ZE_CHECK(ans)                                                          \
  {                                                                            \
    gpuAssert((ans), __FILE__, __LINE__);                                      \
    if (PyErr_Occurred())                                                      \
      return NULL;                                                             \
  }

static PyObject *getDeviceProperties(PyObject *self, PyObject *args) {
  int device_id;
  if (!PyArg_ParseTuple(args, "i", &device_id))
    return NULL;

  if (device_id > sycl_l0_device_list.size()) {
    std::cerr << "Device is not found " << std::endl;
    return NULL;
  }
  auto device = sycl_l0_device_list[device_id];

  // Get device handle
  ze_device_handle_t phDevice = device.second;

  // create a struct to hold device properties
  ze_device_properties_t device_properties = {};
  device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
  zeDeviceGetProperties(phDevice, &device_properties);

  int multiprocessor_count =
      device_properties.numSlices * device_properties.numSubslicesPerSlice;
  int sm_clock_rate = device_properties.coreClockRate;

  // Extract triton::gpu::intel::DeviceArch from pci_device_id
  // https://dgpu-docs.intel.com/devices/hardware-table.html
  int pci_device_id = device_properties.deviceId;
  int gpu_arch = 2; // triton::gpu::intel::DeviceArch::UNKNOWN
  switch ((pci_device_id >> 8) & 0xFF) {
  case 0x56:
    gpu_arch = 0; // Arc GPUs 56xx
    break;
  case 0x0B:      // PVC GPUs 0Bxx
    gpu_arch = 1; // PVC
    break;
  default:; // fall through
  }

  ze_device_compute_properties_t compute_properties = {};
  compute_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES;
  zeDeviceGetComputeProperties(phDevice, &compute_properties);
  int max_shared_mem = compute_properties.maxSharedLocalMemory;
  int max_group_size = compute_properties.maxTotalGroupSize;
  int num_subgroup_sizes = compute_properties.numSubGroupSizes;
  PyObject *subgroup_sizes = PyTuple_New(num_subgroup_sizes);
  for (int i = 0; i < num_subgroup_sizes; i++) {
    PyTuple_SetItem(subgroup_sizes, i,
                    PyLong_FromLong(compute_properties.subGroupSizes[i]));
  }

  uint32_t memoryCount = 0;
  zeDeviceGetMemoryProperties(phDevice, &memoryCount, nullptr);
  auto pMemoryProperties = new ze_device_memory_properties_t[memoryCount];
  for (uint32_t mem = 0; mem < memoryCount; ++mem) {
    pMemoryProperties[mem].stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
    pMemoryProperties[mem].pNext = nullptr;
  }
  zeDeviceGetMemoryProperties(phDevice, &memoryCount, pMemoryProperties);

  int mem_clock_rate = pMemoryProperties[0].maxClockRate;
  int mem_bus_width = pMemoryProperties[0].maxBusWidth;

  delete[] pMemoryProperties;

  return Py_BuildValue(
      "{s:i, s:i, s:i, s:i, s:i, s:i, s:i, s:O}", "max_shared_mem",
      max_shared_mem, "multiprocessor_count", multiprocessor_count,
      "sm_clock_rate", sm_clock_rate, "mem_clock_rate", mem_clock_rate,
      "mem_bus_width", mem_bus_width, "device_arch", gpu_arch,
      "max_work_group_size", max_group_size, "sub_group_sizes", subgroup_sizes);
}

/*Sycl code Start*/
bool getBoolEnv(const std::string &env) {
  const char *s = std::getenv(env.c_str());
  std::string str(s ? s : "");
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return (str == "on" || str == "true" || str == "1");
}

ze_module_handle_t create_module(ze_context_handle_t context,
                                 ze_device_handle_t device,
                                 uint32_t *binary_ptr, size_t binary_size) {
  const char *build_flags = "";
  const ze_module_format_t format = ZE_MODULE_FORMAT_IL_SPIRV;
  ze_module_desc_t module_description = {};
  module_description.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
  module_description.format = format;
  module_description.inputSize =
      static_cast<uint32_t>(binary_size * sizeof(uint32_t));
  module_description.pInputModule = (uint8_t *)binary_ptr;
  module_description.pBuildFlags = build_flags;
  ze_module_build_log_handle_t buildlog;
  ze_module_handle_t module;
  auto context_initial = context;
  auto device_initial = device;
  auto error_no = ZE_RESULT_SUCCESS;
  error_no =
      zeModuleCreate(context, device, &module_description, &module, &buildlog);
  if (error_no != ZE_RESULT_SUCCESS) {
    size_t szLog = 0;
    ZE_CHECK(zeModuleBuildLogGetString(buildlog, &szLog, nullptr));
    char *strLog = (char *)malloc(szLog);
    ZE_CHECK(zeModuleBuildLogGetString(buildlog, &szLog, strLog));
    std::cerr << "L0 build module failed. Log: " << strLog << std::endl;
    free(strLog);
    ZE_CHECK(zeModuleBuildLogDestroy(buildlog));
  }
  ZE_CHECK(error_no);
  return module;
}

void printModuleKernelName(ze_module_handle_t hModule) {
  uint32_t Count = 0;
  auto ret = zeModuleGetKernelNames(hModule, &Count, nullptr);
  assert(ret == ZE_RESULT_SUCCESS);
  std::unique_ptr<const char *[]> PNames(new const char *[Count]);
  ret = zeModuleGetKernelNames(hModule, &Count, PNames.get());
  assert(ret == ZE_RESULT_SUCCESS);
  if (getBoolEnv("MLIR_ENABLE_DUMP")) {
    for (uint32_t i = 0; i < Count; ++i) {
      std::cout << std::string(PNames[i]) << std::endl;
    }
  }
}

ze_kernel_handle_t create_function(ze_module_handle_t module,
                                   ze_kernel_flags_t flag,
                                   std::string func_name) {
  ze_kernel_handle_t kernel;
  ze_kernel_desc_t kernel_description = {};
  kernel_description.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
  kernel_description.pNext = nullptr;
  kernel_description.flags = flag;
  kernel_description.pKernelName = func_name.c_str();
  assert(module);
  auto module_initial = module;
  if (getBoolEnv("MLIR_ENABLE_DUMP")) {
    std::cout << "create kernel:" << func_name << std::endl;
  }
  ZE_CHECK(zeKernelCreate(module, &kernel_description, &kernel));
  return kernel;
}

ze_kernel_handle_t create_function(ze_module_handle_t module,
                                   std::string func_name) {
  return create_function(module, ZE_KERNEL_FLAG_FORCE_RESIDENCY, func_name);
}

std::vector<std::unique_ptr<sycl::kernel>> compiled_kernels;

static PyObject *loadBinary(PyObject *self, PyObject *args) {
  const char *name;
  int shared;
  PyObject *py_bytes;
  int devId;

  if (!PyArg_ParseTuple(args, "sSii", &name, &py_bytes, &shared, &devId)) {
    std::cerr << "loadBinary arg parse failed" << std::endl;
    return NULL;
  }
  int32_t n_regs = 0;
  int32_t n_spills = 0;

  if (devId > sycl_l0_device_list.size()) {
    std::cerr << "Device is not found " << std::endl;
    return NULL;
  }

  auto sycl_l0_device_pair = sycl_l0_device_list[devId];
  sycl::device sycl_device = sycl_l0_device_pair.first;

  std::string kernel_name = name;
  size_t binary_size = PyBytes_Size(py_bytes);
  binary_size = binary_size / sizeof(uint32_t);

  uint32_t *binary_ptr = (uint32_t *)PyBytes_AsString(py_bytes);
  ;
  auto ctx = sycl_device.get_platform().ext_oneapi_get_default_context();
  auto l0_device =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_device);
  auto l0_context = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
  auto l0_module =
      create_module(l0_context, l0_device, binary_ptr, binary_size);

  if (PyErr_Occurred()) {
    // check for errors from module creation
    return NULL;
  }

  auto l0_kernel = create_function(l0_module, kernel_name);

  if (PyErr_Occurred()) {
    // check for errors from kernel creation
    return NULL;
  }

  ze_kernel_properties_t props;
  props.stype = ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
  props.pNext = nullptr;
  ZE_CHECK(zeKernelGetProperties(l0_kernel, &props));
  n_spills = props.spillMemSize;
  auto mod = sycl::make_kernel_bundle<sycl::backend::ext_oneapi_level_zero,
                                      sycl::bundle_state::executable>(
      {l0_module, sycl::ext::oneapi::level_zero::ownership::transfer}, ctx);
  auto fun = sycl::make_kernel<sycl::backend::ext_oneapi_level_zero>(
      {mod, l0_kernel, sycl::ext::oneapi::level_zero::ownership::transfer},
      ctx);
  compiled_kernels.push_back(std::make_unique<sycl::kernel>(fun));
  sycl::kernel *ptr = compiled_kernels[compiled_kernels.size() - 1].get();
  if (getBoolEnv("MLIR_ENABLE_DUMP")) {
    std::cout << "compiled kernel ptr: " << ptr << std::endl;
    std::cout << "total kernels:" << compiled_kernels.size() << std::endl;
    for (auto &k : compiled_kernels) {
      std::cout << "  kernel:"
                << k->get_info<sycl::info::kernel::function_name>() << " @"
                << k.get() << std::endl;
    }
  }
  sycl::kernel *k = new sycl::kernel(*ptr);
  sycl::kernel_bundle<sycl::bundle_state::executable> *kb =
      new sycl::kernel_bundle<sycl::bundle_state::executable>(mod);
  return Py_BuildValue("(KKii)", (uint64_t)kb, (uint64_t)k, n_regs, n_spills);
}
/*Sycl code end*/

bool update(sycl::queue sycl_queue) {
  // Get l0-context
  auto sycl_context = sycl_queue.get_context();
  ze_context_handle_t hCtxt =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_context);
  // Get l0-device
  std::vector<sycl::device> sycl_devices = sycl_context.get_devices();
  ze_device_handle_t hDev =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_devices[0]);
  // Get l0-queue
  bool immediate_cmd_list = false;
  std::variant<ze_command_queue_handle_t, ze_command_list_handle_t> queue_var =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_queue);
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
    devices[i] =
        sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_devices[i]);
  }

  return true;
}

static PyObject *initContext(PyObject *self, PyObject *args) {
  PyObject *cap;
  void *queue = NULL;
  if (!PyArg_ParseTuple(args, "O", &cap))
    return NULL;
  if (!(queue = PyLong_AsVoidPtr(cap)))
    return NULL;
  sycl::queue *sycl_queue = static_cast<sycl::queue *>(queue);
  if (sycl_queue_map.find(*sycl_queue) == sycl_queue_map.end()) {
    update(*sycl_queue);
  }
  context = sycl_queue_map[*sycl_queue].context;
  return Py_BuildValue("(K)", (uint64_t)context);
}

static PyObject *initDevices(PyObject *self, PyObject *args) {
  PyObject *cap;
  void *queue = NULL;
  if (!PyArg_ParseTuple(args, "O", &cap))
    return NULL;
  if (!(queue = PyLong_AsVoidPtr(cap)))
    return NULL;
  sycl::queue *sycl_queue = static_cast<sycl::queue *>(queue);

  auto sycl_context = sycl_queue->get_context();

  // Get sycl-device
  std::vector<sycl::device> sycl_devices = sycl_context.get_devices();

  // Retrieve l0 devices
  uint32_t deviceCount = sycl_devices.size();
  for (uint32_t i = 0; i < deviceCount; ++i) {
    sycl_l0_device_list.push_back(std::make_pair(
        sycl_devices[i], sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
                             sycl_devices[i])));
    devices.push_back(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
        sycl_devices[i]));
  }

  return Py_BuildValue("(i)", deviceCount);
}

static PyMethodDef ModuleMethods[] = {
    {"load_binary", loadBinary, METH_VARARGS,
     "Load provided SPV into ZE driver"},
    {"get_device_properties", getDeviceProperties, METH_VARARGS,
     "Get the properties for a given device"},
    {"init_context", initContext, METH_VARARGS,
     "Initialize the ZE GPU context"},
    {"init_devices", initDevices, METH_VARARGS,
     "Initialize the ZE GPU devices and return device count"},
    {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {PyModuleDef_HEAD_INIT, "spirv_utils",
                                       NULL, // documentation
                                       -1,   // size
                                       ModuleMethods};

PyMODINIT_FUNC PyInit_spirv_utils(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if (m == NULL) {
    return NULL;
  }
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}
