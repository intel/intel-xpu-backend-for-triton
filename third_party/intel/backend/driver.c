//===- driver.c -----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstddef>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "sycl_functions.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

SyclQueueMap sycl_queue_map;
static ze_context_handle_t context = {nullptr};

static std::vector<ze_device_handle_t> devices;
static std::vector<std::pair<sycl::device, ze_device_handle_t>>
    sycl_l0_device_list;

static inline void gpuAssert(ze_result_t code) {
  if (code != ZE_RESULT_SUCCESS) {
    auto str = parseZeResultCode(code);
    char err[1024] = {0};
    strncat(err, str.c_str(), std::min(str.size(), size_t(1024)));
    PyGILState_STATE gil_state;
    gil_state = PyGILState_Ensure();
    PyErr_SetString(PyExc_RuntimeError, err);
    PyGILState_Release(gil_state);
  }
}

template <typename T>
static inline T checkSyclErrors(const std::tuple<T, ze_result_t> tuple) {
  gpuAssert(std::get<1>(tuple));
  if (PyErr_Occurred())
    return nullptr;
  else
    return std::get<0>(tuple);
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

  return Py_BuildValue("{s:i, s:i, s:i, s:i, s:i, s:i, s:N}", "max_shared_mem",
                       max_shared_mem, "multiprocessor_count",
                       multiprocessor_count, "sm_clock_rate", sm_clock_rate,
                       "mem_clock_rate", mem_clock_rate, "mem_bus_width",
                       mem_bus_width, "max_work_group_size", max_group_size,
                       "sub_group_sizes", subgroup_sizes);
}

std::vector<std::unique_ptr<sycl::kernel>> compiled_kernels;

static PyObject *loadBinary(PyObject *self, PyObject *args) {
  const char *name, *build_flags;
  int shared;
  PyObject *py_bytes;
  int devId;

  if (!PyArg_ParseTuple(args, "sSisi", &name, &py_bytes, &shared, &build_flags,
                        &devId)) {
    std::cerr << "loadBinary arg parse failed" << std::endl;
    return NULL;
  }

  if (devId > sycl_l0_device_list.size()) {
    std::cerr << "Device is not found " << std::endl;
    return NULL;
  }

  auto sycl_l0_device_pair = sycl_l0_device_list[devId];
  sycl::device sycl_device = sycl_l0_device_pair.first;

  std::string kernel_name = name;
  size_t binary_size = PyBytes_Size(py_bytes);
  binary_size = binary_size / sizeof(uint32_t);

  uint8_t *binary_ptr = (uint8_t *)PyBytes_AsString(py_bytes);
  auto ctx = sycl_device.get_platform().ext_oneapi_get_default_context();
  auto l0_device =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_device);
  auto l0_context = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
  auto l0_module = checkSyclErrors(create_module(
      l0_context, l0_device, binary_ptr, binary_size, build_flags));

  auto checkL0Errors = [&](auto l0_module) -> ze_kernel_handle_t {
    if (PyErr_Occurred()) {
      // check for errors from module creation
      return NULL;
    }
    ze_kernel_handle_t l0_kernel =
        checkSyclErrors(create_function(l0_module, kernel_name));
    if (PyErr_Occurred()) {
      // check for errors from kernel creation
      return NULL;
    }
    return l0_kernel;
  };

  // Retrieve the kernel properties (e.g. register spills).
  ze_kernel_handle_t l0_kernel = checkL0Errors(l0_module);
  ze_kernel_properties_t props;
  props.stype = ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
  props.pNext = nullptr;
  gpuAssert(zeKernelGetProperties(l0_kernel, &props));

  int32_t n_spills = props.spillMemSize;
  int32_t n_regs = 0;
  constexpr int32_t max_reg_spill = 1000;
  std::string build_flags_str(build_flags);
  bool is_GRF_mode_specified = false;

  // Check whether the GRF mode is specified by the build flags.
  if (build_flags_str.find("-cl-intel-256-GRF-per-thread") !=
          std::string::npos ||
      build_flags_str.find("-cl-intel-128-GRF-per-thread") !=
          std::string::npos ||
      build_flags_str.find("-cl-intel-enable-auto-large-GRF-mode") !=
          std::string::npos) {
    is_GRF_mode_specified = true;
  }

  // If the register mode isn't set, and the number of spills is greater
  // than the threshold, recompile the kernel using large GRF mode.
  if (!is_GRF_mode_specified && n_spills > max_reg_spill) {
    std::cout << "(I): Detected " << n_spills
              << " spills, recompiling the kernel using large GRF mode"
              << std::endl;
    const std::string new_build_flags =
        build_flags_str.append(" -cl-intel-256-GRF-per-thread");
    l0_module =
        checkSyclErrors(create_module(l0_context, l0_device, binary_ptr,
                                      binary_size, new_build_flags.c_str()));
    l0_kernel = checkL0Errors(l0_module);
    gpuAssert(zeKernelGetProperties(l0_kernel, &props));
    n_spills = props.spillMemSize;
    std::cout << "(I): Kernel has now " << n_spills << " spills" << std::endl;
  }

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

static PyObject *initContext(PyObject *self, PyObject *args) {
  PyObject *cap;
  void *queue = NULL;
  if (!PyArg_ParseTuple(args, "O", &cap))
    return NULL;
  if (!(queue = PyLong_AsVoidPtr(cap)))
    return NULL;
  sycl::queue *sycl_queue = static_cast<sycl::queue *>(queue);
  if (sycl_queue_map.find(*sycl_queue) == sycl_queue_map.end()) {
    auto updated_sycl_devices = update(*sycl_queue, sycl_queue_map);
    if (!updated_sycl_devices.empty()) {
      // Update global data
      context = sycl_queue_map[*sycl_queue].context;
      uint32_t deviceCount =
          std::min(updated_sycl_devices.size(), devices.size());
      for (uint32_t i = 0; i < deviceCount; ++i) {
        devices[i] = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
            updated_sycl_devices[i]);
      }
    }
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
