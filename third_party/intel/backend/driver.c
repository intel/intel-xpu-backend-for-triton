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

static SyclQueueMap g_sycl_queue_map;

static std::vector<ze_device_handle_t> g_devices;
static std::vector<std::pair<sycl::device, ze_device_handle_t>>
    g_sycl_l0_device_list;

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

  if (device_id > g_sycl_l0_device_list.size()) {
    std::cerr << "Device is not found " << std::endl;
    return NULL;
  }
  const auto device = g_sycl_l0_device_list[device_id];

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
void freeKernel(PyObject *p) {
  delete reinterpret_cast<sycl::kernel *>(PyCapsule_GetPointer(p, "kernel"));
}

void freeKernelBundle(PyObject *p) {
  delete reinterpret_cast<
      sycl::kernel_bundle<sycl::bundle_state::executable> *>(
      PyCapsule_GetPointer(p, "kernel_bundle"));
}

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

  if (devId > g_sycl_l0_device_list.size()) {
    std::cerr << "Device is not found " << std::endl;
    return NULL;
  }

  const auto &sycl_l0_device_pair = g_sycl_l0_device_list[devId];
  const sycl::device sycl_device = sycl_l0_device_pair.first;

  std::string kernel_name = name;
  const size_t binary_size = PyBytes_Size(py_bytes);

  uint8_t *binary_ptr = (uint8_t *)PyBytes_AsString(py_bytes);
  const auto ctx = sycl_device.get_platform().ext_oneapi_get_default_context();
  const auto l0_device =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_device);
  const auto l0_context =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);

  const auto use_native_code =
      isEnvValueBool(getStrEnv("TRITON_XPU_GEN_NATIVE_CODE"));
  const bool is_spv = use_native_code ? !(*use_native_code) : true;

  auto l0_module = checkSyclErrors(create_module(
      l0_context, l0_device, binary_ptr, binary_size, build_flags, is_spv));

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
  const int32_t n_regs = 0;

  if (is_spv) {
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
      const std::optional<bool> debugEnabled =
          isEnvValueBool(getStrEnv("TRITON_DEBUG"));
      if (debugEnabled)
        std::cout << "(I): Detected " << n_spills
                  << " spills, recompiling kernel \"" << kernel_name
                  << "\" using large GRF mode" << std::endl;

      const std::string new_build_flags =
          build_flags_str.append(" -cl-intel-256-GRF-per-thread");
      l0_module = checkSyclErrors(
          create_module(l0_context, l0_device, binary_ptr, binary_size,
                        new_build_flags.c_str(), is_spv));

      l0_kernel = checkL0Errors(l0_module);
      gpuAssert(zeKernelGetProperties(l0_kernel, &props));
      n_spills = props.spillMemSize;

      if (debugEnabled)
        std::cout << "(I): Kernel has now " << n_spills << " spills"
                  << std::endl;
    }
  }

  auto mod = new sycl::kernel_bundle<sycl::bundle_state::executable>(
      sycl::make_kernel_bundle<sycl::backend::ext_oneapi_level_zero,
                               sycl::bundle_state::executable>(
          {l0_module, sycl::ext::oneapi::level_zero::ownership::transfer},
          ctx));
  sycl::kernel *fun =
      new sycl::kernel(sycl::make_kernel<sycl::backend::ext_oneapi_level_zero>(
          {*mod, l0_kernel, sycl::ext::oneapi::level_zero::ownership::transfer},
          ctx));
  auto kernel_py =
      PyCapsule_New(reinterpret_cast<void *>(fun), "kernel", freeKernel);
  auto kernel_bundle_py = PyCapsule_New(reinterpret_cast<void *>(mod),
                                        "kernel_bundle", freeKernelBundle);

  return Py_BuildValue("(OOii)", kernel_bundle_py, kernel_py, n_regs, n_spills);
}

static PyObject *initContext(PyObject *self, PyObject *args) {
  PyObject *cap;
  void *queue = NULL;
  if (!PyArg_ParseTuple(args, "O", &cap))
    return NULL;
  if (!(queue = PyLong_AsVoidPtr(cap)))
    return NULL;
  sycl::queue *sycl_queue = static_cast<sycl::queue *>(queue);
  if (g_sycl_queue_map.find(*sycl_queue) == g_sycl_queue_map.end()) {
    const auto updated_sycl_devices = update(*sycl_queue, g_sycl_queue_map);
    if (!updated_sycl_devices.empty()) {
      // Update global data
      const uint32_t deviceCount =
          std::min(updated_sycl_devices.size(), g_devices.size());
      for (uint32_t i = 0; i < deviceCount; ++i) {
        g_devices[i] = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
            updated_sycl_devices[i]);
      }
    }
  }
  auto context = g_sycl_queue_map[*sycl_queue].context;
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
  const std::vector<sycl::device> &sycl_devices = sycl_context.get_devices();

  // Retrieve l0 devices
  const uint32_t deviceCount = sycl_devices.size();
  for (uint32_t i = 0; i < deviceCount; ++i) {
    g_sycl_l0_device_list.push_back(std::make_pair(
        sycl_devices[i], sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
                             sycl_devices[i])));
    g_devices.push_back(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
        sycl_devices[i]));
  }

  return Py_BuildValue("(i)", deviceCount);
}

typedef struct {
  void *base;
  uint64_t shape[5];
  uint64_t strides[5];
} TMADesc;

// Simple helper to experiment creating TMA descriptors on the host.
// This is a useful to test TMA operations independently.
static PyObject *fill1DTMADescriptor(PyObject *self, PyObject *args) {
  unsigned long long global_address;
  uint64_t dim;
  uint32_t tensorDim;
  int elementSize;
  unsigned long long desc_address;
  if (!PyArg_ParseTuple(args, "KKiiK", &global_address, &dim, &tensorDim,
                        &elementSize, &desc_address)) {
    return NULL;
  }

  TMADesc *tmaDesc = (TMADesc *)(desc_address);
  tmaDesc->base = (void *)global_address;
  tmaDesc->shape[4] = dim;
  tmaDesc->strides[4] = 1;
  return Py_None;
}

// Simple helper to experiment creating TMA descriptors on the host.
// This is a useful to test TMA operations independently.
static PyObject *fill2DTMADescriptor(PyObject *self, PyObject *args) {
  unsigned long long global_address;
  uint64_t dims[2];
  uint32_t tensorDims[2];
  int elementSize;
  unsigned long long desc_address;
  if (!PyArg_ParseTuple(args, "KKKiiiK", &global_address, &dims[1], &dims[0],
                        &tensorDims[1], &tensorDims[0], &elementSize,
                        &desc_address)) {
    return NULL;
  }
  uint64_t globalStrides[2] = {dims[0] * elementSize,
                               dims[0] * dims[1] * elementSize};
  uint32_t elementStrides[2] = {1, 1};

  TMADesc *tmaDesc = (TMADesc *)(desc_address);
  tmaDesc->base = (void *)global_address;
  tmaDesc->shape[3] = dims[0];
  tmaDesc->shape[4] = dims[1];
  tmaDesc->strides[3] = dims[0];
  tmaDesc->strides[4] = 1;
  return Py_None;
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

    {"fill_1d_tma_descriptor", fill1DTMADescriptor, METH_VARARGS, "doc"},
    {"fill_2d_tma_descriptor", fill2DTMADescriptor, METH_VARARGS, "doc"},

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
