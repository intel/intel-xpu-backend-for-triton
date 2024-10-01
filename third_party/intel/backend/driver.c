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

template <typename T>
static inline T checkSyclErrors(const std::tuple<T, ze_result_t> tuple) {
  const auto code = std::get<1>(tuple);
  if (code != ZE_RESULT_SUCCESS) {
    throw std::runtime_error(parseZeResultCode(code));
  } else {
    return std::get<0>(tuple);
  }
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

using Spills = int32_t;

template <typename L0_DEVICE, typename L0_CONTEXT>
std::tuple<ze_module_handle_t, ze_kernel_handle_t, Spills>
compileLevelZeroObjects(uint8_t *binary_ptr, const size_t binary_size,
                        const std::string &kernel_name, L0_DEVICE l0_device,
                        L0_CONTEXT l0_context,
                        const std::string& build_flags, const bool is_spv) {
  auto l0_module =
      checkSyclErrors(create_module(l0_context, l0_device, binary_ptr,
                                    binary_size, build_flags.data(), is_spv));

  // Retrieve the kernel properties (e.g. register spills).
  auto l0_kernel = checkSyclErrors(create_function(l0_module, kernel_name));

  ze_kernel_properties_t props;
  props.stype = ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
  props.pNext = nullptr;
  checkSyclErrors(
      std::make_tuple(NULL, zeKernelGetProperties(l0_kernel, &props)));

  const int32_t n_spills = props.spillMemSize;

  return std::make_tuple(l0_module, l0_kernel, n_spills);
}

struct BuildFlags {
  std::string build_flags_str;

  const std::string LARGE_GRF_FLAG{"-cl-intel-256-GRF-per-thread"};
  const std::string SMALL_GRF_FLAG{"-cl-intel-128-GRF-per-thread"};
  const std::string AUTO_GRF_FLAG{"-cl-intel-enable-auto-large-GRF-mode"};

  BuildFlags(const char *build_flags) : build_flags_str(build_flags) {}

  const std::string& operator()() const {
    return build_flags_str;
  }

  int32_t n_regs() {
    if (build_flags_str.find(LARGE_GRF_FLAG) != std::string::npos) {
      return 256;
    }
    if (build_flags_str.find(SMALL_GRF_FLAG) != std::string::npos) {
      return 128;
    }
    // TODO: arguably we could return 128 if we find no flag instead of 0. For
    // now, stick with the conservative choice and alert the user only if a
    // specific GRF mode is specified.
    return 0;
  }

  const bool hasGRFSizeFlag() {
    if (build_flags_str.find(LARGE_GRF_FLAG) != std::string::npos ||
        build_flags_str.find(SMALL_GRF_FLAG) != std::string::npos ||
        build_flags_str.find(AUTO_GRF_FLAG) != std::string::npos) {
      return true;
    } else {
      return false;
    }
  }

  void addLargeGRFSizeFlag() { build_flags_str = build_flags_str.append(" " + LARGE_GRF_FLAG); }
};

static PyObject *loadBinary(PyObject *self, PyObject *args) {
  const char *name, *build_flags_ptr;
  int shared;
  PyObject *py_bytes;
  int devId;

  if (!PyArg_ParseTuple(args, "sSisi", &name, &py_bytes, &shared,
                        &build_flags_ptr, &devId)) {
    std::cerr << "loadBinary arg parse failed" << std::endl;
    return NULL;
  }

  if (devId > g_sycl_l0_device_list.size()) {
    std::cerr << "Device is not found " << std::endl;
    return NULL;
  }

  BuildFlags build_flags(build_flags_ptr);

  try {

    const auto &sycl_l0_device_pair = g_sycl_l0_device_list[devId];
    const sycl::device sycl_device = sycl_l0_device_pair.first;

    const std::string kernel_name = name;
    const size_t binary_size = PyBytes_Size(py_bytes);

    uint8_t *binary_ptr = (uint8_t *)PyBytes_AsString(py_bytes);
    const auto ctx =
        sycl_device.get_platform().ext_oneapi_get_default_context();
    const auto l0_device =
        sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_device);
    const auto l0_context =
        sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);

    const auto use_native_code =
        isEnvValueBool(getStrEnv("TRITON_XPU_GEN_NATIVE_CODE"));
    const bool is_spv = use_native_code ? !(*use_native_code) : true;

    auto [l0_module, l0_kernel, n_spills] =
        compileLevelZeroObjects(binary_ptr, binary_size, kernel_name, l0_device,
                                l0_context, build_flags(), is_spv);

    if (is_spv) {
      constexpr int32_t max_reg_spill = 1000;
      const bool is_GRF_mode_specified = build_flags.hasGRFSizeFlag();

      // If the register mode isn't set, and the number of spills is greater
      // than the threshold, recompile the kernel using large GRF mode.
      if (!is_GRF_mode_specified && n_spills > max_reg_spill) {
        const std::optional<bool> debugEnabled =
          isEnvValueBool(getStrEnv("TRITON_DEBUG"));
        if (debugEnabled)
          std::cout << "(I): Detected " << n_spills
                    << " spills, recompiling the kernel using large GRF mode"
                    << std::endl;

        build_flags.addLargeGRFSizeFlag();

        auto [l0_module, l0_kernel, n_spills] = compileLevelZeroObjects(
            binary_ptr, binary_size, kernel_name, l0_device, l0_context,
            build_flags(), is_spv);

      if (debugEnabled)
        std::cout << "(I): Kernel has now " << n_spills << " spills"
                  << std::endl;
      }
    }
    
    auto n_regs = build_flags.n_regs();

    auto mod = new sycl::kernel_bundle<sycl::bundle_state::executable>(
        sycl::make_kernel_bundle<sycl::backend::ext_oneapi_level_zero,
                                 sycl::bundle_state::executable>(
            {l0_module, sycl::ext::oneapi::level_zero::ownership::transfer},
            ctx));
    sycl::kernel *fun = new sycl::kernel(
        sycl::make_kernel<sycl::backend::ext_oneapi_level_zero>(
            {*mod, l0_kernel,
             sycl::ext::oneapi::level_zero::ownership::transfer},
            ctx));
    auto kernel_py =
        PyCapsule_New(reinterpret_cast<void *>(fun), "kernel", freeKernel);
    auto kernel_bundle_py = PyCapsule_New(reinterpret_cast<void *>(mod),
                                          "kernel_bundle", freeKernelBundle);

    return Py_BuildValue("(OOii)", kernel_bundle_py, kernel_py, n_regs,
                         n_spills);

  } catch (const std::exception &e) {
    char err[1024] = {0};
    std::string_view error_str(e.what());
    strncat(err, error_str.data(), std::min(error_str.size(), size_t(1024)));
    PyGILState_STATE gil_state;
    gil_state = PyGILState_Ensure();
    PyErr_SetString(PyExc_RuntimeError, err);
    std::cerr << "Error during Intel loadBinary: " << err << std::endl;
    PyGILState_Release(gil_state);
    return NULL;
  }
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
