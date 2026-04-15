//===- driver.c -----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstddef>
#include <cstring>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>
#if defined(TRITON_INTEL_INJECT_PYTORCH)
#include <ATen/record_function.h>
#endif

#if defined(_WIN32)
#define EXPORT_FUNC __declspec(dllexport)
#else
#define EXPORT_FUNC __attribute__((visibility("default")))
#endif

#include "sycl_functions.h"

#include <Python.h>

static std::vector<std::pair<sycl::device, ze_device_handle_t>>
    g_sycl_l0_device_list;

// Cache for IntelGPUError exception class
static PyObject *g_intel_gpu_error_class = nullptr;

static PyObject *getIntelGPUErrorClass() {
  if (g_intel_gpu_error_class != nullptr) {
    return g_intel_gpu_error_class;
  }

  PyObject *module = PyImport_ImportModule("triton.runtime.errors");
  if (module == nullptr) {
    PyErr_SetString(PyExc_ImportError, "cannot import triton.runtime.errors");
    return NULL;
  }

  g_intel_gpu_error_class = PyObject_GetAttrString(module, "IntelGPUError");
  Py_DECREF(module);

  if (g_intel_gpu_error_class == nullptr) {
    PyErr_SetString(PyExc_AttributeError,
                    "cannot find IntelGPUError in triton.runtime.errors");
    return NULL;
  }

  return g_intel_gpu_error_class;
}

static void zeConstructError(const char *file, int line, const char *message,
                             bool useIntelGPUError = false) {
  const char *prefix = "Triton Error [ZE] %s:%d: ";
  char err[1024] = {0};
  snprintf(err, sizeof(err), prefix, file, line);
  strcat(err, message);

  PyGILState_STATE gil_state;
  gil_state = PyGILState_Ensure();

  if (useIntelGPUError) {
    PyObject *exc_class = getIntelGPUErrorClass();
    if (exc_class != NULL) {
      PyErr_SetString(exc_class, err);
    } else {
      // Fallback to RuntimeError if IntelGPUError class cannot be retrieved.
      // Fetch the actual error message from getIntelGPUErrorClass failure.
      PyObject *orig_type, *orig_value, *orig_tb;
      PyErr_Fetch(&orig_type, &orig_value, &orig_tb);

      if (orig_value != NULL) {
        PyObject *str_repr = PyObject_Str(orig_value);
        if (str_repr != NULL) {
          const char *orig_msg = PyUnicode_AsUTF8(str_repr);
          if (orig_msg != NULL) {
            strcat(err, " (IntelGPUError not available: ");
            strcat(err, orig_msg);
            strcat(err, ")");
          }
          Py_DECREF(str_repr);
        }
      }

      Py_XDECREF(orig_type);
      Py_XDECREF(orig_value);
      Py_XDECREF(orig_tb);

      PyErr_SetString(PyExc_RuntimeError, err);
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, err);
  }

  PyGILState_Release(gil_state);
}

template <typename T>
static inline T
checkZeCodeAndSetPyErr(const std::tuple<T, ze_result_t> syclTuple,
                       const char *file, int line) {
  const auto code = std::get<1>(syclTuple);
  if (code == ZE_RESULT_SUCCESS)
    return std::get<0>(syclTuple);

  zeConstructError(file, line, parseZeResultCode(code).data(),
                   /* useIntelGPUError */ true);
  return std::get<0>(syclTuple);
}

extern "C" EXPORT_FUNC PyObject *get_device_properties(int device_id) {
  if (device_id >= g_sycl_l0_device_list.size()) {
    zeConstructError(__FILE__, __LINE__, "Device is not found");
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
  // To align with other backends - convert MHz to KHz
  int sm_clock_rate = device_properties.coreClockRate * 1000;

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

  // To align with other backends - convert MHz to KHz
  // https://github.com/intel/compute-runtime/blob/cfa007e5519d3a038d726b62237b86fca9a49e2c/shared/source/xe_hpc_core/linux/product_helper_pvc.cpp#L51
  int mem_clock_rate = pMemoryProperties[0].maxClockRate * 1000;
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
                        L0_CONTEXT l0_context, const std::string &build_flags,
                        const bool is_spv) {
  auto l0_module = checkZeCodeAndSetPyErr(
      create_module(l0_context, l0_device, binary_ptr, binary_size,
                    build_flags.data(), is_spv),
      __FILE__, __LINE__);
  if (PyErr_Occurred()) {
    return std::make_tuple(nullptr, nullptr, -1);
  }

  // Retrieve the kernel properties (e.g. register spills).
  auto l0_kernel = checkZeCodeAndSetPyErr(
      create_function(l0_module, kernel_name), __FILE__, __LINE__);
  if (PyErr_Occurred()) {
    return std::make_tuple(nullptr, nullptr, -1);
  }

  ze_kernel_properties_t props;
  props.stype = ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
  props.pNext = nullptr;

  checkZeCodeAndSetPyErr(
      std::make_tuple(NULL, zeKernelGetProperties(l0_kernel, &props)), __FILE__,
      __LINE__);
  if (PyErr_Occurred()) {
    return std::make_tuple(nullptr, nullptr, -1);
  }

  const int32_t n_spills = props.spillMemSize;

  return std::make_tuple(l0_module, l0_kernel, n_spills);
}

struct BuildFlags {
  std::string build_flags_str;

  const char *LARGE_GRF_FLAG{"-cl-intel-256-GRF-per-thread"};
  const char *SMALL_GRF_FLAG{"-cl-intel-128-GRF-per-thread"};
  const char *AUTO_GRF_FLAG{"-cl-intel-enable-auto-large-GRF-mode"};

  BuildFlags(const char *build_flags) : build_flags_str(build_flags) {}

  const std::string &operator()() const { return build_flags_str; }

  int32_t n_regs() const {
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

  const bool hasGRFSizeFlag() const {
    if (build_flags_str.find(LARGE_GRF_FLAG) != std::string::npos ||
        build_flags_str.find(SMALL_GRF_FLAG) != std::string::npos ||
        build_flags_str.find(AUTO_GRF_FLAG) != std::string::npos) {
      return true;
    }

    return false;
  }

  void addLargeGRFSizeFlag() {
    build_flags_str = build_flags_str.append(" ").append(LARGE_GRF_FLAG);
  }
};

sycl::context get_default_context(const sycl::device &sycl_device) {
  const auto &platform = sycl_device.get_platform();
#if defined(_WIN32)
  sycl::context ctx;
  try {
#if __SYCL_COMPILER_VERSION >= 20250604
    ctx = platform.khr_get_default_context();
#else
    ctx = platform.ext_oneapi_get_default_context();
#endif
  } catch (const std::runtime_error &ex) {
    // This exception is thrown on Windows because
    // khr_get_default_context is not implemented. But it can be safely
    // ignored it seems.
#if _DEBUG
    std::cout << "ERROR: " << ex.what() << std::endl;
#endif
  }
  return ctx;
#else
#if __SYCL_COMPILER_VERSION >= 20250604
  return platform.khr_get_default_context();
#else
  return platform.ext_oneapi_get_default_context();
#endif
#endif
}

static BuildFlags last_build_flag("");

extern "C" EXPORT_FUNC PyObject *get_last_selected_build_flags() {
  return Py_BuildValue("s", last_build_flag().data());
}

extern "C" EXPORT_FUNC PyObject *load_binary(PyObject *args) {
  const char *name, *build_flags_ptr;
  int shared;
  PyObject *py_bytes;
  int is_spv;
  int devId;

  if (!PyArg_ParseTuple(args, "sSispi", &name, &py_bytes, &shared,
                        &build_flags_ptr, &is_spv, &devId)) {
    // PyArg_ParseTuple will set a PyErr
    return NULL;
  }

  if (devId >= g_sycl_l0_device_list.size()) {
    zeConstructError(__FILE__, __LINE__, "Device is not found");
    return NULL;
  }

  BuildFlags build_flags(build_flags_ptr);

  const auto &sycl_l0_device_pair = g_sycl_l0_device_list[devId];
  const sycl::device sycl_device = sycl_l0_device_pair.first;
  const auto l0_device = sycl_l0_device_pair.second;

  const std::string kernel_name = name;
  const size_t binary_size = PyBytes_Size(py_bytes);

  uint8_t *binary_ptr = (uint8_t *)PyBytes_AsString(py_bytes);
  const auto &ctx = get_default_context(sycl_device);
  const auto l0_context =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);

  ze_device_compute_properties_t compute_properties = {};
  compute_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES;
  zeDeviceGetComputeProperties(l0_device, &compute_properties);
  int32_t n_max_threads = compute_properties.maxTotalGroupSize;

  auto [l0_module, l0_kernel, n_spills] =
      compileLevelZeroObjects(binary_ptr, binary_size, kernel_name, l0_device,
                              l0_context, build_flags(), is_spv);

  const bool debugEnabled = getBoolEnv("TRITON_DEBUG");

  // If the initial compilation failed entirely (e.g., scratch space exceeds
  // HW limit), and GRF mode was not explicitly set, retry with large GRF mode.
  // This handles cases where the default GRF mode doesn't provide enough
  // registers, causing the backend compiler to fail.
  if (PyErr_Occurred() && is_spv && !build_flags.hasGRFSizeFlag()) {
    // Save the original error before clearing it for the retry attempt.
    PyObject *orig_type, *orig_value, *orig_tb;
    PyErr_Fetch(&orig_type, &orig_value, &orig_tb);

    if (debugEnabled)
      std::cout << "(I): Build failed for \"" << kernel_name
                << "\", retrying with large GRF mode" << std::endl;

    build_flags.addLargeGRFSizeFlag();

    auto [l0_module_retry, l0_kernel_retry, n_spills_retry] =
        compileLevelZeroObjects(binary_ptr, binary_size, kernel_name, l0_device,
                                l0_context, build_flags(), is_spv);
    if (PyErr_Occurred()) {
      // Retry also failed — propagate the original error.
      PyErr_Restore(orig_type, orig_value, orig_tb);
      return NULL;
    }

    // Retry succeeded — discard the saved original error.
    Py_XDECREF(orig_type);
    Py_XDECREF(orig_value);
    Py_XDECREF(orig_tb);

    l0_module = l0_module_retry;
    l0_kernel = l0_kernel_retry;
    n_spills = n_spills_retry;

    // Always print recovery message to stderr to follow up on the
    // "L0 build module failed" error that was already printed.
    std::cerr << "(I): Build failure recovered by retrying with large GRF "
                 "mode for \""
              << kernel_name << "\"" << std::endl;

    if (debugEnabled)
      std::cout << "(I): Retry with large GRF succeeded, kernel has "
                << n_spills << " spills" << std::endl;
  } else if (PyErr_Occurred()) {
    return NULL;
  }

  if (is_spv) {
    constexpr int32_t max_reg_spill = 1000;
    const bool is_GRF_mode_specified = build_flags.hasGRFSizeFlag();

    // If the register mode isn't set, and the number of spills is greater
    // than the threshold, recompile the kernel using large GRF mode.
    if (!is_GRF_mode_specified && n_spills > max_reg_spill) {
      if (debugEnabled)
        std::cout << "(I): Detected " << n_spills
                  << " spills, recompiling the kernel using large GRF mode"
                  << std::endl;

      build_flags.addLargeGRFSizeFlag();

      try {
        auto [l0_module_dgrf, l0_kernel_dgrf, n_spills_dgrf] =
            compileLevelZeroObjects(binary_ptr, binary_size, kernel_name,
                                    l0_device, l0_context, build_flags(),
                                    is_spv);

        if (debugEnabled)
          std::cout << "(I): Kernel has now " << n_spills_dgrf << " spills"
                    << std::endl;

        std::swap(l0_module, l0_module_dgrf);
        std::swap(l0_kernel, l0_kernel_dgrf);
        std::swap(n_spills, n_spills_dgrf);

        // clean up the unused module and kernel.
        auto error_no = zeKernelDestroy(l0_kernel_dgrf);
        if (error_no != ZE_RESULT_SUCCESS) {
          PyErr_WarnEx(
              PyExc_RuntimeWarning,
              "[Ignoring] Intel - Error during destroy unused L0 kernel", 1);
        }
        error_no = zeModuleDestroy(l0_module_dgrf);
        if (error_no != ZE_RESULT_SUCCESS) {
          PyErr_WarnEx(
              PyExc_RuntimeWarning,
              "[Ignoring] Intel - Error during destroy unused L0 module", 1);
        }
      } catch (const std::exception &e) {
        char buf[1024] = {0};
        strcat(buf, "[Ignoring] Intel - Error during Intel loadBinary with "
                    "large registers: ");
        strcat(buf, e.what());
        PyErr_WarnEx(PyExc_RuntimeWarning, buf, 1);
        // construct previous working version
        build_flags = BuildFlags(build_flags_ptr);
      }
    }
  }

  if (debugEnabled && n_spills) {
    std::cout << "(I): Detected " << n_spills << " spills for  \""
              << kernel_name << "\"" << std::endl;
  }

  auto n_regs = build_flags.n_regs();

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
  last_build_flag = build_flags;
  return Py_BuildValue("(OOiii)", kernel_bundle_py, kernel_py, n_regs, n_spills,
                       n_max_threads);
}

extern "C" EXPORT_FUNC PyObject *init_devices(PyObject *cap) {
  void *queue = NULL;
  if (!(queue = PyLong_AsVoidPtr(cap))) {
    zeConstructError(__FILE__, __LINE__,
                     "Failed to convert PyObject to void* for queue");
    return NULL;
  }
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
  }

  return Py_BuildValue("(i)", deviceCount);
}

extern "C" EXPORT_FUNC PyObject *wait_on_sycl_queue(PyObject *cap) {
  void *queue = NULL;
  if (!(queue = PyLong_AsVoidPtr(cap))) {
    zeConstructError(__FILE__, __LINE__,
                     "Failed to convert PyObject to void* for queue");
    return NULL;
  }
  sycl::queue *sycl_queue = static_cast<sycl::queue *>(queue);
  sycl_queue->wait();
  Py_RETURN_NONE;
}

extern "C" EXPORT_FUNC PyObject *sycl_queue_memset(PyObject *args) {
  PyObject *py_queue;
  uint64_t ptr, count;
  int value;

  if (!PyArg_ParseTuple(args, "OKiK", &py_queue, &ptr, &value, &count)) {
    return NULL;
  }

  sycl::queue *queue = static_cast<sycl::queue *>(PyLong_AsVoidPtr(py_queue));
  if (!queue) {
    zeConstructError(__FILE__, __LINE__, "Invalid SYCL queue pointer");
    return NULL;
  }

  try {
    queue->memset((void *)ptr, value, (size_t)count);
  } catch (const sycl::exception &e) {
    zeConstructError(__FILE__, __LINE__, e.what());
    return NULL;
  }

  Py_RETURN_NONE;
}

typedef enum { ARG_CONSTEXPR = 0, ARG_KERNEL = 1, ARG_TUPLE = 2 } ArgType;

// Annotation struct to know how the argument should be handled.
typedef struct {
  PyObject_HEAD;
  PyObject *nested_tuple; // Can be a List of PyKernelArgObjects or None
  ArgType type;
} PyKernelArgObject;

// Deallocator
static void PyKernelArg_dealloc(PyKernelArgObject *self) {
  Py_XDECREF(self->nested_tuple);
  Py_TYPE(self)->tp_free((PyObject *)self);
}

// Constructor
static int PyKernelArg_init(PyKernelArgObject *self, PyObject *args,
                            PyObject *kwds) {
  static char *kwlist[] = {"nested_tuple", "type", NULL};
  PyObject *tup = NULL;
  int type_val = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|i", kwlist, &tup,
                                   &type_val)) {
    return -1;
  }
  Py_XINCREF(tup);
  self->nested_tuple = tup;
  self->type = (ArgType)type_val;
  return 0;
}

static void PyKernelArg_free(void *ptr) { free(ptr); }

static PyTypeObject PyKernelArgType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name =
        "triton.backends.intel.PyKernelArg",
    .tp_basicsize = sizeof(PyKernelArgObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Kernel Argument Metadata",
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)PyKernelArg_init,
    .tp_dealloc = (destructor)PyKernelArg_dealloc,
};

static inline void gpuAssert(ze_result_t code, const char *file, int line) {
  if (code != ZE_RESULT_SUCCESS) {
    const char *prefix = "Triton Error [ZE]: ";
    std::string str = std::to_string(code);
    char err[1024] = {{0}};
    strcat(err, prefix);
    strcat(err, str.c_str());
    PyErr_SetString(PyExc_RuntimeError, err);
  }
}

static inline bool checkDevicePointer(void *ptr, int idx,
                                      const sycl::queue &queue) {
  if (ptr == nullptr) {
    return true;
  }
  auto context = queue.get_context();
  auto handle = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(context);
  ze_memory_allocation_properties_t prop;
  prop.stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
  prop.pNext = nullptr;
  ze_device_handle_t device;
  auto res =
      zeMemGetAllocProperties((ze_context_handle_t)handle, ptr, &prop, &device);
  if (res != ZE_RESULT_SUCCESS) {
    PyErr_Format(
        PyExc_ValueError,
        "Cannot get memory properties for pointer argument (at %d, err=%d)",
        idx, res);
    return false;
  } else if (prop.type == ZE_MEMORY_TYPE_UNKNOWN) {
    // We can work with any memory, known to the driver:
    // ZE_MEMORY_TYPE_DEVICE, ZE_MEMORY_TYPE_SHARED, ZE_MEMORY_TYPE_HOST
    PyErr_Format(
        PyExc_ValueError,
        "Pointer argument (at %d) doesn't reference accessible memory.", idx);
    return false;
  }
  return true;
}

static thread_local const sycl::queue *g_pointer_check_queue = nullptr;
static thread_local int g_pointer_check_arg_idx = -1;

struct PointerCheckScope {
  explicit PointerCheckScope(const sycl::queue &queue) {
    g_pointer_check_queue = &queue;
    g_pointer_check_arg_idx = -1;
  }

  ~PointerCheckScope() {
    g_pointer_check_arg_idx = -1;
    g_pointer_check_queue = nullptr;
  }
};

// start sycl
template <class T>
static inline void set_scalar_arg(sycl::handler &cgh, int index,
                                  const void *value) {
  cgh.set_arg(index, *static_cast<const T *>(value));
}

typedef enum {
  EXTRACTOR_UNKOWN_INDEX = 0,
  // pointers
  EXTRACTOR_POINTER_INDEX = 1,
  // ints
  EXTRACTOR_INT8_INDEX = 2,
  EXTRACTOR_INT16_INDEX = 3,
  EXTRACTOR_INT32_INDEX = 4,
  EXTRACTOR_INT64_INDEX = 5,
  // uints
  EXTRACTOR_UINT8_INDEX = 6,
  EXTRACTOR_UINT16_INDEX = 7,
  EXTRACTOR_UINT32_INDEX = 8,
  EXTRACTOR_UINT64_INDEX = 9,
  // floats
  EXTRACTOR_FP16_INDEX = 10,
  EXTRACTOR_BF16_INDEX = 11,
  EXTRACTOR_FP32_INDEX = 12,
  EXTRACTOR_FP64_INDEX = 13,
  // custom
  // last entry to have a count
  EXTRACTOR_TYPE_COUNT
} ExtractorTypeIndex;

// In C it's not needed
ExtractorTypeIndex &operator++(ExtractorTypeIndex &idx) {
  idx = static_cast<ExtractorTypeIndex>(static_cast<int>(idx) + 1);
  return idx;
}

static inline void setScalarArgByType(sycl::handler &cgh, int index,
                                      const void *value, uint8_t type_idx) {
  switch ((ExtractorTypeIndex)type_idx) {
  case EXTRACTOR_POINTER_INDEX:
    set_scalar_arg<void *>(cgh, index, value);
    break;
  case EXTRACTOR_INT8_INDEX:
    set_scalar_arg<int8_t>(cgh, index, value);
    break;
  case EXTRACTOR_INT16_INDEX:
    set_scalar_arg<int16_t>(cgh, index, value);
    break;
  case EXTRACTOR_INT32_INDEX:
    set_scalar_arg<int32_t>(cgh, index, value);
    break;
  case EXTRACTOR_INT64_INDEX:
    set_scalar_arg<int64_t>(cgh, index, value);
    break;
  case EXTRACTOR_UINT8_INDEX:
    set_scalar_arg<uint8_t>(cgh, index, value);
    break;
  case EXTRACTOR_UINT16_INDEX:
    set_scalar_arg<uint16_t>(cgh, index, value);
    break;
  case EXTRACTOR_UINT32_INDEX:
    set_scalar_arg<uint32_t>(cgh, index, value);
    break;
  case EXTRACTOR_UINT64_INDEX:
    set_scalar_arg<uint64_t>(cgh, index, value);
    break;
  case EXTRACTOR_FP16_INDEX:
    set_scalar_arg<uint16_t>(cgh, index, value);
    break;
  case EXTRACTOR_BF16_INDEX:
    set_scalar_arg<uint16_t>(cgh, index, value);
    break;
  case EXTRACTOR_FP32_INDEX:
    set_scalar_arg<uint32_t>(cgh, index, value);
    break;
  case EXTRACTOR_FP64_INDEX:
    set_scalar_arg<uint64_t>(cgh, index, value);
    break;
  default:
    break;
  }
}

static inline void printScalarArgByType(uint32_t index, const void *value,
                                        uint8_t type_idx) {
  std::cout << "  param[" << index << "] value=";
  switch ((ExtractorTypeIndex)type_idx) {
  case EXTRACTOR_POINTER_INDEX:
    std::cout << *static_cast<void *const *>(value);
    break;
  case EXTRACTOR_INT8_INDEX:
    std::cout << *static_cast<const int8_t *>(value);
    break;
  case EXTRACTOR_INT16_INDEX:
    std::cout << *static_cast<const int16_t *>(value);
    break;
  case EXTRACTOR_INT32_INDEX:
    std::cout << *static_cast<const int32_t *>(value);
    break;
  case EXTRACTOR_INT64_INDEX:
    std::cout << *static_cast<const int64_t *>(value);
    break;
  case EXTRACTOR_UINT8_INDEX:
    std::cout << *static_cast<const uint8_t *>(value);
    break;
  case EXTRACTOR_UINT16_INDEX:
    std::cout << *static_cast<const uint16_t *>(value);
    break;
  case EXTRACTOR_UINT32_INDEX:
    std::cout << *static_cast<const uint32_t *>(value);
    break;
  case EXTRACTOR_UINT64_INDEX:
    std::cout << *static_cast<const uint64_t *>(value);
    break;
  case EXTRACTOR_FP16_INDEX:
  case EXTRACTOR_BF16_INDEX:
    // Stored as 16-bit payload; print raw bits for debugging.
    std::cout << *static_cast<const uint16_t *>(value);
    break;
  case EXTRACTOR_FP32_INDEX: {
    std::cout << *static_cast<const float *>(value);
    break;
  }
  case EXTRACTOR_FP64_INDEX: {
    std::cout << *static_cast<const double *>(value);
    break;
  }
  default:
    std::cout << "<unknown>";
    break;
  }
  std::cout << std::endl;
}

static PyObject *data_ptr_str = NULL;

// Extract a XPU device pointer from a pointer-like PyObject obj, and store
// it to the memory location pointed by ptr.
bool extractPointer(void *ptr, PyObject *obj) {
  if (obj == Py_None) {
    *(void **)ptr = nullptr; // valid nullptr
    return true;
  }
  if (PyLong_Check(obj)) {
    *(void **)ptr = PyLong_AsVoidPtr(obj);
    return checkDevicePointer(*(void **)ptr, g_pointer_check_arg_idx,
                              *g_pointer_check_queue);
  }

  PyObject *data_ptr = PyObject_GetAttrString(obj, "data_ptr");

  if (data_ptr) {
    PyObject *ret = PyObject_CallNoArgs(data_ptr);
    Py_DECREF(data_ptr);
    if (!ret) {
      PyErr_SetString(
          PyExc_TypeError,
          "Pointer argument must be either uint64 or have data_ptr method");
      return false;
    }
    if (!PyLong_Check(ret)) {
      PyErr_SetString(
          PyExc_TypeError,
          "data_ptr method of Pointer object must return 64-bit int");
      return false;
    }
    *(void **)ptr = PyLong_AsVoidPtr(ret);
    Py_DECREF(ret);
    return checkDevicePointer(*(void **)ptr, g_pointer_check_arg_idx,
                              *g_pointer_check_queue);
  }
  PyErr_SetString(
      PyExc_TypeError,
      "Pointer argument must be either uint64 or have data_ptr method");
  return false;
}

bool extractI8(void *ptr, PyObject *obj) {
  *((int8_t *)ptr) = PyLong_AsLong(obj);
  return PyErr_Occurred() == NULL;
}

bool extractI16(void *ptr, PyObject *obj) {
  *((int16_t *)ptr) = PyLong_AsLong(obj);
  return PyErr_Occurred() == NULL;
}

bool extractI32(void *ptr, PyObject *obj) {
  *((int32_t *)ptr) = PyLong_AsLong(obj);
  return PyErr_Occurred() == NULL;
}

bool extractI64(void *ptr, PyObject *obj) {
  *((int64_t *)ptr) = PyLong_AsLongLong(obj);
  return PyErr_Occurred() == NULL;
}

bool extractU8(void *ptr, PyObject *obj) {
  *((uint8_t *)ptr) = PyLong_AsUnsignedLong(obj);
  return PyErr_Occurred() == NULL;
}

bool extractU16(void *ptr, PyObject *obj) {
  *((uint16_t *)ptr) = PyLong_AsUnsignedLong(obj);
  return PyErr_Occurred() == NULL;
}

bool extractU32(void *ptr, PyObject *obj) {
  *((uint32_t *)ptr) = PyLong_AsUnsignedLong(obj);
  return PyErr_Occurred() == NULL;
}

bool extractU64(void *ptr, PyObject *obj) {
  *((uint64_t *)ptr) = PyLong_AsUnsignedLongLong(obj);
  return PyErr_Occurred() == NULL;
}

bool extractFP16(void *ptr, PyObject *obj) {
  double temp_double = PyFloat_AsDouble(obj);
  uint16_t result;
  // from https://github.com/python/pythoncapi-compat
#if 0x030600B1 <= PY_VERSION_HEX && PY_VERSION_HEX <= 0x030B00A1 &&            \
    !defined(PYPY_VERSION)
  _PyFloat_Pack2(temp_double, (unsigned char *)&result, 1);
#else
  PyFloat_Pack2(temp_double, (char *)&result, 1);
#endif
  *((uint16_t *)ptr) = result;
  return PyErr_Occurred() == NULL;
}

bool extractBF16(void *ptr, PyObject *obj) {
  double temp_double = PyFloat_AsDouble(obj);
  float f32 = (float)temp_double;
  uint32_t u32 = *(uint32_t *)&f32;
  *((uint16_t *)ptr) = (u32 >> 16);
  return PyErr_Occurred() == NULL;
}

bool extractFP32(void *ptr, PyObject *obj) {
  double temp_double = PyFloat_AsDouble(obj);
  float f32 = (float)temp_double;
  *((uint32_t *)ptr) = *(uint32_t *)&f32;
  return PyErr_Occurred() == NULL;
}

bool extractFP64(void *ptr, PyObject *obj) {
  double temp_double = PyFloat_AsDouble(obj);
  *((uint64_t *)ptr) = *(uint64_t *)&temp_double;
  return PyErr_Occurred() == NULL;
}

typedef bool (*ExtractorFunc)(void *ptr, PyObject *obj);

#define MAX_NAMES_PER_EXTRACTOR 2

typedef struct {
  ExtractorFunc extract;
  size_t size;
  size_t alignment;
  const char *name[MAX_NAMES_PER_EXTRACTOR];
} Extractor;

Extractor extraction_map[EXTRACTOR_TYPE_COUNT] = {
    [EXTRACTOR_UNKOWN_INDEX] =
        (Extractor){.extract = NULL, .size = 0, .name = NULL},
    [EXTRACTOR_POINTER_INDEX] = (Extractor){.extract = extractPointer,
                                            .size = sizeof(void *),
                                            .name = NULL},
    [EXTRACTOR_INT8_INDEX] = (Extractor){.extract = extractI8,
                                         .size = sizeof(int8_t),
                                         .name = {"i8"}},
    [EXTRACTOR_INT16_INDEX] = (Extractor){.extract = extractI16,
                                          .size = sizeof(int16_t),
                                          .name = {"i16"}},
    [EXTRACTOR_INT32_INDEX] = (Extractor){.extract = extractI32,
                                          .size = sizeof(int32_t),
                                          .name = {"i1", "i32"}},
    [EXTRACTOR_INT64_INDEX] = (Extractor){.extract = extractI64,
                                          .size = sizeof(int64_t),
                                          .name = {"i64"}},
    [EXTRACTOR_UINT8_INDEX] = (Extractor){.extract = extractU8,
                                          .size = sizeof(uint8_t),
                                          .name = {"u8"}},
    [EXTRACTOR_UINT16_INDEX] = (Extractor){.extract = extractU16,
                                           .size = sizeof(uint16_t),
                                           .name = {"u16"}},
    [EXTRACTOR_UINT32_INDEX] = (Extractor){.extract = extractU32,
                                           .size = sizeof(uint32_t),
                                           .name = {"u1", "u32"}},
    [EXTRACTOR_UINT64_INDEX] = (Extractor){.extract = extractU64,
                                           .size = sizeof(uint64_t),
                                           .name = {"u64"}},
    [EXTRACTOR_FP16_INDEX] = (Extractor){.extract = extractFP16,
                                         .size = sizeof(uint16_t),
                                         .name = {"fp16"}},
    [EXTRACTOR_BF16_INDEX] = (Extractor){.extract = extractBF16,
                                         .size = sizeof(uint16_t),
                                         .name = {"bf16"}},
    [EXTRACTOR_FP32_INDEX] = (Extractor){.extract = extractFP32,
                                         .size = sizeof(uint32_t),
                                         .name = {"fp32", "f32"}},
    [EXTRACTOR_FP64_INDEX] = (Extractor){.extract = extractFP64,
                                         .size = sizeof(uint64_t),
                                         .name = {"fp64"}},
};

Extractor getExtractor(uint8_t index) {
  if (index >= EXTRACTOR_TYPE_COUNT) {
    return extraction_map[EXTRACTOR_UNKOWN_INDEX];
  }
  return extraction_map[index];
}

static void sycl_kernel_launch(uint32_t gridX, uint32_t gridY, uint32_t gridZ,
                               int num_warps, int threads_per_warp,
                               int shared_memory, sycl::queue &stream,
                               sycl::kernel &kernel_ptr, void *global_scratch,
                               void *profile_scratch, uint32_t num_params,
                               void **params, uint8_t *extractor_data) {

  std::string kernel_name =
      kernel_ptr.get_info<sycl::info::kernel::function_name>();
#if defined(TRITON_INTEL_INJECT_PYTORCH)
  RECORD_FUNCTION("XPU Triton kernel:" + kernel_name, {});
#endif

  uint32_t expected_num_params =
      kernel_ptr.get_info<sycl::info::kernel::num_args>();
  size_t global_range_x = gridX * threads_per_warp * num_warps;
  size_t global_range_y = gridY;
  size_t global_range_z = gridZ;
  size_t local_range_x = num_warps * threads_per_warp;
  size_t local_range_y = 1;
  size_t local_range_z = 1;
  sycl::range<3> global_range(global_range_z, global_range_y, global_range_x);
  sycl::range<3> local_range(local_range_z, local_range_y, local_range_x);
  sycl::nd_range<3> parallel_work_size(global_range, local_range);
  if (shared_memory) {
    expected_num_params -= 1;
  }

  static bool launchDebug = getBoolEnv("TRITON_INTEL_LAUNCH_DEBUG");
  if (launchDebug) {
    std::cout << "kernel info name:" << kernel_name << " @" << &kernel_ptr
              << std::endl;
    std::cout << "kernel info attributes:"
              << kernel_ptr.get_info<sycl::info::kernel::attributes>()
              << std::endl;
    std::cout << "kernel info reference_count:"
              << kernel_ptr.get_info<sycl::info::kernel::reference_count>()
              << std::endl;
    std::cout << "kernel info num_args:"
              << kernel_ptr.get_info<sycl::info::kernel::num_args>()
              << std::endl;

    std::cout << "launch num param:" << num_params << std::endl;
    std::cout << "  gridx: " << gridX << std::endl;
    std::cout << "  gridY: " << gridY << std::endl;
    std::cout << "  gridZ: " << gridZ << std::endl;
    std::cout << "  num_warps: " << num_warps << std::endl;
    std::cout << "  threads_per_warp: " << threads_per_warp << std::endl;
    std::cout << "  global range:[" << "x:" << global_range_x
              << ", y:" << global_range_y << ", z:" << global_range_z << "]"
              << std::endl;
    std::cout << "  local range:[" << "x:" << local_range_x
              << ", y:" << local_range_y << ", z:" << local_range_z << "]"
              << std::endl;
    std::cout << "  shared_memory: " << shared_memory << std::endl;

    for (uint32_t idx = 0; idx < num_params - 2; ++idx)
      printScalarArgByType(idx, params[idx], extractor_data[idx]);
    // print scratch memory arguments
    printScalarArgByType(num_params - 2, params[num_params - 2],
                         EXTRACTOR_POINTER_INDEX);
    printScalarArgByType(num_params - 1, params[num_params - 1],
                         EXTRACTOR_POINTER_INDEX);
  }
  assert(num_params == expected_num_params &&
         "number of kernel param not matched");
  // Submit the imported kernel.
  auto cgf = [&](sycl::handler &cgh) {
    // Set kernel arguments dynamically using extractor type information
    for (uint32_t idx = 0; idx < num_params - 2; ++idx) {
      setScalarArgByType(cgh, idx, params[idx], extractor_data[idx]);
    }
    // Set scratch memory arguments
    set_scalar_arg<void *>(cgh, num_params - 2, params[num_params - 2]);
    set_scalar_arg<void *>(cgh, num_params - 1, params[num_params - 1]);
    if (shared_memory) {
      using share_mem_t = sycl::local_accessor<int8_t, 1>;
      share_mem_t local_buffer = share_mem_t(shared_memory, cgh);
      cgh.set_arg(num_params, local_buffer);
      cgh.parallel_for(parallel_work_size, kernel_ptr);
    } else {
      cgh.parallel_for(parallel_work_size, kernel_ptr);
    }
  };
  auto event = stream.submit(cgf);
}
// end sycl

bool isMatch(const char *type_bytes, ExtractorTypeIndex idx) {
  Extractor extractor = extraction_map[idx];
  for (int j = 0; j < MAX_NAMES_PER_EXTRACTOR; j++) {
    if (extractor.name[j] != NULL &&
        strcmp(type_bytes, extractor.name[j]) == 0) {
      return true;
    }
  }
  return false;
}

ExtractorTypeIndex getExtractorIndex(PyObject *type) {
  Py_ssize_t type_len = 0;
  const char *type_bytes = PyUnicode_AsUTF8AndSize(type, &type_len);
  if (!type_bytes) {
    return EXTRACTOR_UNKOWN_INDEX;
  }
  if (type_len < 2) {
    PyErr_Format(PyExc_RuntimeError, "Unexpected data type: %R", type);
    return EXTRACTOR_UNKOWN_INDEX;
  }
  // Examples: '*fp32', 'fp32', 'i8', etc.
  if (type_bytes[0] == '*') {
    return EXTRACTOR_POINTER_INDEX;
  }
  for (ExtractorTypeIndex i = EXTRACTOR_INT8_INDEX; i < EXTRACTOR_TYPE_COUNT;
       ++i) {
    if (isMatch(type_bytes, i)) {
      return i;
    }
  }

  PyErr_Format(PyExc_RuntimeError, "Unknown data type: %R", type);
  return EXTRACTOR_UNKOWN_INDEX;
}

// Takes in a list of types (ex: ['*fp32', 'u8', 'nvTmaDesc']) and returns
// a bytes array that represent extractors for quick argument extraction
// when launching.
extern "C" EXPORT_FUNC PyObject *build_signature_metadata(PyObject *args) {
  PyObject *signature = NULL;
  if (!PyArg_ParseTuple(args, "O", &signature)) {
    return NULL;
  }
  PyObject *fast_signature = PySequence_Fast(
      signature, "Expected kernel_arg_types to be a sequence or iterable");
  if (!fast_signature) {
    return NULL;
  }
  Py_ssize_t signature_size = PySequence_Fast_GET_SIZE(fast_signature);
  PyObject **signature_items = PySequence_Fast_ITEMS(fast_signature);

  // Create return bytes object.
  PyObject *ret_bytes = PyBytes_FromStringAndSize(NULL, signature_size);
  if (ret_bytes == NULL) {
    Py_XDECREF(fast_signature);
    return NULL;
  }
  char *buffer = PyBytes_AS_STRING(ret_bytes);
  for (Py_ssize_t i = 0; i < signature_size; ++i) {
    ExtractorTypeIndex extractor_idx = getExtractorIndex(signature_items[i]);
    if (extractor_idx == EXTRACTOR_UNKOWN_INDEX) {
      Py_XDECREF(fast_signature);
      Py_XDECREF(ret_bytes);
      return NULL;
    }
    buffer[i] = (uint8_t)extractor_idx;
  }

  Py_XDECREF(fast_signature);
  return ret_bytes;
}

bool extractArgs(PyObject **final_list, int *list_idx, PyObject *kernel_args,
                 PyObject *arg_annotations) {
  // Extract arg annotations
  PyObject *fast_annotations = PySequence_Fast(
      arg_annotations, "Expected arg_annotations to be a sequence or iterable");
  if (!fast_annotations) {
    Py_XDECREF(fast_annotations);
    return false;
  }
  Py_ssize_t num_annotations = PySequence_Fast_GET_SIZE(fast_annotations);
  PyObject **annotations = PySequence_Fast_ITEMS(fast_annotations);

  PyObject *fast_args = PySequence_Fast(
      kernel_args, "Expected kernel_args to be a sequence or iterable");
  if (!fast_args) {
    Py_XDECREF(fast_annotations);
    Py_XDECREF(fast_args);
    return false;
  }
  PyObject **args = PySequence_Fast_ITEMS(fast_args);

  int arg_idx = 0;
  for (int i = 0; i < num_annotations; ++i) {
    PyKernelArgObject *annotation = (PyKernelArgObject *)annotations[i];
    switch (annotation->type) {
    case ARG_KERNEL:
      final_list[(*list_idx)++] = args[arg_idx++];
      break;
    case ARG_TUPLE:
      if (!extractArgs(final_list, list_idx, args[arg_idx++],
                       annotation->nested_tuple)) {
        Py_XDECREF(fast_annotations);
        Py_XDECREF(fast_args);
        return false;
      }
      break;
    case ARG_CONSTEXPR:
      arg_idx++;
      break;
    }
  }
  Py_DECREF(fast_annotations);
  Py_DECREF(fast_args);
  return true;
}

bool launchHook(PyObject *hook, PyObject *metadata) {
  if (hook != Py_None) {
    PyObject *ret = PyObject_CallOneArg(hook, metadata);
    if (!ret) {
      return false;
    }
    Py_DECREF(ret);
  }
  return true;
}

extern "C" EXPORT_FUNC PyObject *launch(PyObject *args) {
  int gridX, gridY, gridZ;
  PyObject *py_obj_stream;
  PyObject *py_kernel;
  PyObject *kernel_metadata = NULL;
  PyObject *launch_metadata = NULL;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  void *global_scratch = nullptr;
  void *profile_scratch = nullptr;
  PyObject *arg_annotations = NULL;
  Py_buffer signature;
  PyObject *kernel_args = NULL;

  if (!PyArg_ParseTuple(args, "iiiOOOOOOOy*O", &gridX, &gridY, &gridZ,
                        &py_obj_stream, &py_kernel, &kernel_metadata,
                        &launch_metadata, &launch_enter_hook, &launch_exit_hook,
                        &arg_annotations, &signature, &kernel_args)) {
    return NULL;
  }

  // extract kernel metadata
  PyObject *num_warps_attr =
      PyObject_GetAttrString(kernel_metadata, "num_warps");
  int num_warps = PyLong_AsLong(num_warps_attr);
  Py_DECREF(num_warps_attr);
  PyObject *num_ctas_attr = PyObject_GetAttrString(kernel_metadata, "num_ctas");
  int num_ctas = PyLong_AsLong(num_ctas_attr);
  Py_DECREF(num_ctas_attr);
  PyObject *shared_attr = PyObject_GetAttrString(kernel_metadata, "shared");
  int shared_memory = PyLong_AsLong(shared_attr);
  Py_DECREF(shared_attr);
  PyObject *threads_per_warp_attr =
      PyObject_GetAttrString(kernel_metadata, "threads_per_warp");
  int threads_per_warp = PyLong_AsLong(threads_per_warp_attr);
  Py_DECREF(threads_per_warp_attr);

  // launch entry hook.
  if (!launchHook(launch_enter_hook, launch_metadata)) {
    PyBuffer_Release(&signature);
    return NULL;
  }

  uint8_t *extractor_data = (uint8_t *)signature.buf;
  Py_ssize_t num_args = signature.len;

  void *pStream = PyLong_AsVoidPtr(py_obj_stream);
  // error check
  if (pStream == nullptr || py_kernel == nullptr) {
    PyBuffer_Release(&signature);
    return NULL;
  }

  sycl::queue stream = *(static_cast<sycl::queue *>(pStream));

  // Extract kernel parameters - flatten tuples & remove constexpr.
  PyObject **args_data = (PyObject **)alloca(num_args * sizeof(PyObject *));
  if (args_data == NULL) {
    PyBuffer_Release(&signature);
    return NULL;
  }
  int list_idx = 0;
  if (!extractArgs(args_data, &list_idx, kernel_args, arg_annotations)) {
    PyBuffer_Release(&signature);
    return NULL;
  }

  // Number of parameters passed to kernel. + 2 for global & profile scratch.
  int num_params = num_args + 2;
  void **params = (void **)alloca(num_params * sizeof(void *));
  int params_idx = 0;
  PointerCheckScope pointerCheckScope(stream);
  // This loop has to stay in the same function that owns params, since we are
  // using alloca to allocate pointers to it on the stack of the function.
  for (Py_ssize_t i = 0; i < num_args; ++i) {
    g_pointer_check_arg_idx = static_cast<int>(i);
    // Get extractor that will send back a struct with
    // * size for allocation
    // * function to call to put the parameter in params buffer
    Extractor extractor = getExtractor(extractor_data[i]);
    if (extractor.extract == NULL) {
      PyBuffer_Release(&signature);
      return NULL;
    }

    size_t alignment = extractor.alignment;
    if (alignment != 0) {
      // Allocate enough space on the stack to guarantee an aligned block.
      size_t size_with_alignment = extractor.size + alignment - 1;
      void *storage_ptr = alloca(size_with_alignment);
      void *aligned_ptr = (void *)((((uintptr_t)storage_ptr) + alignment - 1) &
                                   ~(alignment - 1));
      if (aligned_ptr == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to align parameter storage");
        PyBuffer_Release(&signature);
        return NULL;
      }
      params[params_idx] = aligned_ptr;
    } else {
      params[params_idx] = alloca(extractor.size);
    }

    PyObject *current_arg = args_data[i];
    if (!extractor.extract(params[params_idx++], current_arg)) {
      PyBuffer_Release(&signature);
      return NULL;
    }
  }
  g_pointer_check_arg_idx = -1;
  // Add scratch objects.
  params[params_idx++] = &global_scratch;
  params[params_idx++] = &profile_scratch;
  sycl::kernel *kernel_ptr = reinterpret_cast<sycl::kernel *>(
      PyCapsule_GetPointer(py_kernel, "kernel"));
  if (kernel_ptr == nullptr)
    return NULL;
  sycl::kernel kernel = *kernel_ptr;

  Py_BEGIN_ALLOW_THREADS;
  sycl_kernel_launch(gridX, gridY, gridZ, num_warps, threads_per_warp,
                     shared_memory, stream, kernel, global_scratch,
                     profile_scratch, num_params, params, extractor_data);
  Py_END_ALLOW_THREADS;

  if (PyErr_Occurred()) {
    PyBuffer_Release(&signature);
    return NULL;
  }

  if (!launchHook(launch_exit_hook, launch_metadata)) {
    PyBuffer_Release(&signature);
    return NULL;
  }
  PyBuffer_Release(&signature);
  Py_RETURN_NONE;
}

extern "C" EXPORT_FUNC PyTypeObject *init_PyKernelArgType() {
  if (PyType_Ready(&PyKernelArgType) < 0)
    return NULL;

  Py_INCREF(&PyKernelArgType);
  return &PyKernelArgType;
}
