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
#include <utility>
#include <vector>

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

// ==========================================================================
// Generic launcher — matches the CUDA/AMD design:
//   - buildSignatureMetadata: called ONCE at kernel init; maps type strings
//     to extractor indices (a bytes object cached in the Python launcher).
//   - generic_launch: called on EVERY launch; receives raw Python arg objects
//     and the cached bytes blob; uses alloca (no heap) and a function-pointer
//     table (no switch) to extract and set each argument.
// ==========================================================================

// --------------------------------------------------------------------------
// Extractor type enum and function-pointer table
// --------------------------------------------------------------------------

typedef bool (*ExtractorFunc)(void *dst, PyObject *obj);
typedef void (*SetArgFunc)(sycl::handler &cgh, int idx, const void *val);

typedef struct {
  ExtractorFunc extract; // write the C value from a Python object into dst
  SetArgFunc setArg;     // call cgh.set_arg with the correctly-typed value
  size_t size;           // sizeof the C type
  const char *names[3];  // Triton type strings that map to this extractor
} Extractor;

// Extractor indices — stored in the pre-built signature metadata bytes.
typedef enum {
  EX_UNKNOWN = 0,
  EX_PTR,  // any '*...' pointer type
  EX_I8,   // i8
  EX_I16,  // i16
  EX_I32,  // i1, i32
  EX_I64,  // i64
  EX_U8,   // u8
  EX_U16,  // u16
  EX_U32,  // u1, u32
  EX_U64,  // u64
  EX_FP16, // fp16
  EX_BF16, // bf16
  EX_FP32, // fp32, f32
  EX_FP64, // fp64
  EX_COUNT
} ExtractorIndex;

// --- individual extractor functions ---

static bool extractPtr(void *dst, PyObject *obj) {
  void **out = static_cast<void **>(dst);
  if (obj == Py_None) {
    *out = nullptr;
    return true;
  }
  if (PyLong_Check(obj)) {
    *out = PyLong_AsVoidPtr(obj);
    return !PyErr_Occurred();
  }
  // obj has a .data_ptr() method (e.g. torch.Tensor)
  static PyObject *s_data_ptr = PyUnicode_InternFromString("data_ptr");
  PyObject *ret = PyObject_CallMethodNoArgs(obj, s_data_ptr);
  if (!ret)
    return false;
  *out = PyLong_AsVoidPtr(ret);
  Py_DECREF(ret);
  return !PyErr_Occurred();
}

static bool extractI8(void *dst, PyObject *obj) {
  *static_cast<int8_t *>(dst) = (int8_t)PyLong_AsLong(obj);
  return !PyErr_Occurred();
}
static bool extractI16(void *dst, PyObject *obj) {
  *static_cast<int16_t *>(dst) = (int16_t)PyLong_AsLong(obj);
  return !PyErr_Occurred();
}
static bool extractI32(void *dst, PyObject *obj) {
  *static_cast<int32_t *>(dst) = (int32_t)PyLong_AsLong(obj);
  return !PyErr_Occurred();
}
static bool extractI64(void *dst, PyObject *obj) {
  *static_cast<int64_t *>(dst) = PyLong_AsLongLong(obj);
  return !PyErr_Occurred();
}
static bool extractU8(void *dst, PyObject *obj) {
  *static_cast<uint8_t *>(dst) = (uint8_t)PyLong_AsUnsignedLong(obj);
  return !PyErr_Occurred();
}
static bool extractU16(void *dst, PyObject *obj) {
  *static_cast<uint16_t *>(dst) = (uint16_t)PyLong_AsUnsignedLong(obj);
  return !PyErr_Occurred();
}
static bool extractU32(void *dst, PyObject *obj) {
  *static_cast<uint32_t *>(dst) = (uint32_t)PyLong_AsUnsignedLong(obj);
  return !PyErr_Occurred();
}
static bool extractU64(void *dst, PyObject *obj) {
  *static_cast<uint64_t *>(dst) = PyLong_AsUnsignedLongLong(obj);
  return !PyErr_Occurred();
}
static bool extractFP16(void *dst, PyObject *obj) {
  double d = PyFloat_AsDouble(obj);
  if (PyErr_Occurred())
    return false;
  uint16_t v;
  PyFloat_Pack2(d, reinterpret_cast<char *>(&v), 1);
  *static_cast<uint16_t *>(dst) = v;
  return !PyErr_Occurred();
}
static bool extractBF16(void *dst, PyObject *obj) {
  float f = (float)PyFloat_AsDouble(obj);
  if (PyErr_Occurred())
    return false;
  uint32_t u;
  memcpy(&u, &f, sizeof(u));
  *static_cast<uint16_t *>(dst) = (uint16_t)(u >> 16);
  return true;
}
static bool extractFP32(void *dst, PyObject *obj) {
  float f = (float)PyFloat_AsDouble(obj);
  if (PyErr_Occurred())
    return false;
  memcpy(dst, &f, sizeof(f));
  return true;
}
static bool extractFP64(void *dst, PyObject *obj) {
  double d = PyFloat_AsDouble(obj);
  if (PyErr_Occurred())
    return false;
  memcpy(dst, &d, sizeof(d));
  return true;
}

// The table — indexed by ExtractorIndex.

// setArg_* helpers: call cgh.set_arg with the correctly-typed value.
// SYCL's set_arg is a template, so we need one wrapper per type.
#define DEFINE_SET_ARG(name, T)                                                \
  static void setArg_##name(sycl::handler &cgh, int idx, const void *val) {    \
    cgh.set_arg(idx, *static_cast<const T *>(val));                            \
  }
// Pointer needs its own implementation: val points to a void* (the pointer
// value), and casting const void* -> const void** removes qualifiers.
// Use memcpy to load the stored pointer value safely.
static void setArg_ptr(sycl::handler &cgh, int idx, const void *val) {
  void *p;
  memcpy(&p, val, sizeof(p));
  cgh.set_arg(idx, p);
}
DEFINE_SET_ARG(i8, int8_t)
DEFINE_SET_ARG(i16, int16_t)
DEFINE_SET_ARG(i32, int32_t)
DEFINE_SET_ARG(i64, int64_t)
DEFINE_SET_ARG(u8, uint8_t)
DEFINE_SET_ARG(u16, uint16_t)
DEFINE_SET_ARG(u32, uint32_t)
DEFINE_SET_ARG(u64, uint64_t)
DEFINE_SET_ARG(fp16, uint16_t) // fp16/bf16 stored as raw bits in uint16_t
DEFINE_SET_ARG(bf16, uint16_t)
DEFINE_SET_ARG(fp32, float)
DEFINE_SET_ARG(fp64, double)
#undef DEFINE_SET_ARG

static const Extractor g_extractors[EX_COUNT] = {
    [EX_UNKNOWN] = {nullptr, nullptr, 0, {}},
    [EX_PTR] = {extractPtr, setArg_ptr, sizeof(void *), {}},
    [EX_I8] = {extractI8, setArg_i8, sizeof(int8_t), {"i8", nullptr, nullptr}},
    [EX_I16] = {extractI16,
                setArg_i16,
                sizeof(int16_t),
                {"i16", nullptr, nullptr}},
    [EX_I32] = {extractI32,
                setArg_i32,
                sizeof(int32_t),
                {"i1", "i32", nullptr}},
    [EX_I64] = {extractI64,
                setArg_i64,
                sizeof(int64_t),
                {"i64", nullptr, nullptr}},
    [EX_U8] = {extractU8, setArg_u8, sizeof(uint8_t), {"u8", nullptr, nullptr}},
    [EX_U16] = {extractU16,
                setArg_u16,
                sizeof(uint16_t),
                {"u16", nullptr, nullptr}},
    [EX_U32] = {extractU32,
                setArg_u32,
                sizeof(uint32_t),
                {"u1", "u32", nullptr}},
    [EX_U64] = {extractU64,
                setArg_u64,
                sizeof(uint64_t),
                {"u64", nullptr, nullptr}},
    [EX_FP16] = {extractFP16,
                 setArg_fp16,
                 sizeof(uint16_t),
                 {"fp16", nullptr, nullptr}},
    [EX_BF16] = {extractBF16,
                 setArg_bf16,
                 sizeof(uint16_t),
                 {"bf16", nullptr, nullptr}},
    [EX_FP32] = {extractFP32,
                 setArg_fp32,
                 sizeof(float),
                 {"fp32", "f32", nullptr}},
    [EX_FP64] = {extractFP64,
                 setArg_fp64,
                 sizeof(double),
                 {"fp64", nullptr, nullptr}},
};

static ExtractorIndex getExtractorIndex(const char *ty) {
  if (ty[0] == '*')
    return EX_PTR;
  for (int i = EX_I8; i < EX_COUNT; ++i) {
    for (int j = 0; j < 3 && g_extractors[i].names[j]; ++j) {
      if (strcmp(ty, g_extractors[i].names[j]) == 0)
        return (ExtractorIndex)i;
    }
  }
  return EX_UNKNOWN;
}

// --------------------------------------------------------------------------
// Pointer validation helper — shared between buildSignatureMetadata (which
// checks at launch time) and the inline check inside generic_launch.
// --------------------------------------------------------------------------

static bool validatePointer(void *ptr, int idx, ze_context_handle_t l0ctx) {
  if (!ptr)
    return true; // nullptr is always valid
  ze_memory_allocation_properties_t prop = {};
  prop.stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
  ze_device_handle_t dev;
  ze_result_t res = zeMemGetAllocProperties(l0ctx, ptr, &prop, &dev);
  if (res != ZE_RESULT_SUCCESS) {
    PyErr_Format(PyExc_ValueError,
                 "Cannot get memory properties for Pointer argument "
                 "(at %d, err=%d)",
                 idx, (int)res);
    return false;
  }
  if (prop.type == ZE_MEMORY_TYPE_UNKNOWN) {
    PyErr_Format(PyExc_ValueError,
                 "Pointer argument (at %d) doesn't reference "
                 "accessible memory.",
                 idx);
    return false;
  }
  return true;
}

// --------------------------------------------------------------------------
// buildSignatureMetadata(sig_list) -> bytes
//
// Called ONCE when a kernel is first compiled.  Converts a Python list of
// Triton type strings (e.g. ['*f32', 'i32', 'constexpr']) into a compact
// bytes object of ExtractorIndex values.  EX_UNKNOWN (0) is used for
// constexpr entries so generic_launch can skip them with a single check.
// --------------------------------------------------------------------------

extern "C" EXPORT_FUNC PyObject *buildSignatureMetadata(PyObject *args) {
  PyObject *sig_list;
  if (!PyArg_ParseTuple(args, "O", &sig_list))
    return NULL;

  PyObject *fast = PySequence_Fast(sig_list, "signature must be a sequence");
  if (!fast)
    return NULL;

  Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);
  PyObject *result = PyBytes_FromStringAndSize(nullptr, n);
  if (!result) {
    Py_DECREF(fast);
    return NULL;
  }
  char *buf = PyBytes_AS_STRING(result);

  PyObject **items = PySequence_Fast_ITEMS(fast);
  for (Py_ssize_t i = 0; i < n; ++i) {
    const char *ty = PyUnicode_AsUTF8(items[i]);
    if (!ty) {
      Py_DECREF(fast);
      Py_DECREF(result);
      return NULL;
    }
    ExtractorIndex idx =
        (strcmp(ty, "constexpr") == 0) ? EX_UNKNOWN : getExtractorIndex(ty);
    if (idx == EX_UNKNOWN && strcmp(ty, "constexpr") != 0) {
      PyErr_Format(PyExc_ValueError,
                   "buildSignatureMetadata: unknown Triton type '%s'", ty);
      Py_DECREF(fast);
      Py_DECREF(result);
      return NULL;
    }
    buf[i] = (char)(uint8_t)idx;
  }

  Py_DECREF(fast);
  return result;
}

// --------------------------------------------------------------------------
// generic_launch(gridX, gridY, gridZ, stream, kernel, kernel_metadata,
//                launch_metadata, enter_hook, exit_hook,
//                sig_metadata,   <- bytes from buildSignatureMetadata
//                kernel_args)    <- tuple of raw Python arg objects
//
// Hot path: no heap allocation, no Python-side packing loop.
// alloca() keeps everything on the C stack (like CUDA/AMD backends).
// --------------------------------------------------------------------------

extern "C" EXPORT_FUNC PyObject *generic_launch(PyObject *args) {
  int gridX, gridY, gridZ;
  PyObject *pyStream, *pyKernel;
  PyObject *kernelMeta, *launchMeta, *enterHook, *exitHook;
  Py_buffer sigBuf;
  PyObject *kernelArgs;

  if (!PyArg_ParseTuple(args, "iiiOOOOOOs*O", &gridX, &gridY, &gridZ, &pyStream,
                        &pyKernel, &kernelMeta, &launchMeta, &enterHook,
                        &exitHook, &sigBuf, &kernelArgs)) {
    return NULL;
  }

  // Extract kernel metadata.
  PyObject *numWarpsAttr = PyObject_GetAttrString(kernelMeta, "num_warps");
  if (!numWarpsAttr) {
    PyBuffer_Release(&sigBuf);
    return NULL;
  }
  int numWarps = (int)PyLong_AsLong(numWarpsAttr);
  Py_DECREF(numWarpsAttr);
  if (numWarps == -1 && PyErr_Occurred()) {
    PyBuffer_Release(&sigBuf);
    return NULL;
  }

  PyObject *sharedAttr = PyObject_GetAttrString(kernelMeta, "shared");
  if (!sharedAttr) {
    PyBuffer_Release(&sigBuf);
    return NULL;
  }
  int sharedMemory = (int)PyLong_AsLong(sharedAttr);
  Py_DECREF(sharedAttr);
  if (sharedMemory == -1 && PyErr_Occurred()) {
    PyBuffer_Release(&sigBuf);
    return NULL;
  }

  PyObject *tpwAttr = PyObject_GetAttrString(kernelMeta, "threads_per_warp");
  if (!tpwAttr) {
    PyBuffer_Release(&sigBuf);
    return NULL;
  }
  int threadsPerWarp = (int)PyLong_AsLong(tpwAttr);
  Py_DECREF(tpwAttr);
  if (threadsPerWarp == -1 && PyErr_Occurred()) {
    PyBuffer_Release(&sigBuf);
    return NULL;
  }

  // Launch enter hook.
  if (enterHook != Py_None) {
    PyObject *ret = PyObject_CallOneArg(enterHook, launchMeta);
    if (!ret) {
      PyBuffer_Release(&sigBuf);
      return NULL;
    }
    Py_DECREF(ret);
  }

  void *pStream = PyLong_AsVoidPtr(pyStream);
  if (!pStream || !pyKernel) {
    PyBuffer_Release(&sigBuf);
    return NULL;
  }
  sycl::queue stream = *static_cast<sycl::queue *>(pStream);
  sycl::kernel *kernelPtr = reinterpret_cast<sycl::kernel *>(
      PyCapsule_GetPointer(pyKernel, "kernel"));
  if (!kernelPtr) {
    PyBuffer_Release(&sigBuf);
    return NULL;
  }
  sycl::kernel kernel = *kernelPtr;

  // sig_metadata is a bytes object built by buildSignatureMetadata.
  // Each byte is an ExtractorIndex; EX_UNKNOWN means constexpr (skip).
  const uint8_t *sigData = static_cast<const uint8_t *>(sigBuf.buf);
  Py_ssize_t sigLen = sigBuf.len;

  // Flatten kernelArgs into a fast sequence.
  PyObject *fastArgs =
      PySequence_Fast(kernelArgs, "kernel_args must be a sequence");
  if (!fastArgs) {
    PyBuffer_Release(&sigBuf);
    return NULL;
  }
  Py_ssize_t nRawArgs = PySequence_Fast_GET_SIZE(fastArgs);
  PyObject **rawArgs = PySequence_Fast_ITEMS(fastArgs);

  // Count kernel (non-constexpr) args.
  int nKernelArgs = 0;
  for (Py_ssize_t i = 0; i < sigLen; ++i)
    if (sigData[i] != EX_UNKNOWN)
      ++nKernelArgs;

  // Stack-allocate an array of (storage_ptr, setArg_fn) pairs — one per
  // kernel arg.  Both alloca calls MUST stay in this function's stack frame.
  void **params = static_cast<void **>(alloca(nKernelArgs * sizeof(void *)));
  SetArgFunc *setArgFns =
      static_cast<SetArgFunc *>(alloca(nKernelArgs * sizeof(SetArgFunc)));

  // Pointer validation flag — read once, static.
  static bool validatePtrs = []() {
    const char *s = std::getenv("TRITON_XPU_VALIDATE_POINTERS");
    if (!s)
      return true;
    std::string v(s);
    std::transform(v.begin(), v.end(), v.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return !(v == "0" || v == "false" || v == "off");
  }();

  ze_context_handle_t l0ctx = nullptr;
  if (validatePtrs)
    l0ctx = static_cast<ze_context_handle_t>(
        sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
            stream.get_context()));

  // Extract each arg from the Python object using the pre-built metadata.
  int paramIdx = 0;
  Py_ssize_t rawIdx = 0;
  for (Py_ssize_t i = 0; i < sigLen; ++i) {
    uint8_t exIdx = sigData[i];
    if (exIdx == EX_UNKNOWN) { // constexpr — skip
      ++rawIdx;
      continue;
    }
    if (rawIdx >= nRawArgs) {
      PyErr_SetString(PyExc_ValueError,
                      "generic_launch: fewer kernel args than signature slots");
      Py_DECREF(fastArgs);
      PyBuffer_Release(&sigBuf);
      return NULL;
    }
    const Extractor &ex = g_extractors[exIdx];
    void *storage = alloca(ex.size);
    PyObject *obj = rawArgs[rawIdx++];
    if (!ex.extract(storage, obj)) {
      Py_DECREF(fastArgs);
      PyBuffer_Release(&sigBuf);
      return NULL;
    }
    // Validate pointers against the Level Zero memory model.
    if (validatePtrs && exIdx == EX_PTR) {
      void *ptr = *static_cast<void **>(storage);
      if (!validatePointer(ptr, paramIdx, l0ctx)) {
        Py_DECREF(fastArgs);
        PyBuffer_Release(&sigBuf);
        return NULL;
      }
    }
    params[paramIdx] = storage;
    setArgFns[paramIdx] = ex.setArg;
    ++paramIdx;
  }

  Py_DECREF(fastArgs);
  PyBuffer_Release(&sigBuf);

  sycl::range<3> globalRange(gridZ, gridY,
                             (size_t)gridX * threadsPerWarp * numWarps);
  sycl::range<3> localRange(1, 1, (size_t)numWarps * threadsPerWarp);
  sycl::nd_range<3> ndRange(globalRange, localRange);

  // Capture nKernelArgs and sharedMemory by value so the lambda is safe
  // after this function returns (submit is asynchronous w.r.t. the host,
  // but the command-group function itself is called synchronously before
  // submit() returns — so stack-allocated params[] are safe here).
  auto cgf = [&](sycl::handler &cgh) {
    for (int i = 0; i < nKernelArgs; ++i)
      setArgFns[i](cgh, i, params[i]);

    // Implicit params: global_scratch, profile_scratch (nullptr).
    void *nullPtr = nullptr;
    cgh.set_arg(nKernelArgs, nullPtr);
    cgh.set_arg(nKernelArgs + 1, nullPtr);

    if (sharedMemory) {
      auto localBuf = sycl::local_accessor<int8_t, 1>(sharedMemory, cgh);
      cgh.set_arg(nKernelArgs + 2, localBuf);
    }
    cgh.parallel_for(ndRange, kernel);
  };

  try {
    stream.submit(cgf);
  } catch (const sycl::exception &e) {
    PyErr_Format(PyExc_RuntimeError, "generic_launch: SYCL submit failed: %s",
                 e.what());
    return NULL;
  }

  // Launch exit hook.
  if (exitHook != Py_None) {
    PyObject *ret = PyObject_CallOneArg(exitHook, launchMeta);
    if (!ret)
      return NULL;
    Py_DECREF(ret);
  }

  Py_RETURN_NONE;
}
