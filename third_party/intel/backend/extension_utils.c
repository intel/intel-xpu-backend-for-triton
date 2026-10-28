/*
 * Lightweight utility for checking device extensions without full driver
 * initialization. This module provides extension checking capabilities without
 * requiring a PyTorch sycl queue.
 */

#include <filesystem>
#include <iostream>
#include <sstream>
#include <sycl/sycl.hpp>
#include <vector>

#if defined(_WIN32)
#define EXPORT_FUNC __declspec(dllexport)
#else
#define EXPORT_FUNC __attribute__((visibility("default")))
#endif

#include <Python.h>

#include "sycl_functions.h"

// Cached device lists separated by platform/backend and their Intel device IDs
// OpenCL devices
static std::vector<sycl::device> g_opencl_devices;
static std::vector<int> g_opencl_device_ids;

// Level Zero devices
static std::vector<sycl::device> g_levelzero_devices;
static std::vector<int> g_levelzero_device_ids;

static bool g_devices_initialized = false;
static bool has_opencl = false;

// Raw Level Zero state for `probe_spirv_module()`, enumerated independently
// of `g_levelzero_devices` above and *without* going through SYCL: plain
// `zeInit`/`zeDriverGet`/`zeDeviceGet`, not `sycl::platform::get_platforms()`.
// Probing a capability needs no SYCL device/context/queue, only a L0 device
// and a L0 context to pass to `zeModuleCreate` (via `create_module()`,
// itself pure L0), so it should not pay for SYCL's backend/plugin init, nor
// be gated by `initializeDevicesIfNeeded()`'s requirement that OpenCL or
// `ocloc` be present (that requirement exists only for `check_extension()`'s
// query path).
static std::vector<ze_device_handle_t> g_l0_probe_devices;
static std::vector<ze_context_handle_t>
    g_l0_probe_contexts; // one per device's driver
static std::vector<int> g_l0_probe_device_ids;
static bool g_l0_probe_devices_initialized = false;

static void initializeLevelZeroDevicesForProbingIfNeeded() {
  if (g_l0_probe_devices_initialized)
    return;
  g_l0_probe_devices_initialized = true;

  // Matches the flag `driver.c`'s `init_devices()` and `proton_utils.cpp`
  // already call `zeInit` with; a redundant call here (before or after
  // either of those) is explicitly documented as safe -- see
  // https://github.com/oneapi-src/level-zero/ -- `zeInit` is multi-call- and
  // multi-library-safe, and only ever initializes each driver once per
  // process.
  if (zeInit(ZE_INIT_FLAG_GPU_ONLY) != ZE_RESULT_SUCCESS)
    return;

  uint32_t driverCount = 0;
  if (zeDriverGet(&driverCount, nullptr) != ZE_RESULT_SUCCESS ||
      driverCount == 0)
    return;
  std::vector<ze_driver_handle_t> drivers(driverCount);
  if (zeDriverGet(&driverCount, drivers.data()) != ZE_RESULT_SUCCESS)
    return;

  for (ze_driver_handle_t driver : drivers) {
    uint32_t deviceCount = 0;
    if (zeDeviceGet(driver, &deviceCount, nullptr) != ZE_RESULT_SUCCESS ||
        deviceCount == 0)
      continue;
    std::vector<ze_device_handle_t> devices(deviceCount);
    if (zeDeviceGet(driver, &deviceCount, devices.data()) != ZE_RESULT_SUCCESS)
      continue;

    // One context per driver, shared by every device behind it; never
    // destroyed, same process-lifetime convention as the device/context
    // caches above and in driver.c.
    ze_context_desc_t context_desc = {};
    context_desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
    ze_context_handle_t context = nullptr;
    if (zeContextCreate(driver, &context_desc, &context) != ZE_RESULT_SUCCESS)
      continue;

    for (ze_device_handle_t device : devices) {
      ze_device_properties_t props = {};
      props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
      if (zeDeviceGetProperties(device, &props) != ZE_RESULT_SUCCESS)
        continue;
      if (props.type != ZE_DEVICE_TYPE_GPU)
        continue;

      g_l0_probe_devices.push_back(device);
      g_l0_probe_contexts.push_back(context);
      // Matches the device_id SYCL callers already have, from
      // sycl::ext::intel::info::device::device_id -- the Level Zero PCI
      // device ID surfaced directly here, with no SYCL layer in between.
      g_l0_probe_device_ids.push_back(static_cast<int>(props.deviceId));
    }
  }
}

bool has_ocloc_in_path() {
  const char *path_env = std::getenv("PATH");
  if (!path_env)
    return false;

#ifdef _WIN32
  const char delimiter = ';';
  const std::string exe = "ocloc.exe";
#else
  const char delimiter = ':';
  const std::string exe = "ocloc";
#endif

  std::stringstream ss(path_env);
  std::string dir;

  while (std::getline(ss, dir, delimiter)) {
    std::filesystem::path p = std::filesystem::path(dir) / exe;
    if (std::filesystem::exists(p))
      return true;
  }
  return false;
}

static void initializeDevicesIfNeeded() {
  if (g_devices_initialized) {
    return;
  }

  // Check if OpenCL backend is available
  auto check_platforms = sycl::platform::get_platforms();
  for (const auto &platform : check_platforms) {
    auto backend = platform.get_backend();
    if (backend == sycl::backend::opencl) {
      has_opencl = true;
      break;
    }
  }

  if (!has_opencl && !has_ocloc_in_path()) {
    throw std::runtime_error(
        "OpenCL backend not found and ocloc is not in PATH. At least one of "
        "these is required for extension checking.");
  }

  // Get all GPU devices and organize them by platform/backend
  // Only consider OpenCL and Level Zero (ext_oneapi) platforms
  for (const auto &platform : sycl::platform::get_platforms()) {
    auto backend = platform.get_backend();
    // Include OpenCL backend and ext_oneapi (Level Zero) backend
    if (backend != sycl::backend::opencl &&
        backend != sycl::backend::ext_oneapi_level_zero) {
      continue;
    }
    auto devices = platform.get_devices(sycl::info::device_type::gpu);
    for (const auto &device : devices) {
      int device_id =
          device.get_info<sycl::ext::intel::info::device::device_id>();
      // Organize devices by backend
      if (device.get_backend() == sycl::backend::opencl) {
        g_opencl_devices.push_back(device);
        g_opencl_device_ids.push_back(device_id);
      } else {
        // Level Zero or other backends
        g_levelzero_devices.push_back(device);
        g_levelzero_device_ids.push_back(device_id);
      }
    }
  }

  g_devices_initialized = true;
}

extern "C" EXPORT_FUNC PyObject *check_extension(int device_id,
                                                 const char *extension) {
  try {
    initializeDevicesIfNeeded();

    // Choose the appropriate device list based on backend availability
    const std::vector<sycl::device> *devices = nullptr;
    const std::vector<int> *device_ids = nullptr;

    if (has_opencl) {
      // Use OpenCL devices if available
      devices = &g_opencl_devices;
      device_ids = &g_opencl_device_ids;
    } else {
      // Fall back to Level Zero devices
      devices = &g_levelzero_devices;
      device_ids = &g_levelzero_device_ids;
    }

    if (devices->empty()) {
      Py_RETURN_NONE;
    }

    // Find the device matching the provided device_id
    int device_idx = -1;
    for (size_t i = 0; i < devices->size(); ++i) {
      if ((*device_ids)[i] == device_id) {
        device_idx = i;
        break;
      }
    }

    if (device_idx == -1) {
      Py_RETURN_NONE;
    }

    const sycl::device target_device = (*devices)[device_idx];

    // Try OpenCL backend if available
    if (has_opencl) {
      if (sycl::opencl::has_extension(target_device, extension)) {
        Py_RETURN_TRUE;
      }
      Py_RETURN_FALSE;
    }

    // Otherwise use Level Zero extension query
    // `ocloc` should be in `PATH` for proper work
    if (target_device.ext_oneapi_supports_cl_extension(extension)) {
      Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;

  } catch (const std::exception &e) {
    Py_RETURN_FALSE;
  }
}

extern "C" EXPORT_FUNC PyObject *get_device_id(int device_idx) {
  try {
    initializeDevicesIfNeeded();

    // Choose the appropriate device list based on backend availability
    const std::vector<int> *device_ids = nullptr;
    size_t device_count = 0;

    if (has_opencl) {
      device_ids = &g_opencl_device_ids;
      device_count = g_opencl_devices.size();
    } else {
      device_ids = &g_levelzero_device_ids;
      device_count = g_levelzero_devices.size();
    }

    if (device_count == 0) {
      PyErr_SetString(PyExc_RuntimeError, "No GPU devices available");
      return NULL;
    }

    // Validate device index is within range
    if (device_idx < 0 || device_idx >= static_cast<int>(device_count)) {
      PyErr_Format(PyExc_RuntimeError,
                   "Invalid device index: %d (must be in range [0, %zu))",
                   device_idx, device_count);
      return NULL;
    }

    // Return the device_id for the device at the given index
    return PyLong_FromLong((*device_ids)[device_idx]);

  } catch (const std::exception &e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return NULL;
  }
}

// Ground-truth probe for a SPIR-V capability/extension: hands `spirv` (the
// smallest module that only declares the capability under test) to the
// driver's own JIT via `zeModuleCreate`, the same call every real kernel
// goes through. Returns True/False once a matching Level Zero device is
// found, None if it isn't (mirrors `check_extension()`'s convention).
extern "C" EXPORT_FUNC PyObject *probe_spirv_module(int device_id,
                                                    const unsigned char *spirv,
                                                    size_t spirv_size) {
  try {
    initializeLevelZeroDevicesForProbingIfNeeded();

    int device_idx = -1;
    for (size_t i = 0; i < g_l0_probe_device_ids.size(); ++i) {
      if (g_l0_probe_device_ids[i] == device_id) {
        device_idx = static_cast<int>(i);
        break;
      }
    }
    if (device_idx == -1) {
      Py_RETURN_NONE;
    }

    auto [module, error_no] = create_module(
        g_l0_probe_contexts[device_idx], g_l0_probe_devices[device_idx],
        const_cast<unsigned char *>(spirv), spirv_size, "");
    if (error_no == ZE_RESULT_SUCCESS) {
      zeModuleDestroy(module);
      Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;

  } catch (const std::exception &e) {
    Py_RETURN_NONE;
  }
}
