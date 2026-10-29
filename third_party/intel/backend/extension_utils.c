/*
 * Lightweight utility for checking device extensions without full driver
 * initialization. This module provides extension checking capabilities without
 * requiring a PyTorch sycl queue.
 */

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <sycl/sycl.hpp>
#include <utility>
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

// Index of `device_id` in `device_ids`, or -1 if absent.
static int findDeviceIndex(const std::vector<int> &device_ids, int device_id) {
  auto it = std::find(device_ids.begin(), device_ids.end(), device_id);
  return it == device_ids.end() ? -1
                                : static_cast<int>(it - device_ids.begin());
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

// The device/device_id lists to query, matching whichever backend
// `initializeDevicesIfNeeded()` found (OpenCL preferred, else Level Zero).
static std::pair<const std::vector<sycl::device> *, const std::vector<int> *>
activeDeviceLists() {
  if (has_opencl)
    return {&g_opencl_devices, &g_opencl_device_ids};
  return {&g_levelzero_devices, &g_levelzero_device_ids};
}

extern "C" EXPORT_FUNC PyObject *check_extension(int device_id,
                                                 const char *extension) {
  try {
    initializeDevicesIfNeeded();

    auto [devices, device_ids] = activeDeviceLists();
    int device_idx = findDeviceIndex(*device_ids, device_id);
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
    // Surface init failures (e.g. no OpenCL and no `ocloc`) instead of
    // reporting every extension as unsupported.
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return NULL;
  }
}

extern "C" EXPORT_FUNC PyObject *get_device_id(int device_idx) {
  try {
    initializeDevicesIfNeeded();

    const std::vector<int> *device_ids = activeDeviceLists().second;
    size_t device_count = device_ids->size();

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
