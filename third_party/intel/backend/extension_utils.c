/*
 * Lightweight utility for checking device extensions without full driver
 * initialization. This module provides extension checking capabilities without
 * requiring a PyTorch sycl queue.
 */

#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>

#if defined(_WIN32)
#define EXPORT_FUNC __declspec(dllexport)
#else
#define EXPORT_FUNC __attribute__((visibility("default")))
#endif

#include <Python.h>

// Cached device list - initialized on first use
static std::vector<sycl::device> g_devices;
static bool g_devices_initialized = false;
static bool g_has_opencl = false;

static void initializeDevicesIfNeeded() {
  if (g_devices_initialized) {
    return;
  }

  // Check if OpenCL backend is available
  for (const auto &platform : sycl::platform::get_platforms()) {
    if (platform.get_backend() == sycl::backend::opencl) {
      g_has_opencl = true;
      break;
    }
  }

  // Get all GPU devices from all platforms
  auto platforms = sycl::platform::get_platforms();
  for (const auto &platform : platforms) {
    auto devices = platform.get_devices(sycl::info::device_type::gpu);
    g_devices.insert(g_devices.end(), devices.begin(), devices.end());
  }

  g_devices_initialized = true;
}

extern "C" EXPORT_FUNC PyObject *check_extension(int device_id,
                                                 const char *extension) {
  try {
    initializeDevicesIfNeeded();

    if (device_id >= g_devices.size()) {
      PyErr_Format(PyExc_RuntimeError,
                   "Device %d not found (only %zu devices available)",
                   device_id, g_devices.size());
      return NULL;
    }

    const sycl::device &device = g_devices[device_id];

    // If OpenCL backend is available, use it
    if (g_has_opencl && device.get_backend() == sycl::backend::opencl) {
      if (sycl::opencl::has_extension(device, extension)) {
        Py_RETURN_TRUE;
      }
      Py_RETURN_FALSE;
    }

    // Otherwise use Level Zero extension query
    if (device.ext_oneapi_supports_cl_extension(extension)) {
      Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;

  } catch (const std::exception &e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return NULL;
  }
}

extern "C" EXPORT_FUNC PyObject *get_device_count() {
  try {
    initializeDevicesIfNeeded();
    return PyLong_FromSize_t(g_devices.size());
  } catch (const std::exception &e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return NULL;
  }
}
