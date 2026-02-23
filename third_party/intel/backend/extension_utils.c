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

// Cached device list and their Intel device IDs - initialized on first use
static std::vector<sycl::device> g_devices;
static std::vector<int> g_device_ids;
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

  // Get all GPU devices from all platforms and extract their Intel device IDs
  auto platforms = sycl::platform::get_platforms();
  for (const auto &platform : platforms) {
    auto devices = platform.get_devices(sycl::info::device_type::gpu);
    for (const auto &device : devices) {
      g_devices.push_back(device);
      // Get the Intel device ID for this device
      try {
        auto device_id =
            device.get_info<sycl::ext::intel::info::device::device_id>();
        g_device_ids.push_back(static_cast<int>(device_id));
      } catch (...) {
        // If Intel device_id is not available, use vendor_id as fallback
        auto vendor_id = device.get_info<sycl::info::device::vendor_id>();
        g_device_ids.push_back(static_cast<int>(vendor_id));
      }
    }
  }

  g_devices_initialized = true;
}

extern "C" EXPORT_FUNC PyObject *check_extension(int device_id,
                                                 const char *extension) {
  try {
    initializeDevicesIfNeeded();

    if (g_devices.empty()) {
      PyErr_SetString(PyExc_RuntimeError, "No GPU devices available");
      return NULL;
    }

    // Find the device matching the provided device_id
    const sycl::device *target_device = nullptr;
    for (size_t i = 0; i < g_devices.size(); ++i) {
      if (g_device_ids[i] == device_id) {
        target_device = &g_devices[i];
        break;
      }
    }

    if (target_device == nullptr) {
      PyErr_Format(PyExc_RuntimeError, "No device found with device_id: %d",
                   device_id);
      return NULL;
    }

    // If OpenCL backend is available, use it
    if (g_has_opencl && target_device->get_backend() == sycl::backend::opencl) {
      if (sycl::opencl::has_extension(*target_device, extension)) {
        Py_RETURN_TRUE;
      }
      Py_RETURN_FALSE;
    }

    // Otherwise use Level Zero extension query
    if (target_device->ext_oneapi_supports_cl_extension(extension)) {
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

extern "C" EXPORT_FUNC PyObject *get_device_id(int device_idx) {
  try {
    initializeDevicesIfNeeded();

    if (g_devices.empty()) {
      PyErr_SetString(PyExc_RuntimeError, "No GPU devices available");
      return NULL;
    }

    // Validate device index is within range
    if (device_idx < 0 || device_idx >= static_cast<int>(g_devices.size())) {
      PyErr_Format(PyExc_RuntimeError,
                   "Invalid device index: %d (must be in range [0, %zu))",
                   device_idx, g_devices.size());
      return NULL;
    }

    // Return the device_id for the device at the given index
    return PyLong_FromLong(g_device_ids[device_idx]);

  } catch (const std::exception &e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return NULL;
  }
}
