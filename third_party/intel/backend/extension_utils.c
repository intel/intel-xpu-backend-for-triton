/*
 * Lightweight utility for checking device extensions without full driver
 * initialization. This module provides extension checking capabilities without
 * requiring a PyTorch sycl queue.
 */

#include <array>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <sycl/sycl.hpp>
#include <utility>
#include <vector>

#include <level_zero/ze_api.h>

#if defined(_WIN32)
#define EXPORT_FUNC __declspec(dllexport)
#include <windows.h>
#else
#define EXPORT_FUNC __attribute__((visibility("default")))
#include <dlfcn.h>
#endif

#include <Python.h>

// Cached Level Zero devices and their Intel device IDs.
static std::vector<sycl::device> g_devices;
static std::vector<int> g_device_ids;

struct DeviceCompilerExtensions {
  bool apiSupported = false;
  std::vector<std::string> extensions;
};

// Indexed like `g_devices`/`g_device_ids`; `std::nullopt` until the first
// query for that device.
static std::vector<std::optional<DeviceCompilerExtensions>> g_extensionsCache;

static bool g_devices_initialized = false;

static void initializeDevicesIfNeeded() {
  if (g_devices_initialized) {
    return;
  }

  for (const auto &platform : sycl::platform::get_platforms()) {
    if (platform.get_backend() != sycl::backend::ext_oneapi_level_zero) {
      continue;
    }
    for (const auto &device :
         platform.get_devices(sycl::info::device_type::gpu)) {
      g_devices.push_back(device);
      g_device_ids.push_back(
          device.get_info<sycl::ext::intel::info::device::device_id>());
    }
  }
  g_extensionsCache.resize(g_devices.size());

  g_devices_initialized = true;
}

// Returns the index into `g_devices`/`g_device_ids` for the given
// `device_id`, or -1 if no enumerated device matches it.
static int findDeviceIndex(int device_id) {
  for (size_t i = 0; i < g_device_ids.size(); ++i) {
    if (g_device_ids[i] == device_id)
      return static_cast<int>(i);
  }
  return -1;
}

// `zeDeviceGetCompilerInfo` and its `ze_device_compiler_info_t` enum postdate
// some Level Zero header packages still used for LTS driver installs (this
// module is JIT-compiled against whatever `<level_zero/ze_api.h>` the host
// has, so an older header may not declare either at all -- this is not just
// a question of whether the symbol links). To compile regardless of header
// vintage, declare the piece we need ourselves, mirroring the stable,
// published ABI (`_FORCE_UINT32`-backed enums), instead of depending on the
// header for them.
constexpr uint32_t kZeDeviceCompilerInfoSpirvExtensions =
    2; // ZE_DEVICE_COMPILER_INFO_SPIRV_EXTENSIONS

using ZeDeviceGetCompilerInfoFn = ze_result_t (*)(ze_device_handle_t, uint32_t,
                                                  const void *, size_t *,
                                                  void *);

// `zeDeviceGetCompilerInfo` postdates LTS Level Zero loaders: on those
// systems the symbol is absent from `libze_loader` entirely, not merely
// unimplemented. Calling it as an ordinarily linked symbol would make the
// dynamic loader abort the process with an undefined-symbol error the first
// time it's referenced, so it must be resolved at runtime instead -- a
// missing symbol then just resolves to nullptr, which is treated the same
// as "unsupported".
static ZeDeviceGetCompilerInfoFn getZeDeviceGetCompilerInfo() {
  static const ZeDeviceGetCompilerInfoFn fn =
      []() -> ZeDeviceGetCompilerInfoFn {
#if defined(_WIN32)
    HMODULE handle = GetModuleHandleA("ze_loader.dll");
    if (!handle)
      return nullptr;
    return reinterpret_cast<ZeDeviceGetCompilerInfoFn>(
        GetProcAddress(handle, "zeDeviceGetCompilerInfo"));
#else
    return reinterpret_cast<ZeDeviceGetCompilerInfoFn>(
        dlsym(RTLD_DEFAULT, "zeDeviceGetCompilerInfo"));
#endif
  }();
  return fn;
}

// Queries the SPIR-V extensions the device's compiler reports via
// `zeDeviceGetCompilerInfo`, writing them into `extensions`. Returns false
// when the query itself is unsupported (e.g. LTS drivers predate this API),
// as opposed to a successful query that simply reports no extensions.
static bool queryCompilerSPIRVExtensions(ze_device_handle_t zeDevice,
                                         std::vector<std::string> &extensions) {
  ZeDeviceGetCompilerInfoFn getCompilerInfo = getZeDeviceGetCompilerInfo();
  if (!getCompilerInfo) {
    return false;
  }

  size_t size = 0;
  if (getCompilerInfo(zeDevice, kZeDeviceCompilerInfoSpirvExtensions, nullptr,
                      &size, nullptr) != ZE_RESULT_SUCCESS) {
    return false;
  }
  if (size == 0) {
    return true;
  }

  std::vector<std::array<char, ZE_MAX_EXTENSION_NAME>> rawExtensions(
      size / ZE_MAX_EXTENSION_NAME);
  if (getCompilerInfo(zeDevice, kZeDeviceCompilerInfoSpirvExtensions, nullptr,
                      &size, rawExtensions.data()) != ZE_RESULT_SUCCESS) {
    return false;
  }

  extensions.reserve(rawExtensions.size());
  for (const auto &extension : rawExtensions)
    extensions.emplace_back(extension.data());
  return true;
}

// Queries (via `zeDeviceGetCompilerInfo`) and caches the SPIR-V extensions
// for the device at `device_idx`. The query runs at most once per device
// per process; subsequent calls (from `check_extension` or
// `get_device_extensions`) reuse the cached result.
static const DeviceCompilerExtensions &
getDeviceCompilerExtensions(size_t device_idx) {
  std::optional<DeviceCompilerExtensions> &cached =
      g_extensionsCache[device_idx];
  if (!cached) {
    DeviceCompilerExtensions result;
    ze_device_handle_t zeDevice =
        sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
            g_devices[device_idx]);
    result.apiSupported =
        queryCompilerSPIRVExtensions(zeDevice, result.extensions);
    cached = std::move(result);
  }
  return *cached;
}

// `zeDeviceGetCompilerInfo` is unavailable on LTS drivers. For the handful of
// SPIR-V extensions this backend actually probes, fall back to the older
// OpenCL-named extension query (`ext_oneapi_supports_cl_extension`), which
// LTS drivers do support.
static const char *spirvToOpenCLExtension(const char *extension) {
  static const std::pair<const char *, const char *> kFallbackNames[] = {
      {"SPV_INTEL_2d_block_io", "cl_intel_subgroup_2d_block_io"},
      {"SPV_INTEL_subgroup_matrix_multiply_accumulate",
       "cl_intel_subgroup_matrix_multiply_accumulate"},
      {"SPV_INTEL_bfloat16_conversion", "cl_intel_bfloat16_conversions"},
      {"SPV_INTEL_tensor_float32_conversion",
       "cl_intel_subgroup_matrix_multiply_accumulate_tensor_float32"},
  };
  for (const auto &entry : kFallbackNames) {
    if (std::strcmp(entry.first, extension) == 0)
      return entry.second;
  }
  return nullptr;
}

extern "C" EXPORT_FUNC PyObject *check_extension(int device_id,
                                                 const char *extension) {
  try {
    initializeDevicesIfNeeded();

    int device_idx = findDeviceIndex(device_id);
    if (device_idx == -1) {
      Py_RETURN_NONE;
    }

    const DeviceCompilerExtensions &cached =
        getDeviceCompilerExtensions(device_idx);
    if (cached.apiSupported) {
      for (const auto &supported : cached.extensions) {
        if (supported == extension) {
          Py_RETURN_TRUE;
        }
      }
      Py_RETURN_FALSE;
    }

    // `zeDeviceGetCompilerInfo` isn't supported by this driver (LTS): fall
    // back to the OpenCL-named extension query for the extensions we know an
    // equivalent name for.
    if (const char *clExtension = spirvToOpenCLExtension(extension)) {
      if (g_devices[device_idx].ext_oneapi_supports_cl_extension(clExtension)) {
        Py_RETURN_TRUE;
      }
    }
    Py_RETURN_FALSE;

  } catch (const std::exception &e) {
    Py_RETURN_FALSE;
  }
}

extern "C" EXPORT_FUNC PyObject *get_device_extensions(int device_id) {
  try {
    initializeDevicesIfNeeded();

    int device_idx = findDeviceIndex(device_id);
    if (device_idx == -1) {
      Py_RETURN_NONE;
    }

    // Reflects only the `zeDeviceGetCompilerInfo` SPIR-V extension list, so
    // it's empty when that query is unsupported (e.g. LTS drivers); callers
    // that need those specific extensions should go through
    // `check_extension`, which falls back to the legacy OpenCL-named query.
    const DeviceCompilerExtensions &cached =
        getDeviceCompilerExtensions(device_idx);

    PyObject *result = PyTuple_New(cached.extensions.size());
    if (!result) {
      return NULL;
    }
    for (size_t i = 0; i < cached.extensions.size(); ++i) {
      PyObject *item = PyUnicode_FromString(cached.extensions[i].c_str());
      if (!item) {
        Py_DECREF(result);
        return NULL;
      }
      PyTuple_SetItem(result, i, item);
    }
    return result;

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
    if (device_idx < 0 || device_idx >= static_cast<int>(g_device_ids.size())) {
      PyErr_Format(PyExc_RuntimeError,
                   "Invalid device index: %d (must be in range [0, %zu))",
                   device_idx, g_device_ids.size());
      return NULL;
    }

    // Return the device_id for the device at the given index
    return PyLong_FromLong(g_device_ids[device_idx]);

  } catch (const std::exception &e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return NULL;
  }
}
