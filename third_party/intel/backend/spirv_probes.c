/*
 * Ground-truth probe for whether a SPIR-V capability/extension is accepted
 * by the installed driver's own JIT. Raw Level Zero (no SYCL), so it skips
 * SYCL's backend/plugin init and doesn't depend on OpenCL or `ocloc` being
 * present -- see `extension_utils.c`'s `check_extension()` for the
 * OpenCL-extension-advertisement side this complements.
 */

#include <algorithm>
#include <vector>

#if defined(_WIN32)
#define EXPORT_FUNC __declspec(dllexport)
#else
#define EXPORT_FUNC __attribute__((visibility("default")))
#endif

#include <Python.h>

#include "sycl_functions.h"

// Index of `device_id` in `device_ids`, or -1 if absent.
static int findDeviceIndex(const std::vector<int> &device_ids, int device_id) {
  auto it = std::find(device_ids.begin(), device_ids.end(), device_id);
  return it == device_ids.end() ? -1
                                : static_cast<int>(it - device_ids.begin());
}

static std::vector<ze_device_handle_t> g_l0_probe_devices;
static std::vector<ze_context_handle_t>
    g_l0_probe_contexts; // one per device's driver
static std::vector<int> g_l0_probe_device_ids;
static bool g_l0_probe_devices_initialized = false;

static void initializeLevelZeroDevicesForProbingIfNeeded() {
  if (g_l0_probe_devices_initialized)
    return;
  g_l0_probe_devices_initialized = true;

  // `zeInit` is multi-call-safe and only initializes each driver once per
  // process, so this is safe alongside `driver.c`/`proton_utils.cpp`'s own
  // `zeInit` calls.
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
    // destroyed (process-lifetime).
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
      // The Level Zero PCI device ID -- matches
      // sycl::ext::intel::info::device::device_id with no SYCL layer here.
      g_l0_probe_device_ids.push_back(static_cast<int>(props.deviceId));
    }
  }
}

// Ground-truth probe for a SPIR-V capability: hands `spirv` (declaring only
// the capability under test) to the driver's JIT via `zeModuleCreate`, the
// same call every real kernel goes through. True/False once a matching
// device is found, None otherwise.
extern "C" EXPORT_FUNC PyObject *probe_spirv_module(int device_id,
                                                    const unsigned char *spirv,
                                                    size_t spirv_size) {
  try {
    initializeLevelZeroDevicesForProbingIfNeeded();

    int device_idx = findDeviceIndex(g_l0_probe_device_ids, device_id);
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
