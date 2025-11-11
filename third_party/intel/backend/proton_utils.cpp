#include <cstring>
#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>

extern "C" void waitOnSyclQueue(void *syclQueue) {
  sycl::queue *queue = static_cast<sycl::queue *>(syclQueue);
  queue->wait();
}

// FIXME: Should it be in DeviceInfo class?
// Inspired by Kineto: `XpuptiActivityProfiler.cpp`
extern "C" void enumDeviceUUIDs(void *deviceUUIDsPtr) {
  auto *deviceUUIDs_ =
      reinterpret_cast<std::vector<std::array<uint8_t, 16>> *>(deviceUUIDsPtr);
  if (!deviceUUIDs_->empty()) {
    return;
  }
  auto platform_list = sycl::platform::get_platforms();
  // Enumerated GPU devices from the specific platform.
  for (const auto &platform : platform_list) {
    if (platform.get_backend() != sycl::backend::ext_oneapi_level_zero) {
      continue;
    }
    auto device_list = platform.get_devices();
    for (const auto &device : device_list) {
      if (device.is_gpu()) {
        if (device.has(sycl::aspect::ext_intel_device_info_uuid)) {
          deviceUUIDs_->push_back(
              device.get_info<sycl::ext::intel::info::device::uuid>());
        } else {
          std::cerr << "Warnings: UUID is not supported for this XPU device. "
                       "The device index of records will be 0."
                    << std::endl;
          deviceUUIDs_->push_back(std::array<uint8_t, 16>{});
        }
      }
    }
  }
}

namespace proton {

namespace xpu {

void check(ze_result_t ret, const char *functionName) {
  if (ret != ZE_RESULT_SUCCESS) {
    throw std::runtime_error("Failed to execute " + std::string(functionName) +
                             " with error " + std::to_string(ret));
  }
}

// FIXME: for this initialization is needed
// ref: initDevices
// static std::vector<std::pair<sycl::device, ze_device_handle_t>>
//    g_sycl_l0_device_list;

// FIXME: rewrite with
// sycl::device.get_info<sycl::ext::intel::info::device::architecture>; cache
// the result
extern "C" void getDeviceProperties(uint64_t index, uint32_t *clockRate,
                                    uint32_t *memoryClockRate,
                                    uint32_t *busWidth, uint32_t *numSms,
                                    char arch[256]) {
  // ref: getDeviceProperties

  // FIXME: double check that initialization is needed
  // At the very least, it shouldn't be for every call
  check(zeInit(ZE_INIT_FLAG_GPU_ONLY), "zeInit");

  // FIXME: For now I use the naive approach that the device index from PTI
  // record coincides with the default numbering of all devices
  uint32_t driverCount = 1;
  ze_driver_handle_t driverHandle;
  check(zeDriverGet(&driverCount, &driverHandle), "zeDriverGet");
  uint32_t deviceCount = 1;
  // Get device handle
  ze_device_handle_t phDevice;
  check(zeDeviceGet(driverHandle, &deviceCount, &phDevice), "zeDeviceGet");
  // create a struct to hold device properties
  ze_device_properties_t device_properties = {};
  device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
  check(zeDeviceGetProperties(phDevice, &device_properties),
        "zeDeviceGetProperties");
  *clockRate = device_properties.coreClockRate;
  *numSms =
      device_properties.numSlices * device_properties.numSubslicesPerSlice;
  // create a struct to hold device memory properties
  uint32_t memoryCount = 0;
  check(zeDeviceGetMemoryProperties(phDevice, &memoryCount, nullptr),
        "zeDeviceGetMemoryProperties");
  auto pMemoryProperties = new ze_device_memory_properties_t[memoryCount];
  for (uint32_t mem = 0; mem < memoryCount; ++mem) {
    pMemoryProperties[mem].stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
    pMemoryProperties[mem].pNext = nullptr;
  }
  check(zeDeviceGetMemoryProperties(phDevice, &memoryCount, pMemoryProperties),
        "zeDeviceGetMemoryProperties");

  // To align with other backends - convert MHz to KHz
  // https://github.com/intel/compute-runtime/blob/cfa007e5519d3a038d726b62237b86fca9a49e2c/shared/source/xe_hpc_core/linux/product_helper_pvc.cpp#L51
  *memoryClockRate = pMemoryProperties[0].maxClockRate * 1000;
  *busWidth = pMemoryProperties[0].maxBusWidth;

  delete[] pMemoryProperties;

  // FIXME: there should be architecture, but not a name
  memcpy(arch, device_properties.name, 256);
}

} // namespace xpu

} // namespace proton
