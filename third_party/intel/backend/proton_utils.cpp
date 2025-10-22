#include "Driver/Dispatch.h"

#include <cstring>
#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>

extern "C" void waitOnSyclQueue(void *syclQueue) {
  sycl::queue *queue = static_cast<sycl::queue *>(syclQueue);
  queue->wait();
}

// FIXME: Should it be in DeviceInfo class?
// Inspired by Kineto: `XpuptiActivityProfiler.cpp`
extern "C" void
enumDeviceUUIDs(std::vector<std::array<uint8_t, 16>> deviceUUIDs_) {
  if (!deviceUUIDs_.empty()) {
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
          deviceUUIDs_.push_back(
              device.get_info<sycl::ext::intel::info::device::uuid>());
        } else {
          std::cerr << "Warnings: UUID is not supported for this XPU device. "
                       "The device index of records will be 0."
                    << std::endl;
          deviceUUIDs_.push_back(std::array<uint8_t, 16>{});
        }
      }
    }
  }
}

namespace proton {

namespace xpu {

struct ExternLibLevelZero : public ExternLibBase {
  using RetType = ze_result_t;

  // FIXME: removeme `/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1`
  static constexpr const char *name = "libze_intel_gpu.so.1";
  static constexpr const char *defaultDir = "";
  static constexpr RetType success = ZE_RESULT_SUCCESS;
  static void *lib;
};

void *ExternLibLevelZero::lib = nullptr;

// moved here to avoid adding dependency `level_zero/ze_api.h` in `XpuApi.h`
template <bool CheckSuccess> ze_result_t init(ze_init_flags_t flags);

template <bool CheckSuccess>
ze_result_t driverGet(uint32_t *pCount, ze_driver_handle_t *phDrivers);

template <bool CheckSuccess>
ze_result_t deviceGet(ze_driver_handle_t hDriver, uint32_t *pCount,
                      ze_device_handle_t *phDevices);

template <bool CheckSuccess>
ze_result_t deviceGetProperties(ze_device_handle_t hDevice,
                                ze_device_properties_t *pDeviceProperties);

template <bool CheckSuccess>
ze_result_t
deviceGetMemoryProperties(ze_device_handle_t hDevice, uint32_t *pCount,
                          ze_device_memory_properties_t *pMemProperties);

// https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/api.html#zeinit
DEFINE_DISPATCH(ExternLibLevelZero, init, zeInit, ze_init_flags_t)
// https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/api.html#zedriverget
DEFINE_DISPATCH(ExternLibLevelZero, driverGet, zeDriverGet, uint32_t *,
                ze_driver_handle_t *)
// https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/api.html#zedeviceget
DEFINE_DISPATCH(ExternLibLevelZero, deviceGet, zeDeviceGet, ze_driver_handle_t,
                uint32_t *, ze_device_handle_t *)
// https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/api.html#zedevicegetproperties
DEFINE_DISPATCH(ExternLibLevelZero, deviceGetProperties, zeDeviceGetProperties,
                ze_device_handle_t, ze_device_properties_t *)
// https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/api.html#zedevicegetmemoryproperties
DEFINE_DISPATCH(ExternLibLevelZero, deviceGetMemoryProperties,
                zeDeviceGetMemoryProperties, ze_device_handle_t, uint32_t *,
                ze_device_memory_properties_t *)

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
  xpu::init<true>(ZE_INIT_FLAG_GPU_ONLY);

  // FIXME: For now I use the naive approach that the device index from PTI
  // record coincides with the default numbering of all devices
  uint32_t driverCount = 1;
  ze_driver_handle_t driverHandle;
  xpu::driverGet<true>(&driverCount, &driverHandle);
  uint32_t deviceCount = 1;
  // Get device handle
  ze_device_handle_t phDevice;
  xpu::deviceGet<true>(driverHandle, &deviceCount, &phDevice);
  // create a struct to hold device properties
  ze_device_properties_t device_properties = {};
  device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
  xpu::deviceGetProperties<true>(phDevice, &device_properties);
  *clockRate = device_properties.coreClockRate;
  *numSms =
      device_properties.numSlices * device_properties.numSubslicesPerSlice;
  // create a struct to hold device memory properties
  uint32_t memoryCount = 0;
  xpu::deviceGetMemoryProperties<true>(phDevice, &memoryCount, nullptr);
  auto pMemoryProperties = new ze_device_memory_properties_t[memoryCount];
  for (uint32_t mem = 0; mem < memoryCount; ++mem) {
    pMemoryProperties[mem].stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
    pMemoryProperties[mem].pNext = nullptr;
  }
  xpu::deviceGetMemoryProperties<true>(phDevice, &memoryCount,
                                       pMemoryProperties);

  *memoryClockRate = pMemoryProperties[0].maxClockRate;
  *busWidth = pMemoryProperties[0].maxBusWidth;

  delete[] pMemoryProperties;

  // FIXME: there should be architecture, but not a name
  memcpy(arch, device_properties.name, 256);
}

} // namespace xpu

} // namespace proton
