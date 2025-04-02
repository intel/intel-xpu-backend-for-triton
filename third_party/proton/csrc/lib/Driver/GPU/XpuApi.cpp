#include "Driver/GPU/XpuApi.h"
#include "Driver/Dispatch.h"

#include <level_zero/ze_api.h>
#include <string>

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

Device getDevice(uint64_t index) {
  // ref: getDeviceProperties

  // FIXME: double check that initialization is needed
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

  uint32_t clockRate = device_properties.coreClockRate;
  uint32_t numSms =
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

  int memoryClockRate = pMemoryProperties[0].maxClockRate;
  int busWidth = pMemoryProperties[0].maxBusWidth;

  delete[] pMemoryProperties;

  // FIXME: there should be architecture, but not a name
  std::string arch = device_properties.name;

  return Device(DeviceType::XPU, index, clockRate, memoryClockRate, busWidth,
                numSms, arch);
}

} // namespace xpu

} // namespace proton
