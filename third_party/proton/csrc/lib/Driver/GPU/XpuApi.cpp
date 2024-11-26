#include "Driver/GPU/XpuApi.h"
#include "Driver/Dispatch.h"

#include "sycl_functions.h"
#include <level_zero/ze_api.h>
#include <string>
#include <vector>

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

// FIXME: DEBUG ref:
// https://spec.oneapi.io/level-zero/1.0.4/core/api.html#zeinit
DEFINE_DISPATCH(ExternLibLevelZero, init, zeInit, ze_init_flags_t)

// FIXME: probably it's better to change `ctxSynchronize` name;
// leave it like this for now, so that it would be easier to compare
// the implementation with other backends
// SPEC:
// https://spec.oneapi.io/level-zero/1.9.3/core/api.html#zecommandqueuesynchronize
DEFINE_DISPATCH(ExternLibLevelZero, ctxSynchronize, zeCommandQueueSynchronize,
                ze_command_queue_handle_t, uint64_t)

/*
DEFINE_DISPATCH(ExternLibCuda, ctxGetCurrent, cuCtxGetCurrent, CUcontext *)

DEFINE_DISPATCH(ExternLibCuda, deviceGet, cuDeviceGet, CUdevice *, int)

*/

// FIXME: for this initialization is needed
// ref: initDevices
static std::vector<std::pair<sycl::device, ze_device_handle_t>>
    g_sycl_l0_device_list;

// FIXME: probably `DEFINE_DISPATCH` should be used in this function
Device getDevice(uint64_t index) {
  // ref: getDeviceProperties
  const auto device = g_sycl_l0_device_list[index];

  // Get device handle
  ze_device_handle_t phDevice = device.second;

  // create a struct to hold device properties
  ze_device_properties_t device_properties = {};
  device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
  // FIXME: should it be: `zeDeviceGetComputeProperties` and
  // `ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES` ref:
  // https://spec.oneapi.io/level-zero/1.0.4/core/api.html
  zeDeviceGetProperties(phDevice, &device_properties);

  uint32_t clockRate = device_properties.coreClockRate;
  uint32_t numSms =
      device_properties.numSlices * device_properties.numSubslicesPerSlice;

  // create a struct to hold device memory properties
  uint32_t memoryCount = 0;
  zeDeviceGetMemoryProperties(phDevice, &memoryCount, nullptr);
  auto pMemoryProperties = new ze_device_memory_properties_t[memoryCount];
  for (uint32_t mem = 0; mem < memoryCount; ++mem) {
    pMemoryProperties[mem].stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
    pMemoryProperties[mem].pNext = nullptr;
  }
  zeDeviceGetMemoryProperties(phDevice, &memoryCount, pMemoryProperties);

  int memoryClockRate = pMemoryProperties[0].maxClockRate;
  int busWidth = pMemoryProperties[0].maxBusWidth;

  delete[] pMemoryProperties;

  // FIXME how this can be defined for XPU?
  // std::string arch = std::to_string(major * 10 + minor);
  std::string arch = "unknown";

  return Device(DeviceType::XPU, index, clockRate, memoryClockRate, busWidth,
                numSms, arch);
}

} // namespace xpu

} // namespace proton
