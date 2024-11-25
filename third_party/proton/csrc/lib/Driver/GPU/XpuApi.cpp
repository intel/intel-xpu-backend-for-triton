#include "Driver/GPU/XpuApi.h"
#include "Driver/Dispatch.h"

#include "sycl_functions.h"
#include <level_zero/ze_api.h>
#include <string>
#include <vector>

namespace proton {

namespace xpu {

/*
struct ExternLibCuda : public ExternLibBase {
  using RetType = CUresult;
  //
https://forums.developer.nvidia.com/t/wsl2-libcuda-so-and-libcuda-so-1-should-be-symlink/236301
  // On WSL, "libcuda.so" and "libcuda.so.1" may not be linked, so we use
  // "libcuda.so.1" instead.
  static constexpr const char *name = "libcuda.so.1";
  static constexpr const char *defaultDir = "";
  static constexpr RetType success = CUDA_SUCCESS;
  static void *lib;
};

void *ExternLibCuda::lib = nullptr;

DEFINE_DISPATCH(ExternLibCuda, init, cuInit, int)

DEFINE_DISPATCH(ExternLibCuda, ctxSynchronize, cuCtxSynchronize)

DEFINE_DISPATCH(ExternLibCuda, ctxGetCurrent, cuCtxGetCurrent, CUcontext *)

DEFINE_DISPATCH(ExternLibCuda, deviceGet, cuDeviceGet, CUdevice *, int)

DEFINE_DISPATCH(ExternLibCuda, deviceGetAttribute, cuDeviceGetAttribute, int *,
                CUdevice_attribute, CUdevice)
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
