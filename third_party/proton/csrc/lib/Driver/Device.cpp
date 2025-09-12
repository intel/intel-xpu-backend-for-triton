#include "Device.h"
#include "Driver/GPU/CudaApi.h"
#include "Driver/GPU/HipApi.h"
#ifdef TRITON_BUILD_PROTON_XPU
#include "Driver/GPU/XpuApi.h"
#endif

#include "Utility/Errors.h"

namespace proton {

Device getDevice(DeviceType type, uint64_t index) {
  if (type == DeviceType::CUDA) {
    return cuda::getDevice(index);
  }
  if (type == DeviceType::HIP) {
    return hip::getDevice(index);
  }
#ifdef TRITON_BUILD_PROTON_XPU
  if (type == DeviceType::XPU) {
    return xpu::getDevice(index);
  }
#endif
  throw std::runtime_error("DeviceType not supported");
}

const std::string getDeviceTypeString(DeviceType type) {
  if (type == DeviceType::CUDA) {
    return DeviceTraits<DeviceType::CUDA>::name;
  } else if (type == DeviceType::HIP) {
    return DeviceTraits<DeviceType::HIP>::name;
  } else if (type == DeviceType::XPU) {
    return DeviceTraits<DeviceType::XPU>::name;
  }
  throw std::runtime_error("DeviceType not supported");
}

} // namespace proton
