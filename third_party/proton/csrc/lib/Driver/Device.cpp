#include "Device.h"
#include "Driver/GPU/CudaApi.h"
#include "Driver/GPU/HipApi.h"
#include "Driver/GPU/XpuApi.h"

#include "Utility/Errors.h"

namespace proton {

Device getDevice(DeviceType type, uint64_t index) {
  if (type == DeviceType::CUDA) {
    return cuda::getDevice(index);
  }
  if (type == DeviceType::HIP) {
    return hip::getDevice(index);
  }
  if (type == DeviceType::XPU) {
    return xpu::getDevice(index);
  }
  throw makeInvalidArgument("DeviceType not supported");
}

const std::string getDeviceTypeString(DeviceType type) {
  if (type == DeviceType::CUDA) {
    return DeviceTraits<DeviceType::CUDA>::name;
  } else if (type == DeviceType::HIP) {
    return DeviceTraits<DeviceType::HIP>::name;
  } else if (type == DeviceType::XPU) {
    return DeviceTraits<DeviceType::XPU>::name;
  }
  throw makeInvalidArgument("DeviceType not supported");
}

} // namespace proton
