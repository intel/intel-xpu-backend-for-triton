#ifndef PROTON_DRIVER_GPU_SYCL_H_
#define PROTON_DRIVER_GPU_SYCL_H_

#include "Device.h"

namespace proton {

namespace xpu {

extern std::string XPU_API_UTILS;

Device getDevice(uint64_t index);

} // namespace xpu

} // namespace proton

#endif // PROTON_DRIVER_GPU_SYCL_H_
