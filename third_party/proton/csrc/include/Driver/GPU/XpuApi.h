#ifndef PROTON_DRIVER_GPU_SYCL_H_
#define PROTON_DRIVER_GPU_SYCL_H_

#include "Device.h"

namespace proton {

namespace xpu {

// The device type the XPU profiler reports: its own DeviceType in libproton,
// the slot for backends registered at run time when the profiler is built as a
// library of its own (PROTON_XPU_EXTERNAL, XpuptiRegistration.cpp).
#ifdef PROTON_XPU_EXTERNAL
inline constexpr DeviceType kDeviceType = DeviceType::EXTERNAL;
#else
inline constexpr DeviceType kDeviceType = DeviceType::XPU;
#endif

extern std::string PROTON_UTILS;

Device getDevice(uint64_t index);

} // namespace xpu

} // namespace proton

#endif // PROTON_DRIVER_GPU_SYCL_H_
