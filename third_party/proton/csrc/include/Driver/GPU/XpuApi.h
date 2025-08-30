#ifndef PROTON_DRIVER_GPU_SYCL_H_
#define PROTON_DRIVER_GPU_SYCL_H_

#include "Device.h"
#include <level_zero/ze_api.h>

namespace proton {

namespace xpu {

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

Device getDevice(uint64_t index);

} // namespace xpu

} // namespace proton

#endif // PROTON_DRIVER_GPU_SYCL_H_
