#ifndef PROTON_DRIVER_GPU_SYCL_H_
#define PROTON_DRIVER_GPU_SYCL_H_

#include "Driver/Device.h"
#include <level_zero/ze_api.h>

namespace proton {

namespace xpu {

template <bool CheckSuccess> ze_result_t init(ze_init_flags_t flags);

template <bool CheckSuccess>
ze_result_t ctxSynchronize(ze_command_queue_handle_t hCommandQueue,
                           uint64_t timeout);

/*

template <bool CheckSuccess> CUresult ctxGetCurrent(CUcontext *pctx);

template <bool CheckSuccess> CUresult deviceGet(CUdevice *device, int ordinal);
*/

Device getDevice(uint64_t index);

} // namespace xpu

} // namespace proton

#endif // PROTON_DRIVER_GPU_SYCL_H_
