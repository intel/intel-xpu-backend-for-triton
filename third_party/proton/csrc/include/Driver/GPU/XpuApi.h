#ifndef PROTON_DRIVER_GPU_SYCL_H_
#define PROTON_DRIVER_GPU_SYCL_H_

#include "Driver/Device.h"

namespace proton {

namespace xpu {

/*
template <bool CheckSuccess> CUresult init(int flags);

template <bool CheckSuccess> CUresult ctxSynchronize();

template <bool CheckSuccess> CUresult ctxGetCurrent(CUcontext *pctx);

template <bool CheckSuccess>
CUresult deviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev);

template <bool CheckSuccess> CUresult deviceGet(CUdevice *device, int ordinal);
*/

Device getDevice(uint64_t index);

} // namespace xpu

} // namespace proton

#endif // PROTON_DRIVER_GPU_SYCL_H_
