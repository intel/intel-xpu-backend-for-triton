#ifndef PROTON_DRIVER_GPU_XPUPTI_H_
#define PROTON_DRIVER_GPU_XPUPTI_H_

#include <pti/pti_view.h>

namespace proton {

namespace xpupti {

using Pti_Activity = pti_view_record_base;

template <bool CheckSuccess> pti_result viewEnable(pti_view_kind kind);

template <bool CheckSuccess> pti_result viewDisable(pti_view_kind kind);

template <bool CheckSuccess> pti_result viewFlushAll();

} // namespace xpupti

} // namespace proton

#endif // PROTON_EXTERN_DISPATCH_H_
