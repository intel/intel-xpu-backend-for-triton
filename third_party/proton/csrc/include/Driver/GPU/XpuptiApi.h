#ifndef PROTON_DRIVER_GPU_XPUPTI_H_
#define PROTON_DRIVER_GPU_XPUPTI_H_

#include <pti/pti_view.h>

namespace proton {

namespace xpupti {

using Pti_Activity = pti_view_record_base;

template <bool CheckSuccess> pti_result viewEnable(pti_view_kind kind);

template <bool CheckSuccess> pti_result viewDisable(pti_view_kind kind);

template <bool CheckSuccess> pti_result viewFlushAll();

template <bool CheckSuccess>
pti_result viewGetNextRecord(uint8_t *buffer, size_t valid_bytes,
                             pti_view_record_base **record);

template <bool CheckSuccess>
pti_result viewSetCallbacks(pti_fptr_buffer_requested fptr_bufferRequested,
                            pti_fptr_buffer_completed fptr_bufferCompleted);

} // namespace xpupti

} // namespace proton

#endif // PROTON_EXTERN_DISPATCH_H_
