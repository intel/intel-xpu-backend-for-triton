#ifndef PROTON_DRIVER_GPU_XPUPTI_H_
#define PROTON_DRIVER_GPU_XPUPTI_H_

#include <pti/pti_callback.h>
#include <pti/pti_view.h>

namespace proton {

namespace xpupti {

using Pti_Activity = pti_view_record_base;

template <bool CheckSuccess> pti_result viewEnable(pti_view_kind kind);

template <bool CheckSuccess> pti_result viewDisable(pti_view_kind kind);

template <bool CheckSuccess> pti_result viewFlushAll();

template <bool CheckSuccess>
pti_result subscribe(pti_callback_subscriber_handle *subscriber,
                     pti_callback_function callback, void *user_data);

template <bool CheckSuccess>
pti_result unsubscribe(pti_callback_subscriber_handle subscriber);

template <bool CheckSuccess>
pti_result enableDomain(pti_callback_subscriber_handle subscriber,
                        pti_callback_domain domain, uint32_t enter_cb,
                        uint32_t exit_cb);

template <bool CheckSuccess>
pti_result disableDomain(pti_callback_subscriber_handle subscriber,
                         pti_callback_domain domain);

template <bool CheckSuccess>
pti_result viewGetNextRecord(uint8_t *buffer, size_t valid_bytes,
                             pti_view_record_base **record);

template <bool CheckSuccess>
pti_result viewSetCallbacks(pti_fptr_buffer_requested fptr_bufferRequested,
                            pti_fptr_buffer_completed fptr_bufferCompleted);

} // namespace xpupti

} // namespace proton

#endif // PROTON_DRIVER_GPU_XPUPTI_H_
