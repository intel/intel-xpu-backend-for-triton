#include "Driver/GPU/XpuptiApi.h"
#include "Device.h"
#include "Driver/Dispatch.h"

namespace proton {

namespace xpupti {

struct ExternLibXpupti : public ExternLibBase {
  using RetType = pti_result;
  static constexpr const char *name = "libpti_view.so";
  static constexpr const char *defaultDir = "";
  static constexpr RetType success = PTI_SUCCESS;
  static void *lib;
};

void *ExternLibXpupti::lib = nullptr;

// For inspiration see CuptiApi.cpp

DEFINE_DISPATCH(ExternLibXpupti, viewEnable, ptiViewEnable, pti_view_kind)

DEFINE_DISPATCH(ExternLibXpupti, viewDisable, ptiViewDisable, pti_view_kind)

DEFINE_DISPATCH(ExternLibXpupti, viewFlushAll, ptiFlushAllViews)

DEFINE_DISPATCH(ExternLibXpupti, viewGetNextRecord, ptiViewGetNextRecord,
                uint8_t *, size_t, pti_view_record_base **)

DEFINE_DISPATCH(ExternLibXpupti, viewSetCallbacks, ptiViewSetCallbacks,
                pti_fptr_buffer_requested, pti_fptr_buffer_completed)

} // namespace xpupti

} // namespace proton
