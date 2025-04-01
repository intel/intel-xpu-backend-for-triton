#include "Driver/GPU/XpuptiApi.h"
#include "Driver/Device.h"
#include "Driver/Dispatch.h"

namespace proton {

namespace xpupti {

struct ExternLibXpupti : public ExternLibBase {
  using RetType = pti_result;
  // FIXME: ref: /opt/intel/oneapi/pti/latest/lib/libpti_view.so
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

} // namespace xpupti

} // namespace proton
