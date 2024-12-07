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

/*
DEFINE_DISPATCH(ExternLibCupti, getVersion, cuptiGetVersion, uint32_t *);

DEFINE_DISPATCH(ExternLibCupti, getContextId, cuptiGetContextId, CUcontext,
                uint32_t *);

DEFINE_DISPATCH(ExternLibCupti, activityRegisterCallbacks,
                cuptiActivityRegisterCallbacks,
                CUpti_BuffersCallbackRequestFunc,
                CUpti_BuffersCallbackCompleteFunc)

DEFINE_DISPATCH(ExternLibCupti, subscribe, cuptiSubscribe,
                CUpti_SubscriberHandle *, CUpti_CallbackFunc, void *)

DEFINE_DISPATCH(ExternLibCupti, enableDomain, cuptiEnableDomain, uint32_t,
                CUpti_SubscriberHandle, CUpti_CallbackDomain)

DEFINE_DISPATCH(ExternLibCupti, enableCallback, cuptiEnableCallback, uint32_t,
                CUpti_SubscriberHandle, CUpti_CallbackDomain, CUpti_CallbackId);
*/

DEFINE_DISPATCH(ExternLibXpupti, viewEnable, ptiViewEnable, pti_view_kind)

DEFINE_DISPATCH(ExternLibXpupti, viewDisable, ptiViewDisable, pti_view_kind)

/*
DEFINE_DISPATCH(ExternLibCupti, activityEnableContext,
                cuptiActivityEnableContext, CUcontext, CUpti_ActivityKind)

DEFINE_DISPATCH(ExternLibCupti, activityDisableContext,
                cuptiActivityDisableContext, CUcontext, CUpti_ActivityKind)
*/

DEFINE_DISPATCH(ExternLibXpupti, viewFlushAll, ptiFlushAllViews)

/*
DEFINE_DISPATCH(ExternLibCupti, activityGetNextRecord,
                cuptiActivityGetNextRecord, uint8_t *, size_t,
                CUpti_Activity **)

DEFINE_DISPATCH(ExternLibCupti, activityPushExternalCorrelationId,
                cuptiActivityPushExternalCorrelationId,
                CUpti_ExternalCorrelationKind, uint64_t)

DEFINE_DISPATCH(ExternLibCupti, activityPopExternalCorrelationId,
                cuptiActivityPopExternalCorrelationId,
                CUpti_ExternalCorrelationKind, uint64_t *)

DEFINE_DISPATCH(ExternLibCupti, activitySetAttribute, cuptiActivitySetAttribute,
                CUpti_ActivityAttribute, size_t *, void *)

DEFINE_DISPATCH(ExternLibCupti, unsubscribe, cuptiUnsubscribe,
                CUpti_SubscriberHandle)

DEFINE_DISPATCH(ExternLibCupti, finalize, cuptiFinalize)

DEFINE_DISPATCH(ExternLibCupti, getGraphExecId, cuptiGetGraphExecId,
                CUgraphExec, uint32_t *);

DEFINE_DISPATCH(ExternLibCupti, getGraphId, cuptiGetGraphId, CUgraph,
                uint32_t *);

DEFINE_DISPATCH(ExternLibCupti, getCubinCrc, cuptiGetCubinCrc,
                CUpti_GetCubinCrcParams *);

DEFINE_DISPATCH(ExternLibCupti, getSassToSourceCorrelation,
                cuptiGetSassToSourceCorrelation,
                CUpti_GetSassToSourceCorrelationParams *);

DEFINE_DISPATCH(ExternLibCupti, pcSamplingGetNumStallReasons,
                cuptiPCSamplingGetNumStallReasons,
                CUpti_PCSamplingGetNumStallReasonsParams *);

DEFINE_DISPATCH(ExternLibCupti, pcSamplingGetStallReasons,
                cuptiPCSamplingGetStallReasons,
                CUpti_PCSamplingGetStallReasonsParams *);

DEFINE_DISPATCH(ExternLibCupti, pcSamplingSetConfigurationAttribute,
                cuptiPCSamplingSetConfigurationAttribute,
                CUpti_PCSamplingConfigurationInfoParams *);

DEFINE_DISPATCH(ExternLibCupti, pcSamplingEnable, cuptiPCSamplingEnable,
                CUpti_PCSamplingEnableParams *);

DEFINE_DISPATCH(ExternLibCupti, pcSamplingDisable, cuptiPCSamplingDisable,
                CUpti_PCSamplingDisableParams *);

DEFINE_DISPATCH(ExternLibCupti, pcSamplingGetData, cuptiPCSamplingGetData,
                CUpti_PCSamplingGetDataParams *);

DEFINE_DISPATCH(ExternLibCupti, pcSamplingStart, cuptiPCSamplingStart,
                CUpti_PCSamplingStartParams *);

DEFINE_DISPATCH(ExternLibCupti, pcSamplingStop, cuptiPCSamplingStop,
                CUpti_PCSamplingStopParams *);

*/
} // namespace xpupti

} // namespace proton
