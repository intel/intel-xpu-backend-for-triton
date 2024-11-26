#ifndef PROTON_DRIVER_GPU_XPUPTI_H_
#define PROTON_DRIVER_GPU_XPUPTI_H_

#include <pti/pti_view.h>

namespace proton {

namespace xpupti {

using Pti_Activity = pti_view_record_base;

/*
template <bool CheckSuccess> CUptiResult getVersion(uint32_t *version);

template <bool CheckSuccess>
CUptiResult getContextId(CUcontext context, uint32_t *pCtxId);

template <bool CheckSuccess>
CUptiResult activityRegisterCallbacks(
    CUpti_BuffersCallbackRequestFunc funcBufferRequested,
    CUpti_BuffersCallbackCompleteFunc funcBufferCompleted);

template <bool CheckSuccess>
CUptiResult subscribe(CUpti_SubscriberHandle *subscriber,
                      CUpti_CallbackFunc callback, void *userdata);

template <bool CheckSuccess>
CUptiResult enableDomain(uint32_t enable, CUpti_SubscriberHandle subscriber,
                         CUpti_CallbackDomain domain);

template <bool CheckSuccess>
CUptiResult enableCallback(uint32_t enable, CUpti_SubscriberHandle subscriber,
                           CUpti_CallbackDomain domain, CUpti_CallbackId cbid);

template <bool CheckSuccess>
CUptiResult activityEnableContext(CUcontext context, CUpti_ActivityKind kind);

template <bool CheckSuccess>
CUptiResult activityDisableContext(CUcontext context, CUpti_ActivityKind kind);
*/

template <bool CheckSuccess> pti_result viewEnable(pti_view_kind kind);

template <bool CheckSuccess> pti_result viewDisable(pti_view_kind kind);

template <bool CheckSuccess> pti_result viewFlushAll();

/*
template <bool CheckSuccess>
CUptiResult activityGetNextRecord(uint8_t *buffer, size_t validBufferSizeBytes,
                                  CUpti_Activity **record);

template <bool CheckSuccess>
CUptiResult
activityPushExternalCorrelationId(CUpti_ExternalCorrelationKind kind,
                                  uint64_t id);

template <bool CheckSuccess>
CUptiResult activityPopExternalCorrelationId(CUpti_ExternalCorrelationKind kind,
                                             uint64_t *lastId);

template <bool CheckSuccess>
CUptiResult activitySetAttribute(CUpti_ActivityAttribute attr,
                                 size_t *valueSize, void *value);

template <bool CheckSuccess>
CUptiResult unsubscribe(CUpti_SubscriberHandle subscriber);

template <bool CheckSuccess> CUptiResult finalize();

template <bool CheckSuccess>
CUptiResult getGraphExecId(CUgraphExec graph, uint32_t *pId);

template <bool CheckSuccess>
CUptiResult getGraphId(CUgraph graph, uint32_t *pId);

template <bool CheckSuccess>
CUptiResult getCubinCrc(CUpti_GetCubinCrcParams *pParams);

template <bool CheckSuccess>
CUptiResult
getSassToSourceCorrelation(CUpti_GetSassToSourceCorrelationParams *pParams);

template <bool CheckSuccess>
CUptiResult
pcSamplingGetNumStallReasons(CUpti_PCSamplingGetNumStallReasonsParams *pParams);

template <bool CheckSuccess>
CUptiResult
pcSamplingGetStallReasons(CUpti_PCSamplingGetStallReasonsParams *pParams);

template <bool CheckSuccess>
CUptiResult pcSamplingSetConfigurationAttribute(
    CUpti_PCSamplingConfigurationInfoParams *pParams);

template <bool CheckSuccess>
CUptiResult pcSamplingEnable(CUpti_PCSamplingEnableParams *pParams);

template <bool CheckSuccess>
CUptiResult pcSamplingDisable(CUpti_PCSamplingDisableParams *pParams);

template <bool CheckSuccess>
CUptiResult pcSamplingGetData(CUpti_PCSamplingGetDataParams *pParams);

template <bool CheckSuccess>
CUptiResult pcSamplingStart(CUpti_PCSamplingStartParams *pParams);

template <bool CheckSuccess>
CUptiResult pcSamplingStop(CUpti_PCSamplingStopParams *pParams);
*/

} // namespace xpupti

} // namespace proton

#endif // PROTON_EXTERN_DISPATCH_H_
