#include "Profiler/CuptiProfiler.h"
#include "Context/Context.h"
#include "Data/Metric.h"
#include "Driver/GPU/Cuda.h"
#include "Driver/GPU/Cupti.h"

#include <cstdlib>
#include <memory>
#include <stdexcept>

namespace proton {

namespace {

std::shared_ptr<Metric> convertActivityToMetric(CUpti_Activity *activity) {
  std::shared_ptr<Metric> metric;
  switch (activity->kind) {
  case CUPTI_ACTIVITY_KIND_KERNEL:
  case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
    auto *kernel = reinterpret_cast<CUpti_ActivityKernel5 *>(activity);
    metric =
        std::make_shared<KernelMetric>(static_cast<uint64_t>(kernel->start),
                                       static_cast<uint64_t>(kernel->end), 1);
    break;
  }
  default:
    break;
  }
  return metric;
}

void addMetric(size_t scopeId, std::set<Data *> &dataSet,
               CUpti_Activity *activity) {
  for (auto *data : dataSet) {
    data->addMetric(scopeId, convertActivityToMetric(activity));
  }
}

void processActivityExternalCorrelation(std::map<uint32_t, size_t> &correlation,
                                        CUpti_Activity *activity) {
  auto *externalCorrelation =
      reinterpret_cast<CUpti_ActivityExternalCorrelation *>(activity);
  correlation[externalCorrelation->correlationId] =
      externalCorrelation->externalId;
}

void processActivityKernel(std::map<uint32_t, size_t> &correlation,
                           std::set<Data *> &dataSet,
                           CUpti_Activity *activity) {
  // Support CUDA >= 11.0
  auto *kernel = reinterpret_cast<CUpti_ActivityKernel5 *>(activity);
  auto correlationId = kernel->correlationId;
  // TODO: non-triton kernels
  if (correlation.find(correlationId) == correlation.end()) {
    return;
  }
  auto externalId = correlation[correlationId];
  addMetric(externalId, dataSet, activity);
  // Track correlation ids from the same stream and erase those < correlationId
  correlation.erase(correlationId);
}

} // namespace

void CuptiProfiler::startOp(const Scope &scope) {
  cupti::activityPushExternalCorrelationId<true>(
      CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, scope.scopeId);
}

void CuptiProfiler::stopOp(const Scope &scope) {
  uint64_t correlationId;
  cupti::activityPopExternalCorrelationId<true>(
      CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, &correlationId);
}

void CuptiProfiler::setOpInProgress(bool value) {
  cuptiState.isRecording = value;
}

bool CuptiProfiler::isOpInProgress() { return cuptiState.isRecording; }

void CuptiProfiler::doStart() {
  cupti::activityRegisterCallbacks<true>(allocBuffer, completeBuffer);
  cupti::activityEnable<true>(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION);
  // Enable driver and runtime activities after external correlation so that
  // external correlation id returned is not 0
  cupti::activityEnable<true>(CUPTI_ACTIVITY_KIND_DRIVER);
  cupti::activityEnable<true>(CUPTI_ACTIVITY_KIND_RUNTIME);
  cupti::activityEnable<true>(CUPTI_ACTIVITY_KIND_FUNCTION);
  cupti::activityEnable<true>(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  // TODO: switch to directly subscribe the APIs and measure overhead
  cupti::subscribe<true>(&subscriber, callback, nullptr);
  cupti::enableDomain<true>(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API);
  cupti::enableDomain<true>(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
}

void CuptiProfiler::doFlush() {
  CUcontext cuContext = nullptr;
  // If the device is not initialized, cuda::ctxGetCurrent will fail.
  // So we don't throw an error here, but instead just skip the synchronization.
  cuda::ctxGetCurrent<false>(&cuContext);
  if (cuContext) {
    cuda::ctxSynchronize<true>();
  }
  cupti::activityFlushAll<true>(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
}

void CuptiProfiler::doStop() {
  cupti::activityDisable<true>(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION);
  cupti::activityDisable<true>(CUPTI_ACTIVITY_KIND_DRIVER);
  cupti::activityDisable<true>(CUPTI_ACTIVITY_KIND_RUNTIME);
  cupti::activityDisable<true>(CUPTI_ACTIVITY_KIND_FUNCTION);
  cupti::activityDisable<true>(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  cupti::enableDomain<true>(0, subscriber, CUPTI_CB_DOMAIN_DRIVER_API);
  cupti::enableDomain<true>(0, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
  cupti::unsubscribe<true>(subscriber);
  cupti::finalize<true>();
}

void CuptiProfiler::allocBuffer(uint8_t **buffer, size_t *bufferSize,
                                size_t *maxNumRecords) {
  *buffer = reinterpret_cast<uint8_t *>(aligned_alloc(AlignSize, BufferSize));
  if (*buffer == nullptr) {
    throw std::runtime_error("aligned_alloc failed");
  }
  *bufferSize = BufferSize;
  *maxNumRecords = 0;
}

void CuptiProfiler::completeBuffer(CUcontext ctx, uint32_t streamId,
                                   uint8_t *buffer, size_t size,
                                   size_t validSize) {
  CuptiProfiler &profiler =
      dynamic_cast<CuptiProfiler &>(CuptiProfiler::instance());
  auto &correlation = profiler.correlation;
  auto &dataSet = profiler.dataSet;

  CUptiResult status;
  CUpti_Activity *activity = nullptr;
  do {
    status = cupti::activityGetNextRecord<false>(buffer, validSize, &activity);
    if (status == CUPTI_SUCCESS) {
      processActivity(correlation, dataSet, activity);
    } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
      break;
    } else {
      throw std::runtime_error("cupti::activityGetNextRecord failed");
    }
  } while (true);

  free(buffer);
}

void CuptiProfiler::processActivity(std::map<uint32_t, size_t> &correlation,
                                    std::set<Data *> &dataSet,
                                    CUpti_Activity *activity) {
  switch (activity->kind) {
  case CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION: {
    processActivityExternalCorrelation(correlation, activity);
    break;
  }
  case CUPTI_ACTIVITY_KIND_KERNEL:
  case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
    processActivityKernel(correlation, dataSet, activity);
    break;
  }
  default:
    break;
  }
}

namespace {

std::pair<bool, bool> matchKernelCbId(CUpti_CallbackId cbId) {
  bool isRuntimeApi = false;
  bool isDriverApi = false;
  switch (cbId) {
  // TODO: switch to directly subscribe the APIs
  case CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020:
  case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000:
  case CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_ptsz_v7000:
  case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000:
  case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060:
  case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_ptsz_v11060:
  case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_v9000:
  case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_ptsz_v9000:
  case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernelMultiDevice_v9000: {
    isRuntimeApi = true;
    break;
  }
  case CUPTI_DRIVER_TRACE_CBID_cuLaunch:
  case CUPTI_DRIVER_TRACE_CBID_cuLaunchGrid:
  case CUPTI_DRIVER_TRACE_CBID_cuLaunchGridAsync:
  case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel:
  case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz:
  case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx:
  case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz:
  case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel:
  case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz:
  case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernelMultiDevice: {
    isDriverApi = true;
    break;
  }
  default:
    break;
  }
  return std::make_pair(isRuntimeApi, isDriverApi);
}

} // namespace

void CuptiProfiler::callback(void *userData, CUpti_CallbackDomain domain,
                             CUpti_CallbackId cbId, const void *cbData) {
  auto [isRuntimeAPI, isDriverAPI] = matchKernelCbId(cbId);
  if (!(isRuntimeAPI || isDriverAPI)) {
    return;
  }
  CuptiProfiler &profiler =
      dynamic_cast<CuptiProfiler &>(CuptiProfiler::instance());
  const CUpti_CallbackData *callbackData =
      reinterpret_cast<const CUpti_CallbackData *>(cbData);
  if (callbackData->callbackSite == CUPTI_API_ENTER) {
    if (callbackData->context && cuptiState.level == 0) {
      // Valid context and outermost level of the kernel launch
      auto scopeId = Scope::getNewScopeId();
      auto scope = Scope(scopeId, callbackData->symbolName);
      cuptiState.record(scope, profiler.getDataSetSnapshot());
      cuptiState.enterOp();
    }
    cuptiState.level++;
  } else if (callbackData->callbackSite == CUPTI_API_EXIT) {
    cuptiState.level--;
    if (cuptiState.level == 0) {
      if (cuptiState.isRecording) {
        cuptiState.exitOp();
      }
      cuptiState.reset();
    }
  }
}

} // namespace proton
