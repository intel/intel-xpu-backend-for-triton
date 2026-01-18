#include "Profiler/Xpupti/XpuptiProfiler.h"
#include "Context/Context.h"
#include "Data/Metric.h"
#include "Device.h"
#include "Driver/GPU/XpuApi.h"
#include "Driver/GPU/XpuptiApi.h"
#include "Runtime/XpuRuntime.h"
#include "Utility/Map.h"
#include <vector>

#include "pti/pti_view.h"
#include <cassert>
#include <cstring>

#include <algorithm>
#include <array>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace proton {

template <>
thread_local GPUProfiler<XpuptiProfiler>::ThreadState
    GPUProfiler<XpuptiProfiler>::threadState(XpuptiProfiler::instance());

namespace {

std::vector<std::array<uint8_t, 16>> deviceUUIDs_ = {};

uint8_t getDeviceIdxFromUUID(const uint8_t deviceUUID[16]) {
  std::array<unsigned char, 16> key;
  memcpy(key.data(), deviceUUID, 16);
  auto it = std::find(deviceUUIDs_.begin(), deviceUUIDs_.end(), key);
  if (it == deviceUUIDs_.end()) {
    std::cerr
        << "Warnings: Can't find the legal XPU device from the given UUID."
        << std::endl;
    return static_cast<uint8_t>(0);
  }
  return static_cast<uint8_t>(std::distance(deviceUUIDs_.begin(), it));
}

std::unique_ptr<Metric>
convertActivityToMetric(xpupti::Pti_Activity *activity) {
  std::unique_ptr<Metric> metric;
  switch (activity->_view_kind) {
  case PTI_VIEW_DEVICE_GPU_KERNEL: {
    auto *kernel = reinterpret_cast<pti_view_record_kernel *>(activity);
    if (kernel->_start_timestamp < kernel->_end_timestamp) {
      metric = std::make_unique<KernelMetric>(
          static_cast<uint64_t>(kernel->_start_timestamp),
          static_cast<uint64_t>(kernel->_end_timestamp), 1,
          static_cast<uint64_t>(getDeviceIdxFromUUID(kernel->_device_uuid)),
          static_cast<uint64_t>(DeviceType::XPU),
          static_cast<uint64_t>(kernel->_sycl_queue_id));
    } // else: not a valid kernel activity
    break;
  }
  default:
    break;
  }
  return metric;
}

uint32_t processActivityKernel(
    XpuptiProfiler::CorrIdToExternIdMap &corrIdToExternId,
    XpuptiProfiler::ExternIdToStateMap &externIdToState,
    std::map<uint64_t, std::reference_wrapper<XpuptiProfiler::ExternIdState>>
        &externIdToStateCache,
    xpupti::Pti_Activity *activity) {
  auto *kernel = reinterpret_cast<pti_view_record_kernel *>(activity);
  auto correlationId = kernel->_correlation_id;
  size_t externId = 0;
  if (!/*not valid*/ corrIdToExternId.withRead(
          correlationId, [&externId](size_t value) { externId = value; })) {
    corrIdToExternId.erase(correlationId);
  }
  if (true) {
    // Non-graph kernels
    bool isMissingName = false;
    DataToEntryMap dataToEntry;
    externIdToState.withRead(externId,
                             [&](const XpuptiProfiler::ExternIdState &state) {
                               isMissingName = state.isMissingName;
                               dataToEntry = state.dataToEntry;
                             });
    if (!isMissingName) {
      for (auto &[data, entry] : dataToEntry) {
        if (auto kernelMetric = convertActivityToMetric(activity)) {
          entry.upsertMetric(std::move(kernelMetric));
        }
      }
    } else {
      for (auto &[data, entry] : dataToEntry) {
        if (auto kernelMetric = convertActivityToMetric(activity)) {
          auto childEntry = data->addOp(entry.id, {Context(kernel->_name)});
          childEntry.upsertMetric(std::move(kernelMetric));
        }
      }
    }
    externIdToState.erase(externId);
    corrIdToExternId.erase(correlationId);
  } else {
    // Graph kernels
    // A single graph launch can trigger multiple kernels.
    // Our solution is to construct the following maps:
    // --- Application threads ---
    // If graph creation has been captured:
    // - parentId, nodeId -> launch context + capture context
    // Otherwise:
    // - parentId -> launch context
    // --- CUPTI thread ---
    // - corrId -> numNodes
    // FIXME: enable it for XPU
  }
  return correlationId;
}

uint32_t processActivity(
    XpuptiProfiler::CorrIdToExternIdMap &corrIdToExternId,
    XpuptiProfiler::ExternIdToStateMap &externIdToState,
    std::map<uint64_t, std::reference_wrapper<XpuptiProfiler::ExternIdState>>
        &externIdToStateCache,
    xpupti::Pti_Activity *activity) {
  auto correlationId = 0;
  switch (activity->_view_kind) {
  case PTI_VIEW_DEVICE_GPU_KERNEL: {
    correlationId = processActivityKernel(corrIdToExternId, externIdToState,
                                          externIdToStateCache, activity);
    break;
  }
  default:
    break;
  }
  return correlationId;
}

} // namespace

struct XpuptiProfiler::XpuptiProfilerPimpl
    : public GPUProfiler<XpuptiProfiler>::GPUProfilerPimplInterface {
  XpuptiProfilerPimpl(XpuptiProfiler &profiler)
      : GPUProfiler<XpuptiProfiler>::GPUProfilerPimplInterface(profiler) {
    // FIXME: enable metrics
    // runtime = &XpuRuntime::instance();
    // metricBuffer = std::make_unique<MetricBuffer>(1024 * 1024 * 64, runtime);
  }
  virtual ~XpuptiProfilerPimpl() = default;

  void doStart() override;
  void doFlush() override;
  void doStop() override;

  static uint32_t get_correlation_id(xpupti::Pti_Activity *activity);

  static void allocBuffer(uint8_t **buffer, size_t *bufferSize);
  static void completeBuffer(uint8_t *buffer, size_t size, size_t validSize);
  static void callbackFn(pti_callback_domain domain,
                         pti_api_group_id driver_api_group_id,
                         uint32_t driver_api_id,
                         pti_backend_ctx_t backend_context, void *cb_data,
                         void *global_user_data, void **instance_user_data);

  static constexpr size_t AlignSize = 8;
  static constexpr size_t BufferSize = 64 * 1024 * 1024;

  pti_callback_subscriber_handle subscriber;

  /*
  CuptiPCSampling pcSampling;

  ThreadSafeMap<uint32_t, size_t, std::unordered_map<uint32_t, size_t>>
      graphIdToNumInstances;
  ThreadSafeMap<uint32_t, uint32_t, std::unordered_map<uint32_t, uint32_t>>
      graphExecIdToGraphId;
  */
};

void XpuptiProfiler::XpuptiProfilerPimpl::allocBuffer(uint8_t **buffer,
                                                      size_t *bufferSize) {
#if defined(_MSC_VER)
  *buffer = static_cast<uint8_t *>(_aligned_malloc(BufferSize, AlignSize));
#else
  *buffer = static_cast<uint8_t *>(aligned_alloc(AlignSize, BufferSize));
#endif
  if (*buffer == nullptr) {
    throw std::runtime_error("aligned_alloc failed");
  }
  *bufferSize = BufferSize;
}

void XpuptiProfiler::XpuptiProfilerPimpl::completeBuffer(uint8_t *buffer,
                                                         size_t size,
                                                         size_t validSize) {
  XpuptiProfiler &profiler = threadState.profiler;
  auto &correlation = profiler.correlation;
  uint32_t maxCorrelationId = 0;
  pti_result status;
  xpupti::Pti_Activity *activity = nullptr;
  std::map<uint64_t, std::reference_wrapper<XpuptiProfiler::ExternIdState>>
      externIdToStateCache;
  do {
    status = xpupti::viewGetNextRecord<true>(buffer, validSize, &activity);
    if (status == pti_result::PTI_SUCCESS) {
      auto correlationId = processActivity(
          profiler.correlation.corrIdToExternId,
          profiler.correlation.externIdToState, externIdToStateCache, activity);
      // Log latest completed correlation id.  Used to ensure we have flushed
      // all data on stop
      maxCorrelationId = std::max<uint64_t>(maxCorrelationId, correlationId);
    } else if (status == pti_result::PTI_STATUS_END_OF_BUFFER) {
      std::cout << "Reached End of buffer" << '\n';
      break;
    } else {
      throw std::runtime_error("xpupti::activityGetNextRecord failed");
    }
  } while (true);

#if defined(_MSC_VER)
  _aligned_free(buffer);
#else
  std::free(buffer);
#endif

  profiler.correlation.complete(maxCorrelationId);
}

void XpuptiProfiler::XpuptiProfilerPimpl::callbackFn(
    pti_callback_domain domain, pti_api_group_id driver_api_group_id,
    uint32_t driver_api_id, pti_backend_ctx_t backend_context, void *cb_data,
    void *global_user_data, void **instance_user_data) {
  pti_callback_gpu_op_data *callback_data =
      static_cast<pti_callback_gpu_op_data *>(cb_data);
  if (callback_data == nullptr) {
    std::cerr << "CallbackGPUOperationAppend: callback_data is null"
              << std::endl;
    return;
  }
  if (callback_data->_phase == PTI_CB_PHASE_API_ENTER) {
    // TODO: Get kernel name from pti_callback_gpu_op_data
    threadState.enterOp(Scope(""));
    auto &dataToEntry = threadState.dataToEntry;
    auto &scope = threadState.scopeStack.back();
    auto isMissingName = scope.name.empty();
    threadState.profiler.correlation.correlate(callback_data->_correlation_id,
                                               scope.scopeId, 1, isMissingName,
                                               dataToEntry);
  } else if (callback_data->_phase == PTI_CB_PHASE_API_EXIT) {
    threadState.exitOp();
    threadState.profiler.correlation.submit(callback_data->_correlation_id);
  } else {
    throw std::runtime_error("[PROTON] callbackFn failed");
  }
}

void CallbackCommon(pti_callback_domain domain,
                    pti_api_group_id driver_group_id, uint32_t driver_api_id,
                    [[maybe_unused]] pti_backend_ctx_t backend_context,
                    [[maybe_unused]] void *cb_data,
                    [[maybe_unused]] void *user_data) {

  switch (domain) {
  case PTI_CB_DOMAIN_DRIVER_GPU_OPERATION_APPENDED:
    std::cout << "PTI_CB_DOMAIN_DRIVER_GPU_OPERATION_APPENDED\n" << std::flush;
    break;
  case PTI_CB_DOMAIN_DRIVER_GPU_OPERATION_COMPLETED:
    std::cout << "PTI_CB_DOMAIN_DRIVER_GPU_OPERATION_COMPLETED\n" << std::flush;
    break;
  default: {
    std::cout << "In " << __func__ << ", domain: " << domain
              << ", driver_group_id: " << driver_group_id
              << ", driver_api_id: " << driver_api_id << std::endl;
    break;
  }
  }
  std::cout << std::endl;
}

typedef void (*EnumDeviceUUIDsFunc)(void *);

#if defined(_WIN32)
int callEnumDeviceUUIDs(const std::string &utils_cache_path) {
  HMODULE handle = LoadLibrary(xpu::PROTON_UTILS.data());
  if (!handle) {
    std::cerr << "Failed to load library: " << GetLastError() << std::endl;
    return 1;
  }

  GetLastError();
  EnumDeviceUUIDsFunc enumDeviceUUIDs =
      (EnumDeviceUUIDsFunc)GetProcAddress(handle, "enumDeviceUUIDs");
  long dlsym_error = GetLastError();
  if (dlsym_error) {
    std::cerr << "Failed to load function: " << dlsym_error << std::endl;
    FreeLibrary(handle);
    return 1;
  }

  enumDeviceUUIDs(&deviceUUIDs_);

  FreeLibrary(handle);
  return 0;
}
#else
int callEnumDeviceUUIDs(const std::string &utils_cache_path) {
  void *handle = dlopen(xpu::PROTON_UTILS.data(), RTLD_LAZY);
  if (!handle) {
    std::cerr << "Failed to load library: " << dlerror() << std::endl;
    return 1;
  }

  dlerror();
  EnumDeviceUUIDsFunc enumDeviceUUIDs =
      (EnumDeviceUUIDsFunc)dlsym(handle, "enumDeviceUUIDs");
  const char *dlsym_error = dlerror();
  if (dlsym_error) {
    std::cerr << "Failed to load function: " << dlsym_error << std::endl;
    dlclose(handle);
    return 1;
  }

  enumDeviceUUIDs(&deviceUUIDs_);

  dlclose(handle);
  return 0;
}
#endif

typedef void (*WaitOnSyclQueueFunc)(void *);

#if defined(_WIN32)
int callWaitOnSyclQueue(void *syclQueue) {
  HMODULE handle = LoadLibrary(xpu::PROTON_UTILS.data());
  if (!handle) {
    std::cerr << "Failed to load library: " << GetLastError() << std::endl;
    return 1;
  }

  GetLastError();
  WaitOnSyclQueueFunc waitOnSyclQueue =
      (WaitOnSyclQueueFunc)GetProcAddress(handle, "waitOnSyclQueue");
  long dlsym_error = GetLastError();
  if (dlsym_error) {
    std::cerr << "Failed to load function: " << dlsym_error << std::endl;
    FreeLibrary(handle);
    return 1;
  }

  waitOnSyclQueue(syclQueue);

  FreeLibrary(handle);
  return 0;
}
#else
int callWaitOnSyclQueue(void *syclQueue) {
  void *handle = dlopen(xpu::PROTON_UTILS.data(), RTLD_LAZY);
  if (!handle) {
    std::cerr << "Failed to load library: " << dlerror() << std::endl;
    return 1;
  }

  dlerror();
  WaitOnSyclQueueFunc waitOnSyclQueue =
      (WaitOnSyclQueueFunc)dlsym(handle, "waitOnSyclQueue");
  const char *dlsym_error = dlerror();
  if (dlsym_error) {
    std::cerr << "Failed to load function: " << dlsym_error << std::endl;
    dlclose(handle);
    return 1;
  }

  waitOnSyclQueue(syclQueue);

  dlclose(handle);
  return 0;
}
#endif

void XpuptiProfiler::XpuptiProfilerPimpl::doStart() {
  // should be call to shared lib
  XpuptiProfiler &profiler = threadState.profiler;
  if (xpu::PROTON_UTILS != "") {
    callEnumDeviceUUIDs(xpu::PROTON_UTILS);
  }

  xpupti::viewSetCallbacks<true>(allocBuffer, completeBuffer);
  xpupti::viewEnable<true>(PTI_VIEW_DEVICE_GPU_KERNEL);
  xpupti::viewEnable<true>(PTI_VIEW_DEVICE_GPU_MEM_FILL);
  xpupti::viewEnable<true>(PTI_VIEW_DEVICE_GPU_MEM_COPY);
  xpupti::subscribe<true>(&subscriber, callbackFn, &subscriber);
  // xpupti::viewEnable<true>(PTI_VIEW_SYCL_RUNTIME_CALLS);
  // xpupti::viewEnable<true>(PTI_VIEW_LEVEL_ZERO_CALLS);
  // setGraphCallbacks(subscriber, /*enable=*/true);
  // setRuntimeCallbacks(subscriber, /*enable=*/true);
  xpupti::enableDomain<true>(subscriber,
                             PTI_CB_DOMAIN_DRIVER_GPU_OPERATION_APPENDED, 1, 1);
  // setDriverCallbacks(subscriber, /*enable=*/true);
}

void XpuptiProfiler::XpuptiProfilerPimpl::doFlush() {
  XpuptiProfiler &profiler = threadState.profiler;
  if (profiler.syclQueue != nullptr) {
    callWaitOnSyclQueue(profiler.syclQueue);
  }

  profiler.correlation.flush(
      /*maxRetries=*/100, /*sleepUs=*/10,
      /*flush=*/[]() { xpupti::viewFlushAll<true>(); });
}

void XpuptiProfiler::XpuptiProfilerPimpl::doStop() {
  xpupti::viewDisable<true>(PTI_VIEW_DEVICE_GPU_KERNEL);
  xpupti::viewDisable<true>(PTI_VIEW_DEVICE_GPU_MEM_FILL);
  xpupti::viewDisable<true>(PTI_VIEW_DEVICE_GPU_MEM_COPY);
  // xpupti::viewDisable<true>(PTI_VIEW_SYCL_RUNTIME_CALLS);
  // xpupti::viewDisable<true>(PTI_VIEW_LEVEL_ZERO_CALLS);
  // setGraphCallbacks(subscriber, /*enable=*/false);
  // setRuntimeCallbacks(subscriber, /*enable=*/false);
  // setDriverCallbacks(subscriber, /*enable=*/false);
  xpupti::disableDomain<true>(subscriber,
                              PTI_CB_DOMAIN_DRIVER_GPU_OPERATION_APPENDED);
  xpupti::unsubscribe<true>(subscriber);
  // cupti::finalize<true>();
}

XpuptiProfiler::XpuptiProfiler() {
  pImpl = std::make_unique<XpuptiProfilerPimpl>(*this);
}

XpuptiProfiler::~XpuptiProfiler() = default;

void XpuptiProfiler::doSetMode(const std::vector<std::string> &modeAndOptions) {
  auto mode = modeAndOptions[0];
  if (!mode.empty()) {
    throw std::invalid_argument("[PROTON] XpuptiProfiler: unsupported mode: " +
                                mode);
  }
}

} // namespace proton
