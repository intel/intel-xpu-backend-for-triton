#include "Profiler/Xpupti/XpuptiProfiler.h"
#include "Context/Context.h"
#include "Data/Metric.h"
#include "Device.h"
#include "Driver/GPU/XpuApi.h"
#include "Driver/GPU/XpuptiApi.h"
#include "Runtime/XpuRuntime.h"
#include "Utility/Env.h"
#include "Utility/Map.h"
#include "Utility/String.h"
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
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>

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

// Phases of GPU operations whose activity record has not been processed yet.
// PTI keeps one activity buffer per submitting thread and hands each one over
// on its own, so a buffer only holding records of later phases does not mean
// that every earlier phase is complete: another thread's buffer may still hold
// records for them. A phase must not be completed (and flushed or cleared)
// before all of its records have been processed.
class PendingPhases {
public:
  using Phases = std::vector<std::pair<Data *, size_t>>;

  void add(uint64_t id, Phases phases) {
    std::lock_guard<std::mutex> lock(mutex);
    addLocked(id, std::move(phases));
  }

  void add(uint64_t id, const DataToEntryMap &dataToEntry) {
    add(id, toPhases(dataToEntry));
  }

  // Track the entries a pending operation was re-correlated to.
  void update(uint64_t id, const DataToEntryMap &dataToEntry) {
    std::lock_guard<std::mutex> lock(mutex);
    if (removeLocked(id))
      addLocked(id, toPhases(dataToEntry));
  }

  void remove(uint64_t id) {
    std::lock_guard<std::mutex> lock(mutex);
    removeLocked(id);
  }

  // Cap each data's end phase at its oldest pending phase, so only phases
  // below it are completed.
  void capEndPhases(DataPhases &dataPhases) const {
    std::lock_guard<std::mutex> lock(mutex);
    for (auto &[data, phases] : dataPhases) {
      auto it = dataToPhaseCounts.find(data);
      if (it != dataToPhaseCounts.end())
        phases.second = std::min(phases.second, it->second.begin()->first);
    }
  }

  void clear() {
    std::lock_guard<std::mutex> lock(mutex);
    idToPhases.clear();
    dataToPhaseCounts.clear();
  }

private:
  static Phases toPhases(const DataToEntryMap &dataToEntry) {
    Phases phases;
    for (const auto &[data, entry] : dataToEntry)
      phases.emplace_back(data, entry.phase);
    return phases;
  }

  // Ids without phases are kept too, so that update() can still tell a
  // pending operation from a processed one.
  void addLocked(uint64_t id, Phases phases) {
    for (const auto &[data, phase] : phases)
      ++dataToPhaseCounts[data][phase];
    idToPhases[id] = std::move(phases);
  }

  bool removeLocked(uint64_t id) {
    auto it = idToPhases.find(id);
    if (it == idToPhases.end())
      return false;
    for (const auto &[data, phase] : it->second) {
      auto &phaseCounts = dataToPhaseCounts[data];
      if (--phaseCounts[phase] == 0)
        phaseCounts.erase(phase);
      if (phaseCounts.empty())
        dataToPhaseCounts.erase(data);
    }
    idToPhases.erase(it);
    return true;
  }

  mutable std::mutex mutex;
  std::unordered_map<uint64_t, Phases> idToPhases;
  std::map<Data *, std::map<size_t, size_t>> dataToPhaseCounts;
};

uint32_t
processActivityKernel(CorrIdToExternIdMap &corrIdToExternId,
                      ExternIdToStateMap &externIdToState,
                      std::map<uint64_t, std::reference_wrapper<ExternIdState>>
                          &externIdToStateCache,
                      DataPhases &dataPhases, xpupti::Pti_Activity *activity) {
  auto *kernel = reinterpret_cast<pti_view_record_kernel *>(activity);
  auto correlationId = kernel->_correlation_id;
  size_t externId = 0;
  if (!/*not valid*/ corrIdToExternId.withRead(
          correlationId, [&externId](size_t value) { externId = value; })) {
    corrIdToExternId.erase(correlationId);
    return correlationId;
  }
  if (true) {
    // Non-graph kernels
    bool isMissingName = false;
    DataToEntryMap dataToEntry;
    externIdToState.withRead(externId, [&](const ExternIdState &state) {
      isMissingName = state.isMissingName;
      dataToEntry = state.dataToEntry;
    });
    if (!isMissingName) {
      for (auto &[data, entry] : dataToEntry) {
        if (auto kernelMetric = convertActivityToMetric(activity)) {
          entry.upsertMetric(std::move(kernelMetric));
          detail::updateDataPhases(dataPhases, data, entry.phase);
        }
      }
    } else {
      for (auto &[data, entry] : dataToEntry) {
        if (auto kernelMetric = convertActivityToMetric(activity)) {
          auto childEntry =
              data->addOp(entry.phase, entry.id, {Context(kernel->_name)});
          childEntry.upsertMetric(std::move(kernelMetric));
          detail::updateDataPhases(dataPhases, data, entry.phase);
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

uint32_t
processActivity(CorrIdToExternIdMap &corrIdToExternId,
                ExternIdToStateMap &externIdToState,
                std::map<uint64_t, std::reference_wrapper<ExternIdState>>
                    &externIdToStateCache,
                DataPhases &dataPhases, PendingPhases &pendingPhases,
                xpupti::Pti_Activity *activity) {
  uint32_t correlationId = 0;
  switch (activity->_view_kind) {
  case PTI_VIEW_DEVICE_GPU_KERNEL: {
    correlationId =
        processActivityKernel(corrIdToExternId, externIdToState,
                              externIdToStateCache, dataPhases, activity);
    pendingPhases.remove(correlationId);
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
    auto runtime = &XpuRuntime::instance();
    profiler.metricBuffer =
        std::make_unique<MetricBuffer>(1024 * 1024 * 64, runtime,
                                       /*mapped=*/true);
    // profiler.pendingGraphPool =
    //    std::make_unique<PendingGraphPool>(profiler.metricBuffer.get());
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

  pti_callback_subscriber_handle subscriber;
  // Keyed by correlation id, from the append until the record is processed.
  PendingPhases pendingPhases;
  // Keyed by scope id, from the start of an op until it stops: an op's entries
  // take their phase when it starts, its kernels only become pending when
  // they are appended.
  PendingPhases pendingOps;

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
  const auto envBufferSize =
      getIntEnv("TRITON_PROFILE_BUFFER_SIZE", 64 * 1024 * 1024);
#if defined(_MSC_VER)
  *buffer = static_cast<uint8_t *>(_aligned_malloc(envBufferSize, AlignSize));
#else
  *buffer = static_cast<uint8_t *>(aligned_alloc(AlignSize, envBufferSize));
#endif
  if (*buffer == nullptr) {
    throw std::runtime_error("aligned_alloc failed");
  }
  *bufferSize = envBufferSize;
}

void XpuptiProfiler::XpuptiProfilerPimpl::completeBuffer(uint8_t *buffer,
                                                         size_t size,
                                                         size_t validSize) {
  XpuptiProfiler &profiler = threadState.profiler;
  auto &pimpl = dynamic_cast<XpuptiProfilerPimpl &>(*profiler.pImpl);
  auto &correlation = profiler.correlation;
  uint32_t maxCorrelationId = 0;
  DataPhases dataPhases;
  pti_result status;
  xpupti::Pti_Activity *activity = nullptr;
  std::map<uint64_t, std::reference_wrapper<ExternIdState>>
      externIdToStateCache;
  do {
    status = xpupti::viewGetNextRecord<false>(buffer, validSize, &activity);
    if (status == pti_result::PTI_SUCCESS) {
      auto correlationId = processActivity(
          profiler.correlation.corrIdToExternId,
          profiler.correlation.externIdToState, externIdToStateCache,
          dataPhases, pimpl.pendingPhases, activity);
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
  // A phase is complete once it has been advanced past and none of its records
  // is pending, even if this buffer holds no record of a later phase. Complete
  // (and flush) it now: Profiler::flush would otherwise mark it complete
  // without it ever being written in periodic flushing mode.
  for (auto &[data, phases] : dataPhases)
    phases.second = data->getPhaseInfo().current;
  pimpl.pendingOps.capEndPhases(dataPhases);
  pimpl.pendingPhases.capEndPhases(dataPhases);
  profiler.flushDataPhases(dataPhases, profiler.pendingGraphPool.get());
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
  auto &pendingPhases =
      dynamic_cast<XpuptiProfilerPimpl &>(*threadState.profiler.pImpl)
          .pendingPhases;
  if (callback_data->_phase == PTI_CB_PHASE_API_ENTER) {
    // TODO: Get kernel name from pti_callback_gpu_op_data
    threadState.enterOp(Scope(""));
    auto &dataToEntry = threadState.dataToEntry;
    auto &scope = threadState.scopeStack.back();
    auto isMissingName = scope.name.empty();
    threadState.profiler.correlation.correlate(callback_data->_correlation_id,
                                               scope.scopeId, 1, isMissingName,
                                               dataToEntry);
    pendingPhases.add(callback_data->_correlation_id, dataToEntry);
  } else if (callback_data->_phase == PTI_CB_PHASE_API_EXIT) {
    // PTI reports the kernel name only on exit: re-enter the unnamed launch op
    // under it, so the kernel's metric lands on the launch op itself.
    const pti_gpu_op_details *op = callback_data->_operation_count > 0
                                       ? callback_data->_operation_details
                                       : nullptr;
    if (op != nullptr && op->_operation_kind == PTI_GPU_OPERATION_KIND_KERNEL &&
        op->_name != nullptr && *op->_name != '\0' &&
        threadState.isApiExternOp &&
        threadState.scopeStack.back().name.empty()) {
      auto scopeId = threadState.scopeStack.back().scopeId;
      threadState.exitOp();
      threadState.enterOp(Scope(scopeId, op->_name));
      threadState.profiler.correlation.correlate(
          callback_data->_correlation_id, scopeId, 1, /*isMissingName=*/false,
          threadState.dataToEntry);
      pendingPhases.update(callback_data->_correlation_id,
                           threadState.dataToEntry);
    }
    // Only records of appended kernels are processed, so a failed append or an
    // operation without a kernel must not hold back the completion of its
    // phase.
    const pti_gpu_op_details *opsEnd =
        op != nullptr ? op + callback_data->_operation_count : nullptr;
    if (callback_data->_return_code != 0 ||
        (op != nullptr &&
         std::none_of(op, opsEnd, [](const pti_gpu_op_details &details) {
           return details._operation_kind == PTI_GPU_OPERATION_KIND_KERNEL;
         })))
      pendingPhases.remove(callback_data->_correlation_id);
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
  if (void *syclQueue = XpuRuntime::instance().getSyclQueue()) {
    callWaitOnSyclQueue(syclQueue);
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
  profiler.periodicFlushingEnabled = false;
  profiler.periodicFlushingFormat.clear();
  pendingPhases.clear();
  pendingOps.clear();
  xpupti::unsubscribe<true>(subscriber);
  // cupti::finalize<true>();
}

XpuptiProfiler::XpuptiProfiler() {
  pImpl = std::make_unique<XpuptiProfilerPimpl>(*this);
}

XpuptiProfiler::~XpuptiProfiler() = default;

// An op entered for a GPU API call becomes pending in the same callback, and a
// kernel's op is re-entered in the current phase when the call returns, so
// only ops that enclose a launch (e.g. from a launch hook) are held back here.
void XpuptiProfiler::startOp(const Scope &scope) {
  if (!threadState.isApiExternOp) {
    // Hold back the current phase before the op's entries are created in it
    // (or in a later one: phases only grow), so it cannot be completed before
    // the op's kernels are pending.
    PendingPhases::Phases phases;
    for (auto *data : dataSet)
      phases.emplace_back(data, data->getPhaseInfo().current);
    dynamic_cast<XpuptiProfilerPimpl &>(*pImpl).pendingOps.add(
        scope.scopeId, std::move(phases));
  }
  GPUProfiler<XpuptiProfiler>::startOp(scope);
}

void XpuptiProfiler::stopOp(const Scope &scope) {
  if (!threadState.isApiExternOp)
    dynamic_cast<XpuptiProfilerPimpl &>(*pImpl).pendingOps.remove(
        threadState.scopeStack.back().scopeId);
  GPUProfiler<XpuptiProfiler>::stopOp(scope);
}

void XpuptiProfiler::doSetMode(const std::vector<std::string> &modeAndOptions) {
  auto mode = modeAndOptions[0];
  if (proton::toLower(mode) == "periodic_flushing") {
    detail::setPeriodicFlushingMode(periodicFlushingEnabled,
                                    periodicFlushingFormat, modeAndOptions,
                                    "XpuptiProfiler");
  } else if (!mode.empty()) {
    throw std::invalid_argument("[PROTON] XpuptiProfiler: unsupported mode: " +
                                mode);
  }
}

} // namespace proton
