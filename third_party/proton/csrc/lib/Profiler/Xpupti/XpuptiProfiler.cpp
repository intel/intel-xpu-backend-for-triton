#include "Profiler/Xpupti/XpuptiProfiler.h"
#include "Context/Context.h"
#include "Data/Metric.h"
#include "Device.h"
#include "Driver/GPU/XpuApi.h"
#include "Driver/GPU/XpuptiApi.h"
#include "Utility/Map.h"

#include "pti/pti_view.h"
#include <cassert>
#include <cstring>

#include <algorithm>
#include <array>
#include <dlfcn.h>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace proton {

template <>
thread_local GPUProfiler<XpuptiProfiler>::ThreadState
    GPUProfiler<XpuptiProfiler>::threadState(XpuptiProfiler::instance());

template <>
thread_local std::deque<size_t>
    GPUProfiler<XpuptiProfiler>::Correlation::externIdQueue{};

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

std::shared_ptr<Metric>
convertActivityToMetric(xpupti::Pti_Activity *activity) {
  std::shared_ptr<Metric> metric;
  switch (activity->_view_kind) {
  case PTI_VIEW_DEVICE_GPU_KERNEL: {
    auto *kernel = reinterpret_cast<pti_view_record_kernel *>(activity);
    if (kernel->_start_timestamp < kernel->_end_timestamp) {
      metric = std::make_shared<KernelMetric>(
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

uint32_t
processActivityKernel(XpuptiProfiler::CorrIdToExternIdMap &corrIdToExternId,
                      XpuptiProfiler::ApiExternIdSet &apiExternIds,
                      std::set<Data *> &dataSet,
                      xpupti::Pti_Activity *activity) {
  auto *kernel = reinterpret_cast<pti_view_record_kernel *>(activity);
  auto correlationId = kernel->_correlation_id;
  if (/*Not a valid context*/ !corrIdToExternId.contain(correlationId)) {
    return correlationId;
  }
  auto [parentId, numInstances] = corrIdToExternId.at(correlationId);
  if (true) {
    // Non-graph kernels
    for (auto *data : dataSet) {
      auto scopeId = parentId;
      if (apiExternIds.contain(scopeId)) {
        // It's triggered by a CUDA op but not triton op
        scopeId = data->addOp(parentId, kernel->_name);
      }
      data->addMetric(scopeId, convertActivityToMetric(activity));
    }
  } else {
    // Graph kernels
    // A single graph launch can trigger multiple kernels.
    // Our solution is to construct the following maps:
    // --- Application threads ---
    // 1. graphId -> numKernels
    // 2. graphExecId -> graphId
    // --- CUPTI thread ---
    // 3. corrId -> numKernels
    for (auto *data : dataSet) {
      auto externId = data->addOp(parentId, kernel->_name);
      data->addMetric(externId, convertActivityToMetric(activity));
    }
  }
  apiExternIds.erase(parentId);
  --numInstances;
  if (numInstances == 0) {
    corrIdToExternId.erase(correlationId);
  } else {
    corrIdToExternId[correlationId].second = numInstances;
  }
  return correlationId;
}

uint32_t processActivity(XpuptiProfiler::CorrIdToExternIdMap &corrIdToExternId,
                         XpuptiProfiler::ApiExternIdSet &apiExternIds,
                         std::set<Data *> &dataSet,
                         xpupti::Pti_Activity *activity) {
  auto correlationId = 0;
  switch (activity->_view_kind) {
  case PTI_VIEW_DEVICE_GPU_KERNEL: {
    correlationId = processActivityKernel(corrIdToExternId, apiExternIds,
                                          dataSet, activity);
    break;
  }
  default:
    break;
  }
  return correlationId;
}

} // namespace

#include <cxxabi.h>

static inline std::string Demangle(const char *name) {

  int status = 0;
  char *demangled = abi::__cxa_demangle(name, nullptr, 0, &status);
  if (status != 0) {
    return name;
  }

  constexpr const char *const prefix_to_skip = "typeinfo name for ";
  const size_t prefix_to_skip_len = strlen(prefix_to_skip);
  const size_t shift =
      (std::strncmp(demangled, prefix_to_skip, prefix_to_skip_len) == 0)
          ? prefix_to_skip_len
          : 0;

  std::string result(demangled + shift);
  free(demangled);
  return result;
}

struct XpuptiProfiler::XpuptiProfilerPimpl
    : public GPUProfiler<XpuptiProfiler>::GPUProfilerPimplInterface {
  XpuptiProfilerPimpl(XpuptiProfiler &profiler)
      : GPUProfiler<XpuptiProfiler>::GPUProfilerPimplInterface(profiler) {}
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
  *buffer = static_cast<uint8_t *>(aligned_alloc(AlignSize, BufferSize));
  if (*buffer == nullptr) {
    throw std::runtime_error("aligned_alloc failed");
  }
  *bufferSize = BufferSize;
}

void XpuptiProfiler::XpuptiProfilerPimpl::completeBuffer(uint8_t *buffer,
                                                         size_t size,
                                                         size_t validSize) {
  XpuptiProfiler &profiler = threadState.profiler;
  auto &dataSet = profiler.dataSet;
  auto &correlation = profiler.correlation;
  uint32_t maxCorrelationId = 0;
  pti_result status;
  xpupti::Pti_Activity *activity = nullptr;
  do {
    status = xpupti::viewGetNextRecord<true>(buffer, validSize, &activity);
    if (status == pti_result::PTI_SUCCESS) {
      auto correlationId =
          processActivity(profiler.correlation.corrIdToExternId,
                          profiler.correlation.apiExternIds, dataSet, activity);
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

  std::free(buffer);

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
    threadState.enterOp();
    threadState.profiler.correlation.correlate(callback_data->_correlation_id,
                                               1);
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

typedef void (*WaitOnSyclQueueFunc)(void *);

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
      /*maxRetries=*/100, /*sleepMs=*/10,
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
