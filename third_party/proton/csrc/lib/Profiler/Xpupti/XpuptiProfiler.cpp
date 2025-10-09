#include "Profiler/Xpupti/XpuptiProfiler.h"
#include "Context/Context.h"
#include "Data/Metric.h"
#include "Device.h"
// #include "Driver/GPU/CudaApi.h"
#include "Driver/GPU/XpuptiApi.h"
#include "Utility/Map.h"

#include "pti/pti_view.h"
#include <cassert>
#include <cstring>
#include <level_zero/layers/zel_tracing_api.h>
#include <level_zero/zet_api.h>

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
  // std::cout << "activity->_name: " << kernel->_name << "\n" << std::flush;
  // std::cout << "activity->_sycl_queue_id: " << kernel->_sycl_queue_id << "\n"
  // << std::flush;
  auto correlationId = kernel->_correlation_id;
  std::cout << "kernel->_correlation_id " << kernel->_correlation_id << "\n"
            << std::flush;
  std::cout << "kernel->_kernel_id " << kernel->_kernel_id << "\n";
  // here doesn't work
  // uint64_t corr_id = 0;
  // auto res =
  // ptiViewPopExternalCorrelationId(pti_view_external_kind::PTI_VIEW_EXTERNAL_KIND_CUSTOM_1,
  // &corr_id); std::cout << "ptiViewPopExternalCorrelationId res: " << res <<
  // "\n" << std::flush; std::cout << "corr_id: " << corr_id << "\n" <<
  // std::flush;
  if (/*Not a valid context*/ !corrIdToExternId.contain(correlationId)) {
    // if (false) {
    std::cout << "MARK#3\n" << std::flush;
    return correlationId;
  }
  auto [parentId, numInstances] = corrIdToExternId.at(correlationId);
  std::cout << "parentId: " << parentId << std::endl;
  if (true) {
    // Non-graph kernels
    for (auto *data : dataSet) {
      auto scopeId = parentId;
      if (apiExternIds.contain(scopeId)) {
        std::cout << "first branch" << std::endl;
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
    std::cout << "MARK#1\n" << std::flush;
    for (auto *data : dataSet) {
      auto externId = data->addOp(parentId, kernel->_name);
      std::cout << "MARK#2\n" << std::flush;
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

uint32_t processActivityExternalCorrelation(
    XpuptiProfiler::CorrIdToExternIdMap &corrIdToExternId,
    xpupti::Pti_Activity *activity) {
  auto *externalActivity =
      reinterpret_cast<pti_view_record_external_correlation *>(activity);
  std::cout << "processActivityExternalCorrelation: _correlation_id: "
            << externalActivity->_correlation_id << "\n";
  std::cout << "processActivityExternalCorrelation: _external_id: "
            << externalActivity->_external_id << "\n";

  // corrIdToExternId[externalActivity->_correlation_id] =
  // {externalActivity->_external_id, 1};
  return externalActivity->_correlation_id;
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
  case PTI_VIEW_EXTERNAL_CORRELATION: {
    // correlationId = processActivityExternalCorrelation(corrIdToExternId,
    // activity);
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

  static void OnEnterCommandListAppendLaunchKernel(
      ze_command_list_append_launch_kernel_params_t *params, ze_result_t result,
      void *global_user_data, void **instance_user_data) {
    std::cout << "Function zeCommandListAppendLaunchKernel is called on enter"
              << std::endl;
    ze_kernel_handle_t kernel = *(params->phKernel);

    size_t size = 0;
    ze_result_t status = zeKernelGetName(kernel, &size, nullptr);
    assert(status == ZE_RESULT_SUCCESS);

    std::vector<char> name(size);
    status = zeKernelGetName(kernel, &size, name.data());
    assert(status == ZE_RESULT_SUCCESS);
    std::string str(name.begin(), name.end());
    std::cout << "OnEnterCommandListAppendLaunchKernel::demangled kernel_name: "
              << Demangle(name.data()) << "\n";

    threadState.enterOp();

    size_t numInstances = 1;
    // FIXME: 4 - debug value
    uint32_t correlationId = 4;
    threadState.profiler.correlation.correlate(correlationId, numInstances);
  }

  static void OnEnterCommandListAppendLaunchCooperativeKernel(
      ze_command_list_append_launch_cooperative_kernel_params_t *params,
      ze_result_t result, void *global_user_data, void **instance_user_data) {
    std::cout << "Function zeCommandListAppendLaunchKernel is called on enter"
              << std::endl;
    threadState.enterOp();
    // FIXME: 4 - debug value
    threadState.profiler.correlation.correlate(4, 1);
  }

  static void OnExitCommandListAppendLaunchKernel(
      ze_command_list_append_launch_kernel_params_t *params, ze_result_t result,
      void *global_user_data, void **instance_user_data) {
    std::cout << "Function zeCommandListAppendLaunchKernel is called on exit"
              << std::endl;
    threadState.exitOp();
    // Track outstanding op for flush
    // FIXME: 4 - debug value
    uint32_t correlationId = 4;
    threadState.profiler.correlation.submit(correlationId);
    // here works
    // uint64_t corr_id = 0;
    // auto res =
    // ptiViewPopExternalCorrelationId(pti_view_external_kind::PTI_VIEW_EXTERNAL_KIND_CUSTOM_1,
    // &corr_id); std::cout << "ptiViewPopExternalCorrelationId res: " << res <<
    // "\n" << std::flush; std::cout << "ptiViewPopExternalCorrelationId
    // corr_id: " << corr_id << "\n";
  }

  static void allocBuffer(uint8_t **buffer, size_t *bufferSize);
  static void completeBuffer(uint8_t *buffer, size_t size, size_t validSize);
  /*
  static void callbackFn(void *userData, CUpti_CallbackDomain domain,
                         CUpti_CallbackId cbId, const void *cbData);
  */

  static constexpr size_t AlignSize = 8;
  static constexpr size_t BufferSize = 64 * 1024 * 1024;

  /*
  static constexpr size_t AttributeSize = sizeof(size_t);

  CUpti_SubscriberHandle subscriber{};
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
      std::cout << "activity->_view_kind: " << activity->_view_kind << "\n"
                << std::flush;
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

zel_tracer_handle_t tracer = nullptr;

typedef void (*EnumDeviceUUIDsFunc)(std::vector<std::array<uint8_t, 16>>);

int callEnumDeviceUUIDs(const std::string &utils_cache_path) {
  void *handle = dlopen(utils_cache_path.data(), RTLD_LAZY);
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

  enumDeviceUUIDs(deviceUUIDs_);

  dlclose(handle);
  return 0;
}

typedef void (*WaitOnSyclQueueFunc)(void *);

int callWaitOnSyclQueue(const std::string &utils_cache_path, void *syclQueue) {
  void *handle = dlopen(utils_cache_path.data(), RTLD_LAZY);
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
  // xpupti::subscribe<true>(&subscriber, callbackFn, nullptr);
  // should be call to shared lib
  XpuptiProfiler &profiler = threadState.profiler;
  if (profiler.utils_cache_path != "") {
    callEnumDeviceUUIDs(profiler.utils_cache_path);
  }
  // auto res = ptiViewPushExternalCorrelationId(
  //     pti_view_external_kind::PTI_VIEW_EXTERNAL_KIND_CUSTOM_1, 42);
  //  std::cout << "res: " << res << "\n" << std::flush;

  ze_result_t status = ZE_RESULT_SUCCESS;
  // status = zeInit(ZE_INIT_FLAG_GPU_ONLY);
  // assert(status == ZE_RESULT_SUCCESS);

  zel_tracer_desc_t tracer_desc = {ZEL_STRUCTURE_TYPE_TRACER_DESC, nullptr,
                                   nullptr /* global user data */};

  status = zelTracerCreate(&tracer_desc, &tracer);
  std::cout << "zelTracerCreate: " << status << "\n" << std::flush;
  assert(status == ZE_RESULT_SUCCESS);

  zet_core_callbacks_t prologue_callbacks = {};
  zet_core_callbacks_t epilogue_callbacks = {};
  prologue_callbacks.CommandList.pfnAppendLaunchKernelCb =
      OnEnterCommandListAppendLaunchKernel;
  // prologue_callbacks.CommandList.pfnAppendLaunchCooperativeKernelCb =
  // OnEnterCommandListAppendLaunchCooperativeKernel;
  epilogue_callbacks.CommandList.pfnAppendLaunchKernelCb =
      OnExitCommandListAppendLaunchKernel;

  status = zelTracerSetPrologues(tracer, &prologue_callbacks);
  assert(status == ZE_RESULT_SUCCESS);
  status = zelTracerSetEpilogues(tracer, &epilogue_callbacks);
  assert(status == ZE_RESULT_SUCCESS);

  status = zelTracerSetEnabled(tracer, true);
  assert(status == ZE_RESULT_SUCCESS);

  xpupti::viewSetCallbacks<true>(allocBuffer, completeBuffer);
  xpupti::viewEnable<true>(PTI_VIEW_DEVICE_GPU_KERNEL);
  // xpupti::viewEnable<true>(PTI_VIEW_DEVICE_GPU_MEM_COPY);
  // xpupti::viewEnable<true>(PTI_VIEW_DEVICE_GPU_MEM_FILL);
  // xpupti::viewEnable<true>(PTI_VIEW_SYCL_RUNTIME_CALLS);
  // xpupti::viewEnable<true>(PTI_VIEW_COLLECTION_OVERHEAD);
  // xpupti::viewEnable<true>(PTI_VIEW_EXTERNAL_CORRELATION);
  // xpupti::viewEnable<true>(PTI_VIEW_LEVEL_ZERO_CALLS);
  // setGraphCallbacks(subscriber, /*enable=*/true);
  // setRuntimeCallbacks(subscriber, /*enable=*/true);
  // setDriverCallbacks(subscriber, /*enable=*/true);
}

void XpuptiProfiler::XpuptiProfilerPimpl::doFlush() {
  std::cout << "flush\n" << std::flush;
  XpuptiProfiler &profiler = threadState.profiler;
  if (profiler.syclQueue != nullptr) {
    callWaitOnSyclQueue(profiler.utils_cache_path, profiler.syclQueue);
  }

  profiler.correlation.flush(
      /*maxRetries=*/100, /*sleepMs=*/10,
      /*flush=*/[]() { xpupti::viewFlushAll<true>(); });
}

void XpuptiProfiler::XpuptiProfilerPimpl::doStop() {
  ze_result_t status = ZE_RESULT_SUCCESS;
  status = zelTracerSetEnabled(tracer, false);
  assert(status == ZE_RESULT_SUCCESS);
  status = zelTracerDestroy(tracer);
  assert(status == ZE_RESULT_SUCCESS);

  xpupti::viewDisable<true>(PTI_VIEW_DEVICE_GPU_KERNEL);
  // xpupti::viewDisable<true>(PTI_VIEW_DEVICE_GPU_MEM_COPY);
  // xpupti::viewDisable<true>(PTI_VIEW_DEVICE_GPU_MEM_FILL);
  // xpupti::viewDisable<true>(PTI_VIEW_SYCL_RUNTIME_CALLS);
  // xpupti::viewDisable<true>(PTI_VIEW_COLLECTION_OVERHEAD);
  // xpupti::viewDisable<true>(PTI_VIEW_EXTERNAL_CORRELATION);
  // xpupti::viewDisable<true>(PTI_VIEW_LEVEL_ZERO_CALLS);
  // setGraphCallbacks(subscriber, /*enable=*/false);
  // setRuntimeCallbacks(subscriber, /*enable=*/false);
  // setDriverCallbacks(subscriber, /*enable=*/false);
  // cupti::unsubscribe<true>(subscriber);
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
