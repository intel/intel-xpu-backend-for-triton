#include "Profiler/Xpupti/XpuptiProfiler.h"
#include "Context/Context.h"
#include "Data/Metric.h"
#include "Driver/Device.h"
// #include "Driver/GPU/CudaApi.h"
#include "Driver/GPU/XpuptiApi.h"
#include "Utility/Map.h"

#include <pti/pti_view.h>
#include <sycl/sycl.hpp>

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

// FIXME: Should it be in DeviceInfo class?
// Inspired by Kineto: `XpuptiActivityProfiler.cpp`
void enumDeviceUUIDs() {
  if (!deviceUUIDs_.empty()) {
    return;
  }
  auto platform_list = sycl::platform::get_platforms();
  // Enumerated GPU devices from the specific platform.
  for (const auto &platform : platform_list) {
    if (platform.get_backend() != sycl::backend::ext_oneapi_level_zero) {
      continue;
    }
    auto device_list = platform.get_devices();
    for (const auto &device : device_list) {
      if (device.is_gpu()) {
        if (device.has(sycl::aspect::ext_intel_device_info_uuid)) {
          deviceUUIDs_.push_back(
              device.get_info<sycl::ext::intel::info::device::uuid>());
        } else {
          std::cerr << "Warnings: UUID is not supported for this XPU device. "
                       "The device index of records will be 0."
                    << std::endl;
          deviceUUIDs_.push_back(std::array<uint8_t, 16>{});
        }
      }
    }
  }
}

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
          static_cast<uint64_t>(DeviceType::CUDA));
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
  if (/*Not a valid context*/ !corrIdToExternId.contain(correlationId))
    return correlationId;
  auto [parentId, numInstances] = corrIdToExternId.at(correlationId);
  // Best guess for now: _sycl_queue_id ~ graphId CUDA
  if (kernel->_sycl_queue_id == 0) {
    // Non-qu kernels
    for (auto *data : dataSet) {
      auto scopeId = parentId;
      if (apiExternIds.contain(scopeId)) {
        // It's triggered by a CUDA op but not triton op
        scopeId = data->addScope(parentId, kernel->_name);
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
      auto externId = data->addScope(parentId, kernel->_name);
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

struct XpuptiProfiler::XpuptiProfilerPimpl
    : public GPUProfiler<XpuptiProfiler>::GPUProfilerPimplInterface {
  XpuptiProfilerPimpl(XpuptiProfiler &profiler)
      : GPUProfiler<XpuptiProfiler>::GPUProfilerPimplInterface(profiler) {}
  virtual ~XpuptiProfilerPimpl() = default;

  void doStart() override;
  void doFlush() override;
  void doStop() override;

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
  uint32_t maxCorrelationId = 0;
  pti_result status;
  xpupti::Pti_Activity *activity = nullptr;
  do {
    status = ptiViewGetNextRecord(buffer, validSize, &activity);
    if (status == pti_result::PTI_SUCCESS) {
      auto correlationId =
          processActivity(profiler.correlation.corrIdToExternId,
                          profiler.correlation.apiExternIds, dataSet, activity);
      maxCorrelationId = std::max(maxCorrelationId, correlationId);
    } else if (status == pti_result::PTI_STATUS_END_OF_BUFFER) {
      std::cout << "Reached End of buffer" << '\n';
      break;
    } else {
      throw std::runtime_error("cupti::activityGetNextRecord failed");
    }
  } while (true);

  std::free(buffer);

  profiler.correlation.complete(maxCorrelationId);
}

void XpuptiProfiler::XpuptiProfilerPimpl::doStart() {
  // xpupti::subscribe<true>(&subscriber, callbackFn, nullptr);
  ptiViewEnable(PTI_VIEW_DEVICE_GPU_KERNEL);
  ptiViewSetCallbacks(allocBuffer, completeBuffer);
  // setGraphCallbacks(subscriber, /*enable=*/true);
  // setRuntimeCallbacks(subscriber, /*enable=*/true);
  // setDriverCallbacks(subscriber, /*enable=*/true);
}

} // namespace proton
