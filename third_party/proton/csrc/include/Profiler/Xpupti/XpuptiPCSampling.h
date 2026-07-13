#ifndef PROTON_PROFILER_XPUPTI_PC_SAMPLING_H_
#define PROTON_PROFILER_XPUPTI_PC_SAMPLING_H_

#include "Driver/GPU/XpuApi.h"
#include "Driver/GPU/XpuptiApi.h"
#include "Utility/Map.h"
#include "Utility/Singleton.h"
#include "XpuptiProfiler.h"
#include <atomic>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include <pti/pti_pc_sampling.h>

namespace proton {

struct XpuptiConfigureData {
  XpuptiConfigureData() = default;

  ~XpuptiConfigureData() {
    if (stallReasonInfos) {
      delete[] stallReasonInfos;
    }
    if (aggregatedSamples) {
      delete[] aggregatedSamples;
    }
  }

  void initialize(pti_device_handle_t device, uint32_t samplingPeriodNs);

  // Default sampling period in nanoseconds = 100 ns (1 microsecond)
  static constexpr uint32_t DefaultSamplingPeriodNs = 100;

  pti_device_handle_t device{};
  pti_pc_sampling_handle_t handle{};
  uint32_t samplingPeriodNs{DefaultSamplingPeriodNs};
  size_t numStallReasons{};
  pti_pc_sampling_stall_reason_info_t *stallReasonInfos{};
  std::map<size_t, size_t> stallReasonIndexToMetricIndex{};
  uint64_t *aggregatedSamples{};
};

class XpuptiPCSampling : public Singleton<XpuptiPCSampling> {

public:
  XpuptiPCSampling() = default;
  virtual ~XpuptiPCSampling() = default;

  void initialize(pti_device_handle_t device, uint32_t samplingPeriodNs);

  void start(pti_device_handle_t device, uint32_t samplingPeriodNs);

  void stop(pti_device_handle_t device,
            const std::map<std::string, std::vector<DataToEntryMap>>
                &kernelNameToEntries);

  // Stop collection without processing data (for use when processing multiple
  // entries)
  void stopCollection(pti_device_handle_t device);

  // Process samples for all saved kernel-name -> entries. Kernel names are
  // used to correlate PTI's per-kernel-binary aggregated samples with the
  // Triton-side scope entries recorded for that kernel (see
  // processPCSamplingData() for why this correlation is needed).
  void processData(pti_device_handle_t device,
                   const std::map<std::string, std::vector<DataToEntryMap>>
                       &kernelNameToEntries);

  void finalize(pti_device_handle_t device);

private:
  XpuptiConfigureData *getConfigureData(pti_device_handle_t device);

  void
  processPCSamplingData(XpuptiConfigureData *configureData,
                        const std::map<std::string, std::vector<DataToEntryMap>>
                            &kernelNameToEntries);

  ThreadSafeMap<pti_device_handle_t, XpuptiConfigureData> deviceToConfigureData;

  std::atomic<bool> pcSamplingStarted{false};
  std::mutex pcSamplingMutex{};
  std::mutex deviceMutex{};
};

} // namespace proton

#endif // PROTON_PROFILER_XPUPTI_PC_SAMPLING_H_
