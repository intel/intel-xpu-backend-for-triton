#include "Profiler/Xpupti/XpuptiPCSampling.h"
#include "Data/Metric.h"
#include "Driver/GPU/XpuApi.h"
#include "Driver/GPU/XpuptiApi.h"
#include "Utility/Atomic.h"
#include "Utility/Errors.h"
#include "Utility/Map.h"
#include "Utility/String.h"
#include <memory>
#include <stdexcept>
#include <tuple>
#include <iostream>

namespace proton {

namespace {

size_t matchStallReasonsToIndices(
    size_t numStallReasons,
    pti_pc_sampling_stall_reason_info_t *stallReasonInfos,
    std::map<size_t, size_t> &stallReasonIndexToMetricIndex) {
  // Match PTI stall reasons to PCSamplingMetric kinds
  size_t numValidStalls = 0;
  for (size_t i = 0; i < numStallReasons; i++) {
    std::string ptiStallName = std::string(stallReasonInfos[i]._name);
    for (size_t j = 0; j < PCSamplingMetric::PCSamplingMetricKind::Count; j++) {
      auto metricName = std::string(PCSamplingMetric().getValueName(j));
      // Map PTI stall reason names to metric kinds
      // PTI uses different naming conventions, so we need fuzzy matching
      if (ptiStallName.find(metricName) != std::string::npos ||
          toLower(ptiStallName).find(toLower(metricName)) != std::string::npos) {
        stallReasonIndexToMetricIndex[i] = j;
        numValidStalls++;
        break;
      }
    }
  }
  return numValidStalls;
}

} // namespace

void XpuptiConfigureData::initialize(pti_device_handle_t device) {
  this->device = device;

  // Note: PTI View must be initialized BEFORE calling ptiPcSamplingEnable.
  // This should be done once globally by the profiler initialization.

  // Create PC sampling handle
  xpupti::pcSamplingEnable<true>(&handle);

  // Configure for all available devices (device filtering not yet implemented in PTI)
  // Pass nullptr to profile all available devices
  xpupti::pcSamplingConfigure<true>(handle, nullptr, 0, samplingPeriodNs);

  // Note: Stall reasons are queried AFTER stopping collection, not during init.
  // See processPCSamplingData() for the actual query.
}

XpuptiConfigureData *
XpuptiPCSampling::getConfigureData(pti_device_handle_t device) {
  return &deviceToConfigureData[device];
}

void XpuptiPCSampling::initialize(pti_device_handle_t device) {
  std::lock_guard<std::mutex> lock(deviceMutex);

  // Check if already initialized for this device
  bool alreadyInitialized = false;
  deviceToConfigureData.withRead(
      device, [&](const XpuptiConfigureData &) { alreadyInitialized = true; });

  if (!alreadyInitialized) {
    getConfigureData(device)->initialize(device);
  }
}

void XpuptiPCSampling::start(pti_device_handle_t device) {
  doubleCheckedLock(
      [&]() -> bool { return !pcSamplingStarted; }, pcSamplingMutex, [&]() {
        initialize(device);
        auto *configureData = getConfigureData(device);
        xpupti::pcSamplingStartCollection<true>(configureData->handle);
        pcSamplingStarted = true;
      });
}

void XpuptiPCSampling::processPCSamplingData(
    XpuptiConfigureData *configureData, const DataToEntryMap &dataToEntry) {

  // Get stall reasons (must be called after StopCollection, not during init)
  size_t reasonCount = 0;
  xpupti::pcSamplingGetStallReasons<true>(configureData->handle, nullptr,
                                          &reasonCount);

  if (reasonCount > 0) {
    configureData->numStallReasons = reasonCount;
    configureData->stallReasonInfos =
        new pti_pc_sampling_stall_reason_info_t[reasonCount];

    // Set struct size for each element
    for (size_t i = 0; i < reasonCount; i++) {
      configureData->stallReasonInfos[i]._struct_size =
          sizeof(pti_pc_sampling_stall_reason_info_t);
    }

    xpupti::pcSamplingGetStallReasons<true>(configureData->handle,
                                            configureData->stallReasonInfos,
                                            &reasonCount);

    // Match stall reasons to metric indices
    matchStallReasonsToIndices(reasonCount, configureData->stallReasonInfos,
                               configureData->stallReasonIndexToMetricIndex);

    // Allocate aggregated samples array
    configureData->aggregatedSamples = new uint64_t[reasonCount];
  }

  // Get profiled devices
  size_t deviceCount = 0;
  xpupti::pcSamplingGetProfiledDevices<true>(configureData->handle, nullptr,
                                             &deviceCount);

  if (deviceCount == 0) {
    return; // No devices with samples
  }

  std::vector<pti_device_handle_t> devices(deviceCount);
  xpupti::pcSamplingGetProfiledDevices<true>(configureData->handle,
                                             devices.data(), &deviceCount);

  // Process each device
  for (size_t devIdx = 0; devIdx < deviceCount; devIdx++) {
    pti_device_handle_t device = devices[devIdx];

    // Get observed kernel handles
    size_t kernelCount = 0;
    xpupti::pcSamplingGetObservedKernelHandles<true>(
        configureData->handle, device, nullptr, &kernelCount);

    if (kernelCount == 0) {
      continue; // No kernels with samples
    }

    std::vector<uint64_t> kernelHandles(kernelCount);
    xpupti::pcSamplingGetObservedKernelHandles<true>(
        configureData->handle, device, kernelHandles.data(), &kernelCount);

    // Process each kernel
    for (size_t kernIdx = 0; kernIdx < kernelCount; kernIdx++) {
      uint64_t kernelHandle = kernelHandles[kernIdx];

      // Get kernel info
      pti_pc_sampling_kernel_info_t kernelInfo;
      kernelInfo._struct_size = sizeof(pti_pc_sampling_kernel_info_t);
      kernelInfo._aggregated_samples = configureData->aggregatedSamples;

      xpupti::pcSamplingGetObservedKernelInfo<true>(
          configureData->handle, device, kernelHandle, &kernelInfo);

      if (kernelInfo._instructions_with_samples_count == 0) {
        continue; // No instructions with samples
      }

      // Allocate buffers for instruction-level data
      std::vector<pti_pc_sampling_instruction_t> instructions(
          kernelInfo._instructions_with_samples_count);
      std::vector<uint64_t> samples(
          kernelInfo._instructions_with_samples_count *
          kernelInfo._reason_count);

      // Get samples per instruction
      xpupti::pcSamplingGetSamplesPerInstruction<true>(
          configureData->handle, device, kernelHandle, instructions.data(),
          instructions.size(), samples.data(), samples.size());

      // Process instruction-level samples
      for (size_t instrIdx = 0; instrIdx < instructions.size(); instrIdx++) {
        auto &instruction = instructions[instrIdx];

        // Build op name with source location if available
        std::string opName = std::string(kernelInfo._kernel_name);
        if (instruction._source_info &&
            instruction._source_info->_file_path != nullptr) {
          opName = formatFileLineFunction(
              std::string(instruction._source_info->_file_path),
              instruction._source_info->_file_line,
              std::string(kernelInfo._kernel_name));
        }

        // Add metrics for each stall reason
        for (size_t reasonIdx = 0; reasonIdx < kernelInfo._reason_count;
             reasonIdx++) {
          uint64_t sampleCount =
              samples[instrIdx * kernelInfo._reason_count + reasonIdx];

          if (sampleCount == 0) {
            continue; // Skip zero samples
          }

          // Find the metric kind for this stall reason
          if (!configureData->stallReasonIndexToMetricIndex.count(reasonIdx)) {
            continue; // Unknown stall reason
          }

          auto metricKind = static_cast<PCSamplingMetric::PCSamplingMetricKind>(
              configureData->stallReasonIndexToMetricIndex[reasonIdx]);

          // Add metric to all entries in dataToEntry
          for (const auto &[data, baseEntry] : dataToEntry) {
            auto entry = baseEntry;
            if (instruction._source_info &&
                instruction._source_info->_file_path != nullptr) {
              entry = data->addOp(entry.phase, entry.id, {opName});
            }

            // For PTI, we consider all samples as "stalled" except for "active"
            // This is a simplification; PTI categorizes differently than CUPTI
            bool isActive = (std::string(
                                 configureData->stallReasonInfos[reasonIdx]._name)
                                 .find("active") != std::string::npos);
            uint64_t stalledSamples = isActive ? 0 : sampleCount;

            entry.upsertMetric(std::make_unique<PCSamplingMetric>(
                metricKind, sampleCount, stalledSamples));
          }
        }
      }
    }
  }
}

void XpuptiPCSampling::stop(pti_device_handle_t device,
                            const DataToEntryMap &dataToEntry) {
  doubleCheckedLock(
      [&]() -> bool { return pcSamplingStarted; }, pcSamplingMutex, [&]() {
        auto *configureData = getConfigureData(device);
        xpupti::pcSamplingStopCollection<true>(configureData->handle);
        pcSamplingStarted = false;
        processPCSamplingData(configureData, dataToEntry);
      });
}

void XpuptiPCSampling::finalize(pti_device_handle_t device) {
  std::lock_guard<std::mutex> lock(deviceMutex);

  bool exists = false;
  deviceToConfigureData.withRead(
      device, [&](const XpuptiConfigureData &) { exists = true; });

  if (!exists) {
    return;
  }

  auto *configureData = getConfigureData(device);
  xpupti::pcSamplingDisable<true>(configureData->handle);
  deviceToConfigureData.erase(device);
}

} // namespace proton
