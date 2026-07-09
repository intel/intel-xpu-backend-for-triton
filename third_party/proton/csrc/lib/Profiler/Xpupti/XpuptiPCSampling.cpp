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

  std::cout << "[PC Sampling] XpuptiConfigureData::initialize() - START" << std::endl;
  std::cout << "[PC Sampling]   Device handle: " << device << std::endl;
  std::cout << "[PC Sampling]   Sampling period: " << samplingPeriodNs << " ns" << std::endl;

  // Note: PTI View must be initialized BEFORE calling ptiPcSamplingEnable.
  // This should be done once globally by the profiler initialization.

  // Create PC sampling handle
  std::cout << "[PC Sampling]   Calling ptiPcSamplingEnable()..." << std::endl;
  xpupti::pcSamplingEnable<true>(&handle);
  std::cout << "[PC Sampling]   ✓ ptiPcSamplingEnable() succeeded, handle: " << handle << std::endl;

  // Configure for all available devices (device filtering not yet implemented in PTI)
  // Pass nullptr to profile all available devices
  std::cout << "[PC Sampling]   Calling ptiPcSamplingConfigure(nullptr, 0, " << samplingPeriodNs << ")..." << std::endl;
  xpupti::pcSamplingConfigure<true>(handle, nullptr, 0, samplingPeriodNs);
  std::cout << "[PC Sampling]   ✓ ptiPcSamplingConfigure() succeeded" << std::endl;

  // Note: Stall reasons are queried AFTER stopping collection, not during init.
  // See processPCSamplingData() for the actual query.

  std::cout << "[PC Sampling] XpuptiConfigureData::initialize() - DONE" << std::endl;
}

XpuptiConfigureData *
XpuptiPCSampling::getConfigureData(pti_device_handle_t device) {
  return &deviceToConfigureData[device];
}

void XpuptiPCSampling::initialize(pti_device_handle_t device) {
  std::cout << "[PC Sampling] XpuptiPCSampling::initialize() - START" << std::endl;
  std::cout << "[PC Sampling]   Device: " << device << std::endl;

  std::lock_guard<std::mutex> lock(deviceMutex);

  // Check if already initialized for this device
  bool alreadyInitialized = false;
  deviceToConfigureData.withRead(
      device, [&](const XpuptiConfigureData &) { alreadyInitialized = true; });

  if (alreadyInitialized) {
    std::cout << "[PC Sampling]   Already initialized for this device, skipping" << std::endl;
  } else {
    std::cout << "[PC Sampling]   Not initialized yet, calling XpuptiConfigureData::initialize()" << std::endl;
    getConfigureData(device)->initialize(device);
    std::cout << "[PC Sampling]   Configuration complete" << std::endl;
  }

  std::cout << "[PC Sampling] XpuptiPCSampling::initialize() - DONE" << std::endl;
}

void XpuptiPCSampling::start(pti_device_handle_t device) {
  std::cout << "[PC Sampling] XpuptiPCSampling::start() - START" << std::endl;
  std::cout << "[PC Sampling]   Device: " << device << std::endl;
  std::cout << "[PC Sampling]   Already started: " << (pcSamplingStarted ? "yes" : "no") << std::endl;

  doubleCheckedLock(
      [&]() -> bool { return !pcSamplingStarted; }, pcSamplingMutex, [&]() {
        std::cout << "[PC Sampling]   Calling initialize()..." << std::endl;
        initialize(device);

        auto *configureData = getConfigureData(device);
        std::cout << "[PC Sampling]   Calling ptiPcSamplingStartCollection()..." << std::endl;
        std::cout << "[PC Sampling]   Handle: " << configureData->handle << std::endl;
        xpupti::pcSamplingStartCollection<true>(configureData->handle);
        std::cout << "[PC Sampling]   ✓ ptiPcSamplingStartCollection() succeeded" << std::endl;

        pcSamplingStarted = true;
        std::cout << "[PC Sampling]   Collection started successfully" << std::endl;
      });

  std::cout << "[PC Sampling] XpuptiPCSampling::start() - DONE" << std::endl;
}

void XpuptiPCSampling::processPCSamplingData(
    XpuptiConfigureData *configureData, const DataToEntryMap &dataToEntry) {

  std::cout << "[PC Sampling] processPCSamplingData() - START" << std::endl;
  std::cout << "[PC Sampling]   Handle: " << configureData->handle << std::endl;

  // Get stall reasons (must be called after StopCollection, not during init)
  std::cout << "[PC Sampling]   Calling ptiPcSamplingGetStallReasons() to query count..." << std::endl;
  size_t reasonCount = 0;
  xpupti::pcSamplingGetStallReasons<true>(configureData->handle, nullptr,
                                          &reasonCount);
  std::cout << "[PC Sampling]   ✓ Found " << reasonCount << " stall reasons" << std::endl;

  if (reasonCount > 0) {
    configureData->numStallReasons = reasonCount;
    configureData->stallReasonInfos =
        new pti_pc_sampling_stall_reason_info_t[reasonCount];

    // Set struct size for each element
    for (size_t i = 0; i < reasonCount; i++) {
      configureData->stallReasonInfos[i]._struct_size =
          sizeof(pti_pc_sampling_stall_reason_info_t);
    }

    std::cout << "[PC Sampling]   Calling ptiPcSamplingGetStallReasons() to retrieve data..." << std::endl;
    xpupti::pcSamplingGetStallReasons<true>(configureData->handle,
                                            configureData->stallReasonInfos,
                                            &reasonCount);
    std::cout << "[PC Sampling]   ✓ Retrieved stall reason details" << std::endl;

    // Print stall reasons
    for (size_t i = 0; i < reasonCount; i++) {
      std::cout << "[PC Sampling]     [" << i << "] "
                << (configureData->stallReasonInfos[i]._name ? configureData->stallReasonInfos[i]._name : "<null>")
                << std::endl;
    }

    // Match stall reasons to metric indices
    std::cout << "[PC Sampling]   Matching stall reasons to metric indices..." << std::endl;
    size_t numMatched = matchStallReasonsToIndices(reasonCount, configureData->stallReasonInfos,
                                                    configureData->stallReasonIndexToMetricIndex);
    std::cout << "[PC Sampling]   ✓ Matched " << numMatched << "/" << reasonCount << " stall reasons" << std::endl;

    // Allocate aggregated samples array
    configureData->aggregatedSamples = new uint64_t[reasonCount];
    std::cout << "[PC Sampling]   ✓ Allocated aggregated samples array" << std::endl;
  }

  // Get profiled devices
  std::cout << "[PC Sampling]   Calling ptiPcSamplingGetProfiledDevices() to query count..." << std::endl;
  size_t deviceCount = 0;
  xpupti::pcSamplingGetProfiledDevices<true>(configureData->handle, nullptr,
                                             &deviceCount);
  std::cout << "[PC Sampling]   ✓ Found " << deviceCount << " profiled devices" << std::endl;

  if (deviceCount == 0) {
    std::cout << "[PC Sampling]   No devices with samples, returning early" << std::endl;
    std::cout << "[PC Sampling] processPCSamplingData() - DONE (no samples)" << std::endl;
    return; // No devices with samples
  }

  std::cout << "[PC Sampling]   Calling ptiPcSamplingGetProfiledDevices() to retrieve device list..." << std::endl;
  std::vector<pti_device_handle_t> devices(deviceCount);
  xpupti::pcSamplingGetProfiledDevices<true>(configureData->handle,
                                             devices.data(), &deviceCount);
  std::cout << "[PC Sampling]   ✓ Retrieved " << deviceCount << " device handles" << std::endl;

  // Process each device
  for (size_t devIdx = 0; devIdx < deviceCount; devIdx++) {
    pti_device_handle_t device = devices[devIdx];
    std::cout << "[PC Sampling]   Processing device " << (devIdx + 1) << "/" << deviceCount
              << " (handle: " << device << ")" << std::endl;

    // Get observed kernel handles
    std::cout << "[PC Sampling]     Calling ptiPcSamplingGetObservedKernelHandles() to query count..." << std::endl;
    size_t kernelCount = 0;
    xpupti::pcSamplingGetObservedKernelHandles<true>(
        configureData->handle, device, nullptr, &kernelCount);
    std::cout << "[PC Sampling]     ✓ Found " << kernelCount << " kernels with samples" << std::endl;

    if (kernelCount == 0) {
      std::cout << "[PC Sampling]     No kernels with samples, skipping device" << std::endl;
      continue; // No kernels with samples
    }

    std::cout << "[PC Sampling]     Calling ptiPcSamplingGetObservedKernelHandles() to retrieve kernel list..." << std::endl;
    std::vector<uint64_t> kernelHandles(kernelCount);
    xpupti::pcSamplingGetObservedKernelHandles<true>(
        configureData->handle, device, kernelHandles.data(), &kernelCount);
    std::cout << "[PC Sampling]     ✓ Retrieved " << kernelCount << " kernel handles" << std::endl;

    // Process each kernel
    for (size_t kernIdx = 0; kernIdx < kernelCount; kernIdx++) {
      uint64_t kernelHandle = kernelHandles[kernIdx];
      std::cout << "[PC Sampling]     Processing kernel " << (kernIdx + 1) << "/" << kernelCount
                << " (handle: 0x" << std::hex << kernelHandle << std::dec << ")" << std::endl;

      // Get kernel info
      pti_pc_sampling_kernel_info_t kernelInfo;
      kernelInfo._struct_size = sizeof(pti_pc_sampling_kernel_info_t);
      kernelInfo._aggregated_samples = configureData->aggregatedSamples;

      std::cout << "[PC Sampling]       Calling ptiPcSamplingGetObservedKernelInfo()..." << std::endl;
      xpupti::pcSamplingGetObservedKernelInfo<true>(
          configureData->handle, device, kernelHandle, &kernelInfo);
      std::cout << "[PC Sampling]       ✓ Kernel name: "
                << (kernelInfo._kernel_name ? kernelInfo._kernel_name : "<null>") << std::endl;
      std::cout << "[PC Sampling]       ✓ Instructions with samples: "
                << kernelInfo._instructions_with_samples_count << std::endl;
      std::cout << "[PC Sampling]       ✓ Reason count: " << kernelInfo._reason_count << std::endl;

      if (kernelInfo._instructions_with_samples_count == 0) {
        std::cout << "[PC Sampling]       No instructions with samples, skipping kernel" << std::endl;
        continue; // No instructions with samples
      }

      // Allocate buffers for instruction-level data
      std::cout << "[PC Sampling]       Allocating buffers for instruction-level data..." << std::endl;
      std::vector<pti_pc_sampling_instruction_t> instructions(
          kernelInfo._instructions_with_samples_count);
      std::vector<uint64_t> samples(
          kernelInfo._instructions_with_samples_count *
          kernelInfo._reason_count);
      std::cout << "[PC Sampling]       ✓ Allocated " << instructions.size() << " instruction slots" << std::endl;
      std::cout << "[PC Sampling]       ✓ Allocated " << samples.size() << " sample slots ("
                << instructions.size() << " instr × " << kernelInfo._reason_count << " reasons)" << std::endl;

      // Get samples per instruction
      std::cout << "[PC Sampling]       Calling ptiPcSamplingGetSamplesPerInstruction()..." << std::endl;
      xpupti::pcSamplingGetSamplesPerInstruction<true>(
          configureData->handle, device, kernelHandle, instructions.data(),
          instructions.size(), samples.data(), samples.size());
      std::cout << "[PC Sampling]       ✓ Retrieved instruction-level samples" << std::endl;

      // Process instruction-level samples
      std::cout << "[PC Sampling]       Processing " << instructions.size() << " instructions..." << std::endl;
      size_t totalSamplesProcessed = 0;
      size_t instructionsWithSamples = 0;

      for (size_t instrIdx = 0; instrIdx < instructions.size(); instrIdx++) {
        auto &instruction = instructions[instrIdx];
        size_t instrSampleCount = 0;

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

          instrSampleCount += sampleCount;

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

        if (instrSampleCount > 0) {
          instructionsWithSamples++;
          totalSamplesProcessed += instrSampleCount;
        }
      }

      std::cout << "[PC Sampling]       ✓ Processed " << instructionsWithSamples << "/"
                << instructions.size() << " instructions with samples" << std::endl;
      std::cout << "[PC Sampling]       ✓ Total samples: " << totalSamplesProcessed << std::endl;
    }
  }

  std::cout << "[PC Sampling] processPCSamplingData() - DONE" << std::endl;
}

void XpuptiPCSampling::stop(pti_device_handle_t device,
                            const DataToEntryMap &dataToEntry) {
  std::cout << "[PC Sampling] XpuptiPCSampling::stop() - START" << std::endl;
  std::cout << "[PC Sampling]   Device: " << device << std::endl;
  std::cout << "[PC Sampling]   Collection started: " << (pcSamplingStarted ? "yes" : "no") << std::endl;

  doubleCheckedLock(
      [&]() -> bool { return pcSamplingStarted; }, pcSamplingMutex, [&]() {
        auto *configureData = getConfigureData(device);
        std::cout << "[PC Sampling]   Calling ptiPcSamplingStopCollection()..." << std::endl;
        std::cout << "[PC Sampling]   Handle: " << configureData->handle << std::endl;
        xpupti::pcSamplingStopCollection<true>(configureData->handle);
        std::cout << "[PC Sampling]   ✓ ptiPcSamplingStopCollection() succeeded" << std::endl;

        pcSamplingStarted = false;

        std::cout << "[PC Sampling]   Calling processPCSamplingData()..." << std::endl;
        processPCSamplingData(configureData, dataToEntry);
        std::cout << "[PC Sampling]   ✓ Data processing complete" << std::endl;
      });

  std::cout << "[PC Sampling] XpuptiPCSampling::stop() - DONE" << std::endl;
}

void XpuptiPCSampling::finalize(pti_device_handle_t device) {
  std::cout << "[PC Sampling] XpuptiPCSampling::finalize() - START" << std::endl;
  std::cout << "[PC Sampling]   Device: " << device << std::endl;

  std::lock_guard<std::mutex> lock(deviceMutex);

  bool exists = false;
  deviceToConfigureData.withRead(
      device, [&](const XpuptiConfigureData &) { exists = true; });

  if (!exists) {
    std::cout << "[PC Sampling]   Device not configured, nothing to finalize" << std::endl;
    std::cout << "[PC Sampling] XpuptiPCSampling::finalize() - DONE (no-op)" << std::endl;
    return;
  }

  auto *configureData = getConfigureData(device);
  std::cout << "[PC Sampling]   Calling ptiPcSamplingDisable()..." << std::endl;
  std::cout << "[PC Sampling]   Handle: " << configureData->handle << std::endl;
  xpupti::pcSamplingDisable<true>(configureData->handle);
  std::cout << "[PC Sampling]   ✓ ptiPcSamplingDisable() succeeded" << std::endl;

  deviceToConfigureData.erase(device);
  std::cout << "[PC Sampling]   ✓ Removed device from configuration map" << std::endl;

  std::cout << "[PC Sampling] XpuptiPCSampling::finalize() - DONE" << std::endl;
}

} // namespace proton
