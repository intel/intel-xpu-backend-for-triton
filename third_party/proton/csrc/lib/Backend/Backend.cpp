#include "Backend/Backend.h"
#include "Driver/GPU/CudaApi.h"
#include "Driver/GPU/HipApi.h"
#include "Driver/GPU/XpuApi.h"
#include "Profiler/Cupti/CuptiProfiler.h"
#include "Profiler/Instrumentation/InstrumentationProfiler.h"
#include "Profiler/RocprofSDK/RocprofSDKProfiler.h"
#include "Profiler/Roctracer/RoctracerProfiler.h"
#include "Profiler/Xpupti/XpuptiProfiler.h"
#include "Runtime/CudaRuntime.h"
#include "Runtime/HipRuntime.h"
#include "Runtime/XpuRuntime.h"
#include <vector>

namespace proton {

const std::vector<ProfilerRegistration> getProfilerRegistrations() {
  std::vector<ProfilerRegistration> registeredProfilers = {
      {"cupti", "cuda", []() { return &CuptiProfiler::instance(); }},
#ifndef _WIN32
      // RocprofSDKProfiler is not built on Windows (see
      // Profiler/CMakeLists.txt).
      {"rocprofiler", "hip", []() { return &RocprofSDKProfiler::instance(); }},
#endif
      {"roctracer", {}, []() { return &RoctracerProfiler::instance(); }},
      {"instrumentation",
       {},
       []() { return &InstrumentationProfiler::instance(); }},
      {"xpupti", "xpu", []() { return &XpuptiProfiler::instance(); }},
  };
  for (const auto &backend : getBackendRegistrations()) {
    const auto &profiler = backend.getProfiler();
    if (profiler)
      registeredProfilers.push_back(*profiler);
  }
  return registeredProfilers;
}

const std::vector<DeviceRegistration> getDeviceRegistrations() {
  std::vector<DeviceRegistration> registeredDevices = {
      {"CUDA", DeviceType::CUDA,
       [](uint64_t index) { return cuda::getDevice(index); }},
      {"HIP", DeviceType::HIP,
       [](uint64_t index) { return hip::getDevice(index); }},
      {"XPU", DeviceType::XPU,
       [](uint64_t index) { return xpu::getDevice(index); }},
  };
  for (const auto &backend : getBackendRegistrations()) {
    const auto &device = backend.getDevice();
    if (device)
      registeredDevices.push_back(*device);
  }
  return registeredDevices;
}

const std::vector<RuntimeRegistration> getRuntimeRegistrations() {
  std::vector<RuntimeRegistration> registeredRuntimes = {
      {"CUDA", []() { return &CudaRuntime::instance(); }},
      {"HIP", []() { return &HipRuntime::instance(); }},
      {"XPU", []() { return &XpuRuntime::instance(); }},
  };
  for (const auto &backend : getBackendRegistrations()) {
    const auto &runtime = backend.getRuntime();
    if (runtime) {
      registeredRuntimes.push_back(*runtime);
    }
  }
  return registeredRuntimes;
}

const std::vector<std::string> getRegisteredProfilerNames() {
  const auto profilers = getProfilerRegistrations();
  std::vector<std::string> profilerNames(profilers.size());
  std::transform(
      profilers.begin(), profilers.end(), profilerNames.begin(),
      [](const ProfilerRegistration &entry) { return entry.getName(); });
  return profilerNames;
}

const std::optional<std::string>
getProfilerForTritonBackend(const std::string &tritonBackend) {
  const auto profilers = getProfilerRegistrations();
  auto itr = std::find_if(profilers.begin(), profilers.end(),
                          [&](const ProfilerRegistration &entry) {
                            return proton::toLower(tritonBackend) ==
                                   proton::toLower(
                                       entry.getTritonBackend().value_or(""));
                          });
  if (itr == profilers.end()) {
    return {};
  }
  return itr->getName();
}

} // namespace proton
