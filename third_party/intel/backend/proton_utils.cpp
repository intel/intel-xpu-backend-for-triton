#include <sycl/sycl.hpp>

extern "C" void waitOnSyclQueue(void *syclQueue) {
  sycl::queue *queue = static_cast<sycl::queue *>(syclQueue);
  queue->wait();
}

// FIXME: Should it be in DeviceInfo class?
// Inspired by Kineto: `XpuptiActivityProfiler.cpp`
extern "C" void
enumDeviceUUIDs(std::vector<std::array<uint8_t, 16>> deviceUUIDs_) {
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
