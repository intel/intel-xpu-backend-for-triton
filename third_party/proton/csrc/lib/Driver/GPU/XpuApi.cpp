#include "Driver/GPU/XpuApi.h"

#include <dlfcn.h>
#include <stdexcept>
#include <string>

namespace proton {

namespace xpu {

std::string XPU_API_UTILS;

typedef void (*GetDevicePropertiesFunc)(uint64_t, uint32_t *, uint32_t *,
                                        uint32_t *, uint32_t *, char[256]);

Device getDevice(uint64_t index) {
  void *handle = dlopen(XPU_API_UTILS.data(), RTLD_LAZY);
  if (!handle) {
    const char *dlopen_error = dlerror();
    throw std::runtime_error(std::string("Failed to load library: ") +
                             std::string(dlopen_error));
  }

  dlerror();
  GetDevicePropertiesFunc getDeviceProperties =
      (GetDevicePropertiesFunc)dlsym(handle, "getDeviceProperties");
  const char *dlsym_error = dlerror();
  if (dlsym_error) {
    dlclose(handle);
    throw std::runtime_error(std::string("Failed to load function: ") +
                             std::string(dlsym_error));
  }

  uint32_t clockRate = 0;
  uint32_t memoryClockRate = 0;
  uint32_t busWidth = 0;
  uint32_t numSms = 0;
  char arch[256];
  getDeviceProperties(index, &clockRate, &memoryClockRate, &busWidth, &numSms,
                      arch);
  dlclose(handle);

  return Device(DeviceType::XPU, index, clockRate, memoryClockRate, busWidth,
                numSms, arch);
}

} // namespace xpu

} // namespace proton
