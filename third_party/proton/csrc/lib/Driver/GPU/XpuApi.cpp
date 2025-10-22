#include "Driver/GPU/XpuApi.h"

#include <dlfcn.h>
#include <iostream>
#include <string>

namespace proton {

namespace xpu {

typedef void (*GetDeviceFunc)(uint64_t, uint32_t *, uint32_t *, uint32_t *,
                              uint32_t *, char[256]);

Device getDevice(uint64_t index) {
  // void *handle = dlopen(utils_cache_path.data(), RTLD_LAZY);
  void *handle = dlopen(std::getenv("PROTON_XPUAPI_LIB_PATH"), RTLD_LAZY);
  if (!handle) {
    const char *dlopen_error = dlerror();
    std::cerr << "Failed to load library: " << dlopen_error << std::endl;
    throw std::runtime_error(std::string("Failed to load library: ") +
                             std::string(dlopen_error));
  }

  dlerror();
  GetDeviceFunc getDeviceFromLib = (GetDeviceFunc)dlsym(handle, "getDevice");
  const char *dlsym_error = dlerror();
  if (dlsym_error) {
    std::cerr << "Failed to load function: " << dlsym_error << std::endl;
    dlclose(handle);
    throw std::runtime_error(std::string("Failed to load function: ") +
                             std::string(dlsym_error));
  }

  uint32_t clockRate = 0;
  uint32_t memoryClockRate = 0;
  uint32_t busWidth = 0;
  uint32_t numSms = 0;
  char arch[256];
  getDeviceFromLib(index, &clockRate, &memoryClockRate, &busWidth, &numSms,
                   arch);
  dlclose(handle);

  return Device(DeviceType::XPU, index, clockRate, memoryClockRate, busWidth,
                numSms, arch);
}

} // namespace xpu

} // namespace proton
