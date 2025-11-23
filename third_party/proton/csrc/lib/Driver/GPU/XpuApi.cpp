#include "Driver/GPU/XpuApi.h"

#ifdef WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <stdexcept>
#include <string>

namespace proton {

namespace xpu {

std::string PROTON_UTILS;

typedef void (*GetDevicePropertiesFunc)(uint64_t, uint32_t *, uint32_t *,
                                        uint32_t *, uint32_t *, char[256]);

#ifdef WIN32
Device getDevice(uint64_t index) {
  HMODULE handle = LoadLibrary(PROTON_UTILS.data());
  if (!handle) {
    long err = GetLastError();
    throw std::runtime_error(std::string("Failed to load library code:") +
                             std::to_string(err));
  }

  GetLastError();
  GetDevicePropertiesFunc getDeviceProperties =
      (GetDevicePropertiesFunc)GetProcAddress(handle, "getDeviceProperties");
  long err = GetLastError();
  if (err) {
    FreeLibrary(handle);
    throw std::runtime_error(std::string("Failed to load function code:") +
                             std::to_string(err));
  }

  uint32_t clockRate = 0;
  uint32_t memoryClockRate = 0;
  uint32_t busWidth = 0;
  uint32_t numSms = 0;
  char arch[256];
  getDeviceProperties(index, &clockRate, &memoryClockRate, &busWidth, &numSms,
                      arch);
  FreeLibrary(handle);

  return Device(DeviceType::XPU, index, clockRate, memoryClockRate, busWidth,
                numSms, arch);
}
#else
Device getDevice(uint64_t index) {
  void *handle = dlopen(PROTON_UTILS.data(), RTLD_LAZY);
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
#endif

} // namespace xpu

} // namespace proton
