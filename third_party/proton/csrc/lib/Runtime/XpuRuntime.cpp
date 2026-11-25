#include "Runtime/XpuRuntime.h"

#include "Driver/GPU/XpuApi.h"
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

typedef void (*CopyDeviceToHostAsyncFunc)(void *, void *, const void *, size_t);
typedef void (*WaitOnSyclQueueFunc)(void *);
typedef void (*AllocateHostBufferFunc)(void *, uint8_t **, size_t);
typedef void (*FreeHostBufferFunc)(void *, uint8_t *);
typedef void (*AllocateDeviceBufferFunc)(void *, uint8_t **, size_t);
typedef void (*FreeDeviceBufferFunc)(void *, uint8_t *);
typedef void (*MemsetAsyncFunc)(void *, void *, int32_t, size_t);
typedef void (*SynchronizeDeviceFunc)(void *);
typedef void *(*GetDeviceKeyFunc)(void *);

template <typename Fn> Fn loadProtonUtilsSymbol(const char *name) {
  const std::string &path = proton::xpu::PROTON_UTILS;
  if (path.empty()) {
    throw std::runtime_error(
        std::string("[PROTON] PROTON_UTILS path is empty; cannot resolve ") +
        name);
  }
#if defined(_WIN32)
  HMODULE handle = LoadLibrary(path.data());
  if (!handle) {
    throw std::runtime_error(std::string("[PROTON] Failed to load ") + path +
                             ", code: " + std::to_string(GetLastError()));
  }
  auto fn = reinterpret_cast<Fn>(GetProcAddress(handle, name));
  if (!fn) {
    long err = GetLastError();
    FreeLibrary(handle);
    throw std::runtime_error(std::string("[PROTON] Failed to resolve ") + name +
                             ", code: " + std::to_string(err));
  }
  return fn;
#else
  void *handle = dlopen(path.data(), RTLD_LAZY);
  if (!handle) {
    throw std::runtime_error(std::string("[PROTON] Failed to load ") + path +
                             ": " + dlerror());
  }
  dlerror();
  auto fn = reinterpret_cast<Fn>(dlsym(handle, name));
  const char *err = dlerror();
  if (err) {
    dlclose(handle);
    throw std::runtime_error(std::string("[PROTON] Failed to resolve ") + name +
                             ": " + err);
  }
  return fn;
#endif
}

static void *requireSyclQueue() {
  void *q = proton::XpuRuntime::instance().getSyclQueue();
  if (q == nullptr) {
    throw std::runtime_error("[PROTON] XpuRuntime has no SYCL queue; "
                             "Session activation did not call setSyclQueue.");
  }
  return q;
}

} // namespace

namespace proton {

void XpuRuntime::launchKernel(void *kernel, unsigned int gridDimX,
                              unsigned int gridDimY, unsigned int gridDimZ,
                              unsigned int blockDimX, unsigned int blockDimY,
                              unsigned int blockDimZ,
                              unsigned int sharedMemBytes, void *stream,
                              void **kernelParams, void **extra) {
  throw std::runtime_error(
      "[PROTON] XpuRuntime::launchKernel is not implemented on XPU.");
}

void XpuRuntime::memset(void *devicePtr, uint32_t value, size_t size,
                        void *stream) {
  static auto memsetFn = loadProtonUtilsSymbol<MemsetAsyncFunc>("memsetAsync");
  memsetFn(stream, devicePtr, static_cast<int32_t>(value), size);
}

void XpuRuntime::allocateHostBuffer(uint8_t **buffer, size_t size,
                                    bool mapped) {
  static auto allocFn =
      loadProtonUtilsSymbol<AllocateHostBufferFunc>("allocateHostBuffer");
  allocFn(requireSyclQueue(), buffer, size);
}

void XpuRuntime::getHostDevicePointer(uint8_t *hostPtr, uint8_t **devicePtr) {
  // SYCL USM host pointers are device-accessible at the same VA.
  *devicePtr = hostPtr;
}

void XpuRuntime::freeHostBuffer(uint8_t *buffer) {
  static auto freeFn =
      loadProtonUtilsSymbol<FreeHostBufferFunc>("freeHostBuffer");
  freeFn(requireSyclQueue(), buffer);
}

void XpuRuntime::allocateDeviceBuffer(uint8_t **buffer, size_t size) {
  static auto allocFn =
      loadProtonUtilsSymbol<AllocateDeviceBufferFunc>("allocateDeviceBuffer");
  allocFn(requireSyclQueue(), buffer, size);
}

void XpuRuntime::freeDeviceBuffer(uint8_t *buffer) {
  static auto freeFn =
      loadProtonUtilsSymbol<FreeDeviceBufferFunc>("freeDeviceBuffer");
  freeFn(requireSyclQueue(), buffer);
}

void XpuRuntime::copyDeviceToHostAsync(void *dst, const void *src, size_t size,
                                       void *stream) {
  static auto copyFn =
      loadProtonUtilsSymbol<CopyDeviceToHostAsyncFunc>("copyDeviceToHostAsync");
  copyFn(stream, dst, src, size);
}

void *XpuRuntime::getDevice() {
  static auto keyFn = loadProtonUtilsSymbol<GetDeviceKeyFunc>("getDeviceKey");
  void *q = XpuRuntime::instance().getSyclQueue();
  if (q == nullptr) {
    // Non-null sentinel so MetricBuffer does not alias with backends that
    // return nullptr from getDevice.
    return reinterpret_cast<void *>(static_cast<uintptr_t>(1));
  }
  return keyFn(q);
}

void *XpuRuntime::getPriorityStream() { return nullptr; }

void XpuRuntime::synchronizeStream(void *stream) {
  static auto waitFn =
      loadProtonUtilsSymbol<WaitOnSyclQueueFunc>("waitOnSyclQueue");
  waitFn(stream);
}

void XpuRuntime::synchronizeDevice() {
  static auto syncFn =
      loadProtonUtilsSymbol<SynchronizeDeviceFunc>("synchronizeDevice");
  syncFn(requireSyclQueue());
}

void XpuRuntime::destroyStream(void *stream) {
  // No-op: SYCL queues are owned by the Triton driver.
}

void XpuRuntime::processHostBuffer(
    uint8_t *hostBuffer, size_t hostBufferSize, uint8_t *deviceBuffer,
    size_t deviceBufferSize, void *stream,
    std::function<void(uint8_t *, size_t)> callback) {
  throw std::runtime_error(
      "[PROTON] XpuRuntime::processHostBuffer is not implemented on XPU.");
}
} // namespace proton
