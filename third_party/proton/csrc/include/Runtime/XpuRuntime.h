#ifndef PROTON_RUNTIME_XPU_RUNTIME_H_
#define PROTON_RUNTIME_XPU_RUNTIME_H_

#include "Runtime.h"
#include "Utility/Singleton.h"

namespace proton {

class XpuRuntime : public Singleton<XpuRuntime>, public Runtime {
public:
  XpuRuntime() : Runtime(DeviceType::XPU) {}
  ~XpuRuntime() = default;

  // Set by Session::makeProfiler for any XPU session. There is exactly one
  // queue process-wide, set once at session start -- not one per device --
  // so XPU instrumentation only supports a single GPU today. `getDevice()`
  // still reports a distinct ordinal per visible device (see
  // `proton_utils.cpp`'s `getDeviceKey`), but every call that resolves a
  // stream or allocates a buffer (`getPriorityStream`,
  // `allocateDeviceBuffer`, `allocateHostBuffer`, ...) goes through this one
  // queue regardless of which device ordinal asked, so multi-GPU callers
  // would silently get device 0's resources for every device.
  void setSyclQueue(void *syclQueue) { syclQueue_ = syclQueue; }
  void *getSyclQueue() const { return syclQueue_; }

  void launchKernel(void *kernel, unsigned int gridDimX, unsigned int gridDimY,
                    unsigned int gridDimZ, unsigned int blockDimX,
                    unsigned int blockDimY, unsigned int blockDimZ,
                    unsigned int sharedMemBytes, void *stream,
                    void **kernelParams, void **extra) override;
  void memset(void *devicePtr, uint32_t value, size_t size,
              void *stream) override;
  void allocateHostBuffer(uint8_t **buffer, size_t size, bool mapped) override;
  void getHostDevicePointer(uint8_t *hostPtr, uint8_t **devicePtr) override;
  void freeHostBuffer(uint8_t *buffer) override;
  void allocateDeviceBuffer(uint8_t **buffer, size_t size) override;
  void freeDeviceBuffer(uint8_t *buffer) override;
  void copyDeviceToHostAsync(void *dst, const void *src, size_t size,
                             void *stream) override;
  void *getDevice() override;
  void *getPriorityStream() override;
  void synchronizeStream(void *stream) override;
  void synchronizeDevice() override;
  void destroyStream(void *stream) override;
  void
  processHostBuffer(uint8_t *hostBuffer, size_t hostBufferSize,
                    uint8_t *deviceBuffer, size_t deviceBufferSize,
                    void *stream,
                    std::function<void(uint8_t *, size_t)> callback) override;

private:
  void *syclQueue_{nullptr};
};

} // namespace proton

#endif // PROTON_RUNTIME_XPU_RUNTIME_H_
