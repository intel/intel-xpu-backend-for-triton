// FIXME: this code is copied from HipRuntime.cpp, need to adapt it.
#include "Runtime/XpuRuntime.h"

#include "Driver/GPU/XpuApi.h"
#include <algorithm>
#include <cstdint>

namespace proton {

void XpuRuntime::launchKernel(void *kernel, unsigned int gridDimX,
                              unsigned int gridDimY, unsigned int gridDimZ,
                              unsigned int blockDimX, unsigned int blockDimY,
                              unsigned int blockDimZ,
                              unsigned int sharedMemBytes, void *stream,
                              void **kernelParams, void **extra) {
  auto status = xpu::launchKernel<true>(
      reinterpret_cast<hipFunction_t>(kernel), gridDimX, gridDimY, gridDimZ,
      blockDimX, blockDimY, blockDimZ, sharedMemBytes,
      reinterpret_cast<hipStream_t>(stream), kernelParams, extra);
  (void)status;
}

void XpuRuntime::memset(void *devicePtr, uint32_t value, size_t size,
                        void *stream) {
  auto status = xpu::memsetD32Async<true>(
      reinterpret_cast<hipDeviceptr_t>(devicePtr), value,
      size / sizeof(uint32_t), reinterpret_cast<hipStream_t>(stream));
  (void)status;
}

void XpuRuntime::allocateHostBuffer(uint8_t **buffer, size_t size) {
  (void)xpu::memAllocHost<true>(reinterpret_cast<void **>(buffer), size);
}

void XpuRuntime::freeHostBuffer(uint8_t *buffer) {
  (void)xpu::memFreeHost<true>(buffer);
}

void XpuRuntime::allocateDeviceBuffer(uint8_t **buffer, size_t size) {
  hipDeviceptr_t devicePtr;
  (void)xpu::memAlloc<true>(reinterpret_cast<void **>(&devicePtr), size);
  *buffer = reinterpret_cast<uint8_t *>(devicePtr);
}

void XpuRuntime::freeDeviceBuffer(uint8_t *buffer) {
  hipDeviceptr_t devicePtr = reinterpret_cast<hipDeviceptr_t>(buffer);
  (void)xpu::memFree<true>(devicePtr);
}

void XpuRuntime::copyDeviceToHostAsync(void *dst, const void *src, size_t size,
                                       void *stream) {
  (void)xpu::memcpyDToHAsync<true>(
      dst, reinterpret_cast<hipDeviceptr_t>(const_cast<void *>(src)), size,
      reinterpret_cast<hipStream_t>(stream));
}

void *XpuRuntime::getDevice() {
  hipDevice_t device;
  (void)xpu::ctxGetDevice<true>(&device);
  return reinterpret_cast<void *>(static_cast<uintptr_t>(device));
}

void *XpuRuntime::getPriorityStream() {
  hipStream_t stream;
  int lowestPriority, highestPriority;
  (void)xpu::ctxGetStreamPriorityRange<true>(&lowestPriority, &highestPriority);
  (void)xpu::streamCreateWithPriority<true>(&stream, hipStreamNonBlocking,
                                            highestPriority);
  return reinterpret_cast<void *>(stream);
}

void XpuRuntime::synchronizeStream(void *stream) {
  (void)xpu::streamSynchronize<true>(reinterpret_cast<hipStream_t>(stream));
}

void XpuRuntime::synchronizeDevice() { (void)xpu::deviceSynchronize<true>(); }

void XpuRuntime::destroyStream(void *stream) {
  (void)xpu::streamDestroy<true>(reinterpret_cast<hipStream_t>(stream));
}

void XpuRuntime::processHostBuffer(
    uint8_t *hostBuffer, size_t hostBufferSize, uint8_t *deviceBuffer,
    size_t deviceBufferSize, void *stream,
    std::function<void(uint8_t *, size_t)> callback) {
  int64_t chunkSize = std::min(hostBufferSize, deviceBufferSize);
  int64_t sizeLeftOnDevice = deviceBufferSize;
  while (chunkSize > 0) {
    (void)xpu::memcpyDToHAsync<true>(
        reinterpret_cast<void *>(hostBuffer),
        reinterpret_cast<hipDeviceptr_t>(deviceBuffer), chunkSize,
        reinterpret_cast<hipStream_t>(stream));
    (void)xpu::streamSynchronize<true>(reinterpret_cast<hipStream_t>(stream));
    callback(hostBuffer, chunkSize);
    hostBuffer += chunkSize;
    deviceBuffer += chunkSize;
    sizeLeftOnDevice -= chunkSize;
    chunkSize =
        std::min(static_cast<int64_t>(hostBufferSize), sizeLeftOnDevice);
  }
}
} // namespace proton
