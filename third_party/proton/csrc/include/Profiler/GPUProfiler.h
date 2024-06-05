#ifndef PROTON_PROFILER_GPU_PROFILER_H_
#define PROTON_PROFILER_GPU_PROFILER_H_

#include "Context/Context.h"
#include "Profiler.h"
#include "Utility/Atomic.h"
#include <thread>

#include <atomic>
#include <map>

namespace proton {

// Singleton<ConcreteProfilerT>: Each concrete GPU profiler, e.g.,
// CuptiProfiler, should be a singleton.
template <typename ConcreteProfilerT>
class GPUProfiler : public Profiler,
                    public OpInterface,
                    public Singleton<ConcreteProfilerT> {
public:
  GPUProfiler() = default;
  virtual ~GPUProfiler() = default;

protected:
  // OpInterface
  void startOp(const Scope &scope) override { pImpl->startOp(scope); }
  void stopOp(const Scope &scope) override { pImpl->stopOp(scope); }

  void setOpInProgress(bool value) override {
    profilerState.isRecording = value;
  }

  bool isOpInProgress() override { return profilerState.isRecording; }

  // Profiler
  virtual void doStart() override { pImpl->doStart(); }
  virtual void doFlush() override { pImpl->doFlush(); }
  virtual void doStop() override { pImpl->doStop(); }

  struct ProfilerState {
    ConcreteProfilerT &profiler;
    std::set<Data *> dataSet;
    bool isRecording{false};
    Scope scope{};

    ProfilerState(ConcreteProfilerT &profiler) : profiler(profiler) {}

    void record(const Scope &scope) {
      this->scope = scope;
      // Take a snapshot of the current dataset
      this->dataSet = profiler.getDataSet();
    }

    void enterOp() {
      profiler.enterOp(scope);
      for (auto data : dataSet)
        data->enterOp(scope);
    }

    void exitOp() {
      profiler.exitOp(scope);
      for (auto data : dataSet)
        data->exitOp(this->scope);
    }
  };

  struct Correlation {
    std::atomic<uint64_t> maxSubmittedCorrelationId{0};
    std::atomic<uint64_t> maxCompletedCorrelationId{0};

    Correlation() = default;

    void submit(const uint64_t correlationId) {
      atomicMax(maxSubmittedCorrelationId, correlationId);
    }

    void complete(const uint64_t correlationId) {
      atomicMax(maxCompletedCorrelationId, correlationId);
    }

    template <typename FlushFnT>
    void flush(uint64_t maxRetries, uint64_t sleepMs, FlushFnT &&flushFn) {
      flushFn();
      auto submittedId = maxSubmittedCorrelationId.load();
      auto completedId = maxCompletedCorrelationId.load();
      auto retries = maxRetries;
      while ((completedId < submittedId) && retries > 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(sleepMs));
        flushFn();
        completedId = maxCompletedCorrelationId.load();
        --retries;
      }
    }
  };

  static thread_local ProfilerState profilerState;
  Correlation correlation;

  // Use the pimpl idiom to hide the implementation details. This lets us avoid
  // including the cupti header from this header. The cupti header and the
  // equivalent header from AMD define conflicting macros, so we want to use
  // those headers only within cpp files.
  class GPUProfilerPimplInterface {
  public:
    GPUProfilerPimplInterface(ConcreteProfilerT &profiler)
        : profiler(profiler) {}
    virtual ~GPUProfilerPimplInterface() = default;

    virtual void startOp(const Scope &scope) = 0;
    virtual void stopOp(const Scope &scope) = 0;
    virtual void doStart() = 0;
    virtual void doFlush() = 0;
    virtual void doStop() = 0;

  protected:
    ConcreteProfilerT &profiler;
  };
  std::unique_ptr<GPUProfilerPimplInterface> pImpl;
};

} // namespace proton

#endif // PROTON_PROFILER_GPU_PROFILER_H_
