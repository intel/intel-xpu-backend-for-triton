#ifndef PROTON_PROFILER_XPUPTI_PROFILER_H_
#define PROTON_PROFILER_XPUPTI_PROFILER_H_

#include "Profiler/GPUProfiler.h"
#include <memory>

namespace proton {

class XpuptiPCSampling;

class XpuptiProfiler : public GPUProfiler<XpuptiProfiler> {
public:
  XpuptiProfiler();
  virtual ~XpuptiProfiler();

private:
  struct XpuptiProfilerPimpl;

  virtual void
  doSetMode(const std::vector<std::string> &modeAndOptions) override;

  bool pcSamplingEnabled{false};
  uint32_t pcSamplingInterval{1000}; // Default interval in cycles
};

} // namespace proton

#endif // PROTON_PROFILER_XPUPTI_PROFILER_H_
