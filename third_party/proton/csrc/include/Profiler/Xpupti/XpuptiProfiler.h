#ifndef PROTON_PROFILER_XPUPTI_PROFILER_H_
#define PROTON_PROFILER_XPUPTI_PROFILER_H_

#include "Profiler/GPUProfiler.h"

namespace proton {

class XpuptiProfiler : public GPUProfiler<XpuptiProfiler> {
public:
  XpuptiProfiler();
  virtual ~XpuptiProfiler();

  int64_t getTimestampOffsetNs() const override { return 0; }

private:
  struct XpuptiProfilerPimpl;

  virtual void
  doSetMode(const std::vector<std::string> &modeAndOptions) override;
};

} // namespace proton

#endif // PROTON_PROFILER_XPUPTI_PROFILER_H_
