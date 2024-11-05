#ifndef PROTON_PROFILER_XPUPTI_PROFILER_H_
#define PROTON_PROFILER_XPUPTI_PROFILER_H_

#include "Profiler/GPUProfiler.h"

namespace proton {

class XpuptiProfiler : public GPUProfiler<XpuptiProfiler> {
public:
  XpuptiProfiler();
  virtual ~XpuptiProfiler();

private:
  struct XpuptiProfilerPimpl;
};

} // namespace proton

#endif // PROTON_PROFILER_Xpupti_PROFILER_H_
