#ifndef PROTON_PROFILER_XPUPTI_PROFILER_H_
#define PROTON_PROFILER_XPUPTI_PROFILER_H_

#include "Profiler/GPUProfiler.h"
#include <memory>
#include <map>

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

public:
  // PC sampling state - public so activity processing can access it
  bool pcSamplingEnabled{false};
  uint32_t pcSamplingInterval{1000}; // Default interval in cycles

  // Save kernel entries created during activity processing for PC sampling
  // These are the ACTUAL entries (with kernel names) that should receive PC samples
  std::vector<DataToEntryMap> pcSamplingKernelEntries;
  std::mutex pcSamplingEntriesMutex;
};

} // namespace proton

#endif // PROTON_PROFILER_XPUPTI_PROFILER_H_
