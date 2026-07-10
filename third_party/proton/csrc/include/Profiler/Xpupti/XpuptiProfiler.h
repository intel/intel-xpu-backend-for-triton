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

  bool pcSamplingEnabled{false};
  uint32_t pcSamplingInterval{1000}; // Default interval in cycles

  // Map correlation IDs to dataToEntry for PC sampling correlation
  std::map<uint64_t, DataToEntryMap> pcSamplingCorrelationToEntry;
};

} // namespace proton

#endif // PROTON_PROFILER_XPUPTI_PROFILER_H_
