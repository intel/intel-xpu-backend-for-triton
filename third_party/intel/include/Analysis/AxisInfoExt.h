#ifndef TRITON_INTEL_ANALYSIS_AXISINFOEXT_H
#define TRITON_INTEL_ANALYSIS_AXISINFOEXT_H

#include "triton/Analysis/AxisInfo.h"

namespace mlir::triton::intel {

/// Intel-specific AxisInfo extension.
///
/// Registers Intel-specific visitors (MakeTensorPtrOp, AdvanceOp,
/// MakeTensorDescOp, DescriptorLoadOp, LLVM dialect ops, IndexCastOp)
/// via the upstream `addVisitors` callback mechanism.
struct AxisInfoExt {
  static void addVisitors(mlir::triton::AxisInfoVisitorList &visitors);
};

/// Module level axis info analysis based on the call graph, assuming that we do
/// not have recursive functions.
///
/// Since each function will be called multiple times, we need to calculate the
/// axis info based on the axis info of all the callers.  In the future, we can
/// perform optimization using function cloning so that each call site will have
/// unique axis info.
class ModuleAxisInfoAnalysis : public triton::ModuleAxisInfoAnalysis {
public:
  explicit ModuleAxisInfoAnalysis(ModuleOp moduleOp);

  AxisInfo *getAxisInfo(Value value) const;

  unsigned getContiguity(Value value) const;
  unsigned getAlignment(Value value) const;

  unsigned getMaskAlignment(Value mask) const;
};

} // namespace mlir::triton::intel

#endif // TRITON_INTEL_ANALYSIS_AXISINFOEXT_H
