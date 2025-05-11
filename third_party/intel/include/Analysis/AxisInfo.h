#ifndef TRITON_INTEL_ANALYSIS_AXISINFO_H
#define TRITON_INTEL_ANALYSIS_AXISINFO_H

#include "triton/Analysis/AxisInfo.h"

namespace mlir::triton::intel {

// Module level axis info analysis based on the call graph, assuming that we do
// not have recursive functions.
//
// Since each function will be called multiple times, we need to calculate the
// axis info based on the axis info of all the callers.  In the future, we can
// perform optimization using function cloning so that each call site will have
// unique axis info.
class ModuleAxisInfoAnalysis : public triton::ModuleAxisInfoAnalysis {
public:
  explicit ModuleAxisInfoAnalysis(ModuleOp moduleOp);

  AxisInfo *getAxisInfo(Value value);

  unsigned getContiguity(Value value);
  unsigned getAlignment(Value value);

  unsigned getMaskAlignment(Value mask);

private:
  void initialize(FunctionOpInterface funcOp);
  void update(CallOpInterface callOp, FunctionOpInterface funcOp);
};

} // namespace mlir::triton::intel

#endif
