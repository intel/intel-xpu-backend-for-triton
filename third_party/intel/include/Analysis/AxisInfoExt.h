#ifndef TRITON_INTEL_ANALYSIS_AXIS_INFO_EXT_H
#define TRITON_INTEL_ANALYSIS_AXIS_INFO_EXT_H

#include "triton/Analysis/AxisInfo.h"

namespace mlir::triton::intel {

struct AxisInfoExt {
  static void addVisitors(mlir::triton::AxisInfoVisitorList &visitors);
};

class ModuleAxisInfoAnalysis : public mlir::triton::ModuleAxisInfoAnalysis {
public:
  explicit ModuleAxisInfoAnalysis(ModuleOp moduleOp);

  AxisInfo *getAxisInfo(Value value);

  unsigned getContiguity(Value value);
  unsigned getAlignment(Value value);

  unsigned getMaskAlignment(Value mask);
};

} // namespace mlir::triton::intel

#endif
