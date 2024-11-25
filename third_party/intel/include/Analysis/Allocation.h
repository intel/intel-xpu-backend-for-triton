#ifndef TRITON_INTEL_ANALYSIS_ALLOCATION_H
#define TRITON_INTEL_ANALYSIS_ALLOCATION_H

#include "triton/Analysis/Allocation.h"

namespace mlir::triton::intel {
unsigned allocationAnalysisScratchSizeFn(Operation *op);
} // namespace mlir::triton::intel

#endif
