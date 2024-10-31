#ifndef TRITON_INTEL_ANALYSIS_ALLOCATION_H
#define TRITON_INTEL_ANALYSIS_ALLOCATION_H

#include "triton/Analysis/Allocation.h"

namespace mlir {
template <>
void Allocation::run<triton::intel::AllocationAnalysis>(
    FuncAllocMapT &funcAllocMap);
} // namespace mlir

#endif
