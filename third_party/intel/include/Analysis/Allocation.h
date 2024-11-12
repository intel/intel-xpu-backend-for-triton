#ifndef TRITON_INTEL_ANALYSIS_ALLOCATION_H
#define TRITON_INTEL_ANALYSIS_ALLOCATION_H

#include "triton/Analysis/Allocation.h"

namespace mlir {
namespace triton::intel {
class AllocationAnalysis;
} // namespace triton::intel
template <>
void Allocation::run<triton::intel::AllocationAnalysis>(
    FuncAllocMapT &funcAllocMap);
} // namespace mlir

#endif
