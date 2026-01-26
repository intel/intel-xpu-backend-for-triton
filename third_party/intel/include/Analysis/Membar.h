#ifndef TRITON_INTEL_ANALYSIS_MEMBAR_H
#define TRITON_INTEL_ANALYSIS_MEMBAR_H

#include "triton/Analysis/Allocation.h"

namespace mlir {
class Operation;
namespace intel {
/// Intel-specific callback to filter operations that need no barriers between
/// each other.
///
/// This is useful as the granularity to check whether barriers are needed is
/// quite coarse. The filter will return true if no barrier is needed between
/// `lhsOp` and `rhsOp`.
bool membarFilter(Operation *lhsOp, Operation *rhsOp, Allocation *allocation);
} // namespace intel
} // namespace mlir

#endif // TRITON_INTEL_ANALYSIS_MEMBAR_H
