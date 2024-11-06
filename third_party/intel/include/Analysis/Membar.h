#ifndef TRITON_INTEL_ANALYSIS_MEMBAR_H
#define TRITON_INTEL_ANALYSIS_MEMBAR_H

namespace mlir {
class Operation;
namespace intel {
/// Intel-specific callback to filter operations that need no barriers between
/// each other.
///
/// This is useful as the granularity to check whether barriers are needed is
/// quite coarse.
bool membarFilter(Operation *lhsOp, Operation *rhsOp);
} // namespace intel
} // namespace mlir

#endif // TRITON_INTEL_ANALYSIS_MEMBAR_H
