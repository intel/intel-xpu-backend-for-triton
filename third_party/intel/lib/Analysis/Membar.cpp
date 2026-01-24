#include "intel/include/Analysis/Membar.h"
#include "intel/include/Analysis/Utility.h"

namespace mlir::intel {
namespace {
triton::gpu::ConvertLayoutOp dynCastToSubGroupTranspose(Operation *op) {
  auto convertLayout = dyn_cast<triton::gpu::ConvertLayoutOp>(op);
  if (!convertLayout)
    return nullptr;

  if (!triton::gpu::intel::cvtIsSubGroupTranspose(
          convertLayout.getSrc().getType(),
          convertLayout.getResult().getType()))
    return nullptr;

  return convertLayout;
}

/// Check if `lhsOp` and `rhsOp` are safe to execute back-to-back sub-group
/// transpose layout conversions.
///
/// Sub-group transposes are implemented as follows:
///
/// - Each sub-group writes all the elements it is handling in a memory block
/// - Each sub-group reads all the elements it is handling from the same memory
/// region.
///
/// As there is no need to synchronize work-items in the same sub-group and we
/// know data won't be shared between sub-groups, executing these operations
/// back-to-back with no barriers in between is safe.
bool areSafeToOverlapSubGroupTransposeOps(Operation *lhsOp, Operation *rhsOp) {
  // Check both are lowered to sub-group transpose operations.
  auto lhsTranspose = dynCastToSubGroupTranspose(lhsOp);
  if (!lhsTranspose)
    return false;
  auto rhsTranspose = dynCastToSubGroupTranspose(rhsOp);
  if (!rhsTranspose)
    return false;

  // Check the types of source and result are the same, i.e., we are expressing
  // the same kind of transposition.
  if (lhsTranspose.getSrc().getType() != lhsTranspose.getSrc().getType() ||
      lhsTranspose.getResult().getType() != lhsTranspose.getResult().getType())
    return false;

  // Check both have the same offset and thus these operation can be overlapped.
  return lhsTranspose->getAttr("allocation.offset") ==
         rhsTranspose->getAttr("allocation.offset");
}
} // namespace
bool membarFilter(Operation *lhsOp, Operation *rhsOp, Allocation *allocation) {
  // For now, we only check these aren't layout conversions implemented as the
  // same sub-group transposition.
  assert(lhsOp && rhsOp && "Expecting valid operations");
  return areSafeToOverlapSubGroupTransposeOps(lhsOp, rhsOp);
}
} // namespace mlir::intel
