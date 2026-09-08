#ifndef TRITON_INTEL_ANALYSIS_UTILITY_H
#define TRITON_INTEL_ANALYSIS_UTILITY_H

#include "triton/Analysis/Utility.h"

namespace mlir::triton::gpu::intel {

bool isDpasToDotShortcut(RankedTensorType dpasTy, RankedTensorType dotTy);

/// Return whether the layout conversion from `srcTy` to `dstTy` can be
/// performed as a sub-group shuffle.
bool cvtIsSubGroupShuffle(RankedTensorType srcTy, RankedTensorType dstTy);
/// Return whether the layout conversion from `srcTy` to `dstTy` can be
/// performed as a sub-group transpose through local memory.
bool cvtIsSubGroupTranspose(RankedTensorType srcTy, RankedTensorType dstTy);
/// Return whether `type` is a valid element type for a fast sub-group
/// transpose.
bool isValidElementTypeForSubGroupTranspose(Type type);

// The following four queries used to live on core's `ReduceOpHelper`, but the
// Intel backend is their only consumer: core's own reduction lowering computes
// its scratch space from linear layouts instead. They are kept here so the
// shared analysis stays vendor-neutral.
//
// `helper` is passed in (rather than constructed here) so that the caller's
// existing helper - and its diagnostics - are reused unchanged.

/// Return whether the reduction needs no inter-warp communication.
bool isWarpSynchronous(ReduceOpHelper &helper, triton::ReduceOp op);

/// Return the shape of the shared memory space needed for the reduction.
SmallVector<unsigned> getScratchRepShape(ReduceOpHelper &helper,
                                         triton::ReduceOp op);

/// Return the source layout order with the reduction axis moved to the front.
SmallVector<unsigned> getOrderWithAxisAtBeginning(ReduceOpHelper &helper,
                                                  triton::ReduceOp op);

/// Return the size in bytes of the shared memory space needed for the
/// reduction.
unsigned getScratchSizeInBytesOld(ReduceOpHelper &helper, triton::ReduceOp op);

} // namespace mlir::triton::gpu::intel

#endif // TRITON_INTEL_ANALYSIS_UTILITY_H
