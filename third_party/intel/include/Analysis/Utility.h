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

} // namespace mlir::triton::gpu::intel

#endif // TRITON_INTEL_ANALYSIS_UTILITY_H
