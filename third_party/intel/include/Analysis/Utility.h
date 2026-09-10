#ifndef TRITON_INTEL_ANALYSIS_UTILITY_H
#define TRITON_INTEL_ANALYSIS_UTILITY_H

#include <optional>

#include "triton/Analysis/Utility.h"

namespace mlir::triton::gpu::intel {

bool isDpasToDotShortcut(RankedTensorType dpasTy, RankedTensorType dotTy);

struct SubGroupReinterpretPackInfo {
  bool isPack;
  unsigned packedRegisterSize;
};

std::optional<LinearLayout>
getReinterpretCastMapping(MLIRContext *ctx, const LinearLayout &srcLayout,
                          const LinearLayout &dstLayout);

/// Return pack/unpack metadata for a valid sub-group reinterpret mapping.
std::optional<SubGroupReinterpretPackInfo>
getSubGroupReinterpretPackInfo(MLIRContext *ctx,
                               const LinearLayout &conversion);

/// Return whether the layout conversion from `srcTy` to `dstTy` can be
/// performed as a sub-group shuffle.
bool cvtIsSubGroupShuffle(RankedTensorType srcTy, RankedTensorType dstTy);
/// Return whether the layout conversion from `srcTy` to `dstTy` can be
/// performed as a sub-group transpose through local memory.
bool cvtIsSubGroupTranspose(RankedTensorType srcTy, RankedTensorType dstTy);
/// Return whether the layout conversion from `srcTy` to `dstTy` can be
/// performed as a sub-group bitcast shuffle (reinterpret cast).
bool cvtIsSubGroupReinterpret(RankedTensorType srcTy, RankedTensorType dstTy);
/// Return whether `type` is a valid element type for a fast sub-group
/// transpose.
bool isValidElementTypeForSubGroupTranspose(Type type);

} // namespace mlir::triton::gpu::intel

#endif // TRITON_INTEL_ANALYSIS_UTILITY_H
