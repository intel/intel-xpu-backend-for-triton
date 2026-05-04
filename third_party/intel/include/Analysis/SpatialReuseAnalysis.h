#ifndef TRITON_INTEL_ANALYSIS_SPATIAL_REUSE_ANALYSIS_H
#define TRITON_INTEL_ANALYSIS_SPATIAL_REUSE_ANALYSIS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::triton::gpu::intel {

namespace tt = ::mlir::triton;

/// Identify the output tensor dimensions on which all warps read the
/// same coordinate -- i.e. the warp id does not move the access along
/// that axis, so every warp in the workgroup touches the same elements
/// there. Those axes are candidates for L1 cross-subgroup reuse.
///
/// Mechanically: in the tensor's LinearLayout, an output dim is
/// "warp-invariant" iff every basis vector for the "warp" input dim is
/// zero on that output dim (changing the warp id does not change the
/// coordinate).
///
/// Example: a DPAS dot-operand-A layout with warpsPerCTA = [4, 1]:
///   - warps tile the M dimension (dim 0) → different warps access
///     different M rows → dim 0 is NOT warp-invariant.
///   - warps do NOT tile the K dimension (dim 1) → every warp reads
///     the same K columns → dim 1 IS warp-invariant.
///   - getWarpInvariantOutDims returns {1}.
///
/// Two entry points:
///   - getWarpInvariantOutDims(...) -- structural primitive, returns
///     the set of warp-invariant out-dim indices.
///   - hasCrossSubgroupReuse(...) -- policy wrapper, returns
///     !getWarpInvariantOutDims(...).empty().
///
/// Generality invariants:
///   1. Pure function of the tensor type (no module walk / no side tables).
///   2. Conservative default: returns the full set {0, ..., rank-1} on
///      non-encoded, non-power-of-2, or layouts lacking a "warp" in-dim
///      (assume sharing on every axis when we cannot prove otherwise).
class SpatialReuseAnalysis {
public:
  explicit SpatialReuseAnalysis(MLIRContext *ctx) : ctx(ctx) {}

  /// Primary structural query: indices of output dims with zero warp
  /// basis. Empty result means warps strictly partition every axis.
  SmallVector<unsigned> getWarpInvariantOutDims(RankedTensorType ty) const;

  /// Policy wrapper: any cross-subgroup reuse at all?
  bool hasCrossSubgroupReuse(RankedTensorType ty) const;

  /// Convenience overloads for read-side memory ops.
  bool hasCrossSubgroupReuse(tt::LoadOp op) const;
  bool hasCrossSubgroupReuse(tt::DescriptorLoadOp op) const;
  bool hasCrossSubgroupReuse(tt::DescriptorGatherOp op) const;
  SmallVector<unsigned> getWarpInvariantOutDims(tt::LoadOp op) const;
  SmallVector<unsigned> getWarpInvariantOutDims(tt::DescriptorLoadOp op) const;
  SmallVector<unsigned>
  getWarpInvariantOutDims(tt::DescriptorGatherOp op) const;

private:
  MLIRContext *ctx;
};

} // namespace mlir::triton::gpu::intel

#endif // TRITON_INTEL_ANALYSIS_SPATIAL_REUSE_ANALYSIS_H
