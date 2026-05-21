#ifndef TRITON_INTEL_ANALYSIS_SPATIAL_REUSE_ANALYSIS_H
#define TRITON_INTEL_ANALYSIS_SPATIAL_REUSE_ANALYSIS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <type_traits>

namespace mlir::triton::gpu::intel {

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
  explicit SpatialReuseAnalysis(ModuleOp module) : ctx(module.getContext()) {}

  /// Primary structural query: indices of output dims with zero warp
  /// basis. Empty result means warps strictly partition every axis.
  SmallVector<unsigned> getWarpInvariantOutDims(RankedTensorType ty) const;

  /// Policy wrapper: any cross-subgroup reuse at all?
  bool hasCrossSubgroupReuse(RankedTensorType ty) const;

  /// Convenience overloads for read-side memory ops.
  bool hasCrossSubgroupReuse(mlir::triton::LoadOp op) const;
  bool hasCrossSubgroupReuse(mlir::triton::DescriptorLoadOp op) const;
  bool hasCrossSubgroupReuse(mlir::triton::DescriptorGatherOp op) const;
  SmallVector<unsigned> getWarpInvariantOutDims(mlir::triton::LoadOp op) const;
  SmallVector<unsigned>
  getWarpInvariantOutDims(mlir::triton::DescriptorLoadOp op) const;
  SmallVector<unsigned>
  getWarpInvariantOutDims(mlir::triton::DescriptorGatherOp op) const;

  /// Return the warp-invariant out-dim set ONLY when the underlying
  /// LinearLayout was actually inspected (no fallback was taken).
  /// Returns std::nullopt when:
  ///   - encoding is null,
  ///   - any shape dim is non-power-of-2,
  ///   - the layout has no "warp" in-dim,
  ///   - load type is not a RankedTensorType (scalar loads).
  ///
  /// Returned dims are warp-broadcast under the inspected LinearLayout
  /// (their lane and register bases vary but the warp basis is zero).
  /// For distributed encodings this generally implies all warps in a
  /// CTA see the same logical out-dim coordinate (and, in turn, the
  /// same address modulo the encoding's element layout), but this is
  /// LAYOUT-DERIVED EVIDENCE, NOT A PROOF OF IDENTICAL ADDRESSES OR
  /// CACHE LINES. Sufficient signal to motivate EVICT_LAST on the
  /// canonical DPAS-operand-load pattern; not sufficient to assert
  /// reuse on arbitrary encodings without a separate same-address
  /// argument.
  ///
  /// Callers that *force* a positive action (e.g. setting EVICT_LAST)
  /// must use this accessor; the existing getWarpInvariantOutDims
  /// returns a conservative full set on fallback paths and is suitable
  /// only for *suppressing* a positive action.
  std::optional<SmallVector<unsigned>>
  knownWarpInvariantOutDims(RankedTensorType ty) const;

  /// Op-result overload for `tt.load` (result may be scalar).
  std::optional<SmallVector<unsigned>>
  knownWarpInvariantOutDims(mlir::triton::LoadOp op) const;

  /// Op-result overload for ops whose result type is constrained to be
  /// a `RankedTensorType` by the op definition (DescriptorLoadOp,
  /// DescriptorGatherOp). Templated and SFINAE-restricted so it cannot
  /// silently match unrelated op types.
  template <
      typename OpTy,
      std::enable_if_t<llvm::is_one_of<OpTy, mlir::triton::DescriptorLoadOp,
                                       mlir::triton::DescriptorGatherOp>::value,
                       int> = 0>
  std::optional<SmallVector<unsigned>>
  knownWarpInvariantOutDims(OpTy op) const {
    return knownWarpInvariantOutDims(cast<RankedTensorType>(op.getType()));
  }

  /// Convenience: known cross-subgroup reuse. Returns true iff
  /// knownWarpInvariantOutDims has a non-empty value. Single template
  /// dispatches across RankedTensorType / LoadOp / DescriptorLoadOp /
  /// DescriptorGatherOp via the corresponding knownWarpInvariantOutDims
  /// overload.
  template <typename T> bool knownCrossSubgroupReuse(T arg) const {
    std::optional<SmallVector<unsigned>> dims = knownWarpInvariantOutDims(arg);
    return dims.has_value() && !dims->empty();
  }

  /// Returns the warp-broadcast factor: the number of warps that map to
  /// the same (lane, register) coordinate of the tensor — i.e., 2^k
  /// where k is the number of all-zero basis vectors of the "warp"
  /// in-dim in the LinearLayout. Factor == 1 means warps strictly
  /// partition the tensor (no broadcast); factor >= 2 means at least 2
  /// warps share the same address.
  ///
  /// Same fallback semantics as `knownWarpInvariantOutDims`: returns
  /// `std::nullopt` when the encoding is null, any shape dim is non-
  /// power-of-2, the layout has no "warp" in-dim, or the load has no
  /// RankedTensorType. Callers that *force* a positive action (e.g.,
  /// gating EVICT_LAST on broadcast >= 2) must use this accessor.
  std::optional<unsigned> knownWarpBroadcastFactor(RankedTensorType ty) const;

  /// Op-result overload for `tt.load` (result may be scalar).
  std::optional<unsigned>
  knownWarpBroadcastFactor(mlir::triton::LoadOp op) const;

  /// Op-result overload for descriptor-load-like ops (see corresponding
  /// `knownWarpInvariantOutDims` template above).
  template <
      typename OpTy,
      std::enable_if_t<llvm::is_one_of<OpTy, mlir::triton::DescriptorLoadOp,
                                       mlir::triton::DescriptorGatherOp>::value,
                       int> = 0>
  std::optional<unsigned> knownWarpBroadcastFactor(OpTy op) const {
    return knownWarpBroadcastFactor(cast<RankedTensorType>(op.getType()));
  }

private:
  MLIRContext *ctx;
};

} // namespace mlir::triton::gpu::intel

#endif // TRITON_INTEL_ANALYSIS_SPATIAL_REUSE_ANALYSIS_H
