#ifndef TRITON_INTEL_ANALYSIS_REUSE_ANALYSIS_H
#define TRITON_INTEL_ANALYSIS_REUSE_ANALYSIS_H

#include "intel/include/Analysis/SpatialReuseAnalysis.h"
#include "intel/include/Analysis/TemporalReuseAnalysis.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::triton::gpu::intel {

/// Thin union of SpatialReuseAnalysis and TemporalReuseAnalysis.
/// Spatial reuse: warps sharing coordinates on some output axis of the loaded
/// tensor. Temporal reuse: axes held across an enclosing loop.
/// `anyReuse(L)` returns true iff either analysis reports reuse for `L`.
/// Consumers needing the richer per-dim or per-loop queries can reach the
/// underlying analyses via `getSpatial()` / `getTemporal()`.
class ReuseAnalysis {
public:
  ReuseAnalysis(ModuleOp m,
                mlir::triton::intel::ModuleStrideAnalysis &strideAnalysis)
      : spatial(m), temporal(strideAnalysis) {}

  template <typename OpTy,
            typename = std::enable_if_t<llvm::is_one_of<
                OpTy, mlir::triton::LoadOp, mlir::triton::DescriptorLoadOp,
                mlir::triton::DescriptorGatherOp>::value>>
  bool anyReuse(OpTy op) const {
    return spatial.hasCrossSubgroupReuse(op) || temporal.hasTemporalReuse(op);
  }

  /// Known reuse: spatial proves cross-subgroup reuse (non-empty known
  /// dims) OR temporal proves Held/Invariant on every enclosing loop.
  /// Either signal alone is sufficient, matching the candidate set:
  /// spatial catches the canonical DPAS-A pattern (load in K-loop, dot-
  /// operand encoding warp-broadcast on K) where the temporal side
  /// reports Streaming on K; temporal catches loop-invariant scalar
  /// broadcasts the spatial side cannot see.
  ///
  /// Unlike `anyReuse`, `knownReuse` requires a positive proof on at
  /// least one of the two sides — neither structural fallback paths in
  /// `SpatialReuseAnalysis` nor `OperandClass::Unknown` on the temporal
  /// side count. Suitable for forcing actions (e.g. setting
  /// EVICT_LAST); use `anyReuse` to suppress an action.
  template <typename OpTy,
            typename = std::enable_if_t<llvm::is_one_of<
                OpTy, mlir::triton::LoadOp, mlir::triton::DescriptorLoadOp,
                mlir::triton::DescriptorGatherOp>::value>>
  bool knownReuse(OpTy op) const {
    return spatial.knownCrossSubgroupReuse(op) ||
           temporal.provenTemporalReuse(op);
  }

  /// Accessors for consumers that need the richer per-dim or per-loop queries.
  const SpatialReuseAnalysis &getSpatial() const { return spatial; }
  const TemporalReuseAnalysis &getTemporal() const { return temporal; }

private:
  SpatialReuseAnalysis spatial;
  TemporalReuseAnalysis temporal;
};

} // namespace mlir::triton::gpu::intel

#endif // TRITON_INTEL_ANALYSIS_REUSE_ANALYSIS_H
