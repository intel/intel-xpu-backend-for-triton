#ifndef TRITON_INTEL_ANALYSIS_TEMPORAL_REUSE_ANALYSIS_H
#define TRITON_INTEL_ANALYSIS_TEMPORAL_REUSE_ANALYSIS_H

#include "intel/include/Analysis/StrideInfo.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::triton::gpu::intel {

/// Determine whether a load op (`tt.load`, `tt.descriptor_load`, or
/// `tt.descriptor_gather`) inside a loop touches the same cache lines on
/// successive iterations of its enclosing loops.
///
/// The analysis walks outward from each load through its enclosing loops.
/// For each enclosing loop L (innermost first), it appends one bool to the
/// result vector:
///   * Case A — operand is loop-invariant w.r.t. L
///     (`loop.isDefinedOutsideOfLoop(v)`). Report reuse.
///   * Case B (Held) — every axis has per-iteration advance strictly
///     less than the tile extent on that axis.  Successive tiles
///     overlap on every axis, so cache-line reuse is possible.
///     Report reuse.
///   * Case C (Streaming) — at least one axis has per-iteration
///     advance >= tile extent on that axis.  Successive tiles are
///     disjoint on that axis, so no cache-line reuse exists.
///     Report no reuse.
///
/// Axis-disjoint rule origin: two axis-aligned tiles overlap iff their
/// intervals overlap on every axis, and intervals overlap iff the
/// shift is strictly less than the extent.
///
/// Advance is reported at element granularity via
/// `StrideInfo::getPerIterationIVStride`, which folds in the loop's
/// constant step.  Dynamic-step loops yield unknown advance and
/// classify as Unknown (conservative reuse).  Sub-cache-line tiles
/// may report false "no reuse" at tile edges — safe direction.
///
/// Unknown handling (conservative): if `StrideInfo` is absent for the
/// operand, the loop has a dynamic (non-constant) step, any axis has
/// unknown advance (StrideInfo sentinel -1), or the StrideInfo rank
/// does not match the tile rank, the operand is classified as
/// "unknown" (distinct from the "held" proof case) and `classify`
/// conservatively reports reuse for L. This errs on the side of
/// preserving loads the analysis cannot prove stream.
///
/// `hasTemporalReuse` is the policy wrapper `any_of(getReuseByLoopDepth)`.
///
/// Examples — load with reuse vs load without reuse:
///
/// Example — small-advance sliding access (reuse). Tile [32, 32],
/// advance [1, 1] per iteration.  Successive tiles overlap on every
/// axis; cache-line reuse is possible.  hasTemporalReuse = true.
///
///   scf.for %i = %c0 to %N step %c1 iter_args(%p = %p_init) {
///     %t = tt.load %p : tensor<32x32 x !tt.ptr<f16>>
///     %next = tt.addptr %p, %one_on_each_axis
///     scf.yield %next
///   }
///
/// Example — full-tile advance (no reuse). Same tile [32, 32] but
/// advance [0, 32].  Axis-1 advance equals tile extent, so successive
/// tiles cover disjoint columns.  hasTemporalReuse = false.
class TemporalReuseAnalysis {
public:
  explicit TemporalReuseAnalysis(
      mlir::triton::intel::ModuleStrideAnalysis &strideAnalysis)
      : strideAnalysis(strideAnalysis) {}

  /// Structural query: one bool per enclosing loop (innermost first,
  /// outermost last). Empty vector => load is not inside any loop.
  SmallVector<bool> getReuseByLoopDepth(mlir::triton::LoadOp op) const;
  SmallVector<bool>
  getReuseByLoopDepth(mlir::triton::DescriptorLoadOp op) const;
  SmallVector<bool>
  getReuseByLoopDepth(mlir::triton::DescriptorGatherOp op) const;

  /// Policy wrapper: true iff any enclosing loop level has temporal reuse.
  bool hasTemporalReuse(mlir::triton::LoadOp op) const;
  bool hasTemporalReuse(mlir::triton::DescriptorLoadOp op) const;
  bool hasTemporalReuse(mlir::triton::DescriptorGatherOp op) const;

private:
  mlir::triton::intel::ModuleStrideAnalysis &strideAnalysis;

  /// Shared implementation: classify every enclosing loop of `op` given the
  /// set of operands that contribute to the effective address of the load
  /// (e.g. the pointer for `tt.load`; the descriptor + all scalar/tensor
  /// index operands for `tt.descriptor_load` and `tt.descriptor_gather`).
  /// A load has reuse at a given loop level iff every address-determining
  /// operand is either loop-invariant or holds at least one tensor axis
  /// fixed across iterations; conversely, if *any* operand advances along
  /// every one of its axes, the effective address streams and the analysis
  /// reports no reuse.
  SmallVector<bool> classify(Operation *op, ValueRange operands) const;

  template <typename OpT> bool hasTemporalReuseImpl(OpT op) const;
};

} // namespace mlir::triton::gpu::intel

#endif // TRITON_INTEL_ANALYSIS_TEMPORAL_REUSE_ANALYSIS_H
