#ifndef TRITON_INTEL_ANALYSIS_REUSE_ANALYSIS_H
#define TRITON_INTEL_ANALYSIS_REUSE_ANALYSIS_H

#include "intel/include/Analysis/SpatialReuseAnalysis.h"
#include "intel/include/Analysis/TemporalReuseAnalysis.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::triton::gpu::intel {

/// Returns true iff the address computation of `ptr` consumes a tensor value
/// read from memory, i.e. the load is a gather/indirect access such as
/// `table[idx]` with `idx` loaded by the kernel. Scalar memory reads (e.g. a
/// per-program offset) do not count: they are uniform across the tile, so its
/// address pattern stays structured. Follows `scf.for` loop-carried values and
/// `scf.for`/`scf.if` results.
bool addressDependsOnLoadedTensor(Value ptr);

/// Thin union of SpatialReuseAnalysis and TemporalReuseAnalysis.
/// Spatial reuse: warps sharing coordinates on some output axis of the loaded
/// tensor. Temporal reuse: axes held across an enclosing loop.
/// `anyReuse(L)` returns true iff either analysis reports reuse for `L`, or,
/// for `tt.load`, iff the address is data-dependent (see
/// `addressDependsOnLoadedTensor`). Both legs derive reuse from the address
/// structure (layout, strides); a gathered address defeats that premise, so
/// "no reuse found" there means "unknown", and the conservative answer for a
/// suppress-side query is reuse. Inside loops the temporal leg already gets
/// this right (unknown strides classify as reuse); outside loops nothing
/// else catches it.
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
    if (spatial.hasCrossSubgroupReuse(op) || temporal.hasTemporalReuse(op))
      return true;
    if constexpr (std::is_same_v<OpTy, mlir::triton::LoadOp>)
      return addressDependsOnLoadedTensor(op.getPtr());
    return false;
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
