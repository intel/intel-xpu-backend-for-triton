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
            typename = std::enable_if_t<
                llvm::is_one_of<OpTy, mlir::triton::LoadOp,
                                mlir::triton::DescriptorLoadOp,
                                mlir::triton::DescriptorGatherOp>::value>>
  bool anyReuse(OpTy op) const {
    return spatial.hasCrossSubgroupReuse(op) || temporal.hasTemporalReuse(op);
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
