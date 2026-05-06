#include "intel/include/Analysis/TemporalReuseAnalysis.h"

#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "temporal-reuse-analysis"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;

namespace mlir::triton::gpu::intel {

TemporalReuseAnalysis::TemporalReuseAnalysis(ModuleOp /*m*/) {}

// TODO(temporal-reuse): Distinguish Case B (tile stays on some axis across
// iterations, true temporal reuse) from Case C (tile streams to a fresh
// cache-line region each iteration, no reuse). Case C detection requires
// per-tensor-axis stride *with respect to a specified loop IV* — not the
// per-tensor-axis spatial stride that `StrideInfo` computes. Concretely,
// `tt.addptr %splat_base, %splat(iv * 128)` has spatial stride [0] (splat
// of one scalar per lane) yet has IV stride 128 — every iteration touches
// a fresh 128-element tile, no reuse. Implementing this correctly means
// adding an IVStrideInfo lattice + sparse dataflow analysis parameterised
// by a LoopLikeOpInterface; see the P2 sub-plan
// `annotate-cc-p2-temporal-reuse.md` "Known limitation — deferred Case C
// precision" section for the design sketch.
// Until this lands, every enclosing loop level that is not Case A
// conservatively reports reuse.
SmallVector<bool> TemporalReuseAnalysis::classify(Operation *op,
                                                  Value ptr) const {
  LDBG("classify: " << *op);

  auto innermost = op->getParentOfType<LoopLikeOpInterface>();
  if (!innermost) {
    LDBG("  no enclosing loop -> {}");
    return {};
  }

  SmallVector<bool> result;
  for (LoopLikeOpInterface loop = innermost; loop;
       loop = loop->getParentOfType<LoopLikeOpInterface>()) {
    if (loop.isDefinedOutsideOfLoop(ptr)) {
      LDBG("  depth " << result.size() << ": loop-invariant -> reuse");
    } else {
      LDBG("  depth " << result.size()
                      << ": non-invariant -> conservative reuse (Case B/C "
                         "not distinguished)");
    }
    // Both branches report `true`. See the TODO above this function.
    result.push_back(true);
  }
  return result;
}

SmallVector<bool>
TemporalReuseAnalysis::getReuseByLoopDepth(tt::LoadOp op) const {
  return classify(op, op.getPtr());
}

SmallVector<bool>
TemporalReuseAnalysis::getReuseByLoopDepth(tt::DescriptorLoadOp op) const {
  return classify(op, op.getDesc());
}

SmallVector<bool>
TemporalReuseAnalysis::getReuseByLoopDepth(tt::DescriptorGatherOp op) const {
  return classify(op, op.getDesc());
}

bool TemporalReuseAnalysis::hasTemporalReuse(tt::LoadOp op) const {
  SmallVector<bool> v = getReuseByLoopDepth(op);
  return llvm::any_of(v, [](bool b) { return b; });
}

bool TemporalReuseAnalysis::hasTemporalReuse(tt::DescriptorLoadOp op) const {
  SmallVector<bool> v = getReuseByLoopDepth(op);
  return llvm::any_of(v, [](bool b) { return b; });
}

bool TemporalReuseAnalysis::hasTemporalReuse(tt::DescriptorGatherOp op) const {
  SmallVector<bool> v = getReuseByLoopDepth(op);
  return llvm::any_of(v, [](bool b) { return b; });
}

} // namespace mlir::triton::gpu::intel
