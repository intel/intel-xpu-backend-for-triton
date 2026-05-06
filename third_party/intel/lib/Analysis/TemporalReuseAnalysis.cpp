#include "intel/include/Analysis/TemporalReuseAnalysis.h"

#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "temporal-reuse-analysis"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
using tt::intel::StrideInfo;

namespace mlir::triton::gpu::intel {

SmallVector<bool> TemporalReuseAnalysis::classify(Operation *op,
                                                  Value ptr) const {
  LDBG("classify: " << *op << " ptr=" << ptr);

  auto innermost = op->getParentOfType<LoopLikeOpInterface>();
  if (!innermost)
    return {};

  StrideInfo *si = strideAnalysis.getStrideInfo(ptr);

  SmallVector<bool> result;
  for (auto loop = innermost; loop;
       loop = loop->getParentOfType<LoopLikeOpInterface>()) {
    // Case A: pointer loop-invariant w.r.t. this loop.
    if (loop.isDefinedOutsideOfLoop(ptr)) {
      LDBG("  depth " << result.size() << ": Case A (loop-invariant) -> reuse");
      result.push_back(true);
      continue;
    }

    if (!si) {
      // No stride info tracked at all -> pessimistic.
      LDBG("  depth " << result.size()
                      << ": no stride info -> pessimistic reuse");
      result.push_back(true);
      continue;
    }

    const StrideInfo::DimVectorT *ivStride = si->getIVStride(loop);
    if (!ivStride || ivStride->empty()) {
      // This loop not tracked for `ptr` -> pessimistic at this level.
      LDBG("  depth " << result.size()
                      << ": no IV-stride for this loop -> pessimistic reuse");
      result.push_back(true);
      continue;
    }

    // Case B vs Case C: any axis with IV stride <= 0 => reuse.
    bool reuse = false;
    for (unsigned d = 0, rank = ivStride->size(); d < rank; ++d) {
      if ((*ivStride)[d] <= 0) {
        reuse = true;
        break;
      }
    }
    LLVM_DEBUG({
      DBGS() << "  depth " << result.size()
             << (reuse ? ": Case B (partial-axis reuse)"
                       : ": Case C (pure streaming, no reuse)")
             << " ivStride=[";
      llvm::interleaveComma(*ivStride, llvm::dbgs());
      llvm::dbgs() << "]\n";
    });
    result.push_back(reuse);
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
