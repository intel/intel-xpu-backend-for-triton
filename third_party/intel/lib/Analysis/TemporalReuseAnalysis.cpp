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

// Per-operand classification at a single loop level.
enum class OperandClass {
  Invariant, // Case A: defined outside the loop.
  Held,      // Case B: at least one axis has IV-stride == 0 and no axis has
             // unknown IV-stride.
  Streaming, // Case C: every axis has IV-stride > 0.
  Unknown,   // No stride info / loop not tracked, or at least one axis has
             // unknown IV-stride (StrideInfo sentinel -1). Conservatively
             // treated as reuse by `classify`.
};

static OperandClass classifyOperandAtLoop(LoopLikeOpInterface loop, Value v,
                                          StrideInfo *si) {
  if (loop.isDefinedOutsideOfLoop(v))
    return OperandClass::Invariant;
  if (!si)
    return OperandClass::Unknown;
  const StrideInfo::DimVectorT *ivStride = si->getIVStride(loop);
  if (!ivStride || ivStride->empty())
    return OperandClass::Unknown;
  bool anyHeld = false;
  for (int64_t s : *ivStride) {
    if (s < 0)
      return OperandClass::Unknown;
    if (s == 0)
      anyHeld = true;
  }
  return anyHeld ? OperandClass::Held : OperandClass::Streaming;
}

SmallVector<bool> TemporalReuseAnalysis::classify(Operation *op,
                                                  ValueRange operands) const {
  LDBG("classify: " << *op << " #operands=" << operands.size());

  auto innermost = op->getParentOfType<LoopLikeOpInterface>();
  if (!innermost)
    return {};

  SmallVector<StrideInfo *> siList;
  siList.reserve(operands.size());
  for (Value v : operands)
    siList.push_back(strideAnalysis.getStrideInfo(v));

  SmallVector<bool> result;
  for (auto loop = innermost; loop;
       loop = loop->getParentOfType<LoopLikeOpInterface>()) {
    // Reuse iff every operand is invariant or holds at least one axis
    // fixed across this loop.  If any operand streams along every axis,
    // the effective address streams and we report no reuse.
    bool anyStreams = false;
    for (unsigned i = 0, e = operands.size(); i < e; ++i) {
      OperandClass c = classifyOperandAtLoop(loop, operands[i], siList[i]);
      if (c == OperandClass::Streaming) {
        anyStreams = true;
        break;
      }
    }
    LDBG("  depth " << result.size() << ": "
                    << (anyStreams ? "no reuse (some operand streams)"
                                   : "reuse"));
    result.push_back(!anyStreams);
  }
  return result;
}

SmallVector<bool>
TemporalReuseAnalysis::getReuseByLoopDepth(tt::LoadOp op) const {
  return classify(op, ValueRange{op.getPtr()});
}

SmallVector<bool>
TemporalReuseAnalysis::getReuseByLoopDepth(tt::DescriptorLoadOp op) const {
  SmallVector<Value> operands;
  operands.push_back(op.getDesc());
  operands.append(op.getIndices().begin(), op.getIndices().end());
  return classify(op, operands);
}

SmallVector<bool>
TemporalReuseAnalysis::getReuseByLoopDepth(tt::DescriptorGatherOp op) const {
  return classify(op,
                  ValueRange{op.getDesc(), op.getXOffsets(), op.getYOffset()});
}

template <typename OpT>
bool TemporalReuseAnalysis::hasTemporalReuseImpl(OpT op) const {
  SmallVector<bool> v = getReuseByLoopDepth(op);
  return llvm::any_of(v, [](bool b) { return b; });
}

bool TemporalReuseAnalysis::hasTemporalReuse(tt::LoadOp op) const {
  return hasTemporalReuseImpl(op);
}

bool TemporalReuseAnalysis::hasTemporalReuse(tt::DescriptorLoadOp op) const {
  return hasTemporalReuseImpl(op);
}

bool TemporalReuseAnalysis::hasTemporalReuse(tt::DescriptorGatherOp op) const {
  return hasTemporalReuseImpl(op);
}

} // namespace mlir::triton::gpu::intel
