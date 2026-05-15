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
  Invariant, // Case A: defined outside the loop (same address every
             // iteration).
  Held,      // Case B: every axis has per-iteration advance strictly
             // less than the tile extent on that axis -> successive
             // tiles overlap in memory -> cache-line reuse possible.
  Streaming, // Case C: at least one axis has per-iteration advance
             // >= tile extent on that axis -> tiles disjoint on that
             // axis -> no cache-line reuse.
  Unknown,   // No stride info for the operand, or at least one axis has
             // unknown advance (StrideInfo sentinel -1 or dynamic loop
             // step).  Conservatively treated as reuse by `classify`.
};

// Extract the element-shape of a load's result tensor.  Returns {1}
// for scalar results so the per-axis rule degenerates correctly
// (|adv| < 1 iff adv == 0, which is Held only for held scalar ops).
static SmallVector<int64_t> getTileShape(Operation *op) {
  assert(op->getNumResults() == 1 && "expected single-result load");
  Type ty = op->getResult(0).getType();
  if (auto tensorTy = dyn_cast<RankedTensorType>(ty))
    return llvm::to_vector(tensorTy.getShape());
  return {1};
}

// Per-operand classification at a single loop level.
//
// `tileShape` is the shape of the load's *result tensor* (not the operand).
// `scalarAxis` is set when `v` is a scalar operand indexing a specific
// tile axis (e.g. a `descriptor_load` scalar index at operand position i
// indexes descriptor block axis i).  For scalar operands, the single
// StrideInfo rank-1 per-iteration advance is compared against
// `tileShape[*scalarAxis]`.
static OperandClass
classifyOperandAtLoop(LoopLikeOpInterface loop, Value v, StrideInfo *si,
                      ArrayRef<int64_t> tileShape,
                      std::optional<unsigned> scalarAxis = std::nullopt) {
  if (loop.isDefinedOutsideOfLoop(v))
    return OperandClass::Invariant;
  if (!si)
    return OperandClass::Unknown;

  // Canonical StrideInfo contract (see `StrideInfo.h`):
  // `getIVStride(loop)` returns `nullptr` iff the value does not depend
  // on `loop`'s IV (every axis held).  That precisely matches `Held`.
  const StrideInfo::DimVectorT *ivStride = si->getIVStride(loop);
  if (!ivStride)
    return OperandClass::Held;
  if (ivStride->empty())
    return OperandClass::Unknown;

  // Scalar-index path: compare the single rank-1 per-iteration advance
  // against the tile extent on the axis this operand indexes.
  if (scalarAxis) {
    assert(*scalarAxis < tileShape.size() && "scalarAxis out of range");
    std::optional<int64_t> adv = si->getPerIterationIVStride(loop, 0);
    if (!adv)
      return OperandClass::Unknown;
    int64_t a = *adv < 0 ? -*adv : *adv;
    return a >= tileShape[*scalarAxis] ? OperandClass::Streaming
                                       : OperandClass::Held;
  }

  // Tensor-operand path: rank must match tile shape.  If not, we can't
  // apply the per-axis rule precisely — fall back to Unknown (safe: maps
  // to reuse in `classify`).
  if (ivStride->size() != tileShape.size())
    return OperandClass::Unknown;

  // Axis-disjoint rule: reuse iff every axis has |advance| < tile extent.
  // If any axis is disjoint (|adv| >= tile), successive tiles share no
  // cache lines on that axis -> no reuse overall.
  bool anyUnknown = false;
  for (unsigned i = 0, e = tileShape.size(); i < e; ++i) {
    std::optional<int64_t> adv = si->getPerIterationIVStride(loop, i);
    if (!adv) {
      anyUnknown = true;
      continue;
    }
    int64_t a = *adv < 0 ? -*adv : *adv;
    if (a >= tileShape[i])
      return OperandClass::Streaming;
  }
  return anyUnknown ? OperandClass::Unknown : OperandClass::Held;
}

SmallVector<bool> TemporalReuseAnalysis::classify(Operation *op,
                                                  ValueRange operands) const {
  LDBG("classify: " << *op << " #operands=" << operands.size());

  auto innermost = op->getParentOfType<LoopLikeOpInterface>();
  if (!innermost)
    return {};

  SmallVector<int64_t> tileShape = getTileShape(op);

  SmallVector<StrideInfo *> siList;
  siList.reserve(operands.size());
  for (Value v : operands)
    siList.push_back(strideAnalysis.getStrideInfo(v));

  // Determine the per-operand scalarAxis mapping based on op type.
  // - tt.load: single pointer-tensor operand, no scalarAxis.
  // - tt.descriptor_load: operand 0 is `desc` (tensor path); operands
  //   1..N are scalar indices, one per descriptor block axis.
  // - tt.descriptor_gather: operands are `desc` (tensor), `xOffsets`
  //   (rank-1 tensor -> rank mismatch -> Unknown path), `yOffset`
  //   (scalar indexing block axis 1).
  SmallVector<std::optional<unsigned>> scalarAxes(operands.size(),
                                                  std::nullopt);
  if (isa<tt::DescriptorLoadOp>(op)) {
    for (unsigned i = 1; i < operands.size(); ++i)
      scalarAxes[i] = i - 1;
  } else if (isa<tt::DescriptorGatherOp>(op)) {
    // operands: [desc, xOffsets, yOffset]
    // yOffset is at index 2 and indexes axis 1 (column) of the 1-row
    // descriptor block.
    if (operands.size() >= 3)
      scalarAxes[2] = 1;
  }

  SmallVector<bool> result;
  for (auto loop = innermost; loop;
       loop = loop->getParentOfType<LoopLikeOpInterface>()) {
    bool anyStreams = false;
    for (unsigned i = 0, e = operands.size(); i < e; ++i) {
      OperandClass c = classifyOperandAtLoop(loop, operands[i], siList[i],
                                             tileShape, scalarAxes[i]);
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

// Honest single-loop check: returns true iff every address-determining
// operand classifies as Invariant or Held at `loop`. Returns false on
// the first operand that classifies as Streaming OR Unknown. Does NOT
// go through `classify`'s aggregated per-loop result (which collapses
// Streaming + Unknown into Streaming, masking the Unknown evidence —
// see the header comment on `provenTemporalReuse`).
static bool everyOperandIsInvariantOrHeld(
    LoopLikeOpInterface loop, Operation *op, ValueRange operands,
    ArrayRef<StrideInfo *> siList, ArrayRef<int64_t> tileShape,
    ArrayRef<std::optional<unsigned>> scalarAxes) {
  for (unsigned i = 0, e = operands.size(); i < e; ++i) {
    OperandClass c = classifyOperandAtLoop(loop, operands[i], siList[i],
                                           tileShape, scalarAxes[i]);
    if (c == OperandClass::Streaming || c == OperandClass::Unknown)
      return false;
  }
  return true;
}

template <typename OpT>
bool TemporalReuseAnalysis::provenTemporalReuseImpl(OpT op) const {
  // Derive address-determining operands the same way `classify` does:
  // - tt.load: pointer.
  // - tt.descriptor_load: descriptor + scalar indices.
  // - tt.descriptor_gather: descriptor + xOffsets + yOffset.
  SmallVector<Value> operands;
  if constexpr (std::is_same_v<OpT, tt::LoadOp>) {
    operands.push_back(op.getPtr());
  } else if constexpr (std::is_same_v<OpT, tt::DescriptorLoadOp>) {
    operands.push_back(op.getDesc());
    operands.append(op.getIndices().begin(), op.getIndices().end());
  } else if constexpr (std::is_same_v<OpT, tt::DescriptorGatherOp>) {
    operands.push_back(op.getDesc());
    operands.push_back(op.getXOffsets());
    operands.push_back(op.getYOffset());
  }

  Operation *opPtr = op.getOperation();
  auto innermost = opPtr->template getParentOfType<LoopLikeOpInterface>();
  if (!innermost)
    return false; // No enclosing loop -> no proof of reuse.

  SmallVector<int64_t> tileShape = getTileShape(opPtr);

  SmallVector<StrideInfo *> siList;
  siList.reserve(operands.size());
  for (Value v : operands)
    siList.push_back(strideAnalysis.getStrideInfo(v));

  // Replicate scalarAxis derivation from `classify`.
  SmallVector<std::optional<unsigned>> scalarAxes(operands.size(),
                                                  std::nullopt);
  if (isa<tt::DescriptorLoadOp>(opPtr)) {
    for (unsigned i = 1; i < operands.size(); ++i)
      scalarAxes[i] = i - 1;
  } else if (isa<tt::DescriptorGatherOp>(opPtr)) {
    if (operands.size() >= 3)
      scalarAxes[2] = 1;
  }

  for (auto loop = innermost; loop;
       loop = loop->getParentOfType<LoopLikeOpInterface>()) {
    if (!everyOperandIsInvariantOrHeld(loop, opPtr, operands, siList, tileShape,
                                       scalarAxes))
      return false;
  }
  return true;
}

bool TemporalReuseAnalysis::provenTemporalReuse(tt::LoadOp op) const {
  return provenTemporalReuseImpl(op);
}

bool TemporalReuseAnalysis::provenTemporalReuse(tt::DescriptorLoadOp op) const {
  return provenTemporalReuseImpl(op);
}

bool TemporalReuseAnalysis::provenTemporalReuse(
    tt::DescriptorGatherOp op) const {
  return provenTemporalReuseImpl(op);
}

} // namespace mlir::triton::gpu::intel
