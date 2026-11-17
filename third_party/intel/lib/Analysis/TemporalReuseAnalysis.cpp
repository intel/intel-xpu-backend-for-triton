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

// Extract the tile shape for the reuse analysis. For descriptor loads, use
// the descriptor's block shape (which matches the number of index operands)
// rather than the result shape (which may have fewer dimensions due to rank
// reduction). For other loads, use the result tensor shape. Returns {1} for
// scalar results so the per-axis rule degenerates correctly.
static SmallVector<int64_t> getTileShape(Operation *op) {
  assert(op->getNumResults() == 1 && "expected single-result load");
  if (auto descLoad = dyn_cast<tt::DescriptorLoadOp>(op)) {
    auto descType = descLoad.getDesc().getType();
    return llvm::to_vector(descType.getBlockType().getShape());
  }
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

// Per-operand `scalarAxis` derivation, replicated for every load op
// type. Encapsulates the wiring between the address-determining
// operand list and the descriptor block axis each scalar operand
// indexes (see comments below for the per-op-type semantics).
static SmallVector<std::optional<unsigned>> getScalarAxes(Operation *op,
                                                          ValueRange operands) {
  SmallVector<std::optional<unsigned>> scalarAxes(operands.size(),
                                                  std::nullopt);
  // - tt.load: single pointer-tensor operand, no scalarAxis.
  // - tt.descriptor_load: operand 0 is `desc` (tensor path); operands
  //   1..N are scalar indices, one per descriptor block axis.
  // - tt.descriptor_gather: operands are `desc` (tensor), `xOffsets`
  //   (rank-1 tensor -> rank mismatch -> Unknown path), `yOffset`
  //   (scalar indexing block axis 1).
  if (isa<tt::DescriptorLoadOp>(op)) {
    for (unsigned i = 1; i < operands.size(); ++i)
      scalarAxes[i] = i - 1;
  } else if (isa<tt::DescriptorGatherOp>(op)) {
    if (operands.size() >= 3)
      scalarAxes[2] = 1;
  }
  return scalarAxes;
}

// Build the full per-loop, per-operand classification matrix for `op`.
// Outer index: enclosing loop depth (0 = innermost). Inner index:
// position in `operands`. Empty result iff `op` has no enclosing loop.
//
// Both `hasTemporalReuse` (suppress-side) and `provenTemporalReuse`
// (force-side) reduce this matrix; they differ only in *how* they
// fold the per-operand classes per loop.
static SmallVector<SmallVector<OperandClass>>
classifyMatrix(tt::intel::ModuleStrideAnalysis &strideAnalysis, Operation *op,
               ValueRange operands) {
  LDBG("classifyMatrix: " << *op << " #operands=" << operands.size());

  auto innermost = op->getParentOfType<LoopLikeOpInterface>();
  if (!innermost)
    return {};

  SmallVector<int64_t> tileShape = getTileShape(op);

  SmallVector<StrideInfo *> siList;
  siList.reserve(operands.size());
  for (Value v : operands)
    siList.push_back(strideAnalysis.getStrideInfo(v));

  SmallVector<std::optional<unsigned>> scalarAxes = getScalarAxes(op, operands);

  SmallVector<SmallVector<OperandClass>> matrix;
  for (auto loop = innermost; loop;
       loop = loop->getParentOfType<LoopLikeOpInterface>()) {
    SmallVector<OperandClass> perOperand;
    perOperand.reserve(operands.size());
    for (unsigned i = 0, e = operands.size(); i < e; ++i)
      perOperand.push_back(classifyOperandAtLoop(loop, operands[i], siList[i],
                                                 tileShape, scalarAxes[i]));
    matrix.push_back(std::move(perOperand));
  }
  return matrix;
}

// Address-determining operands per load op type. Mirrors the per-op
// dispatch previously inlined in `classifyMatrix` callers and in
// `provenTemporalReuseImpl` — keeping it in one place ensures the two
// reductions classify the same operand set.
static SmallVector<Value> getAddressOperands(Operation *op) {
  SmallVector<Value> operands;
  if (auto load = dyn_cast<tt::LoadOp>(op)) {
    operands.push_back(load.getPtr());
  } else if (auto descLoad = dyn_cast<tt::DescriptorLoadOp>(op)) {
    operands.push_back(descLoad.getDesc());
    operands.append(descLoad.getIndices().begin(), descLoad.getIndices().end());
  } else if (auto gather = dyn_cast<tt::DescriptorGatherOp>(op)) {
    operands.push_back(gather.getDesc());
    operands.push_back(gather.getXOffsets());
    operands.push_back(gather.getYOffset());
  } else {
    llvm_unreachable("unsupported load op type");
  }
  return operands;
}

// Suppress-side fold: at each loop, report reuse unless *some* operand
// is `Streaming` (Unknown collapses to reuse — conservative).
static SmallVector<bool>
reduceForReuse(ArrayRef<SmallVector<OperandClass>> matrix) {
  SmallVector<bool> result;
  result.reserve(matrix.size());
  for (unsigned depth = 0, e = matrix.size(); depth < e; ++depth) {
    bool anyStreams = llvm::any_of(matrix[depth], [](OperandClass c) {
      return c == OperandClass::Streaming;
    });
    LDBG("  depth " << depth << ": "
                    << (anyStreams ? "no reuse (some operand streams)"
                                   : "reuse"));
    result.push_back(!anyStreams);
  }
  return result;
}

// Force-side fold: true iff *every* loop has *every* operand in
// {Invariant, Held}. Streaming OR Unknown at any (loop, operand) cell
// defeats the proof.
static bool reduceForProof(ArrayRef<SmallVector<OperandClass>> matrix) {
  if (matrix.empty())
    return false; // No enclosing loop -> no proof of reuse.
  return llvm::all_of(matrix, [](ArrayRef<OperandClass> perOperand) {
    return llvm::all_of(perOperand, [](OperandClass c) {
      return c == OperandClass::Invariant || c == OperandClass::Held;
    });
  });
}

template <typename OpT>
SmallVector<bool> TemporalReuseAnalysis::getReuseByLoopDepthImpl(OpT op) const {
  return reduceForReuse(
      classifyMatrix(strideAnalysis, op, getAddressOperands(op)));
}

SmallVector<bool>
TemporalReuseAnalysis::getReuseByLoopDepth(tt::LoadOp op) const {
  return getReuseByLoopDepthImpl(op);
}

SmallVector<bool>
TemporalReuseAnalysis::getReuseByLoopDepth(tt::DescriptorLoadOp op) const {
  return getReuseByLoopDepthImpl(op);
}

SmallVector<bool>
TemporalReuseAnalysis::getReuseByLoopDepth(tt::DescriptorGatherOp op) const {
  return getReuseByLoopDepthImpl(op);
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

template <typename OpT>
bool TemporalReuseAnalysis::provenTemporalReuseImpl(OpT op) const {
  return reduceForProof(
      classifyMatrix(strideAnalysis, op, getAddressOperands(op)));
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
