#include "intel/include/Dialect/Triton/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-intel-narrow-broadcast-loads"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

STATISTIC(NumNarrowed, "Number of row-broadcast loads narrowed");

using namespace mlir;
namespace tt = mlir::triton;

namespace mlir::triton::intel {
#define GEN_PASS_DEF_TRITONINTELNARROWBROADCASTLOADS
#include "intel/include/Dialect/Triton/Transforms/Passes.h.inc"
} // namespace mlir::triton::intel

namespace {

/// Bounds the walks that look for the dot an epilogue combines with.
constexpr unsigned MaxDepth = 8;

/// A mask leaf `splat(base) + make_range(0, R) <s bound`, broadcast to RxN.
struct RowPrefix {
  Value base;
  Value bound; // splat-like; its uniform value is the scalar bound
};

static bool hasShape(Value v, ArrayRef<int64_t> shape) {
  auto ty = dyn_cast<RankedTensorType>(v.getType());
  return ty && ty.getShape() == shape;
}

static bool isSplatConstant(Value v) {
  DenseElementsAttr attr;
  return matchPattern(v, m_Constant(&attr)) && attr.isSplat();
}

static bool isSplatLike(Value v) {
  return v.getDefiningOp<tt::SplatOp>() || isSplatConstant(v);
}

static Value getSplatScalar(OpBuilder &b, Location loc, Value v) {
  if (auto splat = v.getDefiningOp<tt::SplatOp>())
    return splat.getSrc();
  DenseElementsAttr attr;
  matchPattern(v, m_Constant(&attr));
  return arith::ConstantOp::create(
      b, loc, cast<TypedAttr>(attr.getSplatValue<Attribute>()));
}

/// Returns `b` if \p idx is `splat(b) + make_range(0, n)` in either order.
static Value matchRowIndex(Value idx) {
  auto add = idx.getDefiningOp<arith::AddIOp>();
  if (!add)
    return {};
  for (auto [lhs, rhs] : {std::pair{add.getLhs(), add.getRhs()},
                          std::pair{add.getRhs(), add.getLhs()}}) {
    auto splat = lhs.getDefiningOp<tt::SplatOp>();
    auto range = rhs.getDefiningOp<tt::MakeRangeOp>();
    if (splat && range && range.getStart() == 0)
      return splat.getSrc();
  }
  return {};
}

/// Matches `broadcast(expand_dims(idx < s, 1))` or
/// `broadcast(expand_dims(idx, 1) < s)`, with `sgt` and swapped operands
/// normalized to `slt`.
static std::optional<RowPrefix> matchRowPrefix(Value leaf, int64_t rows) {
  auto bc = leaf.getDefiningOp<tt::BroadcastOp>();
  if (!bc || !hasShape(bc.getSrc(), {rows, 1}))
    return std::nullopt;
  Value x = bc.getSrc();
  bool comparesExpanded = true;
  if (auto ed = x.getDefiningOp<tt::ExpandDimsOp>()) {
    if (ed.getAxis() != 1)
      return std::nullopt;
    x = ed.getSrc();
    comparesExpanded = false;
  }
  auto cmp = x.getDefiningOp<arith::CmpIOp>();
  if (!cmp)
    return std::nullopt;
  Value idx = cmp.getLhs(), bound = cmp.getRhs();
  if (cmp.getPredicate() == arith::CmpIPredicate::sgt)
    std::swap(idx, bound);
  else if (cmp.getPredicate() != arith::CmpIPredicate::slt)
    return std::nullopt;
  if (comparesExpanded) {
    auto ed = idx.getDefiningOp<tt::ExpandDimsOp>();
    if (!ed || ed.getAxis() != 1)
      return std::nullopt;
    idx = ed.getSrc();
  }
  Value base = matchRowIndex(idx);
  if (!base || !isSplatLike(bound))
    return std::nullopt;
  return RowPrefix{base, bound};
}

/// A leaf that does not vary along the broadcast rows: a broadcast from 1xN,
/// a splat, or a splat constant.
static bool isColumnLeaf(Value leaf, ArrayRef<int64_t> narrowShape) {
  if (auto bc = leaf.getDefiningOp<tt::BroadcastOp>())
    return hasShape(bc.getSrc(), narrowShape);
  return isSplatLike(leaf);
}

static Value narrowColumnLeaf(OpBuilder &b, Location loc, Value leaf,
                              RankedTensorType narrowTy) {
  if (auto bc = leaf.getDefiningOp<tt::BroadcastOp>())
    return bc.getSrc();
  if (auto splat = leaf.getDefiningOp<tt::SplatOp>())
    return tt::SplatOp::create(b, loc, narrowTy, splat.getSrc());
  DenseElementsAttr attr;
  matchPattern(leaf, m_Constant(&attr));
  return arith::ConstantOp::create(
      b, loc, narrowTy,
      DenseElementsAttr::get(narrowTy, attr.getSplatValue<Attribute>()));
}

static void collectLeaves(Value mask, SmallVectorImpl<Value> &leaves) {
  if (auto andOp = mask.getDefiningOp<arith::AndIOp>()) {
    collectLeaves(andOp.getLhs(), leaves);
    collectLeaves(andOp.getRhs(), leaves);
    return;
  }
  leaves.push_back(mask);
}

static bool isDotDerived(Value v, unsigned depth) {
  auto res = dyn_cast<OpResult>(v);
  if (!res || depth > MaxDepth)
    return false;
  Operation *op = res.getOwner();
  if (isa<tt::DotOp, tt::DotScaledOp>(op))
    return true;
  if (auto forOp = dyn_cast<scf::ForOp>(op))
    return isDotDerived(forOp.getYieldedValues()[res.getResultNumber()],
                        depth + 1);
  return op->hasTrait<OpTrait::Elementwise>() &&
         llvm::any_of(op->getOperands(), [&](Value operand) {
           return isDotDerived(operand, depth + 1);
         });
}

/// True if \p load's value reaches, through elementwise ops, an elementwise op
/// that combines it with a dot result.
static bool isDotEpilogue(tt::LoadOp load) {
  SmallVector<std::pair<Value, unsigned>> worklist{{load.getResult(), 0}};
  while (!worklist.empty()) {
    auto [v, depth] = worklist.pop_back_val();
    for (Operation *user : v.getUsers()) {
      if (!user->hasTrait<OpTrait::Elementwise>() || user->getNumResults() != 1)
        continue;
      for (Value operand : user->getOperands())
        if (operand != v && isDotDerived(operand, 0))
          return true;
      if (depth < MaxDepth)
        worklist.push_back({user->getResult(0), depth + 1});
    }
  }
  return false;
}

/// As OptimizeLoadMasks' `mayWriteMemory`, except that the structured
/// control-flow ops are judged by their contents, which the callers walk.
static bool mayWriteMemory(Operation *op) {
  // An assume only records a fact; Triton emits one per program-id bound.
  if (isa<LLVM::AssumeOp>(op))
    return false;
  if (isa<scf::ForOp, scf::IfOp, scf::WhileOp, scf::YieldOp, scf::ConditionOp>(
          op))
    return false;
  if (isMemoryEffectFree(op))
    return false;
  if (op->getNumRegions() != 0)
    return true;
  auto memOp = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memOp)
    return true;
  SmallVector<MemoryEffects::EffectInstance> effects;
  memOp.getEffects(effects);
  return llvm::any_of(effects, [](const MemoryEffects::EffectInstance &e) {
    return !isa<MemoryEffects::Read>(e.getEffect());
  });
}

/// The narrow load changes which lanes read which address, so a write that
/// can precede it -- earlier in the function, or anywhere in an enclosing
/// loop -- could turn into a cross-lane race.
static bool hasPriorOrLoopWrite(tt::LoadOp load) {
  Operation *func = load->getParentOfType<FunctionOpInterface>();
  bool found = false;
  func->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op == load.getOperation())
      return WalkResult::interrupt();
    if (op != func && mayWriteMemory(op)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (found)
    return true;
  for (Operation *p = load->getParentOp(); p && p != func; p = p->getParentOp())
    if (isa<LoopLikeOpInterface>(p) && p->walk([&](Operation *op) {
                                          return op != p && mayWriteMemory(op)
                                                     ? WalkResult::interrupt()
                                                     : WalkResult::advance();
                                        }).wasInterrupted())
      return true;
  return false;
}

/// "Some row of `base + arange(rows)` is `<s bound`", exact under iK wrap: the
/// smallest row index is `base`, or INT_MIN once `base + rows - 1` wraps.
static Value buildAnyRowValid(OpBuilder &b, Location loc, RowPrefix row,
                              int64_t rows) {
  auto intTy = cast<IntegerType>(row.base.getType());
  unsigned width = intTy.getWidth();
  Value bound = getSplatScalar(b, loc, row.bound);
  Value wrapAbove = arith::ConstantOp::create(
      b, loc,
      IntegerAttr::get(intTy, APInt::getSignedMaxValue(width) - (rows - 1)));
  Value minValue = arith::ConstantOp::create(
      b, loc, IntegerAttr::get(intTy, APInt::getSignedMinValue(width)));
  Value firstValid =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt, row.base, bound);
  Value wraps = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sgt,
                                      row.base, wrapAbove);
  Value minValid =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sgt, bound, minValue);
  Value wrapValid = arith::AndIOp::create(b, loc, wraps, minValid);
  return arith::OrIOp::create(b, loc, firstValid, wrapValid);
}

static bool isSupportedElementType(Type ty) {
  return isa<IntegerType>(ty) || ty.isF16() || ty.isBF16() || ty.isF32() ||
         ty.isF64();
}

static bool isHintAttr(StringAttr name) {
  return name == "tt.divisibility" || name == "tt.contiguity" ||
         name == "tt.constancy";
}

static bool tryNarrow(tt::LoadOp load, int64_t numThreads) {
  auto bc = load.getPtr().getDefiningOp<tt::BroadcastOp>();
  auto wideTy = dyn_cast<RankedTensorType>(load.getType());
  if (!bc || !wideTy || wideTy.getRank() != 2)
    return false;
  int64_t rows = wideTy.getDimSize(0), cols = wideTy.getDimSize(1);
  if (!hasShape(bc.getSrc(), {1, cols}))
    return false;

  Type elemTy = wideTy.getElementType();
  if (!load.getMask() || load.getIsVolatile() ||
      !isSupportedElementType(elemTy))
    return false;
  for (NamedAttribute attr : load->getDiscardableAttrs())
    if (!isHintAttr(attr.getName()))
      return false;

  SmallVector<Value> leaves, columnLeaves;
  collectLeaves(load.getMask(), leaves);
  std::optional<RowPrefix> row;
  for (Value leaf : leaves) {
    if (isColumnLeaf(leaf, {1, cols})) {
      columnLeaves.push_back(leaf);
      continue;
    }
    std::optional<RowPrefix> r = matchRowPrefix(leaf, rows);
    // `any(D1 & D2)` is not `any(D1) & any(D2)`: allow one row leaf only.
    if (!r || row)
      return false;
    row = r;
  }
  if (!row || cols >= numThreads || !isDotEpilogue(load) ||
      hasPriorOrLoopWrite(load))
    return false;

  OpBuilder b(load);
  Location loc = load.getLoc();
  auto narrowTy = RankedTensorType::get({1, cols}, elemTy);
  auto narrowMaskTy = RankedTensorType::get({1, cols}, b.getI1Type());
  Value mask = tt::SplatOp::create(b, loc, narrowMaskTy,
                                   buildAnyRowValid(b, loc, *row, rows));
  for (Value leaf : columnLeaves)
    mask = arith::AndIOp::create(
        b, loc, narrowColumnLeaf(b, loc, leaf, narrowMaskTy), mask);
  Value zero = arith::ConstantOp::create(
      b, loc, narrowTy,
      DenseElementsAttr::get(narrowTy, b.getZeroAttr(elemTy)));
  auto narrow = tt::LoadOp::create(b, loc, bc.getSrc(), mask, zero,
                                   load.getCachePolicyAttr(),
                                   /*isVolatile=*/false);
  Value wide = tt::BroadcastOp::create(b, loc, wideTy, narrow.getResult());
  Value other = load.getOther();
  // Intel lowers an absent `other` to zero; keep that for masked-off lanes.
  if (!other)
    other = arith::ConstantOp::create(
        b, loc, wideTy, DenseElementsAttr::get(wideTy, b.getZeroAttr(elemTy)));
  auto select = arith::SelectOp::create(b, loc, load.getMask(), wide, other);
  for (NamedAttribute attr : load->getDiscardableAttrs())
    select->setAttr(attr.getName(), attr.getValue());
  LDBG("narrowed " << load);
  load.replaceAllUsesWith(select.getResult());
  load.erase();
  ++NumNarrowed;
  return true;
}

struct NarrowBroadcastLoadsPass
    : public tt::intel::impl::TritonIntelNarrowBroadcastLoadsBase<
          NarrowBroadcastLoadsPass> {
  using Base::Base;

  void runOnOperation() override {
    int64_t numThreads = int64_t(numWarps) * threadsPerWarp;
    SmallVector<tt::LoadOp> loads;
    getOperation().walk([&](tt::LoadOp load) { loads.push_back(load); });
    bool changed = false;
    for (tt::LoadOp load : loads)
      changed |= tryNarrow(load, numThreads);
    if (!changed)
      markAllAnalysesPreserved();
  }
};

} // namespace
