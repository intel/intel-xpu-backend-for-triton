#include "intel/include/Analysis/StrideInfo.h"
#include "intel/include/Analysis/AxisInfoExt.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#include <numeric>

#define DEBUG_TYPE "intel-stride-info"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::triton::intel {

/// Per-axis join for a single stride column, using the "agree or -1" rule.
/// A -1 on either side propagates to -1.  This is the same rule the
/// existing spatial-stride `join` has always used, factored out so both
/// the spatial and IV columns share one implementation.
static StrideInfo::DimVectorT joinColumn(ArrayRef<int64_t> lhs,
                                         ArrayRef<int64_t> rhs) {
  assert(lhs.size() == rhs.size() && "Mismatched column ranks");
  StrideInfo::DimVectorT result;
  result.reserve(lhs.size());
  for (unsigned d = 0, rank = lhs.size(); d < rank; ++d) {
    if (lhs[d] == rhs[d])
      result.push_back(lhs[d]);
    else
      result.push_back(-1);
  }
  return result;
}

// Insert `vec` into `ivStrides[loop]` only if it carries meaningful
// information — i.e. at least one non-zero dimension.  An all-zero vector
// is semantically equivalent to an absent entry (see StrideInfo class
// docstring), and is never stored.  All sites that insert into `ivStrides`
// must go through this helper so the canonical-form invariant holds.
static void maybeStoreIVStride(
    DenseMap<LoopLikeOpInterface, StrideInfo::DimVectorT> &ivStrides,
    LoopLikeOpInterface loop, StrideInfo::DimVectorT vec) {
  if (llvm::all_of(vec, [](int64_t v) { return v == 0; }))
    return;
  ivStrides[loop] = std::move(vec);
}

// StrideInfo static methods
StrideInfo StrideInfo::getPessimisticValueState(Value value) {
  unsigned rank = 1;
  Type ty = value.getType();
  if (auto tensorTy = dyn_cast<RankedTensorType>(ty))
    rank = tensorTy.getRank();
  if (auto descTy = dyn_cast<triton::TensorDescInterface>(ty))
    rank = descTy.getBlockType().getRank();

  // Leave `ivStrides` empty: "no entry" is interpreted as all-zero by
  // consumers, which is correct for values defined outside every loop.
  // Values defined *inside* a loop whose defining op hits the pessimistic
  // path receive pessimism for the IV columns via subsequent arithmetic
  // visitors (which propagate -1 through the per-column Template Method).
  return StrideInfo(DimVectorT(rank, -1));
}

StrideInfo StrideInfo::join(const StrideInfo &lhs, const StrideInfo &rhs) {
  if (lhs.getRank() == 0) {
    assert(lhs.getIVStrides().empty() &&
           "rank-0 StrideInfo should not carry IV-stride entries");
    return rhs;
  }
  if (rhs.getRank() == 0) {
    assert(rhs.getIVStrides().empty() &&
           "rank-0 StrideInfo should not carry IV-stride entries");
    return lhs;
  }
  assert(lhs.getRank() == rhs.getRank() && "Mismatched ranks");

  DimVectorT spatial = joinColumn(lhs.stride, rhs.stride);

  // Union the set of tracked loops.  A missing entry is treated as an
  // all-zero vector of the shared rank, then joined with `joinColumn`'s
  // "agree-or-(-1)" rule.  Concretely, per dimension: {0,0} -> 0;
  // {k,0} or {0,k} -> 0 if k == 0 else -1; {k1,k2} -> k1 if k1 == k2
  // else -1.
  //
  // Note: when one incoming path is loop-variant (stride k != 0) and
  // the other is loop-invariant (absent entry, treated as 0), the
  // merged column is -1 for that loop.  This is not pessimism hiding a
  // better answer — across iterations of the loop the value genuinely
  // advances by k on some paths and by 0 on others, so no single stride
  // describes it.  The lattice's -1 ("unknown") is the correct answer.
  llvm::SmallSetVector<LoopLikeOpInterface, 4> allLoops;
  for (LoopLikeOpInterface loop : llvm::make_first_range(lhs.ivStrides))
    allLoops.insert(loop);
  for (LoopLikeOpInterface loop : llvm::make_first_range(rhs.ivStrides))
    allLoops.insert(loop);
  DimVectorT zeros(lhs.getRank(), 0);
  DenseMap<LoopLikeOpInterface, DimVectorT> ivStrides;
  for (LoopLikeOpInterface loop : allLoops) {
    const DimVectorT *l = lhs.getIVStride(loop);
    const DimVectorT *r = rhs.getIVStride(loop);
    maybeStoreIVStride(
        ivStrides, loop,
        joinColumn(l ? ArrayRef<int64_t>(*l) : ArrayRef<int64_t>(zeros),
                   r ? ArrayRef<int64_t>(*r) : ArrayRef<int64_t>(zeros)));
  }

  // Runtime stride value: keep it on an axis only when both sides agree on
  // the same SSA value, otherwise drop to null (same idea as the join of
  // AxisInfo::constantValue).
  StrideValueVectorT strideValues;
  const StrideValueVectorT &lv = lhs.getStrideValues();
  const StrideValueVectorT &rv = rhs.getStrideValues();
  if (!lv.empty() || !rv.empty()) {
    strideValues.resize(lhs.getRank());
    for (unsigned d = 0, rank = lhs.getRank(); d < rank; ++d) {
      Value a = d < lv.size() ? lv[d] : Value();
      Value b = d < rv.size() ? rv[d] : Value();
      strideValues[d] = (a && a == b) ? a : Value();
    }
  }

  return StrideInfo(std::move(spatial), std::move(ivStrides),
                    std::move(strideValues));
}

int64_t StrideInfo::getIVStride(LoopLikeOpInterface loop, size_t dim) const {
  DenseMap<LoopLikeOpInterface, DimVectorT>::const_iterator it =
      ivStrides.find(loop);
  if (it == ivStrides.end())
    return 0;
  return it->second[dim];
}

const StrideInfo::DimVectorT *
StrideInfo::getIVStride(LoopLikeOpInterface loop) const {
  DenseMap<LoopLikeOpInterface, DimVectorT>::const_iterator it =
      ivStrides.find(loop);
  if (it == ivStrides.end())
    return nullptr;
  assert(!llvm::all_of(it->second, [](int64_t v) { return v == 0; }) &&
         "ivStrides invariant violated: stored entry is all-zero");
  return &it->second;
}

std::optional<int64_t>
StrideInfo::getPerIterationIVStride(LoopLikeOpInterface loop,
                                    size_t dim) const {
  int64_t ivUnitStride = getIVStride(loop, dim);
  if (ivUnitStride < 0)
    return std::nullopt;
  // Only scf.for exposes a constant-step query today; scf.while does not
  // have a single step to fold in and is deferred (see StrideAnalysis).
  auto forOp = dyn_cast<scf::ForOp>(loop.getOperation());
  if (!forOp)
    return std::nullopt;
  std::optional<APInt> step = forOp.getConstantStep();
  if (!step.has_value())
    return std::nullopt;
  // Guard against overflow when the step doesn't fit in int64_t or the
  // product would wrap.
  if (step->getSignificantBits() > 64)
    return std::nullopt;
  int64_t stepVal = step->getSExtValue();
  int64_t product;
  if (llvm::MulOverflow(ivUnitStride, stepVal, product))
    return std::nullopt;
  return product;
}

void StrideInfo::print(raw_ostream &os) const {
  os << "stride = [";
  llvm::interleaveComma(stride, os);
  os << "]";
  if (!ivStrides.empty()) {
    os << ", iv_strides = {";
    bool first = true;
    for (const std::pair<LoopLikeOpInterface, DimVectorT> &kv : ivStrides) {
      if (!first)
        os << ", ";
      first = false;
      os << kv.first->getName() << "@";
      kv.first->getLoc().print(os);
      os << ": [";
      llvm::interleaveComma(kv.second, os);
      os << "]";
    }
    os << "}";
  }
  if (!strideValues.empty()) {
    os << ", stride_values = {";
    bool first = true;
    for (unsigned d = 0, rank = strideValues.size(); d < rank; ++d) {
      if (!strideValues[d])
        continue;
      if (!first)
        os << ", ";
      first = false;
      os << d << ": " << strideValues[d];
    }
    os << "}";
  }
}

using AxisInfoLookupFn = std::function<AxisInfo *(Value)>;

namespace {

/// Try to extract a scalar integer constant from a Value by inspecting the
/// defining op directly. Only recognises arith.constant and llvm.constant —
/// for a more robust check that also consults AxisInfo, use
/// StrideInfoVisitor::getConstantValue() instead.
static std::optional<int64_t> getScalarIntConstant(Value v) {
  Operation *defOp = v.getDefiningOp();
  if (!defOp)
    return std::nullopt;
  Attribute attr;
  if (auto constOp = dyn_cast<arith::ConstantOp>(defOp))
    attr = constOp.getValue();
  else if (auto constOp = dyn_cast<LLVM::ConstantOp>(defOp))
    attr = constOp.getValue();
  else
    return std::nullopt;

  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    APInt apValue = intAttr.getValue();
    // 1-bit integers: use getZExtValue to avoid sign-extending true to -1.
    return apValue.getBitWidth() == 1 ? apValue.getZExtValue()
                                      : apValue.getSExtValue();
  }
  if (auto splatAttr = dyn_cast<SplatElementsAttr>(attr)) {
    if (splatAttr.getElementType().isIntOrIndex()) {
      APInt apValue = splatAttr.getSplatValue<APInt>();
      return apValue.getBitWidth() == 1 ? apValue.getZExtValue()
                                        : apValue.getSExtValue();
    }
  }
  return std::nullopt;
}

class StrideInfoVisitor {
public:
  virtual ~StrideInfoVisitor() = default;
  virtual StrideInfo getStrideInfo(
      Operation *op,
      ArrayRef<const dataflow::Lattice<StrideInfo> *> operands) const = 0;
  virtual bool match(Operation *op) const = 0;

  void setAxisInfoLookup(AxisInfoLookupFn fn) {
    axisInfoLookup = std::move(fn);
  }

  /// Public wrapper over `getConstantValue` so free helpers can reuse the
  /// same constant detection.
  std::optional<int64_t> getConstantValueForStride(Value v) const {
    return getConstantValue(v);
  }

protected:
  /// Try to extract a constant integer value from v.
  /// First checks for a direct constant op (arith.constant / llvm.constant),
  /// then falls back to AxisInfo::getConstantValue() when available.
  std::optional<int64_t> getConstantValue(Value v) const {
    if (auto c = getScalarIntConstant(v))
      return c;
    if (auto *ai = axisInfoLookup(v))
      return ai->getConstantValue();
    return std::nullopt;
  }

  AxisInfoLookupFn axisInfoLookup;
};

template <typename OpTy>
class StrideInfoVisitorImpl : public StrideInfoVisitor {
public:
  StrideInfo getStrideInfo(
      Operation *op,
      ArrayRef<const dataflow::Lattice<StrideInfo> *> operands) const final {
    return getStrideInfo(cast<OpTy>(op), operands);
  }
  bool match(Operation *op) const final { return isa<OpTy>(op); }
  virtual StrideInfo getStrideInfo(
      OpTy op,
      ArrayRef<const dataflow::Lattice<StrideInfo> *> operands) const = 0;
};

/// Template Method base for arithmetic visitors that treat the spatial
/// column and every per-loop IV column the same way.  Subclasses override
/// one per-column hook (`applyOne`); the base class walks every column
/// (spatial + union of IV columns from all operand lattices) and assembles
/// the resulting StrideInfo.
///
/// - Binary arithmetic visitors (AddI, SubI, AddPtr, MulI, DivI, RemI)
///   receive `rhs` as a pointer to the right-hand-side column.
/// - Unary / shape-transform visitors (Splat, ExpandDims, Broadcast, Trans)
///   receive `rhs == nullptr` and should ignore it.
///
/// Leaf / pessimistic visitors that do not perform per-column arithmetic
/// (LoadOp, DescriptorLoadOp, PoisonOp, MakeRangeOp, ConstantOp,
/// MakeTensorDescOp) do **not** use this base — they subclass
/// StrideInfoVisitorImpl directly and produce a full StrideInfo by hand.
///
/// MAINTAINER NOTE: When adding a new arithmetic visitor, prefer
/// `StrideArithVisitor` over `StrideInfoVisitorImpl`.  The base iterates
/// every stride column for you, so you cannot forget to propagate an
/// IV column.  This is the single source of truth for per-axis stride
/// arithmetic.
template <typename OpTy>
class StrideArithVisitor : public StrideInfoVisitorImpl<OpTy> {
public:
  StrideInfo getStrideInfo(
      OpTy op,
      ArrayRef<const dataflow::Lattice<StrideInfo> *> operands) const final {
    assert(!operands.empty() && "Arithmetic visitor requires >= 1 operand");
    const StrideInfo &lhs = operands[0]->getValue();
    const StrideInfo *rhs =
        operands.size() > 1 ? &operands[1]->getValue() : nullptr;

    StrideInfo::DimVectorT spatial =
        applyOne(op, lhs.getStride(), rhs ? &rhs->getStride() : nullptr);

    // Every IV column present in either operand.
    llvm::SmallSetVector<LoopLikeOpInterface, 4> allLoops;
    for (LoopLikeOpInterface loop : llvm::make_first_range(lhs.getIVStrides()))
      allLoops.insert(loop);
    if (rhs)
      for (LoopLikeOpInterface loop :
           llvm::make_first_range(rhs->getIVStrides()))
        allLoops.insert(loop);

    StrideInfo::DimVectorT zeros(lhs.getRank(), 0);
    DenseMap<LoopLikeOpInterface, StrideInfo::DimVectorT> ivStrides;
    for (LoopLikeOpInterface loop : allLoops) {
      const StrideInfo::DimVectorT *lc = lhs.getIVStride(loop);
      const StrideInfo::DimVectorT *rc = rhs ? rhs->getIVStride(loop) : nullptr;
      const StrideInfo::DimVectorT &lCol = lc ? *lc : zeros;
      const StrideInfo::DimVectorT *rCol = rc ? rc : (rhs ? &zeros : nullptr);
      maybeStoreIVStride(ivStrides, loop, applyOne(op, lCol, rCol));
    }

    // Runtime stride value (spatial axes only). The default hook returns
    // nothing; a visitor opts in only where the value is safe to name.
    StrideInfo::StrideValueVectorT strideValues =
        applyStrideValues(op, lhs, rhs, spatial);

    return StrideInfo(std::move(spatial), std::move(ivStrides),
                      std::move(strideValues));
  }

protected:
  /// Subclass hook — compute the result of `op` for one stride column.
  /// For binary ops, `rhs` is non-null and points at the right-hand-side
  /// column (either the operand's actual column or a zero vector of the
  /// correct rank when that side has no entry for this loop).  For
  /// unary/shape-transform ops, `rhs` is `nullptr`.
  virtual StrideInfo::DimVectorT
  applyOne(OpTy op, const StrideInfo::DimVectorT &lhs,
           const StrideInfo::DimVectorT *rhs) const = 0;

  /// Subclass hook: name the runtime stride value for each spatial axis of
  /// the result (`resultSpatial` is the already-computed integer column).
  /// Defaults to "none"; a visitor overrides only when a bare SSA scalar
  /// really is the element stride.
  virtual StrideInfo::StrideValueVectorT
  applyStrideValues(OpTy op, const StrideInfo &lhs, const StrideInfo *rhs,
                    const StrideInfo::DimVectorT &resultSpatial) const {
    return {};
  }
};

class StrideInfoVisitorList {
public:
  template <typename... Ts, typename = std::enable_if_t<sizeof...(Ts) != 0>>
  void append() {
    (visitors.emplace_back(std::make_unique<Ts>()), ...);
  }

  void setAxisInfoLookup(AxisInfoLookupFn fn) {
    for (auto &v : visitors)
      v->setAxisInfoLookup(fn);
  }

  StrideInfo
  apply(Operation *op,
        ArrayRef<const dataflow::Lattice<StrideInfo> *> operands) const {
    for (auto &v : visitors)
      if (v->match(op))
        return v->getStrideInfo(op, operands);
    return StrideInfo();
  }

private:
  std::vector<std::unique_ptr<StrideInfoVisitor>> visitors;
};

// PassThrough: stride passes from operand 0.  Returning `operands[0]->
// getValue()` verbatim correctly carries both the spatial column and the
// full IV-stride map to the result.
template <typename OpTy,
          typename =
              std::enable_if_t<OpTy::template hasTrait<OpTrait::OneOperand>()>>
class PassThroughStrideVisitor final : public StrideInfoVisitorImpl<OpTy> {
public:
  StrideInfo getStrideInfo(
      OpTy op,
      ArrayRef<const dataflow::Lattice<StrideInfo> *> operands) const override {
    assert(op->getNumOperands() == 1 &&
           "PassThroughStrideVisitor expects a single-operand op");
    return operands[0]->getValue();
  }
};

// UnrealizedConversionCastOp: stride passes from operand 0.
// This op has variadic inputs so it cannot use PassThroughStrideVisitor.
class UnrealizedConversionCastStrideVisitor final
    : public StrideInfoVisitorImpl<mlir::UnrealizedConversionCastOp> {
public:
  StrideInfo getStrideInfo(
      mlir::UnrealizedConversionCastOp op,
      ArrayRef<const dataflow::Lattice<StrideInfo> *> operands) const override {
    assert(!operands.empty() &&
           "UnrealizedConversionCastOp must have at least one operand");
    if (op->getNumOperands() > 1)
      return StrideInfo();
    return operands[0]->getValue();
  }
};

class MakeRangeOpStrideVisitor final
    : public StrideInfoVisitorImpl<triton::MakeRangeOp> {
public:
  StrideInfo getStrideInfo(
      triton::MakeRangeOp op,
      ArrayRef<const dataflow::Lattice<StrideInfo> *> operands) const override {
    return StrideInfo(StrideInfo::DimVectorT{1});
  }
};

class PoisonOpStrideVisitor final : public StrideInfoVisitorImpl<ub::PoisonOp> {
public:
  StrideInfo getStrideInfo(
      ub::PoisonOp op,
      ArrayRef<const dataflow::Lattice<StrideInfo> *> operands) const override {
    return StrideInfo::getPessimisticValueState(op.getResult());
  }
};

template <typename OpTy>
class ConstantOpStrideVisitor final : public StrideInfoVisitorImpl<OpTy> {
public:
  StrideInfo getStrideInfo(
      OpTy op,
      ArrayRef<const dataflow::Lattice<StrideInfo> *> operands) const override {
    auto splatAttr = dyn_cast<SplatElementsAttr>(op.getValue());
    if (splatAttr && splatAttr.getElementType().isIntOrIndex()) {
      TensorType ty = cast<TensorType>(splatAttr.getType());
      return StrideInfo(StrideInfo::DimVectorT(ty.getRank(), 0));
    }

    auto intAttr = dyn_cast<IntegerAttr>(op.getValue());
    auto boolAttr = dyn_cast<BoolAttr>(op.getValue());
    if (intAttr || boolAttr)
      return StrideInfo(StrideInfo::DimVectorT{0});

    return StrideInfo();
  }
};

template <typename OpTy>
class AddSubStrideVisitor final : public StrideArithVisitor<OpTy> {
protected:
  StrideInfo::StrideValueVectorT applyStrideValues(
      OpTy op, const StrideInfo &lhs, const StrideInfo *rhs,
      const StrideInfo::DimVectorT &resultSpatial) const override {
    // `a + b` (or addptr): the result stride is the sum, so a single value
    // names it only when one side carries the runtime stride and the other
    // adds nothing (stride 0). If both sides name a value we cannot keep it,
    // since `s + s` is `2*s`. Subtraction keeps only the left side's value.
    if (!rhs)
      return {};
    unsigned rank = lhs.getRank();
    StrideInfo::StrideValueVectorT values(rank);
    for (unsigned d = 0; d < rank; ++d) {
      Value ls = lhs.getStrideValue(d);
      Value rs = rhs->getStrideValue(d);
      if (ls && lhs.getStride(d) < 0 && rhs->getStride(d) == 0) {
        values[d] = ls;
      } else if constexpr (!std::is_same_v<OpTy, arith::SubIOp>) {
        if (rs && rhs->getStride(d) < 0 && lhs.getStride(d) == 0)
          values[d] = rs;
      }
    }
    return values;
  }

  StrideInfo::DimVectorT
  applyOne(OpTy op, const StrideInfo::DimVectorT &lhs,
           const StrideInfo::DimVectorT *rhs) const override {
    assert(rhs && "AddSubStrideVisitor requires two operand columns");
    StrideInfo::DimVectorT stride;
    stride.reserve(lhs.size());
    for (unsigned d = 0, rank = lhs.size(); d < rank; ++d) {
      if (lhs[d] < 0 || (*rhs)[d] < 0) {
        stride.push_back(-1);
      } else if constexpr (std::is_same_v<OpTy, arith::SubIOp>) {
        stride.push_back(std::max(lhs[d] - (*rhs)[d], int64_t(-1)));
      } else {
        stride.push_back(lhs[d] + (*rhs)[d]);
      }
    }
    return stride;
  }
};

/// Look through int casts and a splat; return the splatted scalar if it's a
/// runtime (non-constant) value, else null.
static Value getNonConstSplatScalar(Value v, const StrideInfoVisitor &visitor) {
  while (Operation *def = v.getDefiningOp()) {
    if (isa<arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp,
            arith::IndexCastOp>(def)) {
      v = def->getOperand(0);
      continue;
    }
    break;
  }
  auto splat = v.getDefiningOp<triton::SplatOp>();
  if (!splat)
    return {};
  Value scalar = splat.getSrc();
  if (visitor.getConstantValueForStride(scalar))
    return {};
  return scalar;
}

class MulIOpStrideVisitor final : public StrideArithVisitor<arith::MulIOp> {
protected:
  StrideInfo::StrideValueVectorT applyStrideValues(
      arith::MulIOp op, const StrideInfo &lhs, const StrideInfo *rhs,
      const StrideInfo::DimVectorT &resultSpatial) const override {
    // `index * splat(s)`: when one side steps by exactly 1, the per-element
    // stride is exactly the scalar `s`, so we can name it. Any other step
    // would make the stride `c * s`, which is not a single value, so we skip.
    if (!rhs)
      return {};
    unsigned rank = lhs.getRank();
    StrideInfo::StrideValueVectorT values(rank);
    Value rhsScalar = getNonConstSplatScalar(op.getRhs(), *this);
    Value lhsScalar = getNonConstSplatScalar(op.getLhs(), *this);
    for (unsigned d = 0; d < rank; ++d) {
      if (lhs.getStride(d) == 1 && rhsScalar)
        values[d] = rhsScalar;
      else if (rhs->getStride(d) == 1 && lhsScalar)
        values[d] = lhsScalar;
    }
    return values;
  }

  StrideInfo::DimVectorT
  applyOne(arith::MulIOp op, const StrideInfo::DimVectorT &lhs,
           const StrideInfo::DimVectorT *rhs) const override {
    assert(rhs && "MulIOpStrideVisitor requires two operand columns");

    // Constant detection is an op-level property — it depends on the
    // defining op of the raw Value, not on the per-column stride.
    // Call through to the inherited StrideInfoVisitor::getConstantValue().
    std::optional<int64_t> lhsConst = getConstantValue(op.getLhs());
    std::optional<int64_t> rhsConst = getConstantValue(op.getRhs());

    StrideInfo::DimVectorT stride;
    stride.reserve(lhs.size());
    for (unsigned d = 0, rank = lhs.size(); d < rank; ++d) {
      if (lhs[d] > 0 && rhsConst.has_value()) {
        int64_t product = lhs[d] * rhsConst.value();
        stride.push_back(product >= 0 ? product : -1);
      } else if ((*rhs)[d] > 0 && lhsConst.has_value()) {
        int64_t product = lhsConst.value() * (*rhs)[d];
        stride.push_back(product >= 0 ? product : -1);
      } else {
        auto strideZero = [&](int64_t col, Value v) {
          return getConstantValue(v).has_value() || col == 0 ||
                 !isa<TensorType>(op.getType());
        };
        if (strideZero(lhs[d], op.getLhs()) &&
            strideZero((*rhs)[d], op.getRhs()))
          stride.push_back(0);
        else
          stride.push_back(-1);
      }
    }
    return stride;
  }
};

template <typename OpTy>
class DivOpStrideVisitor final : public StrideArithVisitor<OpTy> {
protected:
  StrideInfo::DimVectorT
  applyOne(OpTy op, const StrideInfo::DimVectorT &lhs,
           const StrideInfo::DimVectorT *rhs) const override {
    // `rhs` column is unused here — RHS-constant detection is an op-level
    // property (we read the defining op via `getConstantValue`).  It must
    // still be non-null because this is a binary-arithmetic visitor.
    assert(rhs && "DivOpStrideVisitor requires two operand columns");

    std::optional<int64_t> rhsConst = this->getConstantValue(op.getRhs());

    StrideInfo::DimVectorT stride;
    stride.reserve(lhs.size());
    for (unsigned d = 0, rank = lhs.size(); d < rank; ++d) {
      if (lhs[d] > 0 && rhsConst.has_value() && rhsConst.value() > 0 &&
          lhs[d] % rhsConst.value() == 0)
        stride.push_back(lhs[d] / rhsConst.value());
      else if (lhs[d] == 0 && rhsConst.has_value() && rhsConst.value() != 0)
        stride.push_back(0);
      else
        stride.push_back(-1);
    }
    return stride;
  }
};

template <typename OpTy>
class RemOpStrideVisitor final : public StrideArithVisitor<OpTy> {
protected:
  StrideInfo::DimVectorT
  applyOne(OpTy op, const StrideInfo::DimVectorT &lhs,
           const StrideInfo::DimVectorT *rhs) const override {
    assert(rhs && "RemOpStrideVisitor requires two operand columns");

    std::optional<int64_t> rhsConst = this->getConstantValue(op.getRhs());

    StrideInfo::DimVectorT stride;
    stride.reserve(lhs.size());
    for (unsigned d = 0, rank = lhs.size(); d < rank; ++d) {
      if (lhs[d] == 0 && (*rhs)[d] == 0) {
        // Both sides are uniform/constant — result is uniform.
        stride.push_back(0);
      } else if (lhs[d] > 0 && rhsConst.has_value() && rhsConst.value() > 0) {
        // Stride preserved when range span doesn't cross a modulus boundary.
        // Effective period is gcd(divisibility, modulus) when AxisInfo is
        // available; falls back to modulus otherwise.
        auto resTy = dyn_cast<RankedTensorType>(op.getType());
        if (resTy) {
          int64_t dimSize = resTy.getDimSize(d);
          int64_t maxVal = lhs[d] * (dimSize - 1);
          int64_t modulus = rhsConst.value();
          int64_t g = modulus; // fallback when no AxisInfo
          if (auto *ai = this->axisInfoLookup(op.getLhs())) {
            int64_t divisibility = ai->getDivisibility(d);
            g = std::gcd(divisibility, modulus);
          }
          if (maxVal < g)
            stride.push_back(lhs[d]);
          else
            stride.push_back(-1);
        } else {
          stride.push_back(-1);
        }
      } else {
        stride.push_back(-1);
      }
    }
    return stride;
  }
};

/// SplatOp: a splat takes a scalar and produces a tensor where every
/// element equals the scalar.  The two stride columns answer different
/// questions and therefore must be computed independently:
///
/// - **Spatial stride** of the result is 0 along every axis, because
///   neighbouring tensor elements within a single loop iteration are
///   identical.  This is independent of the scalar's spatial stride.
/// - **IV stride** of the result equals the scalar's IV stride,
///   broadcast across every tensor axis.  If the scalar advances by N
///   per iteration, every element of the splat tensor advances by N
///   per iteration.  This is the key case for recognising the 1-D
///   streaming pattern `tt.addptr(splat(base), splat(muli(iv, N)))`.
///
/// Because the spatial and IV columns require different logic, `SplatOp`
/// does not use `StrideArithVisitor`; it overrides `getStrideInfo`
/// directly and handles both columns by hand.
class SplatOpStrideVisitor final
    : public StrideInfoVisitorImpl<triton::SplatOp> {
public:
  StrideInfo getStrideInfo(
      triton::SplatOp op,
      ArrayRef<const dataflow::Lattice<StrideInfo> *> operands) const override {
    TensorType retTy = cast<TensorType>(*op->result_type_begin());
    unsigned resRank = retTy.getRank();

    const StrideInfo &scalar = operands[0]->getValue();
    DenseMap<LoopLikeOpInterface, StrideInfo::DimVectorT> ivStrides;
    for (const std::pair<LoopLikeOpInterface, StrideInfo::DimVectorT> &kv :
         scalar.getIVStrides()) {
      // The scalar's IV column is rank-1 (scalars have rank 1 in this
      // analysis, because getRank() treats scalars as 1-element vectors).
      // Broadcast the scalar's single IV-stride value across every
      // axis of the result tensor.
      assert(kv.second.size() == 1 && "scalar should have rank-1 IV stride");
      int64_t ivVal = kv.second.front();
      maybeStoreIVStride(ivStrides, kv.first,
                         StrideInfo::DimVectorT(resRank, ivVal));
    }
    return StrideInfo(StrideInfo::DimVectorT(resRank, 0), std::move(ivStrides));
  }
};

class LoadOpStrideVisitor final : public StrideInfoVisitorImpl<triton::LoadOp> {
public:
  StrideInfo getStrideInfo(
      triton::LoadOp op,
      ArrayRef<const dataflow::Lattice<StrideInfo> *> operands) const override {
    return StrideInfo::getPessimisticValueState(op.getResult());
  }
};

class ExpandDimsOpStrideVisitor final
    : public StrideArithVisitor<triton::ExpandDimsOp> {
protected:
  StrideInfo::DimVectorT
  applyOne(triton::ExpandDimsOp op, const StrideInfo::DimVectorT &lhs,
           const StrideInfo::DimVectorT *rhs) const override {
    assert(!rhs && "ExpandDimsOpStrideVisitor is unary");
    StrideInfo::DimVectorT stride(lhs.begin(), lhs.end());
    stride.insert(stride.begin() + op.getAxis(), 0);
    return stride;
  }

  StrideInfo::StrideValueVectorT
  applyStrideValues(triton::ExpandDimsOp op, const StrideInfo &lhs,
                    const StrideInfo *,
                    const StrideInfo::DimVectorT &) const override {
    const StrideInfo::StrideValueVectorT &src = lhs.getStrideValues();
    if (src.empty())
      return {};
    StrideInfo::StrideValueVectorT values(src.begin(), src.end());
    values.insert(values.begin() + op.getAxis(), Value());
    return values;
  }
};

class BroadcastOpStrideVisitor final
    : public StrideArithVisitor<triton::BroadcastOp> {
protected:
  StrideInfo::DimVectorT
  applyOne(triton::BroadcastOp, const StrideInfo::DimVectorT &lhs,
           const StrideInfo::DimVectorT *rhs) const override {
    assert(!rhs && "BroadcastOpStrideVisitor is unary");
    // Broadcast preserves rank and per-axis values; broadcast axes are
    // already 0 in the operand column, so a verbatim copy is correct for
    // every column (spatial + every IV column).
    return lhs;
  }

  StrideInfo::StrideValueVectorT applyStrideValues(
      triton::BroadcastOp op, const StrideInfo &lhs, const StrideInfo *,
      const StrideInfo::DimVectorT &resultSpatial) const override {
    // A broadcast axis becomes uniform (stride 0), so its runtime stride no
    // longer means anything: drop it there and pass the other axes through.
    const StrideInfo::StrideValueVectorT &src = lhs.getStrideValues();
    if (src.empty())
      return {};
    StrideInfo::StrideValueVectorT values(src.begin(), src.end());
    for (unsigned d = 0, e = values.size(); d < e; ++d)
      if (d < resultSpatial.size() && resultSpatial[d] == 0)
        values[d] = Value();
    return values;
  }
};

class TransOpStrideVisitor final : public StrideArithVisitor<triton::TransOp> {
protected:
  StrideInfo::DimVectorT
  applyOne(triton::TransOp op, const StrideInfo::DimVectorT &lhs,
           const StrideInfo::DimVectorT *rhs) const override {
    assert(!rhs && "TransOpStrideVisitor is unary");
    auto order = op.getOrder();
    StrideInfo::DimVectorT stride;
    stride.reserve(lhs.size());
    for (unsigned d = 0, rank = lhs.size(); d < rank; ++d)
      stride.push_back(lhs[order[d]]);
    return stride;
  }

  StrideInfo::StrideValueVectorT
  applyStrideValues(triton::TransOp op, const StrideInfo &lhs,
                    const StrideInfo *,
                    const StrideInfo::DimVectorT &) const override {
    const StrideInfo::StrideValueVectorT &src = lhs.getStrideValues();
    if (src.empty())
      return {};
    ArrayRef<int32_t> order = op.getOrder();
    StrideInfo::StrideValueVectorT values(src.size());
    for (unsigned d = 0, rank = src.size(); d < rank; ++d)
      values[d] = src[order[d]];
    return values;
  }
};

class MakeTensorDescOpStrideVisitor final
    : public StrideInfoVisitorImpl<triton::MakeTensorDescOp> {
public:
  StrideInfo getStrideInfo(
      triton::MakeTensorDescOp op,
      ArrayRef<const dataflow::Lattice<StrideInfo> *> operands) const override {
    StrideInfo::DimVectorT result;
    for (Value s : op.getStrides()) {
      std::optional<int64_t> val = getConstantValue(s);
      result.push_back(val.has_value() ? val.value() : -1);
    }
    return StrideInfo(std::move(result));
  }
};

class DescriptorLoadOpStrideVisitor final
    : public StrideInfoVisitorImpl<triton::DescriptorLoadOp> {
public:
  StrideInfo getStrideInfo(
      triton::DescriptorLoadOp op,
      ArrayRef<const dataflow::Lattice<StrideInfo> *> operands) const override {
    return StrideInfo::getPessimisticValueState(op.getResult());
  }
};

//===----------------------------------------------------------------------===//
// StrideAnalysis
//===----------------------------------------------------------------------===//

class StrideAnalysis : public dataflow::SparseForwardDataFlowAnalysis<
                           dataflow::Lattice<StrideInfo>> {
private:
  StrideInfoVisitorList visitors;

  void setToEntryState(dataflow::Lattice<StrideInfo> *lattice) override {
    propagateIfChanged(lattice,
                       lattice->join(StrideInfo::getPessimisticValueState(
                           lattice->getAnchor())));
  }

  void visitNonControlFlowArguments(
      Operation *op, const RegionSuccessor & /*successor*/,
      ValueRange /*nonSuccessorInputs*/,
      ArrayRef<dataflow::Lattice<StrideInfo> *> argLattices) override {
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      // Induction variable has spatial stride 0 (scalar) and IV stride 1
      // w.r.t. its own loop's IV; no entries for any other loop
      // (implicitly treated as all-zero by downstream consumers).
      DenseMap<LoopLikeOpInterface, StrideInfo::DimVectorT> ivStrides;
      maybeStoreIVStride(ivStrides, cast<LoopLikeOpInterface>(op),
                         StrideInfo::DimVectorT{1});
      StrideInfo iv(StrideInfo::DimVectorT{0}, std::move(ivStrides));
      (void)argLattices[0]->join(iv);
    } else {
      // scf.while IV-stride seeding is intentionally deferred.  Its
      // before-region args are control-flow-fed from the init operands, which
      // bypasses this hook.  Handling it requires overriding the solver's
      // region-successor propagation; that is out of scope here and will be
      // revisited when a consumer (PR-B TemporalReuseAnalysis) needs it.
      setAllToEntryStates(argLattices);
    }
  }

public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StrideAnalysis)

  StrideAnalysis(DataFlowSolver &solver, AxisInfoLookupFn axisInfoLookup)
      : dataflow::SparseForwardDataFlowAnalysis<dataflow::Lattice<StrideInfo>>(
            solver) {
    // PassThrough visitors
    visitors.append<PassThroughStrideVisitor<arith::ExtSIOp>,
                    PassThroughStrideVisitor<arith::ExtUIOp>,
                    PassThroughStrideVisitor<arith::TruncIOp>,
                    PassThroughStrideVisitor<arith::IndexCastOp>,
                    PassThroughStrideVisitor<triton::gpu::ConvertLayoutOp>,
                    PassThroughStrideVisitor<triton::BitcastOp>>();
    visitors.append<UnrealizedConversionCastStrideVisitor>();
    visitors.append<MakeRangeOpStrideVisitor>();
    visitors.append<PoisonOpStrideVisitor>();
    visitors.append<ConstantOpStrideVisitor<arith::ConstantOp>,
                    ConstantOpStrideVisitor<LLVM::ConstantOp>>();
    visitors.append<AddSubStrideVisitor<triton::AddPtrOp>,
                    AddSubStrideVisitor<arith::AddIOp>,
                    AddSubStrideVisitor<arith::SubIOp>,
                    AddSubStrideVisitor<LLVM::AddOp>>();
    visitors.append<MulIOpStrideVisitor>();
    visitors.append<DivOpStrideVisitor<arith::DivSIOp>,
                    DivOpStrideVisitor<arith::DivUIOp>>();
    visitors.append<RemOpStrideVisitor<arith::RemSIOp>,
                    RemOpStrideVisitor<arith::RemUIOp>>();
    visitors.append<SplatOpStrideVisitor>();
    visitors.append<LoadOpStrideVisitor>();
    visitors.append<ExpandDimsOpStrideVisitor>();
    visitors.append<BroadcastOpStrideVisitor>();
    visitors.append<TransOpStrideVisitor>();
    visitors.append<MakeTensorDescOpStrideVisitor>();
    visitors.append<DescriptorLoadOpStrideVisitor>();
    visitors.setAxisInfoLookup(std::move(axisInfoLookup));
  }

  using dataflow::SparseForwardDataFlowAnalysis<
      dataflow::Lattice<StrideInfo>>::getLatticeElement;

  LogicalResult
  visitOperation(Operation *op,
                 ArrayRef<const dataflow::Lattice<StrideInfo> *> operands,
                 ArrayRef<dataflow::Lattice<StrideInfo> *> results) override {
    // Skip if any operand is uninitialized.
    for (auto op : operands)
      if (op->getValue().getRank() == 0)
        return success();

    StrideInfo curr = visitors.apply(op, operands);
    if (curr.getRank() == 0) {
      setAllToEntryStates(results);
      return success();
    }
    // Override stride from tt.contiguity hint.  Build a fresh StrideInfo
    // carrying the patched spatial column and the IV-stride map verbatim.
    if (auto contiguityAttr = op->getDiscardableAttr("tt.contiguity")) {
      if (auto resTy = dyn_cast<RankedTensorType>(op->getResult(0).getType())) {
        AxisInfo::DimVectorT hintContiguity;
        AxisInfo::initDimVectorFromHint(contiguityAttr, &hintContiguity);
        StrideInfo::DimVectorT newStride(curr.getStride().begin(),
                                         curr.getStride().end());
        for (unsigned d = 0; d < curr.getRank() && d < hintContiguity.size();
             ++d) {
          if (newStride[d] < 0 && hintContiguity[d] >= resTy.getDimSize(d)) {
            newStride[d] = 1;
          }
        }
        curr = StrideInfo(std::move(newStride), curr.getIVStrides(),
                          curr.getStrideValues());
      }
    }
    for (auto *result : results)
      propagateIfChanged(result, result->join(curr));
    return success();
  }
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// ModuleStrideAnalysis
//===----------------------------------------------------------------------===//

ModuleStrideAnalysis::ModuleStrideAnalysis(ModuleOp moduleOp,
                                           ModuleAxisInfoAnalysis &axisInfo)
    : CallGraph<StrideInfoMapT>(moduleOp), axisInfo(axisInfo) {
  SmallVector<FunctionOpInterface> funcs;
  walk<WalkOrder::PreOrder, WalkOrder::PostOrder>(
      [](CallOpInterface callOp, FunctionOpInterface funcOp) {},
      [&](FunctionOpInterface funcOp) {
        funcs.push_back(funcOp);
        funcMap.try_emplace(funcOp, StrideInfoMapT{});
      });
  SetVector<FunctionOpInterface> sortedFuncs(funcs.begin(), funcs.end());
  SymbolTableCollection symbolTable;
  for (auto funcOp : llvm::reverse(sortedFuncs)) {
    initialize(funcOp);
    funcOp.walk([&](CallOpInterface callOp) {
      auto callee = dyn_cast<FunctionOpInterface>(
          callOp.resolveCallableInTable(&symbolTable));
      update(callOp, callee);
    });
  }
}

StrideInfo *ModuleStrideAnalysis::getStrideInfo(Value value) {
  auto funcOp = value.getParentRegion()->getParentOfType<FunctionOpInterface>();
  auto *strideInfoMap = getFuncData(funcOp);
  if (!strideInfoMap)
    return nullptr;
  auto it = strideInfoMap->find(value);
  if (it == strideInfoMap->end())
    return nullptr;
  return &(it->second);
}

void ModuleStrideAnalysis::initialize(FunctionOpInterface funcOp) {
  std::unique_ptr<DataFlowSolver> solver = createDataFlowSolver();
  AxisInfoLookupFn lookupFn = [this](Value v) -> AxisInfo * {
    return axisInfo.getAxisInfo(v);
  };
  StrideAnalysis *analysis = solver->load<StrideAnalysis>(std::move(lookupFn));
  if (failed(solver->initializeAndRun(funcOp)))
    return;
  auto *strideInfoMap = getFuncData(funcOp);
  auto updateMap = [&](Value value) {
    const auto &info = analysis->getLatticeElement(value)->getValue();
    StrideInfo curInfo;
    if (strideInfoMap->count(value))
      curInfo = StrideInfo::join(info, strideInfoMap->lookup(value));
    else
      curInfo = info;
    (*strideInfoMap)[value] = std::move(curInfo);
  };
  funcOp.walk([&](Operation *op) {
    for (auto value : op->getResults())
      updateMap(value);
  });
  funcOp.walk([&](Block *block) {
    for (auto value : block->getArguments())
      updateMap(value);
  });
}

void ModuleStrideAnalysis::update(CallOpInterface callOp,
                                  FunctionOpInterface callee) {
  // StrideInfo does not forward across call boundaries.
}

} // namespace mlir::triton::intel
