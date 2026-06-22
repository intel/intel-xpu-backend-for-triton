#include "intel/include/Analysis/Range.h"
#include "intel/include/Dialect/Triton/Transforms/Passes.h"
#include "intel/include/Utils/Utility.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

#define DEBUG_TYPE "triton-intel-remove-masks"

using namespace mlir;
namespace tt = mlir::triton;

namespace mlir::triton::intel {
#define GEN_PASS_DEF_TRITONINTELREMOVEMASKS
#include "intel/include/Dialect/Triton/Transforms/Passes.h.inc"
} // namespace mlir::triton::intel

namespace {

// Returns true if `pred` is a supported bound-check predicate.
static bool isSupportedBoundPredicate(arith::CmpIPredicate pred) {
  switch (pred) {
  case arith::CmpIPredicate::slt:
  case arith::CmpIPredicate::sle:
  case arith::CmpIPredicate::ult:
  case arith::CmpIPredicate::ule:
  case arith::CmpIPredicate::sge:
  case arith::CmpIPredicate::sgt:
  case arith::CmpIPredicate::uge:
  case arith::CmpIPredicate::ugt:
    return true;
  default:
    return false;
  }
}

// Classification of a mask cmp against a known loop-IV range.
enum class MaskClassification { AlwaysTrue, AlwaysFalse, Unknown };

// Given a bound-check predicate `pred`, the integer range of the loop IV
// `ivRange`, the make_range operand bounds [rangeStart, rangeEnd), and the
// RHS constant `constVal`, classify the comparison as always-true,
// always-false, or unknown.
//
// The varying LHS equals `IV + make_range`, whose element set has:
//   minElem (inclusive) = ivMin + rangeStart
//   maxElem (inclusive) = (ivMax + rangeEnd) - 1
// Signed vs unsigned comparisons use the signed vs unsigned bounds of
// `ivRange` and the corresponding APInt comparison operators.
static MaskClassification classifyMask(arith::CmpIPredicate pred,
                                       const ConstantIntRanges &ivRange,
                                       int64_t rangeStart, int64_t rangeEnd,
                                       const APInt &constVal) {
  assert(isSupportedBoundPredicate(pred) && "Unsupported predicate");

  bool isSigned =
      (pred == arith::CmpIPredicate::slt || pred == arith::CmpIPredicate::sle ||
       pred == arith::CmpIPredicate::sge || pred == arith::CmpIPredicate::sgt);
  bool isLessThan =
      (pred == arith::CmpIPredicate::slt || pred == arith::CmpIPredicate::ult ||
       pred == arith::CmpIPredicate::sle || pred == arith::CmpIPredicate::ule);
  bool isStrict =
      (pred == arith::CmpIPredicate::slt || pred == arith::CmpIPredicate::ult ||
       pred == arith::CmpIPredicate::sgt || pred == arith::CmpIPredicate::ugt);

  unsigned bitWidth = constVal.getBitWidth();
  APInt ivMin = isSigned ? ivRange.smin() : ivRange.umin();
  APInt ivMax = isSigned ? ivRange.smax() : ivRange.umax();

  // Widen or narrow IV bounds to match constVal's bitwidth so APInt arithmetic
  // and comparisons use a consistent width. When narrowing, bail out on any
  // bound that does not fit in `bitWidth`: a silent truncation would alias a
  // wide value to a narrow one and could flip the classification from Unknown
  // to a bogus AlwaysTrue/AlwaysFalse.
  if (ivMin.getBitWidth() < bitWidth) {
    ivMin = isSigned ? ivMin.sext(bitWidth) : ivMin.zext(bitWidth);
    ivMax = isSigned ? ivMax.sext(bitWidth) : ivMax.zext(bitWidth);
  } else if (ivMin.getBitWidth() > bitWidth) {
    auto fits = [&](const APInt &v) {
      return isSigned ? v.isSignedIntN(bitWidth) : v.isIntN(bitWidth);
    };
    if (!fits(ivMin) || !fits(ivMax))
      return MaskClassification::Unknown;
    ivMin = ivMin.trunc(bitWidth);
    ivMax = ivMax.trunc(bitWidth);
  }

  APInt rangeStartAP(bitWidth, static_cast<uint64_t>(rangeStart),
                     /*isSigned=*/true);
  APInt rangeEndAP(bitWidth, static_cast<uint64_t>(rangeEnd),
                   /*isSigned=*/true);

  // minElem = ivMin + rangeStart; maxElem = ivMax + rangeEnd - 1.
  APInt one(bitWidth, 1);
  APInt minElem = ivMin + rangeStartAP;
  APInt maxElem = ivMax + rangeEndAP - one;

  auto lt = [&](const APInt &a, const APInt &b) {
    return isSigned ? a.slt(b) : a.ult(b);
  };
  auto le = [&](const APInt &a, const APInt &b) {
    return isSigned ? a.sle(b) : a.ule(b);
  };
  auto gt = [&](const APInt &a, const APInt &b) {
    return isSigned ? a.sgt(b) : a.ugt(b);
  };
  auto ge = [&](const APInt &a, const APInt &b) {
    return isSigned ? a.sge(b) : a.uge(b);
  };

  if (isLessThan) {
    if (isStrict) {
      // `<`  (slt or ult): AlwaysTrue if maxElem < constVal
      if (lt(maxElem, constVal))
        return MaskClassification::AlwaysTrue;
      if (ge(minElem, constVal))
        return MaskClassification::AlwaysFalse;
    } else {
      // `<=` (sle or ule): AlwaysTrue if maxElem <= constVal
      if (le(maxElem, constVal))
        return MaskClassification::AlwaysTrue;
      if (gt(minElem, constVal))
        return MaskClassification::AlwaysFalse;
    }
  } else {
    if (isStrict) {
      // `>`  (sgt or ugt): AlwaysTrue if minElem > constVal
      if (gt(minElem, constVal))
        return MaskClassification::AlwaysTrue;
      if (le(maxElem, constVal))
        return MaskClassification::AlwaysFalse;
    } else {
      // `>=` (sge or uge): AlwaysTrue if minElem >= constVal
      if (ge(minElem, constVal))
        return MaskClassification::AlwaysTrue;
      if (lt(maxElem, constVal))
        return MaskClassification::AlwaysFalse;
    }
  }
  return MaskClassification::Unknown;
}

static Operation *dropMask(Operation *op, bool maskVal) {
  assert(op && "Expecting a valid operation");

  OpBuilder builder(op);
  Location loc = op->getLoc();
  TypeSwitch<Operation *>(op)
      .Case<tt::LoadOp>([&](auto loadOp) {
        if (maskVal) {
          auto newLoadOp = tt::LoadOp::create(
              builder, loc, loadOp.getPtr(), loadOp.getCache(),
              loadOp.getEvict(), loadOp.getIsVolatile());
          loadOp->replaceAllUsesWith(newLoadOp);
        } else {
          loadOp->replaceAllUsesWith(ValueRange{loadOp.getOther()});
        }
      })
      .Case<arith::SelectOp>([&](auto selectOp) {
        Value origRes = selectOp.getResult();
        Value selectedVal =
            (maskVal ? selectOp.getTrueValue() : selectOp.getFalseValue());
        Value newRes = selectedVal;
        if (auto opResult = dyn_cast<OpResult>(selectedVal)) {
          Operation *defOp = opResult.getDefiningOp();
          newRes = defOp->getOpResult(opResult.getResultNumber());
        }
        origRes.replaceAllUsesWith(newRes);
      });

  return nullptr;
}

// Abstract base class for mask validators.
// Mask validators are used to check whether a given mask has an expected form.
// Concrete subclasses provide a member function used to select masked
// operations that have a mask in a particular (e.g. desired) form.
class MaskValidatorBase {
public:
  virtual ~MaskValidatorBase() = default;

  // Check whether the given mask is valid.
  virtual bool isValidMask(scf::ForOp &forOp, Value mask,
                           Operation *op) const = 0;

  // Create the loop versioning condition based on the mask.
  virtual Value getVersioningCond(scf::ForOp &forOp, Value mask) const = 0;

  virtual std::string getName() const = 0;
};

// A mask validator which ensures the mask is not necessary.
class RemovableMaskValidator final : public MaskValidatorBase {
public:
  RemovableMaskValidator(DataFlowSolver *solver)
      : MaskValidatorBase(), solver(solver) {}

  virtual bool isValidMask(scf::ForOp &forOp, Value mask, Operation *op) const {
    MaskClassification cls = classify(forOp, mask);
    if (cls == MaskClassification::Unknown)
      return false;
    registerMaskValue(op, cls == MaskClassification::AlwaysTrue);
    return true;
  }

  virtual Value getVersioningCond(scf::ForOp &forOp, Value mask) const {
    return {};
  }

  virtual std::string getName() const { return "RemovableMaskValidator"; }

  bool getMaskValue(Operation *op) const {
    assert(opToMaskValue.find(op) != opToMaskValue.end() && "mask not present");
    return opToMaskValue[op];
  }

  // Dispatch on the mask's defining op: `arith.andi` is combined recursively,
  // otherwise fall through to cmp classification.
  MaskClassification classify(scf::ForOp &forOp, Value mask) const {
    Value finalVal = tt::intel::getFinalValue(mask);
    assert(finalVal && "Expecting a valid mask");

    if (auto andOp = dyn_cast_or_null<arith::AndIOp>(finalVal.getDefiningOp()))
      return classifyAnd(forOp, andOp);
    return classifyCmp(forOp, finalVal);
  }

private:
  // Record the final mask value for the given masked operation.
  // Fixes #6871: each operation must record only its own mask, not the masks of
  // its other users (which would poison the map for ops consuming two masks,
  // e.g. an arith.select using one mask as condition and another as
  // true-value).
  void registerMaskValue(Operation *op, bool maskVal) const {
    opToMaskValue.insert({op, maskVal});
  }

  // Combine classifications of the two operands of an `arith.andi` mask:
  //   AlwaysTrue  iff BOTH operands are AlwaysTrue
  //   AlwaysFalse iff EITHER operand is AlwaysFalse
  //   Unknown     otherwise
  MaskClassification classifyAnd(scf::ForOp &forOp, arith::AndIOp andOp) const {
    MaskClassification lhsCls = classify(forOp, andOp.getLhs());
    MaskClassification rhsCls = classify(forOp, andOp.getRhs());

    if (lhsCls == MaskClassification::AlwaysFalse ||
        rhsCls == MaskClassification::AlwaysFalse)
      return MaskClassification::AlwaysFalse;
    if (lhsCls == MaskClassification::AlwaysTrue &&
        rhsCls == MaskClassification::AlwaysTrue)
      return MaskClassification::AlwaysTrue;
    return MaskClassification::Unknown;
  }

  // Check whether a value is the loop induction variable or a loop iter_arg
  // that is equivalent to the IV (same init as lower bound, same step).
  std::optional<ConstantIntRanges> getIVEquivalentRange(scf::ForOp &forOp,
                                                        Value val) const {
    std::optional<ConstantIntRanges> ivRange =
        tt::intel::collectLoopIVRange(forOp, *solver);
    if (!ivRange)
      return std::nullopt;

    if (val == forOp.getSingleInductionVar())
      return ivRange;

    // Check if val resolved (via getFinalValue) to the loop's lower bound
    // constant, meaning it was an iter_arg with that init. Verify there
    // exists an iter_arg with init == lb and yield == self + step.
    OpFoldResult lbOFR = *forOp.getSingleLowerBound();
    OpFoldResult stepOFR = *forOp.getSingleStep();

    auto getConstant = [](OpFoldResult ofr) -> std::optional<int64_t> {
      if (auto attr = dyn_cast<Attribute>(ofr)) {
        if (auto intAttr = dyn_cast_or_null<IntegerAttr>(attr))
          return intAttr.getInt();
        return std::nullopt;
      }
      APInt intVal;
      if (matchPattern(cast<Value>(ofr), m_ConstantInt(&intVal)))
        return intVal.getSExtValue();
      return std::nullopt;
    };

    std::optional<int64_t> lbConst = getConstant(lbOFR);
    std::optional<int64_t> stepConst = getConstant(stepOFR);
    if (!lbConst || !stepConst)
      return std::nullopt;

    auto matchesInt = [](Value v, int64_t expected) -> bool {
      APInt intVal;
      if (matchPattern(v, m_ConstantInt(&intVal)))
        return intVal.getSExtValue() == expected;
      DenseElementsAttr denseAttr;
      if (matchPattern(v, m_Constant(&denseAttr)) && denseAttr.isSplat()) {
        if (auto intAttr =
                dyn_cast<IntegerAttr>(denseAttr.getSplatValue<Attribute>()))
          return intAttr.getInt() == expected;
      }
      return false;
    };

    if (!matchesInt(val, *lbConst))
      return std::nullopt;

    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    for (unsigned i = 0, e = forOp.getNumRegionIterArgs(); i < e; ++i) {
      Value initArg = forOp.getInitArgs()[i];
      if (!matchesInt(initArg, *lbConst))
        continue;

      Value yieldVal = yieldOp.getOperand(i);
      auto yieldAdd = yieldVal.getDefiningOp<arith::AddIOp>();
      if (!yieldAdd)
        continue;

      BlockArgument iterArg = forOp.getRegionIterArg(i);
      bool lhsIsIterArg = (yieldAdd.getLhs() == iterArg);
      bool rhsIsIterArg = (yieldAdd.getRhs() == iterArg);
      if (!lhsIsIterArg && !rhsIsIterArg)
        continue;

      Value stepOperand = lhsIsIterArg ? yieldAdd.getRhs() : yieldAdd.getLhs();
      if (matchesInt(stepOperand, *stepConst))
        return ivRange;
    }

    return std::nullopt;
  }

  // Classify a single `arith.cmpi` mask against the loop IV range.
  MaskClassification classifyCmp(scf::ForOp &forOp, Value finalVal) const {
    if (!finalVal.getDefiningOp() ||
        !isa<arith::CmpIOp>(finalVal.getDefiningOp()))
      return MaskClassification::Unknown;

    auto cmpOp = cast<arith::CmpIOp>(finalVal.getDefiningOp());
    arith::CmpIPredicate pred = cmpOp.getPredicate();
    if (!isSupportedBoundPredicate(pred))
      return MaskClassification::Unknown;

    Value lhs = tt::intel::getFinalValue(cmpOp.getLhs());
    Value rhs = tt::intel::getFinalValue(cmpOp.getRhs());
    Operation *lhsOp = tt::intel::getFinalValue(lhs).getDefiningOp();
    Operation *rhsOp = tt::intel::getFinalValue(rhs).getDefiningOp();
    if (!lhsOp || !rhsOp)
      return MaskClassification::Unknown;

    auto getIntConstantValue = [](Operation *op) -> std::optional<APInt> {
      APInt intVal;
      if (op->getNumResults() > 0 &&
          matchPattern(op->getResult(0), m_ConstantInt(&intVal)))
        return intVal;
      return std::nullopt;
    };

    // TODO: consider the case where the constant is lhs.
    std::optional<APInt> constIntVal = getIntConstantValue(rhsOp);
    if (!constIntVal)
      return MaskClassification::Unknown;

    auto addOp = dyn_cast<arith::AddIOp>(lhsOp);
    if (!addOp)
      return MaskClassification::Unknown;

    Value addLhs = tt::intel::getFinalValue(addOp.getLhs());
    Value addRhs = tt::intel::getFinalValue(addOp.getRhs());

    std::optional<ConstantIntRanges> lhsRange =
        getIVEquivalentRange(forOp, addLhs);
    if (!lhsRange)
      return MaskClassification::Unknown;

    auto makeRangeOp =
        dyn_cast_or_null<tt::MakeRangeOp>(addRhs.getDefiningOp());
    if (!makeRangeOp)
      return MaskClassification::Unknown;

    return classifyMask(pred, *lhsRange, makeRangeOp.getStart(),
                        makeRangeOp.getEnd(), *constIntVal);
  }

  DataFlowSolver *solver;
  mutable std::map<Operation *, bool> opToMaskValue;
};

// A mask validator which ensures that the mask can be reduced to the form:
//  `END-1 < N-i*END`
class CanonicalMaskValidator final : public MaskValidatorBase {
public:
  // This structure is used to store the information about a mask in canonical
  // form (N + END - 1) / END.
  struct MaskInfo {
    Value N;
    unsigned END;
  };

  // Check whether the mask is equivalent to the form: `END-1 < N-i*END`.
  virtual bool isValidMask(scf::ForOp &forOp, Value mask, Operation *op) const {
    Value finalVal = tt::intel::getFinalValue(mask);
    assert(finalVal && "Expecting a valid mask");

    if (!finalVal.getDefiningOp() ||
        !isa<arith::CmpIOp>(finalVal.getDefiningOp()))
      return false;

    auto cmpOp = cast<arith::CmpIOp>(finalVal.getDefiningOp());
    arith::CmpIPredicate pred = cmpOp.getPredicate();
    // The canonical-form recognition and the versioning condition generated
    // below (RemSIOp + sgt, threshold `UB == ((N - END) / END) + 1`, and
    // DivSIOp match for the loop upper bound) assume strict signed `<`.
    // Accepting sle/ult/ule here would silently generate off-by-one
    // versioning conditions against a DivSIOp-folded upper bound and unsafely
    // drop the mask in the "then" region.
    if (pred != arith::CmpIPredicate::slt)
      return false;

    Operation *lhs = tt::intel::getFinalValue(cmpOp.getLhs()).getDefiningOp();
    Operation *rhs = tt::intel::getFinalValue(cmpOp.getRhs()).getDefiningOp();
    if (!lhs || !rhs || !isa<tt::MakeRangeOp>(lhs) || !isa<arith::SubIOp>(rhs))
      return false;

    auto rangeOp = cast<tt::MakeRangeOp>(lhs);
    unsigned end = rangeOp.getEnd();
    assert(end > rangeOp.getStart() && "Invalid range");

    auto subOp = cast<arith::SubIOp>(rhs);
    Operation *subLhs = subOp.getLhs().getDefiningOp();
    Operation *subRhs = subOp.getRhs().getDefiningOp();
    if (subLhs && !isa<arith::ConstantIntOp>(subLhs))
      return false;
    if (!subRhs || !isa<arith::MulIOp>(subRhs))
      return false;

    auto mulOp = cast<arith::MulIOp>(subRhs);
    Operation *defMulLhs = mulOp.getLhs().getDefiningOp();
    Operation *defMulRhs = mulOp.getRhs().getDefiningOp();
    if (defMulLhs && defMulRhs)
      return false;

    std::optional<Value> loopIV = forOp.getSingleInductionVar();
    assert(loopIV.has_value() && "Failed to find loop induction variable");

    if (!defMulLhs && mulOp.getLhs() == *loopIV &&
        isa<arith::ConstantIntOp>(defMulRhs))
      return cast<arith::ConstantIntOp>(defMulRhs).value() == end;

    if (!defMulRhs && mulOp.getRhs() == *loopIV &&
        isa<arith::ConstantIntOp>(defMulLhs))
      return cast<arith::ConstantIntOp>(defMulLhs).value() == end;

    return false;
  }

  // Create the loop versioning condition.
  // At this point the loop upper bound is in canonical form
  // `(N+END-1)/END` (possibly folded), the versioning condition will be:
  // `(N+END-1)%END > 0 && N > END`.
  virtual Value getVersioningCond(scf::ForOp &forOp, Value mask) const {
    Value finalVal = tt::intel::getFinalValue(mask);
    assert(finalVal && "Expecting a valid mask");

    MaskInfo maskInfo = getMaskInfo(forOp, finalVal);
    if (!hasCanonicalUpperBound(forOp, maskInfo))
      return nullptr;

    OpBuilder builder(forOp);
    Location loc = forOp.getLoc();
    Value ub = tt::intel::getFinalValue(forOp.getUpperBound());
    Operation *defOp = ub.getDefiningOp();
    assert(defOp && "Expecting a valid operation");

    // The loop UB is a constant.
    if (isa<arith::ConstantIntOp>(defOp)) {
      int64_t UB = cast<arith::ConstantIntOp>(defOp).value();
      int64_t N =
          cast<arith::ConstantIntOp>(maskInfo.N.getDefiningOp()).value();
      unsigned END = maskInfo.END;
      bool cond = UB == ((N - END) / END) + 1;
      return arith::ConstantIntOp::create(builder, forOp.getLoc(),
                                          builder.getI1Type(), cond);
    }

    auto divOp = cast<arith::DivSIOp>(defOp);
    Operation *divLhsOp = divOp.getLhs().getDefiningOp();
    auto divNumOp = cast<arith::AddIOp>(divLhsOp);
    Value lhs = divNumOp.getLhs();
    Value rhs = divOp.getRhs();

    Value zero = tt::intel::findOrCreateIntConstant(
        loc, 0, lhs.getType().getIntOrFloatBitWidth(), builder);
    Value cmp1 = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::eq,
        arith::RemSIOp::create(builder, loc, lhs, rhs), zero);
    Value cmp2 = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sgt,
                                       lhs, rhs);
    return arith::AndIOp::create(builder, loc, cmp1, cmp2);
  }

  virtual std::string getName() const { return "CanonicalMaskValidator"; }

  // Ensure the loop upper bound is in canonical form (N+END-1)/END.
  static bool hasCanonicalUpperBound(scf::ForOp &forOp,
                                     const MaskInfo &maskInfo) {
    Value ub = tt::intel::getFinalValue(forOp.getUpperBound());
    Operation *defOp = ub.getDefiningOp();
    if (!defOp)
      return false;

    // If the loop UB is constant, use `MaskInfo` to determine whether the UB
    // was folded from a canonical form.
    if (isa<arith::ConstantIntOp>(defOp)) {
      int64_t UB = cast<arith::ConstantIntOp>(defOp).value();
      int64_t N =
          cast<arith::ConstantIntOp>(maskInfo.N.getDefiningOp()).value();
      unsigned END = maskInfo.END;
      return UB == ((N - END) / END) + 1;
    }

    if (!isa<arith::DivSIOp>(defOp))
      return false;

    auto divOp = cast<arith::DivSIOp>(defOp);
    Operation *divLhsOp = divOp.getLhs().getDefiningOp();
    Operation *divRhsOp = divOp.getRhs().getDefiningOp();
    if (!divLhsOp || !divRhsOp || !isa<arith::AddIOp>(divLhsOp) ||
        !isa<arith::ConstantOp>(divRhsOp))
      return false;

    auto divNumOp = cast<arith::AddIOp>(divLhsOp);
    auto divDenOp = cast<arith::ConstantIntOp>(divRhsOp);
    Operation *addLhsOp = divNumOp.getLhs().getDefiningOp();
    Operation *addRhsOp = divNumOp.getRhs().getDefiningOp();
    if (addLhsOp || !isa<arith::ConstantIntOp>(addRhsOp) ||
        (divDenOp.value() != cast<arith::ConstantIntOp>(addRhsOp).value() + 1))
      return false;

    return true;
  }

private:
  // Assuming the mask is equivalent to the form: `END < N-i*END`, returns a
  // structure containing `N` and `END`.
  MaskInfo getMaskInfo(scf::ForOp &forOp, Value mask) const {
    assert(isValidMask(forOp, mask, /*op=*/nullptr) &&
           "Expecting a valid mask");

    Value finalMask = tt::intel::getFinalValue(mask);
    auto cmpOp = cast<arith::CmpIOp>(finalMask.getDefiningOp());
    Operation *lhs = tt::intel::getFinalValue(cmpOp.getLhs()).getDefiningOp();
    Operation *rhs = tt::intel::getFinalValue(cmpOp.getRhs()).getDefiningOp();
    return MaskInfo{cast<arith::SubIOp>(rhs).getLhs(),
                    cast<tt::MakeRangeOp>(lhs).getEnd()};
  }
};

// This mask validator ensures the mask is loop invariant.
class InvariantMaskValidator final : public MaskValidatorBase {
public:
  // The mask must have one of the forms:
  //   - N < M (with i1 data type)
  //   - [0..END] < splat(N)
  //   - splat(N) < [0..END]
  //   - arith.andi of valid sub-masks (compound boundary checks)
  virtual bool isValidMask(scf::ForOp &forOp, Value mask, Operation *op) const {
    Value finalVal = tt::intel::getFinalValue(mask);
    assert(finalVal && "Expecting a valid mask");

    if (!finalVal.getDefiningOp())
      return false;

    // Handle compound andi masks by recursing into both operands.
    if (auto andOp = dyn_cast<arith::AndIOp>(finalVal.getDefiningOp()))
      return isValidMask(forOp, andOp.getLhs(), op) &&
             isValidMask(forOp, andOp.getRhs(), op);

    if (!isa<arith::CmpIOp>(finalVal.getDefiningOp()))
      return false;

    auto cmpOp = cast<arith::CmpIOp>(finalVal.getDefiningOp());
    arith::CmpIPredicate pred = cmpOp.getPredicate();
    if (!isSupportedBoundPredicate(pred))
      return false;

    // The '>=' and '>' predicates are supported in classifyMask (loop-dependent
    // path) but not yet handled by getVersioningCond which always uses END-1 as
    // the scalar threshold -- correct for '<' and '<=' but wrong for '>=' and
    // '>'.
    if (pred == arith::CmpIPredicate::sge ||
        pred == arith::CmpIPredicate::sgt ||
        pred == arith::CmpIPredicate::uge || pred == arith::CmpIPredicate::ugt)
      return false;

    bool isInLoop = (cmpOp->getParentOfType<scf::ForOp>() == forOp);
    if (isInLoop)
      return false;

    Value lhsVal = tt::intel::getFinalValue(cmpOp.getLhs());
    Value rhsVal = tt::intel::getFinalValue(cmpOp.getRhs());
    Operation *lhs = tt::intel::getFinalValue(lhsVal).getDefiningOp();
    Operation *rhs = tt::intel::getFinalValue(rhsVal).getDefiningOp();

    if (!lhs && !rhs) {
      assert(lhsVal.getType() == rhsVal.getType() && "Invalid types");
      assert(isa<IntegerType>(lhsVal.getType()) &&
             cast<IntegerType>(lhsVal.getType()).getWidth() == 1 &&
             "Invalid type");
      return true;
    }

    if (!rhs && isa<tt::MakeRangeOp>(lhs)) {
      [[maybe_unused]] auto rangeOp = cast<tt::MakeRangeOp>(lhs);
      assert(rangeOp.getStart() < rangeOp.getEnd() && "Invalid range");
      return true;
    }

    if (!lhs && isa<tt::MakeRangeOp>(rhs)) {
      [[maybe_unused]] auto rangeOp = cast<tt::MakeRangeOp>(rhs);
      assert(rangeOp.getStart() < rangeOp.getEnd() && "Invalid range");
      return true;
    }

    return false;
  } // namespace

  virtual Value getVersioningCond(scf::ForOp &forOp, Value mask) const {
    assert(isValidMask(forOp, mask, /*op=*/nullptr) && "Invalid mask");

    OpBuilder builder(forOp);
    Location loc = forOp.getLoc();
    Value finalMask = tt::intel::getFinalValue(mask);

    // Handle compound andi: AND the versioning conditions of both operands.
    if (auto andOp = dyn_cast<arith::AndIOp>(finalMask.getDefiningOp())) {
      Value lhsCond = getVersioningCond(forOp, andOp.getLhs());
      Value rhsCond = getVersioningCond(forOp, andOp.getRhs());
      return builder.createOrFold<arith::AndIOp>(loc, lhsCond, rhsCond);
    }

    auto cmpOp = cast<arith::CmpIOp>(finalMask.getDefiningOp());
    arith::CmpIPredicate pred = cmpOp.getPredicate();
    Value lhsVal = tt::intel::getFinalValue(cmpOp.getLhs());
    Value rhsVal = tt::intel::getFinalValue(cmpOp.getRhs());
    Operation *lhs = tt::intel::getFinalValue(lhsVal).getDefiningOp();
    Operation *rhs = tt::intel::getFinalValue(rhsVal).getDefiningOp();

    // N < M (with i1 data type)
    if (!lhs && !rhs)
      return builder.createOrFold<arith::CmpIOp>(loc, pred, lhsVal, rhsVal);

    // [0..END] < splat(N) -- generate versioning condition 'END-1 < N'.
    if (!rhs && isa<tt::MakeRangeOp>(lhs)) {
      [[maybe_unused]] auto rangeOp = cast<tt::MakeRangeOp>(lhs);
      assert(rangeOp.getStart() < rangeOp.getEnd() && "Invalid range");
      unsigned end = rangeOp.getEnd() - 1u;
      auto cstOp = tt::intel::findOrCreateIntConstant(
          loc, end, rhsVal.getType().getIntOrFloatBitWidth(), builder);
      return builder.createOrFold<arith::CmpIOp>(loc, pred, cstOp, rhsVal);
    }

    // splat(N) < [0..END] -- generate versioning condition 'N < END'.
    if (!lhs && isa<tt::MakeRangeOp>(rhs)) {
      [[maybe_unused]] auto rangeOp = cast<tt::MakeRangeOp>(rhs);
      assert(rangeOp.getStart() < rangeOp.getEnd() && "Invalid range");
      unsigned start = rangeOp.getStart();
      auto cstOp = builder.createOrFold<arith::ConstantIntOp>(
          loc, lhsVal.getType(), start);
      return builder.createOrFold<arith::CmpIOp>(loc, pred, lhsVal, cstOp);
    }

    llvm_unreachable("Unexpected mask");
    return {};
  }

  virtual std::string getName() const { return "InvariantMaskValidator"; }
};

// Collects masked operations in a loop that satisfy the condition imposed by
// the mask validator associated with this class.
template <typename MaskValidator> class MaskedOpsCollector {
public:
  using MaskedOperations = SmallPtrSet<Operation *, 8>;

  MaskedOpsCollector(scf::ForOp &forOp, MaskValidator &maskValidator)
      : forOp(forOp), maskValidator(maskValidator) {}

  bool collectMaskedOps() {
    auto collectMaskedOps = [&](auto ops, MaskedOperations &maskedOps) {
      for (Operation *op : ops) {
        Value mask = getMask(op);
        if (mask && maskValidator.isValidMask(forOp, mask, op)) {
          maskedOps.insert(op);
          LLVM_DEBUG(llvm::dbgs()
                     << maskValidator.getName()
                     << ": collected masked operation: " << *op << "\n");
        }
      }
    };

    collectMaskedOps(forOp.getOps<tt::LoadOp>(), maskedOps);
    collectMaskedOps(forOp.getOps<tt::StoreOp>(), maskedOps);
    collectMaskedOps(forOp.getOps<arith::SelectOp>(), maskedOps);
    return maskedOps.size();
  }

  const MaskedOperations &getMaskedOps() const { return maskedOps; };
  const MaskValidator &getMaskValidator() const { return maskValidator; }

  Value getMask(Operation *op) const {
    assert(op && "Expecting a valid operation");
    return TypeSwitch<Operation *, Value>(op)
        .Case<tt::LoadOp, tt::StoreOp>(
            [](auto maskedOp) { return maskedOp.getMask(); })
        .template Case<arith::SelectOp>(
            [](auto selectOp) { return selectOp.getCondition(); })
        .Default([](auto) { return nullptr; });
  }

private:
  scf::ForOp &forOp;
  MaskValidator &maskValidator;
  MaskedOperations maskedOps;
};

class LoopVersioner {
public:
  // Version the \p forOp loop with a condition that makes the masks collected
  // by \p collector unnecessary.
  // TODO: Extend the versioning region to encompass the downward exposed uses
  // of the return values.
  static bool version(scf::ForOp &forOp,
                      MaskedOpsCollector<CanonicalMaskValidator> &collector) {
    assert(!collector.getMaskedOps().empty() &&
           "Expecting a non-empty collection of masked operations");

    // Limitation
    auto getMask = [](Operation *maskedOp) {
      assert(isa<tt::LoadOp>(maskedOp) ||
             isa<tt::StoreOp>(maskedOp) &&
                 "Expecting a load or store operation");
      Value mask = isa<tt::LoadOp>(maskedOp)
                       ? cast<tt::LoadOp>(maskedOp).getMask()
                       : cast<tt::StoreOp>(maskedOp).getMask();
      return tt::intel::getFinalValue(mask);
    };

    // Retrieve the versioning condition, bail out if it doesn't exist (in
    // which case the loop upper bound is not in canonical form).
    Operation *maskedOp = *collector.getMaskedOps().begin();
    Value verCond = collector.getMaskValidator().getVersioningCond(
        forOp, getMask(maskedOp));
    if (!verCond)
      return false;

    // This lambda is used to collect the types for the loop results that are
    // downward exposed (i.e. used by other operations).
    auto getUsedResults = [](const scf::ForOp &forOp) {
      SmallVector<Type> resTypes;
      for (Value res : forOp->getResults()) {
        if (!res.getUsers().empty())
          resTypes.push_back(res.getType());
      }
      return resTypes;
    };

    // Create the versioning branch.
    OpBuilder builder(forOp);
    Location loc = forOp.getLoc();
    auto ifOp = scf::IfOp::create(builder, loc, getUsedResults(forOp), verCond,
                                  /*withThenRegion=*/true);

    // Clone the original loop into the 2 if branches.
    IRMapping map;
    OpBuilder thenB = ifOp.getThenBodyBuilder();
    Operation *thenForLoop = thenB.clone(*forOp.getOperation(), map);
    OpBuilder elseB = ifOp.getElseBodyBuilder();
    Operation *elseForLoop = elseB.clone(*forOp.getOperation());

    // Collect results in 'clonedLoop' corresponding to downward exposed
    // results of the given loop.
    auto pruneUnusedResults = [&](const scf::ForOp &forOp,
                                  Operation *clonedLoop) {
      SmallVector<Value> prunedResults;
      for (auto [idx, val] : llvm::enumerate(forOp->getResults())) {
        if (!val.getUsers().empty())
          prunedResults.push_back(clonedLoop->getResult(idx));
      }
      return prunedResults;
    };

    // Create the yield operations for the two if branches.
    scf::YieldOp::create(thenB, loc, pruneUnusedResults(forOp, thenForLoop));
    scf::YieldOp::create(elseB, loc, pruneUnusedResults(forOp, elseForLoop));

    // Drop the mask from candidate masked operations in the "then" region.
    for (Operation *maskedOp : collector.getMaskedOps()) {
      Operation *mappedOp = map.lookup(maskedOp);
      if (auto loadOp = dyn_cast<tt::LoadOp>(mappedOp)) {
        OpBuilder builder(mappedOp);
        auto newLoad = tt::LoadOp::create(
            builder, loadOp.getLoc(), loadOp.getPtr(), loadOp.getCache(),
            loadOp.getEvict(), loadOp.getIsVolatile());
        mappedOp->replaceAllUsesWith(newLoad);
        mappedOp->erase();
      }
      // TODO: stores
    }

    // Replace the uses of the original loop results.
    unsigned idx = 0;
    for (Value res : forOp.getResults()) {
      if (!res.getUsers().empty())
        res.replaceAllUsesWith(ifOp->getResult(idx++));
    }

    forOp.erase();
    return true;
  }

  static bool version(scf::ForOp &forOp,
                      MaskedOpsCollector<InvariantMaskValidator> &collector) {
    assert(!collector.getMaskedOps().empty() &&
           "Expecting a non-empty collection of masked operations");

    // Collect the (loop invariant) mask conditions, looking through
    // broadcast/splat/expand_dims to find the underlying CmpIOp.
    SmallPtrSet<Operation *, 8> maskConds;
    for (Operation *maskedOp : collector.getMaskedOps()) {
      if (auto loadOp = dyn_cast<tt::LoadOp>(maskedOp))
        maskConds.insert(
            tt::intel::getFinalValue(loadOp.getMask()).getDefiningOp());
      if (auto storeOp = dyn_cast<tt::StoreOp>(maskedOp))
        maskConds.insert(
            tt::intel::getFinalValue(storeOp.getMask()).getDefiningOp());
    }

    // Early return if no mask conditions were collected.
    if (maskConds.empty())
      return false;

    // Combine the versioning conditions.
    OpBuilder builder(forOp);
    Location loc = forOp.getLoc();
    auto it = maskConds.begin();
    Value firstCond = (*it++)->getResult(0);
    auto maskValidator = collector.getMaskValidator();
    Value verCond = maskValidator.getVersioningCond(forOp, firstCond);
    for (; it != maskConds.end(); ++it) {
      Value nextCond = (*it)->getResult(0);
      Value cond = maskValidator.getVersioningCond(forOp, nextCond);
      verCond = arith::AndIOp::create(builder, loc, verCond, cond);
    }

    auto ifOp = scf::IfOp::create(builder, loc, forOp.getResultTypes(), verCond,
                                  /*withThenRegion=*/true);

    // Clone the original loop into the 2 if branches.
    IRMapping map;
    OpBuilder thenB = ifOp.getThenBodyBuilder();
    Operation *thenForLoop = thenB.clone(*forOp.getOperation(), map);
    OpBuilder elseB = ifOp.getElseBodyBuilder();
    Operation *elseForLoop = elseB.clone(*forOp.getOperation());

    // Create the yield operations for the two if branches.
    if (!thenForLoop->getResults().empty()) {
      scf::YieldOp::create(thenB, loc, thenForLoop->getResults());
      scf::YieldOp::create(elseB, loc, elseForLoop->getResults());
    }

    // Drop the mask from candidate masked operations in the "then" region's
    // cloned loop.
    for (Operation *maskedOp : collector.getMaskedOps()) {
      Operation *mappedOp = map.lookup(maskedOp);
      if (auto loadOp = dyn_cast<tt::LoadOp>(mappedOp)) {
        OpBuilder builder(mappedOp);
        auto newLoad = tt::LoadOp::create(
            builder, loadOp.getLoc(), loadOp.getPtr(), loadOp.getCache(),
            loadOp.getEvict(), loadOp.getIsVolatile());
        mappedOp->replaceAllUsesWith(newLoad);
        mappedOp->erase();
      }
      // TODO: stores
    }

    // Replace the uses of the original loop results.
    for (const auto &[i, v] : llvm::enumerate(forOp.getResults()))
      if (!v.getUsers().empty())
        v.replaceAllUsesWith(ifOp->getResult(i));

    forOp.erase();
    return true;
  }
};

struct TritonIntelRemoveMasksBase
    : tt::intel::impl::TritonIntelRemoveMasksBase<TritonIntelRemoveMasksBase> {
public:
  using Base::Base;
  using IndexMapSet = std::map<int, std::set<int>>;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    std::shared_ptr<DataFlowSolver> solver = createDataFlowSolver();
    auto *rangeAnalysis = solver->load<tt::intel::IntegerRangeAnalysis>(
        moduleOp, getAnalysis<DominanceInfo>());

    if (failed(solver->initializeAndRun(getOperation())))
      return signalPassFailure();

    // Remove masks if they are not necessary.
    moduleOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (scf::ForOp forOp = dyn_cast<scf::ForOp>(op)) {
        // Nested loop aren't currently handled.
        if (forOp->template getParentOfType<scf::ForOp>())
          return WalkResult::advance();

        if (!forOp.getSingleInductionVar())
          return WalkResult::advance();

        RemovableMaskValidator maskValidator(solver.get());
        MaskedOpsCollector collector(forOp, maskValidator);
        if (collector.collectMaskedOps()) {
          for (Operation *op : collector.getMaskedOps()) {
            bool maskVal = maskValidator.getMaskValue(op);
            dropMask(op, maskVal);
          }
        }
      }
      return WalkResult::advance();
    });

    // Version loops containing masked operation in canonical form.
    moduleOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (scf::ForOp forOp = dyn_cast<scf::ForOp>(op)) {
        // Nested loop aren't currently handled.
        if (forOp->template getParentOfType<scf::ForOp>())
          return WalkResult::advance();

        if (!forOp.getSingleInductionVar())
          return WalkResult::advance();

        CanonicalMaskValidator maskValidator;
        MaskedOpsCollector collector(forOp, maskValidator);
        if (collector.collectMaskedOps()) {
          [[maybe_unused]] bool loopVersioned =
              LoopVersioner::version(forOp, collector);
          LLVM_DEBUG(if (loopVersioned) llvm::dbgs() << "Loop versioned\n");
        }
      }
      return WalkResult::advance();
    });

    // Version loops containing masked operation with a mask defined before
    // the loop.
    moduleOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (scf::ForOp forOp = dyn_cast<scf::ForOp>(op)) {
        // Nested loop aren't currently handled.
        if (forOp->template getParentOfType<scf::ForOp>())
          return WalkResult::advance();

        InvariantMaskValidator maskValidator;
        MaskedOpsCollector collector(forOp, maskValidator);
        if (collector.collectMaskedOps()) {
          [[maybe_unused]] bool loopVersioned =
              LoopVersioner::version(forOp, collector);
          LLVM_DEBUG(if (loopVersioned) llvm::dbgs() << "Loop versioned\n");
        }
      }
      return WalkResult::advance();
    });

    LLVM_DEBUG(llvm::dbgs() << "After versioning:\n" << moduleOp << "\n");
    assert(succeeded(verify(moduleOp)) && "Module verification failed");
  }
};

} // namespace
