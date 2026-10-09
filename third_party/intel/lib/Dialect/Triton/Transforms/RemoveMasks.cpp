#include "intel/include/Analysis/Range.h"
#include "intel/include/Analysis/SymbolicBounds.h"
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
#include "triton/Tools/Sys/GetEnv.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/xxhash.h"
#include <optional>
#include <type_traits>

#define DEBUG_TYPE "triton-intel-remove-masks"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

// The census trace sits on its own debug type so a corpus run can
// enable it alone: `-debug-only=triton-intel-remove-masks` also turns on this
// pass's after-versioning module dumps, which are two orders of magnitude more
// output than the census itself. The printed prefix stays the pass's, so the
// corpus tooling greps a single pattern either way. Helpers used only under
// CDBG are [[maybe_unused]]: they are dead when debug output is compiled out.
#define CENSUS_DEBUG_TYPE "triton-intel-remove-masks-census"
#define CDBG(X) DEBUG_WITH_TYPE(CENSUS_DEBUG_TYPE, DBGS() << X << "\n")

using namespace mlir;
namespace tt = mlir::triton;

namespace mlir::triton::intel {
#define GEN_PASS_DEF_TRITONINTELREMOVEMASKS
#include "intel/include/Dialect/Triton/Transforms/Passes.h.inc"
} // namespace mlir::triton::intel

namespace {

// Census scaffolding, debug-only: trace lines keyed by a stable per-mask id.
// In `census:` lines, `walk=` numbers the legacy driver's three walks in
// order: 1 is RemovableMaskValidator, 2 CanonicalMaskValidator, 3
// InvariantMaskValidator. Ids are stored as a discardable attribute (no
// dialect prefix, so no dialect verifier sees it): loop clones made by
// versioning inherit them, and stripCensusIds removes them at pass end.
static constexpr StringLiteral kCensusIdAttr = "census_id";

// Null-safe: the validators' internal calls pass op == nullptr.
static StringRef censusId(Operation *op) {
  auto attr = op ? op->getAttrOfType<StringAttr>(kCensusIdAttr) : StringAttr();
  return attr ? attr.getValue() : StringRef("?");
}

// The mask a candidate op carries, or null. A superset of what the collectors
// read: masked stores and atomics are counted but never examined.
static Value censusMask(Operation *op) {
  return TypeSwitch<Operation *, Value>(op)
      .Case<tt::LoadOp, tt::StoreOp, tt::AtomicLoadOp, tt::AtomicStoreOp,
            tt::AtomicRMWOp>([](auto o) { return o.getMask(); })
      .Case<arith::SelectOp>([](auto o) { return o.getCondition(); })
      .Default([](Operation *) { return Value(); });
}

// Classifies a loop bound for the census: "const", "arg" (block argument),
// "cdiv" (divsi of an addi), or the defining op name.
static std::string describeBound(Value v) {
  v = tt::intel::getFinalValue(v);
  // m_ConstantInt binds into its argument, so it needs a real APInt: passing
  // nullptr dereferences null.
  APInt cst;
  if (matchPattern(v, m_ConstantInt(&cst)))
    return "const";
  Operation *def = v.getDefiningOp();
  if (!def)
    return "arg";
  if (auto div = dyn_cast<arith::DivSIOp>(def))
    if (div.getLhs().getDefiningOp<arith::AddIOp>())
      return "cdiv";
  return def->getName().getStringRef().str();
}

// `argN` for a function argument, else `describeBound`'s coarse
// classification: the versioning trace only needs enough to join
// a `versioned:` guard against the census's `candidate:`/`verdict:` lines by
// eye, not a full expression.
[[maybe_unused]] static std::string describeArg(Value v) {
  v = tt::intel::getFinalValue(v);
  if (auto arg = dyn_cast<BlockArgument>(v))
    return "arg" + std::to_string(arg.getArgNumber());
  return describeBound(v);
}

// id = <func>#<xxh3 of the func's printed IR>/L<pre-order loop index>/M<mask
// index>. The hash separates specializations that share a kernel name. Every
// masked op whose innermost enclosing loop is a scf.for gets an id and a
// candidate line, including ops the drivers never examine, so the census
// denominator is complete.
[[maybe_unused]] static void assignCensusIds(ModuleOp mod) {
  mod.walk([&](tt::FuncOp func) {
    std::string text;
    llvm::raw_string_ostream os(text);
    // Full IR, before any attribute is set: elided dense constants could
    // make two different functions hash the same.
    func->print(os);
    std::string funcKey =
        (func.getName() + "#" +
         llvm::utohexstr(llvm::xxh3_64bits(text), /*LowerCase=*/true))
            .str();
    unsigned loopIdx = 0;
    func.walk<WalkOrder::PreOrder>([&](scf::ForOp forOp) {
      std::string loopId = funcKey + "/L" + std::to_string(loopIdx++);
      forOp->setAttr(kCensusIdAttr,
                     StringAttr::get(forOp.getContext(), loopId));
      [[maybe_unused]] StringRef scope =
          forOp->getParentOfType<scf::ForOp>() ? "nested-loop"
          : !forOp.getSingleInductionVar()     ? "multi-iv"
                                               : "outermost";
      unsigned maskIdx = 0;
      forOp.getBody()->walk([&](Operation *op) {
        if (op->getParentOfType<scf::ForOp>() != forOp || !censusMask(op))
          return; // owned by an inner loop, or not masked
        std::string id = loopId + "/M" + std::to_string(maskIdx++);
        op->setAttr(kCensusIdAttr, StringAttr::get(op->getContext(), id));
        CDBG("candidate: id="
             << id << " kind=" << op->getName().getStringRef()
             << " scope=" << scope << " where="
             << (op->getBlock() == forOp.getBody() ? "direct" : "in-region"));
      });
    });
  });
}

[[maybe_unused]] static void stripCensusIds(ModuleOp mod) {
  mod.walk([](Operation *op) { op->removeAttr(kCensusIdAttr); });
}

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
  CDBG("outcome: id=" << censusId(op) << " result=dropped-"
                      << (maskVal ? "true" : "false"));
  TypeSwitch<Operation *>(op)
      .Case<tt::LoadOp>([&](auto loadOp) {
        if (maskVal) {
          auto newLoadOp = tt::LoadOp::create(
              builder, loc, loadOp.getPtr(), /*mask=*/Value(),
              /*other=*/Value(), loadOp.getCachePolicyAttr(),
              loadOp.getIsVolatile());
          loadOp->replaceAllUsesWith(newLoadOp);
        } else if (Value other = loadOp.getOther()) {
          loadOp->replaceAllUsesWith(ValueRange{other});
        } else if (TypedAttr zeroAttr = builder.getZeroAttr(loadOp.getType())) {
          // Note: `other` is optional on `tt.load`. Because the mask is false
          // no element is loaded, therefore the result of the load must be
          // assigned zero.
          auto zeroOp = arith::ConstantOp::create(builder, loc, zeroAttr);
          loadOp->replaceAllUsesWith(ValueRange{zeroOp});
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
    censusOp = op; // census only: classifyCmp has no access to the masked op
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

    // Verify there exists an iter_arg with init == lb, yield == self + step.
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
    std::optional<ConstantIntRanges> optRange =
        tt::intel::collectLoopIVRange(forOp, *solver);
    if (!optRange) {
      censusExit(forOp, "iv-range-unknown");
      return MaskClassification::Unknown;
    }

    if (!finalVal.getDefiningOp() ||
        !isa<arith::CmpIOp>(finalVal.getDefiningOp())) {
      censusExit(forOp, "not-cmpi");
      return MaskClassification::Unknown;
    }

    auto cmpOp = cast<arith::CmpIOp>(finalVal.getDefiningOp());
    arith::CmpIPredicate pred = cmpOp.getPredicate();
    if (!isSupportedBoundPredicate(pred)) {
      censusExit(forOp, "pred-unsupported");
      return MaskClassification::Unknown;
    }

    Value lhs = tt::intel::getFinalValue(cmpOp.getLhs());
    Value rhs = tt::intel::getFinalValue(cmpOp.getRhs());
    Operation *lhsOp = tt::intel::getFinalValue(lhs).getDefiningOp();
    Operation *rhsOp = tt::intel::getFinalValue(rhs).getDefiningOp();
    if (!lhsOp || !rhsOp) {
      censusExit(forOp, "no-defining-op");
      return MaskClassification::Unknown;
    }

    auto getIntConstantValue = [](Operation *op) -> std::optional<APInt> {
      APInt intVal;
      if (op->getNumResults() > 0 &&
          matchPattern(op->getResult(0), m_ConstantInt(&intVal)))
        return intVal;
      DenseElementsAttr constAttr;
      if (matchPattern(op, m_Constant(&constAttr)) && constAttr.isSplat()) {
        auto attr = constAttr.getSplatValue<Attribute>();
        if (auto intAttr = dyn_cast_or_null<IntegerAttr>(attr))
          return intAttr.getValue();
      }
      return std::nullopt;
    };

    // TODO: consider the case where the constant is lhs.
    std::optional<APInt> constIntVal = getIntConstantValue(rhsOp);
    if (!constIntVal) {
      censusExit(forOp, "rhs-not-const");
      return MaskClassification::Unknown;
    }

    auto addOp = dyn_cast<arith::AddIOp>(lhsOp);
    if (!addOp) {
      censusExit(forOp, "lhs-not-addi");
      return MaskClassification::Unknown;
    }

    Value addLhs = tt::intel::getFinalValue(addOp.getLhs());
    Value addRhs = tt::intel::getFinalValue(addOp.getRhs());

    std::optional<ConstantIntRanges> lhsRange =
        getIVEquivalentRange(forOp, addLhs);
    if (!lhsRange) {
      censusExit(forOp, "lhs-not-iv");
      return MaskClassification::Unknown;
    }

    auto makeRangeOp =
        dyn_cast_or_null<tt::MakeRangeOp>(addRhs.getDefiningOp());
    if (!makeRangeOp) {
      censusExit(forOp, "rhs-not-make-range");
      return MaskClassification::Unknown;
    }

    MaskClassification cls =
        classifyMask(pred, *lhsRange, makeRangeOp.getStart(),
                     makeRangeOp.getEnd(), *constIntVal);
    censusExit(forOp, "classified",
               cls == MaskClassification::AlwaysTrue    ? "true"
               : cls == MaskClassification::AlwaysFalse ? "false"
                                                        : "unknown");
    return cls;
  }

  // One census line per walk-1 classification exit (debug-only).
  void censusExit(scf::ForOp &forOp, StringRef exit,
                  StringRef result = StringRef()) const {
    CDBG("census: id=" << censusId(censusOp) << " walk=1 exit=" << exit
                       << (result.empty() ? StringRef() : StringRef(" result="))
                       << result << " ub="
                       << describeBound(forOp.getUpperBound()) << " step="
                       << (forOp.getConstantStep() ? "const" : "dyn"));
  }

  mutable Operation *censusOp = nullptr;
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
        isa<arith::ConstantIntOp>(defMulRhs)) {
      bool matched = cast<arith::ConstantIntOp>(defMulRhs).value() == end;
      if (matched && op)
        CDBG("census: id=" << censusId(op) << " walk=2 exit=canonical-matched");
      return matched;
    }

    if (!defMulRhs && mulOp.getRhs() == *loopIV &&
        isa<arith::ConstantIntOp>(defMulLhs)) {
      bool matched = cast<arith::ConstantIntOp>(defMulLhs).value() == end;
      if (matched && op)
        CDBG("census: id=" << censusId(op) << " walk=2 exit=canonical-matched");
      return matched;
    }

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
      auto nCstOp = maskInfo.N.getDefiningOp<arith::ConstantIntOp>();
      assert(nCstOp && "Expecting a constant `N` (ensured by "
                       "`hasCanonicalUpperBound`)");
      int64_t N = nCstOp.value();
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

  // Returns true if a versioning condition implying `mask` can be generated for
  // `forOp`. Contrary to `getVersioningCond` this does not modify the IR.
  bool canVersion(scf::ForOp &forOp, Value mask) const {
    Value finalVal = tt::intel::getFinalValue(mask);
    if (!finalVal || !isValidMask(forOp, finalVal, /*op=*/nullptr))
      return false;

    return hasCanonicalUpperBound(forOp, getMaskInfo(forOp, finalVal));
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
      // The constant upper bound can only be matched against the canonical form
      // when `N` is a constant too. Note that `isValidMask` accepts a mask
      // whose `N` has no defining operation (e.g. a kernel argument).
      auto nCstOp = maskInfo.N.getDefiningOp<arith::ConstantIntOp>();
      if (!nCstOp)
        return false;

      int64_t UB = cast<arith::ConstantIntOp>(defOp).value();
      int64_t N = nCstOp.value();
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

    // The versioning condition generated by `getVersioningCond` is derived from
    // the loop upper bound's `N` and `END`, therefore it implies the mask only
    // when the mask uses the same `N` and `END`.
    if (tt::intel::getFinalValue(maskInfo.N) !=
            tt::intel::getFinalValue(divNumOp.getLhs()) ||
        divDenOp.value() != maskInfo.END)
      return false;

    return true;
  }

  // Assuming the mask is equivalent to the form: `END < N-i*END`, returns a
  // structure containing `N` and `END`. Public so the versioning trace (Task
  // 11) can render the guard text without re-parsing the mask.
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
  //   - (splat(offset) + ext(make_range(0, END))) cmp dense<constant>
  //     (boundary checks from RewriteTensorDescriptorToPointer)
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

    bool isInLoop = (cmpOp->getParentOfType<scf::ForOp>() == forOp);
    if (isInLoop)
      return false;

    // Boundary-check pattern from RewriteTensorDescriptorToPointer:
    // (splat(offset) + ext(make_range(start, end))) cmp splat(constant)
    // Accepts all comparison predicates (including sge for >= 0 checks).
    if (isBoundaryCheckPattern(cmpOp)) {
      if (op)
        CDBG("census: id=" << censusId(op) << " walk=3 exit=invariant-matched");
      return true;
    }

    if (!isSupportedBoundPredicate(pred))
      return false;

    // The '>=' and '>' predicates are handled by isBoundaryCheckPattern above
    // but not by the legacy getVersioningCond paths below, which always use
    // END-1 as the scalar threshold (correct for '<' and '<=' only).
    if (pred == arith::CmpIPredicate::sge ||
        pred == arith::CmpIPredicate::sgt ||
        pred == arith::CmpIPredicate::uge || pred == arith::CmpIPredicate::ugt)
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
      if (op)
        CDBG("census: id=" << censusId(op) << " walk=3 exit=invariant-matched");
      return true;
    }

    if (!rhs && isa<tt::MakeRangeOp>(lhs)) {
      [[maybe_unused]] auto rangeOp = cast<tt::MakeRangeOp>(lhs);
      assert(rangeOp.getStart() < rangeOp.getEnd() && "Invalid range");
      if (op)
        CDBG("census: id=" << censusId(op) << " walk=3 exit=invariant-matched");
      return true;
    }

    if (!lhs && isa<tt::MakeRangeOp>(rhs)) {
      [[maybe_unused]] auto rangeOp = cast<tt::MakeRangeOp>(rhs);
      assert(rangeOp.getStart() < rangeOp.getEnd() && "Invalid range");
      if (op)
        CDBG("census: id=" << censusId(op) << " walk=3 exit=invariant-matched");
      return true;
    }

    return false;
  }

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

    // Boundary-check pattern: (splat(offset) + ext(make_range)) cmp constant
    // Generate scalar versioning condition.
    if (isBoundaryCheckPattern(cmpOp))
      return getBoundaryCheckVersioningCond(cmpOp, builder, loc);

    llvm_unreachable("Unexpected mask");
    return {};
  }

  virtual std::string getName() const { return "InvariantMaskValidator"; }

private:
  // Extract the scalar constant value from a uniform constant (dense splat or
  // tt.splat of scalar constant).
  static std::optional<APInt> getUniformConstantValue(Value val) {
    DenseElementsAttr attr;
    if (matchPattern(val, m_Constant(&attr)) && attr.isSplat()) {
      if (auto intAttr = dyn_cast<IntegerAttr>(attr.getSplatValue<Attribute>()))
        return intAttr.getValue();
      return std::nullopt;
    }
    if (auto splatOp = val.getDefiningOp<tt::SplatOp>()) {
      Value src = splatOp.getSrc();
      if (auto constOp = src.getDefiningOp<arith::ConstantIntOp>()) {
        unsigned bitWidth = src.getType().getIntOrFloatBitWidth();
        return APInt(bitWidth, constOp.value());
      }
      if (auto constOp = src.getDefiningOp<arith::ConstantOp>()) {
        if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
          return intAttr.getValue();
      }
    }
    return std::nullopt;
  }

  // Check if a cmpi is a boundary-check pattern:
  //   (splat(offset) + ext(make_range(start, end))) cmp uniform_constant
  // Also handles: expand_dims wrapping, commuted operand order.
  static bool isBoundaryCheckPattern(arith::CmpIOp cmpOp) {
    // `getBoundaryCheckVersioningCond` can only generate a correct scalar
    // condition for an ordered bound-check predicate: an `eq`/`ne` mask is
    // satisfied by a single lane, which no scalar condition on the offset
    // implies. Check the predicate here so that both callers are covered.
    if (!isSupportedBoundPredicate(cmpOp.getPredicate()))
      return false;

    // RHS must be a uniform constant (dense splat or tt.splat of scalar).
    Value rhs = cmpOp.getRhs();
    if (!getUniformConstantValue(rhs))
      return false;

    // LHS must be an addi containing a splat and an ext(make_range).
    Value lhs = cmpOp.getLhs();
    // Peel through expand_dims.
    while (auto expandOp = lhs.getDefiningOp<tt::ExpandDimsOp>())
      lhs = expandOp.getSrc();

    auto addOp = lhs.getDefiningOp<arith::AddIOp>();
    if (!addOp)
      return false;

    // One operand should be a splat (the offset), the other should contain
    // a make_range (possibly through extsi).
    auto hasSplat = [](Value v) { return v.getDefiningOp<tt::SplatOp>(); };
    auto hasMakeRange = [](Value v) {
      // Peel extsi.
      if (auto extOp = v.getDefiningOp<arith::ExtSIOp>())
        v = extOp.getIn();
      return v.getDefiningOp<tt::MakeRangeOp>();
    };

    return (hasSplat(addOp.getLhs()) && hasMakeRange(addOp.getRhs())) ||
           (hasSplat(addOp.getRhs()) && hasMakeRange(addOp.getLhs()));
  }

  // Generate scalar versioning condition for a boundary-check pattern.
  // For (splat(offset) + ext(make_range(start, end))) < dense<bound>:
  //   condition = offset + (end - 1) < bound   (all elements in range satisfy)
  // For (splat(offset) + ext(make_range(start, end))) >= dense<bound>:
  //   condition = offset + start >= bound       (min element satisfies)
  static Value getBoundaryCheckVersioningCond(arith::CmpIOp cmpOp,
                                              OpBuilder &builder,
                                              Location loc) {
    arith::CmpIPredicate pred = cmpOp.getPredicate();

    // Extract the constant bound from RHS.
    std::optional<APInt> optBoundVal = getUniformConstantValue(cmpOp.getRhs());
    assert(optBoundVal && "Expected uniform constant RHS");
    APInt boundVal = *optBoundVal;

    // Extract offset scalar and make_range from LHS.
    Value lhs = cmpOp.getLhs();
    while (auto expandOp = lhs.getDefiningOp<tt::ExpandDimsOp>())
      lhs = expandOp.getSrc();
    auto addOp = cast<arith::AddIOp>(lhs.getDefiningOp());

    Value offsetVal;
    tt::MakeRangeOp rangeOp;
    if (addOp.getLhs().getDefiningOp<tt::SplatOp>()) {
      offsetVal = addOp.getLhs().getDefiningOp<tt::SplatOp>().getSrc();
      Value rangeV = addOp.getRhs();
      if (auto extOp = rangeV.getDefiningOp<arith::ExtSIOp>())
        rangeV = extOp.getIn();
      rangeOp = cast<tt::MakeRangeOp>(rangeV.getDefiningOp());
    } else {
      offsetVal = addOp.getRhs().getDefiningOp<tt::SplatOp>().getSrc();
      Value rangeV = addOp.getLhs();
      if (auto extOp = rangeV.getDefiningOp<arith::ExtSIOp>())
        rangeV = extOp.getIn();
      rangeOp = cast<tt::MakeRangeOp>(rangeV.getDefiningOp());
    }

    // For < / <= predicates: use max element = offset + (end - 1)
    //   "all elements < bound" iff "offset + (end-1) < bound"
    // For >= / > predicates: use min element = offset + start
    //   "all elements >= bound" iff "offset + start >= bound"
    unsigned bitWidth = boundVal.getBitWidth();
    int64_t rangeAdjust;
    if (pred == arith::CmpIPredicate::slt ||
        pred == arith::CmpIPredicate::ult ||
        pred == arith::CmpIPredicate::sle || pred == arith::CmpIPredicate::ule)
      rangeAdjust = rangeOp.getEnd() - 1;
    else
      rangeAdjust = rangeOp.getStart(); // sge, sgt, uge, ugt

    // Build: offset + rangeAdjust
    Type scalarTy = builder.getIntegerType(bitWidth);
    assert(offsetVal.getType().getIntOrFloatBitWidth() == bitWidth &&
           "offset and bound widths must match for a valid arith.cmpi");

    Value lhsScalar = offsetVal;
    if (rangeAdjust != 0) {
      Value adjustVal =
          arith::ConstantIntOp::create(builder, loc, scalarTy, rangeAdjust);
      lhsScalar = arith::AddIOp::create(builder, loc, offsetVal, adjustVal);
    }

    Value boundConst = arith::ConstantIntOp::create(builder, loc, scalarTy,
                                                    boundVal.getSExtValue());
    return arith::CmpIOp::create(builder, loc, pred, lhsScalar, boundConst);
  }
};

// A mask validator backed by the symbolic bounds prover. Unlike the validators
// above it recognizes no particular mask shape: it accepts anything the prover
// could conceivably decide and leaves the decision to `proofFor`.
class SymbolicMaskValidator final : public MaskValidatorBase {
public:
  SymbolicMaskValidator(tt::intel::SymbolicBoundsProver &prover)
      : prover(prover) {}

  // Structural pre-filter only, deliberately without proving anything: the
  // collector calls this loads-first and then selects, not in program order,
  // and the Global Constraints require proofs in program order.
  bool isValidMask(scf::ForOp &forOp, Value mask,
                   Operation *op) const override {
    // Mirror the look-through set of `proveTrue` and require it to bottom out
    // in something that has a chance of being decided.
    Value v = mask;
    while (Operation *def = v.getDefiningOp()) {
      if (auto splat = dyn_cast<tt::SplatOp>(def))
        v = splat.getSrc();
      else if (auto expand = dyn_cast<tt::ExpandDimsOp>(def))
        v = expand.getSrc();
      else if (auto bcast = dyn_cast<tt::BroadcastOp>(def))
        v = bcast.getSrc();
      else if (auto ext = dyn_cast<arith::ExtSIOp>(def))
        v = ext.getIn();
      else if (auto ext = dyn_cast<arith::ExtUIOp>(def))
        v = ext.getIn();
      else
        return isa<arith::CmpIOp, arith::AndIOp, arith::ConstantOp>(def);
    }
    return false; // a block argument: `proveTrue` stops there
  }

  // Unused: a symbolic guard comes from the proof's conditions, which the
  // driver materializes once per loop, not from one mask at a time.
  Value getVersioningCond(scf::ForOp &, Value) const override {
    return nullptr;
  }

  std::string getName() const override { return "SymbolicMaskValidator"; }

  // Cached per operation. Operations sharing a mask are still queried
  // separately: each query carries its own program point.
  tt::intel::BoundProof proofFor(Operation *op) const {
    auto it = proofs.find(op);
    if (it != proofs.end())
      return it->second;
    Value mask = censusMask(op);
    tt::intel::BoundProof proof;
    if (mask)
      proof = prover.proveTrue(mask, {op, op->getParentOfType<scf::ForOp>()});
    CDBG("verdict: id=" << censusId(op)
                        << " proof=" << tt::intel::toString(proof));
    proofs.try_emplace(op, proof);
    return proof;
  }

private:
  tt::intel::SymbolicBoundsProver &prover;
  mutable DenseMap<Operation *, tt::intel::BoundProof> proofs;
};

// What the analysis phase decided for one loop, consumed by the mutation
// phases. Held across loops, so it must name no value the mutation of an
// earlier loop could have erased.
// The inline sizes are explicit because these elements embed BoundProof and
// BoundCondition, whose own inline buffers push sizeof past the 256-byte limit
// LLVM asserts on when SmallVector has to pick a default inline count.
struct LoopPlan {
  scf::ForOp loop;
  // Program order. Satisfied | Refuted | ConditionallySatisfied only.
  SmallVector<std::pair<Operation *, tt::intel::BoundProof>, 4> ops;
  // Deduplicated guard conditions, facts before preconditions before guards.
  SmallVector<tt::intel::BoundCondition, 4> conds;
  Value guard; // materialized before any unmasking
};

// ','-joined census ids, in the order given.
[[maybe_unused]] static std::string joinIds(ArrayRef<Operation *> ops) {
  SmallVector<std::string> ids;
  for (Operation *op : ops)
    ids.push_back(censusId(op).str());
  return llvm::join(ids, ",");
}

// ';'-joined conditions, in the order given.
[[maybe_unused]] static std::string
joinConditions(ArrayRef<tt::intel::BoundCondition> cs) {
  SmallVector<std::string> strs;
  for (const tt::intel::BoundCondition &c : cs)
    strs.push_back(tt::intel::toString(c));
  return llvm::join(strs, ";");
}

// \p ops printed one per entry: in program order if they share a block, else
// sorted by text. Program order is only defined within a block, and a
// comparator mixing the two orders is not a strict weak ordering.
[[maybe_unused]] static SmallVector<std::string>
printInTraceOrder(ArrayRef<Operation *> ops) {
  SmallVector<Operation *> sorted(ops);
  bool sameBlock = llvm::all_of(sorted, [&](Operation *op) {
    return op->getBlock() == sorted.front()->getBlock();
  });
  if (sameBlock)
    llvm::sort(sorted, [](Operation *a, Operation *b) {
      return a->isBeforeInBlock(b);
    });
  SmallVector<std::string> texts;
  for (Operation *op : sorted) {
    std::string text;
    llvm::raw_string_ostream os(text);
    op->print(os, OpPrintingFlags().skipRegions());
    texts.push_back(text);
  }
  if (!sameBlock)
    llvm::sort(texts);
  return texts;
}

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
    // `LoopVersioner::version` (the consumer of `CanonicalMaskValidator` and
    // `InvariantMaskValidator`) only knows how to drop masks from `tt.load`;
    // `RemovableMaskValidator` and `SymbolicMaskValidator` consume their ops
    // via `dropMask`, which handles both op kinds, so they also need
    // `arith.select` collected.
    if constexpr (std::is_same_v<MaskValidator, RemovableMaskValidator> ||
                  std::is_same_v<MaskValidator, SymbolicMaskValidator>)
      collectMaskedOps(forOp.getOps<arith::SelectOp>(), maskedOps);
    return maskedOps.size();
  }

  const MaskedOperations &getMaskedOps() const { return maskedOps; };
  const MaskValidator &getMaskValidator() const { return maskValidator; }

  Value getMask(Operation *op) const {
    assert(op && "Expecting a valid operation");
    return TypeSwitch<Operation *, Value>(op)
        .Case<tt::LoadOp>([](auto loadOp) { return loadOp.getMask(); })
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
      assert(isa<tt::LoadOp>(maskedOp) && "Expecting a load operation");
      return tt::intel::getFinalValue(cast<tt::LoadOp>(maskedOp).getMask());
    };

    // The versioning condition is used to drop the mask of *every* collected
    // operation in the "then" region, therefore it must imply all of them. Bail
    // out unless all the collected masks are versionable.
    const CanonicalMaskValidator &maskValidator = collector.getMaskValidator();
    for (Operation *maskedOp : collector.getMaskedOps()) {
      Value mask = collector.getMask(maskedOp);
      if (!mask || !maskValidator.canVersion(forOp, mask))
        return false;
    }

    // Retrieve the versioning condition, bail out if it doesn't exist (in
    // which case the loop upper bound is not in canonical form).
    // Note that a single condition is sufficient: `hasCanonicalUpperBound`
    // ensures every versionable mask shares the loop upper bound's `N` and
    // `END`, therefore the conditions generated for the collected masks are all
    // identical.
    Operation *maskedOp = *collector.getMaskedOps().begin();
    Value verCond = maskValidator.getVersioningCond(forOp, getMask(maskedOp));
    if (!verCond)
      return false;

    DEBUG_WITH_TYPE(CENSUS_DEBUG_TYPE, {
      SmallVector<Operation *> toUnmaskTrace(collector.getMaskedOps().begin(),
                                             collector.getMaskedOps().end());
      llvm::sort(toUnmaskTrace, [](Operation *a, Operation *b) {
        return a->isBeforeInBlock(b);
      });
      CanonicalMaskValidator::MaskInfo info =
          maskValidator.getMaskInfo(forOp, getMask(maskedOp));
      CDBG("versioned: loop="
           << censusId(forOp) << " unmasked=" << joinIds(toUnmaskTrace)
           << " guard=canonical N=" << describeArg(info.N) << " END="
           << info.END << " w=" << info.N.getType().getIntOrFloatBitWidth());
    });

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

    // Create the yield operations for the two if branches. Note that when the
    // 'scf.if' yields no result its regions already contain an implicit
    // terminator (see `scf::IfOp::getThenBodyBuilder`), in which case no
    // explicit yield operation must be created.
    if (ifOp.getNumResults() != 0) {
      scf::YieldOp::create(thenB, loc, pruneUnusedResults(forOp, thenForLoop));
      scf::YieldOp::create(elseB, loc, pruneUnusedResults(forOp, elseForLoop));
    }

    // Drop the mask from candidate masked operations in the "then" region.
    for (Operation *maskedOp : collector.getMaskedOps()) {
      Operation *mappedOp = map.lookup(maskedOp);
      if (auto loadOp = dyn_cast<tt::LoadOp>(mappedOp)) {
        OpBuilder builder(mappedOp);
        auto newLoad = tt::LoadOp::create(
            builder, loadOp.getLoc(), loadOp.getPtr(), /*mask=*/Value(),
            /*other=*/Value(), loadOp.getCachePolicyAttr(),
            loadOp.getIsVolatile());
        mappedOp->replaceAllUsesWith(newLoad);
        mappedOp->erase();
      }
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
      auto loadOp = cast<tt::LoadOp>(maskedOp);
      maskConds.insert(
          tt::intel::getFinalValue(loadOp.getMask()).getDefiningOp());
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

    DEBUG_WITH_TYPE(CENSUS_DEBUG_TYPE, {
      // `maskConds` is a SmallPtrSet with no stable order.
      SmallVector<Operation *> condsTrace(maskConds.begin(), maskConds.end());
      SmallVector<Operation *> toUnmaskTrace(collector.getMaskedOps().begin(),
                                             collector.getMaskedOps().end());
      llvm::sort(toUnmaskTrace, [](Operation *a, Operation *b) {
        return a->isBeforeInBlock(b);
      });
      CDBG("versioned: loop="
           << censusId(forOp) << " unmasked=" << joinIds(toUnmaskTrace)
           << " guard=" << llvm::join(printInTraceOrder(condsTrace), ";"));
    });

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
      auto loadOp = cast<tt::LoadOp>(map.lookup(maskedOp));
      OpBuilder builder(loadOp);
      auto newLoad = tt::LoadOp::create(
          builder, loadOp.getLoc(), loadOp.getPtr(), /*mask=*/Value(),
          /*other=*/Value(), loadOp.getCachePolicyAttr(),
          loadOp.getIsVolatile());
      loadOp->replaceAllUsesWith(newLoad);
      loadOp->erase();
    }

    // Replace the uses of the original loop results.
    for (const auto &[i, v] : llvm::enumerate(forOp.getResults()))
      if (!v.getUsers().empty())
        v.replaceAllUsesWith(ifOp->getResult(i));

    forOp.erase();
    return true;
  }
};

// Versions \p forOp on \p guard and drops the mask of \p opsToUnmask in the
// "then" copy. This is `LoopVersioner::version`'s canonical overload with the
// condition supplied rather than derived, and with the unmasking restricted to
// the operations the prover decided conditionally: the other masked operations
// in the loop must keep their masks in both copies.
static void versionWithGuard(scf::ForOp forOp, Value guard,
                             ArrayRef<Operation *> opsToUnmask) {
  assert(guard && "Expecting a valid versioning condition");

  auto getUsedResults = [](const scf::ForOp &forOp) {
    SmallVector<Type> resTypes;
    for (Value res : forOp->getResults()) {
      if (!res.getUsers().empty())
        resTypes.push_back(res.getType());
    }
    return resTypes;
  };

  OpBuilder builder(forOp);
  Location loc = forOp.getLoc();
  auto ifOp = scf::IfOp::create(builder, loc, getUsedResults(forOp), guard,
                                /*withThenRegion=*/true);

  // Clone the original loop into the 2 if branches.
  IRMapping map;
  OpBuilder thenB = ifOp.getThenBodyBuilder();
  Operation *thenForLoop = thenB.clone(*forOp.getOperation(), map);
  OpBuilder elseB = ifOp.getElseBodyBuilder();
  Operation *elseForLoop = elseB.clone(*forOp.getOperation());

  auto pruneUnusedResults = [&](const scf::ForOp &forOp,
                                Operation *clonedLoop) {
    SmallVector<Value> prunedResults;
    for (auto [idx, val] : llvm::enumerate(forOp->getResults())) {
      if (!val.getUsers().empty())
        prunedResults.push_back(clonedLoop->getResult(idx));
    }
    return prunedResults;
  };

  // When the 'scf.if' yields no result its regions already contain an implicit
  // terminator, in which case no explicit yield must be created.
  if (ifOp.getNumResults() != 0) {
    scf::YieldOp::create(thenB, loc, pruneUnusedResults(forOp, thenForLoop));
    scf::YieldOp::create(elseB, loc, pruneUnusedResults(forOp, elseForLoop));
  }

  // Unmask the clones in the "then" region. `dropMask` replaces the uses but
  // leaves the original in place; erase it, because canonicalization drops a
  // dead non-volatile load yet a volatile one also has a Write effect and
  // would survive and still execute.
  for (Operation *op : opsToUnmask) {
    Operation *mappedOp = map.lookup(op);
    if (!mappedOp)
      continue;
    dropMask(mappedOp, /*maskVal=*/true);
    if (mappedOp->use_empty())
      mappedOp->erase();
  }

  // Replace the uses of the original loop results.
  unsigned idx = 0;
  for (Value res : forOp.getResults()) {
    if (!res.getUsers().empty())
      res.replaceAllUsesWith(ifOp->getResult(idx++));
  }

  forOp.erase();
}

struct TritonIntelRemoveMasksBase
    : tt::intel::impl::TritonIntelRemoveMasksBase<TritonIntelRemoveMasksBase> {
public:
  using Base::Base;
  using IndexMapSet = std::map<int, std::set<int>>;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    // Census scaffolding: assign stable per-mask ids before anything
    // examines or mutates the IR, and strip them at pass end. Setting a
    // discardable attribute creates and erases no values, so this does not
    // affect analysis-state validity.
    DEBUG_WITH_TYPE(CENSUS_DEBUG_TYPE, assignCensusIds(moduleOp));

    if (tt::tools::getBoolEnv("TRITON_INTEL_SYMBOLIC_MASKS"))
      runSymbolic(moduleOp);
    else
      runLegacy(moduleOp);

    LLVM_DEBUG(llvm::dbgs() << "After versioning:\n" << moduleOp << "\n");
    DEBUG_WITH_TYPE(CENSUS_DEBUG_TYPE, stripCensusIds(moduleOp));
    assert(succeeded(verify(moduleOp)) && "Module verification failed");
  }

  // The symbolic driver: one read-only analysis phase over the whole module,
  // then the mutations. Nothing between the solver and the end of the walk
  // below touches the IR, which is what keeps the analysis state valid for
  // every proof.
  void runSymbolic(ModuleOp moduleOp) {
    std::shared_ptr<DataFlowSolver> solver = createDataFlowSolver();
    solver->load<tt::intel::IntegerRangeAnalysis>(moduleOp,
                                                  getAnalysis<DominanceInfo>());
    if (failed(solver->initializeAndRun(moduleOp)))
      return signalPassFailure();

    tt::intel::SymbolicBoundsProver prover(
        *solver, getAnalysis<DominanceInfo>(), moduleOp);
    SymbolicMaskValidator validator(prover);

    SmallVector<LoopPlan, 2> plans;
    moduleOp->walk<WalkOrder::PreOrder>([&](scf::ForOp forOp) {
      // Outermost single-induction-variable loops only.
      if (forOp->getParentOfType<scf::ForOp>() ||
          !forOp.getSingleInductionVar())
        return;
      MaskedOpsCollector<SymbolicMaskValidator> collector(forOp, validator);
      if (!collector.collectMaskedOps())
        return;

      LoopPlan plan{forOp};
      // The collector's SmallPtrSet has no stable order; the collected ops are
      // direct children of the loop body, so isBeforeInBlock sorts them.
      SmallVector<Operation *> ops(collector.getMaskedOps().begin(),
                                   collector.getMaskedOps().end());
      llvm::sort(ops, [](Operation *a, Operation *b) {
        return a->isBeforeInBlock(b);
      });
      for (Operation *op : ops) {
        tt::intel::BoundProof proof = validator.proofFor(op);
        if (proof.verdict != tt::intel::BoundProof::Unknown)
          plan.ops.emplace_back(op, std::move(proof));
      }
      plans.push_back(std::move(plan));
    });

    // Materialize every guard before any other mutation, so no guard
    // refers to a value a later unmasking erased.
    using Prover = tt::intel::SymbolicBoundsProver;
    for (LoopPlan &plan : plans) {
      for (auto &[op, proof] : plan.ops)
        if (proof.verdict == tt::intel::BoundProof::ConditionallySatisfied)
          for (const tt::intel::BoundCondition &c : proof.conditions) {
            // BoundCondition::operator== is structural, never by loc name.
            auto it = llvm::find(plan.conds, c);
            if (it == plan.conds.end())
              plan.conds.push_back(c);
            else if (c.kind < it->kind)
              it->kind = c.kind; // strongest kind wins, whatever the op order
          }
      // Facts, then preconditions, then guards.
      llvm::stable_sort(plan.conds, [](const tt::intel::BoundCondition &a,
                                       const tt::intel::BoundCondition &b) {
        return a.kind < b.kind;
      });
      auto count = [&](tt::intel::ConditionKind k) {
        return llvm::count_if(
            plan.conds,
            [&](const tt::intel::BoundCondition &c) { return c.kind == k; });
      };
      // The budgets bound one proof; this is their union over the loop.
      if (plan.conds.empty() ||
          count(tt::intel::ConditionKind::Fact) > Prover::kMaxFactConditions ||
          count(tt::intel::ConditionKind::Guard) > Prover::kMaxGuards)
        continue;
      OpBuilder builder(plan.loop);
      plan.guard = tt::intel::materialize(plan.conds, plan.loop, builder);
    }

    // Then, per loop in reverse program order: drop the unconditional
    // masks first, so both clones inherit them, then version.
    for (LoopPlan &plan : llvm::reverse(plans)) {
      for (auto &[op, proof] : plan.ops) {
        bool sat = proof.verdict == tt::intel::BoundProof::Satisfied;
        if (!sat && proof.verdict != tt::intel::BoundProof::Refuted)
          continue;
        dropMask(op, sat);
        // `dropMask` does not always replace: for a false load mask with no
        // `other` it falls through when getZeroAttr() gives no attribute for
        // the result type, leaving the op live. Erase only once nothing uses
        // it, so a volatile load does not survive canonicalization and
        // execute, and an unreplaced one is left alone rather than erased
        // while still in use.
        if (op->use_empty())
          op->erase();
      }
      if (!plan.guard)
        continue;
      SmallVector<Operation *> toUnmask; // program order, for joinIds
      for (auto &[op, proof] : plan.ops)
        if (proof.verdict == tt::intel::BoundProof::ConditionallySatisfied)
          toUnmask.push_back(op);
      CDBG("versioned: loop=" << censusId(plan.loop)
                              << " unmasked=" << joinIds(toUnmask)
                              << " guard=" << joinConditions(plan.conds));
      versionWithGuard(plan.loop, plan.guard, toUnmask);
    }
  }

  void runLegacy(ModuleOp moduleOp) {
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
          bool loopVersioned = LoopVersioner::version(forOp, collector);
          LLVM_DEBUG(if (loopVersioned) llvm::dbgs() << "Loop versioned\n");
          // The loop has been erased by the versioner, therefore its regions
          // must not be walked.
          if (loopVersioned)
            return WalkResult::skip();
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
          bool loopVersioned = LoopVersioner::version(forOp, collector);
          LLVM_DEBUG(if (loopVersioned) llvm::dbgs() << "Loop versioned\n");
          // The loop has been erased by the versioner, therefore its regions
          // must not be walked.
          if (loopVersioned)
            return WalkResult::skip();
        }
      }
      return WalkResult::advance();
    });
  }
};

} // namespace
