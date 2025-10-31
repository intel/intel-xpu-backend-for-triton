#include "intel/include/Dialect/Triton/Transforms/Passes.h"
#include "intel/include/Utils/Utility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Support/LLVM.h"
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

static Operation *dropMask(Operation *op, bool maskVal) {
  assert(op && "Expecting a valid operation");

  OpBuilder builder(op);
  Location loc = op->getLoc();
  TypeSwitch<Operation *>(op)
      .Case<tt::LoadOp>([&](auto loadOp) {
        if (maskVal) {
          auto newLoadOp = builder.create<tt::LoadOp>(
              loc, loadOp.getPtr(), loadOp.getBoundaryCheck(),
              loadOp.getPadding(), loadOp.getCache(), loadOp.getEvict(),
              loadOp.getIsVolatile());
          loadOp->replaceAllUsesWith(newLoadOp);
        } else {
          Operation *cstOp =
              builder.create<arith::ConstantOp>(loc, loadOp.getOther());
          loadOp->replaceAllUsesWith(cstOp);
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
  virtual bool isValidMask(scf::ForOp &forOp, Value mask) const = 0;

  // Create the loop versioning condition based on the mask.
  virtual Value getVersioningCond(scf::ForOp &forOp, Value mask) const = 0;

  virtual std::string getName() const = 0;
};

// A mask validator which ensures the mask is not necessary.
class RemovableMaskValidator final : public MaskValidatorBase {
public:
  virtual bool isValidMask(scf::ForOp &forOp, Value mask) const {
    Value finalVal = tt::intel::getFinalValue(mask);
    assert(finalVal && "Expecting a valid mask");

    // Ensure the loop range is known.
    std::optional<ConstantIntRanges> optRange = computeLoopIVRange(forOp);
    if (!optRange)
      return false;

    if (!finalVal.getDefiningOp() ||
        !isa<arith::CmpIOp>(finalVal.getDefiningOp()))
      return false;

    auto cmpOp = cast<arith::CmpIOp>(finalVal.getDefiningOp());
    arith::CmpIPredicate pred = cmpOp.getPredicate();
    if (pred != arith::CmpIPredicate::slt)
      return false;

    Value lhs = tt::intel::getFinalValue(cmpOp.getLhs());
    Value rhs = tt::intel::getFinalValue(cmpOp.getRhs());
    Operation *lhsOp = tt::intel::getFinalValue(lhs).getDefiningOp();
    Operation *rhsOp = tt::intel::getFinalValue(rhs).getDefiningOp();
    if (!lhsOp || !rhsOp)
      return false;

    auto getIntConstantValue = [](Operation *op) -> std::optional<APInt> {
      DenseElementsAttr constAttr;
      if (matchPattern(op, m_Constant(&constAttr))) {
        auto attr = constAttr.getSplatValue<Attribute>();
        if (auto intAttr = dyn_cast_or_null<IntegerAttr>(attr))
          return intAttr.getValue();
      }
      return std::nullopt;
    };

    // TODO: consider the case where the constant is lhs.
    std::optional<APInt> constIntVal = getIntConstantValue(rhsOp);
    if (!constIntVal)
      return false;

    if (auto addOp = dyn_cast<arith::AddIOp>(lhsOp)) {
      Value lhs = tt::intel::getFinalValue(addOp.getLhs());
      Value rhs = tt::intel::getFinalValue(addOp.getRhs());
      if (lhs != forOp.getSingleInductionVar())
        return false;
      if (auto makeRangeOp = dyn_cast<tt::MakeRangeOp>(rhs.getDefiningOp())) {
        APInt maxIV = (*optRange).smax();
        int64_t maxVal = maxIV.getSExtValue() + makeRangeOp.getEnd();

        auto registerMaskValue = [this](Value mask, bool maskVal) {
          for (Operation *user : mask.getUsers())
            opToMaskValue.insert({user, maskVal});
        };

        if (maxVal <= constIntVal->getSExtValue()) {
          registerMaskValue(mask, true);
          return true;
        }
        APInt minIV = (*optRange).smin();
        int64_t minVal = minIV.getSExtValue() + makeRangeOp.getStart();
        if (minVal >= constIntVal->getSExtValue()) {
          registerMaskValue(mask, false);
          return true;
        }
      }
    }

    return false;
  }

  virtual Value getVersioningCond(scf::ForOp &forOp, Value mask) const {
    return {};
  }

  virtual std::string getName() const { return "RemovableMaskValidator"; }

  bool getMaskValue(Operation *op) const {
    assert(opToMaskValue.find(op) != opToMaskValue.end() && "mask not present");
    return opToMaskValue[op];
  }

private:
  mutable std::map<Operation *, bool> opToMaskValue;

  std::optional<ConstantIntRanges> computeLoopIVRange(scf::ForOp forOp) const {
    if (!forOp.getSingleInductionVar())
      return std::nullopt;

    if (std::optional<APInt> tripCount = forOp.getStaticTripCount()) {
      OpFoldResult lb = *forOp.getSingleLowerBound();
      OpFoldResult step = *forOp.getSingleStep();
      int64_t lbVal = *getConstantIntValue(lb);
      int64_t stepVal = *getConstantIntValue(step);
      int64_t lastIVVal = stepVal * (tripCount->getSExtValue() - 1);
      llvm::APInt start(64, lbVal, true);
      llvm::APInt end(64, lastIVVal, true);
      return ConstantIntRanges::range(start, end, true);
    }

    return std::nullopt;
  }
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
  virtual bool isValidMask(scf::ForOp &forOp, Value mask) const {
    Value finalVal = tt::intel::getFinalValue(mask);
    assert(finalVal && "Expecting a valid mask");

    if (!finalVal.getDefiningOp() ||
        !isa<arith::CmpIOp>(finalVal.getDefiningOp()))
      return false;

    auto cmpOp = cast<arith::CmpIOp>(finalVal.getDefiningOp());
    arith::CmpIPredicate pred = cmpOp.getPredicate();
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
      return builder.create<arith::ConstantIntOp>(forOp.getLoc(),
                                                  builder.getI1Type(), cond);
    }

    auto divOp = cast<arith::DivSIOp>(defOp);
    Operation *divLhsOp = divOp.getLhs().getDefiningOp();
    auto divNumOp = cast<arith::AddIOp>(divLhsOp);
    Value lhs = divNumOp.getLhs();
    Value rhs = divOp.getRhs();

    Value zero = tt::intel::findOrCreateIntConstant(
        loc, 0, lhs.getType().getIntOrFloatBitWidth(), builder);
    Value cmp1 = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq,
        builder.create<arith::RemSIOp>(loc, lhs, rhs), zero);
    Value cmp2 =
        builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, lhs, rhs);
    return builder.create<arith::AndIOp>(loc, cmp1, cmp2);
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
    assert(isValidMask(forOp, mask) && "Expecting a valid mask");

    auto cmpOp = cast<arith::CmpIOp>(mask.getDefiningOp());
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
  virtual bool isValidMask(scf::ForOp &forOp, Value mask) const {
    Value finalVal = tt::intel::getFinalValue(mask);
    assert(finalVal && "Expecting a valid mask");

    if (!finalVal.getDefiningOp() ||
        !isa<arith::CmpIOp>(finalVal.getDefiningOp()))
      return false;

    auto cmpOp = cast<arith::CmpIOp>(finalVal.getDefiningOp());
    arith::CmpIPredicate pred = cmpOp.getPredicate();
    if (pred != arith::CmpIPredicate::slt)
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
    assert(isValidMask(forOp, mask) && "Invalid mask");

    OpBuilder builder(forOp);
    Location loc = forOp.getLoc();
    auto cmpOp = cast<arith::CmpIOp>(mask.getDefiningOp());
    Value lhsVal = tt::intel::getFinalValue(cmpOp.getLhs());
    Value rhsVal = tt::intel::getFinalValue(cmpOp.getRhs());
    Operation *lhs = tt::intel::getFinalValue(lhsVal).getDefiningOp();
    Operation *rhs = tt::intel::getFinalValue(rhsVal).getDefiningOp();

    // N < M (with i1 data type)
    if (!lhs && !rhs)
      return builder.createOrFold<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                                 lhsVal, rhsVal);

    // [0..END] < splat(N) -- generate versioning condition 'END-1 < N'.
    if (!rhs && isa<tt::MakeRangeOp>(lhs)) {
      [[maybe_unused]] auto rangeOp = cast<tt::MakeRangeOp>(lhs);
      assert(rangeOp.getStart() < rangeOp.getEnd() && "Invalid range");
      unsigned end = rangeOp.getEnd() - 1u;
      auto cstOp = tt::intel::findOrCreateIntConstant(
          loc, end, rhsVal.getType().getIntOrFloatBitWidth(), builder);
      return builder.createOrFold<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                                 cstOp, rhsVal);
    }

    // splat(N) < [0..END] -- generate versioning condition 'N < END'.
    if (!lhs && isa<tt::MakeRangeOp>(rhs)) {
      [[maybe_unused]] auto rangeOp = cast<tt::MakeRangeOp>(rhs);
      assert(rangeOp.getStart() < rangeOp.getEnd() && "Invalid range");
      unsigned start = rangeOp.getStart();
      auto cstOp = builder.createOrFold<arith::ConstantIntOp>(
          loc, lhsVal.getType(), start);
      return builder.createOrFold<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                                 lhsVal, cstOp);
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
        if (mask && maskValidator.isValidMask(forOp, mask)) {
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
    // Currently we can version the loop only if it doesn't have downward
    // exposed uses of return values that are a tensor of pointers.
    // Note: this is due to the fact the results yielded by the 2 versioning
    // branches have different types for ptr (only in one versioned loop
    // tensor of ptrs are changed to block ptrs) 'then' part of the versioning
    // branch and leave them as is in the 'else' branch).
    auto canVersion = [](scf::ForOp &forOp) {
      return llvm::any_of(forOp.getResults(), [](Value res) {
        return !tt::isTensorPointerType(res.getType()) ||
               res.getUsers().empty();
      });
    };
    if (!canVersion(forOp))
      return false;

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
    auto ifOp = builder.create<scf::IfOp>(loc, getUsedResults(forOp), verCond,
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
    thenB.create<scf::YieldOp>(loc, pruneUnusedResults(forOp, thenForLoop));
    elseB.create<scf::YieldOp>(loc, pruneUnusedResults(forOp, elseForLoop));

    // Drop the mask from candidate masked operations in the "then" region.
    for (Operation *maskedOp : collector.getMaskedOps()) {
      Operation *mappedOp = map.lookup(maskedOp);
      if (auto loadOp = dyn_cast<tt::LoadOp>(mappedOp)) {
        OpBuilder builder(mappedOp);
        auto newLoad = builder.create<tt::LoadOp>(
            loadOp.getLoc(), loadOp.getPtr(), loadOp.getCache(),
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

    // Collect the (loop invariant) mask conditions.
    SmallPtrSet<Operation *, 8> maskConds;
    for (Operation *maskedOp : collector.getMaskedOps()) {
      if (auto loadOp = dyn_cast<tt::LoadOp>(maskedOp))
        maskConds.insert(loadOp.getMask().getDefiningOp());
      if (auto storeOp = dyn_cast<tt::StoreOp>(maskedOp))
        maskConds.insert(storeOp.getMask().getDefiningOp());
    }

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
      verCond = builder.create<arith::AndIOp>(loc, verCond, cond);
    }

    auto ifOp = builder.create<scf::IfOp>(loc, forOp.getResultTypes(), verCond,
                                          /*withThenRegion=*/true);

    // Clone the original loop into the 2 if branches.
    IRMapping map;
    OpBuilder thenB = ifOp.getThenBodyBuilder();
    Operation *thenForLoop = thenB.clone(*forOp.getOperation(), map);
    OpBuilder elseB = ifOp.getElseBodyBuilder();
    Operation *elseForLoop = elseB.clone(*forOp.getOperation());

    // Create the yield operations for the two if branches.
    if (!thenForLoop->getResults().empty()) {
      thenB.create<scf::YieldOp>(loc, thenForLoop->getResults());
      elseB.create<scf::YieldOp>(loc, elseForLoop->getResults());
    }

    // Drop the mask from candidate masked operations in the "then" region's
    // cloned loop.
    for (Operation *maskedOp : collector.getMaskedOps()) {
      Operation *mappedOp = map.lookup(maskedOp);
      if (auto loadOp = dyn_cast<tt::LoadOp>(mappedOp)) {
        OpBuilder builder(mappedOp);
        auto newLoad = builder.create<tt::LoadOp>(
            loadOp.getLoc(), loadOp.getPtr(), loadOp.getCache(),
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

    // Remove masks if they are not necessary.
    moduleOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (scf::ForOp forOp = dyn_cast<scf::ForOp>(op)) {
        // Nested loop aren't currently handled.
        if (forOp->template getParentOfType<scf::ForOp>())
          return WalkResult::advance();

        if (!forOp.getSingleInductionVar())
          return WalkResult::advance();

        RemovableMaskValidator maskValidator;
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
