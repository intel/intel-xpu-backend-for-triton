#include "intel/include/Dialect/Triton/Transforms/Passes.h"
#include "intel/include/Utils/Utility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
// #include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-intel-remove-masks"

using namespace mlir;
namespace tt = mlir::triton;

namespace mlir::triton::intel {
#define GEN_PASS_DEF_TRITONINTELREMOVEMASKS
#include "intel/include/Dialect/Triton/Transforms/Passes.h.inc"
} // namespace mlir::triton::intel

namespace {

// Represent a versioning condition for a loop.
class VersioningCondition {
public:
  VersioningCondition(Value S, Value BS) : S(S), BS(BS) {
    assert(isValid() && "Invalid values supplied");
  }

  // Create the condition: (S % BS == 0 && S > BS)
  Value materialize(OpBuilder &builder, Location loc) const {
    assert(S && BS && "Expecting valid values");
    Value zero =
        builder.createOrFold<arith::ConstantIntOp>(loc, 0, S.getType());
    Value cmp1 = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq,
        builder.create<arith::RemSIOp>(loc, S, BS), zero);
    Value cmp2 =
        builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, S, BS);
    return builder.create<arith::AndIOp>(loc, cmp1, cmp2);
  }

private:
  bool isValid() const {
    Type SType = S.getType(), BSType = BS.getType();
    if (!isa<IntegerType>(SType) || !isa<IntegerType>(BSType))
      return false;

    return cast<IntegerType>(SType).getWidth() ==
           cast<IntegerType>(BSType).getWidth();
  }

  Value S;  // The length of a row/column.
  Value BS; // The block size.
};

// Collects masked operations conditions in a loop.
class MaskedOpsCollector {
public:
  using MaskedOperations = SmallPtrSet<Operation *, 8>;

  MaskedOpsCollector(scf::ForOp &forOp) : forOp(forOp) {
    assert(!forOp->template getParentOfType<scf::ForOp>() &&
           "Nested loop not handled yet");
    createVersioningCondition(forOp);
  }

  // Collect mask condition that can be made loop invariant for the `tt.load`
  // operation in the given loop.
  bool collectMaskedOps() {
    assert(versioningCond && "Versioning condition should be valid");

    // Collect masked loads in the loop if they have canonical mask.
    for (auto op : forOp.getOps<tt::LoadOp>()) {
      Value mask = op.getMask();
      if (mask && isValidMask(tt::intel::getFinalValue(mask)))
        maskedOps.insert(op);
    }

    // TODO: collect masked stores in the loop if they have canonical mask.
    return maskedOps.size();
  }

  VersioningCondition *getVersioningCond() const {
    return versioningCond.get();
  };

  const MaskedOperations &getMaskedOps() const { return maskedOps; };

private:
  // Note: this assumes the loop UB is in canonical form `N+END-1)/END`.
  void createVersioningCondition(scf::ForOp &forOp) {
    Value ub = tt::intel::getFinalValue(forOp.getUpperBound());
    Operation *defOp = ub.getDefiningOp();
    auto divOp = cast<arith::DivSIOp>(defOp);
    Operation *divLhsOp = divOp.getLhs().getDefiningOp();
    auto divNumOp = cast<arith::AddIOp>(divLhsOp);
    versioningCond = std::make_unique<VersioningCondition>(divNumOp.getLhs(),
                                                           divOp.getRhs());
  }

  // Check whether a mask is in canonical form: (0..END) < N - i*END
  bool isValidMask(Value mask) const {
    if (!mask.getDefiningOp() || !isa<arith::CmpIOp>(mask.getDefiningOp()))
      return false;

    auto cmpOp = cast<arith::CmpIOp>(mask.getDefiningOp());
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
    if (subLhs || !isa<arith::MulIOp>(subRhs))
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

private:
  scf::ForOp &forOp;

  // Masked operations that can be have their mask dropped when the loop is
  // versioned using the versioning condition associated with this class.
  MaskedOperations maskedOps;

  std::unique_ptr<VersioningCondition> versioningCond = nullptr;
};

class LoopVersioner {
public:
  // Version the \p forOp loop with a condition that makes the masks collected
  // by \p collector unnecessary.
  // TODO: Extend the versioning region to encompass the downward exposed uses
  // of the return values.
  static bool version(scf::ForOp &forOp, MaskedOpsCollector &collector) {
    assert(collector.getVersioningCond() &&
           "Versioning condition should be present");

    // Limitation: give up if the loop returns tensor of ptrs.
    if (!canVersion(forOp))
      return false;

    // Collect loop results that are downward exposed.
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
    Value versioningCond =
        collector.getVersioningCond()->materialize(builder, loc);
    auto ifOp =
        builder.create<scf::IfOp>(loc, getUsedResults(forOp), versioningCond,
                                  /*withThenRegion=*/true,
                                  /*withElseRegion=*/true);

    // Clone the original loop into the 2 if branches.
    OpBuilder thenB = ifOp.getThenBodyBuilder();
    OpBuilder elseB = ifOp.getElseBodyBuilder();

    IRMapping map;
    Operation *thenForLoop = thenB.clone(*forOp.getOperation(), map);
    Operation *elseForLoop = elseB.clone(*forOp.getOperation());

    // Collect results in 'clonedLoop' corresponding to downward exposed results
    // 'forOp'.
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
    unsigned idx = 0;
    for (Value res : forOp.getResults()) {
      if (!res.getUsers().empty())
        res.replaceAllUsesWith(ifOp->getResult(idx++));
    }

    forOp.erase();

    return true;
  }

  // Ensure the loop upper bound is in canonical form (N+END-1)/END.
  static bool hasValidUpperBound(scf::ForOp &forOp) {
    Value ub = tt::intel::getFinalValue(forOp.getUpperBound());
    Operation *defOp = ub.getDefiningOp();
    if (!defOp || !isa<arith::DivSIOp>(defOp))
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
  // Currently we can version the loop only is it doesn't have downward
  // exposed uses of return values that are a tensor of pointers.
  // Note: this is due to the fact the results yielded by the 2 versioning
  // branches have different types for ptr (only in one versioned loop tensor of
  // ptrs are changed to block ptrs) 'then' part of the versioning branch and
  // leave them as is in the 'else' branch).
  static bool canVersion(scf::ForOp &forOp) {
    return llvm::any_of(forOp.getResults(), [](Value res) {
      return !tt::isTensorPointerType(res.getType()) || res.getUsers().empty();
    });
  }
};

struct TritonIntelRemoveMasksBase
    : tt::intel::impl::TritonIntelRemoveMasksBase<TritonIntelRemoveMasksBase> {
public:
  using Base::Base;
  using IndexMapSet = std::map<int, std::set<int>>;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    // Attempt to version loops so that masked operations in the loop become
    // superfluous.
    moduleOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (scf::ForOp forOp = dyn_cast<scf::ForOp>(op)) {
        // Nested loop aren't currently handled.
        if (forOp->template getParentOfType<scf::ForOp>())
          return WalkResult::advance();

        if (!forOp.getSingleInductionVar() ||
            !LoopVersioner::hasValidUpperBound(forOp))
          return WalkResult::advance();

        MaskedOpsCollector collector(forOp);
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
