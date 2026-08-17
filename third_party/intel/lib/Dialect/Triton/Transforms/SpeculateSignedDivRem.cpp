#include "intel/include/Analysis/AxisInfoExt.h"
#include "intel/include/Analysis/Range.h"
#include "intel/include/Analysis/SignednessProver.h"
#include "intel/include/Dialect/Triton/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include <numeric>

#define DEBUG_TYPE "triton-intel-speculate-signed-div-rem"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = mlir::triton;

namespace mlir::triton::intel {
#define GEN_PASS_DEF_TRITONINTELSPECULATESIGNEDDIVREM
#include "intel/include/Dialect/Triton/Transforms/Passes.h.inc"
} // namespace mlir::triton::intel

namespace {

//===----------------------------------------------------------------------===//
// The deduction mirror
//===----------------------------------------------------------------------===//
//
// The three helpers below mirror their counterparts in
// lib/Analysis/AxisInfo.cpp
// (`gcd`, `AxisInfoVisitor::isContiguousDim`, `AxisInfoVisitor::isConstantDim`)
// so that signedDivRemDeductionApplies() can restate the disabled deduction
// conditions verbatim.

/// Greatest common divisor treating zero as the identity, matching the variadic
/// `gcd` in lib/Analysis/AxisInfo.cpp.
template <typename... Args> int64_t gcd(int64_t a, int64_t b, Args... args) {
  if (a == 0)
    return b;
  if (b == 0)
    return a;
  if constexpr (sizeof...(args) == 0)
    return std::gcd(a, b);
  else
    return gcd(std::gcd(a, b), args...);
}

bool isContiguousDim(const tt::AxisInfo &info, ArrayRef<int64_t> shape,
                     int dim) {
  return info.getContiguity(dim) == shape[dim];
}

bool isConstantDim(const tt::AxisInfo &info, ArrayRef<int64_t> shape, int dim) {
  return info.getConstancy(dim) == shape[dim];
}

} // namespace

namespace mlir::triton::intel {

bool signedDivRemDeductionApplies(Operation *op, const AxisInfo &lhs,
                                  const AxisInfo &rhs) {
  if (!isa<arith::DivSIOp, arith::RemSIOp>(op))
    return false;

  auto resTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!resTy)
    return false;

  // Mirrors DivOpAxisInfoVisitor::getConstancy and
  // RemOpAxisInfoVisitor::getContiguity: both require a contiguous dividend and
  // a constant divisor along the dimension, and both deduce
  // gcd(contiguity, divisibility, divisibility). The sound value is 1 in either
  // case (a contiguous dimension has constancy 1), so the deduction is only
  // worth something when that gcd exceeds 1.
  ArrayRef<int64_t> shape = resTy.getShape();
  for (int d = 0, rank = shape.size(); d < rank; ++d) {
    if (isContiguousDim(lhs, shape, d) && isConstantDim(rhs, shape, d) &&
        gcd(lhs.getContiguity(d), lhs.getDivisibility(d),
            rhs.getDivisibility(d)) > 1)
      return true;
  }

  return false;
}

} // namespace mlir::triton::intel

namespace {

using tt::intel::Goal;
using tt::intel::Proof;
using RequiredCondition = tt::intel::RequiredCondition;

/// Returns true if a `sge 0` check on `value` is meaningful. Excludes `i1`,
/// whose signed values are 0 and -1, so that the check would be wrong rather
/// than merely conservative.
bool isAssertableIntegerType(Value value) {
  Type elemTy = getElementTypeOrSelf(value.getType());
  return elemTy.isSignlessInteger() && elemTy.getIntOrFloatBitWidth() > 1;
}

/// Returns true if the `dividend >= 0` check guarding `op` can be asserted
/// outside every enclosing loop.
bool isCheckHoistableOutOfLoops(Operation *op) {
  // emitAssert() folds the check into a loop-carried flag for each enclosing
  // `scf.for` and stops at any other region-holding parent, so the assertion
  // ends up in the region of the first such parent.
  Operation *parent = op->getParentOp();
  while (isa_and_nonnull<scf::ForOp>(parent))
    parent = parent->getParentOp();

  return parent && !parent->getParentOfType<LoopLikeOpInterface>();
}

/// Folds `cond` into a loop-carried flag on `forOp` and returns the loop result
/// holding its conjunction over every executed iteration. `cond` must be
/// defined inside `forOp`'s body. Returns a null value if `forOp` cannot carry
/// an additional iteration argument.
Value accumulateIntoLoop(RewriterBase &rewriter, scf::ForOp forOp, Value cond) {
  Location loc = cond.getLoc();

  // The flag starts out true and is only ever cleared, so a loop that does not
  // execute yields "no violation" - which is what a division that never runs
  // means.
  rewriter.setInsertionPoint(forOp);
  Value init = arith::ConstantOp::create(rewriter, loc,
                                         rewriter.getOneAttr(cond.getType()));

  // Add the flag as a pass-through iteration argument first and redirect its
  // yielded value afterwards. Computing the conjunction in the
  // `NewYieldValuesFn` callback instead would have it reference `cond`, which
  // lives in the body being moved to the new loop while the callback runs.
  FailureOr<LoopLikeOpInterface> newLoop =
      cast<LoopLikeOpInterface>(forOp.getOperation())
          .replaceWithAdditionalIterOperands(
              rewriter, init, /*replaceInitOperandUsesInLoop=*/false);
  if (failed(newLoop)) {
    rewriter.eraseOp(init.getDefiningOp());
    return {};
  }

  auto newForOp = cast<scf::ForOp>(newLoop->getOperation());
  BlockArgument flag = newForOp.getRegionIterArgs().back();
  Operation *yieldOp = newForOp.getBody()->getTerminator();
  rewriter.setInsertionPoint(yieldOp);
  Value conjunction = arith::AndIOp::create(rewriter, loc, flag, cond);
  rewriter.modifyOpInPlace(yieldOp, [&]() {
    yieldOp->setOperand(yieldOp->getNumOperands() - 1, conjunction);
  });

  return newForOp.getResults().back();
}

class SignedDivRemSpeculator {
public:
  void run(ModuleOp moduleOp) {
    tt::intel::ModuleAxisInfoAnalysis axisInfo(moduleOp);

    SmallVector<Operation *> candidates;
    moduleOp.walk([&](Operation *op) {
      if (isCandidate(op, axisInfo))
        candidates.push_back(op);
    });
    if (candidates.empty())
      return;

    // Deciding which dividends are worth speculating on needs the range
    // analysis. It is built lazily, because it costs far more than the AxisInfo
    // query above and most modules have no candidate at all.
    DominanceInfo domInfo(moduleOp);
    std::unique_ptr<DataFlowSolver> solver = createDataFlowSolver();
    solver->load<tt::intel::IntegerRangeAnalysis>(moduleOp, domInfo);
    if (failed(solver->initializeAndRun(moduleOp))) {
      LDBG("range analysis failed, leaving all candidates signed");
      return;
    }

    // Split the candidates by what the prover says about the dividend: those it
    // proves outright convert for free, those it reduces to exactly one missing
    // fact convert under a runtime check, and the rest are left signed.
    SmallVector<Operation *> proven, speculative;
    for (Operation *op : candidates) {
      tt::intel::Proof proof =
          tt::intel::proveSign(op->getOperand(0), Goal::NonNegative, *solver);

      // The dividend is negative on every launch, or depends on a value that
      // is. Converting changes the result *and* asserts something that fails
      // unconditionally, so a kernel that computes correctly today would start
      // aborting.
      if (proof.verdict == Proof::Refuted) {
        LDBG("skipped, dividend cannot be non-negative: " << *op);
        continue;
      }

      if (proof.verdict == Proof::Satisfied) {
        proven.push_back(op);
        continue;
      }

      // A conjunction cannot be partially satisfied, so every member would have
      // to be checked for the conversion to be sound. One check is the expected
      // shape; more than that means the recursion wandered.
      if (proof.requiredConditions.size() != 1) {
        LDBG("skipped, dividend needs " << proof.requiredConditions.size()
                                        << " facts: " << *op);
        continue;
      }

      // An assertion in a loop body costs several times the kernel runtime -
      // far more than the deduction it recovers is worth - so an operation
      // whose check cannot be kept out of every loop is left signed instead.
      // This gates only the checked conversions; the proven ones emit no check,
      // so where a check would have landed does not concern them. See
      // emitAssert().
      if (!isCheckHoistableOutOfLoops(op)) {
        LDBG("skipped, assertion would land inside a loop: " << *op);
        continue;
      }

      speculative.push_back(op);
    }

    for (Operation *op : proven) {
      LDBG("converted unchecked, dividend is provably non-negative: " << *op);
      convertToUnsigned(op);
    }

    // Convert every remaining candidate and materialize its check before
    // asserting any of them: emitting an assertion can rebuild an enclosing
    // loop, which invalidates every handle to that loop's iteration arguments -
    // and a dividend is often one of them.
    for (Value cond : convertWithChecks(speculative))
      emitAssert(cond);
  }

private:
  bool isCandidate(Operation *op,
                   tt::intel::ModuleAxisInfoAnalysis &axisInfo) const {
    if (!isa<arith::DivSIOp, arith::RemSIOp>(op))
      return false;

    Value lhs = op->getOperand(0), rhs = op->getOperand(1);
    if (!isAssertableIntegerType(lhs))
      return false;

    tt::AxisInfo *lhsInfo = axisInfo.getAxisInfo(lhs);
    tt::AxisInfo *rhsInfo = axisInfo.getAxisInfo(rhs);
    if (!lhsInfo || !rhsInfo)
      return false;

    // The divisor must be a positive constant. `divui`/`remui` reinterpret a
    // negative divisor as a large positive one, so the result would be wrong
    // even for a non-negative dividend, and asserting the divisor is positive
    // would emit a check that fails on every launch for e.g. `x // -4`.
    std::optional<int64_t> divisor = rhsInfo->getConstantValue();
    if (!divisor.has_value() || *divisor <= 0)
      return false;

    if (!tt::intel::signedDivRemDeductionApplies(op, *lhsInfo, *rhsInfo))
      return false;

    LDBG("candidate: " << *op);
    return true;
  }

  /// Replaces `op` with its unsigned counterpart, erasing it, and returns the
  /// replacement. Sound only where the dividend is non-negative, which the
  /// caller must have established or arranged to check.
  Value convertToUnsigned(Operation *op) {
    Value lhs = op->getOperand(0), rhs = op->getOperand(1);
    OpBuilder builder(op);
    Location loc = op->getLoc();

    Value replacement =
        isa<arith::DivSIOp>(op)
            ? arith::DivUIOp::create(builder, loc, lhs, rhs).getResult()
            : arith::RemUIOp::create(builder, loc, lhs, rhs).getResult();
    op->getResult(0).replaceAllUsesWith(replacement);
    op->erase();

    return replacement;
  }

  /// Rewrites every candidate to its unsigned counterpart and materializes the
  /// `dividend >= 0` check it now relies on, returning one check per distinct
  /// (dividend, block) pair. Keying on the block keeps a check dominating the
  /// divisions it guards without a dominance query, and collapses the common
  /// case of one dividend feeding both a division and a remainder.
  SmallVector<Value> convertWithChecks(ArrayRef<Operation *> candidates) {
    SmallVector<Value> conditions;
    DenseSet<std::pair<Value, Block *>> checked;

    for (Operation *op : candidates) {
      Value lhs = op->getOperand(0);
      Location loc = op->getLoc();
      Block *block = op->getBlock();

      Value replacement = convertToUnsigned(op);

      if (checked.insert({lhs, block}).second) {
        OpBuilder builder(replacement.getDefiningOp());
        Value zero = arith::ConstantOp::create(
            builder, loc, builder.getZeroAttr(lhs.getType()));
        conditions.push_back(arith::CmpIOp::create(
            builder, loc, arith::CmpIPredicate::sge, lhs, zero));
        LDBG("checked dividend: " << lhs);
      }
    }

    return conditions;
  }

  /// Emits the single `tt.assert` backing a check built by convertWithChecks().
  ///
  /// A check inside a loop body is not asserted there: it is folded into a
  /// loop-carried flag and asserted once after the outermost enclosing
  /// `scf.for` instead. A `tt.assert` lowers to an opaque call that keeps IGC
  /// from optimizing any loop containing it, which costs several times the
  /// kernel runtime, whereas the same assertion outside the loop is free. The
  /// conjunction flags exactly the same launches, it just no longer points at
  /// the iteration that caused the violation. Note also that an out-of-range
  /// unsigned quotient may fault on a subsequent access before the assertion is
  /// reached; the diagnostic is a best effort either way.
  void emitAssert(Value cond) {
    IRRewriter rewriter(cond.getContext());
    Operation *defOp = cond.getDefiningOp();

    // isCheckHoistableOutOfLoops() kept only candidates whose loop ancestors
    // are all `scf.for`s, so this walk reaches a point outside every loop. It
    // can still stop early if a loop refuses an extra iteration argument,
    // leaving the assertion in the body: slow, but sound, and soundness is the
    // one thing that cannot be traded away here.
    while (auto forOp = dyn_cast<scf::ForOp>(defOp->getParentOp())) {
      Value accumulated = accumulateIntoLoop(rewriter, forOp, cond);
      if (!accumulated)
        break;
      cond = accumulated;
      defOp = cond.getDefiningOp();
    }

    rewriter.setInsertionPointAfter(defOp);
    tt::AssertOp::create(
        rewriter, cond.getLoc(), cond,
        "signed division or remainder with a negative dividend: the compiler "
        "assumed the dividend was non-negative in order to optimize the memory "
        "access pattern. Restructure the kernel so the dividend is "
        "non-negative, or set TRITON_SPECULATE_SIGNED_DIV_REM=0 to compile "
        "without this assumption (slower).");
  }
};

struct TritonIntelSpeculateSignedDivRem
    : tt::intel::impl::TritonIntelSpeculateSignedDivRemBase<
          TritonIntelSpeculateSignedDivRem> {
public:
  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    SignedDivRemSpeculator speculator;
    speculator.run(moduleOp);
    assert(succeeded(verify(moduleOp)) && "Module verification failed");
  }
};

} // namespace
