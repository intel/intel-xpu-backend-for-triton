#include "intel/include/Analysis/AxisInfoExt.h"
#include "intel/include/Analysis/Range.h"
#include "intel/include/Dialect/Triton/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Verifier.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
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

/// Returns true if a `sge 0` check on `value` is meaningful. Excludes `i1`,
/// whose signed values are 0 and -1, so that the check would be wrong rather
/// than merely conservative.
bool isAssertableIntegerType(Value value) {
  Type elemTy = getElementTypeOrSelf(value.getType());
  return elemTy.isSignlessInteger() && elemTy.getIntOrFloatBitWidth() > 1;
}

/// Returns true if `value` is negative in every element on every launch, so
/// that asserting it non-negative would abort unconditionally.
bool isProvablyNegative(Value value, const DataFlowSolver &solver) {
  std::optional<ConstantIntRanges> range =
      tt::intel::collectRange(solver, value);
  return range.has_value() && range->smax().isNegative();
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

    // A provably negative dividend must be left alone: converting it changes
    // the result *and* emits an assertion that fails on every launch, turning a
    // kernel that computes correctly today into one that aborts. The range
    // analysis needed to detect that is built lazily, because it costs far more
    // than the AxisInfo query above and most modules have no candidate at all.
    DominanceInfo domInfo(moduleOp);
    std::unique_ptr<DataFlowSolver> solver = createDataFlowSolver();
    solver->load<tt::intel::IntegerRangeAnalysis>(moduleOp, domInfo);
    if (failed(solver->initializeAndRun(moduleOp))) {
      LDBG("range analysis failed, leaving all candidates signed");
      return;
    }

    llvm::erase_if(candidates, [&](Operation *op) {
      if (!isProvablyNegative(op->getOperand(0), *solver))
        return false;
      LDBG("skipped, dividend is provably negative: " << *op);
      return true;
    });

    for (Operation *op : candidates)
      convertWithAssert(op);
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

  void convertWithAssert(Operation *op) {
    Value lhs = op->getOperand(0), rhs = op->getOperand(1);
    OpBuilder builder(op);
    Location loc = op->getLoc();

    Value replacement =
        isa<arith::DivSIOp>(op)
            ? arith::DivUIOp::create(builder, loc, lhs, rhs).getResult()
            : arith::RemUIOp::create(builder, loc, lhs, rhs).getResult();
    op->getResult(0).replaceAllUsesWith(replacement);
    op->erase();

    assertNonNegative(lhs, replacement.getDefiningOp());
  }

  /// Emits `tt.assert(value >= 0)` before `insertionPoint`, at most once per
  /// (value, block) pair. Keying on the block keeps the assertion dominating
  /// its users without a dominance query, and collapses the common case of one
  /// dividend feeding both a division and a remainder into a single assertion.
  void assertNonNegative(Value value, Operation *insertionPoint) {
    if (!asserted.insert({value, insertionPoint->getBlock()}).second)
      return;

    OpBuilder builder(insertionPoint);
    Location loc = insertionPoint->getLoc();
    Value zero = arith::ConstantOp::create(
        builder, loc, builder.getZeroAttr(value.getType()));
    Value cond = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sge,
                                       value, zero);
    tt::AssertOp::create(
        builder, loc, cond,
        "signed division or remainder with a negative dividend: the compiler "
        "assumed the dividend was non-negative in order to optimize the memory "
        "access pattern. Restructure the kernel so the dividend is "
        "non-negative, or set TRITON_SPECULATE_SIGNED_DIV_REM=0 to compile "
        "without this assumption (slower).");
    LDBG("asserted dividend: " << value);
  }

  DenseSet<std::pair<Value, Block *>> asserted;
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
