#include "intel/include/Dialect/Triton/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Verifier.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-intel-simplify-signed-arithmetic"

using namespace mlir;
namespace tt = mlir::triton;

namespace mlir::triton::intel {
#define GEN_PASS_DEF_TRITONINTELSIMPLIFYSIGNEDARITHMETIC
#include "intel/include/Dialect/Triton/Transforms/Passes.h.inc"
} // namespace mlir::triton::intel

namespace {

class SignedArithmeticSimplifier {
public:
  void run(ModuleOp moduleOp) {
    SmallVector<arith::RemSIOp> remOpsToConvert;
    SmallVector<arith::DivSIOp> divOpsToConvert;
    SmallVector<arith::CeilDivSIOp> ceilDivOpsToConvert;

    // Collect divsi operations first (order matters for tracking)
    moduleOp.walk([&](arith::DivSIOp divOp) {
      if (!isCandidate(divOp))
        return WalkResult::skip();

      LLVM_DEBUG(llvm::dbgs()
                 << "Converting divsi to divui: " << divOp << "\n");
      divOpsToConvert.push_back(divOp);
      return WalkResult::advance();
    });

    // Collect remsi operations
    moduleOp.walk([&](arith::RemSIOp remOp) {
      if (!isCandidate(remOp))
        return WalkResult::skip();

      LLVM_DEBUG(llvm::dbgs()
                 << "Converting remsi to remui: " << remOp << "\n");
      remOpsToConvert.push_back(remOp);
      return WalkResult::advance();
    });

    // Collect ceildivsi operations
    moduleOp.walk([&](arith::CeilDivSIOp ceilDivOp) {
      if (!isCandidate(ceilDivOp))
        return WalkResult::skip();

      LLVM_DEBUG(llvm::dbgs()
                 << "Converting ceildivsi to ceildivui: " << ceilDivOp << "\n");
      ceilDivOpsToConvert.push_back(ceilDivOp);
      return WalkResult::advance();
    });

    // Convert divsi to divui
    for (arith::DivSIOp divOp : divOpsToConvert) {
      OpBuilder builder(divOp);
      auto newOp = arith::DivUIOp::create(builder, divOp.getLoc(),
                                          divOp.getLhs(), divOp.getRhs());
      divOp.replaceAllUsesWith(newOp.getResult());
      divOp.erase();
    }

    // Convert remsi to remui
    for (arith::RemSIOp remOp : remOpsToConvert) {
      OpBuilder builder(remOp);
      auto newOp = arith::RemUIOp::create(builder, remOp.getLoc(),
                                          remOp.getLhs(), remOp.getRhs());
      remOp.replaceAllUsesWith(newOp.getResult());
      remOp.erase();
    }

    // Convert ceildivsi to ceildivui
    for (arith::CeilDivSIOp ceilDivOp : ceilDivOpsToConvert) {
      OpBuilder builder(ceilDivOp);
      auto newOp = arith::CeilDivUIOp::create(
          builder, ceilDivOp.getLoc(), ceilDivOp.getLhs(), ceilDivOp.getRhs());
      ceilDivOp.replaceAllUsesWith(newOp.getResult());
      ceilDivOp.erase();
    }

    LLVM_DEBUG(llvm::dbgs()
               << "Converted " << divOpsToConvert.size() << " divsi, "
               << remOpsToConvert.size() << " remsi, and "
               << ceilDivOpsToConvert.size() << " ceildivsi operations\n");
  }

private:
  /// Returns true if a signed div/rem operation can be converted to unsigned:
  /// - dividend must be provably non-negative
  /// - divisor must be strictly positive
  template <
      typename OpTy,
      typename = std::enable_if_t<llvm::is_one_of<
          OpTy, arith::DivSIOp, arith::RemSIOp, arith::CeilDivSIOp>::value>>
  bool isCandidate(OpTy op) const {
    return isNonNegative(op.getLhs()) && isStrictlyPositive(op.getRhs());
  }

  /// Returns true if value is provably non-negative (>= 0).
  bool isNonNegative(Value value) const {
    Operation *defOp = value.getDefiningOp();
    if (!defOp)
      return false;

    // tt.get_program_id always returns [0, 2^31-1]
    if (isa<tt::GetProgramIdOp>(defOp))
      return true;

    // tt.get_num_programs returns [1, 2^31]
    if (isa<tt::GetNumProgramsOp>(defOp))
      return true;

    // tt.make_range with non-negative start
    // Note: getStart() returns uint32_t, use getStartAttr().getInt() for signed
    if (auto makeRange = dyn_cast<tt::MakeRangeOp>(defOp))
      return makeRange.getStartAttr().getInt() >= 0;

    // Non-negative constant (scalar or tensor)
    if (auto constOp = dyn_cast<arith::ConstantOp>(defOp)) {
      if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
        return intAttr.getValue().isNonNegative();
      if (auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue())) {
        if (denseAttr.getElementType().isSignlessInteger()) {
          return llvm::all_of(denseAttr.getValues<APInt>(),
                              [](const APInt &v) { return v.isNonNegative(); });
        }
      }
    }

    // arith.addi of two non-negative values (assumes no overflow)
    if (auto addOp = dyn_cast<arith::AddIOp>(defOp))
      return isNonNegative(addOp.getLhs()) && isNonNegative(addOp.getRhs());

    // arith.muli of two non-negative values (assumes no overflow)
    if (auto mulOp = dyn_cast<arith::MulIOp>(defOp))
      return isNonNegative(mulOp.getLhs()) && isNonNegative(mulOp.getRhs());

    // arith.remui/divui/extui produce non-negative results. extui zero-extends
    // into a wider type, so the result MSB is always clear.
    if (isa<arith::RemUIOp, arith::DivUIOp, arith::ExtUIOp>(defOp))
      return true;

    // arith.divsi with non-negative dividend and non-negative divisor.
    // For well-defined programs the divisor is non-zero, so non-negative
    // implies positive, and divsi(non-neg, positive) >= 0.
    if (auto divOp = dyn_cast<arith::DivSIOp>(defOp))
      return isNonNegative(divOp.getLhs()) && isNonNegative(divOp.getRhs());

    // arith.remsi: result has the same sign as the dividend (truncation toward
    // zero), so a non-negative dividend guarantees a non-negative result
    // regardless of the divisor's sign.
    if (auto remOp = dyn_cast<arith::RemSIOp>(defOp))
      return isNonNegative(remOp.getLhs());

    // arith.extsi preserves the signed value, hence its sign.
    if (auto extOp = dyn_cast<arith::ExtSIOp>(defOp))
      return isNonNegative(extOp.getIn());

    // arith.shrsi (arithmetic right shift) replicates the sign bit; the result
    // is non-negative iff the shifted value is non-negative.
    if (auto shrOp = dyn_cast<arith::ShRSIOp>(defOp))
      return isNonNegative(shrOp.getLhs());

    // arith.maxsi: non-negative if either operand is non-negative.
    if (auto maxOp = dyn_cast<arith::MaxSIOp>(defOp))
      return isNonNegative(maxOp.getLhs()) || isNonNegative(maxOp.getRhs());

    // arith.minsi: non-negative iff BOTH operands are non-negative.
    if (auto minOp = dyn_cast<arith::MinSIOp>(defOp))
      return isNonNegative(minOp.getLhs()) && isNonNegative(minOp.getRhs());

    // arith.select yields one of its two value operands; non-negative iff both
    // candidate values are non-negative.
    if (auto selOp = dyn_cast<arith::SelectOp>(defOp))
      return isNonNegative(selOp.getTrueValue()) &&
             isNonNegative(selOp.getFalseValue());

    // arith.andi is non-negative if EITHER operand is non-negative, since
    // MSB(a & b) = MSB(a) & MSB(b). Subsumes the constant-mask case.
    if (auto andOp = dyn_cast<arith::AndIOp>(defOp))
      return isNonNegative(andOp.getLhs()) || isNonNegative(andOp.getRhs());

    // tt.splat preserves non-negativity
    if (auto splatOp = dyn_cast<tt::SplatOp>(defOp))
      return isNonNegative(splatOp.getSrc());

    // tt.expand_dims preserves non-negativity
    if (auto expandOp = dyn_cast<tt::ExpandDimsOp>(defOp))
      return isNonNegative(expandOp.getSrc());

    // tt.broadcast preserves non-negativity
    if (auto broadcastOp = dyn_cast<tt::BroadcastOp>(defOp))
      return isNonNegative(broadcastOp.getSrc());

    return false;
  }

  /// Returns true if value is a positive constant (> 0).
  bool isPositiveConstant(Value value) const {
    Operation *defOp = value.getDefiningOp();
    if (!defOp)
      return false;

    if (auto constOp = dyn_cast<arith::ConstantOp>(defOp)) {
      if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
        return intAttr.getValue().isStrictlyPositive();
      if (auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue())) {
        if (denseAttr.getElementType().isSignlessInteger()) {
          return llvm::all_of(denseAttr.getValues<APInt>(), [](const APInt &v) {
            return v.isStrictlyPositive();
          });
        }
      }
    }

    // Splat of positive constant
    if (auto splatOp = dyn_cast<tt::SplatOp>(defOp))
      return isPositiveConstant(splatOp.getSrc());

    return false;
  }

  /// Returns true if value is provably strictly positive (> 0). Superset of
  /// isPositiveConstant.
  bool isStrictlyPositive(Value value) const {
    if (isPositiveConstant(value))
      return true;

    Operation *defOp = value.getDefiningOp();
    if (!defOp)
      return false;

    // tt.get_num_programs is a grid dimension, always >= 1.
    if (isa<tt::GetNumProgramsOp>(defOp))
      return true;

    // arith.maxsi(x, c) >= c; strictly positive if either operand is.
    if (auto maxOp = dyn_cast<arith::MaxSIOp>(defOp))
      return isStrictlyPositive(maxOp.getLhs()) ||
             isStrictlyPositive(maxOp.getRhs());

    // tt.splat preserves strict positivity (tensor divisors).
    if (auto splatOp = dyn_cast<tt::SplatOp>(defOp))
      return isStrictlyPositive(splatOp.getSrc());

    return false;
  }
};

struct TritonIntelSimplifySignedArithmetic
    : tt::intel::impl::TritonIntelSimplifySignedArithmeticBase<
          TritonIntelSimplifySignedArithmetic> {
public:
  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    SignedArithmeticSimplifier simplifier;
    simplifier.run(moduleOp);
    assert(succeeded(verify(moduleOp)) && "Module verification failed");
  }
};

} // namespace
