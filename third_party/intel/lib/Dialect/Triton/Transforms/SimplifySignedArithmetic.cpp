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

/// Check if a constant is a splat of the given signed integer value.
static bool isSplatValue(Value value, int64_t expected) {
  auto constOp = value.getDefiningOp<arith::ConstantOp>();
  if (!constOp)
    return false;
  if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
    return intAttr.getValue().getSExtValue() == expected;
  if (auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue())) {
    if (denseAttr.getElementType().isSignlessInteger()) {
      return denseAttr.isSplat() &&
             denseAttr.getSplatValue<APInt>().getSExtValue() == expected;
    }
  }
  return false;
}

class SignedArithmeticSimplifier {
public:
  void run(ModuleOp moduleOp) {
    // First, optimize XOR-based floor division patterns.
    optimizeFloorDivPatterns(moduleOp);

    SmallVector<arith::RemSIOp> remOpsToConvert;
    SmallVector<arith::DivSIOp> divOpsToConvert;

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

    LLVM_DEBUG(llvm::dbgs()
               << "Converted " << divOpsToConvert.size() << " divsi and "
               << remOpsToConvert.size() << " remsi operations\n");
  }

private:
  /// Returns true if a signed div/rem operation can be converted to unsigned:
  /// - dividend must be provably non-negative
  /// - divisor must be a positive constant
  template <typename OpTy, typename = std::enable_if_t<llvm::is_one_of<
                               OpTy, arith::DivSIOp, arith::RemSIOp>::value>>
  bool isCandidate(OpTy op) const {
    return isNonNegative(op.getLhs()) && isPositiveConstant(op.getRhs());
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

    // arith.remui/divui always produce non-negative results
    if (isa<arith::RemUIOp, arith::DivUIOp>(defOp))
      return true;

    // arith.divsi/remsi with non-negative dividend and positive divisor
    if (auto divOp = dyn_cast<arith::DivSIOp>(defOp))
      return isNonNegative(divOp.getLhs()) &&
             isPositiveConstant(divOp.getRhs());
    if (auto remOp = dyn_cast<arith::RemSIOp>(defOp))
      return isNonNegative(remOp.getLhs()) &&
             isPositiveConstant(remOp.getRhs());

    // arith.andi with non-negative constant mask
    if (auto andOp = dyn_cast<arith::AndIOp>(defOp)) {
      if (isNonNegativeConstant(andOp.getLhs()) ||
          isNonNegativeConstant(andOp.getRhs()))
        return true;
    }

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

  /// Returns true if value is a non-negative constant.
  bool isNonNegativeConstant(Value value) const {
    auto constOp = value.getDefiningOp<arith::ConstantOp>();
    if (!constOp)
      return false;
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
      return intAttr.getValue().isNonNegative();
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

  /// Optimize XOR-based floor division patterns.
  ///
  /// Detects the pattern:
  ///   %cond   = arith.cmpi slt, %x, 0
  ///   %not_x  = arith.xori %x, -1
  ///   %abs_x  = arith.select %cond, %not_x, %x
  ///   %quot   = arith.divsi %abs_x, %divisor
  ///   %not_q  = arith.xori %quot, -1
  ///   %result = arith.select %cond, %not_q, %quot
  ///
  /// When %x is provably non-negative and %divisor is a positive constant,
  /// the entire pattern simplifies to: %result = arith.divsi %x, %divisor
  void optimizeFloorDivPatterns(ModuleOp moduleOp) {
    SmallVector<arith::SelectOp> selectOps;
    moduleOp.walk(
        [&](arith::SelectOp selectOp) { selectOps.push_back(selectOp); });

    unsigned count = 0;
    for (auto outerSelect : selectOps) {
      // Match: %result = select(%cond, %not_q, %quot)
      Value cond = outerSelect.getCondition();
      Value trueVal = outerSelect.getTrueValue();
      Value falseVal = outerSelect.getFalseValue();

      // %not_q = xori(%quot, -1)
      auto notQuotOp = trueVal.getDefiningOp<arith::XOrIOp>();
      if (!notQuotOp)
        continue;

      // Check which operand of xori is -1
      Value quotVal;
      if (isSplatValue(notQuotOp.getRhs(), -1))
        quotVal = notQuotOp.getLhs();
      else if (isSplatValue(notQuotOp.getLhs(), -1))
        quotVal = notQuotOp.getRhs();
      else
        continue;

      // falseVal must be the same %quot
      if (falseVal != quotVal)
        continue;

      // %quot = divsi(%abs_x, %divisor)
      auto divOp = quotVal.getDefiningOp<arith::DivSIOp>();
      if (!divOp)
        continue;

      Value absX = divOp.getLhs();
      Value divisor = divOp.getRhs();

      // %abs_x = select(%same_cond, %not_x, %x)
      auto innerSelect = absX.getDefiningOp<arith::SelectOp>();
      if (!innerSelect || innerSelect.getCondition() != cond)
        continue;

      Value innerTrue = innerSelect.getTrueValue();
      Value innerFalse = innerSelect.getFalseValue();

      // %not_x = xori(%x, -1)
      auto notXOp = innerTrue.getDefiningOp<arith::XOrIOp>();
      if (!notXOp)
        continue;

      Value x;
      if (isSplatValue(notXOp.getRhs(), -1))
        x = notXOp.getLhs();
      else if (isSplatValue(notXOp.getLhs(), -1))
        x = notXOp.getRhs();
      else
        continue;

      // innerFalse must be the same %x
      if (innerFalse != x)
        continue;

      // %cond = cmpi slt, %x, 0
      auto cmpOp = cond.getDefiningOp<arith::CmpIOp>();
      if (!cmpOp || cmpOp.getPredicate() != arith::CmpIPredicate::slt)
        continue;
      if (cmpOp.getLhs() != x || !isSplatValue(cmpOp.getRhs(), 0))
        continue;

      // Full pattern matched! Now check safety conditions.
      if (!isNonNegative(x) || !isPositiveConstant(divisor))
        continue;

      // Safe to replace: floordiv(x, d) == truncdiv(x, d) when x >= 0, d > 0
      LLVM_DEBUG(llvm::dbgs() << "Replacing XOR floor division with divsi: "
                              << outerSelect << "\n");

      OpBuilder builder(outerSelect);
      auto newDiv =
          arith::DivSIOp::create(builder, outerSelect.getLoc(), x, divisor);
      outerSelect.replaceAllUsesWith(newDiv.getResult());

      // Erase dead ops in reverse dependency order.
      if (outerSelect->use_empty())
        outerSelect.erase();
      if (notQuotOp->use_empty())
        notQuotOp.erase();
      if (divOp->use_empty())
        divOp.erase();
      if (innerSelect->use_empty())
        innerSelect.erase();
      if (notXOp->use_empty())
        notXOp.erase();
      if (cmpOp->use_empty())
        cmpOp.erase();

      ++count;
    }

    LLVM_DEBUG({
      if (count > 0)
        llvm::dbgs() << "Replaced " << count
                     << " XOR floor division pattern(s)\n";
    });
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
