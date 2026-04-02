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
    // Optimize XOR-based floor division patterns. This is safe to run after
    // AxisInfo analysis (in make_llir, after gen_to_llvm) because AxisInfo
    // won't re-analyze the replacement divsi ops.
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
  /// Detects:
  ///   %cond   = arith.cmpi slt, %x, 0
  ///   %not_x  = arith.xori %x, -1
  ///   %abs_x  = arith.select %cond, %not_x, %x
  ///   %quot   = arith.divsi %abs_x, %divisor
  ///   %not_q  = arith.xori %quot, -1
  ///   %result = arith.select %cond, %not_q, %quot
  ///
  /// Replaces with (when divisor is a positive constant):
  ///   %quot   = arith.divsi %x, %divisor
  ///   %rem    = arith.remsi %x, %divisor
  ///   (floor adjustment using quot, rem, and sign of x)
  void optimizeFloorDivPatterns(ModuleOp moduleOp) {
    SmallVector<arith::SelectOp> selectOps;
    moduleOp.walk(
        [&](arith::SelectOp selectOp) { selectOps.push_back(selectOp); });

    unsigned count = 0;
    for (auto outerSelect : selectOps) {
      // Only optimize scalar ops (post-scalarization in make_llir).
      // Skip tensor ops (make_ttir) to avoid interfering with AxisInfo.
      if (isa<ShapedType>(outerSelect.getType()))
        continue;

      Value cond = outerSelect.getCondition();
      Value trueVal = outerSelect.getTrueValue();
      Value falseVal = outerSelect.getFalseValue();

      // %not_q = xori(%quot, -1)
      auto notQuotOp = trueVal.getDefiningOp<arith::XOrIOp>();
      if (!notQuotOp)
        continue;
      Value quotVal;
      if (isSplatValue(notQuotOp.getRhs(), -1))
        quotVal = notQuotOp.getLhs();
      else if (isSplatValue(notQuotOp.getLhs(), -1))
        quotVal = notQuotOp.getRhs();
      else
        continue;
      if (falseVal != quotVal)
        continue;

      // %quot = divsi(%abs_x, %divisor)
      auto divOp = quotVal.getDefiningOp<arith::DivSIOp>();
      if (!divOp)
        continue;
      Value absX = divOp.getLhs();
      Value divisor = divOp.getRhs();
      if (!isPositiveConstant(divisor))
        continue;

      // %abs_x = select(%cond, %not_x, %x)
      auto innerSelect = absX.getDefiningOp<arith::SelectOp>();
      if (!innerSelect || innerSelect.getCondition() != cond)
        continue;
      Value notX = innerSelect.getTrueValue();
      Value x = innerSelect.getFalseValue();

      // %not_x = xori(%x, -1)
      auto notXOp = notX.getDefiningOp<arith::XOrIOp>();
      if (!notXOp)
        continue;
      if (!((notXOp->getOperand(0) == x && isSplatValue(notXOp.getRhs(), -1)) ||
            (notXOp->getOperand(1) == x && isSplatValue(notXOp.getLhs(), -1))))
        continue;

      // %cond = cmpi slt, ?, 0 (condition may be shared across lanes)
      auto cmpOp = cond.getDefiningOp<arith::CmpIOp>();
      if (!cmpOp)
        continue;
      if (cmpOp.getPredicate() != arith::CmpIPredicate::slt)
        continue;
      if (!isSplatValue(cmpOp.getRhs(), 0))
        continue;

      // Replace with divsi(x, d) + remsi(x, d) + floor adjustment.
      OpBuilder builder(outerSelect);
      Location loc = outerSelect.getLoc();
      Type xType = x.getType();
      auto newDiv = arith::DivSIOp::create(builder, loc, x, divisor);
      auto newRem = arith::RemSIOp::create(builder, loc, x, divisor);

      auto zero = arith::ConstantOp::create(
          builder, loc, builder.getZeroAttr(xType));
      auto one = arith::ConstantOp::create(
          builder, loc, builder.getIntegerAttr(xType, 1));
      auto isNeg = arith::CmpIOp::create(builder, loc,
                                          arith::CmpIPredicate::slt, x, zero);
      auto hasRem = arith::CmpIOp::create(
          builder, loc, arith::CmpIPredicate::ne, newRem, zero);
      auto needAdj = arith::AndIOp::create(builder, loc, isNeg, hasRem);
      auto adjusted = arith::SubIOp::create(builder, loc, newDiv, one);
      auto result =
          arith::SelectOp::create(builder, loc, needAdj, adjusted, newDiv);

      outerSelect.replaceAllUsesWith(result.getResult());
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
