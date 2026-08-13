#include "intel/include/Analysis/Utility.h"
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
    return tt::gpu::intel::isNonNegative(op.getLhs()) &&
           isStrictlyPositive(op.getRhs());
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
