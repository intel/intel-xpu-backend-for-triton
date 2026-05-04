#include "intel/include/Analysis/Range.h"
#include "intel/include/Dialect/Triton/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/Utils/InferIntRangeCommon.h"
#include "triton/Analysis/Utility.h"

using namespace mlir;
namespace tt = mlir::triton;

namespace mlir::triton::intel {
#define GEN_PASS_DEF_TRITONINTELGPUFOLDTRUECMPI
#include "intel/include/Dialect/Triton/Transforms/Passes.h.inc"
} // namespace mlir::triton::intel

namespace {

/// Evaluate an arith.cmpi operation using integer range analysis.
/// Returns true/false if the comparison is provably always true/false,
/// or std::nullopt if it cannot be determined.
static std::optional<bool> evaluateCmpI(const DataFlowSolver &solver,
                                        arith::CmpIOp cmpOp) {
  if (auto inputRanges =
          tt::intel::collectRanges(solver, ValueRange{cmpOp.getOperands()})) {
    intrange::CmpPredicate pred =
        static_cast<intrange::CmpPredicate>(cmpOp.getPredicate());
    std::optional<ConstantIntRanges> lhs = (*inputRanges)[0];
    std::optional<ConstantIntRanges> rhs = (*inputRanges)[1];
    if (!lhs || !rhs)
      return std::nullopt;

    return intrange::evaluatePred(pred, *lhs, *rhs);
  }
  return std::nullopt;
}

struct FoldTrueCmpIPass
    : public tt::intel::impl::TritonIntelGPUFoldTrueCmpIBase<FoldTrueCmpIPass> {

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    // Phase 1: Run range analysis.
    std::unique_ptr<DataFlowSolver> solver = createDataFlowSolver();
    solver->load<tt::intel::IntegerRangeAnalysis>(mod,
                                                  getAnalysis<DominanceInfo>());
    if (failed(solver->initializeAndRun(mod)))
      return signalPassFailure();

    // Phase 2: Collect all foldable cmpi ops and their results.
    // Using a collect-then-apply approach instead of applyPatternsGreedily
    // avoids interleaved canonicalization that can erase Values and create new
    // ones reusing the same memory addresses, causing the solver to return
    // stale lattice entries with mismatched bitwidths.
    SmallVector<std::pair<arith::CmpIOp, bool>> folds;
    mod.walk([&](arith::CmpIOp cmpOp) {
      if (auto result = evaluateCmpI(*solver, cmpOp))
        folds.emplace_back(cmpOp, *result);
    });

    // Phase 3: Apply all replacements.
    IRRewriter rewriter(&getContext());
    for (auto [cmpOp, result] : folds) {
      rewriter.setInsertionPoint(cmpOp);
      TypedAttr constAttr = result ? rewriter.getOneAttr(cmpOp.getType())
                                   : rewriter.getZeroAttr(cmpOp.getType());
      auto constOp =
          arith::ConstantOp::create(rewriter, cmpOp.getLoc(), constAttr);
      rewriter.replaceAllUsesWith(cmpOp.getResult(), constOp.getResult());
      rewriter.eraseOp(cmpOp);
    }
  }
};

} // namespace
