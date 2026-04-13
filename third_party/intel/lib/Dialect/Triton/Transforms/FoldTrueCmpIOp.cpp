#include "intel/include/Analysis/Range.h"
#include "intel/include/Dialect/Triton/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/Utils/InferIntRangeCommon.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
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
    if (!(*inputRanges)[0] || !(*inputRanges)[1])
      return std::nullopt;
    return intrange::evaluatePred(pred, *(*inputRanges)[0], *(*inputRanges)[1]);
  }
  return std::nullopt;
}

/// Fold arith.cmpi operations that are provably true or false into constants.
struct FoldTrueCmpIOp : OpRewritePattern<arith::CmpIOp> {
  using OpRewritePattern::OpRewritePattern;

  FoldTrueCmpIOp(MLIRContext *context, DataFlowSolver *solver)
      : OpRewritePattern(context), solver(solver) {};

  LogicalResult matchAndRewrite(arith::CmpIOp cmpOp,
                                PatternRewriter &rewriter) const override {
    auto result = evaluateCmpI(*solver, cmpOp);
    if (!result)
      return failure();

    TypedAttr constAttr = *result ? rewriter.getOneAttr(cmpOp.getType())
                                  : rewriter.getZeroAttr(cmpOp.getType());
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(cmpOp, constAttr);
    return success();
  }

  DataFlowSolver *solver;
};

struct FoldTrueCmpIPass
    : public tt::intel::impl::TritonIntelGPUFoldTrueCmpIBase<FoldTrueCmpIPass> {

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    std::unique_ptr<DataFlowSolver> solver = createDataFlowSolver();
    solver->load<tt::intel::IntegerRangeAnalysis>(mod,
                                                  getAnalysis<DominanceInfo>());

    if (failed(solver->initializeAndRun(getOperation())))
      return signalPassFailure();

    RewritePatternSet patterns(&getContext());
    patterns.add<FoldTrueCmpIOp>(&getContext(), solver.get());
    (void)applyPatternsGreedily(mod, std::move(patterns));
  }
};

} // namespace
