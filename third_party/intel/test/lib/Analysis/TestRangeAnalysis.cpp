#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Pass/Pass.h"
#include "third_party/intel/include/Analysis/Range.h"
#include "triton/Analysis/Utility.h"

using namespace mlir;
using namespace mlir::triton::intel;

namespace {

struct TestRangeAnalysisPass
    : public PassWrapper<TestRangeAnalysisPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestRangeAnalysisPass)

  StringRef getArgument() const final { return "test-intel-range-analysis"; }
  StringRef getDescription() const final {
    return "print the result of the range analysis pass";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ModuleOp mod = getOperation();

    std::shared_ptr<DataFlowSolver> solver = createDataFlowSolver();
    auto *rangeAnalysis =
        solver->load<IntegerRangeAnalysis>(mod, getAnalysis<DominanceInfo>());

    if (failed(solver->initializeAndRun(getOperation())))
      return signalPassFailure();

    mod.walk<WalkOrder::PreOrder>([&solver](triton::FuncOp funcOp) {
      ValueRange args = funcOp.getArguments();
      if (auto argRanges = collectRanges(*solver, args)) {
        int i = -1;
        for (const auto &[arg, argR] : llvm::zip(args, *argRanges)) {
          i++;
          if (!argR)
            continue;
          std::string rangeS;
          llvm::raw_string_ostream rangeSt(rangeS);
          if (args.size() > 1)
            rangeSt << "arg " << i << ": " << argR;
          else
            rangeSt << argR;
          emitRemark(arg.getLoc(), rangeS);
        }
      }
    });

    mod->walk<WalkOrder::PreOrder>([&solver, rangeAnalysis](Operation *op) {
      ResultRange results = op->getResults();
      if (auto outputRanges = collectRanges(*solver, results)) {
        int i = -1;
        for (const auto &[res, outR] : llvm::zip(results, *outputRanges)) {
          i++;
          if (!outR)
            continue;
          std::string rangeS;
          llvm::raw_string_ostream rangeSt(rangeS);
          if (results.size() > 1)
            rangeSt << "result " << i << ": " << outR;
          else
            rangeSt << outR;
          emitRemark(res.getLoc(), rangeS);
        }

        if (auto cmpOp = llvm::dyn_cast<arith::CmpIOp>(op)) {
          if (evaluatesToTrue(cmpOp, *solver))
            emitRemark(op->getLoc(), "result is true");
        }
      }

      if (LoopLikeOpInterface loop = llvm::dyn_cast<LoopLikeOpInterface>(op)) {
        uint64_t loopTripCount = getTotalTripCount(loop, *rangeAnalysis);
        emitRemark(loop.getLoc(), "inferred total trip count: " +
                                      std::to_string(loopTripCount));
      }
    });
  }
};

} // namespace

namespace mlir::test::intel {
void registerTestRangeAnalysisPass() {
  PassRegistration<TestRangeAnalysisPass>();
}
} // namespace mlir::test::intel
