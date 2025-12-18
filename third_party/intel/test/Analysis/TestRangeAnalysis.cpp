#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "third_party/intel/include/Analysis/Range.h"
#include "triton/Analysis/Utility.h"

using namespace mlir;
namespace tt = mlir::triton;

namespace mlir::triton::test::intel {

struct TestRangeAnalysisPass
    : PassWrapper<TestRangeAnalysisPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestRangeAnalysisPass)

  StringRef getArgument() const final {
    return "test-triton-intel-range-analysis";
  }
  StringRef getDescription() const final {
    return "print the result of the triton-intel-range-analysis pass";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ModuleOp mod = getOperation();

    std::shared_ptr<DataFlowSolver> solver = createDataFlowSolver();
    auto *rangeAnalysis = solver->load<tt::intel::IntegerRangeAnalysis>(
        mod, getAnalysis<DominanceInfo>());

    rangeAnalysis->initializeModule(mod);

    if (failed(solver->initializeAndRun(getOperation())))
      return signalPassFailure();

    auto isEmpty = [](ConstantIntRanges range) {
      return range.umin().getBitWidth() == 0 ||
             range.umax().getBitWidth() == 0 ||
             range.smin().getBitWidth() == 0 || range.smax().getBitWidth() == 0;
    };

    auto nonNegativePred = [&](Value v) {
      if (const auto *r =
              solver->lookupState<dataflow::IntegerValueRangeLattice>(v)) {
        if (r->getValue().isUninitialized())
          return false;
        if (isEmpty(r->getValue().getValue()))
          return false;
      }
      return succeeded(dataflow::staticallyNonNegative(*solver, v));
    };

    mod.walk<WalkOrder::PreOrder>([&solver](FuncOp funcOp) {
      auto args = funcOp.getArguments();
      if (auto argRanges = tt::intel::collectRanges(*solver, args)) {
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

    mod->walk<WalkOrder::PreOrder>([&solver, nonNegativePred,
                                    rangeAnalysis](Operation *op) {
      auto results = op->getResults();
      if (auto outputRanges = tt::intel::collectRanges(*solver, results)) {
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
          if (tt::intel::evaluatesToTrue(cmpOp, *solver))
            emitRemark(op->getLoc(), "result is true");
        }
      }

      int i = 0;
      for (auto result : results) {
        if (nonNegativePred(result)) {
          std::string nonNegs;
          llvm::raw_string_ostream nonNegSt(nonNegs);
          if (results.size() > 1)
            nonNegSt << "result " << i << ": non-neg";
          else
            nonNegSt << "non-neg";
          emitRemark(result.getLoc(), nonNegs);
        }
        i++;
      }

      if (LoopLikeOpInterface loop = llvm::dyn_cast<LoopLikeOpInterface>(op)) {
        uint64_t loopTripCount = getTotalTripCount(loop, *rangeAnalysis);
        emitRemark(loop.getLoc(), "inferred total trip count: " +
                                      std::to_string(loopTripCount));
      }
    });
  }
};

} // namespace mlir::triton::test::intel

namespace mlir::triton::test::intel {
void registerTestIntelRangeAnalysis() {
  PassRegistration<TestRangeAnalysisPass>();
}
} // namespace mlir::triton::test::intel
