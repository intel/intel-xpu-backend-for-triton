#include "PipelineExpander.h"
#include "Schedule.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/Debug.h"

//===----------------------------------------------------------------------===//
// This file will create a schedule that will be handed over to the pipeline
// expander.
// Software pipeliners are usually separated into two pieces, one that create a
// modulo schedule and an expander that rewrites the loop and emits a prologue
// and epilogue. This pass first calls a helper that will pre-process the IR
// to create async operations and create a modulo schedule. Then we call the
// expander to generate the prologue and new loop.
//===----------------------------------------------------------------------===//

using namespace mlir;

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

static const char *kNumStagesAttrName = "tt.num_stages";

// Return true if the preconditions for pipelining the loop are met.
static bool preCondition(scf::ForOp forOp) {
  // Skip loop with distance > 1 for now.
  // TODO: relax the constraint in the expander.
  if (llvm::any_of(forOp.getBody()->getTerminator()->getOperands(),
                   [](Value operand) {
                     Operation *def = operand.getDefiningOp();
                     return !def;
                   }))
    return false;
  // Don't pipeline outer loops.
  if (forOp
          ->walk([&](Operation *op) {
            if (forOp.getOperation() == op)
              return WalkResult::advance();
            if (isa<scf::ForOp, scf::WhileOp>(op))
              return WalkResult::interrupt();
            return WalkResult::advance();
          })
          .wasInterrupted())
    return false;
  return true;
}

static void tryAndPipelineOuterLoop(scf::ForOp forOp) {
  mlir::triton::PipeliningOption options;
  bool foundSchedule = false;
  // Limit 2 stages to not require extra shared memory.
  foundSchedule = getOuterLoopSchedule(forOp, /*numStage=*/2, options);
  if (!foundSchedule)
    return;
  IRRewriter rewriter(forOp->getContext());
  rewriter.setInsertionPoint(forOp);
  FailureOr<scf::ForOp> newForOp =
      mlir::triton::pipelineForLoop(rewriter, forOp, options);
}

static bool pipelineLoop(scf::ForOp forOp, int numStages) {
  mlir::triton::PipeliningOption options;
  if (!preCondition(forOp))
    return false;

  bool foundSchedule = false;
  foundSchedule = preProcessLoopAndGetSchedule(forOp, numStages, options);

  // TODO: add more pipelines strategy.
  if (!foundSchedule)
    return false;

  IRRewriter rewriter(forOp->getContext());
  rewriter.setInsertionPoint(forOp);
  FailureOr<scf::ForOp> newForOp =
      mlir::triton::pipelineForLoop(rewriter, forOp, options);

  if (failed(newForOp))
    return false;
  mlir::triton::asyncLaunchDots(newForOp.value());
  return true;
}

namespace {
struct PipelinePass : public TritonGPUPipelineBase<PipelinePass> {
  PipelinePass() = default;
  PipelinePass(int numStages, int numWarps, int numCTAs,
               int computeCapability) {
    this->numStages = numStages;
    this->numWarps = numWarps;
    this->numCTAs = numCTAs;
    this->computeCapability = computeCapability;
  }

  int getNumStagesOrDefault(scf::ForOp forOp) {
    // Use the attribute attached to the loop if it exists otherwise use the
    // global control.
    if (!forOp->hasAttr(kNumStagesAttrName))
      return numStages;
    return forOp->getAttr(kNumStagesAttrName).cast<IntegerAttr>().getInt();
  }

  void runOnOperation() override {
    if (this->numStages <= 1)
      return;
    SmallVector<scf::ForOp> loops;
    getOperation()->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });

    llvm::SmallSetVector<scf::ForOp, 8> outerLoops;
    for (scf::ForOp forOp : loops) {
      auto outerLoop = dyn_cast<scf::ForOp>(forOp->getParentOp());
      int loopNumStages = getNumStagesOrDefault(forOp);
      bool pipelined = pipelineLoop(forOp, loopNumStages);
      if (pipelined && outerLoop)
        outerLoops.insert(outerLoop);
    }

    // schedule the waits
    mlir::triton::insertWaits(getOperation());

    // Clean up arithmetic before applying the next level of pipelining to
    // simplify the IR.
    auto arithDialect =
        getOperation().getContext()->getLoadedDialect<arith::ArithDialect>();
    RewritePatternSet patterns(getOperation().getContext());
    arithDialect->getCanonicalizationPatterns(patterns);
    if (applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))
            .failed())
      return signalPassFailure();

    // Try to pipeline the outer loop to overlap the prologue and epilogue of
    // the inner loop.
    for (scf::ForOp outerLoop : outerLoops)
      tryAndPipelineOuterLoop(outerLoop);
  }
};
} // anonymous namespace

std::unique_ptr<Pass>
mlir::triton::gpu::createPipelinePass(int numStages, int numWarps, int numCTAs,
                                      int computeCapability) {
  return std::make_unique<PipelinePass>(numStages, numWarps, numCTAs,
                                        computeCapability);
}
