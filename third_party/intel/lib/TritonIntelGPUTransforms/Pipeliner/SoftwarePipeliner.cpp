
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

#include "Pipeliner/Schedule.h"

#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Utility.h"

using namespace mlir;
namespace ttgi = mlir::triton::gpu::intel;

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONINTELGPUPIPELINE
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

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
            if (isa<LoopLikeOpInterface>(op) && forOp.getOperation() != op)
              return WalkResult::interrupt();
            return WalkResult::advance();
          })
          .wasInterrupted())
    return false;
  return true;
}

static void pipelineLoop(scf::ForOp forOp, int numStages,
                         bool supportRegularPtr) {
  mlir::scf::PipeliningOption options;
  if (!preCondition(forOp))
    return;

  bool foundSchedule = ttgi::preProcessLoopAndGetSchedule(
      forOp, numStages, supportRegularPtr, options);
  if (!foundSchedule)
    return;

  IRRewriter rewriter(forOp->getContext());
  rewriter.setInsertionPoint(forOp);
  FailureOr<scf::ForOp> newForOp =
      mlir::scf::pipelineForLoop(rewriter, forOp, options);
}

namespace {
struct IntelGPUPipelinePass
    : public triton::gpu::intel::impl::TritonIntelGPUPipelineBase<
          IntelGPUPipelinePass> {

  using triton::gpu::intel::impl::TritonIntelGPUPipelineBase<
      IntelGPUPipelinePass>::TritonIntelGPUPipelineBase;

  void runOnOperation() override {
    ModuleOp m = getOperation();

    if (!m->hasAttr(ttgi::TritonIntelGPUDialect::getSupportSG2DBlockAttrName()))
      return;

    if (numStages <= 1)
      return;

    SmallVector<scf::ForOp> loops;
    getOperation()->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });

    for (scf::ForOp forOp : loops) {
      pipelineLoop(forOp, numStages, supportRegularPtr);
    }
  }
};
} // anonymous namespace
