#include "Schedule.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "triton/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "triton/Dialect/TritonIntelGPU/Transforms/Passes.h"

using namespace mlir;
namespace ttgi = mlir::triton::gpu::intel;

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"

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

static void pipelineLoop(scf::ForOp forOp, int numStages) {
  mlir::scf::PipeliningOption options;
  if (!preCondition(forOp))
    return;

  bool foundSchedule =
      ttgi::preProcessLoopAndGetSchedule(forOp, numStages, options);
  if (!foundSchedule)
    return;

  IRRewriter rewriter(forOp->getContext());
  rewriter.setInsertionPoint(forOp);
  FailureOr<scf::ForOp> newForOp =
      mlir::scf::pipelineForLoop(rewriter, forOp, options);
}

namespace {
struct IntelGPUPipelinePass
    : public TritonIntelGPUPipelineBase<IntelGPUPipelinePass> {
  IntelGPUPipelinePass() = default;
  IntelGPUPipelinePass(int numStages, ttgi::DeviceArch arch) {
    numStages = numStages;
    deviceArch = arch;
  }

  void runOnOperation() override {
    if (deviceArch != ttgi::DeviceArch::PVC)
      return;
    if (numStages <= 1)
      return;

    SmallVector<scf::ForOp> loops;
    getOperation()->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });

    for (scf::ForOp forOp : loops) {
      pipelineLoop(forOp, numStages);
    }
  }
};
} // anonymous namespace

std::unique_ptr<Pass>
ttgi::createTritonIntelGPUPipelinePass(int numStages, ttgi::DeviceArch arch) {
  return std::make_unique<IntelGPUPipelinePass>(numStages, arch);
}
