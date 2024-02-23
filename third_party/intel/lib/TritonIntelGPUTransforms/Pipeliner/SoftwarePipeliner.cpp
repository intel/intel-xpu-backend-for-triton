#include "Schedule.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "triton/Dialect/TritonIntelGPU/Transforms/Passes.h"

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
namespace ttgi = triton::gpu::intel;

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"

static void pipelineLoop(scf::ForOp forOp, int numStages) {
  mlir::scf::PipeliningOption options;
  // Skip loop with distance > 1 for now.
  // TODO: relax the constraint in the expander.
  if (llvm::any_of(forOp.getBody()->getTerminator()->getOperands(),
                   [](Value operand) {
                     Operation *def = operand.getDefiningOp();
                     return !def;
                   }))
    return;

  bool foundSchedule = false;
  foundSchedule = mlir::triton::gpu::intel::preProcessLoopAndGetScheduleIntel(
      forOp, numStages, options);

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
    this->numStages = numStages;
    this->deviceArch = arch;
  }

  void runOnOperation() override {
    if (this->numStages <= 1)
      return;
    //  Only the PVC support the prefetching ops.
    if (deviceArch != ttgi::DeviceArch::PVC)
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
