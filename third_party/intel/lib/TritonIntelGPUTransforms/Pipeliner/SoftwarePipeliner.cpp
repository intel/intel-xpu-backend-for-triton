#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

#include "Pipeliner/Schedule.h"

#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"

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

static void pipelineLoop(
    scf::ForOp forOp, int numStages,
    std::optional<triton::TritonGEN::MemScope> barrierScope = std::nullopt) {
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

  if (failed(newForOp))
    return;

  scf::ForOp loop = (*newForOp);
  if (barrierScope) {
    assert((*barrierScope == triton::TritonGEN::MemScope::SUB_GROUP) ||
           (*barrierScope == triton::TritonGEN::MemScope::WORK_GROUP) &&
               "The barrier scope must be SubGroup or Workgroup");
    OpBuilder b(loop);
    Location loc = loop.getLoc();
    b.setInsertionPointToStart(loop.getBody());
    b.create<triton::TritonGEN::SplitBarrierArriveOp>(loc, *barrierScope,
                                                      *barrierScope);
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    b.setInsertionPoint(yield);
    b.create<triton::TritonGEN::SplitBarrierWaitOp>(loc, *barrierScope,
                                                    *barrierScope);
  }
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

    std::optional<triton::TritonGEN::MemScope> barrierScope = std::nullopt;
    switch (splitBarrierScope) {
    case ttgi::SplitBarrierScope::None:
      break;
    case ttgi::SplitBarrierScope::Workgroup:
      barrierScope = triton::TritonGEN::MemScope::WORK_GROUP;
      break;
    case ttgi::SplitBarrierScope::Subgroup:
      barrierScope = triton::TritonGEN::MemScope::SUB_GROUP;
      break;
    }

    SmallVector<scf::ForOp> loops;
    getOperation()->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });

    for (scf::ForOp forOp : loops)
      pipelineLoop(forOp, numStages, barrierScope);
  }
};
} // anonymous namespace
