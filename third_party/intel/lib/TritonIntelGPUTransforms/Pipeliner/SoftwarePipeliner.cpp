#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

#include "Pipeliner/Schedule.h"

#include "intel/include/Dialect/TritonGEN/IR/TritonGENDialect.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "intel/include/Utils/Utility.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

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

// Re-align the warps of a work-group at the top of the loop enclosing the
// pipelined `loop`, as a grid launch does at every tile: their own L1
// prefetches otherwise let them drift apart across a persistent kernel's tiles
// (#8719).
static void insertTileBarrier(scf::ForOp loop) {
  auto outer = loop->getParentOfType<scf::ForOp>();
  auto funcOp = loop->getParentOfType<triton::FuncOp>();
  if (!outer || !funcOp)
    return;
  // Every warp must reach the barrier equally often.
  auto isUniform = [&](Value v) {
    return triton::intel::isProgramUniform(v, funcOp, /*allowLoopIVs=*/true);
  };
  for (Operation *op = outer; op != funcOp; op = op->getParentOp()) {
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      if (!isUniform(forOp.getLowerBound()) ||
          !isUniform(forOp.getUpperBound()) || !isUniform(forOp.getStep()))
        return;
    } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      if (!isUniform(ifOp.getCondition()))
        return;
    } else {
      return;
    }
  }
  Block *body = outer.getBody();
  if (auto barrier = dyn_cast<triton::gpu::BarrierOp>(body->front());
      barrier && !barrier.isWarp())
    return;
  OpBuilder b = OpBuilder::atBlockBegin(body);
  triton::gpu::BarrierOp::create(b, loop.getLoc(),
                                 triton::gpu::AddrSpace::None);
}

static void pipelineLoop(scf::ForOp forOp, int numStages,
                         const ttgi::PipelineTarget &target) {
  mlir::scf::PipeliningOption options;
  if (!preCondition(forOp))
    return;

  bool foundSchedule =
      ttgi::preProcessLoopAndGetSchedule(forOp, numStages, target, options);
  if (!foundSchedule)
    return;

  IRRewriter rewriter(forOp->getContext());
  rewriter.setInsertionPoint(forOp);
  FailureOr<scf::ForOp> newForOp =
      mlir::scf::pipelineForLoop(rewriter, forOp, options);

  if (failed(newForOp))
    return;

  scf::ForOp loop = (*newForOp);
  if (target.useBarrier) {
    OpBuilder b(loop);
    Location loc = loop.getLoc();
    b.setInsertionPointToStart(loop.getBody());
    auto bData = triton::TritonGEN::SplitBarrierArriveOp::create(b, loc);
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    b.setInsertionPoint(yield);
    triton::TritonGEN::SplitBarrierWaitOp::create(b, loc, bData);
  } else if (target.tileBarrier) {
    insertTileBarrier(loop);
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

    if (!m->hasAttr(ttgi::TritonIntelGPUDialect::getSupport2DBlockIOAttrName()))
      return;

    if (numStages <= 1)
      return;

    SmallVector<scf::ForOp> loops;
    getOperation()->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });

    ttgi::PipelineTarget target{deviceArch, coreClockKhz, memClockMhz,
                                useBarrier, tileBarrier};
    for (scf::ForOp forOp : loops)
      pipelineLoop(forOp, numStages, target);
  }
};
} // anonymous namespace
