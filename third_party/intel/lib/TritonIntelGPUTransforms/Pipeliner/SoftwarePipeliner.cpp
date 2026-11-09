#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

#include "Pipeliner/Schedule.h"

#include "intel/include/Dialect/TritonGEN/IR/TritonGENDialect.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "llvm/Support/MathExtras.h"

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

static void pipelineLoop(scf::ForOp forOp, int numStages, bool useBarrier) {
  mlir::scf::PipeliningOption options;
  if (!preCondition(forOp))
    return;

  std::optional<int64_t> prefetchBytes;
  bool foundSchedule = ttgi::preProcessLoopAndGetSchedule(
      forOp, numStages, options, prefetchBytes);
  if (!foundSchedule)
    return;

  IRRewriter rewriter(forOp->getContext());
  rewriter.setInsertionPoint(forOp);
  FailureOr<scf::ForOp> newForOp =
      mlir::scf::pipelineForLoop(rewriter, forOp, options);

  if (failed(newForOp))
    return;

  scf::ForOp loop = (*newForOp);
  if (useBarrier) {
    OpBuilder b(loop);
    Location loc = loop.getLoc();
    b.setInsertionPointToStart(loop.getBody());
    auto bData = triton::TritonGEN::SplitBarrierArriveOp::create(b, loc);
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    b.setInsertionPoint(yield);
    triton::TritonGEN::SplitBarrierWaitOp::create(b, loc, bData);
  }

  Builder builder(loop.getContext());
  NamedAttrList prefetchInfo;
  prefetchInfo.set("barrier", builder.getBoolAttr(useBarrier));
  if (prefetchBytes)
    prefetchInfo.set("total_bytes", builder.getI64IntegerAttr(*prefetchBytes));
  loop->setAttr("ttig.pipelined_prefetch",
                prefetchInfo.getDictionary(loop.getContext()));
}

namespace {
struct IntelGPUPipelinePass
    : public triton::gpu::intel::impl::TritonIntelGPUPipelineBase<
          IntelGPUPipelinePass> {

  using triton::gpu::intel::impl::TritonIntelGPUPipelineBase<
      IntelGPUPipelinePass>::TritonIntelGPUPipelineBase;

  void runOnOperation() override {
    ModuleOp m = getOperation();

    if (m->hasAttr(
            ttgi::TritonIntelGPUDialect::getSupport2DBlockIOAttrName()) &&
        numStages > 1) {
      SmallVector<scf::ForOp> loops;
      m.walk([&](scf::ForOp forOp) { loops.push_back(forOp); });
      for (scf::ForOp forOp : loops)
        pipelineLoop(forOp, numStages, useBarrier);
    }

    bool hasPipeline = false;
    bool hasBarrier = false;
    std::optional<int64_t> totalBytes = 0;
    WalkResult result = m.walk([&](scf::ForOp loop) {
      Attribute attr = loop->getAttr("ttig.pipelined_prefetch");
      if (!attr)
        return WalkResult::advance();
      auto info = dyn_cast<DictionaryAttr>(attr);
      auto barrier = info ? info.getAs<BoolAttr>("barrier") : BoolAttr();
      Attribute bytesAttr = info ? info.get("total_bytes") : Attribute();
      auto bytes = dyn_cast_or_null<IntegerAttr>(bytesAttr);
      if (!barrier || (bytesAttr && (!bytes || !bytes.getType().isInteger(64) ||
                                     bytes.getInt() < 0))) {
        loop.emitError("invalid ttig.pipelined_prefetch annotation");
        return WalkResult::interrupt();
      }
      hasPipeline = true;
      hasBarrier |= barrier.getValue();
      if (!bytes || !totalBytes)
        totalBytes.reset();
      else if (llvm::AddOverflow(*totalBytes, bytes.getInt(), *totalBytes))
        totalBytes.reset();
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
      return;
    }
    if (hasPipeline) {
      Builder builder(m.getContext());
      NamedAttrList info;
      info.set("barrier", builder.getBoolAttr(hasBarrier));
      if (totalBytes)
        info.set("total_bytes", builder.getI64IntegerAttr(*totalBytes));
      m->setAttr("ttig.pipelined_prefetch", info.getDictionary(m.getContext()));
    } else
      m->removeAttr("ttig.pipelined_prefetch");
  }
};
} // anonymous namespace
