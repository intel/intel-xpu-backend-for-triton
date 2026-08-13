#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONINTELGPUREDUCEDATADUPLICATION
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace {

constexpr unsigned kSlmCapBytes = 56u * 1024u;

/// Return the GRF budget in bytes per thread for 256-GRF mode.
/// (Conservative: assumes the auto-escalation has already happened, which is
/// true for the shapes that spill.)
unsigned getGRFBytesPerThread() { return 8192; }

/// Return the per-thread size in bytes for \p type.
unsigned getPerThreadSizeInBytes(RankedTensorType type) {
  Type elType = type.getElementType();
  if (!elType.isIntOrFloat())
    return 0;
  unsigned elemsPerThread = triton::gpu::getTotalElemsPerThread(type);
  return elemsPerThread * (elType.getIntOrFloatBitWidth() / 8);
}

/// Return the total live-in size in bytes for \p blockInfo.
unsigned getBlockLiveInSizeInBytes(const LivenessBlockInfo *blockInfo) {
  unsigned blockInSize = 0;
  for (Value liveVal : blockInfo->in()) {
    Type liveValTy = liveVal.getType();
    if (auto tensorType = dyn_cast<RankedTensorType>(liveValTy))
      blockInSize += getPerThreadSizeInBytes(tensorType);
    else if (liveValTy.isIntOrFloat())
      blockInSize += liveValTy.getIntOrFloatBitWidth() / 8;
  }
  return blockInSize;
}

/// Check if src and dst have different warp distributions (cross-warp).
bool isCrossWarpConversion(Attribute srcEnc, Attribute dstEnc,
                           ArrayRef<int64_t> shape) {
  auto srcWarps = triton::gpu::getWarpsPerCTA(srcEnc, shape);
  auto dstWarps = triton::gpu::getWarpsPerCTA(dstEnc, shape);
  return srcWarps != dstWarps;
}

class TritonIntelGPUReduceDataDuplicationPass
    : public intel::impl::TritonIntelGPUReduceDataDuplicationBase<
          TritonIntelGPUReduceDataDuplicationPass> {
public:
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    Liveness liveness(mod);
    unsigned grfBudget = getGRFBytesPerThread();

    // Pass 1: existing DotOperandEncodingAttr path (unchanged).
    mod.walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
      OpBuilder builder(cvtOp);
      auto srcType = cast<RankedTensorType>(cvtOp.getSrc().getType());
      auto dstType = cast<RankedTensorType>(cvtOp.getType());
      auto srcEncoding = srcType.getEncoding();
      if (isa<triton::gpu::SharedEncodingTrait>(srcEncoding))
        return;
      auto dstDotOp =
          dyn_cast<triton::gpu::DotOperandEncodingAttr>(dstType.getEncoding());
      if (!dstDotOp)
        return;
      if (!cvtNeedsSharedMemory(cvtOp))
        return;
      auto srcOrder = triton::gpu::getOrder(srcType);
      auto rank = srcOrder.size();
      if (auto srcDpasEncoding =
              dyn_cast<triton::gpu::intel::DpasEncodingAttr>(srcEncoding)) {
        auto opIdx =
            static_cast<intel::DpasEncodingAttr::OpIdx>(dstDotOp.getOpIdx());
        if ((opIdx == intel::DpasEncodingAttr::OpIdx::OperandA &&
             dstDotOp.getParent() == srcDpasEncoding &&
             srcDpasEncoding.getWarpsPerCTA()[rank - 1] == 1) ||
            (opIdx == intel::DpasEncodingAttr::OpIdx::OperandB &&
             dstDotOp.getParent() == srcDpasEncoding &&
             srcDpasEncoding.getWarpsPerCTA()[rank - 2] == 1))
          return;
      }
      SmallVector<unsigned> sharedOrder;
      if (rank == 3) {
        for (unsigned i = 0; i < rank; ++i)
          if (srcOrder[i] != 0)
            sharedOrder.emplace_back(srcOrder[i]);
        sharedOrder.emplace_back(0);
      } else {
        sharedOrder = std::move(srcOrder);
      }
      auto sharedMemorySpace =
          triton::gpu::SharedMemorySpaceAttr::get(srcType.getContext());
      auto tmpType = triton::gpu::MemDescType::get(
          dstType.getShape(), dstType.getElementType(),
          triton::gpu::SwizzledSharedEncodingAttr::get(
              mod.getContext(), dstDotOp, srcType.getShape(), sharedOrder,
              triton::gpu::getCGALayout(srcEncoding), srcType.getElementType()),
          sharedMemorySpace);
      auto tmp = triton::gpu::LocalAllocOp::create(builder, cvtOp.getLoc(),
                                                   tmpType, cvtOp.getSrc());
      auto newConvert = triton::gpu::LocalLoadOp::create(
          builder, cvtOp.getLoc(), dstType, tmp);
      cvtOp.replaceAllUsesWith(newConvert.getResult());
      cvtOp.erase();
    });

    // Pass 2: large in-loop cross-warp converts (new).
    SmallVector<triton::gpu::ConvertLayoutOp> cvtsToRoute;
    mod.walk([&](triton::gpu::ConvertLayoutOp cvtOp) {
      auto srcType = cast<RankedTensorType>(cvtOp.getSrc().getType());
      auto dstType = cast<RankedTensorType>(cvtOp.getType());
      auto srcEncoding = srcType.getEncoding();
      auto dstEncoding = dstType.getEncoding();

      // Skip if already shared, or if dst is DotOperandEncodingAttr (Pass 1).
      if (isa<triton::gpu::SharedEncodingTrait>(srcEncoding) ||
          isa<triton::gpu::DotOperandEncodingAttr>(dstEncoding))
        return;

      // Must need shared memory and be cross-warp.
      if (!cvtNeedsSharedMemory(cvtOp) ||
          !isCrossWarpConversion(srcEncoding, dstEncoding, srcType.getShape()))
        return;

      // Must be inside scf.for.
      auto parentForOp = cvtOp->getParentOfType<scf::ForOp>();
      if (!parentForOp || cvtOp->getParentRegion() != &parentForOp.getRegion())
        return;

      // GRF budget gate: per-thread size of src+dst plus live-in must stay
      // within 80% of budget.
      Block *bodyBlock = parentForOp.getBody();
      const LivenessBlockInfo *blockInfo = liveness.getLiveness(bodyBlock);
      if (!blockInfo)
        return;
      unsigned liveInBytes = getBlockLiveInSizeInBytes(blockInfo);
      unsigned cvtBytes =
          getPerThreadSizeInBytes(srcType) + getPerThreadSizeInBytes(dstType);
      if (liveInBytes + cvtBytes < grfBudget * 4 / 5)
        return; // Not large enough to justify SLM.

      // SLM cap gate: total tensor bytes (all threads) must fit in 56 KB.
      auto srcShape = srcType.getShape();
      int64_t totalElems = 1;
      for (int64_t dim : srcShape)
        totalElems *= dim;
      uint64_t totalBytes = totalElems * (srcType.getElementTypeBitWidth() / 8);
      if (totalBytes > kSlmCapBytes)
        return;

      cvtsToRoute.push_back(cvtOp);
    });

    // Route the collected converts through SLM.
    for (auto cvtOp : cvtsToRoute) {
      OpBuilder builder(cvtOp);
      auto srcType = cast<RankedTensorType>(cvtOp.getSrc().getType());
      auto dstType = cast<RankedTensorType>(cvtOp.getType());
      auto srcEncoding = srcType.getEncoding();
      auto srcOrder = triton::gpu::getOrder(srcType);
      auto rank = srcOrder.size();

      SmallVector<unsigned> sharedOrder;
      if (rank == 3) {
        for (unsigned i = 0; i < rank; ++i)
          if (srcOrder[i] != 0)
            sharedOrder.emplace_back(srcOrder[i]);
        sharedOrder.emplace_back(0);
      } else {
        sharedOrder = SmallVector<unsigned>(srcOrder.begin(), srcOrder.end());
      }

      auto sharedMemorySpace =
          triton::gpu::SharedMemorySpaceAttr::get(srcType.getContext());
      auto cgaLayout = triton::gpu::getCGALayout(srcEncoding);
      // Use SwizzledSharedEncodingAttr with no swizzling (1,1,1 = identity).
      auto sharedEnc = triton::gpu::SwizzledSharedEncodingAttr::get(
          mod.getContext(), /*vec=*/1, /*perPhase=*/1, /*maxPhase=*/1,
          sharedOrder, cgaLayout);
      auto tmpType = triton::gpu::MemDescType::get(
          dstType.getShape(), dstType.getElementType(), sharedEnc,
          sharedMemorySpace);
      auto tmp = triton::gpu::LocalAllocOp::create(builder, cvtOp.getLoc(),
                                                   tmpType, cvtOp.getSrc());
      auto newConvert = triton::gpu::LocalLoadOp::create(
          builder, cvtOp.getLoc(), dstType, tmp);
      cvtOp.replaceAllUsesWith(newConvert.getResult());
      cvtOp.erase();
    }
  }
};

} // namespace
