#include "intel/include/Analysis/Allocation.h"
#include "intel/include/Analysis/Utility.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir::triton::intel {
namespace {
constexpr int kPtrBitWidth = 64;

std::pair<unsigned, unsigned> getNumScratchElemsAndRepsSwizzledCvt(
    RankedTensorType srcTy, RankedTensorType dstTy, int numBanks = 32,
    gpu::LocalMemOpTile srcTile = {}, gpu::LocalMemOpTile dstTile = {}) {
  auto srcLayout = gpu::toLinearLayout(srcTy);
  auto dstLayout = gpu::toLinearLayout(dstTy);
  auto *ctx = srcLayout.getInDimNames().begin()->getContext();
  auto srcLayoutNoBroadcast =
      actionRemoveBroadcastedRegs(srcLayout).apply(srcLayout);
  auto dstLayoutNoBroadcast =
      actionRemoveBroadcastedRegs(dstLayout).apply(dstLayout);
  auto smem =
      gpu::optimalSwizzlingLdSt(srcLayoutNoBroadcast, dstLayoutNoBroadcast,
                                getBitwidth(srcTy), numBanks, srcTile, dstTile);
  auto reps = smem.getInDimSize(StringAttr::get(ctx, "reps"));
  auto nBlocks = product(triton::gpu::getCTASplitNum(
      gpu::LinearEncodingAttr::get(ctx, srcLayout)));
  unsigned elemsPerRep = smem.getTotalOutDimSize() / (reps * nBlocks);
  return {elemsPerRep, static_cast<unsigned>(reps)};
}

unsigned allocationAnalysisScratchSizeFn(gpu::ConvertLayoutOp convertLayout) {
  RankedTensorType srcTy = convertLayout.getSrc().getType();
  RankedTensorType dstTy = convertLayout.getResult().getType();
  if (gpu::intel::cvtIsSubGroupShuffle(srcTy, dstTy))
    return 0;
  if (gpu::intel::cvtIsSubGroupTranspose(srcTy, dstTy)) {
    Type elemTy = srcTy.getElementType();
    unsigned bytesPerElement =
        isa<PointerType>(elemTy)
            ? kPtrBitWidth / 8
            : std::max<int>(8, elemTy.getIntOrFloatBitWidth()) / 8;
    unsigned numElements = product(srcTy.getShape());
    Attribute encoding = srcTy.getEncoding();
    int subGroupSize =
        product(gpu::getThreadsPerWarp(encoding, srcTy.getShape()));
    assert(numElements % subGroupSize == 0 &&
           "Sub-group transposable tensors have a number of elements multiple "
           "of the sub-group size");
    // Add an element at the end of the row that will not be accessed. This
    // allows us to avoid bank conflicts.
    unsigned numMatrixCells = (numElements / subGroupSize) * (subGroupSize + 1);
    return numMatrixCells * bytesPerElement;
  }

  if (!cvtNeedsSharedMemory(srcTy, dstTy))
    return 0;

  // For the generic swizzled path, the lowering writes each rep to a disjoint
  // SLM slice (offset = slot * elemsPerRep) so all reps in a batch can be
  // stored before a single barrier.  The lowering processes reps in chunks of
  // batchReps chosen so that batchReps * bytesPerRep fits in the device's
  // local memory (ttig.local_mem_size), matching
  // TargetInfo::getMaxSLMBytesForSwizzledCvt().
  auto [elemsPerRep, reps] = getNumScratchElemsAndRepsSwizzledCvt(srcTy, dstTy);
  unsigned bytesPerRep = elemsPerRep * getBitwidth(srcTy) / 8;
  if (bytesPerRep == 0)
    return 0;
  // Read the local memory size from the module attribute set by AnnotateModule.
  // Falls back to 131072 (128 KB, typical Intel PVC/BMG) if not present.
  unsigned maxSLMBytes = 131072;
  if (auto mod = convertLayout->getParentOfType<ModuleOp>())
    if (auto attr = mod->getAttrOfType<IntegerAttr>(
            gpu::intel::TritonIntelGPUDialect::getLocalMemSizeAttrName()))
      maxSLMBytes = attr.getInt();
  unsigned batchReps = std::min(reps, maxSLMBytes / bytesPerRep);
  batchReps = std::max(1u, batchReps);
  return bytesPerRep * batchReps;
}
} // namespace

unsigned allocationAnalysisScratchSizeFn(Operation *op) {
  return TypeSwitch<Operation *, unsigned>(op)
      .Case<gpu::ConvertLayoutOp>(
          [](auto op) { return allocationAnalysisScratchSizeFn(op); })
      .Case<ReduceOp>(
          [](auto op) { return ReduceOpHelper(op).getScratchSizeInBytesOld(); })
      .Default([](Operation *op) {
        return defaultAllocationAnalysisScratchSizeFn(op);
      });
}
} // namespace mlir::triton::intel
