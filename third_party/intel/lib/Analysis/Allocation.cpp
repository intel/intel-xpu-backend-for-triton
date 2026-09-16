#include "intel/include/Analysis/Allocation.h"
#include "intel/include/Analysis/Utility.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Utils.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "llvm/ADT/TypeSwitch.h"

namespace ttgi = mlir::triton::gpu::intel;

namespace mlir::triton::intel {
namespace {
constexpr int kPtrBitWidth = 64;
constexpr unsigned invalidSize = -1;

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
  if (gpu::intel::cvtIsSubGroupReinterpret(convertLayout))
    return 0;
  return invalidSize;
}
} // namespace

unsigned allocationAnalysisScratchSizeFn(Operation *op) {
  return TypeSwitch<Operation *, unsigned>(op)
      .Case<gpu::ConvertLayoutOp>([](auto op) {
        unsigned size = allocationAnalysisScratchSizeFn(op);
        return size == invalidSize ? defaultAllocationAnalysisScratchSizeFn(op)
                                   : size;
      })
      .Case<ReduceOp>([](auto op) -> unsigned {
        // FIXME: issue #6719 A/B scaffolding. Must stay in lockstep with the
        // pattern selection in TritonIntelGPUToLLVM/PipelineManager.h: the
        // common lowering computes its shared-memory offsets from
        // getScratchSizeInBytes(), the Intel one from the legacy shape-based
        // size, and the two are unordered.
        ReduceOpHelper helper(op);
        unsigned oldSize = ttgi::getScratchSizeInBytesOld(helper, op);
        unsigned newSize = defaultAllocationAnalysisScratchSizeFn(op);

        // FIXME: issue #6719 step-0 scaffolding; strip before the PR. Records
        // the sign of old - new to establish whether the two sizes really are
        // unordered.
        if (::getenv("TRITON_INTEL_REDUCE_DEBUG_COUNTS"))
          llvm::errs() << "[reduce-6719-scratch] old=" << oldSize
                       << " new=" << newSize << " sign="
                       << (oldSize < newSize
                               ? "old<new"
                               : (newSize < oldSize ? "new<old" : "equal"))
                       << " loc=" << op.getLoc() << "\n";

        return ttgi::useCommonReduceLowering() ? newSize : oldSize;
      })
      .Default([](Operation *op) {
        return defaultAllocationAnalysisScratchSizeFn(op);
      });
}
} // namespace mlir::triton::intel
