#include "intel/include/Analysis/Allocation.h"
#include "intel/include/Analysis/Utility.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Utility.h" // isBlockIONoOpConversion
#include "triton/Dialect/Triton/IR/Utility.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir::triton::intel {
namespace {
constexpr int kPtrBitWidth = 64;
constexpr unsigned invalidSize = -1;

unsigned allocationAnalysisScratchSizeFn(gpu::ConvertLayoutOp convertLayout) {
  RankedTensorType srcTy = convertLayout.getSrc().getType();
  RankedTensorType dstTy = convertLayout.getResult().getType();

  if (gpu::intel::isBlockIONoOpConversion(srcTy, dstTy))
    return 0;
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
      .Default([](Operation *op) {
        return defaultAllocationAnalysisScratchSizeFn(op);
      });
}
} // namespace mlir::triton::intel
