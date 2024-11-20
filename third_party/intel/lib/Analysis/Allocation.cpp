#include "intel/include/Analysis/Allocation.h"

#include "llvm/ADT/TypeSwitch.h"

#include "triton/Dialect/Triton/IR/Utility.h"

#include "intel/include/Analysis/Utility.h"

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
    return product(srcTy.getShape()) * bytesPerElement;
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
