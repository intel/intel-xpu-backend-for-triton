//===- Ops.cpp - TritonIntelGPU Operations ------------------------------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlir/IR/Builders.h"
#include "mlir/Support/LLVM.h"

#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"

#include "intel/include/Dialect/TritonGEN/IR/TritonGENMemorySpace.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"

#define GET_OP_CLASSES
#include "intel/include/Dialect/TritonIntelGPU/IR/Ops.cpp.inc"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

/// Return the rank of an input tensor (or ptr to tensor).
static unsigned getRank(Type type) {
  return TypeSwitch<Type, unsigned>(type)
      .Case<RankedTensorType>([](auto ty) { return ty.getRank(); })
      .Case<triton::PointerType>([](auto ty) {
        assert(isa<RankedTensorType>(ty.getPointeeType()) &&
               "Expecting ptr to tensor");
        return cast<RankedTensorType>(ty.getPointeeType()).getRank();
      })
      .Default([](auto) {
        llvm_unreachable("Unexpected type");
        return 0;
      });
}

/// Return the shape of an input tensor (or ptr to tensor).
static SmallVector<int64_t> getShape(Type type) {
  return TypeSwitch<Type, SmallVector<int64_t>>(type)
      .Case<RankedTensorType>([](auto ty) { return ty.getShape(); })
      .Case<triton::PointerType>([](auto ty) {
        assert(isa<RankedTensorType>(ty.getPointeeType()) &&
               "Expecting ptr to tensor");
        return cast<RankedTensorType>(ty.getPointeeType()).getShape();
      })
      .Default([](auto) {
        llvm_unreachable("Unexpected type");
        return SmallVector<int64_t>();
      });
}

/// Return the element type of an input tensor (or ptr to tensor).
static Type getElementType(Type type) {
  return TypeSwitch<Type, Type>(type)
      .Case<ShapedType>([](auto ty) { return ty.getElementType(); })
      .Case<triton::PointerType>([](auto ty) {
        assert(isa<RankedTensorType>(ty.getPointeeType()) &&
               "Expecting ptr to tensor");
        return cast<RankedTensorType>(ty.getPointeeType()).getElementType();
      })
      .Default([](auto ty) { return ty; });
}

/// Return the size of the specified dimension of an input tensor (or ptr to
/// tensor).
static unsigned getDimSize(Type type, unsigned dim) {
  return TypeSwitch<Type, unsigned>(type)
      .Case<RankedTensorType>([dim](auto ty) { return ty.getDimSize(dim); })
      .Case<triton::PointerType>([dim](auto ty) {
        assert(isa<RankedTensorType>(ty.getPointeeType()) &&
               "Expecting ptr to tensor");
        return cast<RankedTensorType>(ty.getPointeeType()).getDimSize(dim);
      })
      .Default([](auto) {
        llvm_unreachable("Unexpected type");
        return 0;
      });
}

namespace mlir::triton::gpu::intel {

void PrefetchOp::build(OpBuilder &builder, OperationState &state, Value ptr,
                       CacheModifier cache, EvictionPolicy evict,
                       bool isVolatile) {
  PrefetchOp::build(builder, state, ptr, /*mask=*/{}, cache, evict, isVolatile);
}

Value PrefetchOp::getPredicateOperand() { return getMask(); }
void PrefetchOp::setPredicateOperand(Value pred) {
  getMaskMutable().assign(pred);
}
Type PrefetchOp::getPredicateOperandTypeLike() { return getPtr().getType(); }

LogicalResult DescriptorPrefetchOp::verify() {
  auto descType = getDesc().getType();
  unsigned blockRank = descType.getBlockType().getRank();
  if (getIndices().size() != blockRank) {
    return emitOpError("expected ")
           << blockRank << " indices, but got " << getIndices().size();
  }
  return success();
}

LogicalResult SubGroupTransposeOp::verify() {
  RankedTensorType srcType = getSrc().getType();
  auto mod = getOperation()->getParentOfType<mlir::ModuleOp>();
  int64_t subGroupSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
  std::array requiredShape{subGroupSize, subGroupSize};
  if (srcType.getEncoding() ||
      srcType.getShape() != ArrayRef<int64_t>(requiredShape))
    return emitOpError("can only be used on tensors of shape <sub_group_size x "
                       "sub_group_size> with no encoding");
  return success();
}

LogicalResult Subgroup2DBlockLoadOp::verify() {
  auto resultType = dyn_cast<RankedTensorType>(getResult().getType());
  if (!resultType)
    return emitOpError("result must be a ranked tensor type");

  if (resultType.getRank() < 2)
    return emitOpError("result tensor must have rank >= 2, got ")
           << resultType.getRank();

  return success();
}

LogicalResult Subgroup2DBlockLoadFromPtrOp::verify() {
  auto resultType = dyn_cast<RankedTensorType>(getResult().getType());
  if (!resultType)
    return emitOpError("result must be a ranked tensor type");

  if (resultType.getRank() < 2)
    return emitOpError("result tensor must have rank >= 2, got ")
           << resultType.getRank();

  if (getMask() && !getOther())
    return emitOpError("'other' must be present when 'mask' is present");

  return success();
}

// -- DescriptorGatherOp
LogicalResult verifyGatherScatterResultType(Operation *op,
                                            ShapedType resultType,
                                            ShapedType indicesType) {
  if (indicesType.getRank() != 1)
    return op->emitOpError("x offsets must be a 1D tensor, but got ")
           << indicesType;
  if (resultType.getRank() != 2)
    return op->emitOpError("result must be a 2D tensor, but got ")
           << resultType;

  Type dtype = resultType.getElementType();
  unsigned bitWidth = dtype.getIntOrFloatBitWidth();
  if (bitWidth > 64)
    return op->emitOpError("dtype cannot be greater than 64 bits");

  unsigned minCols = 256 / bitWidth;
  if (unsigned cols = resultType.getShape()[1]; cols < minCols) {
    return op->emitOpError("must have at least ")
           << minCols << " columns for " << dtype << ", but got " << cols;
  }

  if (resultType.getShape()[0] != indicesType.getShape()[0]) {
    return op->emitOpError("result tensor must have as many rows as indices (")
           << indicesType.getShape()[0] << "), but got " << resultType;
  }

  return success();
}

LogicalResult verifyGatherScatterOp(Operation *op, ShapedType blockType,
                                    ShapedType resultType,
                                    ShapedType indicesType) {
  // Gather from `!tt.tensordesc<1xMxdtype>`.
  if (blockType.getRank() != 2) {
    return op->emitOpError("descriptor block must be a 2D tensor, but got ")
           << blockType;
  }
  if (blockType.getShape()[0] != 1) {
    return op->emitOpError("descriptor block must have exactly 1 row, but got ")
           << blockType;
  }

  // With x offsets `tensor<Nxinttype>` into `tensor<NxMxdtype>`.
  if (failed(verifyGatherScatterResultType(op, resultType, indicesType)))
    return failure();

  if (resultType.getShape()[1] != blockType.getShape()[1]) {
    return op->emitOpError("result tensor number of columns must match block (")
           << blockType.getShape()[1] << "), but got " << resultType;
  }
  if (resultType.getElementType() != blockType.getElementType()) {
    return op->emitOpError("result tensor element type must match block (")
           << blockType.getElementType() << "), but got " << resultType;
  }

  return success();
}

LogicalResult DescriptorGatherOp::verify() {
  return intel::verifyGatherScatterOp(
      *this, getDesc().getType().getSignlessBlockType(), getResult().getType(),
      getXOffsets().getType());
}

// -- DescriptorScatterOp --
LogicalResult DescriptorScatterOp::verify() {
  return verifyGatherScatterOp(
      *this, getDesc().getType().getSignlessBlockType(), getSrc().getType(),
      getXOffsets().getType());
}

} // namespace mlir::triton::gpu::intel
