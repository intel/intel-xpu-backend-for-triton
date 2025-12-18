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

LogicalResult GlueOp::verify() {
  SmallVector<Type> inputTypes;
  for (Value input : getOperands())
    inputTypes.push_back(input.getType());

  Type inputType = inputTypes.front();
  Type resultType = getRes().getType();
  if (!llvm::all_of(inputTypes, [&](Type type) { return type == inputType; }))
    return emitOpError("operands must have the same type");

  if (!isTensorOrTensorPointerType(inputType))
    return success();

  unsigned resultRank = getRank(resultType);
  if (llvm::any_of(inputTypes, [&](Type type) {
        for (unsigned i = 0; i < resultRank; ++i) {
          unsigned resultSize = getDimSize(resultType, i);
          unsigned inputSize = getDimSize(type, i);
          if (inputSize > resultSize)
            return true;
        }
        return false;
      }))
    return emitOpError(
        "operands cannot exceed result size along any dimension");

  for (unsigned i = 0; i < resultRank; ++i) {
    unsigned resultSize = getDimSize(resultType, i);
    unsigned inputSize = getDimSize(inputType, i);
    if (resultSize % inputSize != 0)
      return emitOpError("operands cannot be glued along axis ") << i;
  }

  // Verify that the composition of the input operands covers the output tensor
  // shape.
  SmallVector<int64_t> inputShape = getShape(inputTypes[0]);
  SmallVector<int64_t> resultShape = getShape(resultType);
  unsigned numResultElems = product(resultShape);
  unsigned numInputElems = product(inputShape);

  if (inputTypes.size() * numInputElems != numResultElems)
    return emitOpError("glued operands do not exactly cover the result shape");
  return success();
}

LogicalResult ExtractOp::verify() {
  Type resultType = getRes().getType();
  Type operandType = getBase().getType();

  Type resultElemType = getElementType(resultType);
  Type operandElemType = getElementType(operandType);
  if (operandElemType != resultElemType)
    return emitOpError("operand and result element type must match");

  if (!isTensorOrTensorPointerType(operandType))
    return success();

  unsigned resultRank = getRank(resultType);
  unsigned operandRank = getRank(operandType);
  if (resultRank > operandRank)
    return emitOpError("result rank cannot be greater than operand rank");

  SmallVector<int64_t> resultShape = getShape(resultType);
  SmallVector<int64_t> operandShape = getShape(operandType);

  // Make the result have the same rank as the operand.
  while (resultRank < operandRank) {
    resultShape.insert(resultShape.begin(), operandRank - resultRank, 1);
    ++resultRank;
  }
  assert(operandRank == resultRank && "Expecting same rank");

  // Ensure the input can be partitioned by the requested result.
  unsigned i = 0;
  for (auto [resDim, operandDim] : zip(resultShape, operandShape)) {
    if (operandDim < resDim)
      return emitOpError("operand shape cannot be smaller than result shape ")
             << "along dimension " << i;
    if (operandDim % resDim != 0)
      return emitOpError("operands shape is not divisible by result shape ")
             << "along dimension " << i;
    ++i;
  }

  unsigned numTiles = 1;
  for (auto [resDim, operandDim] : zip(resultShape, operandShape))
    numTiles *= (operandDim / resDim);

  unsigned index = getIndex();
  if (index >= numTiles)
    return emitOpError("index must be less than ") << numTiles;

  return success();
}

OpFoldResult ExtractOp::fold(FoldAdaptor adaptor) {
  // extract (glue %t1, %t2)[1] -> %t2
  if (auto glueOp = getBase().getDefiningOp<GlueOp>()) {
    auto operand = glueOp->getOperand(getIndex());
    if (operand.getType() == getType())
      return operand;
  }

  // %0 =  .... : tensor<16x8xf16>
  // extract %0[0] : tensor<16x8xf16> -> %0
  if (getIndex() == 0 && getBase().getType() == getType())
    return getBase();

  return {};
}

void PrefetchOp::build(OpBuilder &builder, OperationState &state, Value ptr,
                       CacheModifier cache, EvictionPolicy evict,
                       bool isVolatile) {
  PrefetchOp::build(builder, state, ptr, /*mask=*/{}, cache, evict, isVolatile);
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

void LoadTdescOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get());
}

void StoreTdescOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get());
}

} // namespace mlir::triton::gpu::intel
