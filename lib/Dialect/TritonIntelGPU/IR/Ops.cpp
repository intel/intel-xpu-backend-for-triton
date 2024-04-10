//===- Ops.cpp - TritonIntelGPU Operations ------------------------------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonIntelGPU/IR/Dialect.h"

#define GET_OP_CLASSES
#include "triton/Dialect/TritonIntelGPU/IR/Ops.cpp.inc"

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
      .Case<RankedTensorType>([](auto ty) { return ty.getElementType(); })
      .Case<triton::PointerType>([](auto ty) {
        assert(isa<RankedTensorType>(ty.getPointeeType()) &&
               "Expecting ptr to tensor");
        return cast<RankedTensorType>(ty.getPointeeType()).getElementType();
      })
      .Default([](auto) {
        llvm_unreachable("Unexpected type");
        return Type();
      });
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

namespace mlir {
namespace triton {
namespace gpu {
namespace intel {

LogicalResult ConcatOp::verify() {
  if (getOperands().size() < 2)
    return emitOpError("requires at least 2 operands");

  SmallVector<Type> inputTypes;
  for (auto input : getOperands())
    inputTypes.push_back(input.getType());

  Type resultType = getRes().getType();
  unsigned resultRank = getRank(resultType);
  if (llvm::any_of(inputTypes,
                   [&](Type type) { return getRank(type) != resultRank; }))
    return emitOpError("rank of concatenated operands must match result rank");

  unsigned dim = getDim();
  if (dim >= resultRank)
    return emitOpError("concatenation dim must be less than the tensor rank");

  Type resultElementType = getElementType(resultType);
  if (llvm::any_of(inputTypes, [&](Type type) {
        return getElementType(type) != resultElementType;
      }))
    return emitOpError("operands and result element type must match");

  SmallVector<int64_t> sizes(resultRank);
  for (unsigned i = 0; i < resultRank; ++i) {
    if (i == dim)
      continue;
    SaturatedInteger size;
    for (auto inputType : inputTypes) {
      FailureOr<SaturatedInteger> maybeSize =
          size.desaturate(SaturatedInteger::wrap(getDimSize(inputType, i)));
      if (failed(maybeSize))
        return emitOpError("static concatenation size mismatch along ")
               << "non-concatenated dimension " << i;
      size = *maybeSize;
    }
    sizes[i] = size.asInteger();
  }

  return success();
}

LogicalResult ExtractOp::verify() {
  Type resultType = getRes().getType();
  Type operandType = getOperand().getType();

  unsigned resultRank = getRank(resultType);
  unsigned operandRank = getRank(operandType);
  if (operandRank != resultRank)
    return emitOpError("operand and result must have the same rank");

  Type resultElemType = getElementType(resultType);
  Type operandElemType = getElementType(operandType);
  if (operandElemType != resultElemType)
    return emitOpError("operand and result element type must match");

  // ensure the input can be partitioned by the requested result.
  SmallVector<int64_t> resultShape = getShape(resultType);
  SmallVector<int64_t> operandShape = getShape(operandType);

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

} // namespace intel
} // namespace gpu
} // namespace triton
} // namespace mlir
