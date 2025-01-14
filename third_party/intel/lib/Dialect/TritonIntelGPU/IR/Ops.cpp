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

LogicalResult UpcastMXFPOp::verify() {
  auto fpType = getFpType();

  auto xTy = getSrc().getType();
  auto scaleTy = getScale().getType();
  Builder b(getContext());
  if (xTy.getElementType() != b.getBF16Type() &&
      xTy.getElementType() != b.getF16Type() &&
      xTy.getElementType() != b.getI8Type()) {
    return emitOpError(
        "element type of the first operand must be bf16/fp16 or i8");
  }

  if (scaleTy.getElementType() != b.getI8Type()) {
    return emitOpError("element type of the second operand must be uint8");
  }

  auto xShape = xTy.getShape();
  auto scaleShape = scaleTy.getShape();

  if (xShape.size() != scaleShape.size() || xShape.size() < 2) {
    return emitOpError(
        "operands must have the same number of dimensions, at least 2");
  }

  if (!(fpType == ScaleDotElemType::E2M1 || fpType == ScaleDotElemType::E4M3 ||
        fpType == ScaleDotElemType::E5M2)) {
    return emitOpError("NYI: fpType must be E2M1, E4M3, or E5M2");
  }

  auto layoutX = xTy.getEncoding();
  auto layoutScale = scaleTy.getEncoding();
  if (bool(layoutX) != bool(layoutScale)) {
    return emitOpError(
        "Expected either both or neither operands to have an encoding");
  }
  // Nothing to check if no encoding. This is used to infer the return type in
  // AccelerateMatmul.cpp
  if (!layoutX) {
    return success();
  }

  auto dotEncoding = dyn_cast<DotOperandEncodingAttr>(layoutX);
  if (!dotEncoding) {
    return emitOpError("Expected a DotOperandEncodingAttr for values");
  }
  if (!isa<BlockedEncodingAttr, LinearEncodingAttr>(layoutScale)) {
    return emitOpError(
        "Expected a BlockOperandEncoding or LinearOperandEncoding "
        "for scales");
  }

  // Change to support fp8 types
  const auto elemsPacked = fpType == ScaleDotElemType::E2M1 ? 2 : 1;
  // Figure out the K dimension for the input A/B. For A/B scale, the K
  // dimension is always the last dimension.
  const int opIdx = dotEncoding.getOpIdx();
  const bool hasBatch = xShape.size() == 3;
  const int kIdx = (opIdx == 0 ? 1 : 0) + hasBatch;

  if (xShape[kIdx] != (32 / elemsPacked) * scaleShape.back()) {
    return emitOpError("K dimension of first operand must be 16 times "
                       "larger than last/K dimension of the second operand");
  }

  // Check other dimensions match too. For input A/B, we need to figure out the
  // index for the M/N dimension. For scale, it's always {(batch), M/N, K}.
  const int mnIdx = (opIdx == 0 ? 0 : 1) + hasBatch;
  if (hasBatch && xShape[0] != scaleShape[0])
    return emitOpError("batch dimension must match between operands");
  if (xShape[mnIdx] != scaleShape[hasBatch]) {
    return emitOpError("M/N dimension must match between operands");
  }

  return success();
}

RankedTensorType
UpcastMXFPOp::deduceOutputType(TypedValue<RankedTensorType> inputTensor,
                               ScaleDotElemType inputElemType,
                               Type outputElemType) {
  MLIRContext *ctx = inputTensor.getContext();
  auto xTy = inputTensor.getType();
  if (inputElemType != ScaleDotElemType::E2M1)
    return xTy;

  auto xShape = xTy.getShape();
  auto newShape = llvm::to_vector(xShape);
  auto encoding = xTy.getEncoding();
  if (!encoding) {
    newShape.back() *= 2;
    return RankedTensorType::get(xShape, outputElemType);
  }

  auto oldEncoding = cast<DotOperandEncodingAttr>(encoding);
  const int opIdx = oldEncoding.getOpIdx();
  // Note: For Intel the dot operands layout's kWidth parameter must match
  // the parent's DPAS layout opsPerChannel so we need to materialize a
  // new DPAS layout.
  auto dpasEncoding = cast<intel::DpasEncodingAttr>(oldEncoding.getParent());
  unsigned opsPerChannel =
      intel::DpasEncodingAttr::getOpsPerChannel(outputElemType);
  // e2m1 is packed 2 elements per int8, we must handle continuous 2
  // elements when upcasting to bf16
  if (xTy.getElementType() == IntegerType::get(ctx, 8))
    opsPerChannel *= 2;
  auto newDpasEncoding = intel::DpasEncodingAttr::get(
      ctx, dpasEncoding.getRepeatCount(), dpasEncoding.getSystolicDepth(),
      dpasEncoding.getExecutionSize(), opsPerChannel,
      dpasEncoding.getWarpsPerCTA(), dpasEncoding.getRepCluster(),
      product<unsigned>(dpasEncoding.getThreadsPerWarp()));
  Attribute newVEncoding = DotOperandEncodingAttr::get(
      ctx, opIdx, newDpasEncoding, newDpasEncoding.getOpsPerChannel());

  const bool hasBatch = xShape.size() == 3;
  const int kIdx = (opIdx == 0 ? 1 : 0) + hasBatch;
  newShape[kIdx] *= 2;
  return RankedTensorType::get(newShape, outputElemType, newVEncoding);
}

} // namespace mlir::triton::gpu::intel
