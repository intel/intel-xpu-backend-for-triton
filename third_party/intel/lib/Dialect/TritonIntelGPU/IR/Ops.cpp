//===- Ops.cpp - TritonIntelGPU Operations ------------------------------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"

#include "mlir/IR/Builders.h"
#include "mlir/Support/LLVM.h"

#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LinearLayout.h"

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
static LogicalResult verifyGatherScatterResultType(Operation *op,
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

static LogicalResult verifyGatherScatterOp(Operation *op, ShapedType blockType,
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
  return intel::verifyGatherScatterOp(
      *this, getDesc().getType().getSignlessBlockType(), getSrc().getType(),
      getXOffsets().getType());
}

// ---- UpcastScaledOp --------------------------------------------------------

// Derive the compact scale encoding from the src encoding, or {} on failure.
Attribute deriveScaleEncoding(Attribute srcEnc, ArrayRef<int64_t> srcShape,
                              int64_t axis, int64_t scaleFactor) {
  // Only distributed layouts have register/lane/warp/block in-dims. Note that
  // `LayoutEncodingTrait` is *also* implemented by the shared encodings, whose
  // linear layout has an `offset` in-dim -- it is the wrong predicate here.
  if (!srcEnc || !isa<gpu::DistributedEncodingTrait>(srcEnc))
    return {};
  LinearLayout srcLL = gpu::toLinearLayout(srcShape, srcEnc);
  auto scaleLL = UpcastScaledOp::computeScaleLayout(srcLL, axis, scaleFactor);
  if (!scaleLL)
    return {};
  MLIRContext *ctx = srcEnc.getContext();
  // Drop broadcast register bases: one register per distinct scale value.
  // (Stripping before vs. after the compose gives the same result, because
  // compose never turns a zero basis into a non-zero one.)
  LinearLayout compact =
      scaleLL->removeZeroBasesAlongDim(StringAttr::get(ctx, "register"));
  // `LinearEncodingAttr` additionally requires a permutation-matrix layout;
  // bail rather than build an invalid attribute.
  if (!gpu::isGenericLinearEncoding(srcEnc) &&
      !gpu::isPermutationMatrixLayout(compact))
    return {};
  return gpu::inferEncodingFromLinearLayout(ctx, std::move(compact), srcEnc);
}

void UpcastScaledOp::build(OpBuilder &builder, OperationState &state, Value src,
                           Value scale, int32_t axis, int32_t scaleFactor,
                           bool fastMath) {
  build(builder, state, src.getType(), src, scale,
        builder.getI32IntegerAttr(axis), builder.getI32IntegerAttr(scaleFactor),
        fastMath ? builder.getUnitAttr() : UnitAttr{});
}

LogicalResult UpcastScaledOp::verify() {
  auto srcTy = cast<RankedTensorType>(getSrc().getType());
  auto scaleTy = cast<RankedTensorType>(getScale().getType());
  auto resTy = cast<RankedTensorType>(getResult().getType());
  int32_t axis = getAxis();
  int32_t sf = getScaleFactor();

  if (srcTy.getRank() != resTy.getRank())
    return emitOpError("src and result must have the same rank");
  if (srcTy.getRank() != scaleTy.getRank())
    return emitOpError("src and scale must have the same rank");
  if (axis < 0 || axis >= (int32_t)srcTy.getRank())
    return emitOpError("axis out of range");
  if (sf <= 0)
    return emitOpError("scale_factor must be positive");
  // Along the scale axis: src.shape[axis] must equal
  // scale.shape[axis]*scale_factor.
  if (srcTy.getDimSize(axis) != scaleTy.getDimSize(axis) * sf)
    return emitOpError("src.shape[axis] (")
           << srcTy.getDimSize(axis) << ") must equal scale.shape[axis] ("
           << scaleTy.getDimSize(axis) << ") * scale_factor (" << sf << ")";
  // All other dims must match between src and scale.
  for (int i = 0, r = srcTy.getRank(); i < r; ++i) {
    if (i == axis)
      continue;
    if (srcTy.getDimSize(i) != scaleTy.getDimSize(i))
      return emitOpError("src and scale must have equal size on non-axis dim ")
             << i;
  }

  // Verify src and result match, and src element type is bf16.
  if (srcTy.getElementType() != resTy.getElementType() ||
      srcTy.getShape() != resTy.getShape())
    return emitOpError(
        "src and result must have the same shape and element type");
  if (!srcTy.getElementType().isBF16())
    return emitOpError("only bf16 src is supported (E8M0 exponent-add relies "
                       "on bf16's 8-bit exponent field)");

  // Verify scale layout is compatible with the derived encoding.
  Attribute srcEnc = srcTy.getEncoding();
  Attribute scaleEnc = scaleTy.getEncoding();
  if (static_cast<bool>(srcEnc) != static_cast<bool>(scaleEnc))
    return emitOpError("src and scale must both have an encoding, or neither");
  if (!srcEnc)
    return success();

  Attribute expectedEnc =
      deriveScaleEncoding(srcEnc, srcTy.getShape(), axis, sf);
  if (!expectedEnc)
    return emitOpError("cannot derive a scale layout from the src encoding");

  auto kRegister = StringAttr::get(getContext(), "register");
  LinearLayout expected = gpu::toLinearLayout(scaleTy.getShape(), expectedEnc)
                              .removeZeroBasesAlongDim(kRegister);
  LinearLayout actual = gpu::toLinearLayout(scaleTy.getShape(), scaleEnc)
                            .removeZeroBasesAlongDim(kRegister);
  if (expected != actual)
    return emitOpError("scale encoding is incompatible with the scale layout "
                       "derived from the src encoding");

  return success();
}

std::optional<LinearLayout>
UpcastScaledOp::computeScaleLayout(const LinearLayout &srcLayout, int64_t axis,
                                   int64_t scaleFactor) {
  // For identity scales (scaleFactor == 1), the scale layout is the same as
  // src.
  if (scaleFactor == 1)
    return srcLayout;
  if (scaleFactor <= 0 || !llvm::isPowerOf2_64(scaleFactor))
    return std::nullopt;

  auto ctx = srcLayout.getOutDimNames().begin()->getContext();
  auto axisDim = StringAttr::get(ctx, llvm::formatv("dim{0}", axis).str());

  if (!srcLayout.hasOutDim(axisDim) ||
      srcLayout.getOutDimSize(axisDim) % scaleFactor != 0)
    return std::nullopt;

  // Build a divisor map from src coordinates to scale coordinates that
  // floor-divides (right shift) the scaled axis by scaleFactor and
  // leaves every other dimension unchanged (identity).
  LinearLayout scaleDivisor = LinearLayout::empty();
  for (StringAttr outDim : srcLayout.getOutDimNames()) {
    int32_t size = srcLayout.getOutDimSize(outDim);
    if (outDim != axisDim) {
      scaleDivisor *= LinearLayout::identity1D(size, outDim, outDim);
      continue;
    }

    // floor-divide the scaled axis by scaleFactor: drop the low
    // log2(scaleFactor) bits, shift the remaining bits down.
    scaleDivisor *=
        LinearLayout::zeros1D(scaleFactor, outDim, outDim) *
        LinearLayout::identity1D(size / scaleFactor, outDim, outDim);
  }

  // Compose the divisor with the src layout to get the scale layout. For
  // each input it yields the scale coordinate that the output element at that
  // input needs.
  return srcLayout.compose(scaleDivisor);
}

// UpcastFpOpInterface: encoding propagation with scale derivation.
Attribute UpcastScaledOp::inferDstEncoding(unsigned opIdx, Attribute srcEnc) {
  // Only src -> result is a well-defined identity; the scale layout is a
  // quotient (broadcasting), so it cannot be inverted to a unique result.
  return opIdx == 0 ? srcEnc : Attribute();
}

Attribute UpcastScaledOp::inferSrcEncoding(unsigned opIdx, Attribute dstEnc) {
  if (opIdx == 0)
    return dstEnc;
  assert(opIdx == 1 && "upcast_scaled has two operands");
  // src and result have identical shapes, so dstEnc is also the src layout.
  return deriveScaleEncoding(dstEnc, getSrc().getType().getShape(), getAxis(),
                             getScaleFactor());
}

} // namespace mlir::triton::gpu::intel
