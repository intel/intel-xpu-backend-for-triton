//===- TritonGENOps.cpp - TritonGEN dialect operations --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "intel/include/Dialect/TritonGEN/IR/TritonGENDialect.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/ADT/STLExtras.h"
#include <cstdint>

#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

template <typename Op>
static LogicalResult verify2DBlockAddressPayloadRestriction(Op op) {
  static_assert(llvm::is_one_of<Op, TritonGEN::Matrix2DBlockLoadOp,
                                TritonGEN::Matrix2DBlockStoreOp,
                                TritonGEN::Matrix2DBlockPrefetchOp>::value,
                "Unexpected template parameter");

  unsigned elemSize = op.getElemSizeInBits() / 8;
  std::optional<int64_t> width = getConstantIntValue(op.getBaseWidth());
  if (width) {
    if (*width > (1 << 24))
      return op->emitOpError("2nd operand (base width) should be <= 24 bits");
    if (*width < 64)
      return op->emitOpError("2nd operand (base width) should be >= 64");
    if (*width % std::max(4u, elemSize) != 0)
      return op->emitOpError(
          "2nd operand (base width) should be aligned to MAX(4, element_size)");
  }
  std::optional<int64_t> height = getConstantIntValue(op.getBaseHeight());
  if (height)
    if (*height > (1 << 24))
      return op->emitOpError("3rd operand (base height) should be <= 24 bits");
  std::optional<int64_t> pitch = getConstantIntValue(op.getBasePitch());
  if (pitch) {
    if (*pitch > (1 << 24))
      return op->emitOpError("4th operand (base pitch) should be <= 24 bits");
    if (*pitch < 64)
      return op->emitOpError("4th operand (base pitch) should be >= 64");
    if (*pitch % 16 != 0)
      return op->emitOpError(
          "4th operand (base pitch) should be a multiple of 16 bytes");
  }
  if (pitch && width && *pitch < *width)
    return op->emitOpError(
        "4th operand (base pitch) should be >= 2nd operand (base width)");
  std::optional<int64_t> x = getConstantIntValue(op.getX());
  if (x) {
    if (elemSize == 1 && (*x % 4 != 0))
      return op->emitOpError(
          "5th operand (x) should be a multiple of 4 for 8 bit elements");
    else if (elemSize == 2 && (*x % 2 != 0))
      return op->emitOpError(
          "5th operand (x) should be a multiple of 2 for 16 bit elements");
  }

  uint32_t tileWidth = op.getTileWidth();
  uint32_t tileHeight = op.getTileHeight();
  uint32_t vBlocks = op.getVBlocks();
  auto isPowerOfTwo = [](uint32_t x) { return x && !(x & (x - 1)); };
  if (!isPowerOfTwo(tileWidth) || !isPowerOfTwo(tileHeight) ||
      !isPowerOfTwo(vBlocks))
    return op->emitOpError("expecting tile shape to be power of two");

  if (tileWidth > 64)
    return op->emitOpError("expecting tile_width to be between 1 and 64");
  if (tileHeight > 32)
    return op->emitOpError("expecting tile_height to be between 1 and 32");
  if (vBlocks > 4)
    return op->emitOpError("expecting v_blocks to be between 1 and 4");

  return success();
}

template <typename Op> static LogicalResult verify2DBlockHWRestriction(Op op) {
  static_assert(llvm::is_one_of<Op, TritonGEN::Matrix2DBlockLoadOp,
                                TritonGEN::Matrix2DBlockStoreOp,
                                TritonGEN::Matrix2DBlockPrefetchOp>::value,
                "Unexpected template parameter");

  if (verify2DBlockAddressPayloadRestriction(op).failed())
    return failure();

  unsigned elemSizeInBits = op.getElemSizeInBits();
  if (elemSizeInBits != 8 && elemSizeInBits != 16 && elemSizeInBits != 32 &&
      elemSizeInBits != 64)
    return op->emitOpError(
        "expecting 'elem_size_in_bits' to be 8, 16, 32, or 64");

  uint32_t tileWidth = op.getTileWidth();
  uint32_t vBlocks = op.getVBlocks();
  if (elemSizeInBits * tileWidth * vBlocks > 512)
    return op->emitOpError(
        "expecting elem_size_in_bits * tile_width * v_blocks <= 512");

  assert(tileWidth >= 1 && tileWidth <= 64 &&
         "tile_width should be between 1 and 64");
  switch (elemSizeInBits) {
  case 8:
    if (tileWidth < 4)
      return op->emitOpError("expecting tile_width to be between 4 and 64");
    break;
  case 16:
    if (tileWidth < 2 || tileWidth > 32)
      return op->emitOpError("expecting tile_width to be between 2 and 32");
    break;
  case 32:
    if (tileWidth > 16)
      return op->emitOpError("expecting tile_width to be between 1 and 16");
    if (vBlocks == 4)
      return op->emitOpError("v_blocks for 32 bit elements should be 1 or 2");
    break;
  case 64:
    if (tileWidth > 8)
      return op->emitOpError("expecting tile_width to be between 1 and 8");
    if (vBlocks != 1)
      return op->emitOpError("v_blocks for 64 bit elements should be 1");
    break;
  default:
    llvm_unreachable("unexpected element size");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// gen.matrix.dpas
//===----------------------------------------------------------------------===//

LogicalResult TritonGEN::MatrixDPASOp::verify() {
  if (getRc() != 1 && getRc() != 2 && getRc() != 4 && getRc() != 8)
    return this->emitOpError("expecting repeat count to be 1, 2, 4, or 8");

  TritonGEN::PrecisionType precision = getPa();
  if (getPa() != getPb())
    return this->emitOpError(
        "expecting precision of matrix A and B to be the same");

  VectorType ATy = getA().getType();
  VectorType BTy = getB().getType();
  VectorType CTy = getC().getType();
  VectorType DTy = getD().getType();
  if (CTy != DTy)
    return this->emitOpError(
        "1st operand (C) and result (D) should have the same type");

  if (CTy.getNumElements() != getRc() || DTy.getNumElements() != getRc())
    return this->emitOpError("the dimension for 1st operand (C) and "
                             "result (D) should match repeat count");

  constexpr unsigned SD = 8;
  if (BTy.getNumElements() != SD)
    return this->emitOpError("the dimension for the 3rd operand (B) should "
                             "match the systolic depth of 8");

  Type AElemTy = ATy.getElementType();
  Type BElemTy = BTy.getElementType();
  Type CElemTy = CTy.getElementType();

  switch (precision) {
  case PrecisionType::U8:
  case PrecisionType::S8:
    if (!CElemTy.isInteger(32))
      return this->emitOpError(
          "the element type for 1st operand (C) and the result should be i32");
    break;
  case PrecisionType::FP16:
    if (!(CElemTy.isF16() || CElemTy.isF32()))
      return this->emitOpError("the element type for 1st operand (C) and the "
                               "result should be f16 or f32");
    break;
  case PrecisionType::BF16:
    if (!(CElemTy.isBF16() || CElemTy.isF32()))
      return this->emitOpError("the element type for 1st operand (C) and the "
                               "result should be bf16 or f32");
    break;
  case PrecisionType::TF32:
    if (!CElemTy.isF32())
      return this->emitOpError(
          "the element type for 1st operand (C) and the result should be f32");
    break;
  default:
    return this->emitOpError(
        "expecting precision type to be tf32, bf16, fp16, u8, or s8");
  }

  switch (precision) {
  case TritonGEN::PrecisionType::TF32:
    if (ATy.getNumElements() != getRc() / 2)
      return this->emitOpError("the dimension for the 2nd operand (A) should "
                               "be equal to half of the repeat count");
    if (!isa<Float32Type>(AElemTy) && !AElemTy.isInteger(32))
      return this->emitOpError("2nd operand (A) element type should be f32 or "
                               "i32 when the precision type is tf32");
    if (!isa<Float32Type>(BElemTy) && !BElemTy.isInteger(32))
      return this->emitOpError("3rd operand (B) element type should be f32 or "
                               "i32 when the precision type is tf32");
    break;
  case TritonGEN::PrecisionType::BF16:
  case TritonGEN::PrecisionType::FP16:
  case TritonGEN::PrecisionType::U8:
  case TritonGEN::PrecisionType::S8:
    if (ATy.getNumElements() != getRc())
      return this->emitOpError("2nd operand (A) should have the same number of "
                               "elements as repeat count");
    if (!AElemTy.isInteger(16))
      return this->emitOpError(
          "2nd operand (A) element type should be i16 when "
          "the precision type is not tf32");
    if (!BElemTy.isInteger(32))
      return this->emitOpError(
          "3rd operand (B) element type should be i32 when "
          "the precision type is not tf32");
    break;
  default:
    llvm_unreachable("unhandled precision type");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// gen.2Dblockload
//===----------------------------------------------------------------------===//

static LogicalResult
verify2DBlockLoadHWRestriction(TritonGEN::Matrix2DBlockLoadOp op) {
  VectorType resTy = op.getRes().getType();
  unsigned resElemTySize = resTy.getElementType().getIntOrFloatBitWidth();
  unsigned resSize = resTy.getNumElements() * resElemTySize;
  unsigned subgroupSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(
      op->getParentOfType<mlir::ModuleOp>());
  ;
  unsigned expectedSize = op.getElemSizeInBits() * op.getTileHeight() *
                          op.getTileWidth() * op.getVBlocks() / subgroupSize;
  if (resSize != expectedSize)
    return op->emitOpError() << "result size of " << resSize
                             << " bits does not match the expected size of "
                             << expectedSize << " bits";

  if (op.getTranspose() && op.getVnniTransform())
    return op->emitOpError(
        "transpose and vnni_transform are mutually exclusive");

  if (op.getTranspose()) {
    assert(!op.getVnniTransform() &&
           "Expecting vnni_transform should be false");

    uint32_t vBlocks = op.getVBlocks();
    if (vBlocks != 1)
      return op->emitOpError("expecting v_blocks to be 1");

    uint32_t tileHeight = op.getTileHeight();
    uint32_t tileWidth = op.getTileWidth();
    switch (op.getElemSizeInBits()) {
    case 32:
      assert(tileWidth >= 1 &&
             "tile_width should be greater than or equal to 1");
      if (tileWidth > 8)
        return op->emitOpError("expecting tile_width to be between 1 and 8");
      break;
    case 64:
      if (tileHeight != 8)
        return op->emitOpError(
            "expecting tile_height to be 8 for 64 bit elements");
      if (tileWidth != 1 && tileWidth != 2 && tileWidth != 4)
        return op->emitOpError("expecting tile_width to be 1, 2, or 4");
      break;
    default:
      return op->emitOpError(
          "transpose is only supported for 32 and 64 bit elements");
    }

    return success();
  }

  if (op.getVnniTransform()) {
    assert(!op.getTranspose() && "Expecting transpose should be false");

    uint32_t tileHeight = op.getTileHeight();
    assert(tileHeight <= 32 &&
           "tile_height should be less than or equal to 32");
    switch (op.getElemSizeInBits()) {
    case 8:
      if (tileHeight < 4)
        return op->emitOpError("expecting tile_height to be between 4 and 32");
      break;
    case 16:
      if (tileHeight < 2)
        return op->emitOpError("expecting tile_height to be between 2 and 32");
      break;
    default:
      return op->emitOpError(
          "vnni_transform is only supported for 8 and 16 bit elements");
    }

    return success();
  }

  return success();
}

LogicalResult TritonGEN::Matrix2DBlockLoadOp::verify() {
  if (verify2DBlockHWRestriction(*this).failed())
    return failure();

  if (verify2DBlockLoadHWRestriction(*this).failed())
    return failure();

  VectorType resTy = getRes().getType();
  unsigned resElemTySize = resTy.getElementType().getIntOrFloatBitWidth();
  if (getElemSizeInBits() == 32 || getVnniTransform()) {
    if (resElemTySize != 32)
      return emitOpError() << "expecting result element type to be 32 bits";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// gen.2Dblockstore
//===----------------------------------------------------------------------===//

LogicalResult TritonGEN::Matrix2DBlockStoreOp::verify() {
  if (verify2DBlockHWRestriction(*this).failed())
    return failure();

  uint32_t tileHeight = getTileHeight();
  assert(tileHeight >= 1 && "tile_height should be greater than or equal to 1");
  if (tileHeight > 8)
    return emitOpError("expecting tile_height to be between 1 and 8");

  uint32_t vBlocks = getVBlocks();
  if (vBlocks != 1)
    return emitOpError("expecting v_blocks to be 1");

  return success();
}

//===----------------------------------------------------------------------===//
// gen.2Dblockprefetch
//===----------------------------------------------------------------------===//

LogicalResult TritonGEN::Matrix2DBlockPrefetchOp::verify() {
  return verify2DBlockHWRestriction(*this);
}
