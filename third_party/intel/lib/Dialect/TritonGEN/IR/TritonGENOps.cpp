//===- TritonGENOps.cpp - TritonGEN dialect operations --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/OpDefinition.h"

#include "intel/include/Dialect/TritonGEN/IR/TritonGENDialect.h"

#include "llvm/ADT/STLExtras.h"
#include <cstdint>

using namespace mlir;
using namespace mlir::triton;

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

template <typename Op> static LogicalResult verifyInput(Op op) {
  static_assert(llvm::is_one_of<Op, TritonGEN::Matrix2DBlockLoadOp,
                                TritonGEN::Matrix2DBlockStoreOp,
                                TritonGEN::Matrix2DBlockPrefetchOp>::value,
                "Unexpected template parameter");

  if (op.getElemSizeInBits() != 8 && op.getElemSizeInBits() != 16 &&
      op.getElemSizeInBits() != 32)
    return op->emitOpError("expecting 'elem_size_in_bits' to be 8, 16, or 32");

  if (op.getTranspose() && op.getVnniTransform())
    return op->emitOpError(
        "transpose and vnni transform are mutually exclusive");

  std::optional<int64_t> width = getConstantIntValue(op.getBaseWidth());
  std::optional<int64_t> pitch = getConstantIntValue(op.getBasePitch());
  if (pitch && width && *pitch < *width)
    return op->emitOpError(
        "4th operand (base pitch) should be >= 2nd operand (base width)");

  uint32_t TileHeight = op.getTileHeight();
  if (TileHeight != 1 && TileHeight != 2 && TileHeight != 4 &&
      TileHeight != 8 && TileHeight != 16 && TileHeight != 32)
    return op->emitOpError("expecting tile_height to be 1, 2, 4, 8, 16, or 32");

  uint32_t TileWidth = op.getTileWidth();
  switch (op.getElemSizeInBits()) {
  case 32:
    if (TileWidth != 8)
      return op->emitOpError("tile_width for 32 bit elements should be equal "
                             "to systolic depth, i.e., 8 elements");
    break;
  case 16:
    if (TileWidth != 16)
      return op->emitOpError("tile_width for 16 bit elements should be equal "
                             "to systolic depth times 2, i.e., 16 elements");
    break;
  case 8:
    if (TileWidth != 32)
      return op->emitOpError("tile_width for 8 bit elements should be equal "
                             "to systolic depth times 4, i.e., 32 elements");
    break;
  default:
    return op->emitOpError("element size should be 8, 16 or 32 bits");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// gen.sub_group_reduce
//===----------------------------------------------------------------------===//

LogicalResult TritonGEN::SubGroupReduceOp::verify() {
  spirv::TargetEnvAttr attr = spirv::lookupTargetEnv(op);
  if (!attr)
    return this->emitOpError("expecting valid target env attribute");

  Type ty = getValue().getType();
  switch (getKind()) {
  case TritonGEN::ReduceKind::FSUM:
  case TritonGEN::ReduceKind::FPROD:
  case TritonGEN::ReduceKind::FMIN:
  case TritonGEN::ReduceKind::FMAX:
    if (!isa<FloatType>(ty))
      return this->emitOpError(
          "expecting floating point type for floating point reduction");
    break;
  default:
    if (!isa<IntegerType>(ty))
      return this->emitOpError("expecting integer type for integer reduction");
  }

  if (getSize() < 1 || getSize() > TritonGEN::getSubgroupSize(*this) ||
      !llvm::isPowerOf2_32(getSize()))
    return this->emitOpError(
        "expecting size to be a power of 2 between 1 and subgroup size");

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

  if (precision == TritonGEN::PrecisionType::U8 ||
      precision == TritonGEN::PrecisionType::S8) {
    if (!CElemTy.isInteger(32))
      return this->emitOpError("the element type for 1st operand (C) and "
                               "the result should be i32");
  } else if (!CElemTy.isF32())
    return this->emitOpError("the element type for 1st operand (C) and the "
                             "result should be f32");

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
    return this->emitOpError(
        "expecting precision type to be tf32, bf16, fp16, u8, or s8");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// gen.2Dblockload
//===----------------------------------------------------------------------===//

LogicalResult TritonGEN::Matrix2DBlockLoadOp::verify() {
  return verifyInput(*this);
}

//===----------------------------------------------------------------------===//
// gen.2Dblockstore
//===----------------------------------------------------------------------===//

LogicalResult TritonGEN::Matrix2DBlockStoreOp::verify() {
  return verifyInput(*this);
}

//===----------------------------------------------------------------------===//
// gen.2Dblockprefetch
//===----------------------------------------------------------------------===//

LogicalResult TritonGEN::Matrix2DBlockPrefetchOp::verify() {
  return verifyInput(*this);
}
