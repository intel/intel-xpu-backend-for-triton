//===- Utility.cpp - Code generation utilities ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Utility.h"

#include "mlir/Conversion/ArithCommon/AttrToLLVMConverter.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir::LLVM::intel {

static Type findShuffleType(RewriterBase &rewriter, Type valType) {
  if (valType.isBF16())
    return rewriter.getI16Type();

  unsigned bitWidth = valType.getIntOrFloatBitWidth();
  if (bitWidth < 8)
    return rewriter.getI8Type();

  assert((valType.isInteger(8) || valType.isInteger(16) ||
          valType.isInteger(32) || valType.isInteger(64) || valType.isF16() ||
          valType.isF32() || valType.isF64()) &&
         "Invalid Shuffle Type");
  return valType;
}

static Value shuffleCommonImpl(Location loc, RewriterBase &rewriter, Value val,
                               Value i, mlir::gpu::ShuffleMode mode) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Type valType = val.getType();
  Type shuffleType = findShuffleType(rewriter, valType);

  const unsigned bitWidth = valType.getIntOrFloatBitWidth();
  if (shuffleType != valType) {
    assert(shuffleType.isInteger() &&
           "expected to bitcast to an integer for unsupported shuffles");
    if (!valType.isInteger()) {
      val = b.bitcast(val, int_ty(bitWidth));
    }
    if (bitWidth < shuffleType.getIntOrFloatBitWidth()) {
      val = b.zext(shuffleType, val);
    }
  }

  int width = triton::gpu::TritonGPUDialect::getThreadsPerWarp(
      i.getDefiningOp()->getParentOfType<ModuleOp>());
  Value widthConstant = b.i32_val(width);
  Value result =
      rewriter.create<mlir::gpu::ShuffleOp>(loc, val, i, widthConstant, mode)
          .getShuffleResult();

  if (shuffleType != valType) {
    if (bitWidth < shuffleType.getIntOrFloatBitWidth()) {
      result = b.trunc(int_ty(bitWidth), result);
    }
    if (!valType.isInteger()) {
      result = b.bitcast(result, valType);
    }
  }

  return result;
}

static Value shuffleCommon(Location loc, RewriterBase &rewriter, Value val,
                           Value i, mlir::gpu::ShuffleMode mode) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  // To shuffle pointers, convert them to i64.
  Type valTy = val.getType();
  if (isa<LLVM::LLVMPointerType>(valTy))
    val = b.ptrtoint(i64_ty, val);
  Value result = shuffleCommonImpl(loc, rewriter, val, i, mode);
  if (isa<LLVM::LLVMPointerType>(valTy))
    result = b.inttoptr(valTy, result);
  return result;
}

Value shuffleXor(Location loc, RewriterBase &rewriter, Value val, int i) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return shuffleCommon(loc, rewriter, val, b.i32_val(i),
                       mlir::gpu::ShuffleMode::XOR);
}

Value shuffleUp(Location loc, RewriterBase &rewriter, Value val, int i) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return shuffleCommon(loc, rewriter, val, b.i32_val(i),
                       mlir::gpu::ShuffleMode::UP);
}

Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, int i) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return shuffleIdx(loc, rewriter, val, b.i32_val(i));
}

Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, Value i) {
  return shuffleCommon(loc, rewriter, val, i, mlir::gpu::ShuffleMode::IDX);
}

Value permute(Location loc, RewriterBase &rewriter, Value x, Value y,
              Value selector) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  x = b.bitcast(x, int_ty(32));
  y = b.bitcast(y, int_ty(32));
  selector = b.bitcast(selector, int_ty(32));

  Value result = b.i32_val(0);

  for (int i = 0; i < 4; ++i) {
    Value shiftAmount = b.i32_val(4 * i);
    Value nibble = b.and_(b.lshr(selector, shiftAmount), b.i32_val(0xF));

    Value sourceIsY = b.and_(b.lshr(nibble, b.i32_val(2)), b.i32_val(1));
    Value byteIdx = b.and_(nibble, b.i32_val(0x3));
    Value replicate = b.and_(nibble, b.i32_val(0x8));

    Value extractedX = b.i32_val(0);
    Value extractedY = b.i32_val(0);

    for (int j = 0; j < 4; ++j) {
      Value isThisByte = b.icmp_eq(byteIdx, b.i32_val(j));

      Value xByte = b.and_(b.lshr(x, b.i32_val(8 * j)), b.i32_val(0xFF));
      Value yByte = b.and_(b.lshr(y, b.i32_val(8 * j)), b.i32_val(0xFF));

      extractedX = b.select(isThisByte, xByte, extractedX);
      extractedY = b.select(isThisByte, yByte, extractedY);
    }

    Value selectedByte =
        b.select(b.icmp_ne(sourceIsY, b.i32_val(0)), extractedY, extractedX);

    // Handle replication of MSB if bit 3 is set. It is not required
    // by Triton's permute, but other backends do it this way.
    Value msb = b.and_(b.lshr(selectedByte, b.i32_val(7)), b.i32_val(1));
    Value replicated = b.select(b.icmp_ne(msb, b.i32_val(0)), b.i32_val(0xFF),
                                b.i32_val(0x00));

    Value finalByte =
        b.select(b.icmp_ne(replicate, b.i32_val(0)), replicated, selectedByte);

    Value shiftedByte = b.shl(finalByte, b.i32_val(8 * i));
    result = b.or_(result, shiftedByte);
  }

  return result;
}

LLVM::RoundingMode
convertTritonRoundingModeToLLVM(const triton::RoundingMode rounding) {
  LLVM::RoundingMode roundingMode;
  switch (rounding) {
  case triton::RoundingMode::RTNE:
    return LLVM::RoundingMode::NearestTiesToEven;
  case triton::RoundingMode::RTZ:
    return LLVM::RoundingMode::TowardZero;
  default:
    llvm_unreachable(("WARNING: unsupported rounding mode for f32->f16 "
                      "conversion: " +
                      stringifyRoundingMode(rounding))
                         .str()
                         .c_str());
  }
}

} // namespace mlir::LLVM::intel
