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
