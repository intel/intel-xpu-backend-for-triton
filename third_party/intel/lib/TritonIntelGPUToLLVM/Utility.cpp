//===- Utility.cpp - Code generation utilities ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Utility.h"
#include "mlir/Conversion/ArithCommon/AttrToLLVMConverter.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"

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
  Type valType = val.getType();
  Type shuffleType = findShuffleType(rewriter, valType);

  const unsigned bitWidth = valType.getIntOrFloatBitWidth();
  if (shuffleType != valType) {
    assert(shuffleType.isInteger() &&
           "expected to bitcast to an integer for unsupported shuffles");
    if (!valType.isInteger()) {
      val = bitcast(val, int_ty(bitWidth));
    }
    if (bitWidth < shuffleType.getIntOrFloatBitWidth()) {
      val = zext(shuffleType, val);
    }
  }

  int width = TritonGEN::getSubgroupSize(i.getDefiningOp());
  Value widthConstant = i32_val(width);
  Value result =
      rewriter.create<mlir::gpu::ShuffleOp>(loc, val, i, widthConstant, mode)
          .getShuffleResult();

  if (shuffleType != valType) {
    if (bitWidth < shuffleType.getIntOrFloatBitWidth()) {
      result = trunc(int_ty(bitWidth), result);
    }
    if (!valType.isInteger()) {
      result = bitcast(result, valType);
    }
  }

  return result;
}

static Value shuffleCommon(Location loc, RewriterBase &rewriter, Value val,
                           Value i, mlir::gpu::ShuffleMode mode) {
  // To shuffle pointers, convert them to i64.
  Type valTy = val.getType();
  if (isa<LLVM::LLVMPointerType>(valTy))
    val = ptrtoint(i64_ty, val);
  Value result = shuffleCommonImpl(loc, rewriter, val, i, mode);
  if (isa<LLVM::LLVMPointerType>(valTy))
    result = inttoptr(valTy, result);
  return result;
}

Value shuffleXor(Location loc, RewriterBase &rewriter, Value val, int i) {
  return shuffleCommon(loc, rewriter, val, i32_val(i),
                       mlir::gpu::ShuffleMode::XOR);
}

Value shuffleUp(Location loc, RewriterBase &rewriter, Value val, int i) {
  return shuffleCommon(loc, rewriter, val, i32_val(i),
                       mlir::gpu::ShuffleMode::UP);
}

Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, int i) {
  return shuffleIdx(loc, rewriter, val, i32_val(i));
}

Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, Value i) {
  return shuffleCommon(loc, rewriter, val, i, mlir::gpu::ShuffleMode::IDX);
}

// declare __spirv_ocl_printf(i8*, ...) as external function
LLVM::LLVMFuncOp getSpirvPrintfDeclaration(RewriterBase &rewriter) {
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  StringRef funcName("_Z18__spirv_ocl_printf");
  Operation *funcOp = moduleOp.lookupSymbol(funcName);
  if (funcOp)
    return cast<LLVM::LLVMFuncOp>(*funcOp);

  MLIRContext *context = rewriter.getContext();
  auto ptrTy = LLVM::LLVMPointerType::get(
      context, TritonGEN::TritonGENMemorySpace::kUniformConstant);
  SmallVector<Type> argsType{ptrTy};
  auto retType = i32_ty;
  auto funcType =
      LLVM::LLVMFunctionType::get(retType, argsType, /*isVarArg*/ true);

  ConversionPatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(moduleOp.getBody());

  auto printFunc = rewriter.create<LLVM::LLVMFuncOp>(
      UnknownLoc::get(context), funcName, funcType, LLVM::Linkage::External,
      /*dsoLocal*/ false, LLVM::CConv::SPIR_FUNC, /*comdat=*/SymbolRefAttr{});
  printFunc->setAttr("nounwind", rewriter.getUnitAttr());

  return printFunc;
}

static LLVM::RoundingMode
convertTritonRoundingModeToLLVM(const triton::RoundingMode rounding) {
  LLVM::RoundingMode roundingMode;
  switch (rounding) {
  case triton::RoundingMode::RTNE:
    return LLVM::RoundingMode::NearestTiesToEven;
  case triton::RoundingMode::RTZ:
    return LLVM::RoundingMode::TowardZero;
  default:
    llvm::errs() << "WARNING: unsupported rounding mode for f32->f16 "
                    "conversion: "
                 << stringifyRoundingMode(rounding) << "\n";
    llvm_unreachable("");
  }
}

Value convertFp32ToFp16(Location loc, ConversionPatternRewriter &rewriter,
                        const Value &v, const triton::RoundingMode rounding) {
  MLIRContext *ctx = rewriter.getContext();
  return rewriter.create<LLVM::ConstrainedFPTruncIntr>(
      loc, f16_ty, v,
      LLVM::RoundingModeAttr::get(ctx,
                                  convertTritonRoundingModeToLLVM(rounding)),
      arith::getLLVMDefaultFPExceptionBehavior(*ctx));
}

} // namespace mlir::LLVM::intel
