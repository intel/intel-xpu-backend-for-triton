//===- TargetInfo.cpp - Target dependent information ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TargetInfo.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"

using namespace mlir;
namespace mlir::triton::intel {

bool TargetInfo::supportMaximumMinimum() const { return true; }
Value TargetInfo::ballot(ConversionPatternRewriter &rewriter, Location loc,
                         Type type, Value cmp) const {
  assert("TODO: implement ballot on XPU");
  return Value();
}
Value TargetInfo::storeShared(ConversionPatternRewriter &rewriter, Location loc,
                              Value ptr, Value val, Value pred) const {
  mlir::LLVM::utils::createPredicatedBlock(rewriter, loc, pred, [&] {
    store(val, ptr);
    return ArrayRef<Value>();
  });
  return Value();
}

Value TargetInfo::loadShared(ConversionPatternRewriter &rewriter, Location loc,
                             Value ptr, Type elemTy, Value pred) const {
  assert(ptr.getType().cast<mlir::LLVM::LLVMPointerType>().getAddressSpace() ==
             3 &&
         "Invalid addr space for loadShared");
  Value undef = undef(elemTy);
  Block &endBlock = mlir::LLVM::utils::createPredicatedBlock(
      rewriter, loc, pred, SmallVector<Value, 1>{undef}, [&] {
        Value ret = load(elemTy, ptr);
        return SmallVector<Value, 1>{ret};
      });
  return *endBlock.args_begin();
}

static TritonGEN::ShflKind toGenShuffleMode(NVVM::ShflKind mode) {
  switch (mode) {
  case NVVM::ShflKind::bfly:
    return TritonGEN::ShflKind::XOR;
  case NVVM::ShflKind::up:
    return TritonGEN::ShflKind::UP;
  case NVVM::ShflKind::down:
    return TritonGEN::ShflKind::DOWN;
  case NVVM::ShflKind::idx:
    return TritonGEN::ShflKind::IDX;
  }
  llvm_unreachable("unsupported NVVM::ShflKind");
}

static Value commonShflSync(Location loc, ConversionPatternRewriter &rewriter,
                            Value val, Value i, NVVM::ShflKind mode,
                            Value clamp) {
  unsigned bits = val.getType().getIntOrFloatBitWidth();

  if (bits == 64) {
    Type vecTy = vec_ty(f32_ty, 2);
    Value vec = bitcast(val, vecTy);
    Value val0 = extract_element(f32_ty, vec, i32_val(0));
    Value val1 = extract_element(f32_ty, vec, i32_val(1));
    val0 = commonShflSync(loc, rewriter, val0, i, mode, clamp);
    val1 = commonShflSync(loc, rewriter, val1, i, mode, clamp);
    vec = undef(vecTy);
    vec = insert_element(vecTy, vec, val0, i32_val(0));
    vec = insert_element(vecTy, vec, val1, i32_val(1));
    return bitcast(vec, val.getType());
  }
  Type type = val.getType();
  return rewriter.create<TritonGEN::SubGroupShuffleOp>(loc, type, val, i,
                                                       toGenShuffleMode(mode));
}

Value TargetInfo::shuffleXor(Location loc, ConversionPatternRewriter &rewriter,
                             Value val, int i) const {
  return commonShflSync(loc, rewriter, val, i32_val(i), NVVM::ShflKind::bfly,
                        i32_val(0x1f));
}

Value TargetInfo::shuffleUp(Location loc, ConversionPatternRewriter &rewriter,
                            Value val, int i) const {
  return commonShflSync(loc, rewriter, val, i32_val(i), NVVM::ShflKind::up,
                        i32_val(0x0));
}

Value TargetInfo::shuffleIdx(Location loc, ConversionPatternRewriter &rewriter,
                             Value val, int i) const {
  return mlir::LLVM::utils::shflIdxSync(loc, rewriter, val, i32_val(i));
}

Value TargetInfo::shuffleIdx(Location loc, ConversionPatternRewriter &rewriter,
                             Value val, Value i) const {
  return commonShflSync(loc, rewriter, val, i, NVVM::ShflKind::idx,
                        i32_val(0x1f));
}

Value TargetInfo::programId(Location loc, ConversionPatternRewriter &rewriter,
                            ModuleOp moduleOp, int axis) const {
  return LLVM::utils::llGetPid(loc, rewriter, moduleOp, axis);
}

bool TargetInfo::warpReduce(ConversionPatternRewriter &rewriter, Location loc,
                            SmallVector<Value> &acc, triton::ReduceOp op,
                            unsigned numLaneToReduce) const {
  assert("TODO: implement warpReduce on XPU");
  return false;
}

bool TargetInfo::processReplicaUsingStMatrix(
    ConversionPatternRewriter &rewriter, Location loc, Value smemBase,
    SmallVector<Value> &vals, RankedTensorType srcTy, Type elemTy,
    ArrayRef<unsigned> paddedRepShape, ArrayRef<unsigned> origRepShape,
    ArrayRef<unsigned> outOrd, unsigned accumNumReplicates) const {
  return false;
}

} // namespace mlir::triton::intel
