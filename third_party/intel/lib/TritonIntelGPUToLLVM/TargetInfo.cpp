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
  LLVM::Intel::createPredicatedBlock(rewriter, loc, pred, [&] {
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
  Block &endBlock = LLVM::Intel::createPredicatedBlock(
      rewriter, loc, pred, SmallVector<Value, 1>{undef}, [&] {
        Value ret = load(elemTy, ptr);
        return SmallVector<Value, 1>{ret};
      });
  return *endBlock.args_begin();
}

// FIXME: Copy ShflKind from NVVM dialect to GEN dialect.
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
  Type type = val.getType();
  return rewriter.create<TritonGEN::SubGroupShuffleOp>(loc, type, val, i,
                                                       toGenShuffleMode(mode));
}

Value TargetInfo::shuffleXor(ConversionPatternRewriter &rewriter, Location loc,
                             Value val, int i) const {
  return commonShflSync(loc, rewriter, val, i32_val(i), NVVM::ShflKind::bfly,
                        i32_val(0x1f));
}

Value TargetInfo::shuffleUp(ConversionPatternRewriter &rewriter, Location loc,
                            Value val, int i) const {
  return commonShflSync(loc, rewriter, val, i32_val(i), NVVM::ShflKind::up,
                        i32_val(0x0));
}

Value TargetInfo::shuffleIdx(ConversionPatternRewriter &rewriter, Location loc,
                             Value val, int i) const {
  return LLVM::Intel::shflIdxSync(loc, rewriter, val, i32_val(i));
}

Value TargetInfo::shuffleIdx(ConversionPatternRewriter &rewriter, Location loc,
                             Value val, Value i) const {
  return commonShflSync(loc, rewriter, val, i, NVVM::ShflKind::idx,
                        i32_val(0x1f));
}

Value TargetInfo::programId(ConversionPatternRewriter &rewriter, Location loc,
                            ModuleOp moduleOp, int axis) const {
  return LLVM::Intel::llGetPid(loc, rewriter, moduleOp, axis);
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

std::string TargetInfo::getMulhiFuncName(Type resultElementTy) const {
  std::string funcName =
      resultElementTy.isInteger(32) ? "__imf_umulhi" : "__imf_umul64hi";
  return funcName;
}

// declare __spirv_ocl_printf(i8*, ...) as external function
LLVM::LLVMFuncOp
getSpirvPrintfDeclaration(ConversionPatternRewriter &rewriter) {
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

void TargetInfo::printf(ConversionPatternRewriter &rewriter,
                        Value formatStrStart, int /*formatStrByteCount*/,
                        ValueRange args) const {
  auto *ctx = rewriter.getContext();
  Type ptr = ptr_ty(ctx);
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto funcOp = getSpirvPrintfDeclaration(rewriter);
  auto loc = UnknownLoc::get(ctx);

  SmallVector<Value> operands;
  operands.push_back(formatStrStart);
  for (auto arg : args) {
    operands.push_back(arg);
  }
  call(funcOp, operands);
}

static LLVM::LLVMFuncOp
getAssertfailDeclaration(ConversionPatternRewriter &rewriter) {
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  StringRef funcName = "__assert_fail";
  Operation *funcOp = moduleOp.lookupSymbol(funcName);
  if (funcOp)
    return cast<LLVM::LLVMFuncOp>(*funcOp);

  // void __assert_fail(const char * assertion, const char * file, unsigned
  // int line, const char * function);
  auto *ctx = rewriter.getContext();
  SmallVector<Type> argsType;
  argsType = {ptr_ty(ctx, TritonGEN::TritonGENMemorySpace::kGeneric),
              ptr_ty(ctx, TritonGEN::TritonGENMemorySpace::kGeneric), i32_ty,
              ptr_ty(ctx, TritonGEN::TritonGENMemorySpace::kGeneric)};
  auto funcType = LLVM::LLVMFunctionType::get(void_ty(ctx), argsType);

  ConversionPatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(moduleOp.getBody());

  auto func = rewriter.create<LLVM::LLVMFuncOp>(UnknownLoc::get(ctx), funcName,
                                                funcType);
  func.setCConv(LLVM::cconv::CConv::SPIR_FUNC);
  return func;
}

void TargetInfo::assertFail(ConversionPatternRewriter &rewriter, Location loc,
                            StringRef message, StringRef file, StringRef func,
                            int line) const {
  auto funcOp = getAssertfailDeclaration(rewriter);
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  unsigned addrSpace = TritonGEN::TritonGENMemorySpace::kCrossWorkgroup;
  Value messageString = LLVM::Intel::addStringToModule(
      loc, rewriter, "assertMessage_", message, addrSpace);
  Value fileString = LLVM::Intel::addStringToModule(
      loc, rewriter, "assertFile_", file, addrSpace);
  Value funcString = LLVM::Intel::addStringToModule(
      loc, rewriter, "assertFunc_", func, addrSpace);
  Value lineNumber = i32_val(line);

  auto *ctx = rewriter.getContext();
  SmallVector<Value> operands;
  Value messageStringPtr = addrspacecast(
      ptr_ty(ctx, TritonGEN::TritonGENMemorySpace::kGeneric), messageString);
  Value fileStringPtr = addrspacecast(
      ptr_ty(ctx, TritonGEN::TritonGENMemorySpace::kGeneric), fileString);
  Value funcStringPtr = addrspacecast(
      ptr_ty(ctx, TritonGEN::TritonGENMemorySpace::kGeneric), funcString);
  operands = {messageStringPtr, fileStringPtr, lineNumber, funcStringPtr};
  auto ret = call(funcOp, operands);
  ret.setCConv(LLVM::cconv::CConv::SPIR_FUNC);
}

} // namespace mlir::triton::intel
