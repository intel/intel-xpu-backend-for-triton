//===- Utility.cpp - Code generation utilities ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Utility.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace LLVM {
namespace intel {

Value storeShared(ConversionPatternRewriter &rewriter, Location loc, Value ptr,
                  Value val, Value pred) {
  createPredicatedBlock(rewriter, loc, pred, [&] {
    store(val, ptr);
    return ArrayRef<Value>();
  });
  return Value();
}

Value loadShared(ConversionPatternRewriter &rewriter, Location loc, Value ptr,
                 Type elemTy, Value pred) {
  assert(ptr.getType().cast<LLVMPointerType>().getAddressSpace() == 3 &&
         "Invalid addr space for loadShared");
  Value undef = undef(elemTy);
  Block &endBlock = createPredicatedBlock(rewriter, loc, pred,
                                          SmallVector<Value, 1>{undef}, [&] {
                                            Value ret = load(elemTy, ptr);
                                            return SmallVector<Value, 1>{ret};
                                          });
  return *endBlock.args_begin();
}

static Value shuffleCommon(Location loc, ConversionPatternRewriter &rewriter,
                           Value val, Value i, TritonGEN::ShflKind mode,
                           Value clamp) {
  Type type = val.getType();
  return rewriter.create<TritonGEN::SubGroupShuffleOp>(loc, type, val, i, mode);
}

Value shuffleXor(Location loc, ConversionPatternRewriter &rewriter, Value val,
                 int i) {
  return shuffleCommon(loc, rewriter, val, i32_val(i), TritonGEN::ShflKind::XOR,
                       i32_val(0x1f));
}

Value shuffleUp(Location loc, ConversionPatternRewriter &rewriter, Value val,
                int i) {
  return shuffleCommon(loc, rewriter, val, i32_val(i), TritonGEN::ShflKind::UP,
                       i32_val(0x0));
}

Value shuffleIdx(Location loc, ConversionPatternRewriter &rewriter, Value val,
                 int i) {
  return shuffleIdx(loc, rewriter, val, i32_val(i));
}

Value shuffleIdx(Location loc, ConversionPatternRewriter &rewriter, Value val,
                 Value i) {
  return shuffleCommon(loc, rewriter, val, i, TritonGEN::ShflKind::IDX,
                       i32_val(0x1f));
}

Value addStringToModule(Location loc, ConversionPatternRewriter &rewriter,
                        StringRef key, StringRef content,
                        unsigned addressSpace) {
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto ctx = moduleOp.getContext();
  unsigned stringNumber = 0;
  SmallString<16> stringConstName;
  do {
    stringConstName.clear();
    (key + Twine(stringNumber++)).toStringRef(stringConstName);
  } while (moduleOp.lookupSymbol(stringConstName));

  llvm::SmallString<64> contentStr(content);
  // SPIRV needs c-string
  contentStr.push_back('\0');
  size_t contentSize = contentStr.size_in_bytes();
  auto globalType = LLVM::LLVMArrayType::get(i8_ty, contentSize);

  LLVM::GlobalOp global;
  {
    ConversionPatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    global = rewriter.create<LLVM::GlobalOp>(
        UnknownLoc::get(ctx), globalType,
        /*isConstant=*/true, LLVM::Linkage::Internal, stringConstName,
        rewriter.getStringAttr(contentStr), /*alignment=*/0, addressSpace);
  }

  Value zero = i32_val(0);
  Type globalPtrType = LLVM::LLVMPointerType::get(ctx, global.getAddrSpace());
  Value globalPtr = rewriter.create<LLVM::AddressOfOp>(
      UnknownLoc::get(ctx), globalPtrType, global.getSymName());
  Value stringStart = gep(ptr_ty(ctx, global.getAddrSpace()), i8_ty, globalPtr,
                          SmallVector<Value>({zero}));
  return stringStart;
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

void llPrintf(ConversionPatternRewriter &rewriter, Value msg, ValueRange args) {
  auto *ctx = rewriter.getContext();
  Type ptr = ptr_ty(ctx);
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto funcOp = getSpirvPrintfDeclaration(rewriter);
  auto loc = UnknownLoc::get(ctx);

  SmallVector<Value> operands;
  operands.push_back(msg);
  for (auto arg : args) {
    operands.push_back(arg);
  }
  call(funcOp, operands);
}

Value llPrintf(ConversionPatternRewriter &rewriter, StringRef msg,
               ValueRange args) {
  assert(!msg.empty() && "printf with empty string not supported");
  llvm::SmallString<64> msgNewline(msg);
  msgNewline.push_back('\n');
  Value msgValue = addStringToModule(
      UnknownLoc::get(rewriter.getContext()), rewriter, "printfFormat_",
      msgNewline, TritonGEN::TritonGENMemorySpace::kUniformConstant);
  llPrintf(rewriter, msgValue, args);
  return msgValue;
}

} // namespace intel
} // namespace LLVM
} // namespace mlir
