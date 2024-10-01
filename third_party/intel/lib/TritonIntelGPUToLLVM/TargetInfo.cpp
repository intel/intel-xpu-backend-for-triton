//===- TargetInfo.cpp - Target dependent information ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TargetInfo.h"
#include "Utility.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

namespace mlir::triton::intel {

bool TargetInfo::supportMaximumMinimum() const { return true; }
Value TargetInfo::ballot(RewriterBase &rewriter, Location loc, Type type,
                         Value cmp) const {
  assert("TODO: implement ballot on XPU");
  return Value();
}

Value TargetInfo::getClusterCTAId(RewriterBase &rewriter, Location loc) const {
  // Clusters of thread blocks aren't supported.
  return i32_val(0);
}

void TargetInfo::storeDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              std::optional<Value> ctaId, Value val,
                              Value pred) const {
  LLVM::intel::createPredicatedBlock(rewriter, loc, pred, [&] {
    store(val, ptr);
    return ArrayRef<Value>();
  });
}

void TargetInfo::storeMatrixShared(RewriterBase &rewriter, Location loc,
                                   Value ptr, Value val) const {
  llvm::report_fatal_error("IntelGPU does not support stmatrix");
}

Value TargetInfo::loadDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              std::optional<Value> ctaId, Type elemTy,
                              Value pred) const {
  assert(cast<mlir::LLVM::LLVMPointerType>(ptr.getType()).getAddressSpace() ==
             3 &&
         "Invalid addr space for loadShared");
  Value undef = undef(elemTy);
  Block &endBlock = LLVM::intel::createPredicatedBlock(
      rewriter, loc, pred, SmallVector<Value, 1>{undef}, [&] {
        Value ret = load(elemTy, ptr);
        return SmallVector<Value, 1>{ret};
      });
  return *endBlock.args_begin();
}

Value TargetInfo::shuffleXor(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  return LLVM::intel::shuffleXor(loc, rewriter, val, i);
}

Value TargetInfo::shuffleUp(RewriterBase &rewriter, Location loc, Value val,
                            int i) const {
  return LLVM::intel::shuffleUp(loc, rewriter, val, i);
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  return LLVM::intel::shuffleIdx(loc, rewriter, val, i);
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             Value i) const {
  return LLVM::intel::shuffleIdx(loc, rewriter, val, i);
}

Value TargetInfo::programId(RewriterBase &rewriter, Location loc,
                            ModuleOp moduleOp, int axis) const {
  assert(axis >= 0);
  assert(axis < 3);
  assert(moduleOp);

  constexpr mlir::gpu::Dimension dims[] = {mlir::gpu::Dimension::x,
                                           mlir::gpu::Dimension::y,
                                           mlir::gpu::Dimension::z};

  Value blockId = rewriter.create<::mlir::gpu::BlockIdOp>(loc, dims[axis]);
  return rewriter.create<arith::IndexCastOp>(loc, i32_ty, blockId);
}

bool TargetInfo::warpReduce(RewriterBase &rewriter, Location loc,
                            SmallVector<Value> &acc, triton::ReduceOp op,
                            unsigned numLaneToReduce,
                            unsigned interleave) const {
  // No horizontal reduce required.
  if (numLaneToReduce == 1)
    return false;
  // Horizontal reduce with interleave stride not supported.
  if (interleave > 1)
    return false;
  // Check if it is a simple reduce operation supported by
  // TritonGEN::SubGroupReduceOp.
  if (op.getNumOperands() != 1 || op.getNumResults() != 1)
    return false;
  Region &combineOp = op.getCombineOp();
  if (combineOp.getBlocks().size() > 1)
    return false;
  Block &block = *combineOp.begin();
  Operation *yield = block.getTerminator();
  Operation *reduceOp = yield->getOperand(0).getDefiningOp();
  if (!reduceOp || reduceOp->getNumOperands() != 2 ||
      reduceOp->getNumResults() != 1)
    return false;
  if (reduceOp->getOperand(0) != block.getArgument(0) ||
      reduceOp->getOperand(1) != block.getArgument(1))
    return false;

  auto reduceKind =
      llvm::TypeSwitch<mlir::Operation *, std::optional<TritonGEN::ReduceKind>>(
          reduceOp)
          .Case<arith::AddFOp, arith::AddIOp>(
              [&](auto) { return TritonGEN::ReduceKind::ADD; })
          .Case<arith::MulFOp, arith::MulIOp>(
              [&](auto) { return TritonGEN::ReduceKind::MUL; })
          .Case<arith::MaxNumFOp>(
              [&](auto) { return TritonGEN::ReduceKind::MAX; })
          .Case<arith::MinNumFOp>(
              [&](auto) { return TritonGEN::ReduceKind::MIN; })
          .Case<arith::AndIOp>([&](auto) { return TritonGEN::ReduceKind::AND; })
          .Case<arith::OrIOp>([&](auto) { return TritonGEN::ReduceKind::OR; })
          .Case<arith::XOrIOp>([&](auto) { return TritonGEN::ReduceKind::XOR; })
          .Default([](auto) { return std::nullopt; });
  if (reduceKind == std::nullopt)
    return false;

  for (unsigned i = 0; i < acc.size(); ++i) {
    acc[i] = rewriter.create<TritonGEN::SubGroupReduceOp>(
        loc, reduceOp->getResult(0).getType(), acc[i], *reduceKind,
        numLaneToReduce);
  }

  return true;
}

std::string TargetInfo::getMulhiFuncName(Type resultElementTy) const {
  std::string funcName =
      resultElementTy.isInteger(32) ? "__imf_umulhi" : "__imf_umul64hi";
  return funcName;
}

void TargetInfo::printf(RewriterBase &rewriter, Value formatStrStart,
                        int /*formatStrByteCount*/, ValueRange args) const {
  auto *ctx = rewriter.getContext();
  Type ptr = ptr_ty(ctx);
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto funcOp = LLVM::intel::getSpirvPrintfDeclaration(rewriter);
  auto loc = UnknownLoc::get(ctx);

  SmallVector<Value> operands;
  operands.push_back(formatStrStart);
  for (auto arg : args) {
    operands.push_back(arg);
  }
  call(funcOp, operands);
}

void TargetInfo::printf(RewriterBase &rewriter, StringRef msg,
                        ValueRange args) const {
  assert(!msg.empty() && "printf with empty string not supported");
  llvm::SmallString<64> msgNewline(msg);
  msgNewline.push_back('\n');
  msgNewline.push_back('\0');
  Value msgValue = LLVM::intel::addStringToModule(
      UnknownLoc::get(rewriter.getContext()), rewriter, "printfFormat_",
      msgNewline, /*AddressSpace*/ TritonGEN::kUniformConstant);
  printf(rewriter, msgValue, msgNewline.size_in_bytes(), args);
}

static LLVM::LLVMFuncOp getAssertfailDeclaration(RewriterBase &rewriter) {
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

  RewriterBase::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(moduleOp.getBody());

  auto func = rewriter.create<LLVM::LLVMFuncOp>(UnknownLoc::get(ctx), funcName,
                                                funcType);
  func.setCConv(LLVM::cconv::CConv::SPIR_FUNC);
  return func;
}

void TargetInfo::assertFail(RewriterBase &rewriter, Location loc,
                            StringRef message, StringRef file, StringRef func,
                            int line) const {
  auto funcOp = getAssertfailDeclaration(rewriter);
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  unsigned addrSpace = TritonGEN::TritonGENMemorySpace::kCrossWorkgroup;
  llvm::SmallString<64> messageString(message), fileString(file),
      funcString(func);
  messageString.push_back('\0');
  fileString.push_back('\0');
  funcString.push_back('\0');
  Value messageStringVal = LLVM::intel::addStringToModule(
      loc, rewriter, "assertMessage_", messageString, addrSpace);
  Value fileStringVal = LLVM::intel::addStringToModule(
      loc, rewriter, "assertFile_", fileString, addrSpace);
  Value funcStringVal = LLVM::intel::addStringToModule(
      loc, rewriter, "assertFunc_", funcString, addrSpace);
  Value lineNumber = i32_val(line);

  auto *ctx = rewriter.getContext();
  SmallVector<Value> operands;
  Value messageStringPtr = addrspacecast(
      ptr_ty(ctx, TritonGEN::TritonGENMemorySpace::kGeneric), messageStringVal);
  Value fileStringPtr = addrspacecast(
      ptr_ty(ctx, TritonGEN::TritonGENMemorySpace::kGeneric), fileStringVal);
  Value funcStringPtr = addrspacecast(
      ptr_ty(ctx, TritonGEN::TritonGENMemorySpace::kGeneric), funcStringVal);
  operands = {messageStringPtr, fileStringPtr, lineNumber, funcStringPtr};
  auto ret = call(funcOp, operands);
  ret.setCConv(LLVM::cconv::CConv::SPIR_FUNC);
}

} // namespace mlir::triton::intel
