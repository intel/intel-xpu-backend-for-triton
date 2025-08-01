//===- TargetInfo.cpp - Target dependent information ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TargetInfo.h"
#include "Dialect/TritonIntelGPU/IR/Utils.h"
#include "SPIRVTargetInfo.h"
#include "Utility.h"

using namespace mlir;

namespace mlir::triton::intel {

bool TargetInfo::supportMaximumMinimum() const { return false; }
Value TargetInfo::ballot(RewriterBase &rewriter, Location loc, Type type,
                         Value cmp) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  // Emulate vote.ballot.sync behavior using shift, shuffle, and or.
  // TODO: check for more efficient solution.
  auto mod = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  Value threadId = getThreadId(rewriter, loc);
  int numThreadPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
  Value laneId = b.and_(threadId, b.i32_val(numThreadPerWarp - 1));
  Value reduced_val = b.shl(b.select(cmp, b.i32_val(1), b.i32_val(0)), laneId);
  for (int offs = 1; offs < numThreadPerWarp; offs = offs << 1) {
    Value other_val = LLVM::intel::shuffleXor(loc, rewriter, reduced_val, offs);
    reduced_val = b.or_(reduced_val, other_val);
  }
  return reduced_val;
}

Value TargetInfo::getClusterCTAId(RewriterBase &rewriter, Location loc) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  // Clusters of thread blocks aren't supported.
  return b.i32_val(0);
}

void TargetInfo::storeDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              std::optional<Value> ctaId, Value val,
                              Value pred) const {
  LLVM::intel::createPredicatedBlock(rewriter, loc, pred, [&] {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    b.store(val, ptr);
    return ArrayRef<Value>();
  });
}

bool TargetInfo::canUseStMatrix(RankedTensorType tensorTy,
                                ArrayRef<unsigned> repShape,
                                ArrayRef<unsigned> paddedRepShape,
                                ArrayRef<unsigned> order,
                                int swizzleByteSize) const {
  return false;
}

void TargetInfo::storeMatrixShared(RewriterBase &rewriter, Location loc,
                                   Value ptr, Value val) const {
  llvm::report_fatal_error("IntelGPU does not support stmatrix");
}

Value TargetInfo::loadDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              std::optional<Value> ctaId, Type elemTy,
                              Value pred, Operation *localLoadOp) const {
  assert(cast<mlir::LLVM::LLVMPointerType>(ptr.getType()).getAddressSpace() ==
             3 &&
         "Invalid addr space for loadShared");
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value undef = b.undef(elemTy);
  Block &endBlock = LLVM::intel::createPredicatedBlock(
      rewriter, loc, pred, SmallVector<Value, 1>{undef}, [&] {
        Value ret = b.load(elemTy, ptr);
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

  auto mod = op->getParentOfType<ModuleOp>();
  unsigned warpSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);

  if (!isSupportedWarpReduceOp(reduceOp, numLaneToReduce, warpSize))
    return false;

  for (unsigned i = 0; i < acc.size(); ++i) {
    acc[i] = genWarpReduce(rewriter, loc, acc[i], reduceOp, numLaneToReduce,
                           warpSize);
  }

  return true;
}

std::string TargetInfo::getMulhiFuncName(Type resultElementTy) const {
  std::string funcName =
      resultElementTy.isInteger(32) ? "__imf_umulhi" : "__imf_umul64hi";
  return funcName;
}

Value printfPromoteValue(RewriterBase &rewriter, Value value, bool isSigned) {
  auto type = value.getType();
  if (isa<IntegerType>(type) && type.getIntOrFloatBitWidth() == 1) {
    // FIXME: There is some problem when using i1 type now,
    // remove this code once IGC fix the problem.
    TritonLLVMOpBuilder b(rewriter.getUnknownLoc(), rewriter);
    return b.zext(i8_ty, value);
  } else if (type.isIntOrIndex() && type.getIntOrFloatBitWidth() < 32) {
    TritonLLVMOpBuilder b(rewriter.getUnknownLoc(), rewriter);
    if (isSigned) {
      return b.sext(i32_ty, value);
    } else {
      return b.zext(i32_ty, value);
    }
  } else {
    return value;
  }
}

// declare __spirv_ocl_printf(i8*, ...) as external function
static LLVM::LLVMFuncOp getSpirvPrintfDeclaration(RewriterBase &rewriter) {
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

void TargetInfo::printf(RewriterBase &rewriter, Value formatStrStart,
                        int /*formatStrByteCount*/, ValueRange args,
                        ArrayRef<bool> isSigned) const {
  auto *ctx = rewriter.getContext();
  Type ptr = ptr_ty(ctx);
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto funcOp = getSpirvPrintfDeclaration(rewriter);
  auto loc = UnknownLoc::get(ctx);
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  SmallVector<Value> operands;
  operands.push_back(formatStrStart);
  for (auto [i, arg] : llvm::enumerate(args)) {
    operands.push_back(printfPromoteValue(
        rewriter, arg, isSigned.empty() ? true : isSigned[i]));
  }
  auto callOp = b.call(funcOp, operands);
  callOp.setCConv(triton::gpu::intel::getRequiredCConv(callOp));
}

void TargetInfo::printf(RewriterBase &rewriter, StringRef msg, ValueRange args,
                        ArrayRef<bool> isSigned) const {
  assert(!msg.empty() && "printf with empty string not supported");
  llvm::SmallString<64> msgNewline(msg);
  msgNewline.push_back('\n');
  msgNewline.push_back('\0');
  Value msgValue = getGlobalStringStart(
      rewriter.getUnknownLoc(), rewriter, "printfFormat_", msgNewline,
      /*addressSpace=*/TritonGEN::kUniformConstant);
  printf(rewriter, msgValue, msgNewline.size_in_bytes(), args, isSigned);
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
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto funcOp = getAssertfailDeclaration(rewriter);
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  unsigned addrSpace = TritonGEN::TritonGENMemorySpace::kCrossWorkgroup;
  llvm::SmallString<64> messageString(message), fileString(file),
      funcString(func);
  messageString.push_back('\0');
  fileString.push_back('\0');
  funcString.push_back('\0');
  Value messageStringVal =
      getGlobalStringStart(loc, rewriter, "assertMessage_", messageString,
                           /*addressSpace=*/TritonGEN::kCrossWorkgroup);
  Value fileStringVal =
      getGlobalStringStart(loc, rewriter, "assertFile_", fileString,
                           /*addressSpace=*/TritonGEN::kCrossWorkgroup);
  Value funcStringVal =
      getGlobalStringStart(loc, rewriter, "assertFunc_", funcString,
                           /*addressSpace=*/TritonGEN::kCrossWorkgroup);
  Value lineNumber = b.i32_val(line);

  auto *ctx = rewriter.getContext();
  SmallVector<Value> operands;
  Value messageStringPtr = b.addrspacecast(
      ptr_ty(ctx, TritonGEN::TritonGENMemorySpace::kGeneric), messageStringVal);
  Value fileStringPtr = b.addrspacecast(
      ptr_ty(ctx, TritonGEN::TritonGENMemorySpace::kGeneric), fileStringVal);
  Value funcStringPtr = b.addrspacecast(
      ptr_ty(ctx, TritonGEN::TritonGENMemorySpace::kGeneric), funcStringVal);
  operands = {messageStringPtr, fileStringPtr, lineNumber, funcStringPtr};
  auto ret = b.call(funcOp, operands);
  ret.setCConv(LLVM::cconv::CConv::SPIR_FUNC);
}

int TargetInfo::getSharedAddressSpace() const {
  return TritonGEN::TritonGENMemorySpace::kWorkgroup;
}

bool TargetInfo::supportVectorizedAtomics() const {
  // Note: not currently tested or used, but AMD generally supports vectorized
  // atomics.
  return true;
}

int TargetInfo::getAddressSpace(Attribute addressSpace) const {
  int spaceId = 0;
  if (isa<triton::gpu::SharedMemorySpaceAttr>(addressSpace)) {
    spaceId = 3;
  } else {
    llvm::report_fatal_error("Only support SharedMemorySpace for now");
  }
  return spaceId;
}

Value TargetInfo::getGlobalStringStart(Location loc, RewriterBase &rewriter,
                                       StringRef name, StringRef value,
                                       unsigned addressSpace) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  LLVM::GlobalOp global =
      getGlobalString(loc, rewriter, name, value, addressSpace);
  MLIRContext *ctx = rewriter.getContext();
  Type globalPtrType = ptr_ty(ctx, addressSpace);
  Value globalPtr = rewriter.create<LLVM::AddressOfOp>(loc, global);
  return b.gep(globalPtrType, i8_ty, globalPtr, LLVM::GEPArg{0});
}

LLVM::GlobalOp TargetInfo::getGlobalString(Location loc, RewriterBase &rewriter,
                                           StringRef name, StringRef value,
                                           unsigned addressSpace) const {
  StringAttr valueAttr = rewriter.getStringAttr(value);
  std::pair<unsigned, StringAttr> cacheKey{addressSpace, valueAttr};
  auto pos = globals.find(cacheKey);
  if (pos != globals.end())
    return pos->second;

  ModuleOp moduleOp = rewriter.getInsertionPoint()->getParentOfType<ModuleOp>();

  llvm::SmallString<64> contentStr(value);
  size_t contentSize = contentStr.size_in_bytes();
  auto globalType = LLVM::LLVMArrayType::get(i8_ty, contentSize);

  auto createGlobal = [&](StringRef name) {
    RewriterBase::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    return rewriter.create<LLVM::GlobalOp>(
        rewriter.getUnknownLoc(), globalType,
        /*isConstant=*/true, LLVM::Linkage::Internal, name, valueAttr,
        /*alignment=*/0, addressSpace);
  };

  LLVM::GlobalOp global =
      moduleOp.lookupSymbol(name)
          ? createGlobal(Twine{name}.concat(Twine{globals.size()}).str())
          : createGlobal(name);

  globals.try_emplace(cacheKey, global);

  return global;
}

std::unique_ptr<TargetInfo> createTargetInfo(ModuleOp mod) {
  if (triton::gpu::intel::hasSpirvTargetArch(mod))
    return std::unique_ptr<TargetInfo>(new SPIRVTargetInfo());
  llvm_unreachable("createTargetInfo: unsupported target arch");
}

} // namespace mlir::triton::intel
