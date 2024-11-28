//===- TargetInfo.cpp - Target dependent information ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TargetInfo.h"
#include "intel/include/TritonIntelGPUToLLVM/vISAAsmFormat.h"

#include "SPIRVSubgroupOps.h"
#include "Utility.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "llvm/ADT/TypeSwitch.h"

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
                              Value pred) const {
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

namespace {

template <typename GroupOp>
Value createSPIRVGroupOp(RewriterBase &rewriter, Location loc, Type resultTy,
                         Value acc, unsigned numLanesToReduce,
                         unsigned warpSize) {
  auto spvGroupOp = spirv::GroupOperation::Reduce;
  Value clusterSize;
  if (numLanesToReduce != warpSize) {
    spvGroupOp = spirv::GroupOperation::ClusteredReduce;
    clusterSize = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32Type(),
        rewriter.getI32IntegerAttr(numLanesToReduce));
  }

  Value result = rewriter.create<GroupOp>(loc, resultTy, spirv::Scope::Subgroup,
                                          spvGroupOp, acc, clusterSize);
  return result;
}

Value warpReduceHelper(RewriterBase &rewriter, Location loc, Value acc,
                       Operation *reduceOp, unsigned numLanesToReduce,
                       unsigned warpSize) {
  auto resultType = reduceOp->getResult(0).getType();
  // Use bit-equivalent logical operation for Boolean values.
  if (resultType.isInteger(1))
    return TypeSwitch<mlir::Operation *, Value>(reduceOp)
        .Case<arith::AddIOp, arith::MulIOp, arith::MaxSIOp, arith::MaxUIOp,
              arith::MinSIOp, arith::MinUIOp, arith::AndIOp, arith::OrIOp,
              arith::XOrIOp>([&](auto groupOp) {
          return createSPIRVGroupOp<SPIRVLogicalGroupOpTy<decltype(groupOp)>>(
              rewriter, loc, resultType, acc, numLanesToReduce, warpSize);
        });
  return TypeSwitch<mlir::Operation *, Value>(reduceOp)
      .Case<arith::AddFOp, arith::AddIOp, arith::MulFOp, arith::MulIOp,
            arith::MaxSIOp, arith::MaxUIOp, arith::MinSIOp, arith::MinUIOp,
            arith::MaxNumFOp, arith::MinNumFOp, arith::AndIOp, arith::OrIOp,
            arith::XOrIOp>([&](auto groupOp) {
        return createSPIRVGroupOp<SPIRVGroupOpTy<decltype(groupOp)>>(
            rewriter, loc, resultType, acc, numLanesToReduce, warpSize);
      });
}

} // namespace

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

  auto supportedOp =
      isa<arith::AddFOp, arith::AddIOp, arith::MulFOp, arith::MulIOp,
          arith::MaxSIOp, arith::MaxUIOp, arith::MinSIOp, arith::MinUIOp,
          arith::MaxNumFOp, arith::MinNumFOp, arith::AndIOp, arith::OrIOp,
          arith::XOrIOp>(reduceOp);

  if (!supportedOp)
    return false;

  if (acc.size() == 16 && isa<arith::AddFOp, arith::MaxNumFOp>(reduceOp)) {
    VectorType reduceTy = vec_ty(acc[0].getType(), acc.size());
    Value batchedReduceVal = rewriter.create<LLVM::UndefOp>(loc, reduceTy);
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    for (unsigned i = 0; i < acc.size(); ++i) {
      batchedReduceVal =
          b.insert_element(reduceTy, batchedReduceVal, acc[i], b.i32_val(i));
    }
    VISABuilder vISABuilder;
    std::string batchedHorizontalReduce;
    if (isa<arith::AddFOp>(reduceOp)) {
      batchedHorizontalReduce =
          "{\n"
          ".decl temp_result v_type=G type=f num_elts=128 align=wordx32\n"
          // 1st round 2x8 + 2x8 -> 1x16
          "add (M1_NM, 16) temp_result(0, 0)<1>  $1(0, 0)<16;8,1> $1(0, "
          "8)<16;8,1> \n"
          "add (M1_NM, 16) temp_result(1, 0)<1>  $1(2, 0)<16;8,1> $1(2, "
          "8)<16;8,1> \n"
          "add (M1_NM, 16) temp_result(2, 0)<1>  $1(4, 0)<16;8,1> $1(4, "
          "8)<16;8,1> \n"
          "add (M1_NM, 16) temp_result(3, 0)<1>  $1(6, 0)<16;8,1> $1(6, "
          "8)<16;8,1> \n"
          "add (M1_NM, 16) temp_result(4, 0)<1>  $1(8, 0)<16;8,1> $1(8, "
          "8)<16;8,1> \n"
          "add (M1_NM, 16) temp_result(5, 0)<1>  $1(10, 0)<16;8,1> $1(10, "
          "8)<16;8,1> \n"
          "add (M1_NM, 16) temp_result(6, 0)<1>  $1(12, 0)<16;8,1> $1(12, "
          "8)<16;8,1> \n"
          "add (M1_NM, 16) temp_result(7, 0)<1>  $1(14, 0)<16;8,1> $1(14, "
          "8)<16;8,1> \n"

          // 2nd round 2x2x4 + 2x2x4 -> 1x16
          "add (M1_NM, 16) temp_result(0, 0)<1>  temp_result(0, 0)<8;4,1> "
          "temp_result(0, 4)<8;4,1> \n"
          "add (M1_NM, 16) temp_result(1, 0)<1>  temp_result(2, 0)<8;4,1> "
          "temp_result(2, 4)<8;4,1> \n"
          "add (M1_NM, 16) temp_result(2, 0)<1>  temp_result(4, 0)<8;4,1> "
          "temp_result(4, 4)<8;4,1> \n"
          "add (M1_NM, 16) temp_result(3, 0)<1>  temp_result(6, 0)<8;4,1> "
          "temp_result(6, 4)<8;4,1> \n"

          // 3rd round 4x2x2 + 4x2x2 -> 1x16
          "add (M1_NM, 16) temp_result(0, 0)<1>  temp_result(0, 0)<4;2,1> "
          "temp_result(0, 2)<4;2,1> \n"
          "add (M1_NM, 16) temp_result(1, 0)<1>  temp_result(2, 0)<4;2,1> "
          "temp_result(2, 2)<4;2,1> \n"

          // 4th round 8x2x1 + 8x2x1 -> 1x16
          "add (M1_NM, 16) $0(0, 0)<1>  temp_result(0, 0)<2;1,0> "
          "temp_result(0, 1)<2;1,0> \n"
          "}\n";
    } else if (isa<arith::MaxNumFOp>(reduceOp)) {
      batchedHorizontalReduce =
          "{\n"
          ".decl temp_result v_type=G type=f num_elts=128 align=wordx32\n"
          // 1st round 2x8 + 2x8 -> 1x16
          "max (M1_NM, 16) temp_result(0, 0)<1>  $1(0, 0)<16;8,1> $1(0, "
          "8)<16;8,1> \n"
          "max (M1_NM, 16) temp_result(1, 0)<1>  $1(2, 0)<16;8,1> $1(2, "
          "8)<16;8,1> \n"
          "max (M1_NM, 16) temp_result(2, 0)<1>  $1(4, 0)<16;8,1> $1(4, "
          "8)<16;8,1> \n"
          "max (M1_NM, 16) temp_result(3, 0)<1>  $1(6, 0)<16;8,1> $1(6, "
          "8)<16;8,1> \n"
          "max (M1_NM, 16) temp_result(4, 0)<1>  $1(8, 0)<16;8,1> $1(8, "
          "8)<16;8,1> \n"
          "max (M1_NM, 16) temp_result(5, 0)<1>  $1(10, 0)<16;8,1> $1(10, "
          "8)<16;8,1> \n"
          "max (M1_NM, 16) temp_result(6, 0)<1>  $1(12, 0)<16;8,1> $1(12, "
          "8)<16;8,1> \n"
          "max (M1_NM, 16) temp_result(7, 0)<1>  $1(14, 0)<16;8,1> $1(14, "
          "8)<16;8,1> \n"

          // 2nd round 2x2x4 + 2x2x4 -> 1x16
          "max (M1_NM, 16) temp_result(0, 0)<1>  temp_result(0, 0)<8;4,1> "
          "temp_result(0, 4)<8;4,1> \n"
          "max (M1_NM, 16) temp_result(1, 0)<1>  temp_result(2, 0)<8;4,1> "
          "temp_result(2, 4)<8;4,1> \n"
          "max (M1_NM, 16) temp_result(2, 0)<1>  temp_result(4, 0)<8;4,1> "
          "temp_result(4, 4)<8;4,1> \n"
          "max (M1_NM, 16) temp_result(3, 0)<1>  temp_result(6, 0)<8;4,1> "
          "temp_result(6, 4)<8;4,1> \n"

          // 3rd round 4x2x2 + 4x2x2 -> 1x16
          "max (M1_NM, 16) temp_result(0, 0)<1>  temp_result(0, 0)<4;2,1> "
          "temp_result(0, 2)<4;2,1> \n"
          "max (M1_NM, 16) temp_result(1, 0)<1>  temp_result(2, 0)<4;2,1> "
          "temp_result(2, 2)<4;2,1> \n"

          // 4th round 8x2x1 + 8x2x1 -> 1x16
          "max (M1_NM, 16) $0(0, 0)<1>  temp_result(0, 0)<2;1,0> "
          "temp_result(0, 1)<2;1,0> \n"
          "}\n";
    } else {
      llvm_unreachable("batched reduce WIP");
    }

    auto &bReduceOp = *vISABuilder.create<>(batchedHorizontalReduce);
    //    auto res = vISABuilder.newOperand("=rw.u");
    auto res = vISABuilder.newOperand("=rw");
    auto in = vISABuilder.newOperand(batchedReduceVal, "rw");
    bReduceOp({res, in}, /*onlyAttachMLIRArgs=*/true);
    Type resultTy = reduceTy.getElementType();
    Value ret = vISABuilder.launch(rewriter, loc, resultTy, true);
    for (unsigned i = 0; i < acc.size(); ++i) {
      // The output of the inline vISA has to be the non-uniform value.
      // Have to shuffle the result to get the reduce value.
      acc[i] = LLVM::intel::shuffleIdx(loc, rewriter, ret, i);
    }

  } else {
    auto mod = op->getParentOfType<ModuleOp>();
    unsigned warpSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);

    for (unsigned i = 0; i < acc.size(); ++i) {
      acc[i] = warpReduceHelper(rewriter, loc, acc[i], reduceOp,
                                numLaneToReduce, warpSize);
    }
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
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  SmallVector<Value> operands;
  operands.push_back(formatStrStart);
  for (auto arg : args) {
    operands.push_back(arg);
  }
  b.call(funcOp, operands);
}

void TargetInfo::printf(RewriterBase &rewriter, StringRef msg,
                        ValueRange args) const {
  assert(!msg.empty() && "printf with empty string not supported");
  llvm::SmallString<64> msgNewline(msg);
  msgNewline.push_back('\n');
  msgNewline.push_back('\0');
  Value msgValue = getGlobalStringStart(
      rewriter.getUnknownLoc(), rewriter, "printfFormat_", msgNewline,
      /*addressSpace=*/TritonGEN::kUniformConstant);
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

Value TargetInfo::getStackPointer(RewriterBase &rewriter,
                                  FunctionOpInterface funcOp) const {
  auto mod = funcOp->getParentOfType<ModuleOp>();
  LLVM::LLVMPointerType ptrTy = ptr_ty(
      rewriter.getContext(), TritonGEN::TritonGENMemorySpace::kWorkgroup);
  if (mod->getAttrOfType<IntegerAttr>("ttg.shared").getInt() == 0)
    return rewriter.create<LLVM::PoisonOp>(funcOp.getLoc(), ptrTy);
  return funcOp.getArgument(funcOp.getNumArguments() - 1);
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

} // namespace mlir::triton::intel
