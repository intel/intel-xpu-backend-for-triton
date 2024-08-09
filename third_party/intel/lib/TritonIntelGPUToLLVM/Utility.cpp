//===- Utility.cpp - Code generation utilities ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Utility.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/LinearLayoutConversions.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Utility.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir::LLVM::intel {

static Value shuffleCommon(Location loc, RewriterBase &rewriter, Value val,
                           Value i, TritonGEN::ShflKind mode) {
  Type type = val.getType();
  return rewriter.create<TritonGEN::SubGroupShuffleOp>(loc, type, val, i, mode);
}

Value loadShared(ConversionPatternRewriter &rewriter, Location loc, Value ptr,
                 Type elemTy, Value pred) {
  assert(cast<LLVMPointerType>(ptr.getType()).getAddressSpace() == 3 &&
         "Invalid addr space for loadShared");
  Value undef = undef(elemTy);
  Block &endBlock = createPredicatedBlock(rewriter, loc, pred,
                                          SmallVector<Value, 1>{undef}, [&] {
                                            Value ret = load(elemTy, ptr);
                                            return SmallVector<Value, 1>{ret};
                                          });
  return *endBlock.args_begin();
}

Value shuffleXor(Location loc, RewriterBase &rewriter, Value val, int i) {
  return shuffleCommon(loc, rewriter, val, i32_val(i),
                       TritonGEN::ShflKind::XOR);
}

Value shuffleUp(Location loc, RewriterBase &rewriter, Value val, int i) {
  return shuffleCommon(loc, rewriter, val, i32_val(i), TritonGEN::ShflKind::UP);
}

Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, int i) {
  return shuffleIdx(loc, rewriter, val, i32_val(i));
}

Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, Value i) {
  return shuffleCommon(loc, rewriter, val, i, TritonGEN::ShflKind::IDX);
}

Value addStringToModule(Location loc, RewriterBase &rewriter, StringRef key,
                        StringRef content, unsigned addressSpace) {
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto ctx = moduleOp.getContext();
  unsigned stringNumber = 0;
  SmallString<16> stringConstName;
  do {
    stringConstName.clear();
    (key + Twine(stringNumber++)).toStringRef(stringConstName);
  } while (moduleOp.lookupSymbol(stringConstName));

  llvm::SmallString<64> contentStr(content);
  size_t contentSize = contentStr.size_in_bytes();
  auto globalType = LLVM::LLVMArrayType::get(i8_ty, contentSize);

  LLVM::GlobalOp global;
  {
    RewriterBase::InsertionGuard guard(rewriter);
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

} // namespace mlir::LLVM::intel

namespace mlir::triton::intel {
bool emitTransferBetweenDPASAndShared(
    RankedTensorType registerTy, MemDescType sharedTy, Type elemLlvmTy,
    std::optional<int32_t> maxVecElems, Value shmemBase,
    ArrayRef<Value> shmemStrides, Location loc, RewriterBase &rewriter,
    const TargetInfoBase &target,
    std::function<void(VectorType, Value /*shmemAddr*/)> perVectorCallback) {
  MLIRContext *ctx = rewriter.getContext();

  auto shape = registerTy.getShape();
  int rank = shape.size();

  StringAttr kBlock = str_attr("block");
  StringAttr kRegister = str_attr("register");
  StringAttr kLane = str_attr("lane");
  StringAttr kWarp = str_attr("warp");

  std::optional<LinearLayout> regLayout;
  if (auto dpas = dyn_cast<DpasEncodingAttr>(registerTy.getEncoding())) {
    // Default is operandC (opidx == 2)
    regLayout = triton::gpu::DPAStoLinearLayout(shape, dpas);
  } else {
    regLayout = triton::gpu::toLinearLayout(shape, registerTy.getEncoding());
  }

  std::optional<LinearLayout> sharedLayout;
  if (auto dpas = dyn_cast<DpasEncodingAttr>(sharedTy.getEncoding())) {
    sharedLayout = triton::gpu::DPAStoLinearLayout(shape, dpas);
  } else {
    sharedLayout = triton::gpu::toLinearLayout(
        shape, sharedTy.getEncoding(), elemLlvmTy.getIntOrFloatBitWidth());
  }

  if (!regLayout.has_value() || !sharedLayout.has_value()) {
    return false;
  }
  auto sharedOrder = triton::gpu::getOrder(sharedTy.getEncoding());

  // sharedLayout's in-dims are currently (offset, block).  Reshape to
  // (offsetX1, offsetX2, ..., block) so that we can apply the N-dimensional
  // shmem strides.  (The offsetX's appear in minor-to-major order.)
  auto sharedLegacy =
      cast<triton::gpu::SharedEncodingAttr>(sharedTy.getEncoding());
  SmallVector<std::pair<StringAttr, int32_t>> multiDimSharedSize;
  for (int i = 0; i < rank; i++) {
    int dim = sharedOrder[i];
    int64_t size = std::max(
        int64_t{1},
        shape[dim] / sharedLegacy.getCTALayout().getCTASplitNum()[dim]);
    multiDimSharedSize.push_back(
        {str_attr("offset" + std::to_string(dim)), size});
  }
  multiDimSharedSize.push_back({kBlock, sharedLayout->getInDimSize(kBlock)});
  sharedLayout = sharedLayout->reshapeIns(multiDimSharedSize);

  // regToSharedLayout maps from (register, lane, warp, block) to (offsetX1,
  // ..., offsetXN, block), where the offsetX's are in minor-to-major order.
  LinearLayout regToSharedLayout = regLayout->invertAndCompose(*sharedLayout);

  // TODO(jlebar): We don't currently support loading from shared memory in a
  // different CTA.  We'd need to emit `mapa.shared::cluster` instructions.
  for (int inBlock = 1; inBlock < regToSharedLayout.getInDimSize(kBlock);
       inBlock *= 2) {
    auto idx = llvm::to_vector(llvm::make_second_range(regToSharedLayout.apply(
        {{kRegister, 0}, {kLane, 0}, {kWarp, 0}, {kBlock, inBlock}})));
    // offsetX1, ..., offsetXN must all be 0.
    if (!llvm::all_of(ArrayRef(idx).drop_back(1),
                      [&](auto offset) { return offset == 0; })) {
      return false;
    }
    int32_t outBlock = idx.back();
    if (outBlock != inBlock) {
      return false;
    }
  }

  // Determine how many consecutive registers map to consecutive shmem elements
  // in out-dimension offsetN.  This is our load instruction's vector width.
  //
  // It's OK if the vector width we choose here is wider than the hardware
  // supports; LLVM will legalize it.
  //
  // TODO(jlebar): shmemStrides are Values, but most of them are usually integer
  // constants.  We could add those constant strides to the LL, and then before
  // calling getNumConsecutiveInOut(), we could flatten consecutive out-dims
  // which have known strides.  This would allow us to vectorize across multiple
  // shmem out dimensions where possible.
  const int vecElems =
      std::min(regToSharedLayout.getNumConsecutiveInOut(),
               maxVecElems.value_or(std::numeric_limits<int>::max()));

  Value threadId = getThreadId(rewriter, loc);
  Value threadsPerWarp = i32_val(regToSharedLayout.getInDimSize(kLane));
  Value laneId = urem(threadId, threadsPerWarp);
  Value warpId = udiv(threadId, threadsPerWarp);

  int numElems = regToSharedLayout.getInDimSize(kRegister);
  auto vecTy = vec_ty(elemLlvmTy, vecElems);
  auto ptrTy = ptr_ty(ctx, /*addressSpace=*/3);
  Value zero = i32_val(0);
  SmallVector<Value> ret;
  for (int i = 0; i < numElems / vecElems; i++) {
    // Get the address to load/store.  The multi-dim address is (offsetX1, ...,
    // offsetXN, block), where the offsets appear in minor-to-major order, and
    // we drop_end to drop block, which we know from above will be 0.
    auto multiDimShmemOffset =
        llvm::to_vector(llvm::drop_end(llvm::make_second_range(
            applyLinearLayout(loc, rewriter, regToSharedLayout,
                              {{kRegister, i32_val(i * vecElems)},
                               {kLane, laneId},
                               {kWarp, warpId},
                               {kBlock, zero}}))));

    // Reorder strides according to `order`.  This way they match the
    // multi-dimensional offsets in regToSharedLayout.
    Value shmemOffset = dot(rewriter, loc, multiDimShmemOffset,
                            applyPermutation(shmemStrides, sharedOrder));
    auto vecAddr = gep(ptrTy, elemLlvmTy, shmemBase, shmemOffset);
    vecAddr.setInbounds(true);
    perVectorCallback(vecTy, vecAddr);
  }
  return true;
}
} // namespace mlir::triton::intel
