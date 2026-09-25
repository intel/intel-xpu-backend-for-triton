//===- MemoryOpToLLVM.cpp - Intel memory-op lowering overrides ------------===//
//
// Intel GPUs have no usable sub-word atomic: 8-bit ones have no SPIR-V builtin
// and 16-bit integer ones need SPV_INTEL_16bit_atomics, which not every target
// supports (issues #7390, #8027). Emulate them over the containing 4-byte word;
// anything else defers to the upstream patterns via a lower PatternBenefit.
//
//===----------------------------------------------------------------------===//

#include "Dialect/TritonIntelGPU/IR/Dialect.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"

#include "PatternTritonGPUOpToLLVM.h"

#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace {

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;

// 16-bit floats map onto the OpenCL half/bfloat atomic builtins, so only
// integers need emulating.
static bool needsWordEmulation(ModuleOp moduleOp, Type elemTy) {
  if (!elemTy.isInteger())
    return false;
  unsigned bitWidth = elemTy.getIntOrFloatBitWidth();
  return bitWidth == 8 ||
         (bitWidth == 16 &&
          !moduleOp->hasAttr(
              ttgi::TritonIntelGPUDialect::getSupport16BitAtomicsAttrName()));
}

// The word containing a sub-word address, seen as a vector of sub-words, plus
// the index of the addressed one. Assumes the address is naturally aligned, as
// the atomic it replaces does.
struct WordSlot {
  Value alignedPtr;
  Value index;
  Type vecTy;
};

static WordSlot getWordSlot(RewriterBase &rewriter, Location loc, Value ptr,
                            Type subWordTy) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  unsigned byteWidth = subWordTy.getIntOrFloatBitWidth() / 8;
  Value intPtr = b.ptrtoint(i64_ty, ptr);
  Value byteOffset = b.and_(intPtr, b.i64_val(3));
  Value index = b.trunc(i32_ty, byteOffset);
  if (unsigned shift = llvm::Log2_32(byteWidth))
    index = b.lshr(index, b.i32_val(shift));
  Value alignedPtr =
      b.inttoptr(ptr.getType(), b.sub(intPtr, byteOffset).getResult());
  return {alignedPtr, index, vec_ty(subWordTy, 4 / byteWidth)};
}

static Value loadWord(RewriterBase &rewriter, Location loc, Value alignedPtr,
                      StringRef syncScope) {
  return LLVM::LoadOp::create(
      rewriter, loc, i32_ty, alignedPtr, /*alignment=*/4,
      /*isVolatile=*/false, /*isNonTemporal=*/false, /*isInvariant=*/false,
      /*isInvariantGroup=*/false, LLVM::AtomicOrdering::monotonic, syncScope);
}

// Load the containing word and extract the addressed sub-word.
static Value emulateAtomicLoad(ConversionPatternRewriter &rewriter,
                               Location loc, Value ptr, Type elemTy, Value pred,
                               StringRef syncScope) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return emitPredicated(
             rewriter, loc, pred, ValueRange{b.undef(elemTy)},
             [&]() -> SmallVector<Value> {
               WordSlot slot = getWordSlot(rewriter, loc, ptr, elemTy);
               Value word = loadWord(rewriter, loc, slot.alignedPtr, syncScope);
               return {
                   b.extract_element(b.bitcast(word, slot.vecTy), slot.index)};
             })
      .front();
}

// CAS loop over the containing word, so that concurrent updates of the
// neighbouring sub-words are not lost.
static void emulateAtomicStore(ConversionPatternRewriter &rewriter,
                               Location loc, Value ptr, Value value, Value pred,
                               StringRef syncScope) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Block *currentBlock = rewriter.getInsertionBlock();
  Block *doneBlock =
      rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
  Region *region = currentBlock->getParent();
  Block *initBlock = rewriter.createBlock(region, Region::iterator(doneBlock));
  Block *casBlock = rewriter.createBlock(region, Region::iterator(doneBlock));
  BlockArgument word = casBlock->addArgument(i32_ty, loc);

  rewriter.setInsertionPointToEnd(currentBlock);
  WordSlot slot = getWordSlot(rewriter, loc, ptr, value.getType());
  if (pred)
    LLVM::CondBrOp::create(rewriter, loc, pred, initBlock, ValueRange{},
                           doneBlock, ValueRange{});
  else
    LLVM::BrOp::create(rewriter, loc, initBlock);

  rewriter.setInsertionPointToEnd(initBlock);
  Value initialWord = loadWord(rewriter, loc, slot.alignedPtr, syncScope);
  LLVM::BrOp::create(rewriter, loc, ValueRange{initialWord}, casBlock);

  rewriter.setInsertionPointToEnd(casBlock);
  Value newWord = b.bitcast(
      b.insert_element(b.bitcast(word, slot.vecTy), value, slot.index), i32_ty);
  auto cas = LLVM::AtomicCmpXchgOp::create(
      rewriter, loc, slot.alignedPtr, word, newWord,
      LLVM::AtomicOrdering::monotonic, LLVM::AtomicOrdering::monotonic,
      syncScope, /*alignment=*/4);
  Value staleWord = b.extract_val(cas, 0);
  Value stored = b.extract_val(cas, 1);
  LLVM::CondBrOp::create(rewriter, loc, stored, doneBlock, ValueRange{},
                         casBlock, ValueRange{staleWord});

  rewriter.setInsertionPointToStart(doneBlock);
}

struct AtomicPollOpConversion
    : public ConvertOpToLLVMPattern<triton::AtomicPollOp> {
  AtomicPollOpConversion(LLVMTypeConverter &converter,
                         const TargetInfoBase &targetInfo,
                         PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::AtomicPollOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::AtomicPollOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto moduleOp = op->getParentOfType<ModuleOp>();
    assert(moduleOp && "Parent ModuleOp not found for AtomicPollOp");

    Type expectedTy = op.getExpected().getType();
    if (!expectedTy.isIntOrFloat())
      return rewriter.notifyMatchFailure(
          op, "tensor atomic_poll defers to upstream lowering");

    if (!needsWordEmulation(moduleOp, expectedTy))
      return rewriter.notifyMatchFailure(
          op, "defer to upstream atomic_poll lowering");

    int numCTAs = TritonGPUDialect::getNumCTAs(moduleOp);
    if (numCTAs != 1 && !targetInfo.isCuda())
      return rewriter.notifyMatchFailure(
          op, "multi-CTA atomic_poll requires cross-CTA shared memory");

    insertAtomicOrderingBarriers(op, op.getSem(),
                                 /*emitBarrierAfter=*/false, rewriter,
                                 targetInfo);

    auto freeVarMasks = getFreeVariableMasks(op.getPtr().getType());
    Value threadPred =
        emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);
    StringRef syncScope = targetInfo.getAtomicSyncScope(op.getScope());

    Block *currentBlock = rewriter.getInsertionBlock();
    Block *doneBlock = currentBlock->splitBlock(rewriter.getInsertionPoint());
    Region *region = currentBlock->getParent();
    Block *pollInitBlock =
        rewriter.createBlock(region, Region::iterator(doneBlock));
    Block *pollLoopBlock =
        rewriter.createBlock(region, Region::iterator(doneBlock));
    Block *pollSuccessBlock =
        rewriter.createBlock(region, Region::iterator(doneBlock));
    Block *timeoutCheckBlock =
        adaptor.getTimeout()
            ? rewriter.createBlock(region, Region::iterator(doneBlock))
            : nullptr;
    BlockArgument matched = doneBlock->addArgument(i1_ty, loc);

    rewriter.setInsertionPointToEnd(currentBlock);
    LLVM::CondBrOp::create(rewriter, loc, threadPred, pollInitBlock,
                           ValueRange{}, doneBlock, ValueRange{b.false_val()});

    rewriter.setInsertionPointToEnd(pollInitBlock);
    Value start;
    if (adaptor.getTimeout())
      start = targetInfo.getGlobalTimer(rewriter, loc);
    LLVM::BrOp::create(rewriter, loc, pollLoopBlock);

    rewriter.setInsertionPointToEnd(pollLoopBlock);
    // Widen the unsupported sub-word load; ordering and sync scope match the
    // upstream poll.
    WordSlot slot = getWordSlot(rewriter, loc, adaptor.getPtr(), expectedTy);
    Value word = loadWord(rewriter, loc, slot.alignedPtr, syncScope);
    Value loaded = b.extract_element(b.bitcast(word, slot.vecTy), slot.index);
    Value pollMatched = b.icmp_eq(loaded, adaptor.getExpected());
    if (adaptor.getTimeout()) {
      LLVM::CondBrOp::create(rewriter, loc, pollMatched, pollSuccessBlock,
                             timeoutCheckBlock);

      rewriter.setInsertionPointToEnd(timeoutCheckBlock);
      Value elapsed = b.sub(targetInfo.getGlobalTimer(rewriter, loc), start);
      Value timedOut = b.icmp_uge(elapsed, adaptor.getTimeout());
      LLVM::CondBrOp::create(rewriter, loc, timedOut, doneBlock,
                             ValueRange{b.false_val()}, pollLoopBlock,
                             ValueRange{});
    } else {
      LLVM::CondBrOp::create(rewriter, loc, pollMatched, pollSuccessBlock,
                             pollLoopBlock);
    }

    rewriter.setInsertionPointToEnd(pollSuccessBlock);
    if (op.getSem() == triton::MemSemantic::ACQUIRE)
      LLVM::FenceOp::create(rewriter, loc, LLVM::AtomicOrdering::acquire,
                            syncScope);
    LLVM::BrOp::create(rewriter, loc, ValueRange{b.true_val()}, doneBlock);

    rewriter.setInsertionPointToStart(doneBlock);
    if (!adaptor.getTimeout()) {
      if (numCTAs == 1)
        targetInfo.barrier(loc, rewriter, AddrSpace::Local);
      else
        targetInfo.clusterBarrier(loc, rewriter, op);
      rewriter.replaceOp(op, b.true_val());
      return success();
    }

    if (op.getResult().use_empty()) {
      if (numCTAs == 1)
        targetInfo.barrier(loc, rewriter, AddrSpace::Local);
      else
        targetInfo.clusterBarrier(loc, rewriter, op);
      rewriter.eraseOp(op);
      return success();
    }

    Value atomPtr =
        LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
    atomPtr = b.bitcast(atomPtr, ptr_ty(rewriter.getContext(),
                                        targetInfo.getSharedAddressSpace()));
    targetInfo.storeShared(rewriter, loc, atomPtr, matched, threadPred);
    if (numCTAs == 1)
      targetInfo.barrier(loc, rewriter, AddrSpace::Local);
    else
      targetInfo.clusterBarrier(loc, rewriter, op);

    Value result;
    if (numCTAs == 1) {
      result = b.load(i1_ty, atomPtr);
    } else {
      result = targetInfo.loadDShared(rewriter, loc, atomPtr, b.i32_val(0),
                                      i1_ty, b.true_val());
    }
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

struct AtomicLoadOpConversion
    : public ConvertOpToLLVMPattern<triton::AtomicLoadOp> {
  AtomicLoadOpConversion(LLVMTypeConverter &converter,
                         const TargetInfoBase &targetInfo,
                         PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::AtomicLoadOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::AtomicLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type valueElemTy = getTypeConverter()->convertType(
        getElementTypeOrSelf(op.getResult().getType()));
    if (!needsWordEmulation(op->getParentOfType<ModuleOp>(), valueElemTy))
      return rewriter.notifyMatchFailure(
          op, "defer to upstream atomic_load lowering");

    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    MemSemantic sem = op.getSem();
    insertAtomicOrderingBarriers(op, sem, !atomicResultHasOrderingBarrier(op),
                                 rewriter, targetInfo);
    StringRef syncScope = targetInfo.getAtomicSyncScope(op.getScope());

    SmallVector<Value> ptrElements =
        unpackUniqueTensorElements(loc, adaptor.getPtr(), rewriter);
    SmallVector<Value> maskElements;
    if (adaptor.getMask())
      maskElements =
          unpackUniqueTensorElements(loc, adaptor.getMask(), rewriter);
    Value threadPred = emitRedundantThreadPredicate(
        getFreeVariableMasks(op.getPtr().getType()), rewriter, loc, targetInfo);

    SmallVector<Value> resultVals;
    resultVals.reserve(ptrElements.size());
    for (auto [i, ptrElement] : llvm::enumerate(ptrElements)) {
      Value pred = maskElements.empty()
                       ? threadPred
                       : maybeAnd(rewriter, loc, threadPred, maskElements[i]);
      resultVals.push_back(emulateAtomicLoad(rewriter, loc, ptrElement,
                                             valueElemTy, pred, syncScope));
    }

    if (sem == MemSemantic::ACQUIRE)
      LLVM::FenceOp::create(rewriter, loc, LLVM::AtomicOrdering::acquire,
                            syncScope);
    finalizeAtomicResults(op, rewriter, resultVals, valueElemTy, b, threadPred,
                          targetInfo, getTypeConverter());
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

struct AtomicStoreOpConversion
    : public ConvertOpToLLVMPattern<triton::AtomicStoreOp> {
  AtomicStoreOpConversion(LLVMTypeConverter &converter,
                          const TargetInfoBase &targetInfo,
                          PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::AtomicStoreOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::AtomicStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type valueElemTy = getTypeConverter()->convertType(
        getElementTypeOrSelf(op.getValue().getType()));
    if (!needsWordEmulation(op->getParentOfType<ModuleOp>(), valueElemTy))
      return rewriter.notifyMatchFailure(
          op, "defer to upstream atomic_store lowering");

    Location loc = op.getLoc();
    insertAtomicOrderingBarriers(op, op.getSem(), /*emitBarrierAfter=*/true,
                                 rewriter, targetInfo);
    StringRef syncScope = targetInfo.getAtomicSyncScope(op.getScope());

    SmallVector<Value> ptrElements =
        unpackUniqueTensorElements(loc, adaptor.getPtr(), rewriter);
    SmallVector<Value> valueElements =
        unpackUniqueTensorElements(loc, adaptor.getValue(), rewriter);
    SmallVector<Value> maskElements;
    if (adaptor.getMask())
      maskElements =
          unpackUniqueTensorElements(loc, adaptor.getMask(), rewriter);
    Value threadPred = emitRedundantThreadPredicate(
        getFreeVariableMasks(op.getPtr().getType()), rewriter, loc, targetInfo);

    if (op.getSem() == MemSemantic::RELEASE)
      LLVM::FenceOp::create(rewriter, loc, LLVM::AtomicOrdering::release,
                            syncScope);

    for (auto [i, ptrElement] : llvm::enumerate(ptrElements)) {
      Value pred = maskElements.empty()
                       ? threadPred
                       : maybeAnd(rewriter, loc, threadPred, maskElements[i]);
      emulateAtomicStore(rewriter, loc, ptrElement, valueElements[i], pred,
                         syncScope);
    }
    rewriter.eraseOp(op);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

} // namespace

void mlir::triton::intel::populateMemoryOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, const TargetInfoBase &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<AtomicLoadOpConversion, AtomicPollOpConversion,
               AtomicStoreOpConversion>(typeConverter, targetInfo,
                                        benefit.getBenefit() + 1);
}
