//===- MemoryOpToLLVM.cpp - Intel memory-op lowering overrides ------------===//
//
// Intel override of the shared `tt.atomic_poll` lowering: the upstream pattern
// polls with a native atomic load of the operand type, but 16-bit atomics are
// unsupported on Intel GPUs and crash in ocloc/IGC (GitHub issue #7390). This
// widens the 16-bit case to a 4-byte-aligned 32-bit atomic load; every other
// case defers to the upstream pattern via a lower PatternBenefit.
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

    unsigned bitWidth = adaptor.getExpected().getType().getIntOrFloatBitWidth();
    bool support16BitAtomics = moduleOp->hasAttr(
        ttgi::TritonIntelGPUDialect::getSupport16BitAtomicsAttrName());
    if (bitWidth != 16 || support16BitAtomics)
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
    // Widen the unsupported 16-bit atomic load to a 4-byte-aligned 32-bit one
    // and extract the target half (mirrors `emulate16BitsCAS` in
    // LoadStoreOpToLLVM.cpp); ordering and sync scope match the upstream poll.
    Value expected = adaptor.getExpected();
    Value intPtr = b.ptrtoint(i64_ty, adaptor.getPtr());
    Value lowPtrBits = b.and_(intPtr, b.i64_val(3));
    Value elemIndex = b.trunc(i32_ty, b.lshr(lowPtrBits, b.i64_val(1)));
    Value alignedPtr = b.inttoptr(adaptor.getPtr().getType(),
                                  b.sub(intPtr, lowPtrBits).getResult());
    Value word = LLVM::LoadOp::create(
        rewriter, loc, i32_ty, alignedPtr, /*alignment=*/4,
        /*isVolatile=*/false, /*isNonTemporal=*/false, /*isInvariant=*/false,
        /*isInvariantGroup=*/false, LLVM::AtomicOrdering::monotonic, syncScope);
    Value loaded =
        b.extract_element(b.bitcast(word, vec_ty(i16_ty, 2)), elemIndex);
    Value pollMatched = b.icmp_eq(loaded, expected);
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

} // namespace

void mlir::triton::intel::populateMemoryOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, const TargetInfoBase &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<AtomicPollOpConversion>(typeConverter, targetInfo,
                                       benefit.getBenefit() + 1);
}
