#include "PatternTritonGPUOpToLLVM.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Utils.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include <algorithm>

namespace {

struct FixCallCConv : public ConvertOpToLLVMPattern<LLVM::CallOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(LLVM::CallOp op, LLVM::CallOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.startOpModification(op);
    op.setCConv(triton::gpu::intel::getRequiredCConv(op));
    rewriter.finalizeOpModification(op);
    return success();
  }
};

struct CallOpConversion : public ConvertOpToLLVMPattern<triton::CallOp> {
  CallOpConversion(LLVMTypeConverter &converter,
                   const TargetInfoBase &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::CallOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::CallOp callOp,
                  typename triton::CallOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto promotedOperands = promoteOperands(callOp, adaptor, rewriter);
    auto newCallOp =
        convertCallOpToLLVMCallOp(callOp, promotedOperands, rewriter);
    if (!newCallOp)
      return failure();
    auto results = getCallOpResults(callOp, newCallOp, rewriter);
    rewriter.replaceOp(callOp, results);
    return success();
  }

private:
  SmallVector<Value, 4>
  promoteOperands(triton::CallOp callOp,
                  typename triton::CallOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const {
    // Get the last argument of the caller, which is the current stack pointer
    // of shared memory and append it to the operands of the callOp.
    auto loc = callOp.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto caller = callOp->getParentOfType<FunctionOpInterface>();
    auto promotedOperands = this->getTypeConverter()->promoteOperands(
        callOp.getLoc(), /*opOperands=*/callOp->getOperands(),
        adaptor.getOperands(), rewriter);
    if (!caller->hasAttr("allocation.offset")) {
      auto base = LLVM::getStackPointer(rewriter, caller);
      promotedOperands.push_back(base);
    } else {
      auto base = LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, callOp);
      promotedOperands.push_back(base);
    }

    auto opOffsetAttr = callOp->getAttrOfType<mlir::IntegerAttr>(
        "ttg.global_scratch_memory_offset");
    Value opOffsetVal;
    if (opOffsetAttr) {
      auto opOffset = opOffsetAttr.getValue().getZExtValue();
      opOffsetVal = b.i32_val(opOffset);
    }

#if 0
    Value globalScratchPtr = LLVM::getGlobalScratchPtr(
        loc, rewriter, targetInfo, caller, opOffsetVal);
    auto callee = cast<FunctionOpInterface>(callOp.resolveCallable());
    auto lastArgType =
        callee.getArguments()[callee.getNumArguments() - 1].getType();
    if (lastArgType != globalScratchPtr.getType()) {
      auto zeroOp = rewriter.create<LLVM::ZeroOp>(loc, lastArgType);
      promotedOperands.push_back(zeroOp);
      return promotedOperands;
    }

    promotedOperands.push_back(globalScratchPtr);
#else
    promotedOperands.push_back(LLVM::getGlobalScratchPtr(
        loc, rewriter, targetInfo, caller, opOffsetVal));
    promotedOperands.push_back(
        LLVM::getProfileScratchPtr(loc, rewriter, caller));
#endif
    return promotedOperands;
  }
  LLVM::CallOp
  convertCallOpToLLVMCallOp(triton::CallOp callOp,
                            ArrayRef<Value> promotedOperands,
                            ConversionPatternRewriter &rewriter) const {
    // Pack the result types into a struct.
    Type packedResult = nullptr;
    unsigned numResults = callOp.getNumResults();
    auto resultTypes = llvm::to_vector<4>(callOp.getResultTypes());

    if (numResults != 0) {
      if (!(packedResult =
                this->getTypeConverter()->packFunctionResults(resultTypes)))
        return nullptr;
    }
    auto newCallOp = rewriter.create<LLVM::CallOp>(
        callOp.getLoc(), packedResult ? TypeRange(packedResult) : TypeRange(),
        promotedOperands, callOp->getAttrs());
    newCallOp.getProperties().setOpBundleSizes(
        rewriter.getDenseI32ArrayAttr({}));
    newCallOp.getProperties().setOperandSegmentSizes(
        {static_cast<int>(promotedOperands.size()), 0});
    return newCallOp;
  }

  SmallVector<Value>
  getCallOpResults(triton::CallOp callOp, LLVM::CallOp newCallOp,
                   ConversionPatternRewriter &rewriter) const {
    auto numResults = callOp.getNumResults();
    SmallVector<Value> results;
    if (numResults < 2) {
      // If < 2 results, packing did not do anything and we can just return.
      results.append(newCallOp.result_begin(), newCallOp.result_end());
    } else {
      // Otherwise, it had been converted to an operation producing a structure.
      // Extract individual results from the structure and return them as list.
      results.reserve(numResults);
      for (unsigned i = 0; i < numResults; ++i) {
        results.push_back(rewriter.create<LLVM::ExtractValueOp>(
            callOp.getLoc(), newCallOp->getResult(0), i));
      }
    }
    return results;
  }
  const TargetInfoBase &targetInfo;
};

} // namespace

void mlir::triton::intel::populateControlFlowOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<FixCallCConv>(typeConverter);
  // Overwrite the CallOpConversion pattern added by the call to
  // populateControlFlowOpToLLVMPattern.
  patterns.add<CallOpConversion>(typeConverter, targetInfo,
                                 benefit.getBenefit() + 1);
  mlir::triton::populateControlFlowOpToLLVMPattern(typeConverter, patterns,
                                                   targetInfo, benefit);
}
