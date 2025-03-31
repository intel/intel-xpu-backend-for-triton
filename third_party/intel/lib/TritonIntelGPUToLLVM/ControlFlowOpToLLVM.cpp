#include "PatternTritonGPUOpToLLVM.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Utils.h"

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

} // namespace

void mlir::triton::intel::populateControlFlowOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<FixCallCConv>(typeConverter);
  mlir::triton::populateControlFlowOpToLLVMPattern(typeConverter, patterns,
                                                   targetInfo, benefit);
}
