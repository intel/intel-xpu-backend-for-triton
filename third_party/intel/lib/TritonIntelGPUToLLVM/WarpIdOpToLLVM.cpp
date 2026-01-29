#include "PatternTritonGPUOpToLLVM.h"

namespace {

using namespace mlir;
using namespace mlir::triton;

struct WarpIdOpPattern : public ConvertOpToLLVMPattern<triton::gpu::WarpIdOp> {
public:
  WarpIdOpPattern(LLVMTypeConverter &typeConverter, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::gpu::WarpIdOp>(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(triton::gpu::WarpIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    int threadsPerWarp = triton::gpu::lookupThreadsPerWarp(rewriter);
    Value warpSizeVal = b.i32_val(threadsPerWarp);
    Value tid = getThreadId(rewriter, loc);
    Value warpId = b.udiv(tid, warpSizeVal);
    rewriter.replaceOp(op, warpId);
    return success();
  }
};

} // namespace

void mlir::triton::intel::populateWarpIdOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<WarpIdOpPattern>(typeConverter, benefit);
}
