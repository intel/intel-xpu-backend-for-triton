#include "PatternTritonGPUOpToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "triton/Dialect/NVGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace {
struct RegAllocOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::RegAllocOp> {
  using ConvertOpToLLVMPattern<
      triton::nvidia_gpu::RegAllocOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::RegAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    rewriter.replaceOpWithNewOp<triton::nvgpu::RegAllocOp>(
        op, adaptor.getRegCount());
    return success();
  }
};

struct RegDeallocOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::RegDeallocOp> {
  using ConvertOpToLLVMPattern<
      triton::nvidia_gpu::RegDeallocOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::RegDeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    rewriter.replaceOpWithNewOp<triton::nvgpu::RegDeallocOp>(
        op, adaptor.getRegCount());
    return success();
  }
};
} // namespace

void mlir::triton::populateRegReallocOpToLLVMPatterns(
<<<<<<< HEAD
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int numWarps, ModuleAxisInfoAnalysis &axisInfoAnalysis, Target target,
    PatternBenefit benefit) {
  patterns.add<RegAllocOpConversion>(typeConverter, target, benefit);
  patterns.add<RegDeallocOpConversion>(typeConverter, target, benefit);
=======
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns, int numWarps,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, PatternBenefit benefit) {
  patterns.add<RegAllocOpConversion>(typeConverter, benefit);
  patterns.add<RegDeallocOpConversion>(typeConverter, benefit);
>>>>>>> 2dd9d74527f431e5e822b8e67c01900e4d0bfef3
  return;
}
