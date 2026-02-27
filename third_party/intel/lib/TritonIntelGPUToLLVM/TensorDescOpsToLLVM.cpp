#include "PatternTritonGPUOpToLLVM.h"

using namespace mlir;
using namespace mlir::triton;

namespace {
struct MakeTensorDescOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::MakeTensorDescOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::MakeTensorDescOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::MakeTensorDescOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // The converted TensorDescType struct layout is:
    // struct { shape0, shape1, ..., stride0, stride1, ..., base_ptr }
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    auto shapes = adaptor.getShape();
    auto strides = adaptor.getStrides();
    auto base = adaptor.getBase();
    auto result = op.getResult();

    SmallVector<Value> elems;

    // Shapes: extend from i32 to i64
    for (auto shape : shapes)
      elems.push_back(b.sext(i64_ty, shape));

    // Strides: already i64
    for (auto stride : strides)
      elems.push_back(stride);

    // Base pointer
    elems.push_back(base);

    auto newValue = packLLElements(op.getLoc(), getTypeConverter(), elems,
                                   rewriter, result.getType());
    rewriter.replaceOp(op, newValue);
    return success();
  }
};
} // namespace

void mlir::triton::intel::populateTensorDescOpsToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<MakeTensorDescOpConversion>(typeConverter, benefit);
}
