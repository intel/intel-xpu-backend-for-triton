#include "PatternTritonGPUOpToLLVM.h"

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

/// Custom lowering for converting arith ops to LLVMIR dialect.
/// These ops lowering ONLY works for the advanced path.
/// This is a temporary solution until we have done upstreaming or have other
/// proper lowering strategy for these ops.

namespace {
// FIXME: Remove this lowering when upstream ConstantOpLowering has such
// lowering.
class ArithConstantOpLowering
    : public ConvertTritonGPUOpToLLVMPattern<mlir::arith::ConstantOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      mlir::arith::ConstantOp>::ConvertTritonGPUOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(mlir::arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto srcType = dyn_cast<RankedTensorType>(op.getType());
    if (!srcType || srcType.getNumElements() == 1)
      return failure();

    Type dstType = getTypeConverter()->convertType(srcType);
    if (!dyn_cast_or_null<VectorType>(dstType))
      return failure();

    auto dstElementsAttr = dyn_cast<DenseElementsAttr>(op.getValue());
    if (!dstElementsAttr || !dstElementsAttr.isSplat())
      return failure();

    auto vecType = cast<VectorType>(dstType);
    rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(
        op, dstType, dstElementsAttr.resizeSplat(vecType));
    return success();
  }
};

// FIXME: This optimization should be done in IGC.
class ArithDivFOpLowering
    : public ConvertTritonGPUOpToLLVMPattern<mlir::arith::DivFOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      mlir::arith::DivFOp>::ConvertTritonGPUOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(mlir::arith::DivFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = dyn_cast<ShapedType>(op.getType());
    if (!srcType)
      return failure();

    Type dstType = getTypeConverter()->convertType(srcType);
    if (!dyn_cast_or_null<VectorType>(dstType))
      return failure();

    Location loc = op->getLoc();
    auto vecType = cast<VectorType>(dstType);
    auto attr = rewriter.getFloatAttr(vecType.getElementType(), 1.0);
    auto dstAttr = DenseElementsAttr::get(vecType, attr.getValue());
    auto one = rewriter.create<LLVM::ConstantOp>(loc, dstType, dstAttr);
    auto rcp =
        rewriter.create<LLVM::FDivOp>(loc, dstType, one, adaptor.getRhs());
    rewriter.replaceOpWithNewOp<LLVM::FMulOp>(op, dstType, adaptor.getLhs(),
                                              rcp);
    return success();
  }
};
} // namespace

void mlir::triton::intel::populateArithOpsToLLVMPatterns(
    TritonIntelGPUToLLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<ArithConstantOpLowering>(typeConverter, benefit);
  patterns.add<ArithDivFOpLowering>(typeConverter, benefit);
}
