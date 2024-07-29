#include "PatternTritonGPUOpToLLVM.h"

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

/// Custom lowering for converting arith ops to LLVMIR dialect.
/// These ops lowering ONLY works for Advanced Path.
/// This is a temporary solution until we have done upstreaming or have other
/// proper lowering stratefy for these ops.

namespace {
// FIXME: remove this when upstream ConstantOpLowering has such lowering
class ArithConstantOpLowering
    : public ConvertTritonGPUOpToLLVMPattern<mlir::arith::ConstantOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      mlir::arith::ConstantOp>::ConvertTritonGPUOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(mlir::arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto srcType = dyn_cast<ShapedType>(op.getType());
    if (!srcType || srcType.getNumElements() == 1)
      return failure();

    // arith.constant should only have vector or tensor types.
    if (!isa<VectorType, RankedTensorType>(srcType))
      return failure();

    Type dstType = getTypeConverter()->convertType(srcType);
    if (!dstType)
      return failure();

    auto dstElementsAttr = dyn_cast<DenseElementsAttr>(op.getValue());
    if (!dstElementsAttr)
      return failure();

    auto vecType = cast<VectorType>(dstType);
    dstElementsAttr = dstElementsAttr.resizeSplat(vecType);
    auto newOp =
        rewriter.create<LLVM::ConstantOp>(loc, dstType, dstElementsAttr);
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

class ArithDivFOpLowering
    : public ConvertTritonGPUOpToLLVMPattern<mlir::arith::DivFOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      mlir::arith::DivFOp>::ConvertTritonGPUOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(mlir::arith::DivFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto srcType = dyn_cast<ShapedType>(op.getType());
    Type dstType = getTypeConverter()->convertType(srcType);
    auto vecType = cast<VectorType>(dstType);
    auto attr = rewriter.getFloatAttr(vecType.getElementType(), 1.0);
    auto dstAttr = DenseElementsAttr::get(vecType, attr.getValue());
    auto one = rewriter.create<LLVM::ConstantOp>(loc, dstType, dstAttr);
    auto rcp =
        rewriter.create<LLVM::FDivOp>(loc, dstType, one, adaptor.getRhs());
    auto res =
        rewriter.create<LLVM::FMulOp>(loc, dstType, adaptor.getLhs(), rcp);
    rewriter.replaceOp(op, res);
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
