#include "PatternTritonGPUOpToLLVM.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::gpu::intel;

namespace {
SmallVector<Value> convertMxfp4x2ToFloat(RewriterBase &rewriter, Location loc,
                                         ArrayRef<Value> values,
                                         FloatType floatTy) {
  Value table;
  { // Create a constant vector containing all the possible values
    auto vecTy = VectorType::get({16}, floatTy);
    SmallVector<Attribute, 16> values;
    for (double v : {0., 0.5, 1., 1.5, 2., 3., 4., 6., -0., -0.5, -1., -1.5,
                     -2., -3., -4., -6.})
      values.push_back(rewriter.getFloatAttr(floatTy, v));
    table = rewriter.create<LLVM::ConstantOp>(
        loc, vecTy, DenseElementsAttr::get(vecTy, values));
  }

  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value i8_4 = b.i8_val(4);
  Value i8_15 = b.i8_val(15);
  SmallVector<Value> results;
  for (Value v : values) {
    // The first and last 4 bits are the values induces in the table
    Value idx1 = b.and_(v, i8_15);
    Value idx2 = b.lshr(v, i8_4);
    results.push_back(b.extract_element(table, idx1));
    results.push_back(b.extract_element(table, idx2));
  }
  return results;
}

SmallVector<Value> convertMxfp4x2ToBf16x2(RewriterBase &rewriter, Location loc,
                                          ArrayRef<Value> values) {
  return convertMxfp4x2ToFloat(rewriter, loc, values, bf16_ty);
}

SmallVector<Value> convertMxfp4x2ToFp16x2(RewriterBase &rewriter, Location loc,
                                          ArrayRef<Value> values) {
  return convertMxfp4x2ToFloat(rewriter, loc, values, f16_ty);
}

class Fp4ToFpOpPattern : public ConvertOpToLLVMPattern<Fp4ToFpOp> {
public:
  Fp4ToFpOpPattern(LLVMTypeConverter &typeConverter, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<Fp4ToFpOp>(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(Fp4ToFpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *ctx = op.getContext();
    Type elemType = op.getType().getElementType();
    assert(elemType == f16_ty || elemType == bf16_ty);
    bool toFp16 = elemType == f16_ty;

    SmallVector<Value> xVals =
        unpackLLElements(loc, adaptor.getSrc(), rewriter);
    xVals = toFp16 ? convertMxfp4x2ToFp16x2(rewriter, loc, xVals)
                   : convertMxfp4x2ToBf16x2(rewriter, loc, xVals);

    Value result =
        packLLElements(loc, getTypeConverter(), xVals, rewriter, op.getType());
    rewriter.replaceOp(op, result);
    return success();
  }
};
} // anonymous namespace

void mlir::triton::intel::populateFp4ToFpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<Fp4ToFpOpPattern>(typeConverter, benefit);
}
