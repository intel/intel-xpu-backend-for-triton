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
class Fp4ToFpOpPattern : public ConvertOpToLLVMPattern<Fp4ToFpOp> {
public:
  Fp4ToFpOpPattern(LLVMTypeConverter &typeConverter, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<Fp4ToFpOp>(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(Fp4ToFpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type elemType = op.getType().getElementType();
    assert(elemType == f16_ty || elemType == bf16_ty);

    SmallVector<Value> results;
    {
      SmallVector<Value> xVals =
          unpackLLElements(loc, adaptor.getSrc(), rewriter);
      convertMxfp4x2ToFloat(rewriter, loc, xVals, results,
                            elemType == f16_ty ? f16_ty : bf16_ty);
    }
    rewriter.replaceOp(op, packLLElements(loc, getTypeConverter(), results,
                                          rewriter, op.getType()));
    return success();
  }

private:
  static void convertMxfp4x2ToFloat(RewriterBase &rewriter, Location loc,
                                    SmallVector<Value> &values,
                                    SmallVector<Value> &results,
                                    FloatType floatTy) {
    assert(results.empty() && !values.empty());

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

    TritonLLVMOpBuilder b(loc, rewriter);
    Value i8_4 = b.i8_val(4);
    Value i8_15 = b.i8_val(15);
    results.reserve(values.size() * 2);
    for (Value v : values) {
      // The first and last 4 bits are the values indices in the table
      Value idx1 = b.and_(v, i8_15);
      Value idx2 = b.lshr(v, i8_4);
      results.push_back(b.extract_element(table, idx1));
      results.push_back(b.extract_element(table, idx2));
    }
  }
};
} // anonymous namespace

void mlir::triton::intel::populateFp4ToFpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<Fp4ToFpOpPattern>(typeConverter, benefit);
}
