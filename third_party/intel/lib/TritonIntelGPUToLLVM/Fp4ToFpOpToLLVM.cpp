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
SmallVector<Value> convertMxfp4x2ToBf16x2(RewriterBase &rewriter, Location loc,
                                          ArrayRef<Value> values) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  SmallVector<Value> results;
  for (auto v : values) {
    auto em0 = b.and_(v, b.i8_val(0x7));
    auto em1 = b.and_(v, b.i8_val(0x70));
    Value v0 =
        b.or_(b.shl(b.zext(i16_ty, em0), b.i16_val(6)),
              b.shl(b.zext(i16_ty, b.and_(v, b.i8_val(0x8))), b.i16_val(12)));
    Value v1 =
        b.or_(b.shl(b.zext(i16_ty, em1), b.i16_val(2)),
              b.shl(b.zext(i16_ty, b.and_(v, b.i8_val(0x80))), b.i16_val(8)));
    // Three cases:
    // 1) x is normal and non-zero: Correct bias
    v0 = b.select(b.icmp_ne(b.and_(em0, b.i8_val(0x6)), b.i8_val(0)),
                  b.add(v0, b.i16_val((127 - 1) << 7)), v0);
    v1 = b.select(b.icmp_ne(b.and_(em1, b.i8_val(0x60)), b.i8_val(0)),
                  b.add(v1, b.i16_val((127 - 1) << 7)), v1);
    // 2) x is subnormal (x == 0bs001 where s is the sign): Map to +-0.5 in
    // bf16
    v0 = b.bitcast(
        b.select(b.icmp_eq(em0, b.i8_val(0x1)),
                 b.or_(b.i16_val(16128), b.and_(v0, b.i16_val(0x8000))), v0),
        bf16_ty);
    v1 = b.bitcast(
        b.select(b.icmp_eq(em1, b.i8_val(0x10)),
                 b.or_(b.i16_val(16128), b.and_(v1, b.i16_val(0x8000))), v1),
        bf16_ty);
    // 3) x is zero, nothing to do
    results.push_back(v0);
    results.push_back(v1);
  }
  return results;
}

SmallVector<Value> convertMxfp4x2ToFp16x2(RewriterBase &rewriter, Location loc,
                                          ArrayRef<Value> values) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  SmallVector<Value> results;
  for (auto v : values) {
    auto em0 = b.and_(v, b.i8_val(0x7));
    auto em1 = b.and_(v, b.i8_val(0x70));
    // FP16 bits: sign = 1, exponent = 5, mantissa = 10
    Value v0 =
        b.or_(b.shl(b.zext(i16_ty, em0), b.i16_val(10 - 1)),
              b.shl(b.zext(i16_ty, b.and_(v, b.i8_val(0x8))), b.i16_val(12)));
    Value v1 =
        b.or_(b.shl(b.zext(i16_ty, em1), b.i16_val(10 - 1 - 4)),
              b.shl(b.zext(i16_ty, b.and_(v, b.i8_val(0x80))), b.i16_val(8)));

    // Three cases:
    // 1) x is normal and non-zero: Correct bias
    v0 = b.select(b.icmp_ne(b.and_(em0, b.i8_val(0x6)), b.i8_val(0)),
                  b.add(v0, b.i16_val((15 - 1) << 10)), v0);
    v1 = b.select(b.icmp_ne(b.and_(em1, b.i8_val(0x60)), b.i8_val(0)),
                  b.add(v1, b.i16_val((15 - 1) << 10)), v1);

    // 2) x is subnormal (x == 0bs001 where s is the sign): Map to fp16 +-0.5
    v0 = b.bitcast(
        b.select(b.icmp_eq(em0, b.i8_val(0x1)),
                 b.or_(b.i16_val(0x3800), b.and_(v0, b.i16_val(0x8000))), v0),
        f16_ty);
    v1 = b.bitcast(
        b.select(b.icmp_eq(em1, b.i8_val(0x10)),
                 b.or_(b.i16_val(0x3800), b.and_(v1, b.i16_val(0x8000))), v1),
        f16_ty);
    // 3) x is zero, nothing to do
    results.push_back(v0);
    results.push_back(v1);
  }
  return results;
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
