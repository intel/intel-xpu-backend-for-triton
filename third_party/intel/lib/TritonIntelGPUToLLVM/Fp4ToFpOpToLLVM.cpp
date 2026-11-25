#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::LLVM::intel;
using namespace mlir::triton;
using namespace mlir::triton::intel;
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
    auto elemTy = dyn_cast<FloatType>(op.getType().getElementType());
    assert(elemTy == f16_ty || elemTy == bf16_ty);

    Location loc = op.getLoc();
    SmallVector<Value> srcVals =
        unpackLLElements(loc, adaptor.getSrc(), rewriter);

    if (hasModuleAttr(
            op.getOperation(),
            TritonIntelGPUDialect::getSupportF4ConversionAttrName())) {
      auto convertFn = [&](TritonLLVMIRRewriter &rw, Value v) {
        int64_t numEls = 2;
        if (auto vecTy = dyn_cast<VectorType>(v.getType())) {
          numEls *= vecTy.getNumElements();
        }
        auto baseName = elemTy == f16_ty
                            ? "__builtin_spirv_ConvertE2M1ToFP16EXT"
                            : "__builtin_spirv_ConvertE2M1ToBF16EXT";
        auto i4Ty = rw.builder->getI4Type();
        v = rw.bitcast(v, VectorType::get({numEls}, i4Ty));
        return convertWithFunctionCall(rw, v, baseName, i4Ty, elemTy);
      };

      auto res = vectorize(convertFn, loc, rewriter, srcVals, 8);
      if (!res.empty()) {
        rewriter.replaceOp(op, packLLElements(loc, getTypeConverter(), res,
                                              rewriter, op.getType()));
        return success();
      }
    }

    rewriter.replaceOp(op,
                       packLLElements(loc, getTypeConverter(),
                                      decode(loc, rewriter, srcVals, elemTy),
                                      rewriter, op.getType()));
    return success();
  }

private:
  // Decodes the two e2m1 values of each byte with integer operations. A
  // nibble is shifted to bits [31:28], and an arithmetic shift right by 6
  // moves its exponent and mantissa bits to f32 bits [24:22] and keeps the
  // sign at bit 31; the mask clears the sign copies in between. That f32 value
  // is 2^-126 times the e2m1 value (0.5 becomes a denormal), so multiplying it
  // by 2^126 gives the exact value. Every e2m1 value is exact in fp16 and
  // bf16, so the bf16 result is the upper half of the f32 value.
  //
  // This avoids a lookup into a constant table of the 16 values: its per-lane
  // dynamic vector index is lowered by IGC to indirect register moves that
  // occupy the address register.
  static SmallVector<Value> decode(Location loc,
                                   ConversionPatternRewriter &rewriter,
                                   ArrayRef<Value> srcVals, FloatType elemTy) {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value mask = b.i32_val(0x81C00000);
    Value shiftRight = b.i32_val(6);
    Value upperHalf = b.i32_val(16);
    Value scale = b.f32_val(0x1p126f);
    // The low nibble holds the first element, the high nibble the second.
    Value shiftLeft[] = {b.i32_val(28), b.i32_val(24)};
    SmallVector<Value> results;
    for (Value packed : srcVals) {
      assert(packed.getType().isInteger(8) && "expected packed e2m1 bytes");
      Value x = b.zext(i32_ty, packed);
      for (Value shift : shiftLeft) {
        Value bits = b.and_(b.ashr(b.shl(x, shift), shiftRight), mask);
        Value val = b.fmul(b.bitcast(bits, f32_ty), scale);
        if (elemTy.isBF16())
          results.push_back(b.bitcast(
              b.trunc(i16_ty, b.lshr(b.bitcast(val, i32_ty), upperHalf)),
              bf16_ty));
        else
          results.push_back(b.fptrunc(f16_ty, val));
      }
    }
    return results;
  }
};
} // anonymous namespace

void mlir::triton::intel::populateFp4ToFpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<Fp4ToFpOpPattern>(typeConverter, benefit);
}
