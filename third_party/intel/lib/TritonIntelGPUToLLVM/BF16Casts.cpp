#include "PatternTritonGPUOpToLLVM.h"

#include "Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "Utils/LLVMIntr.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/TypeSwitch.h"

#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Utils.h"

using namespace mlir;
using namespace triton::gpu::intel;

namespace {
static bool isBF16OrTensorOf(Type type) {
  return TypeSwitch<Type, bool>(type)
      .Case(
          [](RankedTensorType type) { return type.getElementType().isBF16(); })
      .Default([](Type type) { return type.isBF16(); });
}

static bool isF32OrTensorOf(Type type) {
  return TypeSwitch<Type, bool>(type)
      .Case([](RankedTensorType type) { return type.getElementType().isF32(); })
      .Default([](Type type) { return type.isF32(); });
}

static Type getTypeWithSameShape(Type type, Type elementType) {
  return TypeSwitch<Type, Type>(type)
      .Case([elementType](VectorType type) {
        return VectorType::get(type.getShape(), elementType,
                               type.getScalableDims());
      })
      .Default(elementType);
}

struct ExtBF16 : ConvertOpToLLVMPattern<arith::ExtFOp> {
  using ConvertOpToLLVMPattern<arith::ExtFOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(arith::ExtFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (!isBF16OrTensorOf(op.getIn().getType()) ||
        !isF32OrTensorOf(op.getOut().getType()))
      return failure();

    rewriter.replaceOp(
        op, intel::convertBf16ToFp32(op.getLoc(), rewriter, adaptor.getIn()));
    return success();
  }
};

struct TruncBF16 : ConvertOpToLLVMPattern<arith::TruncFOp> {
  using ConvertOpToLLVMPattern<arith::TruncFOp>::ConvertOpToLLVMPattern;

  constexpr static arith::RoundingMode validRoundingMode =
      arith::RoundingMode::to_nearest_even;

  LogicalResult
  matchAndRewrite(arith::TruncFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    std::optional<arith::RoundingMode> roundingMode = op.getRoundingmode();
    if ((roundingMode && *roundingMode != validRoundingMode) ||
        !isF32OrTensorOf(op.getIn().getType()) ||
        !isBF16OrTensorOf(op.getOut().getType()))
      return failure();

    rewriter.replaceOp(op, intel::convertFp32ToBf16(op.getLoc(), rewriter,
                                                    adaptor.getIn(),
                                                    RoundingMode::RTNE));
    return success();
  }
};
} // namespace

namespace mlir::triton::intel {
Value convertBf16ToFp32(Location loc, ConversionPatternRewriter &rewriter,
                        Value v) {
  TritonLLVMIRRewriter b(loc, rewriter);
  auto as_int16 = b.bitcast(v, i16_ty).getResult();
  auto result = convertWithFunctionCall(
      b, as_int16, "__spirv_ConvertBF16ToFINTEL", i16_ty, f32_ty,
      TritonIntelGPUDialect::getSupportBF16ConversionAttrName());
  if (result) {
    return result;
  }

  auto as_int32 = b.zext(i32_ty, as_int16);
  auto shifted = b.shl(i32_ty, as_int32, b.i32_val(16));
  return (b.bitcast(shifted, f32_ty));
}

Value convertFp32ToBf16(Location loc, ConversionPatternRewriter &rewriter,
                        Value v, RoundingMode rounding) {
  TritonLLVMIRRewriter b(loc, rewriter);
  auto result = convertWithFunctionCall(
      b, v, "__spirv_ConvertFToBF16INTEL", f32_ty, i16_ty,
      TritonIntelGPUDialect::getSupportBF16ConversionAttrName());
  if (result) {
    return b.bitcast(result, bf16_ty);
  }

  assert(!isa<VectorType>(v.getType()) && "Not yet supported");

  auto as_uint32 = b.bitcast(v, i32_ty);
  auto check_exponent =
      b.and_(i32_ty, b.xor_(i32_ty, as_uint32, b.i32_val(0xffffffff)),
             b.i32_val(0x7f800000));
  auto exponent_not_all1s = b.icmp_ne(check_exponent, b.i32_val(0));
  auto exponent_all1s = b.icmp_eq(check_exponent, b.i32_val(0));
  Value rounded = as_uint32;
  if (rounding == RoundingMode::RTNE) {
    rounded = b.add(
        i32_ty, b.i32_val(0x7fff),
        b.and_(i32_ty, b.lshr(i32_ty, as_uint32, b.i32_val(16)), b.i32_val(1)));
    rounded = b.add(i32_ty, rounded, as_uint32);
    rounded = b.select(exponent_not_all1s, rounded, as_uint32);
  }

  auto preserve_nan = b.and_(
      i1_ty, exponent_all1s,
      b.icmp_ne(b.and_(i32_ty, as_uint32, b.i32_val(0xffff)), b.i32_val(0)));
  auto nan = b.or_(i32_ty, as_uint32, b.i32_val(0x10000));
  Value res = b.select(preserve_nan, nan, rounded);

  auto shifted = b.lshr(i32_ty, res, b.i32_val(16));
  auto truncated = b.trunc(i16_ty, shifted);
  return b.bitcast(truncated, bf16_ty);
}

void populateBF16CastsLLVMPatterns(LLVMTypeConverter &typeConverter,
                                   RewritePatternSet &patterns,
                                   PatternBenefit benefit) {
  patterns.add<ExtBF16, TruncBF16>(typeConverter, benefit);
}
} // namespace mlir::triton::intel
