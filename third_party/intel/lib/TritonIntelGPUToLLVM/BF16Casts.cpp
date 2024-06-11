#include "PatternTritonGPUOpToLLVM.h"

#include "Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "Utils/Mangling.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

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

  LogicalResult match(arith::ExtFOp op) const final {
    return success(isBF16OrTensorOf(op.getIn().getType()) &&
                   isF32OrTensorOf(op.getOut().getType()));
  }

  void rewrite(arith::ExtFOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(
        op, intel::convertBf16ToFp32(op.getLoc(), rewriter, adaptor.getIn()));
  }
};

struct TruncBF16 : ConvertOpToLLVMPattern<arith::TruncFOp> {
  using ConvertOpToLLVMPattern<arith::TruncFOp>::ConvertOpToLLVMPattern;

  constexpr static arith::RoundingMode validRoundingMode =
      arith::RoundingMode::to_nearest_even;

  LogicalResult match(arith::TruncFOp op) const final {
    std::optional<arith::RoundingMode> roundingMode = op.getRoundingmode();
    return success((!roundingMode || *roundingMode == validRoundingMode) &&
                   isF32OrTensorOf(op.getIn().getType()) &&
                   isBF16OrTensorOf(op.getOut().getType()));
  }

  void rewrite(arith::TruncFOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(op, intel::convertFp32ToBf16(op.getLoc(), rewriter,
                                                    adaptor.getIn(),
                                                    RoundingMode::RTNE));
  }
};
} // namespace

namespace mlir::triton::intel {
Value convertBf16ToFp32(Location loc, ConversionPatternRewriter &rewriter,
                        Value v) {
  auto moduleOp = v.getDefiningOp()->getParentWithTrait<OpTrait::SymbolTable>();
  constexpr StringLiteral baseName = "__spirv_ConvertBF16ToFINTEL";
  Type inTy = getTypeWithSameShape(v.getType(), i16_ty);
  Type outTy = getTypeWithSameShape(inTy, f32_ty);
  std::string name = mlir::triton::gpu::intel::mangle(baseName, inTy);
  auto ext_func =
      triton::gpu::intel::lookupOrCreateSPIRVFn(moduleOp, name, inTy, outTy);
  auto call = triton::gpu::intel::createSPIRVBuiltinCall(
      loc, rewriter, ext_func, bitcast(v, inTy).getResult());
  return call.getResult();
}

Value convertFp32ToBf16(Location loc, ConversionPatternRewriter &rewriter,
                        Value v, RoundingMode rounding) {
  if (rounding == RoundingMode::RTNE) {
    auto moduleOp =
        v.getDefiningOp()->getParentWithTrait<OpTrait::SymbolTable>();
    // Intel SPIR-V extension only supports round-to-nearest-even
    constexpr StringLiteral baseName = "__spirv_ConvertFToBF16INTEL";
    Type inTy = v.getType();
    Type funcOutTy = getTypeWithSameShape(inTy, i16_ty);
    Type outTy = getTypeWithSameShape(inTy, bf16_ty);
    std::string name = mlir::triton::gpu::intel::mangle(baseName, inTy);
    auto trunc_func = triton::gpu::intel::lookupOrCreateSPIRVFn(
        moduleOp, name, inTy, funcOutTy);
    auto call = triton::gpu::intel::createSPIRVBuiltinCall(loc, rewriter,
                                                           trunc_func, v);
    return bitcast(call.getResult(), outTy);
  }

  assert(!isa<VectorType>(v.getType()) && "Not yet supported");

  auto as_uint32 = bitcast(v, i32_ty);
  auto check_exponent =
      and_(i32_ty, xor_(i32_ty, as_uint32, i32_val(0xffffffff)),
           i32_val(0x7f800000));
  auto exponent_not_all1s = icmp_ne(check_exponent, i32_val(0));
  auto exponent_all1s = icmp_eq(check_exponent, i32_val(0));
  Value rounded = as_uint32;

  auto preserve_nan =
      and_(i1_ty, exponent_all1s,
           icmp_ne(and_(i32_ty, as_uint32, i32_val(0xffff)), i32_val(0)));
  auto nan = or_(i32_ty, as_uint32, i32_val(0x10000));
  Value res = select(preserve_nan, nan, rounded);

  auto shifted = lshr(i32_ty, res, i32_val(16));
  auto truncated = trunc(i16_ty, shifted);
  return bitcast(truncated, bf16_ty);
}

void populateBF16CastsLLVMPatterns(LLVMTypeConverter &typeConverter,
                                   RewritePatternSet &patterns,
                                   PatternBenefit benefit) {
  patterns.add<ExtBF16, TruncBF16>(typeConverter, benefit);
}
} // namespace mlir::triton::intel
