#include "Dialect/TritonIntelGPU/Transforms/DecomposeScaledBlocked.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"

#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace {

SmallVector<int, 2> getTransposeOrder(int rank) {
  assert(rank >= 2);
  auto transOrder = llvm::to_vector<2>(llvm::seq<int>(rank - 2));
  transOrder.push_back(rank - 1);
  transOrder.push_back(rank - 2);
  return transOrder;
}

class DecomposeScaledBlocked : public OpRewritePattern<DotScaledOp> {

public:
  DecomposeScaledBlocked(MLIRContext *context, int benefit)
      : OpRewritePattern<DotScaledOp>(context, benefit) {}

  LogicalResult matchAndRewrite(DotScaledOp scaledDotOp,
                                PatternRewriter &rewriter) const override {
    RankedTensorType oldRetType = scaledDotOp.getType();
    if (!oldRetType.getEncoding() ||
        isa<intel::DpasEncodingAttr>(oldRetType.getEncoding()))
      return failure();

    // Types
    auto computeType = getComputeType(scaledDotOp.getAElemType(),
                                      scaledDotOp.getBElemType(), rewriter);
    auto loc = scaledDotOp.getLoc();

    auto cvtDotOperand = [&](TypedValue<RankedTensorType> v,
                             int opIdx) -> TypedValue<RankedTensorType> {
      auto *ctx = rewriter.getContext();
      auto retEnc = scaledDotOp.getType().getEncoding();
      auto vType = v.getType();
      auto encoding = DotOperandEncodingAttr::get(ctx, opIdx, retEnc,
                                                  vType.getElementType());
      RankedTensorType retTy = vType.cloneWithEncoding(encoding);
      return ConvertLayoutOp::create(rewriter, loc, retTy, v);
    };

    auto scaledA = scaleArg(rewriter, scaledDotOp, 0, computeType);
    scaledA = cvtDotOperand(scaledA, 0);
    auto scaledB = scaleArg(rewriter, scaledDotOp, 1, computeType);
    scaledB = cvtDotOperand(scaledB, 1);
    auto newDot =
        DotOp::create(rewriter, scaledDotOp.getLoc(), scaledA, scaledB,
                      scaledDotOp.getC(), InputPrecision::TF32, 0);

    rewriter.replaceOpWithNewOp<ConvertLayoutOp>(scaledDotOp,
                                                 scaledDotOp.getType(), newDot);
    return success();
  }

private:
  FloatType getComputeType(ScaleDotElemType aType, ScaleDotElemType bType,
                           PatternRewriter &rewriter) const {
    if (aType == ScaleDotElemType::FP16 || bType == ScaleDotElemType::FP16)
      return rewriter.getF16Type();
    return rewriter.getBF16Type();
  }

  TypedValue<RankedTensorType> scaleTo16(PatternRewriter &rewriter,
                                         TypedValue<RankedTensorType> scale,
                                         FloatType computeType) const {
    auto loc = scale.getLoc();
    auto scaleTy = scale.getType();
    assert(computeType == rewriter.getBF16Type() ||
           computeType == rewriter.getF16Type());

    if (isa<FloatType>(scaleTy.getElementType())) {
      auto scaleType = scaleTy.clone(computeType);
      return cast<TypedValue<RankedTensorType>>(
          FpToFpOp::create(rewriter, loc, scaleType, scale).getResult());
    }

    // Choose an fp type that can fit the scale value.
    FloatType largeFpType = computeType == rewriter.getF16Type()
                                ? rewriter.getF32Type()
                                : computeType;
    int intWidth = largeFpType.getIntOrFloatBitWidth();
    auto intType = rewriter.getIntegerType(intWidth);

    auto zexted =
        arith::ExtUIOp::create(rewriter, loc, scaleTy.clone(intType), scale);
    // getFpMantissaWidth() returns the number of bits in the mantissa plus the
    // sign bit!
    int shiftValue = largeFpType.getFPMantissaWidth() - 1;
    auto shiftConst =
        arith::ConstantIntOp::create(rewriter, loc, shiftValue, intWidth);
    auto shift =
        SplatOp::create(rewriter, loc, scaleTy.clone(intType), shiftConst);
    auto shlRes = arith::ShLIOp::create(rewriter, loc, zexted, shift);
    Value scaleFP =
        BitcastOp::create(rewriter, loc, scaleTy.clone(largeFpType), shlRes);
    if (largeFpType != computeType) {
      scaleFP = arith::TruncFOp::create(rewriter, loc,
                                        scaleTy.clone(computeType), scaleFP);
    }
    return cast<TypedValue<RankedTensorType>>(scaleFP);
  }

  // Broadcast `scale` along `dim` by `scaleFactor` and produce a tensor with
  // encoding `dstEncoding`.  Uses inferReshapeOpEncoding to place the layout
  // conversion on the small pre-broadcast tensor, avoiding a full-size
  // ConvertLayoutOp on the post-broadcast tensor.  Ported from upstream
  // `lib/Dialect/TritonGPU/Transforms/DecomposeScaledBlocked.cpp`.
  TypedValue<RankedTensorType>
  broadcastScale(PatternRewriter &rewriter, DotScaledOp scaledDotOp,
                 TypedValue<RankedTensorType> scale, int dim,
                 Attribute dstEncoding) const {
    auto loc = scale.getLoc();
    auto scaleTy = scale.getType();
    int32_t scaleFactor = scaledDotOp.deduceScaleFactor();

    // Shapes at each step: insert a size-1 dim after `dim`, broadcast to
    // scaleFactor along that dim, then reshape to fold the broadcast into
    // `dim`.
    auto expandedShape = to_vector(scaleTy.getShape());
    expandedShape.insert(expandedShape.begin() + dim + 1, 1);
    auto broadcastShape = expandedShape;
    broadcastShape[dim + 1] = scaleFactor;
    auto resultShape = to_vector(scaleTy.getShape());
    resultShape[dim] *= scaleFactor;

    // Infer the pre-broadcast encoding that will yield dstEncoding after the
    // reshape + broadcast chain.
    auto interface =
        cast<DialectInferLayoutInterface>(&dstEncoding.getDialect());
    Attribute broadcastEncoding;
    auto result = interface->inferReshapeOpEncoding(
        resultShape, dstEncoding, broadcastShape, broadcastEncoding,
        /*allowReorder=*/false, loc);
    assert(succeeded(result));
    Attribute srcEncoding;
    result = interface->inferReshapeOpEncoding(expandedShape, broadcastEncoding,
                                               scaleTy.getShape(), srcEncoding,
                                               /*allowReorder=*/false, loc);
    assert(succeeded(result));

    // Convert on the small pre-broadcast tensor.
    auto srcType = scaleTy.cloneWithEncoding(srcEncoding);
    scale = ConvertLayoutOp::create(rewriter, loc, srcType, scale);

    // Expand the extra dim (via reshape) — mark as efficient so forward layout
    // propagation doesn't try to sink another convert layout in.
    auto expandType = RankedTensorType::get(
        expandedShape, scaleTy.getElementType(), broadcastEncoding);
    auto expandScale =
        ReshapeOp::create(rewriter, loc, expandType, scale,
                          /*allow_reorder=*/nullptr,
                          /*efficient_layout=*/rewriter.getUnitAttr());
    // Broadcast to microscaling factor.
    auto broadcastType = RankedTensorType::get(
        broadcastShape, scaleTy.getElementType(), broadcastEncoding);
    auto broadcastScale =
        BroadcastOp::create(rewriter, loc, broadcastType, expandScale);
    // Reshape to fold the broadcast into `dim`.
    auto resultType = RankedTensorType::get(
        resultShape, scaleTy.getElementType(), dstEncoding);
    return ReshapeOp::create(rewriter, loc, resultType, broadcastScale);
  }

  TypedValue<RankedTensorType>
  extendAndBroadcastScale(PatternRewriter &rewriter, DotScaledOp scaledDotOp,
                          TypedValue<RankedTensorType> &scale,
                          FloatType computeType, RankedTensorType dstType,
                          int opIdx) const {
    auto loc = scale.getLoc();
    auto v = opIdx == 0 ? scaledDotOp.getA() : scaledDotOp.getB();
    auto rank = v.getType().getRank();
    auto kDim = opIdx == 0 ? rank - 1 : rank - 2;

    // Transpose scale for RHS operand (inplace — caller sees the change).
    if (opIdx == 1) {
      auto order = getTransposeOrder(rank);
      scale = TransOp::create(rewriter, loc, scale, order);
    }

    // 1) Cast scale to compute type (fp16/bf16)
    auto scale16 = scaleTo16(rewriter, scale, computeType);

    // 2) Broadcast scale to the same shape as v — the ConvertLayoutOp lands
    // on the small pre-broadcast tensor inside broadcastScale.
    return broadcastScale(rewriter, scaledDotOp, scale16, kDim,
                          dstType.getEncoding());
  }

  TypedValue<RankedTensorType> maskNan(PatternRewriter &rewriter,
                                       DotScaledOp scaledDotOp,
                                       TypedValue<RankedTensorType> mxfp,
                                       TypedValue<RankedTensorType> scale,
                                       int dim) const {
    // Skip NaN checks if fastMath
    if (scaledDotOp.getFastMath())
      return mxfp;

    // Implement tl.where(scale == 0xFF, float("nan"), mxfp)
    auto loc = scale.getLoc();

    // Scale is NaN
    auto scaleTy = scale.getType();
    TypedValue<RankedTensorType> scaleIsNan;
    if (isa<FloatType>(scaleTy.getElementType())) {
      auto computeType = cast<FloatType>(mxfp.getType().getElementType());
      auto scaleFp = scaleTo16(rewriter, scale, computeType);
      scaleIsNan = cast<TypedValue<RankedTensorType>>(
          arith::CmpFOp::create(rewriter, loc, arith::CmpFPredicate::UNO,
                                scaleFp, scaleFp)
              .getResult());
    } else {
      auto constFF = arith::ConstantOp::create(
          rewriter, loc, scaleTy,
          DenseElementsAttr::get(
              scaleTy, APInt(scaleTy.getElementTypeBitWidth(), 0xff)));
      scaleIsNan = cast<TypedValue<RankedTensorType>>(
          arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq, scale,
                                constFF)
              .getResult());
    }
    // Broadcast the i1 NaN mask directly into mxfp's encoding — the layout
    // conversion happens on the small pre-broadcast tensor.
    auto cond = broadcastScale(rewriter, scaledDotOp, scaleIsNan, dim,
                               mxfp.getType().getEncoding());

    // Create NaN
    auto mxfpTy = mxfp.getType();
    auto nan = APFloat::getNaN(
        cast<FloatType>(mxfpTy.getElementType()).getFloatSemantics());
    auto constNan = arith::ConstantOp::create(
        rewriter, loc, mxfpTy, DenseElementsAttr::get(mxfpTy, nan));

    auto result = arith::SelectOp::create(rewriter, loc, cond, constNan, mxfp);
    return cast<TypedValue<RankedTensorType>>(result.getResult());
  }

  // Apply an E8M0 scale to a bf16 operand as an exponent-add on the bf16 raw
  // bits.  The E8M0 byte value equals the bf16 biased exponent that represents
  // 2^(byte-127); adding (byte << 7) - 0x3F80 to the operand's i16 bit pattern
  // is bit-exact `operand * 2^(byte-127)` when neither the input nor the
  // result exponent overflows the 8-bit field.  Skips the bf16 -> f32 -> bf16
  // widening that `arith::MulFOp` would incur under
  // `arith_emulate_unsupported_floats` (BMG has no native bf16 arithmetic).
  TypedValue<RankedTensorType> applyE8M0ScaleViaExponentAdd(
      PatternRewriter &rewriter, DotScaledOp scaledDotOp,
      TypedValue<RankedTensorType> v, TypedValue<RankedTensorType> &scale,
      int opIdx) const {
    auto loc = v.getLoc();
    auto vType = v.getType();
    auto rank = vType.getRank();
    auto kDim = opIdx == 0 ? rank - 1 : rank - 2;
    auto i16Type = rewriter.getIntegerType(16);
    auto i16OperandType = vType.clone(i16Type);

    // Transpose scale for RHS operand (mirrors `extendAndBroadcastScale`).
    if (opIdx == 1) {
      auto order = getTransposeOrder(rank);
      scale = TransOp::create(rewriter, loc, scale, order);
    }

    // 1) Widen scale i8 -> i16 and shift into the bf16 exponent slot.
    auto scaleTy = scale.getType();
    auto zexted =
        arith::ExtUIOp::create(rewriter, loc, scaleTy.clone(i16Type), scale);
    auto shiftConst = arith::ConstantIntOp::create(rewriter, loc, 7, 16);
    auto shiftSplat =
        SplatOp::create(rewriter, loc, scaleTy.clone(i16Type), shiftConst);
    auto shifted = cast<TypedValue<RankedTensorType>>(
        arith::ShLIOp::create(rewriter, loc, zexted, shiftSplat).getResult());

    // 2) Broadcast shifted scale to operand shape directly in i16 operand
    // layout — the layout convert lands on the small pre-broadcast tensor.
    auto broadcastShifted = broadcastScale(rewriter, scaledDotOp, shifted, kDim,
                                           i16OperandType.getEncoding());

    // 3) Bitcast operand bf16 -> i16.
    auto vI16 = BitcastOp::create(rewriter, loc, i16OperandType, v);

    // 4) result_i16 = operand_i16 + (scale_byte << 7) - 0x3F80
    //    (0x3F80 is bf16 biased exp 127 << 7, i.e. bf16 1.0)
    auto sum = arith::AddIOp::create(rewriter, loc, vI16, broadcastShifted);
    auto biasConst = arith::ConstantIntOp::create(rewriter, loc, 0x3F80, 16);
    auto biasSplat = SplatOp::create(rewriter, loc, i16OperandType, biasConst);
    auto biased = arith::SubIOp::create(rewriter, loc, sum, biasSplat);

    // 5) Bitcast result i16 -> bf16.
    auto rescaled = cast<TypedValue<RankedTensorType>>(
        BitcastOp::create(rewriter, loc, vType, biased).getResult());

    // 6) NaN mask (byte == 0xFF).
    return maskNan(rewriter, scaledDotOp, rescaled, scale, kDim);
  }

  TypedValue<RankedTensorType> scaleArg(PatternRewriter &rewriter,
                                        DotScaledOp scaledDotOp, int opIdx,
                                        FloatType computeType) const {
    auto v = opIdx == 0 ? scaledDotOp.getA() : scaledDotOp.getB();
    auto res = scaledDotOp.getD();
    auto scale = opIdx == 0 ? scaledDotOp.getAScale() : scaledDotOp.getBScale();
    auto isFp4 =
        ScaleDotElemType::E2M1 ==
        (opIdx == 0 ? scaledDotOp.getAElemType() : scaledDotOp.getBElemType());

    auto loc = v.getLoc();
    auto rank = v.getType().getRank();
    auto kDim = opIdx == 0 ? rank - 1 : rank - 2;

    // 0) Upcast value to computeType (fp16/bf16)
    if (isFp4) {
      auto resShape = res.getType().getShape();
      auto vShape = v.getType().getShape();
      auto packDim = kDim;
      if ((opIdx == 0 && resShape[rank - 2] != vShape[rank - 2]) ||
          (opIdx == 1 && resShape[rank - 1] != vShape[rank - 1])) {
        packDim = (packDim + 1) % 2;
      }
      v = Fp4ToFpOp::create(rewriter, loc, v, computeType, packDim);
    } else {
      auto vType16 = v.getType().clone(computeType);
      v = cast<TypedValue<RankedTensorType>>(
          FpToFpOp::create(rewriter, loc, vType16, v).getResult());
    }
    if (!scale)
      return v;

    // Fast path: E8M0 (i8) scale on a bf16 operand.  Apply the scale as an
    // integer exponent-add on the bf16 raw bits, avoiding the bf16 mulf that
    // `arith_emulate_unsupported_floats` widens into an f32 intermediate.
    auto scaleElemType = scale.getType().getElementType();
    if (!isa<FloatType>(scaleElemType) &&
        computeType == rewriter.getBF16Type()) {
      return applyE8M0ScaleViaExponentAdd(rewriter, scaledDotOp, v, scale,
                                          opIdx);
    }

    // 1) Cast scale to fp16/bf16, broadcast it and convert its layout
    auto reshapeScale = extendAndBroadcastScale(
        rewriter, scaledDotOp, scale, computeType, v.getType(), opIdx);

    // 2) Multiply
    auto mxfp = cast<TypedValue<RankedTensorType>>(
        arith::MulFOp::create(rewriter, loc, v, reshapeScale).getResult());

    // 3) If the scale is NaN, return NaN, else return the scaled value.
    return maskNan(rewriter, scaledDotOp, mxfp, scale, kDim);
  }
};

} // namespace

namespace mlir::triton::gpu::intel {

void populateDecomposeScaledBlockedPatterns(RewritePatternSet &patterns,
                                            int benefit) {
  patterns.add<DecomposeScaledBlocked>(patterns.getContext(), benefit);
}

} // namespace mlir::triton::gpu::intel
