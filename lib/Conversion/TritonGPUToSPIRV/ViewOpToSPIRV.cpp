#include "ViewOpToSPIRV.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::spirv::getSharedMemoryObjectFromStruct;
using ::mlir::triton::gpu::getTotalElemsPerThread;

struct SplatOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<triton::SplatOp> {
  using ConvertTritonGPUOpToSPIRVPattern<
      triton::SplatOp>::ConvertTritonGPUOpToSPIRVPattern;

  // Convert SplatOp or arith::ConstantOp with SplatElementsAttr to a
  // spirv::StructType value.
  //
  // @elemType: the element type in operand.
  // @resType: the return type of the Splat-like op.
  // @constVal: a spirv::ConstantOp or other scalar value.
  static Value convertSplatLikeOp(Type elemType, Type resType, Value constVal,
                                  TritonGPUToSPIRVTypeConverter *typeConverter,
                                  ConversionPatternRewriter &rewriter,
                                  Location loc) {
    auto tensorTy = resType.cast<RankedTensorType>();
    auto srcType = typeConverter->convertType(elemType);
    auto spirvSrc = bitcast(constVal, srcType);
    size_t elemsPerThread = getTotalElemsPerThread(tensorTy);
    llvm::SmallVector<Value> elems(elemsPerThread, spirvSrc);
    return typeConverter->packLLElements(loc, elems, rewriter, resType);
  }

  LogicalResult matchAndRewrite(triton::SplatOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    auto loc = op->getLoc();
    auto src = adaptor.getSrc();
    auto spirvStruct = convertSplatLikeOp(src.getType(), op.getType(), src,
                                          getTypeConverter(), rewriter, loc);
    rewriter.replaceOp(op, {spirvStruct});
    return success();
  }
};

// This pattern helps to convert arith::ConstantOp(with SplatElementsAttr),
// the logic is the same as triton::SplatOp, so the underlying implementation
// is reused.
struct ArithConstantSplatOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<arith::ConstantOp> {
  using ConvertTritonGPUOpToSPIRVPattern<
      arith::ConstantOp>::ConvertTritonGPUOpToSPIRVPattern;

  Value _convertFp32ToBf16(Location loc, ConversionPatternRewriter &rewriter,
                           const Value &v) const {
    // Copied from `ElementwiseOpToSPIRV.cpp`. This code is used for special
    // treatment for BF16.

    // Convert the FP32 value by RNE(Rounding to Nearest Even).
    // Algorithm is as follows:
    //   STEP1: U32_VAL = BITCAST(F32_VAL)
    //   STEP2: U32_VAL_TMP = U32_VAL >> 16
    //   STEP3: U32_VAL_TMP = U32_VAL_TMP & 1
    //   STEP4: ROUNDING_BIAS = U32_VAL_TMP + UINT32(0x7FFF)
    //   STEP5: U32_VAL_TMP = U32_VAL + ROUNDING_BIAS
    //   STEP6: BF16_VAL = static_cast<UINT16>(U32_VAL_TMP >> 16)
    Value val = v;
    auto mask = fcmp_oeq(val, val);
    // STEP1
    auto fp32_i32_value = bitcast(v, i32_ty);
    // STEP2
    val = lshr(fp32_i32_value, i32_val(16));
    // val = rewriter.create<arith::TruncIOp>(loc, i16_ty, val);
    val = itrunc(i16_ty, val);
    // STEP3
    val = and_(val, int_val(16, 1));
    // STEP4
    auto rounding_bias = int_val(16, 0x7FF);
    val = add(val, rounding_bias);
    val = zext(i32_ty, val);
    // Step 5
    val = add(val, fp32_i32_value);
    // Step6
    val = lshr(val, int_val(32, 16));
    // val = rewriter.create<arith::TruncIOp>(loc, i16_ty, val);
    val = itrunc(i16_ty, val);
    val = bitcast(val, i16_ty);
    // If the value is NaN, return BF16 NaN.
    val = select(mask, val, int_val(16, 0xFFFF));
    return val;
  }

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto value = op.getValue();
    if (!value.dyn_cast<SplatElementsAttr>())
      return failure();

    auto loc = op->getLoc();

    auto values = op.getValue().dyn_cast<SplatElementsAttr>();
    auto elemType = values.getElementType();

    Attribute val;
    if (elemType.isBF16()) {
      // spirv::ConstantOp does not support bf16, Thus it needs special
      // treatment first.
      auto v = values.getValues<FloatAttr>()[0];
      auto lit_v = v.getValue();
      val = rewriter.getF32FloatAttr(lit_v.convertToFloat());
    } else if (type::isFloat(elemType)) {
      val = values.getValues<FloatAttr>()[0];
    } else if (type::isInt(elemType)) {
      val = values.getValues<IntegerAttr>()[0];
    } else {
      llvm::errs()
          << "ArithConstantSplatOpSPIRVConversion get unsupported type: "
          << value.getType() << "\n";
      return failure();
    }

    Value constOp;
    if (elemType.isBF16()) {
      // spirv::ConstantOp does not support bf16.
      constOp = rewriter.create<spirv::ConstantOp>(loc, f32_ty, val);
    } else {
      constOp = rewriter.create<spirv::ConstantOp>(loc, elemType, val);
    }

    if (elemType.isBF16()) {
      constOp = _convertFp32ToBf16(loc, rewriter, constOp);
    }
    auto llStruct = SplatOpSPIRVConversion::convertSplatLikeOp(
        elemType, op.getType(), constOp, getTypeConverter(), rewriter, loc);
    rewriter.replaceOp(op, llStruct);

    return success();
  }
};

#if 0
struct CatOpConversion : public ConvertTritonGPUOpToLLVMPattern<CatOp> {
  using OpAdaptor = typename CatOp::Adaptor;

  explicit CatOpConversion(TritonGPUToLLVMTypeConverter &typeConverter,
                           PatternBenefit benefit = 1)
          : ConvertTritonGPUOpToLLVMPattern<CatOp>(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(CatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto resultTy = op.getType().template cast<RankedTensorType>();
    unsigned elems = getTotalElemsPerThread(resultTy);
    Type elemTy =
            this->getTypeConverter()->convertType(resultTy.getElementType());
    SmallVector<Type> types(elems, elemTy);
    // unpack input values
    auto lhsVals = getTypeConverter()->unpackLLElements(
            loc, adaptor.getLhs(), rewriter, op.getOperand(0).getType());
    auto rhsVals = getTypeConverter()->unpackLLElements(
            loc, adaptor.getRhs(), rewriter, op.getOperand(1).getType());
    // concatenate (and potentially reorder) values
    SmallVector<Value> retVals;
    for (Value v : lhsVals)
      retVals.push_back(v);
    for (Value v : rhsVals)
      retVals.push_back(v);
    // pack and replace
    Value ret =
            getTypeConverter()->packLLElements(loc, retVals, rewriter, resultTy);
    rewriter.replaceOp(op, ret);
    return success();
  }
};
#endif

struct ViewOpSPIRVConversion : public ConvertTritonGPUOpToSPIRVPattern<ViewOp> {
  using OpAdaptor = typename ViewOp::Adaptor;
  using ConvertTritonGPUOpToSPIRVPattern<
      ViewOp>::ConvertTritonGPUOpToSPIRVPattern;

  LogicalResult
  matchAndRewrite(ViewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto resultTy = op.getType().template cast<RankedTensorType>();
    auto vals = this->getTypeConverter()->unpackLLElements(
        loc, adaptor.getSrc(), rewriter, op.getOperand().getType());
    Value ret =
        this->getTypeConverter()->packLLElements(loc, vals, rewriter, resultTy);
    rewriter.replaceOp(op, ret);
    return success();
  }
};

struct ExpandDimsOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<ExpandDimsOp> {
  using OpAdaptor = typename ExpandDimsOp::Adaptor;
  using ConvertTritonGPUOpToSPIRVPattern<
      ExpandDimsOp>::ConvertTritonGPUOpToSPIRVPattern;

  LogicalResult
  matchAndRewrite(ExpandDimsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto srcVals = this->getTypeConverter()->unpackLLElements(
        loc, adaptor.getSrc(), rewriter, op.getOperand().getType());

    auto srcTy = op.getSrc().getType().cast<RankedTensorType>();
    auto resultTy = op.getType().template cast<RankedTensorType>();

    assert(srcTy.getEncoding().isa<SliceEncodingAttr>() &&
           "ExpandDimsOp only support SliceEncodingAttr");
    auto srcLayout = srcTy.getEncoding().dyn_cast<SliceEncodingAttr>();
    auto resultLayout = resultTy.getEncoding();

    auto srcOffsets = emitOffsetForLayout(srcLayout, srcTy);
    auto resultOffsets = emitOffsetForLayout(resultLayout, resultTy);
    DenseMap<SmallVector<unsigned>, Value, SmallVectorKeyInfo> srcValues;
    for (size_t i = 0; i < srcOffsets.size(); i++) {
      srcValues[srcOffsets[i]] = srcVals[i];
    }

    SmallVector<Value> resultVals;
    for (size_t i = 0; i < resultOffsets.size(); i++) {
      auto offset = resultOffsets[i];
      offset.erase(offset.begin() + srcLayout.getDim());
      resultVals.push_back(srcValues.lookup(offset));
    }
    Value ret = this->getTypeConverter()->packLLElements(loc, resultVals,
                                                         rewriter, resultTy);
    rewriter.replaceOp(op, ret);
    return success();
  }
};

#if 0
struct TransOpConversion
        : public ConvertTritonGPUOpToLLVMPattern<triton::TransOp> {
  using ConvertTritonGPUOpToLLVMPattern<
          triton::TransOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::TransOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto srcSmemObj =
            getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(), rewriter);
    SmallVector<Value> dstStrides = {srcSmemObj.strides[1],
                                     srcSmemObj.strides[0]};
    SmallVector<Value> dstOffsets = {srcSmemObj.offsets[1],
                                     srcSmemObj.offsets[0]};
    auto dstSmemObj =
            SharedMemoryObject(srcSmemObj.base, dstStrides, dstOffsets);
    auto retVal = getStructFromSharedMemoryObject(loc, dstSmemObj, rewriter);
    rewriter.replaceOp(op, retVal);
    return success();
  }
};
#endif

void populateViewOpToSPIRVPatterns(
    TritonGPUToSPIRVTypeConverter &typeConverter, mlir::MLIRContext *context,
    mlir::RewritePatternSet &patterns, int numWarps,
    mlir::ModuleAxisInfoAnalysis &axisInfoAnalysis,
    mlir::ModuleAllocation *allocation, mlir::Value smem,
    mlir::PatternBenefit benefit) {
  patterns.add<ViewOpSPIRVConversion>(typeConverter, context, benefit);
  patterns.add<ExpandDimsOpSPIRVConversion>(typeConverter, context, benefit);
  patterns.add<SplatOpSPIRVConversion>(typeConverter, context, benefit);
  patterns.add<ArithConstantSplatOpSPIRVConversion>(typeConverter, context,
                                                    benefit);
}
