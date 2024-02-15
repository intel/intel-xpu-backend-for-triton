#include "PatternTritonGPUOpToLLVM.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.cpp.inc"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

using ::mlir::LLVM::getSharedMemoryObjectFromStruct;

namespace {
struct SplatOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::SplatOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::SplatOp>::ConvertTritonGPUOpToLLVMPattern;

  // Convert SplatOp or arith::ConstantOp with SplatElementsAttr to a
  // LLVM::StructType value.
  //
  // @elemType: the element type in operand.
  // @resType: the return type of the Splat-like op.
  // @constVal: a LLVM::ConstantOp or other scalar value.
  static Value convertSplatLikeOp(Type elemType, Type resType, Value constVal,
                                  TritonGPUToLLVMTypeConverter *typeConverter,
                                  ConversionPatternRewriter &rewriter,
                                  Location loc) {
    auto tensorTy = resType.cast<RankedTensorType>();
    // Check the converted type for the tensor as depending on the encoding the
    // converter may pick different element types.
    auto srcType = typeConverter->convertType(tensorTy);
    if (auto structTy = dyn_cast<LLVM::LLVMStructType>(srcType))
      srcType = structTy.getBody()[0];
    // If the type sizes don't match we need to pack constants.
    if (srcType.isIntOrFloat() && constVal.getType().getIntOrFloatBitWidth() !=
                                      srcType.getIntOrFloatBitWidth()) {
      unsigned cstBitWidth = constVal.getType().getIntOrFloatBitWidth();
      unsigned srcBitWidth = srcType.getIntOrFloatBitWidth();
      assert(cstBitWidth <= srcBitWidth && srcBitWidth % cstBitWidth == 0);
      unsigned ratio = srcBitWidth / cstBitWidth;
      Type intTy = IntegerType::get(elemType.getContext(), cstBitWidth);
      VectorType vecType = VectorType::get(ratio, intTy);
      Value intCst = bitcast(constVal, intTy);
      Value vec = undef(vecType);
      for (unsigned i = 0; i < ratio; ++i)
        vec = insert_element(vecType, vec, intCst, int_val(32, i));
      constVal = vec;
    }
    auto llSrc = bitcast(constVal, srcType);
    size_t elemsPerThread = getTotalElemsPerThread(tensorTy);
    llvm::SmallVector<Value> elems(elemsPerThread, llSrc);
    return typeConverter->packLLElements(loc, elems, rewriter, resType);
  }

  LogicalResult matchAndRewrite(triton::SplatOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    auto loc = op->getLoc();
    auto src = adaptor.getSrc();
    auto llStruct = convertSplatLikeOp(src.getType(), op.getType(), src,
                                       getTypeConverter(), rewriter, loc);
    rewriter.replaceOp(op, {llStruct});
    return success();
  }
};

// This pattern helps to convert arith::ConstantOp(with SplatElementsAttr),
// the logic is the same as triton::SplatOp, so the underlying implementation
// is reused.
struct ArithConstantSplatOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<arith::ConstantOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      arith::ConstantOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto value = op.getValue();
    if (!value.dyn_cast<SplatElementsAttr>())
      return failure();

    auto loc = op->getLoc();

    LLVM::ConstantOp arithConstantOp;
    auto values = op.getValue().dyn_cast<SplatElementsAttr>();
    auto elemType = values.getElementType();

    Attribute val;
    if (elemType.isBF16() || type::isFloat(elemType)) {
      val = values.getValues<FloatAttr>()[0];
    } else if (type::isInt(elemType)) {
      val = values.getValues<IntegerAttr>()[0];
    } else {
      llvm::errs() << "ArithConstantSplatOpConversion get unsupported type: "
                   << value.getType() << "\n";
      return failure();
    }

    auto constOp = rewriter.create<LLVM::ConstantOp>(loc, elemType, val);
    auto llStruct = SplatOpConversion::convertSplatLikeOp(
        elemType, op.getType(), constOp, getTypeConverter(), rewriter, loc);
    rewriter.replaceOp(op, llStruct);

    return success();
  }
};

struct CatOpConversion : public ConvertTritonGPUOpToLLVMPattern<CatOp> {
  using OpAdaptor = typename CatOp::Adaptor;

  explicit CatOpConversion(TritonGPUToLLVMTypeConverter &typeConverter,
                           Target target, PatternBenefit benefit = 1)
      : ConvertTritonGPUOpToLLVMPattern<CatOp>(typeConverter, target, benefit) {
  }

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
    auto lhsVals =
        getTypeConverter()->unpackLLElements(loc, adaptor.getLhs(), rewriter);
    auto rhsVals =
        getTypeConverter()->unpackLLElements(loc, adaptor.getRhs(), rewriter);
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

struct InterleaveOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<ExperimentalInterleaveOp> {
  using OpAdaptor = typename ExperimentalInterleaveOp::Adaptor;

  explicit InterleaveOpConversion(TritonGPUToLLVMTypeConverter &typeConverter,
                                  Target target, PatternBenefit benefit = 1)
      : ConvertTritonGPUOpToLLVMPattern<ExperimentalInterleaveOp>(
            typeConverter, target, benefit) {}

  LogicalResult
  matchAndRewrite(ExperimentalInterleaveOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // We rely on the following invariants of this op (which are checked by its
    // verifier):
    //
    // - The op has a blocked encoding.
    // - The last dimension (the one we're interleaving) is also the most minor
    //   dimension.
    // - The input and output encodings are the same, except the output has
    //   twice as many elements per thread in the last dimension.
    //
    // With these invariants, interleaving is trivial: We just return the i'th
    // element from lhs, followed by the i'th elem from rhs.
    Location loc = op->getLoc();
    auto resultTy = op.getType().cast<RankedTensorType>();

    SmallVector<Value> lhsVals =
        getTypeConverter()->unpackLLElements(loc, adaptor.getLhs(), rewriter);
    SmallVector<Value> rhsVals =
        getTypeConverter()->unpackLLElements(loc, adaptor.getRhs(), rewriter);
    assert(lhsVals.size() == rhsVals.size());

    SmallVector<Value> interleavedVals;
    for (int i = 0; i < lhsVals.size(); i++) {
      interleavedVals.push_back(lhsVals[i]);
      interleavedVals.push_back(rhsVals[i]);
    }

    Value ret = getTypeConverter()->packLLElements(loc, interleavedVals,
                                                   rewriter, resultTy);
    rewriter.replaceOp(op, ret);
    return success();
  }
};

struct ReshapeOpConversion : public ConvertTritonGPUOpToLLVMPattern<ReshapeOp> {
  using OpAdaptor = typename ReshapeOp::Adaptor;
  explicit ReshapeOpConversion(TritonGPUToLLVMTypeConverter &typeConverter,

                               Target target, PatternBenefit benefit = 1)
      : ConvertTritonGPUOpToLLVMPattern<ReshapeOp>(typeConverter, target,
                                                   benefit) {}

  LogicalResult
  matchAndRewrite(ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    if (triton::gpu::isExpensiveView(op.getSrc().getType(), op.getType())) {
      return emitOptionalError(loc,
                               "expensive view not supported on reshape op");
    }
    auto resultTy = op.getType().template cast<RankedTensorType>();
    auto srcTy = op.getSrc().getType().template cast<RankedTensorType>();

    auto vals = this->getTypeConverter()->unpackLLElements(
        loc, adaptor.getSrc(), rewriter);
    Value ret =
        this->getTypeConverter()->packLLElements(loc, vals, rewriter, resultTy);
    rewriter.replaceOp(op, ret);
    return success();
  }
};

struct ExpandDimsOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<ExpandDimsOp> {
  using OpAdaptor = typename ExpandDimsOp::Adaptor;
  explicit ExpandDimsOpConversion(TritonGPUToLLVMTypeConverter &typeConverter,
                                  Target target, PatternBenefit benefit = 1)
      : ConvertTritonGPUOpToLLVMPattern<ExpandDimsOp>(typeConverter, target,
                                                      benefit) {}

  LogicalResult
  matchAndRewrite(ExpandDimsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto srcVals = this->getTypeConverter()->unpackLLElements(
        loc, adaptor.getSrc(), rewriter);

    auto srcTy = op.getSrc().getType().cast<RankedTensorType>();
    auto resultTy = op.getType().template cast<RankedTensorType>();
    auto srcLayout = srcTy.getEncoding().dyn_cast<SliceEncodingAttr>();
    if (!srcLayout) {
      return emitOptionalError(
          loc, "ExpandDimsOp only supports SliceEncodingAttr as its input");
    }

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

struct TransOpConversion : public ConvertTritonGPUOpToLLVMPattern<TransOp> {
  using ConvertTritonGPUOpToLLVMPattern::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(TransOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto resultTy = op.getType().cast<RankedTensorType>();

    if (auto enc = resultTy.getEncoding().dyn_cast<SharedEncodingAttr>()) {
      auto llvmElemTy =
          getTypeConverter()->convertType(resultTy.getElementType());
      auto srcSmemObj = getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                        llvmElemTy, rewriter);
      auto dstSmemObj = SharedMemoryObject(
          srcSmemObj.base, srcSmemObj.baseElemType,
          /*strides=*/applyPermutation(srcSmemObj.strides, op.getOrder()),
          /*offsets=*/applyPermutation(srcSmemObj.offsets, op.getOrder()));
      auto retVal = getStructFromSharedMemoryObject(loc, dstSmemObj, rewriter);
      rewriter.replaceOp(op, retVal);
      return success();
    } else if (auto enc =
                   resultTy.getEncoding().dyn_cast<BlockedEncodingAttr>()) {
      // If the dst encoding is blocked, then TransOp::inferReturnTypes
      // ensures that:
      //  - the src encoding is also blocked, and
      //  - the translation from src to dst is just a "renaming" of the
      //    registers, i.e. each thread has exactly the same values.
      // Thus the transpose op simply returns the same values it got.
      auto vals = this->getTypeConverter()->unpackLLElements(
          loc, adaptor.getSrc(), rewriter);
      Value ret = this->getTypeConverter()->packLLElements(loc, vals, rewriter,
                                                           resultTy);
      rewriter.replaceOp(op, ret);
      return success();
    }

    return emitOptionalError(loc, "unsupported encoding for TransOp");
  }
};

struct BroadcastOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::BroadcastOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::BroadcastOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Following the order of indices in the legacy code, a broadcast of:
    //   [s(0), s(1) ... s(k-1),    1, s(k+1), s(k+2) ... s(n-1)]
    // =>
    //   [s(0), s(1) ... s(k-1), s(k), s(k+1), s(k+2) ... s(n-1)]
    //
    // logically maps to a broadcast within a thread's scope:
    //   [cta(0)..cta(k-1),     1,cta(k+1)..cta(n-1),spt(0)..spt(k-1),
    //   1,spt(k+1)..spt(n-1)]
    // =>
    //   [cta(0)..cta(k-1),cta(k),cta(k+1)..cta(n-1),spt(0)..spt(k-1),spt(k),spt(k+1)..spt(n-1)]
    //
    // regardless of the order of the layout
    //
    Location loc = op->getLoc();
    Value src = adaptor.getSrc();
    Value result = op.getResult();
    auto srcTy = op.getSrc().getType().cast<RankedTensorType>();
    auto resultTy = result.getType().cast<RankedTensorType>();
    auto srcLayout = srcTy.getEncoding();
    auto resultLayout = resultTy.getEncoding();
    auto srcShape = srcTy.getShape();
    auto resultShape = resultTy.getShape();
    unsigned rank = srcTy.getRank();
    auto typeConverter = getTypeConverter();

    assert(rank == resultTy.getRank());
    auto order = triton::gpu::getOrder(srcLayout);
    auto srcOffsets = emitOffsetForLayout(srcLayout, srcTy);
    auto resultOffsets = emitOffsetForLayout(resultLayout, resultTy);
    SmallVector<Value> srcVals =
        typeConverter->unpackLLElements(loc, src, rewriter);

    std::map<SmallVector<unsigned>, Value> srcValues;
    for (size_t i = 0; i < srcOffsets.size(); i++) {
      srcValues[srcOffsets[i]] = srcVals[i];
    }

    SmallVector<Value> resultVals;
    for (size_t i = 0; i < resultOffsets.size(); i++) {
      auto offset = resultOffsets[i];
      for (size_t j = 0; j < srcShape.size(); j++)
        if (srcShape[j] == 1)
          offset[j] = 0;
      resultVals.push_back(srcValues.at(offset));
    }

    Value resultStruct =
        typeConverter->packLLElements(loc, resultVals, rewriter, resultTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

} // namespace

void mlir::triton::populateViewOpToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    Target target, PatternBenefit benefit) {
  patterns.add<ReshapeOpConversion>(typeConverter, target, benefit);
  patterns.add<ExpandDimsOpConversion>(typeConverter, target, benefit);
  patterns.add<SplatOpConversion>(typeConverter, target, benefit);
  patterns.add<ArithConstantSplatOpConversion>(typeConverter, target, benefit);
  patterns.add<CatOpConversion>(typeConverter, target, benefit);
  patterns.add<InterleaveOpConversion>(typeConverter, target, benefit);
  patterns.add<TransOpConversion>(typeConverter, target, benefit);
  patterns.add<BroadcastOpConversion>(typeConverter, target, benefit);
}
