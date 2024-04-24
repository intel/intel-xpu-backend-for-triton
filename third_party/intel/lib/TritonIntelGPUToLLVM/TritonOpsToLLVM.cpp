#include "PatternTritonGPUOpToLLVM.h"
#include "triton/Analysis/Utility.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGEN/IR/TritonGENDialect.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu::intel;

namespace {

VectorType getVectorType(RankedTensorType tensorType, Type elemType) {
  unsigned ratio =
      elemType.getIntOrFloatBitWidth() / tensorType.getElementTypeBitWidth();
  unsigned num = (tensorType.getNumElements() / 16) / ratio;
  return vec_ty(elemType, num);
};

/// v2i32 [offsetX, offsetY] for 2D tensor desc.
class MakeTensorPtrOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<MakeTensorPtrOp> {
public:
  using ConvertTritonGPUOpToLLVMPattern<
      MakeTensorPtrOp>::ConvertTritonGPUOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(MakeTensorPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    VectorType v2i32 = vec_ty(i32_ty, 2);
    Value offsetX = op.getOffsets()[1];
    Value offsetY = op.getOffsets()[0];
    Value payLoad = undef(v2i32);
    payLoad = insert_element(payLoad, offsetX, i32_val(0));
    payLoad = insert_element(payLoad, offsetY, i32_val(1));
    rewriter.replaceOp(op, payLoad);
    return success();
  }
};

/// %oldOffset = llvm.extract %v2i32, 0/1
/// %newOffset = llvm.add %oldOffset, %advanceStep
/// offset = llvm.insert %v2i32, 0/1
class AdvanceOpConversion : public ConvertTritonGPUOpToLLVMPattern<AdvanceOp> {
public:
  using ConvertTritonGPUOpToLLVMPattern<
      AdvanceOp>::ConvertTritonGPUOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(AdvanceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ValueRange offsets = adaptor.getOffsets();
    Value ptr = adaptor.getPtr();

    for (size_t i = 0; i < offsets.size(); ++i) {
      Value offset = offsets[i];
      if (auto cst = dyn_cast<LLVM::ConstantOp>(offset.getDefiningOp()))
        if (auto attr = dyn_cast<mlir::IntegerAttr>(cst.getValue());
            attr && attr.getInt() == 0)
          continue;

      Value idx = i32_val(!i);
      Value oldOffset = extract_element(ptr, idx);
      Value newOffset = add(i32_ty, oldOffset, offset);
      ptr = insert_element(ptr, newOffset, idx);
    }

    rewriter.replaceOp(op, ptr);
    return success();
  }
};

/// TritonGen 2DBlock Prefetch/LoadOp Desc: LSC 2d block prefetch/load
/// Output: for prefetch, nothing is returned. for load a vector is returned
/// Arg 0: flat image base offset
/// Arg 1: flat image base width
/// Arg 2: flat image base height
/// Arg 3: flat image base pitch
/// Arg 4: offset x
/// Arg 5: offset y
/// Arg 6: elemSize
/// Arg 7: tile width
/// Arg 8: tile height
/// Arg 9: V - num blocks (2 for simple 2d block read)
/// Arg 10: transpose
/// Arg 11: vnni transform (for transpose+transform use transpose only and
///         elemSize 32)
/// Arg 12: cache controls options (LSC_CACHE_OPTS)

/// TritonGen 2DBlockStoreOp Desc: LSC 2d block write
/// Output: nothing is returned
/// Arg 0: flat image base offset
/// Arg 1: flat image base width
/// Arg 2: flat image base height
/// Arg 3: flat image base pitch
/// Arg 4: offset x
/// Arg 5: offset y
/// Arg 6: elemSize
/// Arg 7: tile width
/// Arg 8: tile height
/// Arg 9: V - num blocks (2 for simple 2d block read)
/// Arg 10: transpose
/// Arg 11: vnni transform (for transpose+transform use transpose only and
///         elemSize 32)
/// Arg 12: cache controls options (LSC_CACHE_OPTS)
/// Arg 13: stored value
template <typename OpType, typename = std::enable_if_t<llvm::is_one_of<
                               OpType, PrefetchOp, LoadOp, StoreOp>::value>>
class LoadStorePrefetchOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<OpType> {
public:
  using ConvertTritonGPUOpToLLVMPattern<
      OpType>::ConvertTritonGPUOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(OpType op, typename OpType::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ptrType = cast<PointerType>(op.getPtr().getType());
    auto tensorType = cast<RankedTensorType>(ptrType.getPointeeType());
    assert(tensorType.getRank() <= 2 &&
           "only support 1d/2d load/store/prefetch for now");

    unsigned dataSize = tensorType.getElementType().getIntOrFloatBitWidth();
    unsigned blockWidth = tensorType.getShape()[1];
    assert(blockWidth == 16 || blockWidth == 32 && "only support 16/32 block");
    unsigned vBlks = blockWidth == 32 ? 2 : 1;
    blockWidth = 16;
    unsigned blockHeight = tensorType.getShape()[0];
    Value ptr = op.getPtr();
    if (auto cast =
            dyn_cast<mlir::UnrealizedConversionCastOp>(ptr.getDefiningOp()))
      ptr = cast.getInputs()[0];

    MakeTensorPtrOp ptrOp = getMakeTensorPtrOp(ptr);
    Value base = ptrOp.getBase();
    if (auto cast =
            dyn_cast<mlir::UnrealizedConversionCastOp>(base.getDefiningOp()))
      base = cast.getInputs()[0];

    OpBuilder::InsertPoint insertPoint = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointAfter(ptrOp);
    Location loc = op.getLoc();
    Value bytes =
        i32_val(tensorType.getElementType().getIntOrFloatBitWidth() / 8);
    Value one = i32_val(1);
    Value surfaceW = sub(mul(trunc(i32_ty, ptrOp.getShape()[1]), bytes), one);
    Value surfaceH = sub(trunc(i32_ty, ptrOp.getShape()[0]), one);
    Value surfaceP = sub(mul(trunc(i32_ty, ptrOp.getStrides()[0]), bytes), one);
    rewriter.restoreInsertionPoint(insertPoint);

    Value tensorPtr = adaptor.getPtr();
    Value offsetX = extract_element(tensorPtr, i32_val(0));
    Value offsetY = extract_element(tensorPtr, i32_val(1));

    if constexpr (std::is_same_v<OpType, LoadOp>) {
      auto idxAttr = op->template getAttrOfType<mlir::IntegerAttr>("DotIdx");
      unsigned idx = idxAttr.getInt();
      Type resType =
          this->getTypeConverter()->convertType(op->getResult(0).getType());
      Type vectorType =
          getVectorType(cast<RankedTensorType>(op.getResult().getType()),
                        idx == 0 ? i16_ty : i32_ty);
      bool vnni = (idx == 1) && dataSize <= 32;
      auto load = rewriter.create<TritonGEN::Matrix2DBlockLoadOp>(
          loc, vectorType, base, surfaceW, surfaceH, surfaceP, offsetX, offsetY,
          dataSize, blockWidth, blockHeight, vBlks, false /* transpose*/, vnni);
      rewriter.replaceOp(op, bitcast(load, resType));
    } else if constexpr (std::is_same_v<OpType, PrefetchOp>) {
      rewriter.create<TritonGEN::Matrix2DBlockPrefetchOp>(
          loc, base, surfaceW, surfaceH, surfaceP, offsetX, offsetY, dataSize,
          blockWidth, blockHeight, vBlks, false /*transpose*/, false /*vnni*/,
          TritonGEN::PrefetchCacheControl::L1C_L3C);
      rewriter.eraseOp(op);
    } else {
      VectorType vectorType = getVectorType(
          cast<RankedTensorType>(op.getValue().getType()), i32_ty);
      rewriter.create<TritonGEN::Matrix2DBlockStoreOp>(
          loc, base, surfaceW, surfaceH, surfaceP, offsetX, offsetY, dataSize,
          blockWidth, blockHeight, vBlks, false /*transpose*/, false /*vnni*/,
          bitcast(adaptor.getValue(), vectorType));
      rewriter.eraseOp(op);
    }

    return success();
  }
};

/// TritonGen DpasOp Desc: XeHP SDV: dot product accumulate systolic
/// Output: dst
/// Arg 0: src0(acc)
/// Arg 1: src1
/// Arg 2: src2
/// Arg 3: src1's precision
/// Arg 4: src2's precision
/// Arg 5: systolic depth
/// Arg 6: repeat count
/// Arg 7: isDpasw
class DotOpConversion : public ConvertTritonGPUOpToLLVMPattern<DotOp> {
public:
  using ConvertTritonGPUOpToLLVMPattern<DotOp>::ConvertTritonGPUOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto encodePrecision = [&](Type type) -> TritonGEN::PrecisionType {
      if (type == bf16_ty)
        return TritonGEN::PrecisionType::BF16;
      else if (type == f16_ty)
        return TritonGEN::PrecisionType::FP16;
      else if (type == rewriter.getTF32Type())
        return TritonGEN::PrecisionType::TF32;
      assert(false && "add more support");
      return TritonGEN::PrecisionType::UNUSED;
    };

    TritonGEN::PrecisionType precATy =
        encodePrecision(op.getA().getType().getElementType());
    TritonGEN::PrecisionType precBTy =
        encodePrecision(op.getB().getType().getElementType());
    auto precA =
        TritonGEN::PrecisionTypeAttr::get(rewriter.getContext(), precATy);
    auto precB =
        TritonGEN::PrecisionTypeAttr::get(rewriter.getContext(), precBTy);

    Location loc = op.getLoc();
    Type typeA =
        getVectorType(cast<RankedTensorType>(op.getA().getType()), i16_ty);
    Value castA = bitcast(adaptor.getA(), typeA);
    VectorType typeB =
        getVectorType(cast<RankedTensorType>(op.getB().getType()), i32_ty);
    Value castB = bitcast(adaptor.getB(), typeB);
    auto rc = IntegerAttr::get(i32_ty, 8);
    // sd dpasW fixed in genx.dpas lowering.
    rewriter.replaceOpWithNewOp<TritonGEN::MatrixDPASOp>(
        op, adaptor.getC().getType(), adaptor.getC(), castA, castB, precA,
        precB, rc);
    return success();
  }
};

/// %glue = ttgi.glue %a, %b : tensor<4xf16>, tensor<4xf16> : tensor<8xf16>
/// is converted to:
/// %glue = llvm.shufflevector %a, %b : [0, 1, 2, 3, 4, 5, 6, 7] : vector<8xf16>
class GlueOpConversion : public ConvertTritonGPUOpToLLVMPattern<GlueOp> {
public:
  using ConvertTritonGPUOpToLLVMPattern<
      GlueOp>::ConvertTritonGPUOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(GlueOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    SmallVector<Value> operands = adaptor.getOperands();
    auto dstType =
        cast<VectorType>(getTypeConverter()->convertType(op.getType()));
    unsigned numElts = dstType.getNumElements();
    SmallVector<int32_t> indices(numElts);
    std::iota(indices.begin(), indices.end(), 0);
    DenseI32ArrayAttr attr = rewriter.getDenseI32ArrayAttr(indices);

    switch (operands.size()) {
    case 1:
      rewriter.replaceOp(op, operands[0]);
      break;
    case 2:
      rewriter.replaceOpWithNewOp<LLVM::ShuffleVectorOp>(
          op, dstType, operands[0], operands[1], attr);
      break;
    case 4: {
      auto subType = vec_ty(dstType.getElementType(), numElts / 2);
      indices.pop_back_n(numElts / 2);
      DenseI32ArrayAttr attr01 = rewriter.getDenseI32ArrayAttr(indices);
      auto shfl01 = rewriter.create<LLVM::ShuffleVectorOp>(
          loc, subType, operands[0], operands[1], attr01);
      DenseI32ArrayAttr attr23 = rewriter.getDenseI32ArrayAttr(indices);
      auto shfl23 = rewriter.create<LLVM::ShuffleVectorOp>(
          loc, subType, operands[2], operands[3], attr23);
      rewriter.replaceOpWithNewOp<LLVM::ShuffleVectorOp>(op, dstType, shfl01,
                                                         shfl23, attr);
    } break;
    default:
      llvm_unreachable("add more support for glue op to llvm");
    }

    return success();
  }
};

/// %extract = ttgi.extract %a[0] : tensor<8xf16> -> tensor<4xf16>
/// is converted to
/// %extract = llvm.shufflevector %a, %a : [0, 1, 2, 3] : vector<4xf16>
class ExtractOpConversion : public ConvertTritonGPUOpToLLVMPattern<ExtractOp> {
public:
  using ConvertTritonGPUOpToLLVMPattern<
      ExtractOp>::ConvertTritonGPUOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value base = adaptor.getBase();
    auto dstType =
        cast<VectorType>(getTypeConverter()->convertType(op.getType()));
    unsigned numElts = dstType.getNumElements();
    SmallVector<int32_t> indices(numElts);
    unsigned start = op.getIndex() * numElts;
    std::iota(indices.begin(), indices.end(), start);
    DenseI32ArrayAttr attr = rewriter.getDenseI32ArrayAttr(indices);
    rewriter.replaceOpWithNewOp<LLVM::ShuffleVectorOp>(op, dstType, base, base,
                                                       attr);
    return success();
  }
};

// FIXME: support it in upstream constantOpLowering
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

    assert((isa<VectorType, RankedTensorType>(srcType)) &&
           "arith.constant should only have vector or tensor type");

    if (Type dstType = getTypeConverter()->convertType(srcType)) {
      if (auto dstElementsAttr = dyn_cast<DenseElementsAttr>(op.getValue())) {
        auto vecType = cast<VectorType>(dstType);
        VectorType dstAttrType =
            vec_ty(vecType.getElementType(), vecType.getNumElements());
        dstElementsAttr = dstElementsAttr.resizeSplat(dstAttrType);
        rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(op, vecType,
                                                      dstElementsAttr);
        return success();
      }
    }

    return failure();
  }
};

} // namespace

void mlir::triton::intel::populateTritonOpsToLLVMPatterns(
    TritonIntelGPUToLLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<MakeTensorPtrOpConversion>(typeConverter, benefit);
  patterns.add<AdvanceOpConversion>(typeConverter, benefit);
  patterns.add<DotOpConversion>(typeConverter, benefit);
  patterns.add<LoadStorePrefetchOpConversion<PrefetchOp>>(typeConverter,
                                                          benefit);
  patterns.add<LoadStorePrefetchOpConversion<LoadOp>>(typeConverter, benefit);
  patterns.add<LoadStorePrefetchOpConversion<StoreOp>>(typeConverter, benefit);
  patterns.add<GlueOpConversion>(typeConverter, benefit);
  patterns.add<ExtractOpConversion>(typeConverter, benefit);
  patterns.add<ArithConstantOpLowering>(typeConverter, benefit);
}
