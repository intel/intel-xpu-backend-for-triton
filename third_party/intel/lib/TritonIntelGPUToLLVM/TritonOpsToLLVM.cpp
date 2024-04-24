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
  unsigned num = tensorType.getNumElements() / 16 / ratio;
  return vec_ty(elemType, num);
};

/// v2i32 [offsetX, offsetY] for 2D tensor desc
class MakeTensorPtrOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<MakeTensorPtrOp> {
public:
  using ConvertTritonGPUOpToLLVMPattern<
      MakeTensorPtrOp>::ConvertTritonGPUOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(MakeTensorPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    IntegerType i32Type = rewriter.getI32Type();
    VectorType v2i32 = vec_ty(i32Type, 2);
    Value offsetX = op.getOffsets()[1];
    Value offsetY = op.getOffsets()[0];
    Value payLoad = undef(v2i32);
    Value idx0 = i32_val(0);
    Value idx1 = i32_val(1);
    payLoad = insert_element(payLoad, offsetX, idx0);
    payLoad = insert_element(payLoad, offsetY, idx1);
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
    SmallVector<Value> offsets = adaptor.getOffsets();
    Value ptr = adaptor.getPtr();
    for (size_t i = 0; i < offsets.size(); ++i) {
      Value offset = offsets[i];
      if (auto cst = dyn_cast<LLVM::ConstantOp>(offset.getDefiningOp()))
        if (auto attr = dyn_cast<mlir::IntegerAttr>(cst.getValue());
            attr && attr.getInt() == 0)
          continue;

      IntegerType i32Type = rewriter.getI32Type();
      Value idx = (i == 0)
                      ? rewriter.create<LLVM::ConstantOp>(
                            loc, i32Type, rewriter.getIntegerAttr(i32Type, 1))
                      : rewriter.create<LLVM::ConstantOp>(
                            loc, i32Type, rewriter.getIntegerAttr(i32Type, 0));
      Value oldOffset = extract_element(ptr, idx);
      Value newOffset = add(i32Type, oldOffset, offset);
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
template <typename OpType>
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

    Location loc = op.getLoc();
    constexpr bool isLoad = std::is_same_v<OpType, LoadOp>;
    constexpr bool isPrefetch = std::is_same_v<OpType, PrefetchOp>;

    IntegerType i16Type = rewriter.getI16Type();
    IntegerType i32Type = rewriter.getI32Type();
    IntegerType i64Type = rewriter.getI64Type();
    bool vnni = false, transpose = false;
    if constexpr (isLoad) {
      auto idxAttr = op->template getAttrOfType<mlir::IntegerAttr>("DotIdx");
      vnni = idxAttr.getInt() == 1 ? true : false;
    }

    unsigned dataSize = tensorType.getElementType().getIntOrFloatBitWidth();
    unsigned blockWidth = tensorType.getShape()[1];
    assert(blockWidth == 16 || blockWidth == 32 && "only support 16/32 block");
    unsigned vBlks = blockWidth == 32 ? 2 : 1;
    blockWidth = 16;
    unsigned blockHeight = tensorType.getShape()[0];
    Value idx0 = i32_val(0);
    Value idx1 = i32_val(1);
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
    Value bytes =
        i32_val(tensorType.getElementType().getIntOrFloatBitWidth() / 8);
    Value one = i32_val(1);
    Value surfaceW =
        rewriter.create<arith::TruncIOp>(loc, i32Type, ptrOp.getShape()[1]);
    surfaceW = rewriter.create<arith::MulIOp>(loc, surfaceW, bytes);
    surfaceW = rewriter.create<arith::SubIOp>(loc, surfaceW, one);
    Value surfaceH =
        rewriter.create<arith::TruncIOp>(loc, i32Type, ptrOp.getShape()[0]);
    surfaceH = rewriter.create<arith::SubIOp>(loc, surfaceH, one);
    Value surfaceP =
        rewriter.create<arith::TruncIOp>(loc, i32Type, ptrOp.getStrides()[0]);
    surfaceP = rewriter.create<arith::MulIOp>(loc, surfaceP, bytes);
    surfaceP = rewriter.create<arith::SubIOp>(loc, surfaceP, one);
    rewriter.restoreInsertionPoint(insertPoint);

    Value tensorPtr = adaptor.getPtr();
    Value offsetX = extract_element(tensorPtr, idx0);
    Value offsetY = extract_element(tensorPtr, idx1);

    if constexpr (isLoad) {
      Type resType =
          this->getTypeConverter()->convertType(op->getResult(0).getType());
      auto idxAttr = op->template getAttrOfType<mlir::IntegerAttr>("DotIdx");
      unsigned idx = idxAttr.getInt();
      Type vectorType =
          getVectorType(cast<RankedTensorType>(op->getResult(0).getType()),
                        idx == 0 ? i16Type : i32Type);
      auto load = rewriter.create<TritonGEN::Matrix2DBlockLoadOp>(
          loc, vectorType, base, surfaceW, surfaceH, surfaceP, offsetX, offsetY,
          dataSize, blockWidth, blockHeight, vBlks, transpose, vnni);
      auto cast = rewriter.create<LLVM::BitcastOp>(loc, resType, load);
      rewriter.replaceOp(op, cast);
    } else if constexpr (isPrefetch) {
      rewriter.create<TritonGEN::Matrix2DBlockPrefetchOp>(
          loc, base, surfaceW, surfaceH, surfaceP, offsetX, offsetY, dataSize,
          blockWidth, blockHeight, vBlks, transpose, vnni,
          TritonGEN::PrefetchCacheControl::L1C_L3C);
      rewriter.eraseOp(op);
    } else {
      VectorType vectorType = getVectorType(
          cast<RankedTensorType>(op.getValue().getType()), i32Type);
      Value cast =
          rewriter.create<LLVM::BitcastOp>(loc, vectorType, adaptor.getValue());
      rewriter.create<TritonGEN::Matrix2DBlockStoreOp>(
          loc, base, surfaceW, surfaceH, surfaceP, offsetX, offsetY, dataSize,
          blockWidth, blockHeight, vBlks, transpose, vnni, cast);
      rewriter.eraseOp(op);
    }
    return success();
  }
};

// TritonGen DpasOp Desc: XeHP SDV: dot product accumulate systolic
// Output: dst
// Arg 0: src0(acc)
// Arg 1: src1
// Arg 2: src2
// Arg 3: src1's precision
// Arg 4: src2's precision
// Arg 5: systolic depth
// Arg 6: repeat count
// Arg 7: isDpasw
class DotOpConversion : public ConvertTritonGPUOpToLLVMPattern<DotOp> {
public:
  using ConvertTritonGPUOpToLLVMPattern<DotOp>::ConvertTritonGPUOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto encodePrecision = [&](Type type) -> TritonGEN::PrecisionType {
      if (type == rewriter.getBF16Type())
        return TritonGEN::PrecisionType::BF16;
      else if (type == rewriter.getF16Type())
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
    IntegerType i16Type = rewriter.getI16Type();
    IntegerType i32Type = rewriter.getI32Type();
    VectorType typeA =
        getVectorType(cast<RankedTensorType>(op.getA().getType()), i16Type);
    Value castA = rewriter.create<LLVM::BitcastOp>(loc, typeA, adaptor.getA());
    VectorType typeB =
        getVectorType(cast<RankedTensorType>(op.getB().getType()), i32Type);
    Value castB = rewriter.create<LLVM::BitcastOp>(loc, typeB, adaptor.getB());
    auto rc = IntegerAttr::get(i32Type, 8);
    // sd dpasW fixed in genx.dpas lowering.
    auto dpas = rewriter.create<TritonGEN::MatrixDPASOp>(
        loc, adaptor.getC().getType(), adaptor.getC(), castA, castB, precA,
        precB, rc);
    rewriter.replaceOp(op, dpas);
    return success();
  }
};

/// %glue = ttgi.glue %a, %b : tensor<4xf16>, tensor<4xf16> : tensor<8xf16>
/// is converted to
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
      auto shfl = rewriter.create<LLVM::ShuffleVectorOp>(loc, dstType, shfl01,
                                                         shfl23, attr);
      rewriter.replaceOp(op, shfl);
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

    // arith.constant should only have vector or tenor types.
    assert((isa<VectorType, RankedTensorType>(srcType)));

    Type dstType = getTypeConverter()->convertType(srcType);
    if (!dstType)
      return failure();

    auto dstElementsAttr = dyn_cast<DenseElementsAttr>(op.getValue());
    if (!dstElementsAttr)
      return failure();

    auto vecType = cast<VectorType>(dstType);
    ShapedType dstAttrType =
        vec_ty(vecType.getElementType(), vecType.getNumElements());
    dstElementsAttr = dstElementsAttr.resizeSplat(dstAttrType);
    auto newOp =
        rewriter.create<LLVM::ConstantOp>(loc, dstType, dstElementsAttr);
    rewriter.replaceOp(op, newOp);
    return success();
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
