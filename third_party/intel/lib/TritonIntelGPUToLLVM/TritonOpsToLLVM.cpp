#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGEN/IR/TritonGENDialect.h"
#include "triton/Tools/Sys/GetEnv.hpp"

#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu::intel;

namespace {

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
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();
    VectorType v2i32 = VectorType::get(2, i32Type);
    Value payLoad = rewriter.create<LLVM::UndefOp>(loc, v2i32);
    auto createIntConstant = [&](Type type, unsigned value) {
      auto attr = rewriter.getIntegerAttr(type, value);
      return rewriter.create<LLVM::ConstantOp>(loc, type, attr);
    };
    // assert(rank == 2 && "add more support for rank != 2");
    Value offsetX = op.getOffsets()[1];
    Value offsetY = op.getOffsets()[0];
    Value idx0 = createIntConstant(i32Type, 0);
    Value idx1 = createIntConstant(i32Type, 1);
    payLoad =
        rewriter.create<LLVM::InsertElementOp>(loc, payLoad, offsetX, idx0);
    payLoad =
        rewriter.create<LLVM::InsertElementOp>(loc, payLoad, offsetY, idx1);
    rewriter.replaceOp(op, payLoad);
    return success();
  }
};

/// %oldOffset =  llvm.extract %v2i32, 0/1
/// %newOffset =  llvm.add %oldOffset, %advanceStep
/// offset = llvm.insert %v2i32, 0/1
class AdvanceOpConversion : public ConvertTritonGPUOpToLLVMPattern<AdvanceOp> {
public:
  using ConvertTritonGPUOpToLLVMPattern<
      AdvanceOp>::ConvertTritonGPUOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(AdvanceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type i32Type = rewriter.getI32Type();
    SmallVector<Value> offsets = adaptor.getOffsets();
    Value ptr = adaptor.getPtr();
    for (size_t i = 0; i < offsets.size(); i++) {
      Value offset = offsets[i];
      if (auto cst = dyn_cast<LLVM::ConstantOp>(offset.getDefiningOp()))
        if (auto attr = dyn_cast<mlir::IntegerAttr>(cst.getValue());
            attr && attr.getInt() == 0)
          continue;
      Value idx0 = rewriter.create<LLVM::ConstantOp>(
          loc, i32Type, rewriter.getIntegerAttr(i32Type, 0));
      Value idx1 = rewriter.create<LLVM::ConstantOp>(
          loc, i32Type, rewriter.getIntegerAttr(i32Type, 1));
      Value idx = i == 0 ? idx1 : idx0;
      Value oldOffset = rewriter.create<LLVM::ExtractElementOp>(loc, ptr, idx);
      Value newOffset =
          rewriter.create<LLVM::AddOp>(loc, i32Type, oldOffset, offset);
      ptr = rewriter.create<LLVM::InsertElementOp>(loc, ptr, newOffset, idx);
    }
    rewriter.replaceOp(op, ptr);
    return success();
  }
};

// TritonGen 2DBlock Prefetch/LoadOp Desc: LSC 2d block prefetch/load
// Output: for prefetch, nothing is returned. for load a vector is returned
// Arg 0: flat image base offset
// Arg 1: flat image base width
// Arg 2: flat image base height
// Arg 3: flat image base pitch
// Arg 4: offset x
// Arg 5: offset y
// Arg 6: elemSize
// Arg 7: tile width
// Arg 8: tile height
// Arg 9: V - num blocks (2 for simple 2d block read)
// Arg 10: transpose
// Arg 11: vnni transform (for transpose+transform use transpose only and
// elemSize 32)
// Arg 12: cache controls options (LSC_CACHE_OPTS)

// TritonGen 2DBlockStoreOp Desc: LSC 2d block write
// Output: nothing is returned
// Arg 0: flat image base offset
// Arg 1: flat image base width
// Arg 2: flat image base height
// Arg 3: flat image base pitch
// Arg 4: offset x
// Arg 5: offset y
// Arg 6: elemSize
// Arg 7: tile width
// Arg 8: tile height
// Arg 9: V - num blocks (2 for simple 2d block read)
// Arg 10: transpose
// Arg 11: vnni transform (for transpose+transform use transpose only and
// elemSize 32)
// Arg 12: cache controls options (LSC_CACHE_OPTS)
// Arg 13: stored value
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
    auto tType = cast<RankedTensorType>(ptrType.getPointeeType());
    unsigned rank = tType.getRank();
    assert(rank <= 2 && "only support 1d/2d load/store/prefetch for now");
    Location loc = op.getLoc();
    constexpr bool isLoad = std::is_same_v<OpType, LoadOp>;
    constexpr bool isPrefetch = std::is_same_v<OpType, PrefetchOp>;
    auto createIntConstant = [&](Type type, unsigned value) {
      auto attr = rewriter.getIntegerAttr(type, value);
      return rewriter.create<LLVM::ConstantOp>(loc, type, attr);
    };
    Type i16Type = rewriter.getI16Type();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();
    bool vnni = false;
    bool transpose = false;
    if constexpr (isLoad) {
      auto idxAttr = op->template getAttrOfType<mlir::IntegerAttr>("DotIdx");
      vnni = idxAttr.getInt() == 1 ? true : false;
    }
    unsigned dataSize = tType.getElementType().getIntOrFloatBitWidth();
    unsigned blockWidth = tType.getShape()[1];
    assert(blockWidth == 16 || blockWidth == 32 && "only support 16/32 block");
    unsigned vBlks = blockWidth == 32 ? 2 : 1;
    blockWidth = 16;
    unsigned blockHeight = tType.getShape()[0];
    Value idx0 = createIntConstant(i32Type, 0);
    Value idx1 = createIntConstant(i32Type, 1);
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
    Value bytes = createIntConstant(
        i32Type, tType.getElementType().getIntOrFloatBitWidth() / 8);
    Value one = createIntConstant(i32Type, 1);
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

    auto getIntType = [&](Type type, bool is16Bit = false) {
      auto tType = cast<RankedTensorType>(type);
      auto elemType = is16Bit ? i16Type : i32Type;
      auto ratio =
          elemType.getIntOrFloatBitWidth() / tType.getElementTypeBitWidth();
      auto num = tType.getNumElements() / 16 / ratio;
      return VectorType::get(num, elemType);
    };
    Value tensorPtr = adaptor.getPtr();
    Value offsetX =
        rewriter.create<LLVM::ExtractElementOp>(loc, tensorPtr, idx0);
    Value offsetY =
        rewriter.create<LLVM::ExtractElementOp>(loc, tensorPtr, idx1);
    if constexpr (isLoad) {
      Type resType =
          this->getTypeConverter()->convertType(op->getResult(0).getType());
      auto idxAttr = op->template getAttrOfType<mlir::IntegerAttr>("DotIdx");
      unsigned idx = idxAttr.getInt();
      Type intType = getIntType(op->getResult(0).getType(), idx == 0);
      auto load = rewriter.create<TritonGEN::Matrix2DBlockLoadOp>(
          loc, intType, base, surfaceW, surfaceH, surfaceP, offsetX, offsetY,
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
      Type intType = getIntType(op.getValue().getType());
      Value cast =
          rewriter.create<LLVM::BitcastOp>(loc, intType, adaptor.getValue());
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
    Location loc = op.getLoc();
    Type i16Type = rewriter.getI16Type();
    Type i32Type = rewriter.getI32Type();
    auto encodePrecision = [&](Type type) -> TritonGEN::PrecisionType {
      if (type == rewriter.getBF16Type())
        return TritonGEN::PrecisionType::BF16;
      else if (type == rewriter.getF16Type())
        return TritonGEN::PrecisionType::FP16;
      else if (type == rewriter.getTF32Type())
        return TritonGEN::PrecisionType::TF32;
      else {
        assert(0 && "add more support");
        return TritonGEN::PrecisionType::UNUSED;
      }
    };
    TritonGEN::PrecisionType preca =
        encodePrecision(op.getA().getType().getElementType());
    TritonGEN::PrecisionType precb =
        encodePrecision(op.getB().getType().getElementType());
    auto precA =
        TritonGEN::PrecisionTypeAttr::get(rewriter.getContext(), preca);
    auto precB =
        TritonGEN::PrecisionTypeAttr::get(rewriter.getContext(), precb);
    auto rc = IntegerAttr::get(i32Type, 8);
    auto getIntType = [&](Type type, bool is16Bit = false) {
      auto tType = cast<RankedTensorType>(type);
      Type elemType = is16Bit ? i16Type : i32Type;
      unsigned ratio =
          elemType.getIntOrFloatBitWidth() / tType.getElementTypeBitWidth();
      unsigned num = tType.getNumElements() / 16 / ratio;
      return VectorType::get(num, elemType);
    };
    Type intTypeA = getIntType(op.getA().getType(), true);
    Value castA =
        rewriter.create<LLVM::BitcastOp>(loc, intTypeA, adaptor.getA());
    Type intTypeB = getIntType(op.getB().getType());
    Value castB =
        rewriter.create<LLVM::BitcastOp>(loc, intTypeB, adaptor.getB());
    // sd dpasW fixed in genx.dpas lowering
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
    unsigned num = operands.size();
    if (num == 1) {
      rewriter.replaceOp(op, operands[0]);
    } else if (num == 2) {
      rewriter.replaceOpWithNewOp<LLVM::ShuffleVectorOp>(
          op, dstType, operands[0], operands[1], attr);
    } else if (num == 4) {
      auto subType = VectorType::get(numElts / 2, dstType.getElementType());
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
    } else {
      assert(0 && "add more support for glue op to llvm");
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
    Location loc = op.getLoc();
    Value base = adaptor.getBase();
    unsigned idx = op.getIndex();
    auto dstType =
        cast<VectorType>(getTypeConverter()->convertType(op.getType()));
    unsigned numElts = dstType.getNumElements();
    SmallVector<int32_t> indices(numElts);
    unsigned start = idx * numElts;
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

    ShapedType dstAttrType = dstElementsAttr.getType();
    auto vecType = cast<VectorType>(dstType);
    dstAttrType =
        VectorType::get(vecType.getNumElements(), vecType.getElementType());
    dstElementsAttr = dstElementsAttr.resizeSplat(dstAttrType);
    auto newOp =
        rewriter.create<LLVM::ConstantOp>(loc, dstType, dstElementsAttr);
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

} // namespace

void mlir::triton::intel::populateTritonOpsToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
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
