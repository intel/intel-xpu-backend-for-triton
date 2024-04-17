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

/// offsetX, offsetY for 2D tensor desc
class MakeTensorPtrOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<MakeTensorPtrOp> {
public:
  using ConvertTritonGPUOpToLLVMPattern<
      MakeTensorPtrOp>::ConvertTritonGPUOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(MakeTensorPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto i32Type = rewriter.getI32Type();
    auto i64Type = rewriter.getI64Type();
    auto v2i32 = VectorType::get(2, i32Type);
    Value payLoad = rewriter.create<LLVM::UndefOp>(loc, v2i32);
    auto createIntConstant = [&](Type type, unsigned value) {
      auto attr = rewriter.getIntegerAttr(type, value);
      return rewriter.create<LLVM::ConstantOp>(loc, type, attr);
    };
    // if (rank == 2) {
    auto offsetX = op.getOffsets()[1];
    auto offsetY = op.getOffsets()[0];
    auto idx0 = createIntConstant(i32Type, 0);
    auto idx1 = createIntConstant(i32Type, 1);
    payLoad =
        rewriter.create<LLVM::InsertElementOp>(loc, payLoad, offsetX, idx0);
    payLoad =
        rewriter.create<LLVM::InsertElementOp>(loc, payLoad, offsetY, idx1);
    rewriter.replaceOp(op, payLoad);
    return success();
  }
};

class AdvanceOpConversion : public ConvertTritonGPUOpToLLVMPattern<AdvanceOp> {
public:
  using ConvertTritonGPUOpToLLVMPattern<
      AdvanceOp>::ConvertTritonGPUOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(AdvanceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto i32Type = rewriter.getI32Type();
    auto offsets = adaptor.getOffsets();
    Value ptr = adaptor.getPtr();
    for (size_t i = 0; i < offsets.size(); i++) {
      auto offset = offsets[i];
      if (auto cst = dyn_cast<LLVM::ConstantOp>(offset.getDefiningOp()))
        if (auto attr = dyn_cast<mlir::IntegerAttr>(cst.getValue());
            attr && attr.getInt() == 0)
          continue;
      auto idx0 = rewriter.create<LLVM::ConstantOp>(
          loc, i32Type, rewriter.getIntegerAttr(i32Type, 0));
      auto idx1 = rewriter.create<LLVM::ConstantOp>(
          loc, i32Type, rewriter.getIntegerAttr(i32Type, 1));
      Value idx = i == 0 ? idx1 : idx0;
      auto oldOffset = rewriter.create<LLVM::ExtractElementOp>(loc, ptr, idx);
      auto newOffset =
          rewriter.create<LLVM::AddOp>(loc, i32Type, oldOffset, offset);
      ptr = rewriter.create<LLVM::InsertElementOp>(loc, ptr, newOffset, idx);
    }
    rewriter.replaceOp(op, ptr);
    return success();
  }
};

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
    auto rank = tType.getRank();
    assert(rank <= 2 && "only support 1d/2d load/store/prefetch for now");
    auto loc = op.getLoc();
    constexpr bool isLoad = std::is_same_v<OpType, LoadOp>;
    constexpr bool isPrefetch = std::is_same_v<OpType, PrefetchOp>;
    auto createIntConstant = [&](Type type, unsigned value) {
      auto attr = rewriter.getIntegerAttr(type, value);
      return rewriter.create<LLVM::ConstantOp>(loc, type, attr);
    };
    auto i16Type = rewriter.getI16Type();
    auto i32Type = rewriter.getI32Type();
    auto i64Type = rewriter.getI64Type();
    auto v4i64 = VectorType::get(4, i64Type);
    auto vnni = false;
    auto transpose = false;
    if constexpr (isLoad) {
      auto idxAttr = op->template getAttrOfType<mlir::IntegerAttr>("DotIdx");
      vnni = idxAttr.getInt() == 1 ? true : false;
    }
    unsigned dataSize = tType.getElementType().getIntOrFloatBitWidth();
    auto blockWidth = tType.getShape()[1];
    assert(blockWidth == 16 || blockWidth == 32 && "only support 16/32 block");
    auto vBlks = blockWidth == 32 ? 2 : 1;
    blockWidth = 16;
    auto blockHeight = tType.getShape()[0];
    auto idx0 = createIntConstant(i32Type, 0);
    auto idx1 = createIntConstant(i32Type, 1);
    Value ptr = op.getPtr();
    if (auto cast =
            dyn_cast<mlir::UnrealizedConversionCastOp>(ptr.getDefiningOp()))
      ptr = cast.getInputs()[0];
    MakeTensorPtrOp ptrOp = getMakeTensorPtrOp(ptr);
    Value base = ptrOp.getBase();
    if (auto cast =
            dyn_cast<mlir::UnrealizedConversionCastOp>(base.getDefiningOp()))
      base = cast.getInputs()[0];

    auto insertPoint = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointAfter(ptrOp);
    auto bytes = createIntConstant(
        i32Type, tType.getElementType().getIntOrFloatBitWidth() / 8);
    auto one = createIntConstant(i32Type, 1);
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
    auto tensorPtr = adaptor.getPtr();
    auto offsetX =
        rewriter.create<LLVM::ExtractElementOp>(loc, tensorPtr, idx0);
    auto offsetY =
        rewriter.create<LLVM::ExtractElementOp>(loc, tensorPtr, idx1);
    if constexpr (isLoad) {
      auto resType =
          this->getTypeConverter()->convertType(op->getResult(0).getType());
      auto idxAttr = op->template getAttrOfType<mlir::IntegerAttr>("DotIdx");
      auto idx = idxAttr.getInt();
      auto intType = getIntType(op->getResult(0).getType(), idx == 0);
      auto load = rewriter.create<TritonGEN::Matrix2DBlockLoadOp>(
          loc, intType, base, surfaceW, surfaceH, surfaceP, offsetX, offsetY,
          dataSize, blockWidth, blockHeight, vBlks, transpose, vnni);
      auto cast = rewriter.create<LLVM::BitcastOp>(loc, resType, load);
      rewriter.replaceOp(op, cast);
    } else if constexpr (isPrefetch) {
      auto load = rewriter.create<TritonGEN::Matrix2DBlockPrefetchOp>(
          loc, base, surfaceW, surfaceH, surfaceP, offsetX, offsetY, dataSize,
          blockWidth, blockHeight, 1, transpose, vnni,
          TritonGEN::PrefetchCacheControl::L1C_L3C);
      rewriter.eraseOp(op);
    } else {
      auto intType = getIntType(op.getValue().getType());
      auto cast =
          rewriter.create<LLVM::BitcastOp>(loc, intType, adaptor.getValue());
      rewriter.create<TritonGEN::Matrix2DBlockStoreOp>(
          loc, base, surfaceW, surfaceH, surfaceP, offsetX, offsetY, dataSize,
          blockWidth, blockHeight, vBlks, transpose, vnni, cast);
      rewriter.eraseOp(op);
    }
    return success();
  }
};

class DotOpConversion : public ConvertTritonGPUOpToLLVMPattern<DotOp> {
public:
  using ConvertTritonGPUOpToLLVMPattern<DotOp>::ConvertTritonGPUOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto i16Type = rewriter.getI16Type();
    auto i32Type = rewriter.getI32Type();
    auto encodePrecision = [&](Type type) -> TritonGEN::PrecisionType {
      if (type == rewriter.getBF16Type())
        return TritonGEN::PrecisionType::BF16; // 9;
      else if (type == rewriter.getF16Type())
        return TritonGEN::PrecisionType::FP16; // 10;
      else if (type == rewriter.getTF32Type())
        return TritonGEN::PrecisionType::TF32; // 12;
      else {
        assert(0 && "add more support");
        return TritonGEN::PrecisionType::UNUSED;
      }
    };
    auto preca = encodePrecision(op.getA().getType().getElementType());
    auto precb = encodePrecision(op.getB().getType().getElementType());
    auto precA =
        TritonGEN::PrecisionTypeAttr::get(rewriter.getContext(), preca);
    auto precB =
        TritonGEN::PrecisionTypeAttr::get(rewriter.getContext(), precb);
    auto rc = IntegerAttr::get(i32Type, 8);
    // sd dpasW fixed in genx.dpas lowering
    auto getIntType = [&](Type type, bool is16Bit = false) {
      auto tType = cast<RankedTensorType>(type);
      auto elemType = is16Bit ? i16Type : i32Type;
      auto ratio =
          elemType.getIntOrFloatBitWidth() / tType.getElementTypeBitWidth();
      auto num = tType.getNumElements() / 16 / ratio;
      return VectorType::get(num, elemType);
    };
    auto intTypeA = getIntType(op.getA().getType(), true);
    auto castA =
        rewriter.create<LLVM::BitcastOp>(loc, intTypeA, adaptor.getA());
    auto intTypeB = getIntType(op.getB().getType());
    auto castB =
        rewriter.create<LLVM::BitcastOp>(loc, intTypeB, adaptor.getB());
    auto dpas = rewriter.create<TritonGEN::MatrixDPASOp>(
        loc, adaptor.getC().getType(), adaptor.getC(), castA, castB, precA,
        precB, rc);
    rewriter.replaceOp(op, dpas);
    return success();
  }
};

class GlueOpConversion : public ConvertTritonGPUOpToLLVMPattern<GlueOp> {
public:
  using ConvertTritonGPUOpToLLVMPattern<
      GlueOp>::ConvertTritonGPUOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(GlueOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto operands = adaptor.getOperands();
    auto dstType =
        cast<VectorType>(getTypeConverter()->convertType(op.getType()));
    auto numElts = dstType.getNumElements();
    SmallVector<int32_t> indices(numElts);
    std::iota(indices.begin(), indices.end(), 0);
    auto attr = rewriter.getDenseI32ArrayAttr(indices);
    auto num = operands.size();
    if (num == 1) {
      rewriter.replaceOp(op, operands[0]);
    } else if (num == 2) {
      rewriter.replaceOpWithNewOp<LLVM::ShuffleVectorOp>(
          op, dstType, operands[0], operands[1], attr);
    } else if (num == 4) {
      auto subType = VectorType::get(numElts / 2, dstType.getElementType());
      indices.pop_back_n(numElts / 2);
      auto attr01 = rewriter.getDenseI32ArrayAttr(indices);
      auto shfl01 = rewriter.create<LLVM::ShuffleVectorOp>(
          loc, subType, operands[0], operands[1], attr01);
      auto attr23 = rewriter.getDenseI32ArrayAttr(indices);
      auto shfl23 = rewriter.create<LLVM::ShuffleVectorOp>(
          loc, subType, operands[2], operands[3], attr23);
      auto shfl = rewriter.create<LLVM::ShuffleVectorOp>(loc, dstType, shfl01,
                                                         shfl23, attr);
      rewriter.replaceOp(op, shfl);
    } else {
      assert(0 && "add more support for tt.glue to llvm");
    }
    return success();
  }
};

class CastOpConversion : public ConvertTritonGPUOpToLLVMPattern<CastOp> {
public:
  using ConvertTritonGPUOpToLLVMPattern<
      CastOp>::ConvertTritonGPUOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(CastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto dstType = getTypeConverter()->convertType(op.getType());
    auto cast =
        rewriter.create<LLVM::BitcastOp>(loc, dstType, adaptor.getSrc());
    rewriter.replaceOp(op, cast);
    return success();
  }
};

class ExtractOpConversion : public ConvertTritonGPUOpToLLVMPattern<ExtractOp> {
public:
  using ConvertTritonGPUOpToLLVMPattern<
      ExtractOp>::ConvertTritonGPUOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto base = adaptor.getBase();
    auto idx = op.getIdx();
    auto dstType =
        cast<VectorType>(getTypeConverter()->convertType(op.getType()));
    auto numElts = dstType.getNumElements();
    SmallVector<int32_t> indices(numElts);
    auto start = idx * numElts;
    std::iota(indices.begin(), indices.end(), start);
    auto attr = rewriter.getDenseI32ArrayAttr(indices);
    rewriter.replaceOpWithNewOp<LLVM::ShuffleVectorOp>(op, dstType, base, base,
                                                       attr);
    return success();
  }
};

// fixme: support it in gputogenx
class GPUSubgroupIdOpLowering
    : public ConvertTritonGPUOpToLLVMPattern<mlir::gpu::SubgroupIdOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      mlir::gpu::SubgroupIdOp>::ConvertTritonGPUOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(mlir::gpu::SubgroupIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto i32Type = rewriter.getI32Type();
    Value threadX =
        rewriter.create<mlir::gpu::ThreadIdOp>(loc, mlir::gpu::Dimension::x);
    Value threadY =
        rewriter.create<mlir::gpu::ThreadIdOp>(loc, mlir::gpu::Dimension::y);
    Value threadZ =
        rewriter.create<mlir::gpu::ThreadIdOp>(loc, mlir::gpu::Dimension::z);
    Value blockX =
        rewriter.create<mlir::gpu::BlockDimOp>(loc, mlir::gpu::Dimension::x);
    Value blockY =
        rewriter.create<mlir::gpu::BlockDimOp>(loc, mlir::gpu::Dimension::y);
    Value llid = rewriter.create<arith::MulIOp>(loc, threadZ, blockY);
    llid = rewriter.create<arith::AddIOp>(loc, llid, threadY);
    llid = rewriter.create<arith::MulIOp>(loc, llid, blockX);
    llid = rewriter.create<arith::AddIOp>(loc, llid, threadX);
    // fixme: replace cst16 with subgroupSize
    Value cst16 = rewriter.create<arith::ConstantOp>(
        loc, i32Type, rewriter.getIntegerAttr(i32Type, 16));
    Value subgroupId = rewriter.create<arith::DivUIOp>(loc, llid, cst16);
    rewriter.replaceOp(op, subgroupId);
    return success();
  }
};

// fixme: support it in upstream constantOpLowering
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
  patterns.add<CastOpConversion>(typeConverter, benefit);
  patterns.add<GPUSubgroupIdOpLowering>(typeConverter, benefit);
  patterns.add<ArithConstantOpLowering>(typeConverter, benefit);
}
