#include "PatternTritonGPUOpToLLVM.h"

#include "intel/include/Dialect/TritonGEN/IR/TritonGENDialect.h"

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/ArrayRef.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu::intel;

namespace {

// Note: this macro is used to explicitly invoke the verifier because
// `triton_gen` ops are immediately lowered further to a builtin call.
#define VERIFY_OPERATION(op)                                                   \
  if (failed(op.verify()))                                                     \
    return failure();

VectorType getVectorType(RankedTensorType tensorType, Type elemType) {
  // Determine a vector type of the given `elemType` that covers 1/16 of
  // `tensorType`, i.e. the amout of data a single subgroup lane will work on.
  size_t tensorSize =
      tensorType.getNumElements() * tensorType.getElementTypeBitWidth();
  size_t num = (tensorSize / 16) / elemType.getIntOrFloatBitWidth();
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
    assert(tensorType.getRank() == 2 &&
           "only support 2d load/store/prefetch for now");

    Type elemType = tensorType.getElementType();
    unsigned dataSize = elemType.getIntOrFloatBitWidth();
    unsigned blockHeight = tensorType.getShape()[0];
    unsigned blockWidth = tensorType.getShape()[1];
    assert((blockWidth == 8 || blockWidth == 16 || blockWidth == 32 ||
            blockWidth == 64) &&
           "only support 8/16/32/64 block");
    auto idxAttr = op->template getAttrOfType<mlir::IntegerAttr>("DotIdx");
    unsigned vBlks = 1;
    if (dataSize == 16) {
      vBlks = ceil(blockWidth, 16U);
      blockWidth = 16;
    } else if (dataSize == 8 && idxAttr) {
      unsigned blockWidthUnit = idxAttr.getInt() == 0 ? 32 : 16;
      vBlks = ceil(blockWidth, blockWidthUnit);
      blockWidth = blockWidthUnit;
    }
    assert((vBlks == 1 || vBlks == 2) && "only support 1 or 2 blocks");

    Value ptr = op.getPtr();
    if (auto cast =
            dyn_cast<mlir::UnrealizedConversionCastOp>(ptr.getDefiningOp()))
      ptr = cast.getInputs()[0];

    MakeTensorPtrOp ptrOp = getMakeTensorPtrOp(ptr);
    Value base = ptrOp.getBase();
    if (auto cast =
            dyn_cast<mlir::UnrealizedConversionCastOp>(base.getDefiningOp()))
      base = cast.getInputs()[0];
    else
      base = rewriter.getRemappedValue(base);

    OpBuilder::InsertPoint insertPoint = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointAfter(ptrOp);
    Location loc = op.getLoc();
    bool transpose = ptrOp.getOrder()[0] == 0;
    Value bytes =
        i32_val(tensorType.getElementType().getIntOrFloatBitWidth() / 8);

    auto calculateSurface = [&](Value shape, bool multiplyBytes) {
      Value truncatedShape = trunc(i32_ty, shape);
      if (multiplyBytes)
        truncatedShape = mul(truncatedShape, bytes);
      return truncatedShape;
    };

    Value surfaceW = calculateSurface(ptrOp.getShape()[!transpose], true);
    Value surfaceH = calculateSurface(ptrOp.getShape()[transpose], false);
    Value surfaceP = calculateSurface(ptrOp.getStrides()[transpose], true);
    rewriter.restoreInsertionPoint(insertPoint);

    Value tensorPtr = adaptor.getPtr();
    Value offsetX = extract_element(tensorPtr, i32_val(0));
    Value offsetY = extract_element(tensorPtr, i32_val(1));

    if constexpr (std::is_same_v<OpType, LoadOp>) {
      assert(idxAttr && "Dot index attribute missing");
      unsigned idx = idxAttr.getInt();
      Type resType =
          this->getTypeConverter()->convertType(op->getResult(0).getType());
      bool isDword = idx == 1 || elemType == f32_ty;
      Type vectorType =
          getVectorType(cast<RankedTensorType>(op.getResult().getType()),
                        isDword ? i32_ty : i16_ty);
      bool vnni = (idx == 1) && dataSize < 32;

      // FIXME: only support fp16/bf16 transpose for now, add more support like
      // tf32 and fp8.
      if (transpose) {
        assert(getElementBitWidth(tensorType) == 16 &&
               "only support 16-bit element type for now");
        vnni = false;
        dataSize = 32;
        blockWidth /= 2;
        Value tmp = offsetX;
        offsetX = lshr(offsetY, i32_val(1));
        offsetY = tmp;
      }
      auto load = rewriter.create<TritonGEN::Matrix2DBlockLoadOp>(
          loc, vectorType, base, surfaceW, surfaceH, surfaceP, offsetX, offsetY,
          dataSize, blockWidth, blockHeight, vBlks, false /*transpose*/, vnni);
      VERIFY_OPERATION(load)

      rewriter.replaceOp(op, bitcast(load, resType));
    } else if constexpr (std::is_same_v<OpType, PrefetchOp>) {
      if (transpose)
        std::swap(offsetX, offsetY);
      auto newOp = rewriter.create<TritonGEN::Matrix2DBlockPrefetchOp>(
          loc, base, surfaceW, surfaceH, surfaceP, offsetX, offsetY, dataSize,
          blockWidth, blockHeight, vBlks, TritonGEN::LoadCacheControl::L1C_L3C);
      VERIFY_OPERATION(newOp);

      rewriter.eraseOp(op);
    } else {
      VectorType vectorType =
          getVectorType(cast<RankedTensorType>(op.getValue().getType()),
                        rewriter.getIntegerType(dataSize));
      auto newOp = rewriter.create<TritonGEN::Matrix2DBlockStoreOp>(
          loc, base, surfaceW, surfaceH, surfaceP, offsetX, offsetY, dataSize,
          blockWidth, blockHeight, vBlks,
          bitcast(adaptor.getValue(), vectorType));
      VERIFY_OPERATION(newOp);

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
    auto encodePrecision =
        [&](Type type, InputPrecisionAttr attr) -> TritonGEN::PrecisionType {
      if (type == bf16_ty)
        return TritonGEN::PrecisionType::BF16;
      else if (type == f16_ty)
        return TritonGEN::PrecisionType::FP16;
      else if (type == f32_ty && attr &&
               attr.getValue() == InputPrecision::TF32)
        return TritonGEN::PrecisionType::TF32;
      else if (type.isInteger(8)) {
        if (type.isUnsignedInteger())
          return TritonGEN::PrecisionType::U8;
        return TritonGEN::PrecisionType::S8;
      }

      llvm_unreachable("add more support for PrecisionType");
      return TritonGEN::PrecisionType::UNUSED;
    };

    TritonGEN::PrecisionType precATy = encodePrecision(
        op.getA().getType().getElementType(), op.getInputPrecisionAttr());
    TritonGEN::PrecisionType precBTy = encodePrecision(
        op.getB().getType().getElementType(), op.getInputPrecisionAttr());
    auto precA =
        TritonGEN::PrecisionTypeAttr::get(rewriter.getContext(), precATy);
    auto precB =
        TritonGEN::PrecisionTypeAttr::get(rewriter.getContext(), precBTy);

    Location loc = op.getLoc();
    Type typeA = getVectorType(
        cast<RankedTensorType>(op.getA().getType()),
        precATy == TritonGEN::PrecisionType::TF32 ? i32_ty : i16_ty);
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
    default: {
      unsigned num = operands.size();
      Value undef = undef(dstType);
      for (unsigned i = 0; i < num; ++i)
        undef =
            insert_element(dstType, undef, operands[i],
                           rewriter.create<LLVM::ConstantOp>(loc, i32_ty, i));

      rewriter.replaceOp(op, undef);
    }
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
    unsigned idx = adaptor.getIndex();
    Type dstType = getTypeConverter()->convertType(op.getType());
    Value result;
    if (auto vecTy = dyn_cast<VectorType>(dstType)) {
      unsigned numElts = vecTy.getNumElements();
      SmallVector<int32_t> indices(numElts);
      unsigned start = idx * numElts;
      std::iota(indices.begin(), indices.end(), start);
      result = rewriter.create<LLVM::ShuffleVectorOp>(
          loc, vecTy, base, base, rewriter.getDenseI32ArrayAttr(indices));
    } else {
      Value idxVal = rewriter.create<LLVM::ConstantOp>(loc, i32_ty, idx);
      result = extract_element(base, idxVal);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

class SplatOpConversion : public ConvertTritonGPUOpToLLVMPattern<SplatOp> {
public:
  using ConvertTritonGPUOpToLLVMPattern<
      SplatOp>::ConvertTritonGPUOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(SplatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    RankedTensorType resultType = op.getType();
    TritonIntelGPUToLLVMTypeConverter *typeConverter = getTypeConverter();
    Type srcTy = adaptor.getSrc().getType();
    VectorType vecTy = VectorType::get(1, srcTy);
    auto poison = rewriter.create<LLVM::PoisonOp>(loc, vecTy);
    auto splat =
        insert_element(vecTy, poison, adaptor.getSrc(),
                       rewriter.create<LLVM::ConstantOp>(loc, i32_ty, 0));
    Type convertedTy = typeConverter->convertType(resultType);
    int64_t num = cast<VectorType>(convertedTy).getNumElements();
    SmallVector<int32_t> indices(num, 0);
    Value result = rewriter.create<LLVM::ShuffleVectorOp>(
        loc, convertedTy, splat, poison,
        rewriter.getDenseI32ArrayAttr(indices));
    rewriter.replaceOp(op, result);
    return success();
  }
};

class ReduceOpConversion : public ConvertTritonGPUOpToLLVMPattern<ReduceOp> {
public:
  using ConvertTritonGPUOpToLLVMPattern<
      ReduceOp>::ConvertTritonGPUOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    int subgroupSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    int axis = op.getAxis();
    ArrayRef<int64_t> shape =
        cast<RankedTensorType>(op.getInputTypes()[0]).getShape();
    assert(shape[axis] <= subgroupSize &&
           "Reduce size should be split into subgroups");

    Location loc = op.getLoc();
    Type resultType = op.getType(0);
    TritonIntelGPUToLLVMTypeConverter *typeConverter = getTypeConverter();
    Type convertedTy = typeConverter->convertType(resultType);
    Region &combineOp = op.getCombineOp();
    if (!combineOp.hasOneBlock() ||
        combineOp.front().getOperations().size() != 2)
      return failure();

    Operation *combine = &*combineOp.front().getOperations().begin();

    // FIXME: support all possible reduction modes
    using AllReduceOperation = mlir::gpu::AllReduceOperation;
    AllReduceOperation redKind;
    if (isa<arith::AddFOp>(combine))
      redKind = AllReduceOperation::ADD;
    else if (isa<arith::MaxNumFOp>(combine))
      redKind = AllReduceOperation::MAXNUMF;
    else
      llvm_unreachable("Unhandled reduction kind");

    Value result = rewriter.create<mlir::gpu::SubgroupReduceOp>(
        loc, convertedTy, adaptor.getSrcs()[0], redKind, true);
    rewriter.replaceOp(op, result);
    return success();
  }
};

class ExpandDimsOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<ExpandDimsOp> {
public:
  using ConvertTritonGPUOpToLLVMPattern<
      ExpandDimsOp>::ConvertTritonGPUOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(ExpandDimsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getSrc());
    return success();
  }
};

class BroadcastOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::BroadcastOp> {
public:
  using ConvertTritonGPUOpToLLVMPattern<
      triton::BroadcastOp>::ConvertTritonGPUOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(triton::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getSrc());
    return success();
  }
};

class AddPtrOpConversion : public ConvertTritonGPUOpToLLVMPattern<AddPtrOp> {
public:
  using ConvertTritonGPUOpToLLVMPattern<
      AddPtrOp>::ConvertTritonGPUOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type resultType = op.getType();
    LLVMTypeConverter *typeConverter = getTypeConverter();
    Type resultPtrTy = typeConverter->convertType(resultType);
    Type resultElmTy = typeConverter->convertType(
        cast<PointerType>(resultType).getPointeeType());
    Value result =
        gep(resultPtrTy, resultElmTy, adaptor.getPtr(), adaptor.getOffset());
    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void mlir::triton::intel::populateTritonOpsToLLVMPatterns(
    TritonIntelGPUToLLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<AddPtrOpConversion>(typeConverter, benefit);
  patterns.add<AdvanceOpConversion>(typeConverter, benefit);
  patterns.add<BroadcastOpConversion>(typeConverter, benefit);
  patterns.add<DotOpConversion>(typeConverter, benefit);
  patterns.add<ExpandDimsOpConversion>(typeConverter, benefit);
  patterns.add<ExtractOpConversion>(typeConverter, benefit);
  patterns.add<GlueOpConversion>(typeConverter, benefit);
  patterns.add<LoadStorePrefetchOpConversion<PrefetchOp>>(typeConverter,
                                                          benefit);
  patterns.add<LoadStorePrefetchOpConversion<LoadOp>>(typeConverter, benefit);
  patterns.add<LoadStorePrefetchOpConversion<StoreOp>>(typeConverter, benefit);
  patterns.add<MakeTensorPtrOpConversion>(typeConverter, benefit);
  patterns.add<ReduceOpConversion>(typeConverter, benefit);
  patterns.add<SplatOpConversion>(typeConverter, benefit);
}
