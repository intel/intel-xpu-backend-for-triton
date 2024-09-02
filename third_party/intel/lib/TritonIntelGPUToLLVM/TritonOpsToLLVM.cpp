#include "PatternTritonGPUOpToLLVM.h"

#include "intel/include/Dialect/TritonGEN/IR/TritonGENDialect.h"
#include "intel/include/TritonGENToLLVM/GenIntrinsics.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/LLVMIR/TypeToLLVM.h"

#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"

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
    auto addrSpace = ptrType.getAddressSpace();
    const bool isLocalSpace = (addrSpace == 3);
    auto moduleOp =
        rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
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
    else
      base = rewriter.getRemappedValue(base);

    OpBuilder::InsertPoint insertPoint = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointAfter(ptrOp);
    Location loc = op.getLoc();
    bool transpose = ptrOp.getOrder()[0] == 0;
    Value bytes =
        i32_val(tensorType.getElementType().getIntOrFloatBitWidth() / 8);
    if (isLocalSpace) {
      Value threadId = getThreadId(rewriter, loc);
      Value laneId = urem(threadId, i32_val(16));

      if constexpr (std::is_same_v<OpType, LoadOp>) {

        auto oriResTy = VectorType::get(1, i64_ty);
        auto v4i16Ty = VectorType::get(4, i16_ty);
        auto v8i16Ty = VectorType::get(8, i16_ty);
        auto vecf16ResTy = VectorType::get(8, f16_ty);
        Value llPtr = adaptor.getPtr();

        auto iPtrTy = LLVM::LLVMPointerType::get(rewriter.getContext(), 3);
        std::string funcName = llvm::GenISAIntrinsic::getName(
            llvm::GenISAIntrinsic::GenISA_LSCLoad);

        SmallVector<Type> argTypes{iPtrTy, i32_ty, i32_ty, i32_ty, i32_ty};
        LLVM::LLVMFuncOp funcOp =
            LLVM::lookupOrCreateFn(moduleOp, funcName, argTypes, oriResTy);
        funcOp.setCConv(LLVM::cconv::CConv::SPIR_FUNC);
        rewriter.restoreInsertionPoint(insertPoint);
        Value offsetX = extract_element(llPtr, i32_val(0));
        Value offsetY = extract_element(llPtr, i32_val(1));

        Value blkId = add(mul(udiv(offsetY, i32_val(8)), i32_val(4)),
                          udiv(offsetX, i32_val(16)));

        Value offset = i32_val(0);

        Value index = mul(blkId, i32_val(128));

        base = gep(ptr_ty(rewriter.getContext(), 3), i16_ty, base, index);

        // laneoffset
        base = gep(ptr_ty(rewriter.getContext(), 3), i64_ty, base, laneId);

        // LSC_DATA_SIZE_64b in visa_igc_common_header.h
        Value dataSize = i32_val(4);
        // vectorSize = 1 LSC_DATA_ELEMS_1 in visa_igc_common_header.h
        Value vSize = i32_val(1);
        // LSC_L1DEF_L3DEF
        Value cacheOpt = i32_val(0);
        SmallVector<Value> args{base, offset, dataSize, vSize, cacheOpt};

        auto localLoadHead = rewriter.create<LLVM::CallOp>(loc, funcOp, args);
        base = gep(ptr_ty(rewriter.getContext(), 3), i64_ty, base, i32_val(16));

        SmallVector<Value> tailArgs{base, offset, dataSize, vSize, cacheOpt};

        auto localLoadTail =
            rewriter.create<LLVM::CallOp>(loc, funcOp, tailArgs);

        SmallVector<int32_t> indices(8);
        std::iota(indices.begin(), indices.end(), 0);
        DenseI32ArrayAttr attr = rewriter.getDenseI32ArrayAttr(indices);

        auto concatVec = rewriter.create<LLVM::ShuffleVectorOp>(
            loc, v8i16Ty, bitcast(localLoadHead.getResult(), v4i16Ty),
            bitcast(localLoadTail.getResult(), v4i16Ty), attr);
        rewriter.replaceOp(op, bitcast(concatVec, vecf16ResTy));

        return success();
      }
      if constexpr (std::is_same_v<OpType, StoreOp>) {
        auto voidTy = LLVM::LLVMVoidType::get(rewriter.getContext());

        llvm::LLVMContext llvmContext;
        LLVM::TypeToLLVMIRTranslator typeTranslator(llvmContext);
        auto v2Ty = VectorType::get(2, i64_ty);
        auto v1Ty = VectorType::get(1, i64_ty);
        auto iPtrTy = LLVM::LLVMPointerType::get(rewriter.getContext(), 3);
        Value llPtr = adaptor.getPtr();

        std::string funcName = llvm::GenISAIntrinsic::getName(
            llvm::GenISAIntrinsic::GenISA_LSCStore);
        rewriter.restoreInsertionPoint(insertPoint);
        Value val = adaptor.getValue();

        SmallVector<Type> argTypes{iPtrTy, i32_ty, i64_ty,
                                   i32_ty, i32_ty, i32_ty};
        LLVM::LLVMFuncOp funcOp =
            LLVM::lookupOrCreateFn(moduleOp, funcName, argTypes, voidTy);
        funcOp.setCConv(LLVM::cconv::CConv::SPIR_FUNC);
        Value offsetX = extract_element(llPtr, i32_val(0));
        Value offsetY = extract_element(llPtr, i32_val(1));
        Value offset = i32_val(0);
        // 16x64xhf = 2x8x16xhf
        Value blkId = add(mul(udiv(offsetY, i32_val(8)), i32_val(4)),
                          udiv(offsetX, i32_val(16)));

        Value index = mul(blkId, i32_val(128));

        base = gep(ptr_ty(rewriter.getContext(), 3), i16_ty, base, index);
        // laneoffset
        base = gep(ptr_ty(rewriter.getContext(), 3), i64_ty, base, laneId);

        // LSC_DATA_SIZE_64b in visa_igc_common_header.h
        Value dataSize = i32_val(4);
        // vectorSize = 1 LSC_DATA_ELEMS_1 in visa_igc_common_header.h
        Value vSize = i32_val(1);

        // LSC_L1DEF_L3DEF
        Value cacheOpt = i32_val(0);
        Value v2Val = bitcast(val, v2Ty);
        val = extract_element(v2Val, i32_val(0));

        SmallVector<Value> args{base, offset, val, dataSize, vSize, cacheOpt};

        auto localStore = rewriter.create<LLVM::CallOp>(loc, funcOp, args);
        Value val1 = extract_element(v2Val, i32_val(1));

        // one i64
        // laneoffset
        // for second store, next line, 16(sg) x 8bytes
        base = gep(ptr_ty(rewriter.getContext(), 3), i64_ty, base, i32_val(16));

        SmallVector<Value> args1{base, offset, val1, dataSize, vSize, cacheOpt};
        auto localStore1 = rewriter.create<LLVM::CallOp>(loc, funcOp, args1);

        rewriter.eraseOp(op);
        return success();
      }
    } // isLocalSpace

    auto calculateSurface = [&](Value shape, bool multiplyBytes) {
      Value truncatedShape = trunc(i32_ty, shape);
      if (multiplyBytes)
        truncatedShape = mul(truncatedShape, bytes);
      return sub(truncatedShape, i32_val(1));
    };

    Value surfaceW = calculateSurface(
        transpose ? ptrOp.getShape()[0] : ptrOp.getShape()[1], true);
    Value surfaceH = calculateSurface(
        transpose ? ptrOp.getShape()[1] : ptrOp.getShape()[0], false);
    Value surfaceP = calculateSurface(
        transpose ? ptrOp.getStrides()[1] : ptrOp.getStrides()[0], true);
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
      // fixed f16 for now
      if (ptrOp.getOrder()[0] == 0) {
        transpose = true;
        vnni = false;
        dataSize = 32;
        blockWidth /= 2;
        // auto one = createIntConstant(i32Type, 1);
        Value tmp = offsetX;
        offsetX = rewriter.create<LLVM::LShrOp>(loc, offsetY, i32_val(1));
        offsetY = tmp;
      }
      auto load = rewriter.create<TritonGEN::Matrix2DBlockLoadOp>(
          loc, vectorType, base, surfaceW, surfaceH, surfaceP, offsetX, offsetY,
          dataSize, blockWidth, blockHeight, vBlks, transpose, vnni);
      rewriter.replaceOp(op, bitcast(load, resType));
    } else if constexpr (std::is_same_v<OpType, PrefetchOp>) {
      if (ptrOp.getOrder()[0] == 0) {
        // transpose = false;
        // vnni = false;
        Value tmp = offsetX;
        offsetX = offsetY;
        offsetY = tmp;
      }
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
      llvm_unreachable("add more support for PrecisionType");
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

class GlueOpConversion : public ConvertTritonGPUOpToLLVMPattern<GlueOp> {
public:
  using ConvertTritonGPUOpToLLVMPattern<
      GlueOp>::ConvertTritonGPUOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(GlueOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ValueRange operands = adaptor.getOperands();
    Value result = TypeSwitch<Type, Value>(operands.front().getType())
                       .Case([this, &rewriter, op, operands](VectorType) {
                         return vectorGlueOp(op, operands, rewriter);
                       })
                       .Default([this, &rewriter, op, operands](auto) {
                         return scalarGlueOp(op, operands, rewriter);
                       });
    if (!result) {
      // Non-power of 2 number of vector operands case. See comment below.
      return failure();
    }
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  Value vectorGlueOp(GlueOp op, ValueRange operands,
                     ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    if (!llvm::isPowerOf2_64(operands.size())) {
      // Legal vector widths: 2, 3, 4, 8, 16. We cannot obtain any of these from
      // a non-power of 2 number of vector operands (2 and 3 can only be created
      // from scalar values). Bail out to keep algorithm simple and avoid
      // illegal codegen.
      return {};
    }
    return treeVectorGlueOp(op.getLoc(), operands, rewriter);
  }

  Value treeVectorGlueOp(Location loc, ValueRange operands,
                         ConversionPatternRewriter &rewriter) const {
    if (operands.size() == 1)
      return operands.front();
    assert(llvm::isPowerOf2_64(operands.size()) &&
           "Expecting power of 2 number of operands");
    SmallVector<Value> lhs;
    SmallVector<Value> rhs;
    for (auto [index, value] : llvm::enumerate(operands))
      (index % 2 == 0 ? lhs : rhs).push_back(value);
    SmallVector<Value> res;
    int32_t numElements =
        cast<VectorType>(operands.front().getType()).getNumElements();
    SmallVector<int32_t> concatMask(numElements * 2);
    std::iota(std::begin(concatMask), std::end(concatMask), 0);
    llvm::transform(llvm::zip_equal(lhs, rhs), std::back_inserter(res),
                    [loc, &rewriter, &concatMask](const auto &pair) -> Value {
                      auto [lhs, rhs] = pair;
                      return rewriter.create<LLVM::ShuffleVectorOp>(
                          loc, lhs, rhs, concatMask);
                    });
    return treeVectorGlueOp(loc, res, rewriter);
  }

  Value scalarGlueOp(GlueOp op, ValueRange operands,
                     ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    auto dstType =
        cast<VectorType>(getTypeConverter()->convertType(op.getType()));
    Value poison = rewriter.create<LLVM::PoisonOp>(loc, dstType);

    auto enumeratedOperands = llvm::enumerate(operands);
    return std::accumulate(std::begin(enumeratedOperands),
                           std::end(enumeratedOperands), poison,
                           [&rewriter, loc](Value acc, const auto &pair) {
                             auto [index, operand] = pair;
                             Value idx = i32_val(index);
                             return insert_element(acc, operand, idx);
                           });
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
    auto dstType = getTypeConverter()->convertType(op.getType());
    Value result;
    if (auto vecTy = dyn_cast<VectorType>(dstType)) {
      unsigned numElts = vecTy.getNumElements();
      SmallVector<int32_t> indices(numElts);
      unsigned start = idx * numElts;
      std::iota(indices.begin(), indices.end(), start);
      DenseI32ArrayAttr attr = rewriter.getDenseI32ArrayAttr(indices);
      result =
          rewriter.create<LLVM::ShuffleVectorOp>(loc, vecTy, base, base, attr);
    } else {
      Type i32Ty = rewriter.getI32Type();
      Value idxVal = rewriter.create<LLVM::ConstantOp>(loc, i32Ty, idx);
      result = rewriter.create<LLVM::ExtractElementOp>(loc, base, idxVal);
    }
    rewriter.replaceOp(op, result);
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

class AddPtrOpConversion : public ConvertTritonGPUOpToLLVMPattern<AddPtrOp> {
public:
  using ConvertTritonGPUOpToLLVMPattern<
      AddPtrOp>::ConvertTritonGPUOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resultType = op.getType();
    assert(isa<PointerType>(resultType));
    auto typeConverter = getTypeConverter();
    auto resultPtrTy = typeConverter->convertType(resultType);
    auto resultElmTy = typeConverter->convertType(
        cast<PointerType>(resultType).getPointeeType());
    Value result = rewriter.create<LLVM::GEPOp>(
        loc, resultPtrTy, resultElmTy, adaptor.getPtr(), adaptor.getOffset());
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
    auto loc = op.getLoc();
    auto resultType = op.getType();
    auto typeConverter = getTypeConverter();
    auto srcTy = adaptor.getSrc().getType();
    auto vecTy = VectorType::get(1, srcTy);
    auto undef = rewriter.create<LLVM::UndefOp>(loc, vecTy);
    auto splat = rewriter.create<LLVM::InsertElementOp>(
        loc, vecTy, undef, adaptor.getSrc(),
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), 0));
    auto convertedTy = typeConverter->convertType(resultType);
    auto num = convertedTy.cast<VectorType>().getNumElements();
    SmallVector<int32_t> indices(num, 0);
    auto attr = rewriter.getDenseI32ArrayAttr(indices);
    Value result = rewriter.create<LLVM::ShuffleVectorOp>(loc, convertedTy,
                                                          splat, splat, attr);
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
    Location loc = op.getLoc();
    ValueRange srcs = adaptor.getSrcs();
    if (srcs.size() != 1)
      return failure();
    Value src = srcs.front();
    auto srcTy = dyn_cast<VectorType>(src.getType());
    if (!srcTy)
      return failure();
    SmallVector<Value> elements;
    for (int i = 0, size = srcTy.getNumElements(); i < size; ++i)
      elements.push_back(extract_element(src, i32_val(i)));
    // Help tree reduce.
    if (!llvm::isPowerOf2_64(elements.size()))
      return failure();
    Value res = treeReduce(op, rewriter, elements);
    rewriter.replaceOp(op, res);
    return success();
  }

private:
  static LLVM::LLVMFuncOp buildReduceFunc(ReduceOp op,
                                          ConversionPatternRewriter &rewriter) {
    Location loc = op.getLoc();

    std::string name = llvm::formatv("__reduceOp_{0}", static_cast<void *>(op));
    Block *combineBlock = &op.getCombineOp().front();
    Type elementTy = combineBlock->getArgument(0).getType();

    auto type = rewriter.getType<LLVM::LLVMFunctionType>(
        elementTy, ArrayRef<Type>{elementTy, elementTy}, /*isVarArg=*/false);

    OpBuilder::InsertionGuard ig(rewriter);
    auto modOp = op->getParentOfType<ModuleOp>();
    rewriter.setInsertionPointToStart(modOp.getBody());

    auto funcOp = rewriter.create<LLVM::LLVMFuncOp>(loc, name, type);
    funcOp.setCConv(LLVM::cconv::CConv::SPIR_FUNC);

    Block *entryBlock = funcOp.addEntryBlock(rewriter);
    rewriter.mergeBlocks(combineBlock, entryBlock, entryBlock->getArguments());

    {
      OpBuilder::InsertionGuard ig(rewriter);
      rewriter.setInsertionPointToEnd(entryBlock);
      auto terminatorOp = cast<ReduceReturnOp>(entryBlock->getTerminator());
      rewriter.create<LLVM::ReturnOp>(loc, terminatorOp.getResult());
      rewriter.eraseOp(terminatorOp);
    }

    return funcOp;
  }

  static Value treeReduce(ReduceOp op, ConversionPatternRewriter &rewriter,
                          ArrayRef<Value> values) {
    LLVM::LLVMFuncOp reduceFunc = buildReduceFunc(op, rewriter);
    SmallVector<Value> res = treeReduce(op, reduceFunc, rewriter, values);
    assert(res.size() == 1 && "Expecting single result");
    return res.front();
  }

  static SmallVector<Value> treeReduce(ReduceOp op, LLVM::LLVMFuncOp reduceFunc,
                                       ConversionPatternRewriter &rewriter,
                                       ArrayRef<Value> values) {
    if (values.size() == 1)
      return {values.front()};
    SmallVector<Value> lhs;
    SmallVector<Value> rhs;
    for (auto [index, value] : llvm::enumerate(values))
      (index % 2 == 0 ? lhs : rhs).push_back(value);
    SmallVector<Value> res;
    Location loc = op.getLoc();
    llvm::transform(llvm::zip_equal(lhs, rhs), std::back_inserter(res),
                    [&](auto pair) -> Value {
                      auto [lhs, rhs] = pair;
                      auto callOp = rewriter.create<LLVM::CallOp>(
                          loc, reduceFunc, ValueRange{lhs, rhs});
                      callOp.setCConv(LLVM::cconv::CConv::SPIR_FUNC);
                      return callOp->getResult(0);
                    });
    return treeReduce(op, reduceFunc, rewriter, res);
  }
};

class SubGroupTransposeOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<SubGroupTranspose> {
public:
  using ConvertTritonGPUOpToLLVMPattern<
      SubGroupTranspose>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SubGroupTranspose op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (op.getType().getRank() != 2)
      return failure();
    Value src = adaptor.getSrc();
    auto vecTy = dyn_cast<VectorType>(src.getType());
    auto mod = op->getParentOfType<ModuleOp>();
    int threadsPerWarp =
        mlir::triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    if (!vecTy || vecTy.getNumElements() != threadsPerWarp)
      return failure();
    Location loc = op.getLoc();
    Value localBuffer = adaptor.getLocalBuffer();
    Type offsetType = getTypeConverter()->getIndexType();
    Value subGroupId = getValueOrCreateCastToIndexLike(
        rewriter, loc, offsetType,
        rewriter.create<mlir::gpu::SubgroupIdOp>(loc));
    Value subGroupLocalId = getValueOrCreateCastToIndexLike(
        rewriter, loc, offsetType, rewriter.create<mlir::gpu::LaneIdOp>(loc));
    Value wiStride =
        rewriter.create<LLVM::ConstantOp>(loc, offsetType, threadsPerWarp);
    Value sgStride = rewriter.create<LLVM::ConstantOp>(
        loc, offsetType, threadsPerWarp * threadsPerWarp);
    Value subGroupOffset = mul(sgStride, subGroupId);
    Type ptrType = localBuffer.getType();
    Type elementType =
        cast<RankedTensorType>(op.getSrc().getType()).getElementType();
    Value subGroupBasePtr = gep(ptrType, elementType, localBuffer,
                                ValueRange{subGroupOffset}, /*inbounds=*/true);

    // Store matrix in local memory.
    rewriter.create<TritonGEN::SIMDBlockWriteOp>(loc, subGroupBasePtr, src);

    // Load from matrix, trasposed.
    Value workItemOffset = mul(wiStride, subGroupLocalId);
    Value workItemBasePtr = gep(ptrType, elementType, subGroupBasePtr,
                                ValueRange{workItemOffset}, /*inbounds=*/true);
    rewriter.replaceOp(op, load(src.getType(), workItemBasePtr));
    return success();
  }
};

class ConvertLayoutOpConversion : public ConvertTritonGPUOpToLLVMPattern<
                                      mlir::triton::gpu::ConvertLayoutOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      mlir::triton::gpu::ConvertLayoutOp>::ConvertTritonGPUOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(mlir::triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value src = adaptor.getSrc();
    Type type = getTypeConverter()->convertType(op.getType());
    if (!type)
      return failure();
    Value res = rewriter.create<LLVM::PoisonOp>(loc, type);
    Type idxType = rewriter.getI32Type();
    constexpr int size = 16;
    constexpr StringLiteral fnName = "_Z17sub_group_shufflefj";
    auto moduleOp = op->getParentOfType<ModuleOp>();
    assert(moduleOp);
    LLVM::LLVMFuncOp func = LLVM::lookupOrCreateFn(
        moduleOp, fnName, {rewriter.getF32Type(), idxType},
        rewriter.getF32Type());
    for (int i = 0; i < size; ++i) {
      Value idx = rewriter.create<LLVM::ConstantOp>(loc, idxType, i);
      Value element =
          rewriter.create<LLVM::CallOp>(loc, func, ArrayRef<Value>{src, idx})
              ->getResult(0);
      res = rewriter.create<LLVM::InsertElementOp>(loc, res, element, idx);
    }
    rewriter.replaceOp(op, res);
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
    : public ConvertTritonGPUOpToLLVMPattern<BroadcastOp> {
public:
  using ConvertTritonGPUOpToLLVMPattern<
      BroadcastOp>::ConvertTritonGPUOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // keep it simple for now
    auto src = adaptor.getSrc();
    assert(src.getType() == getTypeConverter()->convertType(op.getType()));
    rewriter.replaceOp(op, src);
    return success();
  }
};

class ArithDivFOpLowering
    : public ConvertTritonGPUOpToLLVMPattern<mlir::arith::DivFOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      mlir::arith::DivFOp>::ConvertTritonGPUOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(mlir::arith::DivFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto srcType = dyn_cast<ShapedType>(op.getType());
    Type dstType = getTypeConverter()->convertType(srcType);
    auto vecType = cast<VectorType>(dstType);
    auto attr = rewriter.getFloatAttr(vecType.getElementType(), 1.0);
    auto dstAttr = DenseElementsAttr::get(vecType, attr.getValue());
    auto one = rewriter.create<LLVM::ConstantOp>(loc, dstType, dstAttr);
    auto rcp = rewriter.create<LLVM::FDivOp>(
        loc, dstType, one, adaptor.getRhs(),
        LLVM::FastmathFlagsAttr::get(rewriter.getContext(),
                                     LLVM::FastmathFlags::fast));
    auto res = rewriter.create<LLVM::FMulOp>(
        loc, dstType, adaptor.getLhs(), rcp,
        LLVM::FastmathFlagsAttr::get(rewriter.getContext(),
                                     LLVM::FastmathFlags::fast));

    rewriter.replaceOp(op, res);
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
  patterns.add<ArithDivFOpLowering>(typeConverter, benefit);
  patterns.add<AddPtrOpConversion>(typeConverter, benefit);
  patterns.add<SplatOpConversion>(typeConverter, benefit);
  patterns.add<ReduceOpConversion>(typeConverter, benefit);
  patterns.add<SubGroupTransposeOpConversion>(typeConverter, benefit);
  patterns.add<ExpandDimsOpConversion>(typeConverter, benefit);
  patterns.add<ConvertLayoutOpConversion>(typeConverter, benefit);
  patterns.add<BroadcastOpConversion>(typeConverter, benefit);
}
