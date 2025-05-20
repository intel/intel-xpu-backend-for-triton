#include "Dialect/TritonIntelGPU/IR/Utils.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "SPIRVSubgroupOps.h"

#include "intel/include/Dialect/TritonGEN/IR/TritonGENDialect.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;
using namespace mlir::triton;
namespace ttgi = mlir::triton::gpu::intel;

namespace {

// Note: this macro is used to explicitly invoke the verifier because
// `triton_gen` ops are immediately lowered further to a builtin call.
#define VERIFY_OPERATION(op)                                                   \
  if (failed(op.verify()))                                                     \
    return failure();

VectorType getVectorType(RankedTensorType tensorType, Type elemType) {
  // Determine a vector type of the given `elemType` that covers 1/16 of
  // `tensorType`, i.e. the amount of data a single subgroup lane will work on.
  size_t tensorSize =
      tensorType.getNumElements() * tensorType.getElementTypeBitWidth();
  size_t num = (tensorSize / 16) / elemType.getIntOrFloatBitWidth();
  return vec_ty(elemType, num);
};

MakeTensorPtrOp getMakePtrOp(Value v) {
  using BranchOps = llvm::SetVector<std::pair<Operation *, int>>;
  llvm::DenseMap<Block *, BranchOps> blockToBrOps;
  auto moduleOp =
      v.getParentBlock()->getParentOp()->getParentOfType<ModuleOp>();

  moduleOp.walk([&](Operation *op) {
    if (auto br = dyn_cast<LLVM::BrOp>(op)) {
      Block *block = br.getDest();
      blockToBrOps[block].insert({op, -1});
    }
    if (auto condBr = dyn_cast<LLVM::CondBrOp>(op)) {
      Block *blockT = condBr.getTrueDest();
      Block *blockF = condBr.getFalseDest();
      blockToBrOps[blockT].insert({condBr, 1});
      blockToBrOps[blockF].insert({condBr, 0});
    }
  });

  if (Operation *def = v.getDefiningOp()) {
    if (auto cast = dyn_cast<mlir::UnrealizedConversionCastOp>(def))
      def = cast.getInputs()[0].getDefiningOp();
    if (auto make = dyn_cast<MakeTensorPtrOp>(def))
      return make;
    if (auto advanceOp = dyn_cast<AdvanceOp>(def))
      return getMakePtrOp(advanceOp.getPtr());
    llvm_unreachable("Unable to getMakePtr()");
  }

  // If there is no defining op, v must be a BlockArgument.
  BlockArgument arg = cast<BlockArgument>(v);
  unsigned argNum = arg.getArgNumber();
  Operation *argOwner = arg.getOwner()->getParentOp();

  if (auto funcOp = dyn_cast<FunctionOpInterface>(argOwner)) {
    Block *block = arg.getOwner();
    auto [op, tOrF] = blockToBrOps[block][0];
    if (auto br = dyn_cast<LLVM::BrOp>(op))
      return getMakePtrOp(br.getDestOperands()[argNum]);
    if (auto condBr = dyn_cast<LLVM::CondBrOp>(op))
      return getMakePtrOp(tOrF ? condBr.getTrueDestOperands()[argNum]
                               : condBr.getFalseDestOperands()[argNum]);
    return getMakePtrOp(argOwner->getOperand(argNum));
  }
  llvm_unreachable("Unable to getMakePtr()");
}

static void decomposeBlockStore(ConversionPatternRewriter &rewriter,
                                Location loc, Value base, Value val,
                                VectorType vecTy, unsigned subGroupSize) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  constexpr unsigned maxBlockStoreWidth = 8;
  VectorType decomposedVecTy =
      VectorType::get(maxBlockStoreWidth, vecTy.getElementType());
  Value offset = b.i32_val(subGroupSize);
  for (int i = 0; i < vecTy.getNumElements() / maxBlockStoreWidth; ++i) {
    rewriter.create<TritonGEN::SubGroupBlockWriteOp>(
        loc, base,
        rewriter
            .create<triton::gpu::intel::ExtractOp>(loc, decomposedVecTy, val, i)
            .getRes());
    base = b.gep(base.getType(), decomposedVecTy, base, offset);
  }
}

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
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    VectorType v2i32 = vec_ty(i32_ty, 2);
    Value offsetX = op.getOffsets()[1];
    Value offsetY = op.getOffsets()[0];
    Value payLoad = b.undef(v2i32);
    payLoad = b.insert_element(payLoad, offsetX, b.i32_val(0));
    payLoad = b.insert_element(payLoad, offsetY, b.i32_val(1));
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
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    ValueRange offsets = adaptor.getOffsets();
    Value ptr = adaptor.getPtr();

    for (size_t i = 0; i < offsets.size(); ++i) {
      Value offset = offsets[i];
      if (auto cst = offset.getDefiningOp<LLVM::ConstantOp>())
        if (auto attr = dyn_cast<mlir::IntegerAttr>(cst.getValue());
            attr && attr.getInt() == 0)
          continue;

      Value idx = b.i32_val(!i);
      Value oldOffset = b.extract_element(ptr, idx);
      Value newOffset = b.add(i32_ty, oldOffset, offset);
      ptr = b.insert_element(ptr, newOffset, idx);
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
template <typename OpType,
          typename = std::enable_if_t<llvm::is_one_of<OpType, ttgi::PrefetchOp,
                                                      LoadOp, StoreOp>::value>>
class LoadStorePrefetchOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<OpType> {
public:
  using ConvertTritonGPUOpToLLVMPattern<
      OpType>::ConvertTritonGPUOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(OpType op, typename OpType::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ptrType = cast<PointerType>(op.getPtr().getType());
    // scalar load/store
    if (!isa<RankedTensorType>(ptrType.getPointeeType())) {
      if constexpr (std::is_same_v<OpType, LoadOp>) {
        auto newLoad = rewriter.create<LLVM::LoadOp>(op.getLoc(), op.getType(),
                                                     adaptor.getPtr());
        rewriter.replaceOp(op, newLoad);
        return success();
      }
      assert(0 && "add more support");
      return failure();
    }
    // blocked load/store
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
    bool isLocalSpace = (ptrType.getAddressSpace() ==
                         TritonGEN::TritonGENMemorySpace::kWorkgroup);

    assert(isLocalSpace ||
           (vBlks == 1 || vBlks == 2) && "only support 1 or 2 blocks");

    Value ptr = op.getPtr();
    if (auto cast = ptr.getDefiningOp<mlir::UnrealizedConversionCastOp>())
      ptr = cast.getInputs()[0];

    MakeTensorPtrOp ptrOp = getMakePtrOp(ptr);
    Value base = ptrOp.getBase();
    if (auto cast = base.getDefiningOp<mlir::UnrealizedConversionCastOp>())
      base = cast.getInputs()[0];
    else
      base = rewriter.getRemappedValue(base);

    OpBuilder::InsertPoint insertPoint = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointAfter(ptrOp);
    if (isLocalSpace)
      return rewriteLocalSpace(op, base, insertPoint, adaptor, rewriter);

    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    bool transpose = ptrOp.getOrder()[0] == 0;
    Value bytes =
        b.i32_val(tensorType.getElementType().getIntOrFloatBitWidth() / 8);

    auto calculateSurface = [&](Value shape, bool multiplyBytes) {
      Value truncatedShape = b.trunc(i32_ty, shape);
      if (multiplyBytes)
        truncatedShape = b.mul(truncatedShape, bytes);
      return truncatedShape;
    };

    Value surfaceW = calculateSurface(ptrOp.getShape()[!transpose], true);
    Value surfaceH = calculateSurface(ptrOp.getShape()[transpose], false);
    Value surfaceP = calculateSurface(ptrOp.getStrides()[transpose], true);
    rewriter.restoreInsertionPoint(insertPoint);

    Value tensorPtr = adaptor.getPtr();
    Value offsetX = b.extract_element(tensorPtr, b.i32_val(0));
    Value offsetY = b.extract_element(tensorPtr, b.i32_val(1));

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
        offsetX = b.lshr(offsetY, b.i32_val(1));
        offsetY = tmp;
      }
      auto load = rewriter.create<TritonGEN::Matrix2DBlockLoadOp>(
          loc, vectorType, base, surfaceW, surfaceH, surfaceP, offsetX, offsetY,
          dataSize, blockWidth, blockHeight, vBlks, transpose, vnni);
      VERIFY_OPERATION(load)

      rewriter.replaceOp(op, b.bitcast(load, resType));
    } else if constexpr (std::is_same_v<OpType, ttgi::PrefetchOp>) {
      if (transpose)
        std::swap(offsetX, offsetY);
      auto newOp = rewriter.create<TritonGEN::Matrix2DBlockPrefetchOp>(
          loc, base, surfaceW, surfaceH, surfaceP, offsetX, offsetY, dataSize,
          blockWidth, blockHeight, vBlks, TritonGEN::LoadCacheControl::L1C_L3C);
      VERIFY_OPERATION(newOp)

      rewriter.eraseOp(op);
    } else {
      VectorType vectorType =
          getVectorType(cast<RankedTensorType>(op.getValue().getType()),
                        rewriter.getIntegerType(dataSize));
      auto newOp = rewriter.create<TritonGEN::Matrix2DBlockStoreOp>(
          loc, base, surfaceW, surfaceH, surfaceP, offsetX, offsetY, dataSize,
          blockWidth, blockHeight, vBlks,
          b.bitcast(adaptor.getValue(), vectorType));
      VERIFY_OPERATION(newOp)

      rewriter.eraseOp(op);
    }

    return success();
  }

private:
  LogicalResult rewriteLocalSpace(OpType op, Value base,
                                  OpBuilder::InsertPoint insertPoint,
                                  typename OpType::Adaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
    auto ptrType = cast<PointerType>(op.getPtr().getType());
    assert(ptrType.getAddressSpace() ==
               TritonGEN::TritonGENMemorySpace::kWorkgroup &&
           "expecting local space");
    auto elemType =
        cast<RankedTensorType>(ptrType.getPointeeType()).getElementType();

    MLIRContext *ctx = rewriter.getContext();
    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value llPtr = adaptor.getPtr();
    if (auto cast = llPtr.getDefiningOp<mlir::UnrealizedConversionCastOp>())
      llPtr = cast.getInputs()[0];

    // sg_size(16) x i64 = 64 x i16
    VectorType v64i16Ty = VectorType::get(64, i16_ty);
    LLVM::LLVMPointerType ptrToSharedMemTy =
        ptr_ty(ctx, ptrType.getAddressSpace());
    Value offsetX = b.extract_element(llPtr, b.i32_val(0));
    Value offsetY = b.extract_element(llPtr, b.i32_val(1));

    Value blkId = b.add(b.mul(b.udiv(offsetY, b.i32_val(8)), b.i32_val(4)),
                        b.udiv(offsetX, b.i32_val(16)));
    Value index = b.mul(blkId, b.i32_val(128));
    base = b.gep(ptrToSharedMemTy, i16_ty, base, index);

    if constexpr (std::is_same_v<OpType, LoadOp>) {
      rewriter.restoreInsertionPoint(insertPoint);

      constexpr unsigned maxBlockLoadi16Width = 8;
      VectorType decomposedVecTy =
          VectorType::get(maxBlockLoadi16Width, i16_ty);
      auto mod = op->template getParentOfType<mlir::ModuleOp>();
      Value offset =
          b.i32_val(triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod));
      SmallVector<Value> values;
      for (int i = 0; i < 64 / maxBlockLoadi16Width; ++i) {
        auto simdRead = rewriter.create<TritonGEN::SubGroupBlockReadOp>(
            loc, decomposedVecTy, base);
        values.push_back(simdRead.getRes());
        base = b.gep(ptrToSharedMemTy, decomposedVecTy, base, offset);
      }
      auto simdRead =
          rewriter.create<triton::gpu::intel::GlueOp>(loc, v64i16Ty, values);

      VectorType v64Ty = VectorType::get(64, elemType);
      rewriter.replaceOp(op, b.bitcast(simdRead.getRes(), v64Ty));

      return success();
    }

    if constexpr (std::is_same_v<OpType, StoreOp>) {
      rewriter.restoreInsertionPoint(insertPoint);
      Value val = adaptor.getValue();
      if (auto shuffleOp = val.getDefiningOp<LLVM::ShuffleVectorOp>())
        val = shuffleOp.getRes();
      if (isa<LLVM::LLVMStructType>(val.getType())) {
        SmallVector<Value> unpackedVal = unpackLLElements(loc, val, rewriter);
        val = packLLVector(loc, unpackedVal, rewriter);
      }
      val = b.bitcast(val, v64i16Ty);

      auto mod = op->template getParentOfType<mlir::ModuleOp>();
      decomposeBlockStore(
          rewriter, loc, base, val, v64i16Ty,
          triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod));

      rewriter.eraseOp(op);
      return success();
    }

    return failure();
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
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Type typeA = getVectorType(
        cast<RankedTensorType>(op.getA().getType()),
        precATy == TritonGEN::PrecisionType::TF32 ? i32_ty : i16_ty);
    Value castA = b.bitcast(adaptor.getA(), typeA);
    VectorType typeB =
        getVectorType(cast<RankedTensorType>(op.getB().getType()), i32_ty);
    Value castB = b.bitcast(adaptor.getB(), typeB);
    auto rc = IntegerAttr::get(i32_ty, 8);
    // sd dpasW fixed in genx.dpas lowering.
    rewriter.replaceOpWithNewOp<TritonGEN::MatrixDPASOp>(
        op, adaptor.getC().getType(), adaptor.getC(), castA, castB, precA,
        precB, rc);
    return success();
  }
};

class GlueOpConversion : public ConvertTritonGPUOpToLLVMPattern<ttgi::GlueOp> {
public:
  using ConvertTritonGPUOpToLLVMPattern<
      ttgi::GlueOp>::ConvertTritonGPUOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(ttgi::GlueOp op, OpAdaptor adaptor,
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
  Value vectorGlueOp(ttgi::GlueOp op, ValueRange operands,
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

  Value scalarGlueOp(ttgi::GlueOp op, ValueRange operands,
                     ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    auto dstType =
        cast<VectorType>(getTypeConverter()->convertType(op.getType()));
    Value poison = rewriter.create<LLVM::PoisonOp>(loc, dstType);

    auto enumeratedOperands = llvm::enumerate(operands);
    return std::accumulate(std::begin(enumeratedOperands),
                           std::end(enumeratedOperands), poison,
                           [&rewriter, loc](Value acc, const auto &pair) {
                             auto b = TritonLLVMOpBuilder(loc, rewriter);
                             auto [index, operand] = pair;
                             Value idx = b.i32_val(index);
                             return b.insert_element(acc, operand, idx);
                           });
  }
};

/// %extract = ttgi.extract %a[0] : tensor<8xf16> -> tensor<4xf16>
/// is converted to
/// %extract = llvm.shufflevector %a, %a : [0, 1, 2, 3] : vector<4xf16>
class ExtractOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<ttgi::ExtractOp> {
public:
  using ConvertTritonGPUOpToLLVMPattern<
      ttgi::ExtractOp>::ConvertTritonGPUOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(ttgi::ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
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
      result = b.extract_element(base, idxVal);
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
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    RankedTensorType resultType = op.getType();
    TritonIntelGPUToLLVMTypeConverter *typeConverter = getTypeConverter();
    Type srcTy = adaptor.getSrc().getType();
    VectorType vecTy = VectorType::get(1, srcTy);
    auto poison = rewriter.create<LLVM::PoisonOp>(loc, vecTy);
    auto splat =
        b.insert_element(vecTy, poison, adaptor.getSrc(),
                         rewriter.create<LLVM::ConstantOp>(loc, i32_ty, 0));
    Type convertedTy = typeConverter->convertType(resultType);
    if (!isa<VectorType>(convertedTy)) {
      // On the advance path, the type converter reduces 1-element vectors to
      // their element type, making this splat a no-op.
      rewriter.replaceOp(op, adaptor.getSrc());
      return success();
    }
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
  matchAndRewrite(ReduceOp op, ReduceOpAdaptor adaptor,
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
    Region &combineOp = op.getCombineOp();
    if (!combineOp.hasOneBlock() ||
        combineOp.front().getOperations().size() != 2)
      return failure();

    Operation *combine = &*combineOp.front().getOperations().begin();

    // FIXME: support all possible reduction modes
    TypeSwitch<Operation *>(combine).Case<arith::AddFOp, arith::MaxNumFOp>(
        [&](auto reduce) {
          rewriter.replaceOpWithNewOp<intel::SPIRVGroupOpTy<decltype(reduce)>>(
              op, typeConverter->convertType(op.getType(0)),
              spirv::Scope::Subgroup, spirv::GroupOperation::Reduce,
              adaptor.getSrcs()[0], Value());
        });

    return success();
  }
};

class TransposedReduceOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<ReduceOp> {
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
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    ValueRange srcs = adaptor.getSrcs();
    if (srcs.size() != 1)
      return failure();
    Value src = srcs.front();
    auto srcTy = dyn_cast<VectorType>(src.getType());
    if (!srcTy)
      return failure();
    SmallVector<Value> elements;
    for (int i = 0, size = srcTy.getNumElements(); i < size; ++i)
      elements.push_back(b.extract_element(src, b.i32_val(i)));
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
    funcOp.setAlwaysInline(true);

    Block *entryBlock = funcOp.addEntryBlock(rewriter);
    rewriter.mergeBlocks(combineBlock, entryBlock, entryBlock->getArguments());

    // Clone function
    {
      OpBuilder::InsertionGuard ig(rewriter);
      rewriter.setInsertionPointToEnd(entryBlock);
      // Replace terminator for llvm.func
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
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    llvm::transform(llvm::zip_equal(lhs, rhs), std::back_inserter(res),
                    [&](auto pair) -> Value {
                      auto [lhs, rhs] = pair;
                      auto callOp = b.call(reduceFunc, ValueRange{lhs, rhs});
                      callOp.setCConv(LLVM::cconv::CConv::SPIR_FUNC);
                      return callOp->getResult(0);
                    });
    return treeReduce(op, reduceFunc, rewriter, res);
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
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value src = adaptor.getSrc();
    Type type = getTypeConverter()->convertType(op.getType());
    if (!type)
      return failure();

    auto m = op->getParentOfType<ModuleOp>();
    int size = triton::gpu::TritonGPUDialect::getThreadsPerWarp(m);

    Value res = rewriter.create<LLVM::PoisonOp>(loc, type);
    for (int i = 0; i < size; ++i) {
      Value idx = b.i32_val(i);
      Value element = LLVM::intel::shuffleIdx(loc, rewriter, src, idx);
      res = b.insert_element(res, element, idx);
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
    : public ConvertTritonGPUOpToLLVMPattern<ttgi::BroadcastOp> {
public:
  using ConvertTritonGPUOpToLLVMPattern<
      ttgi::BroadcastOp>::ConvertTritonGPUOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(ttgi::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    constexpr unsigned subgroupSize = 16;

    auto srcShape = op.getSrc().getType().getShape();
    auto dstShape = op.getType().getShape();
    assert(srcShape.size() == 2 && dstShape.size() == 2 &&
           "Expected 2D broadcast");
    assert(dstShape[1] == subgroupSize && "Unexpected result shape");

    if (srcShape[0] == dstShape[0]) {
      // Example: 16x1 --> 16x16 broadcast. Each thread in the subgroup will get
      // the same value, so we use the source operand directly.
      rewriter.replaceOp(op, adaptor.getSrc());
      return success();
    }

    if (srcShape[1] == dstShape[1]) {
      // Example: 1x16 --> 8x16 broadcast. We have extract the element
      // corresponding to the thread's lane ID and splat it to the desired
      // result size.
      Location loc = op.getLoc();
      Value laneId = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getI32Type(),
          rewriter.create<mlir::gpu::LaneIdOp>(loc, /*upperBound=*/nullptr));
      Value extract = rewriter.create<LLVM::ExtractElementOp>(
          loc, adaptor.getSrc(), laneId);
      Value splat =
          rewriter.create<mlir::triton::SplatOp>(loc, op.getType(), extract);
      rewriter.replaceOp(op, splat);
      return success();
    }

    return failure();
  }
};

class SubGroupTransposeOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<ttgi::SubGroupTransposeOp> {
public:
  using ConvertTritonGPUOpToLLVMPattern<
      ttgi::SubGroupTransposeOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ttgi::SubGroupTransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Value src = adaptor.getSrc();
    auto vecTy = cast<VectorType>(src.getType());
    auto mod = op->getParentOfType<ModuleOp>();
    int threadsPerWarp =
        mlir::triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    assert(vecTy.getNumElements() == threadsPerWarp &&
           "Valid input tensor types should convert to a vector of sub-group "
           "size");

    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value localBuffer = adaptor.getLocalBuffer();
    Type offsetType = getTypeConverter()->getIndexType();
    Value subGroupId = getValueOrCreateCastToIndexLike(
        rewriter, loc, offsetType,
        rewriter.create<mlir::gpu::SubgroupIdOp>(
            loc, /*upper_bound=*/IntegerAttr{}));
    Value subGroupLocalId = getValueOrCreateCastToIndexLike(
        rewriter, loc, offsetType,
        rewriter.create<mlir::gpu::LaneIdOp>(loc,
                                             /*upper_bound=*/IntegerAttr{}));
    Value wiStride =
        rewriter.create<LLVM::ConstantOp>(loc, offsetType, threadsPerWarp);
    Value sgStride = rewriter.create<LLVM::ConstantOp>(
        loc, offsetType, threadsPerWarp * threadsPerWarp);
    Value subGroupOffset = b.mul(sgStride, subGroupId);
    Type ptrType = localBuffer.getType();
    Type elementType =
        cast<RankedTensorType>(op.getSrc().getType()).getElementType();
    Value subGroupBasePtr =
        b.gep(ptrType, elementType, localBuffer, ValueRange{subGroupOffset},
              LLVM::GEPNoWrapFlags::inbounds);

    // Store matrix in local memory.
    VectorType intVecTy =
        vec_ty(int_ty(vecTy.getElementType().getIntOrFloatBitWidth()),
               vecTy.getNumElements());
    Value val =
        vecTy.getElementType().isInteger() ? src : b.bitcast(src, intVecTy);
    decomposeBlockStore(rewriter, loc, subGroupBasePtr, val, intVecTy,
                        threadsPerWarp);

    // Load from matrix, trasposed.
    Value workItemOffset = b.mul(wiStride, subGroupLocalId);
    Value workItemBasePtr =
        b.gep(ptrType, elementType, subGroupBasePtr, ValueRange{workItemOffset},
              LLVM::GEPNoWrapFlags::inbounds);
    rewriter.replaceOp(op, b.load(src.getType(), workItemBasePtr));
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
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Type resultType = op.getType();
    LLVMTypeConverter *typeConverter = getTypeConverter();
    Type resultPtrTy = typeConverter->convertType(resultType);
    Type resultElmTy = typeConverter->convertType(
        cast<PointerType>(resultType).getPointeeType());
    Value result =
        b.gep(resultPtrTy, resultElmTy, adaptor.getPtr(), adaptor.getOffset());
    rewriter.replaceOp(op, result);
    return success();
  }
};

class MakeRangeOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<MakeRangeOp> {
public:
  using ConvertTritonGPUOpToLLVMPattern<
      MakeRangeOp>::ConvertTritonGPUOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(MakeRangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Note: On the default path, the lowering of `tt.make_range` takes the
    // tensor layout into account. To that end, there is a dedicated lowering
    // pattern in `MakeRangeOpToLLVM.cpp`. However, with the assumed dense
    // layout in the advanced path, we can just emit a sequence of integers.

    Location loc = op->getLoc();
    Value vec = rewriter.create<LLVM::UndefOp>(
        loc, getTypeConverter()->convertType(op.getType()));
    for (int i = op.getStart(); i < op.getEnd(); ++i) {
      auto valI = LLVM::createConstantI32(loc, rewriter, i);
      vec = rewriter.create<LLVM::InsertElementOp>(loc, vec, valI, valI);
    }

    rewriter.replaceOp(op, vec);
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
  patterns.add<ConvertLayoutOpConversion>(typeConverter, benefit);
  patterns.add<DotOpConversion>(typeConverter, benefit);
  patterns.add<ExpandDimsOpConversion>(typeConverter, benefit);
  patterns.add<ExtractOpConversion>(typeConverter, benefit);
  patterns.add<GlueOpConversion>(typeConverter, benefit);
  patterns.add<LoadStorePrefetchOpConversion<ttgi::PrefetchOp>>(typeConverter,
                                                                benefit);
  patterns.add<LoadStorePrefetchOpConversion<LoadOp>>(typeConverter, benefit);
  patterns.add<LoadStorePrefetchOpConversion<StoreOp>>(typeConverter, benefit);
  patterns.add<MakeTensorPtrOpConversion>(typeConverter, benefit);
  if (ttgi::applyTransposedReduction())
    patterns.add<TransposedReduceOpConversion>(typeConverter, benefit);
  else
    patterns.add<ReduceOpConversion>(typeConverter, benefit);
  patterns.add<SubGroupTransposeOpConversion>(typeConverter, benefit);
  patterns.add<SplatOpConversion>(typeConverter, benefit);
  patterns.add<MakeRangeOpConversion>(typeConverter, benefit);
}
