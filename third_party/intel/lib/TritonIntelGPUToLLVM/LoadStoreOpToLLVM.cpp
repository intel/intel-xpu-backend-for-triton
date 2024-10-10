#include "Dialect/TritonIntelGPU/IR/Dialect.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "Utility.h"

#include "intel/include/Dialect/TritonIntelGPU/IR/Attributes.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::gpu::intel;
namespace {

// Return the mask for the unique data accessed by given tensor type.
// Used to mask out the redundant data accessed by threads.
Value redundantDataMask(Type valueTy, ConversionPatternRewriter &rewriter,
                        Location loc,
                        const triton::intel::TargetInfo &targetInfo) {
  auto tensorTy = dyn_cast<RankedTensorType>(valueTy);
  Value mask = int_val(1, 1);
  auto tid = tid_val();
  auto clusterCTAId = targetInfo.getClusterCTAId(rewriter, loc);
  if (tensorTy) {
    auto layout = tensorTy.getEncoding();
    auto shape = tensorTy.getShape();
    unsigned rank = shape.size();
    auto sizePerThread = triton::gpu::getSizePerThread(layout);
    auto threadsPerWarp = triton::gpu::getThreadsPerWarp(layout);
    auto warpsPerCTA = triton::gpu::getWarpsPerCTA(layout);
    auto order = triton::gpu::getOrder(layout);
    auto shapePerCTATile = triton::gpu::getShapePerCTATile(layout, shape);
    Value warpSize = LLVM::intel::getModuleWarpSize(rewriter, loc);
    Value laneId = urem(tid, warpSize);
    Value warpId = udiv(tid, warpSize);
    SmallVector<Value> multiDimWarpId =
        delinearize(rewriter, loc, warpId, warpsPerCTA, order);
    SmallVector<Value> multiDimThreadId =
        delinearize(rewriter, loc, laneId, threadsPerWarp, order);
    for (unsigned dim = 0; dim < rank; ++dim) {
      // if there is no data replication across threads on this dimension
      if (shape[dim] >= shapePerCTATile[dim])
        continue;
      // Otherwise, we need to mask threads that will replicate data on this
      // dimension. Calculate the thread index on this dimension for the CTA
      Value threadDim =
          add(mul(multiDimWarpId[dim], i32_val(threadsPerWarp[dim])),
              multiDimThreadId[dim]);
      mask = and_(mask, icmp_slt(mul(threadDim, i32_val(sizePerThread[dim])),
                                 i32_val(shape[dim])));
    }
    // Do not write duplicated data when multicast is enabled
    if (triton::gpu::getNumCTAs(layout) > 1) {
      auto _0 = i32_val(0);
      auto CTAsPerCGA = triton::gpu::getCTAsPerCGA(layout);
      auto CTASplitNum = triton::gpu::getCTASplitNum(layout);
      auto CTAOrder = triton::gpu::getCTAOrder(layout);

      auto multiDimClusterCTAId =
          delinearize(rewriter, loc, clusterCTAId, CTAsPerCGA, CTAOrder);

      for (unsigned dim = 0; dim < rank; ++dim) {
        // Skip when multicast is not enabled in this dimension
        if (CTAsPerCGA[dim] == CTASplitNum[dim])
          continue;
        // This wrapping rule must be consistent with emitCTAOffsetForLayout
        unsigned splitNum = std::min<unsigned>(shape[dim], CTASplitNum[dim]);
        Value repId = udiv(multiDimClusterCTAId[dim], i32_val(splitNum));
        // Consider the example where CTAsPerCGA = [4] and CTASplitNum = [2]:
        //     CTA0 and CTA2 holds data of block0,
        //     CTA1 and CTA3 holds data of block1.
        // Only CTA0 and CTA1 are expected to write while CTA2 and CTA3 should
        // be masked. We add the following mask:
        //     multiDimClusterCTAId[dim] / splitNum == 0
        // Actually in all existing cases of multicast, splitNum is always 1.
        // The mask is equivalent to:
        //     multiDimClusterCTAId[dim] == 0
        mask = and_(mask, icmp_eq(repId, _0));
      }
    }
  } else {
    // If the tensor is not ranked, then it is a scalar and only thread 0 of
    // CTA0 can write
    mask = and_(mask, icmp_eq(clusterCTAId, i32_val(0)));
    mask = and_(mask, icmp_eq(tid, i32_val(0)));
  }
  return mask;
}

/// Holds the values related to a block pointer.
/// It includes the base pointer, base width and height, row and column
/// stride, and offset base for X and Y.
struct BlockPointerValues {
  Value base;
  Value baseWidth;
  Value baseHeight;
  Value rowStride;
  Value colStride;
  Value offsetBaseX;
  Value offsetBaseY;
};

// Unpack values as the params to 2DBlockLoad Payload: offsetBaseY,
// offsetBaseX, baseHeight, baseWidth, rowStride, colStride, base.
// FIXME: Only supports 2D matrices for now.
BlockPointerValues
getValuesFromBlockPointerStruct(Value blockPointerStruct,
                                ConversionPatternRewriter &rewriter) {
  const SmallVector<Value> &elems = unpackLLElements(
      blockPointerStruct.getLoc(), blockPointerStruct, rewriter);
  assert(elems.size() == 7 &&
         "unexpected number of values unpacked from a block pointer");
  BlockPointerValues values{
      .base = elems[6],
      .baseWidth = elems[3],
      .baseHeight = elems[2],
      .rowStride = elems[4],
      .colStride = elems[5],
      .offsetBaseX = elems[1],
      .offsetBaseY = elems[0],
  };
  return values;
}

/// Compute the 2D prefetch shape for each warp given an input 2D tensor.
/// Because a cache line is 64 bytes, and we want to prefetch one cache line a
/// time (per thread), the maximum number of bytes per column is 64. We know
/// that the maximum size for each 2D prefetch is 2048 bytes, therefore the
/// maximum number of rows is given by 2048/64=32.
SmallVector<unsigned, 2> get2DPrefetchShapePerWarp(RankedTensorType tensorTy) {
  Type eltTy = tensorTy.getElementType();
  const ArrayRef<int64_t> tensorShape = tensorTy.getShape();
  unsigned elemSizeInBits = eltTy.getIntOrFloatBitWidth();
  unsigned elemSizeInBytes = elemSizeInBits / 8;
  unsigned maxBytesPerCol = 64;
  unsigned numRows = std::min<unsigned>(tensorShape[0], 32);
  unsigned numCols = maxBytesPerCol / elemSizeInBytes;
  return {numRows, numCols};
}

/// Get the 2D warps per CTA given the tensor shape and the prefetch
/// shape per warp.
SmallVector<unsigned, 2>
getWarpsPerCTA(const ArrayRef<int64_t> tensorShape,
               const SmallVector<unsigned, 2> &shapePerWarp,
               unsigned numWarps) {
  assert(tensorShape.size() == 2 && shapePerWarp.size() == 2 &&
         "only 2D tensors are supported");

  unsigned repNumPerRow = mlir::ceil((unsigned)tensorShape[1], shapePerWarp[1]);
  unsigned warpNumPerRow = std::min(numWarps, repNumPerRow);
  unsigned warpNumRow = mlir::ceil(numWarps, warpNumPerRow);
  return {warpNumRow, warpNumPerRow};
}

// Contains some helper functions for both Load and Store conversions.
struct LoadStoreConversionBase {
  explicit LoadStoreConversionBase(const triton::intel::TargetInfo &targetInfo,
                                   ModuleAxisInfoAnalysis &axisAnalysisPass)
      : targetInfo(targetInfo), axisAnalysisPass(axisAnalysisPass) {}

  unsigned getContiguity(Value ptr) const {
    auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
    if (!tensorTy)
      return 1;
    return axisAnalysisPass.getPtrContiguity(ptr);
  }

  unsigned getVectorSize(Value ptr) const {
    auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
    if (!tensorTy)
      return 1;
    auto contiguity = getContiguity(ptr);
    auto pointeeBitWidth = triton::getPointeeBitWidth(tensorTy);
    // The maximum vector size is 128 bits.
    return std::min<unsigned>(128 / pointeeBitWidth, contiguity);
  }

  unsigned getMaskAlignment(Value mask) const {
    return axisAnalysisPass.getMaskAlignment(mask);
  }

  std::tuple<SmallVector<Value>, SmallVector<Value>, SmallVector<Value>>
  convertBlockPtrToTensorOfPtr(
      Location loc, Value blockPointerStruct, RankedTensorType tensorType,
      Type valueElemTy, ConversionPatternRewriter &rewriter,
      ArrayRef<int32_t> boundaryCheck = {},
      std::optional<PaddingOption> padding = std::nullopt) const {

    auto rank = tensorType.getRank();
    // The block pointer struct is expected to have the following layout:
    //    Struct {
    //      Value offset[rank];
    //      Value shape[rank];
    //      Value stride[rank];
    //      Value base;
    //    }
    // All the values are decomposed by `unpackLLElements` into a vector.
    // Defines the indices for the block pointer struct.
    unsigned blockOffset = 0, blockShape = 1 * rank, blockStride = 2 * rank,
             blockBase = 3 * rank;
    const SmallVector<Value> &blockPtr =
        unpackLLElements(loc, blockPointerStruct, rewriter);

    unsigned numElems = getTotalElemsPerThread(tensorType);

    // Get the LLVM values for indices in block
    auto indices = mlir::triton::intel::emitIndices(
        loc, rewriter, targetInfo, tensorType.getEncoding(), tensorType, true);

    auto linearize =
        [](ArrayRef<Value> A, ArrayRef<Value> B, Value init,
           std::function<Value(const Value &, const Value &, const Value &)>
               linearizeFunc) {
          auto rank = A.size();
          Value accumulate = init;
          if (rank > 0) {
            for (auto [a, b] : llvm::zip(A, B)) {
              accumulate = linearizeFunc(a, b, accumulate);
            }
          }
          return accumulate;
        };

    SmallVector<Value> ptrElems(numElems);
    SmallVector<Value> maskElems;
    for (unsigned i = 0; i < numElems; ++i) {
      auto index = indices[i];
      SmallVector<Value> indicesInTensor(rank);
      for (unsigned j = 0; j < rank; ++j) {
        indicesInTensor[j] = add(index[j], blockPtr[blockOffset + j]);
      }

      // Get the LLVM values for pointers
      Value offset = linearize(
          indicesInTensor,
          {blockPtr.begin() + blockStride, blockPtr.begin() + blockBase},
          i32_val(0),
          [&](const Value &index, const Value &stride, const Value &off) {
            // off = off + index * stride
            return add(mul(index, trunc(i32_ty, stride)), off);
          });

      ptrElems[i] = gep(ptr_ty(rewriter.getContext(), 1 /*global*/),
                        valueElemTy, blockPtr[blockBase], offset);

      if (boundaryCheck.size() > 0) {
        // Get the LLVM values for mask
        maskElems.push_back(linearize(
            indicesInTensor,
            {blockPtr.begin() + blockShape, blockPtr.begin() + blockStride},
            int_val(1, 1),
            [&](const Value &index, const Value &shape, const Value &mask) {
              // mask = mask && (index < shape)
              return and_(icmp_slt(index, trunc(i32_ty, shape)), mask);
            }));
      }
    }

    // Get the LLVM values for `other`
    SmallVector<Value> otherElems;
    if (padding) {
      Value other;
      if (*padding == PaddingOption::PAD_ZERO) {
        other = rewriter.create<LLVM::ConstantOp>(
            loc, valueElemTy, rewriter.getZeroAttr(valueElemTy));
      } else if (*padding == PaddingOption::PAD_NAN) {
        assert(!valueElemTy.isIntOrIndex() &&
               "Expect element type to be non-integer type");
        auto apNaN = llvm::APFloat::getNaN(
            cast<FloatType>(valueElemTy).getFloatSemantics());
        other = rewriter.create<LLVM::ConstantOp>(
            loc, valueElemTy, rewriter.getFloatAttr(valueElemTy, apNaN));
      } else {
        llvm_unreachable("Unexpected padding option");
      }
      for (unsigned i = 0; i < numElems; ++i) {
        otherElems.push_back(other);
      }
    }

    return std::make_tuple(ptrElems, maskElems, otherElems);
  }

protected:
  ModuleAxisInfoAnalysis &axisAnalysisPass;
  const triton::intel::TargetInfo &targetInfo;
};

struct PrefetchOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::gpu::intel::PrefetchOp>,
      public LoadStoreConversionBase {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::gpu::intel::PrefetchOp>::ConvertTritonGPUOpToLLVMPattern;

  PrefetchOpConversion(TritonGPUToLLVMTypeConverter &converter,
                       const triton::intel::TargetInfo &targetInfo,
                       ModuleAxisInfoAnalysis &axisAnalysisPass,
                       PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<triton::gpu::intel::PrefetchOp>(
            converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::gpu::intel::PrefetchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Value ptr = op.getPtr();
    if (isTensorPointerType(ptr.getType()))
      return rewriteTensorPointerPrefetch(op, adaptor, rewriter);

    llvm_unreachable("Unexpected prefetch operation on 'regular' ptr");
    return failure();
  }

  LogicalResult
  rewriteTensorPointerPrefetch(triton::gpu::intel::PrefetchOp op,
                               OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {

    Attribute blockIOAttr =
        op->getAttr(TritonIntelGPUDialect::getBlockIOAttrName());
    if (!blockIOAttr) {
      // TODO: Fallback to gather semantic prefetching. Simply erase the
      // prefetching op which is not supported for now.
      rewriter.eraseOp(op);
      return success();
    }

    // Only support rank 2 block pointer, either row major or column major.
    StringRef memoryLayoutInfo = cast<StringAttr>(blockIOAttr).getValue();
    assert((memoryLayoutInfo == "row_major" ||
            memoryLayoutInfo == "column_major") &&
           "Only row_major or column_major is supported");

    const bool memoryRowMajor = (memoryLayoutInfo == "row_major");

    auto mod = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
    Location loc = op.getLoc();
    Value ptr = op.getPtr();
    auto ptrType = cast<PointerType>(ptr.getType());
    auto tensorType = cast<RankedTensorType>(ptrType.getPointeeType());
    Type eltTy = tensorType.getElementType();
    const ArrayRef<int64_t> shapeRef = tensorType.getShape();
    SmallVector<int64_t> tensorShape{shapeRef.begin(), shapeRef.end()};

    if (!memoryRowMajor) {
      // Swap the shape to make it row major and then get the tiling
      // size base on row major shape.
      std::swap(tensorShape[0], tensorShape[1]);
    }

    unsigned numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);

    SmallVector<unsigned, 2> shapePerWarp =
        get2DPrefetchShapePerWarp(tensorType);

    SmallVector<unsigned, 2> warpsPerCTA =
        getWarpsPerCTA(tensorShape, shapePerWarp, numWarps);

    // To adjust the row shape per warp to fit the tensor shape and avoid
    // duplication in prefetching.
    unsigned factor =
        mlir::ceil(shapePerWarp[0] * warpsPerCTA[0], (unsigned)tensorShape[0]);
    shapePerWarp[0] = mlir::ceil(shapePerWarp[0], factor);

    SmallVector<int64_t> numReps = {
        mlir::ceil<int64_t>(tensorShape[0], shapePerWarp[0] * warpsPerCTA[0]),
        mlir::ceil<int64_t>(tensorShape[1], shapePerWarp[1] * warpsPerCTA[1])};

    unsigned elemSizeInBits = eltTy.getIntOrFloatBitWidth();
    unsigned tileWidthInElem = shapePerWarp[1];
    unsigned tileHeightInElem = shapePerWarp[0];
    unsigned vBlocks = 1;
    switch (elemSizeInBits) {
    case 8:
      if (tileWidthInElem == 64) {
        // OCL interface supports 8b_?r32x2c for 64 bytes per row of 8 bits
        // element.
        vBlocks = 2;
        tileWidthInElem = 32;
      }
      break;
    case 16:
      if (tileWidthInElem == 32) {
        // OCL interface supports 16b_?r16x2c for 64 bytes per row of 16 bits
        // element.
        vBlocks = 2;
        tileWidthInElem = 16;
      }
      break;
    }

    Value warpId = rewriter.create<arith::IndexCastOp>(
        loc, i32_ty,
        rewriter.create<mlir::gpu::SubgroupIdOp>(loc, /*upperBound=*/nullptr));
    SmallVector<Value> multiDimWarpId =
        mlir::LLVM::delinearize(rewriter, loc, warpId, warpsPerCTA, {1, 0});

    auto [base, baseWidth, baseHeight, rowStride, colStride, offsetBaseX,
          offsetBaseY] =
        getValuesFromBlockPointerStruct(adaptor.getPtr(), rewriter);

    if (!memoryRowMajor) {
      // Swap the width/height and strides to the row major.
      std::swap(baseWidth, baseHeight);
      std::swap(colStride, rowStride);
    }

    baseWidth = mul(baseWidth, i64_val(eltTy.getIntOrFloatBitWidth() / 8));
    baseWidth = trunc(i32_ty, baseWidth);

    baseHeight = trunc(i32_ty, baseHeight);

    Value rowStrideInBytes =
        mul(rowStride, i64_val(eltTy.getIntOrFloatBitWidth() / 8));
    rowStrideInBytes = trunc(i32_ty, rowStrideInBytes);

    for (int row = 0; row < numReps[0]; ++row) {
      for (int col = 0; col < numReps[1]; ++col) {
        Value offsetX, offsetY;
        offsetX = add(
            // the offset of this warp.
            mul(multiDimWarpId[1], i32_val(shapePerWarp[1])),
            // add the replica offset with a warp stride.
            i32_val(col * warpsPerCTA[1] * shapePerWarp[1]));
        // Round the offset into to the tensor shape
        offsetX = urem(offsetX, i32_val(tensorShape[1]));
        offsetX = add(offsetX, offsetBaseX);
        offsetY = add(
            // the offset of this warp.
            mul(multiDimWarpId[0], i32_val(shapePerWarp[0])),
            // add the replica offset with a warp stride.
            i32_val(row * warpsPerCTA[0] * shapePerWarp[0]));
        // Round the offset into to the tensor shape
        offsetY = urem(offsetY, i32_val(tensorShape[0]));
        offsetY = add(offsetY, offsetBaseY);

        auto newOp = rewriter.create<TritonGEN::Matrix2DBlockPrefetchOp>(
            loc,
            /*ptr*/ base,
            /*base_width*/ baseWidth,
            /*base_height*/ baseHeight,
            /*base_pitch*/ rowStrideInBytes,
            /*x*/ trunc(i32_ty, offsetX),
            /*y*/ trunc(i32_ty, offsetY),
            /*elem_size_in_bits*/ elemSizeInBits,
            /*tile_width*/ tileWidthInElem,
            /*tile_height*/ tileHeightInElem,
            /*v_blocks*/ vBlocks,
            /*cache_opt*/ TritonGEN::LoadCacheControl::L1C_L3C);
        if (failed(newOp.verify())) {
          // Explicitly invoke verifier because `triton_gen` ops are immediately
          // lowered further to a builtin call.
          return failure();
        }
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct LoadOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::LoadOp>,
      public LoadStoreConversionBase {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::LoadOp>::ConvertTritonGPUOpToLLVMPattern;

  using ValueTable = std::map<std::pair<int, int>, Value>;

  LoadOpConversion(TritonIntelGPUToLLVMTypeConverter &converter,
                   const triton::intel::TargetInfo &targetInfo,
                   ModuleAxisInfoAnalysis &axisAnalysisPass,
                   PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<triton::LoadOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  rewriteTensorPointerLoad(triton::LoadOp op, OpAdaptor adaptor,
                           ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    Value ptr = op.getPtr();
    Value mask = op.getMask();
    Value other = op.getOther();
    Type resultType = op.getType();
    auto tensorType = cast<RankedTensorType>(resultType);

    // Only lower loadOp with dpas layout encoding.
    if (!hasDotDpasEncoding(tensorType))
      return failure();

    Attribute blockIOAttr =
        op->getAttr(TritonIntelGPUDialect::getBlockIOAttrName());
    if (!blockIOAttr)
      return failure();

    // Only support rank 2 dot layout, either row major or column major.
    StringRef memoryLayoutInfo = cast<StringAttr>(blockIOAttr).getValue();
    assert((memoryLayoutInfo == "row_major" ||
            memoryLayoutInfo == "column_major") &&
           "Only row_major or column_major is supported");
    const bool memoryRowMajor = (memoryLayoutInfo == "row_major");

    DotOperandEncodingAttr dotLayout = getDotEncoding(tensorType).value();
    auto dotOrder = dotLayout.getThreadOrder();
    const bool valueRowMajor = (dotOrder[0] == 1 && dotOrder[1] == 0);
    assert((valueRowMajor || (dotOrder[0] == 0 && dotOrder[1] == 1)) &&
           "Only row_major or column_major is allowed");
    const bool isTransposeRequired = valueRowMajor ^ memoryRowMajor;

    auto dpasLayout = cast<DpasEncodingAttr>(dotLayout.getParent());

    const unsigned opIdx = dotLayout.getOpIdx();
    Type eltTy = tensorType.getElementType();
    const ArrayRef<int64_t> tensorShape = tensorType.getShape();
    unsigned numElems = getTotalElemsPerThread(resultType);
    SmallVector<int64_t> numReps =
        dpasLayout.getDPASRepetitions(tensorShape, opIdx);
    const SmallVector<unsigned> warpsPerCTA = dpasLayout.getWarpsPerCTA();
    SmallVector<unsigned> dpasOrder = triton::gpu::getOrder(dpasLayout);
    int threadsPerWarp = triton::gpu::getWarpSize(dpasLayout);

    Value warpId = rewriter.create<arith::IndexCastOp>(
        loc, i32_ty,
        rewriter.create<mlir::gpu::SubgroupIdOp>(loc, /*upperBound=*/nullptr));

    SmallVector<Value> multiDimWarpId =
        delinearize(rewriter, loc, warpId, warpsPerCTA, dpasOrder);

    bool isOperandA = (opIdx == 0);
    SmallVector<unsigned> dpasInstShape = isOperandA
                                              ? dpasLayout.getDPASInstShapeA()
                                              : dpasLayout.getDPASInstShapeB();
    SmallVector<unsigned> elemsPerDPASInst = {dpasInstShape[0],
                                              dpasInstShape[1]};
    unsigned elemsPerLanePerDPASInst =
        product<unsigned>(elemsPerDPASInst) / threadsPerWarp;
    TritonGPUToLLVMTypeConverter *typeConverter = getTypeConverter();
    Type unpackedDPASOperandType = LLVM::getFixedVectorType(
        typeConverter->convertType(eltTy), elemsPerLanePerDPASInst);

    // By default, use the unpacked type for the 2D load result type.
    Type loadResultElemType = typeConverter->convertType(eltTy);
    bool usePackedType = false;
    unsigned packedElemsPerLanePerDPASInst = elemsPerLanePerDPASInst;

    // The tensor values are distributed as DotOp layout of DPAS.
    // If the element size of the tensor matches the DPAS packed layout, then
    // use the packed type for the 2D load result type. For example,
    // The intermediate ops generated by ConvertTritonGPUToLLVM:
    //   %0 = load_2d %ptr : vector<8 x i32>
    //   %1 = bitcast %0 : vector<8 x i32> -> vector<16 x f16>
    //   %2 = bitcast %1 : vector<16 x f16> -> vector<8 x i32>
    //   %3 = dpas %2
    // And the LLVM dialect optimization pass can eliminate the duplicated
    // bitcast. Then there is a shortcut to use the load result directly as the
    // input operands to DPAS.
    // TODO: add support for int4 and int2.
    unsigned opsPerChannel = dpasLayout.getOpsPerChannel();
    unsigned elemBits = eltTy.getIntOrFloatBitWidth();
    if ((opsPerChannel == 4 && elemBits == 8) ||
        (opsPerChannel == 2 && elemBits == 16) ||
        (opsPerChannel == 1 && elemBits == 32)) {
      loadResultElemType = (isOperandA && elemBits != 32) ? i16_ty : i32_ty;
      packedElemsPerLanePerDPASInst =
          isOperandA ? elemsPerLanePerDPASInst / (opsPerChannel == 4 ? 2 : 1)
                     : elemsPerLanePerDPASInst / opsPerChannel;
      usePackedType = true;
    }

    Type packedDPASOperandType = LLVM::getFixedVectorType(
        loadResultElemType, packedElemsPerLanePerDPASInst);

    // Outer dim: Dim M or N. Inner dim: Dim K.
    // Round the warp id fit into the tensor shape.
    auto repCluster = dpasLayout.getRepCluster();
    SmallVector<unsigned> warpShape =
        isOperandA ? dpasLayout.getShapeA() : dpasLayout.getShapeB();
    unsigned outerDimRequiredWarpNum =
        mlir::ceil<unsigned>(tensorShape[opIdx], warpShape[opIdx]);
    unsigned outerDimWarpNum =
        std::min<unsigned>(warpsPerCTA[opIdx], outerDimRequiredWarpNum);
    Value outerDimWarpId =
        urem(multiDimWarpId[opIdx], i32_val(outerDimWarpNum));

    auto [base, baseWidth, baseHeight, rowStride, colStride, offsetBaseX,
          offsetBaseY] =
        getValuesFromBlockPointerStruct(adaptor.getPtr(), rewriter);

    unsigned tileWidth = elemsPerDPASInst[dotOrder[0]];
    unsigned tileHeight = elemsPerDPASInst[dotOrder[1]];
    unsigned vBlocks = 1;
    unsigned numOperandsOuterDimPerLoad = 1;
    unsigned numOperandsInnerDimPerLoad = 1;

    unsigned numOperandsPer2DLoadM, numOperandsPer2DloadN;
    if (!isTransposeRequired) {
      numOperandsPer2DLoadM = isOperandA ? repCluster[opIdx] : numReps[!opIdx];
      numOperandsPer2DloadN = isOperandA ? numReps[!opIdx] : repCluster[opIdx];
    } else {
      if (isOperandA)
        return failure();

      if (!usePackedType)
        return failure();

      std::swap(tileHeight, tileWidth);

      // We can decompose the matrix returned by transposed large 2d load
      // when threads per warp < column size. Otherwise we have to load one
      // operand per inst.
      // Note: the tileHeight and numOperandsPer2DLoadM are the column size
      // now.
      numOperandsPer2DLoadM =
          (threadsPerWarp <= tileHeight) ? repCluster[1] : 1;
      // The transpose 2d load only support 1 operand per inst on column.
      // (vBlocks = 1)
      numOperandsPer2DloadN = 1;
    }

    // PVC 2D load supports 32 rows at most. Load multiple dot operands in by
    // enlarging the tileHeight.
    numOperandsPer2DLoadM = std::min(numOperandsPer2DLoadM, 32 / tileHeight);
    tileHeight = tileHeight * numOperandsPer2DLoadM;

    // PVC 2D load supports 64 bytes per row at most. Load multiple dot operands
    // by enlarging the vBlocks.
    unsigned totalBytesPerRowPerDPASOp = tileWidth * elemBits / 8;
    numOperandsPer2DloadN =
        std::min(numOperandsPer2DloadN, 64 / totalBytesPerRowPerDPASOp);
    vBlocks = numOperandsPer2DloadN;

    numOperandsOuterDimPerLoad =
        isOperandA ? numOperandsPer2DLoadM : numOperandsPer2DloadN;
    numOperandsInnerDimPerLoad =
        isOperandA ? numOperandsPer2DloadN : numOperandsPer2DLoadM;

    if (isTransposeRequired)
      std::swap(numOperandsOuterDimPerLoad, numOperandsInnerDimPerLoad);

    unsigned numLoadPerOutRepCluster =
        mlir::ceil<unsigned>(repCluster[opIdx], numOperandsOuterDimPerLoad);

    unsigned numValuesPerLoad = packedElemsPerLanePerDPASInst *
                                numOperandsOuterDimPerLoad *
                                numOperandsInnerDimPerLoad;
    Type load2DGenXType =
        LLVM::getFixedVectorType(loadResultElemType, numValuesPerLoad);

    // The stride for the replicates.
    unsigned repOuterStride = warpShape[opIdx] * outerDimWarpNum;
    unsigned repStride = elemsPerDPASInst[opIdx] * numOperandsOuterDimPerLoad;
    unsigned warpOuterStride = warpShape[opIdx];
    unsigned repKStride = elemsPerDPASInst[opIdx == 0 ? 1 : 0];

    unsigned numRepOuter = numReps[opIdx];
    unsigned numRepInner = numReps[!opIdx];

    Value pitch;
    if (memoryRowMajor) {
      pitch = trunc(i32_ty, rowStride);
    } else {
      // Column major memory. We need to swap the width and height because HW
      // only support row major memory layout.
      pitch = trunc(i32_ty, colStride);
      std::swap(baseWidth, baseHeight);
    }
    baseWidth = trunc(i32_ty, baseWidth);
    baseHeight = trunc(i32_ty, baseHeight);

    unsigned originalElemBits = elemBits;
    if (isTransposeRequired) {
      // adjust the block io parameter to align HW's limitations on
      // transposing load.
      tileWidth = tileWidth / (32 / originalElemBits);
      elemBits = 32;
    }
    Value elemSizeInBytes = i32_val(originalElemBits / 8);

    ValueTable loadVals;
    for (int outer = 0; outer < numRepOuter; ++outer) {
      for (int rep = 0; rep < numLoadPerOutRepCluster; ++rep) {
        for (int k = 0; k < numRepInner; k += numOperandsInnerDimPerLoad) {
          Value offsetX, offsetY;
          if (opIdx == 0) {
            // A
            offsetY = add(mul(outerDimWarpId, i32_val(warpOuterStride)),
                          i32_val(outer * repOuterStride + rep * repStride));
            offsetX = i32_val(k * repKStride);
          } else {
            // B
            offsetX = add(mul(outerDimWarpId, i32_val(warpOuterStride)),
                          i32_val(outer * repOuterStride + rep * repStride));
            offsetY = i32_val(k * repKStride);
          }

          offsetX = add(offsetX, offsetBaseX);
          offsetY = add(offsetY, offsetBaseY);

          if (!memoryRowMajor) {
            // Column major memory. We need to swap the X and Y because HW only
            // support row major memory layout.
            std::swap(offsetX, offsetY);
          }

          if (isTransposeRequired) {
            // adjust the block io parameter to align HW's limitations on
            // transposing load.
            offsetX = udiv(offsetX, i32_val(32 / originalElemBits));
          }

          auto load2dOp = rewriter.create<TritonGEN::Matrix2DBlockLoadOp>(
              loc, load2DGenXType,
              /*ptr*/ base,
              /*base_width*/ mul(baseWidth, elemSizeInBytes),
              /*base_height*/ baseHeight,
              /*base_pitch*/ mul(pitch, elemSizeInBytes),
              /*x*/ trunc(i32_ty, offsetX),
              /*y*/ trunc(i32_ty, offsetY),
              /*elem_size_in_bits*/ elemBits,
              /*tile_width*/ tileWidth,
              /*tile_height*/ tileHeight,
              /*v_blocks*/ vBlocks,
              /*transpose*/ isTransposeRequired,
              /*vnni_transform*/
              (usePackedType && !isOperandA && !isTransposeRequired &&
               eltTy.getIntOrFloatBitWidth() != 32));
          if (failed(load2dOp.verify())) {
            // Explicitly invoke verifier because `triton_gen` ops are
            // immediately lowered further to a builtin call.
            return failure();
          }

          unsigned packedRowNum = opIdx == 0 ? numOperandsOuterDimPerLoad
                                             : numOperandsInnerDimPerLoad;
          unsigned packedColNum = opIdx == 0 ? numOperandsInnerDimPerLoad
                                             : numOperandsOuterDimPerLoad;

          // Decompose the return value to multiple operands.
          unsigned packedColNumPerVBlock = packedColNum / vBlocks;
          for (int vblk = 0; vblk < vBlocks; ++vblk)
            for (int row = 0; row < packedRowNum; ++row)
              for (int col = 0; col < packedColNumPerVBlock; ++col) {

                unsigned operandStartOffset = (vblk * packedRowNum + row) *
                                              packedColNumPerVBlock *
                                              packedElemsPerLanePerDPASInst;

                SmallVector<int32_t> indices(packedElemsPerLanePerDPASInst);
                for (int elemIdx = 0; elemIdx < packedElemsPerLanePerDPASInst;
                     ++elemIdx) {
                  indices[elemIdx] = operandStartOffset +
                                     elemIdx * packedColNumPerVBlock + col;
                }
                DenseI32ArrayAttr attr = rewriter.getDenseI32ArrayAttr(indices);
                Value loadVal = rewriter.create<LLVM::ShuffleVectorOp>(
                    loc, packedDPASOperandType, load2dOp, load2dOp, attr);

                // Save the decomposed vals to the map;
                if (opIdx == 0) {
                  loadVals[{outer * packedRowNum * numLoadPerOutRepCluster +
                                rep * packedRowNum + row,
                            k + vblk * packedColNumPerVBlock + col}] =
                      bitcast(loadVal, unpackedDPASOperandType);
                } else {
                  loadVals[{outer * packedColNum * numLoadPerOutRepCluster +
                                rep * packedColNum +
                                vblk * packedColNumPerVBlock + col,
                            k + row}] =
                      bitcast(loadVal, unpackedDPASOperandType);
                }
              }
        }
      }
    }

    // Extract the value returned by the load ops. And put the values in the
    // expected order for the layout.
    SmallVector<Value> unpackedLoadedVals;
    for (int outer = 0; outer < numRepOuter; ++outer) {
      for (int k = 0; k < numRepInner; ++k) {
        for (int rep = 0; rep < repCluster[opIdx]; ++rep) {
          Value loadVal = loadVals.at({outer * repCluster[opIdx] + rep, k});
          VectorType loadTy = cast<VectorType>(loadVal.getType());
          for (int i = 0; i < loadTy.getNumElements(); ++i) {
            auto val = extract_element(loadVal, i32_val(i));
            unpackedLoadedVals.push_back(val);
          }
        }
      }
    }

    Type llvmResultStructTy = typeConverter->convertType(op.getType());
    Value resultStruct = packLLElements(loc, typeConverter, unpackedLoadedVals,
                                        rewriter, llvmResultStructTy);
    rewriter.replaceOp(op, {resultStruct});

    return success();
  }

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto typeConverter = getTypeConverter();
    auto *ctx = rewriter.getContext();

    // Determine the vectorization size
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(op.getType()));
    unsigned numElems = getTotalElemsPerThread(op.getType());
    unsigned vec = 1;

    SmallVector<Value> ptrElems;
    SmallVector<Value> maskElems;

    bool otherIsSplatConstInt = false;
    int64_t splatVal = 0;
    SmallVector<Value> otherElems;

    if (isTensorPointerType(op.getPtr().getType())) {
      if (rewriteTensorPointerLoad(op, adaptor, rewriter).succeeded()) {
        return success();
      } else {
        // TODO: (johnlu) set the vector size > 1; Need to prove the memory is
        // contiguous on the fast changing dim when fallback to gather load.
        Type resultType = op.getType();
        auto tensorType = cast<RankedTensorType>(resultType);
        std::tie(ptrElems, maskElems, otherElems) =
            convertBlockPtrToTensorOfPtr(
                loc, adaptor.getPtr(), tensorType, valueElemTy, rewriter,
                op.getBoundaryCheck(), op.getPadding());
      }
    } else {
      // original values
      Value ptr = op.getPtr();
      Value other = op.getOther();
      Value mask = op.getMask();

      // adaptor values
      Value llPtr = adaptor.getPtr();
      Value llMask = adaptor.getMask();
      Value llOther = adaptor.getOther();
      vec = getVectorSize(ptr);
      if (llMask)
        vec = std::min<size_t>(vec, getMaskAlignment(mask));

      // Get the LLVM values for pointers
      ptrElems = unpackLLElements(loc, llPtr, rewriter);
      assert(ptrElems.size() == numElems);

      // Get the LLVM values for mask
      if (llMask) {
        maskElems = unpackLLElements(loc, llMask, rewriter);
        assert(maskElems.size() == numElems);
      }

      // Get the LLVM values for `other`
      // TODO: (goostavz) handle when other is const but not splat, which
      //       should be rarely seen
      DenseElementsAttr constAttr;
      if (other && isa<IntegerType>(valueElemTy) &&
          matchPattern(other, m_Constant(&constAttr)) && constAttr.isSplat() &&
          isa<IntegerType>(constAttr.getElementType())) {
        otherIsSplatConstInt = true;
        splatVal = constAttr.getSplatValue<APInt>().getSExtValue();
      }
      if (other) {
        otherElems = unpackLLElements(loc, llOther, rewriter);
      }
    }

    // vectorized iteration through all the pointer/mask/other elements
    const int valueElemNBits =
        std::max(8u, valueElemTy.getIntOrFloatBitWidth());
    const int numVecs = numElems / vec;

    SmallVector<Value> loadedVals;
    for (size_t vecStart = 0; vecStart < numElems; vecStart += vec) {
      // TODO: optimization when ptr is GEP with constant offset
      size_t in_off = 0;

      const size_t maxWordWidth = std::max<size_t>(32, valueElemNBits);
      const size_t totalWidth = valueElemNBits * vec;
      const size_t width = std::min(totalWidth, maxWordWidth);
      const size_t nWords = std::max<size_t>(1, totalWidth / width);
      const size_t wordNElems = width / valueElemNBits;
      const size_t movWidth = width < 16 ? 16 : width;
      assert(wordNElems * nWords * numVecs == numElems);

      Value pred = maskElems.size() ? maskElems[vecStart] : int_val(1, 1);

      SmallVector<Type> retTys(nWords, IntegerType::get(getContext(), width));
      Type retTy = retTys.size() > 1
                       ? vec_ty(IntegerType::get(ctx, width), nWords)
                       : retTys[0];

      Value other_ = undef(retTy);
      if (otherElems.size()) {
        for (size_t ii = 0; ii < nWords; ++ii) {
          size_t size = width / valueElemNBits;

          auto vecTy = vec_ty(valueElemTy, size);
          Value v = undef(vecTy);
          for (size_t s = 0; s < size; ++s) {
            Value falseVal = otherElems[vecStart + ii * size + s];
            Value sVal = createIndexAttrConstant(
                rewriter, loc, this->getTypeConverter()->getIndexType(), s);
            v = insert_element(vecTy, v, falseVal, sVal);
          }
          v = bitcast(v, IntegerType::get(ctx, width));

          if (otherIsSplatConstInt) {
            for (size_t s = 0; s < 32; s += valueElemNBits)
              splatVal |= splatVal << valueElemNBits;
            v = int_val(width, splatVal);
          }

          Value iiVal = createIndexAttrConstant(
              rewriter, loc, this->getTypeConverter()->getIndexType(), ii);
          if (nWords > 1) {
            other_ = insert_element(retTy, other_, v, iiVal);
          } else {
            other_ = v;
          }
        }
      } else {
        other_ = rewriter.create<LLVM::ConstantOp>(loc, retTy,
                                                   rewriter.getZeroAttr(retTy));
      }

      // Create a predicated load operation.
      Block &endBlock = LLVM::intel::createPredicatedBlock(
          rewriter, loc, pred, SmallVector<Value, 1>{other_}, [&]() {
            Value addrElem =
                bitcast(ptrElems[vecStart], ptr_ty(ctx, 1 /*global*/));
            uint32_t alignment = nWords * width / 8;
            Value ret = load(retTy, addrElem, alignment);
            return SmallVector<Value, 1>{ret};
          });
      Value ret = *endBlock.args_begin();

      // Extract and store return values
      SmallVector<Value> rets;
      for (unsigned int ii = 0; ii < nWords; ++ii) {
        Value curr;
        if (isa<VectorType>(retTy)) {
          curr =
              extract_element(IntegerType::get(ctx, width), ret, i32_val(ii));
        } else {
          curr = ret;
        }
        curr = bitcast(curr, LLVM::getFixedVectorType(valueElemTy,
                                                      width / valueElemNBits));
        rets.push_back(curr);
      }
      int tmp = width / valueElemNBits;
      for (size_t ii = 0; ii < vec; ++ii) {
        Value loaded =
            extract_element(valueElemTy, rets[ii / tmp], i32_val(ii % tmp));
        loadedVals.push_back(loaded);
      }
    } // end vec

    Type llvmResultStructTy = typeConverter->convertType(op.getType());
    Value resultStruct = packLLElements(loc, typeConverter, loadedVals,
                                        rewriter, llvmResultStructTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

struct StoreOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::StoreOp>,
      public LoadStoreConversionBase {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::StoreOp>::ConvertTritonGPUOpToLLVMPattern;

  StoreOpConversion(TritonIntelGPUToLLVMTypeConverter &converter,
                    const triton::intel::TargetInfo &targetInfo,
                    ModuleAxisInfoAnalysis &axisAnalysisPass,
                    PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<triton::StoreOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  rewriteTensorPointerStore(triton::StoreOp op, OpAdaptor adaptor,
                            ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    Type resultType = op.getValue().getType();
    auto tensorType = cast<RankedTensorType>(resultType);

    // Only lower StoreOp with dpas layout encoding.
    if (!hasDpasEncoding(tensorType))
      return failure();

    auto dpasLayout = cast<DpasEncodingAttr>(tensorType.getEncoding());
    TritonGPUToLLVMTypeConverter *typeConverter = getTypeConverter();
    MLIRContext *ctx = rewriter.getContext();

    Type eltTy = tensorType.getElementType();
    unsigned elemSizeInBits = eltTy.getIntOrFloatBitWidth();
    Value elemSizeInBytes = i32_val(elemSizeInBits / 8);
    const ArrayRef<int64_t> tensorShape = tensorType.getShape();
    unsigned numElems = getTotalElemsPerThread(tensorType);
    SmallVector<unsigned> elemsPerInstr = dpasLayout.getDPASInstShapeC();
    const SmallVector<unsigned> warpsPerCTA = dpasLayout.getWarpsPerCTA();
    SmallVector<int64_t> numReps =
        dpasLayout.getDPASRepetitions(tensorShape, 2);
    SmallVector<unsigned> order = triton::gpu::getOrder(dpasLayout);
    unsigned threadsPerWarp = triton::gpu::getWarpSize(dpasLayout);

    Value warpId = rewriter.create<arith::IndexCastOp>(
        loc, i32_ty,
        rewriter.create<mlir::gpu::SubgroupIdOp>(loc, /*upperBound=*/nullptr));
    SmallVector<Value> multiDimWarpId =
        mlir::LLVM::delinearize(rewriter, loc, warpId, warpsPerCTA, order);

    int64_t elemsPerLane = product<unsigned>(elemsPerInstr) / threadsPerWarp;
    Type store2DGenXType =
        LLVM::getFixedVectorType(IntegerType::get(ctx, elemSizeInBits),
                                 elemsPerLane); // make it opaque type.

    Value blockPtr = adaptor.getPtr();
    auto [base, width, height, rowStride, colStride, offsetBaseX, offsetBaseY] =
        getValuesFromBlockPointerStruct(blockPtr, rewriter);

    auto vals = unpackLLElements(loc, adaptor.getValue(), rewriter);
    assert(vals.size() == numElems);

    width = trunc(i32_ty, width);
    height = trunc(i32_ty, height);
    rowStride = trunc(i32_ty, rowStride);
    // encoded as bytes.
    Value baseWidth = mul(width, elemSizeInBytes);
    // encoded as bytes.
    Value basePitch = mul(rowStride, elemSizeInBytes);

    // A warp stride for the replicates.
    SmallVector<unsigned> repClusterShape = dpasLayout.getShapeC();
    unsigned outerDimWarpNum = std::min<unsigned>(
        warpsPerCTA[0],
        mlir::ceil<unsigned>(tensorShape[0], repClusterShape[0]));
    unsigned innerDimWarpNum = std::min<unsigned>(
        warpsPerCTA[1],
        mlir::ceil<unsigned>(tensorShape[1], repClusterShape[1]));
    Value outerDimWarpId = urem(multiDimWarpId[0], i32_val(outerDimWarpNum));
    Value innerDimWarpId = urem(multiDimWarpId[1], i32_val(innerDimWarpNum));
    int64_t numRepOuter = numReps[0];
    int64_t numRepInner = numReps[1];

    std::array<unsigned, 2> replicaStride = {
        outerDimWarpNum * repClusterShape[0],
        innerDimWarpNum * repClusterShape[1]};
    std::array<unsigned, 2> warpStride = {repClusterShape[0],
                                          repClusterShape[1]};

    Value dimWarpId0 = mul(outerDimWarpId, i32_val(warpStride[0]));
    Value dimWarpId1 = mul(innerDimWarpId, i32_val(warpStride[1]));
    Value warpId0Offset = add(dimWarpId0, offsetBaseY);
    Value warpId1Offset = add(dimWarpId1, offsetBaseX);

    ArrayRef<unsigned> repCluster = dpasLayout.getRepCluster();
    unsigned valOffset = 0;
    for (int m = 0; m < numRepOuter; ++m) {
      for (int n = 0; n < numRepInner; ++n) {
        for (int repM = 0; repM < repCluster[0]; ++repM) {
          Value offsetY = add(warpId0Offset, i32_val(m * replicaStride[0] +
                                                     repM * elemsPerInstr[0]));
          for (int repN = 0; repN < repCluster[1]; ++repN) {
            Value offsetX =
                add(warpId1Offset,
                    i32_val(n * replicaStride[1] + repN * elemsPerInstr[1]));
            Value storeVal = rewriter.create<LLVM::UndefOp>(
                loc, LLVM::getFixedVectorType(typeConverter->convertType(eltTy),
                                              elemsPerLane));
            for (size_t i = 0; i < elemsPerLane; ++i) {
              storeVal = insert_element(storeVal, vals[valOffset], i32_val(i));
              ++valOffset;
            }

            auto newOp = rewriter.create<TritonGEN::Matrix2DBlockStoreOp>(
                loc,
                /*ptr*/ base,
                /*base_width*/ baseWidth,
                /*base_height*/ height,
                /*base_pitch*/ basePitch,
                /*x*/ trunc(i32_ty, offsetX),
                /*y*/ trunc(i32_ty, offsetY),
                /*elem_size_in_bits*/ elemSizeInBits,
                /*tile_width*/ elemsPerInstr[1],
                /*tile_height*/ elemsPerInstr[0],
                /*v_blocks*/ 1,
                /*stored_val*/ bitcast(storeVal, store2DGenXType));

            if (failed(newOp.verify())) {
              // Explicitly invoke verifier because `triton_gen` ops are
              // immediately lowered further to a builtin call.
              return failure();
            }
          }
        }
      }
    }
    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();

    Value ptr = op.getPtr();
    Value value = op.getValue();

    auto valueTy = value.getType();
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));
    SmallVector<Value> ptrElems;
    SmallVector<Value> maskElems;
    unsigned vec = 1;

    if (isTensorPointerType(ptr.getType())) {
      if (rewriteTensorPointerStore(op, adaptor, rewriter).succeeded()) {
        return success();
      } else {
        // fallback to scatter store.
        auto tensorType = cast<RankedTensorType>(valueTy);
        SmallVector<Value> dummyOther;
        std::tie(ptrElems, maskElems, dummyOther) =
            convertBlockPtrToTensorOfPtr(loc, adaptor.getPtr(), tensorType,
                                         valueElemTy, rewriter,
                                         op.getBoundaryCheck());
      }
    } else {
      Value llPtr = adaptor.getPtr();
      Value llMask = adaptor.getMask();

      vec = getVectorSize(ptr);

      ptrElems = unpackLLElements(loc, llPtr, rewriter);

      // Determine the vectorization size
      if (llMask) {
        Value mask = op.getMask();
        maskElems = unpackLLElements(loc, llMask, rewriter);

        unsigned maskAlign = getMaskAlignment(mask);
        vec = std::min(vec, maskAlign);
      }
    }

    Value llValue = adaptor.getValue();
    auto valueElems = unpackLLElements(loc, llValue, rewriter);
    assert(ptrElems.size() == valueElems.size());
    assert(!maskElems.size() ||
           valueElems.size() == maskElems.size() && "Mask size mismatch");

    Value mask = redundantDataMask(valueTy, rewriter, loc, targetInfo);
    const size_t dtsize =
        std::max<int>(1, valueElemTy.getIntOrFloatBitWidth() / 8);
    const size_t valueElemNBits = dtsize * 8;

    unsigned elemsPerThread = getTotalElemsPerThread(valueTy);
    const int numVecs = elemsPerThread / vec;
    for (size_t vecStart = 0; vecStart < elemsPerThread; vecStart += vec) {
      // TODO: optimization when ptr is AddPtr with constant offset
      size_t in_off = 0;

      const size_t maxWordWidth = std::max<size_t>(32, valueElemNBits);
      const size_t totalWidth = valueElemNBits * vec;
      const size_t width = std::min(totalWidth, maxWordWidth);
      const size_t nWords = std::max<size_t>(1, totalWidth / width);
      const size_t wordNElems = width / valueElemNBits;
      assert(wordNElems * nWords * numVecs == elemsPerThread);

      // TODO(Superjomn) Add cache policy fields to StoreOp.
      // TODO(Superjomn) Deal with cache policy here.

      Type valArgTy = IntegerType::get(ctx, width);
      auto wordTy = vec_ty(valueElemTy, wordNElems);

      SmallVector<std::pair<Value, std::string>> asmArgs;
      for (size_t wordIdx = 0; wordIdx < nWords; ++wordIdx) {
        // llWord is a width-len composition
        Value llWord = undef(wordTy);
        // Insert each value element to the composition
        for (size_t elemIdx = 0; elemIdx < wordNElems; ++elemIdx) {
          const size_t elemOffset = vecStart + wordIdx * wordNElems + elemIdx;
          assert(elemOffset < valueElems.size());
          Value elem = valueElems[elemOffset];
          if (elem.getType().isInteger(1))
            elem = sext(i8_ty, elem);
          elem = bitcast(elem, valueElemTy);

          llWord = insert_element(wordTy, llWord, elem, i32_val(elemIdx));
        }
        llWord = bitcast(llWord, valArgTy);
        std::string constraint =
            (width == 64) ? "l" : ((width == 32) ? "r" : "c");
        asmArgs.emplace_back(llWord, constraint);
      }

      Value maskVal = maskElems.size() ? and_(mask, maskElems[vecStart]) : mask;

      auto vecTy = vec_ty(valArgTy, nWords);
      Value vecWord = undef(vecTy);
      for (int index = 0; index < asmArgs.size(); ++index) {
        auto llWord = asmArgs[index].first;
        vecWord = insert_element(vecTy, vecWord, llWord, i32_val(index));
      }

      // Create a predicated store operation.
      LLVM::intel::createPredicatedBlock(rewriter, loc, maskVal, [&] {
        Value addrElem = bitcast(ptrElems[vecStart], ptr_ty(ctx, 1 /*global*/));
        uint32_t alignment = nWords * width / 8;
        store(vecWord, addrElem, alignment);
        return ArrayRef<Value>();
      });
    } // for
    rewriter.eraseOp(op);
    return success();
  }
};

void createBarrier(ConversionPatternRewriter &rewriter, Location loc,
                   int numCTAs) {
  assert(numCTAs == 1 && "Expecting numCTA to be 1");
  barrier();
}

struct AtomicCASOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::AtomicCASOp>,
      public LoadStoreConversionBase {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::AtomicCASOp>::ConvertTritonGPUOpToLLVMPattern;

  AtomicCASOpConversion(TritonIntelGPUToLLVMTypeConverter &converter,
                        const triton::intel::TargetInfo &targetInfo,
                        ModuleAxisInfoAnalysis &axisAnalysisPass,
                        PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<triton::AtomicCASOp>(converter,
                                                             benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::AtomicCASOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto moduleOp = op->getParentOfType<ModuleOp>();
    assert(moduleOp && "Parent ModuleOp not found for AtomicCASOp");
    int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(moduleOp);

    Value llPtr = adaptor.getPtr();
    Value llCmp = adaptor.getCmp();
    Value llVal = adaptor.getVal();

    auto ptrElements = unpackLLElements(loc, llPtr, rewriter);
    auto cmpElements = unpackLLElements(loc, llCmp, rewriter);
    auto valElements = unpackLLElements(loc, llVal, rewriter);

    auto valueTy = op.getType();
    auto tensorTy = dyn_cast<RankedTensorType>(valueTy);
    Type valueElemTy =
        tensorTy ? getTypeConverter()->convertType(tensorTy.getElementType())
                 : valueTy;
    auto valueElemNBits = valueElemTy.getIntOrFloatBitWidth();
    auto elemsPerThread = getTotalElemsPerThread(op.getVal().getType());
    // vec = 1 for scalar
    auto vec = getVectorSize(op.getPtr());
    // tensor
    if (tensorTy) {
      auto valTy = cast<RankedTensorType>(op.getVal().getType());
      vec = std::min<unsigned>(vec, valTy.getElementType().isF16() ? 2 : 1);
    }

    Value mask = redundantDataMask(valueTy, rewriter, loc, targetInfo);
    auto vecTy = vec_ty(valueElemTy, vec);
    SmallVector<Value> resultVals(elemsPerThread);

    for (size_t i = 0; i < elemsPerThread; i += vec) {
      Value casVal = undef(vecTy);
      for (int ii = 0; ii < vec; ++ii) {
        Value iiVal = createIndexAttrConstant(
            rewriter, loc, getTypeConverter()->getIndexType(), ii);
        casVal = insert_element(vecTy, casVal, valElements[i + ii], iiVal);
      }

      Value casPtr = ptrElements[i];
      Value casCmp = cmpElements[i];
      casVal = valElements[i];

      assert((valueElemNBits == 32 || valueElemNBits == 64) &&
             "Unexpected width");

      Value zero = (valueElemNBits == 32) ? i32_val(0) : i64_val(0);
      if (!atomicNeedsSharedMemory(op.getResult()))
        rewriter.create<spirv::ControlBarrierOp>(
            loc, mlir::spirv::Scope::Workgroup, mlir::spirv::Scope::Workgroup,
            mlir::spirv::MemorySemantics::WorkgroupMemory);
      Block &endBlock =
          LLVM::intel::createPredicatedBlock(rewriter, loc, mask, {zero}, [&] {
            // casPtr = bitcast(casPtr, ptr_ty(ctx, 1));
            casCmp = bitcast(casCmp, zero.getType());
            casVal = bitcast(casVal, zero.getType());

            auto cmpxchg = rewriter.create<LLVM::AtomicCmpXchgOp>(
                loc, casPtr, casCmp, casVal, LLVM::AtomicOrdering::acq_rel,
                LLVM::AtomicOrdering::monotonic);
            Value newLoaded =
                rewriter.create<LLVM::ExtractValueOp>(loc, cmpxchg, 0);
            return SmallVector<Value, 1>{newLoaded};
          });

      Value ret = endBlock.getArgument(0);
      Type retType = (!tensorTy || vec == 1) ? valueElemTy : vecTy;
      ret = bitcast(ret, retType);

      if (tensorTy) {
        for (int ii = 0; ii < vec; ++ii) {
          resultVals[i + ii] =
              vec == 1 ? ret : extract_element(valueElemTy, ret, i32_val(ii));
        }
      } else {
        if (!atomicNeedsSharedMemory(op.getResult())) {
          rewriter.eraseOp(op);
          return success();
        }
        Value atomPtr = LLVM::intel::getSharedMemoryBase(
            loc, rewriter, targetInfo, op.getOperation());
        atomPtr = bitcast(atomPtr, ptr_ty(ctx, 3));
        targetInfo.storeShared(rewriter, loc, atomPtr, ret, mask);
        createBarrier(rewriter, loc, numCTAs);
        Value ret = load(valueElemTy, atomPtr);
        rewriter.replaceOp(op, {ret});
      }
    }

    if (tensorTy) {
      Type structTy = getTypeConverter()->convertType(tensorTy);
      Value resultStruct = packLLElements(loc, getTypeConverter(), resultVals,
                                          rewriter, structTy);
      rewriter.replaceOp(op, {resultStruct});
    }
    return success();
  }
};

struct AtomicRMWOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::AtomicRMWOp>,
      public LoadStoreConversionBase {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::AtomicRMWOp>::ConvertTritonGPUOpToLLVMPattern;

  AtomicRMWOpConversion(TritonIntelGPUToLLVMTypeConverter &converter,
                        const triton::intel::TargetInfo &targetInfo,
                        ModuleAxisInfoAnalysis &axisAnalysisPass,
                        PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<triton::AtomicRMWOp>(converter,
                                                             benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::AtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto moduleOp = op->getParentOfType<ModuleOp>();
    assert(moduleOp && "Parent ModuleOp not found for AtomicRMWOp");
    int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(moduleOp);

    auto atomicRmwAttr = op.getAtomicRmwOp();

    Value val = op.getVal();
    Value ptr = op.getPtr();

    Value llPtr = adaptor.getPtr();
    Value llVal = adaptor.getVal();
    Value llMask = adaptor.getMask();

    auto valElements = unpackLLElements(loc, llVal, rewriter);
    auto ptrElements = unpackLLElements(loc, llPtr, rewriter);
    SmallVector<Value> maskElements;
    if (llMask)
      maskElements = unpackLLElements(loc, llMask, rewriter);

    auto valueTy = op.getType();
    auto tensorTy = dyn_cast<RankedTensorType>(valueTy);
    Type valueElemTy =
        tensorTy ? getTypeConverter()->convertType(tensorTy.getElementType())
                 : valueTy;
    const size_t valueElemNBits = valueElemTy.getIntOrFloatBitWidth();
    auto elemsPerThread = getTotalElemsPerThread(val.getType());
    // vec = 1, numElements = 1 for scalar
    auto vec = getVectorSize(ptr);
    int numElems = 1;
    // tensor
    if (tensorTy) {
      auto valTy = cast<RankedTensorType>(val.getType());
      auto maxVecSize =
          valueElemNBits / valTy.getElementType().getIntOrFloatBitWidth();
      vec = std::min<unsigned>(vec,
                               valTy.getElementType().isF16() ? maxVecSize : 1);
      // mask
      numElems = tensorTy.getNumElements();
    }
    Value mask = redundantDataMask(valueTy, rewriter, loc, targetInfo);

    auto vecTy = vec_ty(valueElemTy, vec);
    SmallVector<Value> resultVals(elemsPerThread);
    for (size_t i = 0; i < elemsPerThread; i += vec) {
      Value rmwVal = undef(vecTy);
      for (int ii = 0; ii < vec; ++ii) {
        Value iiVal = createIndexAttrConstant(
            rewriter, loc, getTypeConverter()->getIndexType(), ii);
        rmwVal = insert_element(vecTy, rmwVal, valElements[i + ii], iiVal);
      }

      Value rmwPtr = ptrElements[i];
      Value rmwMask = llMask ? and_(mask, maskElements[i]) : mask;

      assert((valueElemNBits == 16 || valueElemNBits == 32 ||
              valueElemNBits == 64) &&
             "Unexpected width");

      Value zero;
      llvm::TypeSwitch<mlir::Type>(valueElemTy)
          .Case<mlir::IntegerType>(
              [&](auto ty) { zero = int_val(valueElemNBits, 0); })
          .Case<mlir::Float16Type>([&](auto ty) { zero = f16_val(0); })
          .Case<mlir::Float32Type>([&](auto ty) { zero = f32_val(0); })
          .Case<mlir::Float64Type>([&](auto ty) { zero = f64_val(0); });

      Block *endBlock = nullptr;
      // TODO: check device capabilities to avoid unnecessary emulation or
      // emit unsupported feature error.
      if (valueElemNBits == 16) {
        op.emitWarning(
            "'tt.atomic_rmw' op fp16 datatype is not supported in the target "
            "HW, software emulation is an experimental feature (use at own "
            "risk)");
        endBlock =
            emulateFp16AtomicRmw(rewriter, loc, atomicRmwAttr, valueElemTy,
                                 rmwPtr, rmwVal, rmwMask, {zero});
      } else {
        if (!atomicNeedsSharedMemory(op.getResult()))
          rewriter.create<spirv::ControlBarrierOp>(
              loc, mlir::spirv::Scope::Workgroup, mlir::spirv::Scope::Workgroup,
              mlir::spirv::MemorySemantics::WorkgroupMemory |
                  mlir::spirv::MemorySemantics::CrossWorkgroupMemory);
        endBlock = &LLVM::intel::createPredicatedBlock(
            rewriter, loc, rmwMask, {zero}, [&] {
              mlir::LLVM::AtomicBinOp rmwKind;
              switch (atomicRmwAttr) {
              case RMWOp::AND:
                rmwKind = LLVM::AtomicBinOp::_and;
                break;
              case RMWOp::OR:
                rmwKind = LLVM::AtomicBinOp::_or;
                break;
              case RMWOp::XOR:
                rmwKind = LLVM::AtomicBinOp::_xor;
                break;
              case RMWOp::ADD:
                rmwKind = LLVM::AtomicBinOp::add;
                break;
              case RMWOp::FADD:
                rmwKind = LLVM::AtomicBinOp::fadd;
                break;
              case RMWOp::MAX:
                rmwKind = LLVM::AtomicBinOp::max;
                break;
              case RMWOp::UMAX:
                rmwKind = LLVM::AtomicBinOp::umax;
                break;
              case RMWOp::MIN:
                rmwKind = LLVM::AtomicBinOp::min;
                break;
              case RMWOp::UMIN:
                rmwKind = LLVM::AtomicBinOp::umin;
                break;
              case RMWOp::XCHG:
                rmwKind = LLVM::AtomicBinOp::xchg;
                break;
              }

              rmwVal = bitcast(rmwVal, valueElemTy);
              auto atomRMW = rewriter.create<LLVM::AtomicRMWOp>(
                  loc, rmwKind, rmwPtr, rmwVal, LLVM::AtomicOrdering::acq_rel);
              return SmallVector<Value, 1>{atomRMW.getRes()};
            });
      }

      Value ret = endBlock->getArgument(0);
      Type retType = (!tensorTy || vec == 1) ? valueElemTy : vecTy;
      ret = bitcast(ret, retType);

      if (tensorTy) {
        for (int ii = 0; ii < vec; ++ii) {
          resultVals[i + ii] =
              vec == 1 ? ret : extract_element(valueElemTy, ret, i32_val(ii));
        }
      } else {
        if (!atomicNeedsSharedMemory(op.getResult())) {
          rewriter.eraseOp(op);
          return success();
        }
        Value atomPtr = LLVM::intel::getSharedMemoryBase(
            loc, rewriter, targetInfo, op.getOperation());
        atomPtr = bitcast(atomPtr, ptr_ty(ctx, 3));
        // Only threads with rmwMask = True store the result
        targetInfo.storeShared(rewriter, loc, atomPtr, ret, rmwMask);
        createBarrier(rewriter, loc, numCTAs);
        Value loadVal = load(valueElemTy, atomPtr);
        rewriter.replaceOp(op, {loadVal});
      }
    }

    if (tensorTy) {
      Type structTy = getTypeConverter()->convertType(tensorTy);
      Value resultStruct = packLLElements(loc, getTypeConverter(), resultVals,
                                          rewriter, structTy);
      rewriter.replaceOp(op, {resultStruct});
    }
    return success();
  }

  // Emulate 16-bit atomicrmw through a loop with 32-bit cmpxchg.
  Block *emulateFp16AtomicRmw(ConversionPatternRewriter &rewriter, Location loc,
                              mlir::triton::RMWOp atomicOp, Type valueElemTy,
                              Value rmwPtr, Value rmwVal, Value rmwMask,
                              ArrayRef<Value> ops) const {
    Block *insertionBlock = rewriter.getInsertionBlock();
    Block *headerBlock =
        rewriter.splitBlock(insertionBlock, rewriter.getInsertionPoint());
    Block *endBlock = rewriter.splitBlock(headerBlock, headerBlock->begin());
    rewriter.setInsertionPointToEnd(insertionBlock);
    rewriter.create<cf::CondBranchOp>(loc, rmwMask, headerBlock, endBlock, ops);
    rewriter.setInsertionPointToStart(headerBlock);

    rmwVal = bitcast(rmwVal, valueElemTy);

    // Align pointer by 4 bytes by zeroing lower address bits. Atomically read
    // a vector of two fp16 values as a single i32. The second lowest bit is
    // extracted to later be used as an index to extract the required vector
    // element.
    assert(isa<LLVM::LLVMPointerType>(rmwPtr.getType()));
    auto intPtr = ptrtoint(i64_ty, rmwPtr);
    auto lowPtrBits = and_(intPtr, i64_val(3));
    auto elemIndex = trunc(i32_ty, lshr(lowPtrBits, i64_val(1)));
    auto alignPtr = inttoptr(rmwPtr.getType(), sub(intPtr, lowPtrBits));
    auto firstValInt = load(i32_ty, alignPtr, 4, false, false, false,
                            LLVM::AtomicOrdering::acquire);

    // Create a loop body block. It has a single parameter which holds the
    // latest loaded i32 value.
    Block *bodyBlock =
        rewriter.splitBlock(headerBlock, rewriter.getInsertionPoint());
    auto origValInt =
        bodyBlock->addArgument(firstValInt.getType(), firstValInt.getLoc());
    rewriter.setInsertionPointToEnd(headerBlock);
    rewriter.create<cf::BranchOp>(loc, bodyBlock,
                                  SmallVector<Value, 1>{firstValInt});
    rewriter.setInsertionPointToEnd(bodyBlock);

    // Extract value for modification.
    auto origValVec = bitcast(origValInt, vec_ty(valueElemTy, 2));
    Value origVal = extract_element(origValVec, elemIndex);

    // Apply operation.
    Value newVal = nullptr;
    switch (atomicOp) {
    case RMWOp::FADD:
      newVal = rewriter.create<LLVM::FAddOp>(loc, origVal, rmwVal);
      break;
    case RMWOp::MAX:
      newVal = rewriter.create<LLVM::MaximumOp>(loc, origVal, rmwVal);
      break;
    case RMWOp::MIN:
      newVal = rewriter.create<LLVM::MinimumOp>(loc, origVal, rmwVal);
      break;
    case RMWOp::XCHG:
      newVal = rmwVal;
      break;
    default:
      llvm_unreachable("Unsupported FP16 atomic op");
    }

    // Use modified value to form a new i32 value to write to memory.
    assert(newVal);
    Value newValVec = insert_element(origValVec, newVal, elemIndex);
    Value newValInt = bitcast(newValVec, i32_ty);

    // Execute cmpxchg and loop back if it fails.
    auto successOrdering = LLVM::AtomicOrdering::acq_rel;
    auto failureOrdering = LLVM::AtomicOrdering::monotonic;
    auto cmpxchg = rewriter.create<LLVM::AtomicCmpXchgOp>(
        loc, alignPtr, origValInt, newValInt, successOrdering, failureOrdering);
    auto newLoaded = extract_val(cmpxchg, 0);
    auto done = extract_val(cmpxchg, 1);
    assert(ops.size() == (size_t)1);
    SmallVector<Value, 1> endOps = {origVal};
    rewriter.create<cf::CondBranchOp>(loc, done, endBlock, endOps, bodyBlock,
                                      SmallVector<Value, 1>{newLoaded});

    for (Value op : ops)
      endBlock->addArgument(op.getType(), op.getLoc());

    rewriter.setInsertionPointToStart(endBlock);
    return endBlock;
  }
};

} // namespace

void mlir::triton::intel::populateLoadStoreOpToLLVMPatterns(
    TritonIntelGPUToLLVMTypeConverter &typeConverter,
    const TargetInfo &targetInfo, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, PatternBenefit benefit) {
  patterns.add<AtomicCASOpConversion, AtomicRMWOpConversion, LoadOpConversion,
               StoreOpConversion, PrefetchOpConversion>(
      typeConverter, targetInfo, axisInfoAnalysis, benefit);
}
