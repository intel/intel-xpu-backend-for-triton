#include "Dialect/TritonIntelGPU/IR/Dialect.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "Utility.h"

#include "intel/include/Dialect/TritonIntelGPU/IR/Attributes.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Utility.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::gpu::intel;

#define S(v) StringAttr::get(ctx, (v))

namespace {

// Return the mask for the unique data accessed by given tensor type.
// Used to mask out the redundant data accessed by threads.
Value redundantDataMask(Type valueTy, ConversionPatternRewriter &rewriter,
                        Location loc,
                        const triton::intel::TargetInfo &targetInfo) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto tensorTy = dyn_cast<RankedTensorType>(valueTy);
  Value mask = b.int_val(1, 1);
  auto tid = getThreadId(rewriter, loc);
  auto clusterCTAId = targetInfo.getClusterCTAId(rewriter, loc);
  if (tensorTy) {
    auto layout = tensorTy.getEncoding();
    auto shape = tensorTy.getShape();
    unsigned rank = shape.size();
    auto sizePerThread = triton::gpu::getSizePerThread(layout);
    auto threadsPerWarp = triton::gpu::getThreadsPerWarp(layout);
    auto warpsPerCTA = triton::gpu::getWarpsPerCTA(layout);
    auto order = triton::gpu::getOrder(layout);
    auto shapePerCTATile = triton::gpu::getShapePerCTATile(layout);
    Value warpSize = LLVM::intel::getModuleWarpSize(rewriter, loc);
    Value laneId = b.urem(tid, warpSize);
    Value warpId = b.udiv(tid, warpSize);
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
          b.add(b.mul(multiDimWarpId[dim], b.i32_val(threadsPerWarp[dim])),
                multiDimThreadId[dim]);
      mask = b.and_(mask,
                    b.icmp_slt(b.mul(threadDim, b.i32_val(sizePerThread[dim])),
                               b.i32_val(shape[dim])));
    }
    // Do not write duplicated data when multicast is enabled
    if (triton::gpu::getNumCTAs(layout) > 1) {
      auto _0 = b.i32_val(0);
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
        Value repId = b.udiv(multiDimClusterCTAId[dim], b.i32_val(splitNum));
        // Consider the example where CTAsPerCGA = [4] and CTASplitNum = [2]:
        //     CTA0 and CTA2 holds data of block0,
        //     CTA1 and CTA3 holds data of block1.
        // Only CTA0 and CTA1 are expected to write while CTA2 and CTA3 should
        // be masked. We add the following mask:
        //     multiDimClusterCTAId[dim] / splitNum == 0
        // Actually in all existing cases of multicast, splitNum is always 1.
        // The mask is equivalent to:
        //     multiDimClusterCTAId[dim] == 0
        mask = b.and_(mask, b.icmp_eq(repId, _0));
      }
    }
  } else {
    // If the tensor is not ranked, then it is a scalar and only thread 0 of
    // CTA0 can write
    mask = b.and_(mask, b.icmp_eq(clusterCTAId, b.i32_val(0)));
    mask = b.and_(mask, b.icmp_eq(tid, b.i32_val(0)));
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
  BlockPointerValues values{/*base=*/elems[6],
                            /*baseWidth=*/elems[3],
                            /*baseHeight=*/elems[2],
                            /*rowStride=*/elems[4],
                            /*colStride=*/elems[5],
                            /*offsetBaseX=*/elems[1],
                            /*offsetBaseY=*/elems[0]};
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
  explicit LoadStoreConversionBase(
      const triton::intel::TargetInfo &targetInfo,
      const triton::intel::ModuleAxisInfoAnalysis &axisAnalysisPass)
      : targetInfo(targetInfo), axisAnalysisPass(axisAnalysisPass) {}

  unsigned getContiguity(Value ptr) const {
    return const_cast<triton::intel::ModuleAxisInfoAnalysis &>(axisAnalysisPass)
        .getPtrContiguity(ptr);
  }

  unsigned getVectorSize(Value ptr) const {
    auto tensorTy = getRankedTensorType(ptr.getType());
    if (!tensorTy)
      return 1;

    unsigned contiguity = getContiguity(ptr);
    unsigned pointeeBitWidth =
        isTensorPointerType(ptr.getType())
            ? tensorTy.getElementType().getIntOrFloatBitWidth()
            : triton::getPointeeBitWidth(tensorTy);
    // The maximum vector size is 128 bits.
    return std::min<unsigned>(128 / pointeeBitWidth, contiguity);
  }

  unsigned getMaskAlignment(Value mask) const {
    return const_cast<triton::intel::ModuleAxisInfoAnalysis &>(axisAnalysisPass)
        .getMaskAlignment(mask);
  }

  std::tuple<SmallVector<Value>, SmallVector<Value>, SmallVector<Value>>
  convertBlockPtrToTensorOfPtr(
      Location loc, Value blockPointerStruct, RankedTensorType tensorType,
      Type valueElemTy, ConversionPatternRewriter &rewriter,
      ArrayRef<int32_t> boundaryCheck = {},
      std::optional<PaddingOption> padding = std::nullopt) const {

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    size_t rank = tensorType.getRank();
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
    auto indices = emitIndices(loc, rewriter, targetInfo,
                               tensorType.getEncoding(), tensorType, true);

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
        indicesInTensor[j] = b.add(index[j], blockPtr[blockOffset + j]);
      }

      // Get the LLVM values for pointers
      Value offset = linearize(
          indicesInTensor,
          {blockPtr.begin() + blockStride, blockPtr.begin() + blockBase},
          b.i32_val(0),
          [&](const Value &index, const Value &stride, const Value &off) {
            // off = off + index * stride
            return b.add(b.mul(index, b.trunc(i32_ty, stride)), off);
          });

      ptrElems[i] = b.gep(ptr_ty(rewriter.getContext(), 1 /*global*/),
                          valueElemTy, blockPtr[blockBase], offset);

      if (boundaryCheck.size() > 0) {
        // Get the LLVM values for mask
        maskElems.push_back(linearize(
            indicesInTensor,
            {blockPtr.begin() + blockShape, blockPtr.begin() + blockStride},
            b.int_val(1, 1),
            [&](const Value &index, const Value &shape, const Value &mask) {
              // mask = mask && (index < shape) && idx >= 0
              auto is_pos_idx = b.icmp_sge(index, b.int_val(32, 0));
              return b.and_(
                  b.and_(b.icmp_slt(index, b.trunc(i32_ty, shape)), mask),
                  is_pos_idx);
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
  const triton::intel::ModuleAxisInfoAnalysis &axisAnalysisPass;
  const triton::intel::TargetInfo &targetInfo;
};

struct PrefetchOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::gpu::intel::PrefetchOp>,
      public LoadStoreConversionBase {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::gpu::intel::PrefetchOp>::ConvertTritonGPUOpToLLVMPattern;

  PrefetchOpConversion(
      TritonGPUToLLVMTypeConverter &converter,
      const triton::intel::TargetInfo &targetInfo,
      const triton::intel::ModuleAxisInfoAnalysis &axisAnalysisPass,
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
    auto b = TritonLLVMOpBuilder(loc, rewriter);
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

    unsigned numWarps = triton::gpu::lookupNumWarps(op);

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

    baseWidth = b.mul(baseWidth, b.i64_val(eltTy.getIntOrFloatBitWidth() / 8));
    baseWidth = b.trunc(i32_ty, baseWidth);

    baseHeight = b.trunc(i32_ty, baseHeight);

    Value rowStrideInBytes =
        b.mul(rowStride, b.i64_val(eltTy.getIntOrFloatBitWidth() / 8));
    rowStrideInBytes = b.trunc(i32_ty, rowStrideInBytes);

    for (int row = 0; row < numReps[0]; ++row) {
      for (int col = 0; col < numReps[1]; ++col) {
        Value offsetX, offsetY;
        offsetX = b.add(
            // the offset of this warp.
            b.mul(multiDimWarpId[1], b.i32_val(shapePerWarp[1])),
            // add the replica offset with a warp stride.
            b.i32_val(col * warpsPerCTA[1] * shapePerWarp[1]));
        // Round the offset into to the tensor shape
        offsetX = b.urem(offsetX, b.i32_val(tensorShape[1]));
        offsetX = b.add(offsetX, offsetBaseX);
        offsetY = b.add(
            // the offset of this warp.
            b.mul(multiDimWarpId[0], b.i32_val(shapePerWarp[0])),
            // add the replica offset with a warp stride.
            b.i32_val(row * warpsPerCTA[0] * shapePerWarp[0]));
        // Round the offset into to the tensor shape
        offsetY = b.urem(offsetY, b.i32_val(tensorShape[0]));
        offsetY = b.add(offsetY, offsetBaseY);

        auto newOp = rewriter.create<TritonGEN::Matrix2DBlockPrefetchOp>(
            loc,
            /*ptr*/ base,
            /*base_width*/ baseWidth,
            /*base_height*/ baseHeight,
            /*base_pitch*/ rowStrideInBytes,
            /*x*/ b.trunc(i32_ty, offsetX),
            /*y*/ b.trunc(i32_ty, offsetY),
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

  LoadOpConversion(
      TritonIntelGPUToLLVMTypeConverter &converter,
      const triton::intel::TargetInfo &targetInfo,
      const triton::intel::ModuleAxisInfoAnalysis &axisAnalysisPass,
      PatternBenefit benefit, bool oneMatrixPerLoadForBT)
      : ConvertTritonGPUOpToLLVMPattern<triton::LoadOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass),
        oneMatrixPerLoadForBT(oneMatrixPerLoadForBT) {}

  LogicalResult
  rewriteTensorPointerLoad(triton::LoadOp op, OpAdaptor adaptor,
                           ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value ptr = op.getPtr();
    Value mask = op.getMask();
    Value other = op.getOther();
    Type resultType = op.getType();
    auto tensorType = cast<RankedTensorType>(resultType);

    // Only lower loadOp with dpas layout encoding.
    auto encoding = tensorType.getEncoding();
    const bool hasDpasLayout = isa<DpasEncodingAttr>(encoding);
    if (!hasDpasLayout && !hasDotDpasEncoding(tensorType))
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

    auto getOpIdx = [&]() -> DpasEncodingAttr::OpIdx {
      if (hasDpasLayout) {
        return DpasEncodingAttr::OpIdx::OperandC;
      } else {
        auto dotLayout = getDotEncoding(tensorType).value();
        return static_cast<DpasEncodingAttr::OpIdx>(dotLayout.getOpIdx());
      }
    };
    auto opIdx = getOpIdx();

    LLVM_DEBUG(llvm::dbgs() << "Tensor type for op " << int(opIdx) << ": "
                            << tensorType << "\n");

    std::optional<LinearLayout> llEncoding =
        cast<DistributedEncodingTrait>(encoding).toLinearLayout(
            tensorType.getShape());
    assert(llEncoding.has_value() && "invalid dot layout to linear layout");
    LinearEncodingAttr llAttr =
        LinearEncodingAttr::get(rewriter.getContext(), *llEncoding);
    SmallVector<unsigned> threadOrder = llAttr.getThreadOrder();
    size_t rank = threadOrder.size();
    const bool valueRowMajor =
        (threadOrder[rank - 2] == 1 && threadOrder[rank - 1] == 0);
    assert((valueRowMajor ||
            (threadOrder[rank - 2] == 0 && threadOrder[rank - 1] == 1)) &&
           "Only row_major or column_major is allowed");
    const bool isTransposeRequired = valueRowMajor ^ memoryRowMajor;

    Type eltTy = tensorType.getElementType();
    unsigned elemSizeInBits = eltTy.getIntOrFloatBitWidth();

    auto dpasLayout = hasDpasLayout
                          ? cast<DpasEncodingAttr>(encoding)
                          : cast<DpasEncodingAttr>(
                                getDotEncoding(tensorType).value().getParent());

    const ArrayRef<int64_t> tensorShape = tensorType.getShape();
    unsigned numElems = getTotalElemsPerThread(resultType);
    SmallVector<int64_t> numReps =
        dpasLayout.getDPASRepetitions(tensorShape, opIdx);
    LLVM_DEBUG({
      llvm::dbgs() << "numReps: ";
      for (size_t i = 0; i < numReps.size(); i++)
        llvm::dbgs() << " " << numReps[i];
      llvm::dbgs() << "\n";
    });
    const SmallVector<unsigned> warpsPerCTA = dpasLayout.getWarpsPerCTA();
    SmallVector<unsigned> dpasWarpsOrder = triton::gpu::getOrder(dpasLayout);
    int threadsPerWarp = triton::gpu::getWarpSize(dpasLayout);

    Value warpId = rewriter.create<arith::IndexCastOp>(
        loc, i32_ty,
        rewriter.create<mlir::gpu::SubgroupIdOp>(loc, /*upperBound=*/nullptr));

    SmallVector<Value> multiDimWarpId =
        delinearize(rewriter, loc, warpId, warpsPerCTA, dpasWarpsOrder);

    if (hasDpasLayout) {
      // A block load with the DPAS layout but without the DotDpasLayout is
      // expected to follow the ordering of the DPAS output. For a 2D block
      // load, the rows are distributed across work items/SIMD lanes and the
      // column vectors are available for each work item to process. This layout
      // aligns to the DPAS layout as the DPAS operation output layout
      // distributes rows across work items.

      if (isTransposeRequired) {
        // TODO: this would likely require a shuffle to match the expected
        // ordering coming out of the DPAS layout and requires more
        // investigation
        return failure();
      }

      MLIRContext *ctx = rewriter.getContext();

      Value elemSizeInBytes = b.i32_val(elemSizeInBits / 8);

      SmallVector<unsigned> elemsPerInstr = dpasLayout.getDPASInstShapeC();
      int64_t elemsPerLane = product<unsigned>(elemsPerInstr) / threadsPerWarp;
      Type load2DGenXType =
          LLVM::getFixedVectorType(IntegerType::get(ctx, elemSizeInBits),
                                   elemsPerLane); // make it opaque type.

      auto [base, baseWidth, baseHeight, rowStride, colStride, offsetBaseX,
            offsetBaseY] =
          getValuesFromBlockPointerStruct(adaptor.getPtr(), rewriter);
      baseWidth = b.trunc(i32_ty, baseWidth);
      baseHeight = b.trunc(i32_ty, baseHeight);

      auto pitch = b.trunc(i32_ty, rowStride);

      SmallVector<unsigned> repClusterShape = dpasLayout.getShapeC();
      unsigned outerDimWarpNum =
          std::min<unsigned>(warpsPerCTA[rank - 2],
                             mlir::ceil<unsigned>(tensorShape[rank - 2],
                                                  repClusterShape[rank - 2]));
      unsigned innerDimWarpNum =
          std::min<unsigned>(warpsPerCTA[rank - 1],
                             mlir::ceil<unsigned>(tensorShape[rank - 1],
                                                  repClusterShape[rank - 1]));
      Value outerDimWarpId =
          b.urem(multiDimWarpId[rank - 2], b.i32_val(outerDimWarpNum));
      Value innerDimWarpId =
          b.urem(multiDimWarpId[rank - 1], b.i32_val(innerDimWarpNum));
      int64_t numRepOuter = numReps[1];
      int64_t numRepInner = numReps[2];

      std::array<unsigned, 2> replicaStride = {
          outerDimWarpNum * repClusterShape[rank - 2],
          innerDimWarpNum * repClusterShape[rank - 1]};
      std::array<unsigned, 2> warpStride = {repClusterShape[rank - 2],
                                            repClusterShape[rank - 1]};

      Value dimWarpId0 = b.mul(outerDimWarpId, b.i32_val(warpStride[0]));
      Value dimWarpId1 = b.mul(innerDimWarpId, b.i32_val(warpStride[1]));
      Value warpId0Offset = b.add(dimWarpId0, offsetBaseY);
      Value warpId1Offset = b.add(dimWarpId1, offsetBaseX);

      ArrayRef<unsigned> repCluster = dpasLayout.getRepCluster();
      unsigned valOffset = 0;

      SmallVector<Value> unpackedLoadedVals;

      for (int m = 0; m < numRepOuter; ++m) {
        for (int n = 0; n < numRepInner; ++n) {
          for (int repM = 0; repM < repCluster[0]; ++repM) {

            Value offsetY =
                b.add(warpId0Offset, b.i32_val(m * replicaStride[0] +
                                               repM * elemsPerInstr[0]));
            for (int repN = 0; repN < repCluster[1]; ++repN) {
              Value offsetX =
                  b.add(warpId1Offset, b.i32_val(n * replicaStride[1] +
                                                 repN * elemsPerInstr[1]));

              auto load2dOp = rewriter.create<TritonGEN::Matrix2DBlockLoadOp>(
                  loc, load2DGenXType,
                  /*ptr*/ base,
                  /*base_width*/ b.mul(baseWidth, elemSizeInBytes),
                  /*base_height*/ baseHeight,
                  /*base_pitch*/ b.mul(pitch, elemSizeInBytes),
                  /*x*/ b.trunc(i32_ty, offsetX),
                  /*y*/ b.trunc(i32_ty, offsetY),
                  /*elem_size_in_bits*/ elemSizeInBits,
                  /*tile_width*/ elemsPerInstr[1],
                  /*tile_height*/ elemsPerInstr[0],
                  /*v_blocks*/ 1,
                  /*transpose*/ false,
                  /*vnni_transform*/ false);
              if (failed(load2dOp.verify())) {
                // Explicitly invoke verifier because `triton_gen` ops are
                // immediately lowered further to a builtin call.
                return failure();
              }

              Value ret = b.bitcast(
                  load2dOp, LLVM::getFixedVectorType(eltTy, elemsPerLane));

              for (size_t i = 0; i < elemsPerLane; i++) {
                Value loaded = b.extract_element(eltTy, ret, b.i32_val(i));
                unpackedLoadedVals.push_back(loaded);
              }
            }
          }
        }
      }

      TritonGPUToLLVMTypeConverter *typeConverter = getTypeConverter();
      Type llvmResultStructTy = typeConverter->convertType(op.getType());
      Value resultStruct = packLLElements(
          loc, typeConverter, unpackedLoadedVals, rewriter, llvmResultStructTy);
      rewriter.replaceOp(op, {resultStruct});

      return success();
    }

    bool isOperandA = (opIdx == DpasEncodingAttr::OpIdx::OperandA);
    SmallVector<unsigned> dpasInstShape = isOperandA
                                              ? dpasLayout.getDPASInstShapeA()
                                              : dpasLayout.getDPASInstShapeB();
    const SmallVector<unsigned> elemsPerDPASInst = {dpasInstShape[0],
                                                    dpasInstShape[1]};
    LLVM_DEBUG(llvm::dbgs()
               << "Elements per DPAS Instruction: " << elemsPerDPASInst[0]
               << ", " << elemsPerDPASInst[1] << "\n");
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
    if ((opsPerChannel == 4 && elemSizeInBits == 8) ||
        (opsPerChannel == 2 && elemSizeInBits == 16) ||
        (opsPerChannel == 1 && elemSizeInBits == 32)) {
      loadResultElemType =
          (isOperandA && elemSizeInBits != 32) ? i16_ty : i32_ty;
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

    unsigned dimOuter = bool(opIdx) ? rank - 1 : rank - 2;
    unsigned dimInner = bool(opIdx) ? rank - 2 : rank - 1;
    unsigned outerDimRequiredWarpNum =
        mlir::ceil<unsigned>(tensorShape[dimOuter], warpShape[dimOuter]);
    unsigned outerDimWarpNum =
        std::min<unsigned>(warpsPerCTA[dimOuter], outerDimRequiredWarpNum);
    Value outerDimWarpId =
        b.urem(multiDimWarpId[dimOuter], b.i32_val(outerDimWarpNum));

    auto [base, baseWidth, baseHeight, rowStride, colStride, offsetBaseX,
          offsetBaseY] =
        getValuesFromBlockPointerStruct(adaptor.getPtr(), rewriter);

    unsigned tileWidth = elemsPerDPASInst[threadOrder[rank - 2]];
    unsigned tileHeight = elemsPerDPASInst[threadOrder[rank - 1]];
    unsigned vBlocks = 1;
    unsigned numOperandsOuterDimPerLoad = 1;
    unsigned numOperandsInnerDimPerLoad = 1;

    LLVM_DEBUG({
      llvm::dbgs() << "dimOuter: " << dimOuter << "\n";
      llvm::dbgs() << "dimInner: " << dimInner << "\n";
      llvm::dbgs() << "width (elemsPerDPASInst[" << threadOrder[rank - 2]
                   << "]): " << elemsPerDPASInst[threadOrder[rank - 2]] << "\n";
      llvm::dbgs() << "height: (elemsPerDPASInst[" << threadOrder[rank - 1]
                   << "]): " << elemsPerDPASInst[threadOrder[rank - 1]] << "\n";
    });

    MLIRContext *ctx = rewriter.getContext();
    auto dimOuterStr = S("dim" + std::to_string(dimOuter));
    auto dimInnerStr = S("dim" + std::to_string(dimInner));

    // Create the linear layout for the load.
    // First, we create a tile layout corresponding to a single invocation of
    // the DPAS instruction across all threads/work-items in a sub-group. The
    // layout will later be expanded to cover multiple DPAS invocations
    // (iteration) and multiple loads (load).
    StringAttr kOffset = S("offset");
    StringAttr kIteration = S("iteration");
    StringAttr kLoad = S("load");
    auto createTileLayout = [&](const SmallVector<unsigned> &threadOrder,
                                SmallVector<unsigned> tileShape) {
      auto outDimNames = standardOutDimNames(ctx, tensorShape.size());
      LinearLayout layout = LinearLayout::empty();
      SmallVector<StringAttr> kOffsetDims;
      auto totalOffsets = 1;
      assert(tileShape.size() == 2); // only support 2D layouts for now

      if (isTransposeRequired && opIdx == DpasEncodingAttr::OpIdx::OperandB) {
        const auto widthDim = threadOrder[rank - 2];
        const auto origTileWidth = tileShape[widthDim];
        tileShape[widthDim] = origTileWidth / (32 / elemSizeInBits);
      }

      for (int i = 0; i < tileShape.size(); i++) {
        int dim = threadOrder[i];
        StringAttr kOffset = S("offset" + std::to_string(dim));

        kOffsetDims.push_back(kOffset);

        assert(llvm::isPowerOf2_32(tileShape[dim]));
        layout *=
            LinearLayout::identity1D(tileShape[dim], kOffset, outDimNames[dim]);
        totalOffsets *= tileShape[dim];
      }
      SmallVector<StringAttr> newDims;
      newDims.append(kOffsetDims.begin(), kOffsetDims.end());
      auto ret = layout.transposeIns(newDims);
      ret = ret.transposeOuts(outDimNames);
      return ret.reshapeIns({{kOffset, totalOffsets}});
    };
    auto tileLayout = createTileLayout(threadOrder, elemsPerDPASInst);

    LLVM_DEBUG({
      llvm::dbgs() << "Block load tile layout: " << tileLayout << "\n";
      for (size_t i = 0; i < tileLayout.getOutDimSize(dimOuterStr) *
                                 tileLayout.getOutDimSize(dimInnerStr);
           i++) {
        auto tensorVals = tileLayout.apply({{kOffset, i}});
        assert(tensorVals.size() == 2);
        llvm::dbgs() << i << " : " << tensorVals[0].second << ", "
                     << tensorVals[1].second << "\n";
      }
      llvm::dbgs() << "tile layout done\n";
    });

    unsigned numOperandsPer2DLoadM, numOperandsPer2DloadN;
    if (!isTransposeRequired) {
      numOperandsPer2DLoadM =
          isOperandA ? repCluster[dimOuter] : numReps[unsigned(opIdx) ? 1 : 2];
      numOperandsPer2DloadN =
          isOperandA ? numReps[unsigned(opIdx) ? 1 : 2] : repCluster[dimOuter];
    } else {
      if (isOperandA)
        return failure();

      if (!usePackedType)
        return failure();

      std::swap(tileHeight, tileWidth);

      if (oneMatrixPerLoadForBT) {
        // Only load 1 operand per inst on row.
        numOperandsPer2DLoadM = 1;
      } else {
        // We can decompose the matrix returned by transposed large 2d load
        // when threads per warp < column size. Otherwise we have to load one
        // operand per inst.
        // Note: the tileHeight and numOperandsPer2DLoadM are the column size
        // now.
        numOperandsPer2DLoadM =
            (threadsPerWarp <= tileHeight) ? repCluster[rank - 1] : 1;
      }
      // The transpose 2d load only support 1 operand per inst on column.
      // (vBlocks = 1)
      numOperandsPer2DloadN = 1;
    }

    // PVC 2D load supports 32 rows at most. Load multiple dot operands in by
    // enlarging the tileHeight.
    if (numOperandsPer2DLoadM > (32 / tileHeight)) {
      llvm::errs() << "replacing " << numOperandsPer2DLoadM << " with "
                   << 32 / tileHeight << "\n";
    }
    numOperandsPer2DLoadM = std::min(numOperandsPer2DLoadM, 32 / tileHeight);
    tileHeight = tileHeight * numOperandsPer2DLoadM;

    // PVC 2D load supports 64 bytes per row at most. Load multiple dot operands
    // by enlarging the vBlocks.
    unsigned totalBytesPerRowPerDPASOp = tileWidth * elemSizeInBits / 8;
    const auto origNumOperandsPer2DloadN = numOperandsPer2DloadN;
    if (numOperandsPer2DloadN > 64 / totalBytesPerRowPerDPASOp)
      llvm::errs() << "enlarge vblocks : " << 64 / totalBytesPerRowPerDPASOp
                   << " vs " << numOperandsPer2DloadN << "\n";
    numOperandsPer2DloadN =
        std::min(numOperandsPer2DloadN, 64 / totalBytesPerRowPerDPASOp);
    vBlocks = numOperandsPer2DloadN;

    numOperandsOuterDimPerLoad =
        isOperandA ? numOperandsPer2DLoadM : numOperandsPer2DloadN;
    numOperandsInnerDimPerLoad =
        isOperandA ? numOperandsPer2DloadN : numOperandsPer2DLoadM;

    // vblocks are not represented using the load layout
    auto numIterationsOuterDimPerLoad =
        isOperandA ? numOperandsPer2DLoadM : origNumOperandsPer2DloadN;
    auto numIterationsInnerDimPerLoad =
        isOperandA ? numOperandsPer2DloadN : numOperandsPer2DLoadM;

    if (!isTransposeRequired) {
#if 0
      tileLayout *= LinearLayout::identity1D(numIterationsOuterDimPerLoad,
        kIteration, S("dim0"));
      tileLayout *= LinearLayout::identity1D(numIterationsInnerDimPerLoad,
        kIteration, S("dim1"));
#else
      tileLayout *= LinearLayout::identity1D(numIterationsOuterDimPerLoad,
                                             kIteration, dimOuterStr);
      tileLayout *= LinearLayout::identity1D(numIterationsInnerDimPerLoad,
                                             kIteration, dimInnerStr);
#endif

      llvm::errs() << "\nnumOperandsOuterDimPerLoad: "
                   << numOperandsOuterDimPerLoad << "\n";
      llvm::errs() << "numOperandsInnerDimPerLoad: "
                   << numOperandsInnerDimPerLoad << "\n";
      llvm::errs() << "numIterationsOuterDimPerLoad: "
                   << numIterationsOuterDimPerLoad << "\n";
      llvm::errs() << "numIterationsInnerDimPerLoad: "
                   << numIterationsInnerDimPerLoad << "\n";
      llvm::errs() << "repCluster[dimOuter]: " << repCluster[dimOuter] << "\n";
      llvm::errs() << "repCluster[dimInner]: " << repCluster[dimInner] << "\n";
      llvm::errs() << "numReps[unsigned(opIdx) ? 1 : 2]: "
                   << numReps[unsigned(opIdx) ? 1 : 2] << "\n\n";
    } else {
      if (isOperandA)
        return failure();

      if (!usePackedType)
        return failure();

      if (oneMatrixPerLoadForBT) {
        // Only load 1 operand per inst on row.
        tileLayout *= LinearLayout::identity1D(1, kIteration, dimOuterStr);
      } else {
        // We can decompose the matrix returned by transposed large 2d load
        // when threads per warp < column size. Otherwise we have to load one
        // operand per inst.
        // Note: the tileHeight and numOperandsPer2DLoadM are the column size
        // now.
        llvm::errs() << "\n"
                     << std::to_string((threadsPerWarp <= tileHeight)) << " ? "
                     << repCluster[dimOuter] << " : " << 1 << "\n";
        llvm::errs() << "numOperandsOuterDimPerLoad: "
                     << numOperandsOuterDimPerLoad << "\n";
        llvm::errs() << "numOperandsInnerDimPerLoad: "
                     << numOperandsInnerDimPerLoad << "\n";
        llvm::errs() << "numIterationsOuterDimPerLoad: "
                     << numIterationsOuterDimPerLoad << "\n";
        llvm::errs() << "numIterationsInnerDimPerLoad: "
                     << numIterationsInnerDimPerLoad << "\n";
        tileLayout *= LinearLayout::identity1D(
            (threadsPerWarp <= tileHeight) ? repCluster[dimOuter] : 1,
            kIteration, dimOuterStr);
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "Block load tile layout after adding iterations: "
                   << tileLayout << "\n";
      for (size_t itr = 0; itr < tileLayout.getInDimSize(kIteration); itr++) {
        {
          size_t offset = 0;
          auto tensorVals =
              tileLayout.apply({{kOffset, offset}, {kIteration, itr}});
          assert(tensorVals.size() == 2);
          llvm::dbgs() << itr << ", " << offset << " : " << tensorVals[0].second
                       << ", " << tensorVals[1].second << "\n";
        }
        {
          size_t offset = 1;
          auto tensorVals =
              tileLayout.apply({{kOffset, offset}, {kIteration, itr}});
          assert(tensorVals.size() == 2);
          llvm::dbgs() << itr << ", " << offset << " : " << tensorVals[0].second
                       << ", " << tensorVals[1].second << "\n";
        }
        {
          size_t offset = tileLayout.getInDimSize(kOffset) - 2;
          auto tensorVals =
              tileLayout.apply({{kOffset, offset}, {kIteration, itr}});
          assert(tensorVals.size() == 2);
          llvm::dbgs() << itr << ", " << offset << " : " << tensorVals[0].second
                       << ", " << tensorVals[1].second << "\n";
        }
        {
          size_t offset = tileLayout.getInDimSize(kOffset) - 1;
          auto tensorVals =
              tileLayout.apply({{kOffset, offset}, {kIteration, itr}});
          assert(tensorVals.size() == 2);
          llvm::dbgs() << itr << ", " << offset << " : " << tensorVals[0].second
                       << ", " << tensorVals[1].second << "\n";
        }
      }
      llvm::dbgs() << "\n";
    });

    unsigned numRepOuter = numReps[bool(opIdx) ? 2 : 1];
    unsigned numRepInner = numReps[bool(opIdx) ? 1 : 2];

    const unsigned packedElementsPerSlot =
        isTransposeRequired ? tileHeight / elemsPerDPASInst[dimInner] : 1;
    LLVM_DEBUG(llvm::dbgs() << "Packed elements per slot: "
                            << packedElementsPerSlot << "\n");

    if (isTransposeRequired) {
      llvm::errs() << "transpose required!\n";
      std::swap(numOperandsOuterDimPerLoad, numOperandsInnerDimPerLoad);
      std::swap(numIterationsOuterDimPerLoad, numIterationsInnerDimPerLoad);
    }

#if 0
    llvm::errs() << "dim outer load: \n";
    llvm::errs() << "\tnumRepOuter: " << numRepOuter << "\n";
    llvm::errs() << "\tpackedElementsPerSlot: " << packedElementsPerSlot
                 << "\n";
    llvm::errs() << "\tvblocks: " << vBlocks << "\n";
    llvm::errs() << "\tnumRepOuter * packedElementsPerSlot * vBlocks: "
                 << numRepOuter * packedElementsPerSlot * vBlocks << "\n";
    llvm::errs() << "\tnumIterationsOuterDimPerLoad: "
                 << numIterationsOuterDimPerLoad << "\n";
    llvm::errs() << "\tdimOuter Val: "
                 << (numRepOuter * packedElementsPerSlot * vBlocks) /
                        numIterationsOuterDimPerLoad
                 << "\n";
    llvm::errs() << "\tdimOuter Val swapped iterations: "
                 << (numRepOuter * packedElementsPerSlot * vBlocks) /
                        numIterationsInnerDimPerLoad
                 << "\n";

    llvm::errs() << "dim inner load: \n";
    llvm::errs() << "\tnumRepInner: " << numRepInner << "\n";
    llvm::errs() << "\tnumIterationsInnerDimPerLoad: "
                 << numIterationsInnerDimPerLoad << "\n";
    llvm::errs() << "\tdimInner Val: "
                 << numRepInner / numIterationsInnerDimPerLoad << "\n";
    llvm::errs() << "\tdimInner Val swapped iterations: "
                 << numRepInner / numIterationsOuterDimPerLoad << "\n";
#endif

    tileLayout *= LinearLayout::identity1D(
        numRepInner / numIterationsInnerDimPerLoad, kLoad, dimInnerStr);
    if (isTransposeRequired && oneMatrixPerLoadForBT) {
      tileLayout *= LinearLayout::identity1D(numRepOuter * repCluster[dimOuter],
                                             kLoad, dimOuterStr);
    } else {
      tileLayout *= LinearLayout::identity1D(
          (numRepOuter * packedElementsPerSlot * vBlocks) /
              numIterationsOuterDimPerLoad,
          kLoad, dimOuterStr);
    }

    LLVM_DEBUG({
      llvm::dbgs() << "Block load tile layout after adding loads: "
                   << tileLayout << "\n";
      for (size_t load = 0; load < tileLayout.getInDimSize(kLoad); load++) {
        for (size_t itr = 0; itr < tileLayout.getInDimSize(kIteration); itr++) {
          {
            size_t offset = 0;
            auto tensorVals = tileLayout.apply(
                {{kOffset, offset}, {kIteration, itr}, {kLoad, load}});
            assert(tensorVals.size() == 2);
            llvm::dbgs() << load << ", " << itr << ", " << offset << " : "
                         << tensorVals[0].second << ", " << tensorVals[1].second
                         << "\n";
          }
          {
            size_t offset = 1;
            auto tensorVals = tileLayout.apply(
                {{kOffset, offset}, {kIteration, itr}, {kLoad, load}});
            assert(tensorVals.size() == 2);
            llvm::dbgs() << load << ", " << itr << ", " << offset << " : "
                         << tensorVals[0].second << ", " << tensorVals[1].second
                         << "\n";
          }
          {
            size_t offset = tileLayout.getInDimSize(kOffset) - 2;
            auto tensorVals = tileLayout.apply(
                {{kOffset, offset}, {kIteration, itr}, {kLoad, load}});
            assert(tensorVals.size() == 2);
            llvm::dbgs() << load << ", " << itr << ", " << offset << " : "
                         << tensorVals[0].second << ", " << tensorVals[1].second
                         << "\n";
          }
          {
            size_t offset = tileLayout.getInDimSize(kOffset) - 1;
            auto tensorVals = tileLayout.apply(
                {{kOffset, offset}, {kIteration, itr}, {kLoad, load}});
            assert(tensorVals.size() == 2);
            llvm::dbgs() << load << ", " << itr << ", " << offset << " : "
                         << tensorVals[0].second << ", " << tensorVals[1].second
                         << "\n";
          }
        }
        llvm::dbgs() << "\n";
      }
    });

    unsigned numLoadPerOutRepCluster =
        mlir::ceil<unsigned>(repCluster[dimOuter], numOperandsOuterDimPerLoad);

    unsigned numValuesPerLoad = packedElemsPerLanePerDPASInst *
                                numOperandsOuterDimPerLoad *
                                numOperandsInnerDimPerLoad;

    Type load2DGenXType =
        LLVM::getFixedVectorType(loadResultElemType, numValuesPerLoad);

    // The stride for the replicates.
    unsigned repOuterStride = warpShape[dimOuter] * outerDimWarpNum;
    unsigned repStride =
        elemsPerDPASInst[dimOuter] * numOperandsOuterDimPerLoad;
    unsigned warpOuterStride = warpShape[dimOuter];
    unsigned repKStride = elemsPerDPASInst[dimInner];

    Value pitch;
    if (memoryRowMajor) {
      pitch = b.trunc(i32_ty, rowStride);
    } else {
      // Column major memory. We need to swap the width and height because HW
      // only support row major memory layout.
      pitch = b.trunc(i32_ty, colStride);
      std::swap(baseWidth, baseHeight);
    }
    baseWidth = b.trunc(i32_ty, baseWidth);
    baseHeight = b.trunc(i32_ty, baseHeight);

    const unsigned originalElemBits = elemSizeInBits;
    if (isTransposeRequired) {
      // adjust the block io parameter to align HW's limitations on
      // transposing load.
      tileWidth = tileWidth / (32 / originalElemBits);
      elemSizeInBits = 32;
    }
    Value elemSizeInBytes = b.i32_val(originalElemBits / 8);

    auto ll = *llEncoding;

    StringAttr kRegister = StringAttr::get(ctx, "register");
    StringAttr kLane = StringAttr::get(ctx, "lane");
    StringAttr kWarp = StringAttr::get(ctx, "warp");
    StringAttr kBlock = StringAttr::get(ctx, "block");

    LLVM_DEBUG(llvm::dbgs() << "DPAS Linear Layout: " << ll << "\n");

    LLVM_DEBUG({
      llvm::dbgs() << "tileWidth: " << tileWidth << "\n";
      llvm::dbgs() << "tileHeight: " << tileHeight << "\n";

      llvm::dbgs() << "offset dim size: " << tileLayout.getInDimSize(kOffset)
                   << "\n";
      llvm::dbgs() << "iteration dim size: "
                   << tileLayout.getInDimSize(kIteration) << "\n";

      llvm::dbgs() << "ll lane size: " << ll.getInDimSize(kLane) << "\n";
      llvm::dbgs() << "ll warp size: " << ll.getInDimSize(kWarp) << "\n";
      llvm::dbgs() << "ll register size: " << ll.getInDimSize(kRegister)
                   << "\n";
      llvm::dbgs() << "ll block size: " << ll.getInDimSize(kBlock) << "\n";

      llvm::dbgs() << "outerDimWarpNum: " << outerDimWarpNum << "\n";
      llvm::dbgs() << "originalElemBits: " << originalElemBits << "\n";
      llvm::dbgs() << "elemSizeInBits: " << elemSizeInBits << "\n";
    });

    LLVM_DEBUG({
      llvm::dbgs() << "numRepOuter: " << numRepOuter << "\n";
      llvm::dbgs() << "numLoadPerOutRepCluster: " << numLoadPerOutRepCluster
                   << "\n";
      llvm::dbgs() << "numOperandsOuterDimPerLoad: "
                   << numOperandsOuterDimPerLoad << "\n";
      llvm::dbgs() << "numRepInner: " << numRepInner << "\n";
      llvm::dbgs() << "numOperandsInnerDimPerLoad: "
                   << numOperandsInnerDimPerLoad << "\n";
    });

    LLVM_DEBUG({
      llvm::dbgs() << "\nrepOuterStride: " << repOuterStride << "\n";
      llvm::dbgs() << "repStride: " << repStride << "\n";
      llvm::dbgs() << "warpShape: (outer, inner) " << warpShape[dimOuter]
                   << ", " << warpShape[dimInner] << "\n";
      llvm::dbgs() << "outerDimWarpNum: " << outerDimWarpNum << "\n";
    });

    ValueTable loadVals;
    for (int outer = 0; outer < numRepOuter; ++outer) {
      for (int rep = 0; rep < numLoadPerOutRepCluster; ++rep) {
        for (int k = 0; k < numRepInner; k += numOperandsInnerDimPerLoad) {
          LLVM_DEBUG({
            llvm::dbgs() << "outer, rep, k: " << outer << ", " << rep << ", "
                         << k << "\n";
          });

          const int loadIdx = (outer * numLoadPerOutRepCluster *
                               (numRepInner / numOperandsInnerDimPerLoad)) +
                              rep * (numRepInner / numOperandsInnerDimPerLoad) +
                              k / numOperandsInnerDimPerLoad;
          LLVM_DEBUG(llvm::dbgs() << "loadIdx: " << loadIdx << "\n");

          Value offsetX, offsetY;
          auto offset = tileLayout.apply(
              {{kOffset, 0}, {kIteration, 0}, {kLoad, loadIdx}});
          assert(offset.size() == 2);
          // adjust the load offset to compensate for strides related to the
          // DPAS layout
          LLVM_DEBUG({
            llvm::dbgs() << "x offset from layout: " << offset[0].second
                         << "\n";
            llvm::dbgs() << "y offset from layout: " << offset[1].second
                         << "\n";
          });
          llvm::errs() << "outer dim warp num: " << outerDimWarpNum << "\n";
          llvm::errs() << "iteration num: "
                       << tileLayout.getInDimSize(kIteration) << "\n";
          llvm::errs() << "load num: " << tileLayout.getInDimSize(kLoad)
                       << "\n";
          llvm::errs() << "numLoadPerOutRepCluster: " << numLoadPerOutRepCluster
                       << "\n";
          llvm::errs() << "packedElementsPerSlot: " << packedElementsPerSlot
                       << "\n";
          const auto loadOffsetX =
              isOperandA ? offset[1].second
                         : offset[1].second *
                               (tileLayout.getInDimSize(kIteration) / vBlocks) *
                               tileLayout.getInDimSize(kLoad);
          llvm::errs() << "itr size: " << tileLayout.getInDimSize(kIteration)
                       << "\n";
          llvm::errs() << "itr / vblocks: "
                       << tileLayout.getInDimSize(kIteration) / vBlocks << "\n";
          llvm::errs() << "load size: " << tileLayout.getInDimSize(kLoad)
                       << "\n";
          llvm::errs() << "numLoadPerOutRepCluster: " << numLoadPerOutRepCluster
                       << "\n";
          const auto loadOffsetY = offset[0].second;
          LLVM_DEBUG({
            llvm::dbgs() << "isOperandA? " << isOperandA << "\n";
            llvm::dbgs() << "x offset ll: " << loadOffsetX << "\n";
            llvm::dbgs() << "y offset ll: " << loadOffsetY << "\n";
          });

          switch (opIdx) {
          case DpasEncodingAttr::OpIdx::OperandA: {
            LLVM_DEBUG({
              llvm::dbgs() << "x offset: " << k * repKStride << "\n";
              llvm::dbgs() << "y offset: "
                           << outer * repOuterStride + rep * repStride << "\n";
            });
            offsetY = b.add(b.mul(outerDimWarpId, b.i32_val(warpOuterStride)),
                            b.i32_val(loadOffsetY));
            offsetX = b.i32_val(loadOffsetX);
          } break;
          case DpasEncodingAttr::OpIdx::OperandB: {
            LLVM_DEBUG({
              llvm::dbgs() << "x offset: "
                           << outer * repOuterStride + rep * repStride << "\n";
              llvm::dbgs() << "y offset: " << k * repKStride << "\n";
            });
            offsetX = b.add(b.mul(outerDimWarpId, b.i32_val(warpOuterStride)),
                            b.i32_val(loadOffsetX));
            offsetY = b.i32_val(loadOffsetY);
          } break;
          case DpasEncodingAttr::OpIdx::OperandC: {
            llvm_unreachable("unexpected OpIdx::OperandC");
          } break;
          }

          offsetX = b.add(offsetX, offsetBaseX);
          offsetY = b.add(offsetY, offsetBaseY);

          if (!memoryRowMajor) {
            // Column major memory. We need to swap the X and Y because HW only
            // support row major memory layout.
            std::swap(offsetX, offsetY);
          }

          if (isTransposeRequired) {
            // adjust the block io parameter to align HW's limitations on
            // transposing load.
            offsetX = b.udiv(offsetX, b.i32_val(32 / originalElemBits));
          }

          auto load2dOp = rewriter.create<TritonGEN::Matrix2DBlockLoadOp>(
              loc, load2DGenXType,
              /*ptr*/ base,
              /*base_width*/ b.mul(baseWidth, elemSizeInBytes),
              /*base_height*/ baseHeight,
              /*base_pitch*/ b.mul(pitch, elemSizeInBytes),
              /*x*/ b.trunc(i32_ty, offsetX),
              /*y*/ b.trunc(i32_ty, offsetY),
              /*elem_size_in_bits*/ elemSizeInBits,
              /*tile_width*/ tileWidth,
              /*tile_height*/ tileHeight,
              /*v_blocks*/ vBlocks,
              /*transpose*/ isTransposeRequired,
              /*vnni_transform*/
              (usePackedType && !isOperandA && !isTransposeRequired &&
               originalElemBits != 32));
          if (failed(load2dOp.verify())) {
            // Explicitly invoke verifier because `triton_gen` ops are
            // immediately lowered further to a builtin call.
            return failure();
          }
          LLVM_DEBUG(llvm::dbgs() << "Generated load op: " << load2dOp << "\n");

          const auto loadRowStride = (isTransposeRequired && !isOperandA)
                                         ? 0
                                         : packedElemsPerLanePerDPASInst;
          ;

          unsigned loadColStride;
          if (isTransposeRequired && !isOperandA) {
            loadColStride = (ll.getOutDimSize(dimOuterStr) >= tileHeight)
                                ? tileHeight / elemsPerDPASInst[dimOuter] - 1
                                : tileHeight;
          } else {
            llvm::errs() << "tileWidth = " << tileWidth << "\n";
            llvm::errs() << "elemSizeInBits/16 = " << elemSizeInBits / 16
                         << "\n";
#if 1
            loadColStride = isOperandA ? tileHeight : tileWidth;
#else
            loadColStride = isOperandA ? tileHeight / (elemSizeInBits / 16)
                                       : tileWidth / (elemSizeInBits / 16);
#endif
          }

          LLVM_DEBUG({
            llvm::dbgs() << "row stride: " << loadRowStride << "\n";
            llvm::dbgs() << "col stride: " << loadColStride << "\n";
          });

          auto itrOffsetCoord_ = tileLayout.apply(
              {{kLoad, loadIdx}, {kOffset, 0}, {kIteration, 0}});
          assert(itrOffsetCoord_.size() == 2);
          LLVM_DEBUG(llvm::dbgs()
                     << "itrOffsetCoord_: " << itrOffsetCoord_[0].second << ", "
                     << itrOffsetCoord_[1].second << "\n");

          for (size_t i = 0; i < tileLayout.getInDimSize(kIteration); i++) {
            LLVM_DEBUG(llvm::dbgs() << "Emitting shuffle vector for iteration "
                                    << i << ", load: " << loadIdx << "\n");

#if 0

            // Use the tile layout to determine the starting coordinate for
            // this iteration in a generic load
            auto itrOffsetCoord =
                tileLayout.apply({{kLoad, 0}, {kOffset, 0}, {kIteration, i}});
            assert(itrOffsetCoord.size() == 2);
            LLVM_DEBUG(llvm::dbgs()
                       << "itrOffsetCoord: " << itrOffsetCoord[0].second << ", "
                       << itrOffsetCoord[1].second << "\n");
            auto itrOffsetRowCoord =
                itrOffsetCoord[0].second / elemsPerDPASInst[0];
            auto itrOffsetColCoord = itrOffsetCoord[1].second / tileWidth;
            LLVM_DEBUG({
              llvm::dbgs() << "row: " << itrOffsetRowCoord << "\n";
              llvm::dbgs() << "col: " << itrOffsetColCoord << "\n";
            });

#if 0
            const auto itrRowOffset = itrOffsetRowCoord * loadColStride;
            const auto itrColOffset = itrOffsetColCoord * loadRowStride;
#else
            const auto itrRowOffset = itrOffsetRowCoord * loadRowStride;
            const auto itrColOffset = itrOffsetColCoord * loadColStride;
#endif
            LLVM_DEBUG({
              llvm::dbgs() << "row offset: " << itrRowOffset << "\n";
              llvm::dbgs() << "col offset: " << itrColOffset << "\n";
            });
#endif

            SmallVector<int32_t> indices(packedElemsPerLanePerDPASInst);
            for (int elemIdx = 0; elemIdx < packedElemsPerLanePerDPASInst;
                 elemIdx += packedElementsPerSlot) {

              auto shuffleVectorOffset = tileLayout.apply(
                  {{kOffset, elemIdx * tileWidth * packedElementsPerSlot},
                   {kIteration, i},
                   {kLoad, loadIdx}});
              assert(shuffleVectorOffset.size() == 2);
              LLVM_DEBUG(llvm::dbgs() << "\tshuffle vector offset: "
                                      << shuffleVectorOffset[0].second << ", "
                                      << shuffleVectorOffset[1].second << "\n");
              shuffleVectorOffset[0].second -= itrOffsetCoord_[0].second;
              shuffleVectorOffset[1].second -= itrOffsetCoord_[1].second;
              LLVM_DEBUG(llvm::dbgs()
                         << "\tshuffle vector offset after handing loads: "
                         << shuffleVectorOffset[0].second << ", "
                         << shuffleVectorOffset[1].second << "\n");

              for (size_t s = 0; s < packedElementsPerSlot; s++) {
                // TODO: this indexing is wrong, we don't care about
                // itrRowOffset or itrColOffset. we care about how many
                // iterations have come before us in this load, and what the
                // last offset was.
#if 1
                llvm::errs()
                    << "\tshuffle x coord: " << (shuffleVectorOffset[0].second)
                    << "\n";
                llvm::errs()
                    << "\tshuffle y coord: " << shuffleVectorOffset[1].second
                    << "\n";

                const auto x_offset =
                    (shuffleVectorOffset[0].second / elemsPerDPASInst[0]) *
                    packedElemsPerLanePerDPASInst;
                auto y_offset =
                    (shuffleVectorOffset[1].second * packedElementsPerSlot) /
                    elemsPerDPASInst[0];
                llvm::errs() << "\tshuffle y offset: " << y_offset << "\n";

#if 1
                y_offset *= isTransposeRequired ? 1 : tileHeight / opsPerChannel; 
#else
                y_offset *= isTransposeRequired ? 1 : (isOperandA ? elemsPerDPASInst[1] : packedElemsPerLanePerDPASInst);
#endif 
                llvm::errs() << "\tshuffle y stride: " << y_offset << "\n";

                llvm::errs()
                    << "\tshuffle slot offset: " << s * packedElementsPerSlot
                    << "\n";

                indices[elemIdx + s] =
                    x_offset + y_offset +
                    (shuffleVectorOffset[0].second %
                     (packedElementsPerSlot * packedElemsPerLanePerDPASInst)) +
                    s * packedElementsPerSlot;
#else
                indices[elemIdx + s] =
                    itrRowOffset + itrColOffset +
                    (shuffleVectorOffset[0].second %
                     (packedElementsPerSlot * packedElemsPerLanePerDPASInst)) +
                    s * packedElementsPerSlot;
#endif
                LLVM_DEBUG({
                  llvm::dbgs() << "indices[" << elemIdx + s << "]" << " = "
                               << indices[elemIdx + s] << "\n";
                });
              }
            }

            DenseI32ArrayAttr attr = rewriter.getDenseI32ArrayAttr(indices);
            Value loadVal = rewriter.create<LLVM::ShuffleVectorOp>(
                loc, packedDPASOperandType, load2dOp, load2dOp, attr);

            auto tensorCoord = tileLayout.apply(
                {{kLoad, loadIdx}, {kOffset, 0}, {kIteration, i}});
            assert(tensorCoord.size() == 2);
            LLVM_DEBUG(llvm::dbgs() << "tensorCoord: " << tensorCoord[0].second
                                    << ", " << tensorCoord[1].second << "\n");
            llvm::errs() << "packedElementsPerSlot: " << packedElementsPerSlot
                         << "\n";
            llvm::errs() << "packedElemsPerLanePerDPASInst: " << packedElemsPerLanePerDPASInst << "\n";
            llvm::errs() << "elemsPerLanePerDPASInst: " << elemsPerLanePerDPASInst << "\n";
            llvm::errs() << "opsPerChannel: " << opsPerChannel << "\n";
            llvm::errs() << "elemsPerDPASInst: " << elemsPerDPASInst[0] << ", "
                         << elemsPerDPASInst[1] << "\n";
            llvm::errs() << "tile height, width: " << tileHeight << ", "
                         << tileWidth << "\n";

            auto tensorRowCoord = (tensorCoord[0].second) / elemsPerDPASInst[0];
            auto tensorColCoord = tensorCoord[1].second /
                                  (elemsPerDPASInst[1] / packedElementsPerSlot);
            if (oneMatrixPerLoadForBT) {
              // HACK
              llvm::errs() << "overwriting tensor col coord\n";
              tensorColCoord = tensorCoord[1].second / tileWidth;
            }

            if (isOperandA) {
              LLVM_DEBUG(llvm::dbgs()
                         << "storing load vals index: " << tensorRowCoord
                         << ", " << tensorColCoord << "\n");
              loadVals[{tensorRowCoord, tensorColCoord}] =
                  b.bitcast(loadVal, unpackedDPASOperandType);
            } else {
              LLVM_DEBUG(llvm::dbgs()
                         << "storing load vals index: " << tensorColCoord
                         << ", " << tensorRowCoord << "\n");
              loadVals[{tensorColCoord, tensorRowCoord}] =
                  b.bitcast(loadVal, unpackedDPASOperandType);
            }
          }

          LLVM_DEBUG({
            llvm::dbgs() << "\n";
            unsigned packedRowNum = opIdx == DpasEncodingAttr::OpIdx::OperandA
                                        ? numOperandsOuterDimPerLoad
                                        : numOperandsInnerDimPerLoad;
            unsigned packedColNum = opIdx == DpasEncodingAttr::OpIdx::OperandA
                                        ? numOperandsInnerDimPerLoad
                                        : numOperandsOuterDimPerLoad;
            unsigned packedColNumPerVBlock = packedColNum / vBlocks;
            llvm::dbgs() << "packedRowNum: " << packedRowNum << "\n";
            llvm::dbgs() << "packedColNum: " << packedColNum << "\n";
            llvm::dbgs() << "packedColNumPerVBlock: " << packedColNumPerVBlock
                         << "\n";

            // Decompose the return value to multiple operands.
            for (int vblk = 0; vblk < vBlocks; ++vblk)
              for (int row = 0; row < packedRowNum; ++row)
                for (int col = 0; col < packedColNumPerVBlock; ++col) {
                  llvm::dbgs() << "Generating shuffle vector " << vblk << ", "
                               << row << ", " << col << "\n";
                  unsigned operandStartOffset = (vblk * packedRowNum + row) *
                                                packedColNumPerVBlock *
                                                packedElemsPerLanePerDPASInst;

                  SmallVector<int32_t> indices(packedElemsPerLanePerDPASInst);
                  for (int elemIdx = 0; elemIdx < packedElemsPerLanePerDPASInst;
                       ++elemIdx) {
                    indices[elemIdx] = operandStartOffset +
                                       elemIdx * packedColNumPerVBlock + col;
                    llvm::dbgs() << "indices[" << elemIdx << "]" << " = "
                                 << indices[elemIdx] << "\n";
                  }
                  // Save the decomposed vals to the map;
                  switch (opIdx) {
                  case DpasEncodingAttr::OpIdx::OperandA: {
                    llvm::dbgs() << "load vals index: "
                                 << std::to_string(outer * packedRowNum *
                                                       numLoadPerOutRepCluster +
                                                   rep * packedRowNum + row)
                                 << ", "
                                 << std::to_string(
                                        k + vblk * packedColNumPerVBlock + col)
                                 << "\n";
                  } break;
                  case DpasEncodingAttr::OpIdx::OperandB: {
                    llvm::dbgs()
                        << "load vals index: "
                        << std::to_string(outer * packedColNum *
                                              numLoadPerOutRepCluster +
                                          rep * packedColNum +
                                          vblk * packedColNumPerVBlock + col)
                        << ", " << std::to_string(k + row) << "\n";
                  } break;
                  case DpasEncodingAttr::OpIdx::OperandC: {
                    llvm_unreachable("unexpected OpIdx::OperandC");
                  } break;
                  }
                }
          });
        }
      }
    }

    // Extract the value returned by the load ops. And put the values in the
    // expected order for the layout.
    SmallVector<Value> unpackedLoadedVals;
    for (int outer = 0; outer < numRepOuter; ++outer) {
      for (int k = 0; k < numRepInner; ++k) {
        for (int rep = 0; rep < repCluster[unsigned(opIdx)]; ++rep) {
          if (loadVals.find({outer * repCluster[unsigned(opIdx)] + rep, k}) ==
              loadVals.end()) {
            llvm::errs() << "Failed to find key at "
                         << outer * repCluster[unsigned(opIdx)] + rep << ", "
                         << k << "\n";
          }
          Value loadVal =
              loadVals.at({outer * repCluster[unsigned(opIdx)] + rep, k});
          VectorType loadTy = cast<VectorType>(loadVal.getType());
          for (int i = 0; i < loadTy.getNumElements(); ++i) {
            auto val = b.extract_element(loadVal, b.i32_val(i));
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
    if (isTensorPointerType(op.getPtr().getType()))
      if (rewriteTensorPointerLoad(op, adaptor, rewriter).succeeded())
        return success();

    Location loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto typeConverter = getTypeConverter();
    MLIRContext *ctx = rewriter.getContext();
    Value ptr = op.getPtr();
    Value mask = op.getMask();
    Value llMask = adaptor.getMask();

    // Determine the vectorization size
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(op.getType()));
    unsigned numElems = getTotalElemsPerThread(op.getType());
    unsigned vec = getVectorSize(ptr);
    if (llMask)
      vec = std::min<size_t>(vec, getMaskAlignment(mask));

    SmallVector<Value> ptrElems, maskElems, otherElems;
    bool otherIsSplatConstInt = false;
    int64_t splatVal = 0;

    if (isTensorPointerType(ptr.getType())) {
      // fallback to gather load.
      auto tensorType = cast<RankedTensorType>(op.getType());
      std::tie(ptrElems, maskElems, otherElems) = convertBlockPtrToTensorOfPtr(
          loc, adaptor.getPtr(), tensorType, valueElemTy, rewriter,
          op.getBoundaryCheck(), op.getPadding());
    } else {
      Value other = op.getOther();
      Value llPtr = adaptor.getPtr();
      Value llOther = adaptor.getOther();

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

      Value pred = maskElems.size() ? maskElems[vecStart] : b.int_val(1, 1);

      SmallVector<Type> retTys(nWords, IntegerType::get(getContext(), width));
      Type retTy = retTys.size() > 1
                       ? vec_ty(IntegerType::get(ctx, width), nWords)
                       : retTys[0];

      Value other_ = b.undef(retTy);
      if (otherElems.size()) {
        for (size_t ii = 0; ii < nWords; ++ii) {
          size_t size = width / valueElemNBits;

          auto vecTy = vec_ty(valueElemTy, size);
          Value v = b.undef(vecTy);
          for (size_t s = 0; s < size; ++s) {
            Value falseVal = otherElems[vecStart + ii * size + s];
            Value sVal = createIndexAttrConstant(
                rewriter, loc, typeConverter->getIndexType(), s);
            v = b.insert_element(vecTy, v, falseVal, sVal);
          }
          v = b.bitcast(v, IntegerType::get(ctx, width));

          if (otherIsSplatConstInt) {
            for (size_t s = 0; s < 32; s += valueElemNBits)
              splatVal |= splatVal << valueElemNBits;
            v = b.int_val(width, splatVal);
          }

          Value iiVal = createIndexAttrConstant(
              rewriter, loc, typeConverter->getIndexType(), ii);
          if (nWords > 1) {
            other_ = b.insert_element(retTy, other_, v, iiVal);
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
                b.bitcast(ptrElems[vecStart], ptr_ty(ctx, 1 /*global*/));
            uint32_t alignment = nWords * width / 8;
            Value ret = b.load(retTy, addrElem, alignment);
            return SmallVector<Value, 1>{ret};
          });
      Value ret = *endBlock.args_begin();

      // Extract and store return values
      SmallVector<Value> rets;
      for (unsigned int ii = 0; ii < nWords; ++ii) {
        Value curr;
        if (isa<VectorType>(retTy)) {
          curr = b.extract_element(IntegerType::get(ctx, width), ret,
                                   b.i32_val(ii));
        } else {
          curr = ret;
        }
        curr = b.bitcast(curr, LLVM::getFixedVectorType(
                                   valueElemTy, width / valueElemNBits));
        rets.push_back(curr);
      }
      int tmp = width / valueElemNBits;
      for (size_t ii = 0; ii < vec; ++ii) {
        Value loaded =
            b.extract_element(valueElemTy, rets[ii / tmp], b.i32_val(ii % tmp));
        loadedVals.push_back(loaded);
      }
    } // end vec

    Type llvmResultStructTy = typeConverter->convertType(op.getType());
    Value resultStruct = packLLElements(loc, typeConverter, loadedVals,
                                        rewriter, llvmResultStructTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }

private:
  bool oneMatrixPerLoadForBT;
};

struct StoreOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::StoreOp>,
      public LoadStoreConversionBase {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::StoreOp>::ConvertTritonGPUOpToLLVMPattern;

  StoreOpConversion(
      TritonIntelGPUToLLVMTypeConverter &converter,
      const triton::intel::TargetInfo &targetInfo,
      const triton::intel::ModuleAxisInfoAnalysis &axisAnalysisPass,
      PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<triton::StoreOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  rewriteTensorPointerStore(triton::StoreOp op, OpAdaptor adaptor,
                            ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
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
    Value elemSizeInBytes = b.i32_val(elemSizeInBits / 8);
    const ArrayRef<int64_t> tensorShape = tensorType.getShape();
    size_t rank = tensorShape.size();
    unsigned numElems = getTotalElemsPerThread(tensorType);
    SmallVector<unsigned> elemsPerInstr = dpasLayout.getDPASInstShapeC();
    const SmallVector<unsigned> warpsPerCTA = dpasLayout.getWarpsPerCTA();
    SmallVector<int64_t> numReps =
        dpasLayout.getDPASRepetitions(tensorShape, 2);
    SmallVector<unsigned> order = triton::gpu::getOrder(dpasLayout);
    unsigned threadsPerWarp = triton::gpu::getWarpSize(dpasLayout);

    Value warpId = rewriter.create<arith::IndexCastOp>(
        loc, i32_ty,
        rewriter.create<mlir::gpu::SubgroupIdOp>(loc,
                                                 /*upperBound=*/nullptr));
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

    width = b.trunc(i32_ty, width);
    height = b.trunc(i32_ty, height);
    rowStride = b.trunc(i32_ty, rowStride);
    // encoded as bytes.
    Value baseWidth = b.mul(width, elemSizeInBytes);
    // encoded as bytes.
    Value basePitch = b.mul(rowStride, elemSizeInBytes);

    // A warp stride for the replicates.
    SmallVector<unsigned> repClusterShape = dpasLayout.getShapeC();
    unsigned outerDimWarpNum = std::min<unsigned>(
        warpsPerCTA[rank - 2],
        mlir::ceil<unsigned>(tensorShape[rank - 2], repClusterShape[rank - 2]));
    unsigned innerDimWarpNum = std::min<unsigned>(
        warpsPerCTA[rank - 1],
        mlir::ceil<unsigned>(tensorShape[rank - 1], repClusterShape[rank - 1]));
    Value outerDimWarpId =
        b.urem(multiDimWarpId[rank - 2], b.i32_val(outerDimWarpNum));
    Value innerDimWarpId =
        b.urem(multiDimWarpId[rank - 1], b.i32_val(innerDimWarpNum));
    int64_t numRepOuter = numReps[1];
    int64_t numRepInner = numReps[2];

    std::array<unsigned, 2> replicaStride = {
        outerDimWarpNum * repClusterShape[rank - 2],
        innerDimWarpNum * repClusterShape[rank - 1]};
    std::array<unsigned, 2> warpStride = {repClusterShape[rank - 2],
                                          repClusterShape[rank - 1]};

    Value dimWarpId0 = b.mul(outerDimWarpId, b.i32_val(warpStride[0]));
    Value dimWarpId1 = b.mul(innerDimWarpId, b.i32_val(warpStride[1]));
    Value warpId0Offset = b.add(dimWarpId0, offsetBaseY);
    Value warpId1Offset = b.add(dimWarpId1, offsetBaseX);

    ArrayRef<unsigned> repCluster = dpasLayout.getRepCluster();
    unsigned valOffset = 0;
    for (int m = 0; m < numRepOuter; ++m) {
      for (int n = 0; n < numRepInner; ++n) {
        for (int repM = 0; repM < repCluster[0]; ++repM) {
          Value offsetY =
              b.add(warpId0Offset,
                    b.i32_val(m * replicaStride[0] + repM * elemsPerInstr[0]));
          for (int repN = 0; repN < repCluster[1]; ++repN) {
            Value offsetX =
                b.add(warpId1Offset, b.i32_val(n * replicaStride[1] +
                                               repN * elemsPerInstr[1]));
            Value storeVal = rewriter.create<LLVM::UndefOp>(
                loc, LLVM::getFixedVectorType(typeConverter->convertType(eltTy),
                                              elemsPerLane));
            for (size_t i = 0; i < elemsPerLane; ++i) {
              storeVal =
                  b.insert_element(storeVal, vals[valOffset], b.i32_val(i));
              ++valOffset;
            }

            auto newOp = rewriter.create<TritonGEN::Matrix2DBlockStoreOp>(
                loc,
                /*ptr*/ base,
                /*base_width*/ baseWidth,
                /*base_height*/ height,
                /*base_pitch*/ basePitch,
                /*x*/ b.trunc(i32_ty, offsetX),
                /*y*/ b.trunc(i32_ty, offsetY),
                /*elem_size_in_bits*/ elemSizeInBits,
                /*tile_width*/ elemsPerInstr[1],
                /*tile_height*/ elemsPerInstr[0],
                /*v_blocks*/ 1,
                /*stored_val*/ b.bitcast(storeVal, store2DGenXType));

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
    if (isTensorPointerType(op.getPtr().getType()))
      if (rewriteTensorPointerStore(op, adaptor, rewriter).succeeded())
        return success();

    Location loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto *typeConverter = getTypeConverter();
    MLIRContext *ctx = rewriter.getContext();
    Value ptr = op.getPtr();
    Value mask = op.getMask();
    Value llMask = adaptor.getMask();

    // Determine the vectorization size
    Type valueTy = op.getValue().getType();
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));
    SmallVector<Value> ptrElems, maskElems;
    unsigned vec = getVectorSize(ptr);
    if (llMask)
      vec = std::min<size_t>(vec, getMaskAlignment(mask));

    if (isTensorPointerType(ptr.getType())) {
      // fallback to scatter store.
      auto tensorType = cast<RankedTensorType>(valueTy);
      SmallVector<Value> dummyOther;
      std::tie(ptrElems, maskElems, dummyOther) = convertBlockPtrToTensorOfPtr(
          loc, adaptor.getPtr(), tensorType, valueElemTy, rewriter,
          op.getBoundaryCheck());
    } else {
      Value llPtr = adaptor.getPtr();
      ptrElems = unpackLLElements(loc, llPtr, rewriter);
      if (llMask)
        maskElems = unpackLLElements(loc, llMask, rewriter);
    }

    Value llValue = adaptor.getValue();
    auto valueElems = unpackLLElements(loc, llValue, rewriter);
    assert(ptrElems.size() == valueElems.size());
    assert(!maskElems.size() ||
           valueElems.size() == maskElems.size() && "Mask size mismatch");

    mask = redundantDataMask(valueTy, rewriter, loc, targetInfo);
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
        Value llWord = b.undef(wordTy);
        // Insert each value element to the composition
        for (size_t elemIdx = 0; elemIdx < wordNElems; ++elemIdx) {
          const size_t elemOffset = vecStart + wordIdx * wordNElems + elemIdx;
          assert(elemOffset < valueElems.size());
          Value elem = valueElems[elemOffset];
          if (elem.getType().isInteger(1))
            elem = b.sext(i8_ty, elem);
          elem = b.bitcast(elem, valueElemTy);

          llWord = b.insert_element(wordTy, llWord, elem, b.i32_val(elemIdx));
        }
        llWord = b.bitcast(llWord, valArgTy);
        std::string constraint =
            (width == 64) ? "l" : ((width == 32) ? "r" : "c");
        asmArgs.emplace_back(llWord, constraint);
      }

      Value maskVal =
          maskElems.size() ? b.and_(mask, maskElems[vecStart]) : mask;

      auto vecTy = vec_ty(valArgTy, nWords);
      Value vecWord = b.undef(vecTy);
      for (int index = 0; index < asmArgs.size(); ++index) {
        auto llWord = asmArgs[index].first;
        vecWord = b.insert_element(vecTy, vecWord, llWord, b.i32_val(index));
      }

      // Create a predicated store operation.
      LLVM::intel::createPredicatedBlock(rewriter, loc, maskVal, [&] {
        Value addrElem =
            b.bitcast(ptrElems[vecStart], ptr_ty(ctx, 1 /*global*/));
        uint32_t alignment = nWords * width / 8;
        b.store(vecWord, addrElem, alignment);
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
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  b.barrier();
}

static LLVM::AtomicOrdering getMemoryOrdering(MemSemantic memOrdering) {
  switch (memOrdering) {
  case MemSemantic::RELAXED:
    return LLVM::AtomicOrdering::monotonic;
  case MemSemantic::ACQUIRE:
    return LLVM::AtomicOrdering::acquire;
  case MemSemantic::RELEASE:
    return LLVM::AtomicOrdering::release;
  case MemSemantic::ACQUIRE_RELEASE:
    return LLVM::AtomicOrdering::acq_rel;
  default:
    return LLVM::AtomicOrdering::acq_rel;
  }
}

struct AtomicCASOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::AtomicCASOp>,
      public LoadStoreConversionBase {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::AtomicCASOp>::ConvertTritonGPUOpToLLVMPattern;

  AtomicCASOpConversion(
      TritonIntelGPUToLLVMTypeConverter &converter,
      const triton::intel::TargetInfo &targetInfo,
      const triton::intel::ModuleAxisInfoAnalysis &axisAnalysisPass,
      PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<triton::AtomicCASOp>(converter,
                                                             benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::AtomicCASOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
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

    MemSemantic memSem = op.getSem();
    LLVM::AtomicOrdering successOrdering = getMemoryOrdering(memSem);
    LLVM::AtomicOrdering failureOrdering = LLVM::AtomicOrdering::monotonic;
    for (size_t i = 0; i < elemsPerThread; i += vec) {
      Value casVal = b.undef(vecTy);
      for (int ii = 0; ii < vec; ++ii) {
        Value iiVal = createIndexAttrConstant(
            rewriter, loc, getTypeConverter()->getIndexType(), ii);
        casVal = b.insert_element(vecTy, casVal, valElements[i + ii], iiVal);
      }

      Value casPtr = ptrElements[i];
      Value casCmp = cmpElements[i];
      casVal = valElements[i];

      assert((valueElemNBits == 32 || valueElemNBits == 64) &&
             "Unexpected width");

      Value zero = (valueElemNBits == 32) ? b.i32_val(0) : b.i64_val(0);
      if (!atomicNeedsSharedMemory(op.getResult()))
        rewriter.create<spirv::ControlBarrierOp>(
            loc, spirv::Scope::Workgroup, spirv::Scope::Workgroup,
            spirv::MemorySemantics::SequentiallyConsistent |
                spirv::MemorySemantics::CrossWorkgroupMemory);
      Block &endBlock =
          LLVM::intel::createPredicatedBlock(rewriter, loc, mask, {zero}, [&] {
            // casPtr = b.bitcast(casPtr, ptr_ty(ctx, 1));
            casCmp = b.bitcast(casCmp, zero.getType());
            casVal = b.bitcast(casVal, zero.getType());

            auto cmpxchg = rewriter.create<LLVM::AtomicCmpXchgOp>(
                loc, casPtr, casCmp, casVal, successOrdering, failureOrdering);
            Value newLoaded =
                rewriter.create<LLVM::ExtractValueOp>(loc, cmpxchg, 0);
            return SmallVector<Value, 1>{newLoaded};
          });

      Value ret = endBlock.getArgument(0);
      Type retType = (!tensorTy || vec == 1) ? valueElemTy : vecTy;
      ret = b.bitcast(ret, retType);

      if (tensorTy) {
        for (int ii = 0; ii < vec; ++ii) {
          resultVals[i + ii] =
              vec == 1 ? ret
                       : b.extract_element(valueElemTy, ret, b.i32_val(ii));
        }
      } else {
        if (!atomicNeedsSharedMemory(op.getResult())) {
          rewriter.eraseOp(op);
          return success();
        }
        Value atomPtr = LLVM::getSharedMemoryBase(loc, rewriter, targetInfo,
                                                  op.getOperation());
        atomPtr = b.bitcast(atomPtr, ptr_ty(ctx, 3));
        targetInfo.storeShared(rewriter, loc, atomPtr, ret, mask);
        createBarrier(rewriter, loc, numCTAs);
        Value ret = b.load(valueElemTy, atomPtr);
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

  AtomicRMWOpConversion(
      TritonIntelGPUToLLVMTypeConverter &converter,
      const triton::intel::TargetInfo &targetInfo,
      const triton::intel::ModuleAxisInfoAnalysis &axisAnalysisPass,
      PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<triton::AtomicRMWOp>(converter,
                                                             benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::AtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    MLIRContext *ctx = rewriter.getContext();

    auto moduleOp = op->getParentOfType<ModuleOp>();
    assert(moduleOp && "Parent ModuleOp not found for AtomicRMWOp");
    int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(moduleOp);

    auto atomicRmwAttr = op.getAtomicRmwOp();
    MemSemantic memSem = op.getSem();
    LLVM::AtomicOrdering llvmMemOrdering = getMemoryOrdering(memSem);

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
      Value rmwVal = b.undef(vecTy);
      for (int ii = 0; ii < vec; ++ii) {
        Value iiVal = createIndexAttrConstant(
            rewriter, loc, getTypeConverter()->getIndexType(), ii);
        rmwVal = b.insert_element(vecTy, rmwVal, valElements[i + ii], iiVal);
      }

      Value rmwPtr = ptrElements[i];
      Value rmwMask = llMask ? b.and_(mask, maskElements[i]) : mask;

      assert((valueElemNBits == 16 || valueElemNBits == 32 ||
              valueElemNBits == 64) &&
             "Unexpected width");

      Value zero;
      llvm::TypeSwitch<mlir::Type>(valueElemTy)
          .Case<mlir::IntegerType>(
              [&](auto ty) { zero = b.int_val(valueElemNBits, 0); })
          .Case<mlir::Float16Type>([&](auto ty) { zero = b.f16_val(0); })
          .Case<mlir::Float32Type>([&](auto ty) { zero = b.f32_val(0); })
          .Case<mlir::Float64Type>([&](auto ty) { zero = b.f64_val(0); });

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
              loc, spirv::Scope::Workgroup, spirv::Scope::Workgroup,
              spirv::MemorySemantics::SequentiallyConsistent |
                  spirv::MemorySemantics::CrossWorkgroupMemory);
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

              rmwVal = b.bitcast(rmwVal, valueElemTy);
              auto atomRMW = rewriter.create<LLVM::AtomicRMWOp>(
                  loc, rmwKind, rmwPtr, rmwVal, llvmMemOrdering);
              return SmallVector<Value, 1>{atomRMW.getRes()};
            });
      }

      Value ret = endBlock->getArgument(0);
      Type retType = (!tensorTy || vec == 1) ? valueElemTy : vecTy;
      ret = b.bitcast(ret, retType);

      if (tensorTy) {
        for (int ii = 0; ii < vec; ++ii) {
          resultVals[i + ii] =
              vec == 1 ? ret
                       : b.extract_element(valueElemTy, ret, b.i32_val(ii));
        }
      } else {
        if (!atomicNeedsSharedMemory(op.getResult())) {
          rewriter.eraseOp(op);
          return success();
        }
        Value atomPtr = LLVM::getSharedMemoryBase(loc, rewriter, targetInfo,
                                                  op.getOperation());
        atomPtr = b.bitcast(atomPtr, ptr_ty(ctx, 3));
        // Only threads with rmwMask = True store the result
        targetInfo.storeShared(rewriter, loc, atomPtr, ret, rmwMask);
        createBarrier(rewriter, loc, numCTAs);
        Value loadVal = b.load(valueElemTy, atomPtr);
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
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Block *insertionBlock = rewriter.getInsertionBlock();
    Block *headerBlock =
        rewriter.splitBlock(insertionBlock, rewriter.getInsertionPoint());
    Block *endBlock = rewriter.splitBlock(headerBlock, headerBlock->begin());
    rewriter.setInsertionPointToEnd(insertionBlock);
    rewriter.create<cf::CondBranchOp>(loc, rmwMask, headerBlock, endBlock, ops);
    rewriter.setInsertionPointToStart(headerBlock);

    rmwVal = b.bitcast(rmwVal, valueElemTy);

    // Align pointer by 4 bytes by zeroing lower address bits. Atomically read
    // a vector of two fp16 values as a single i32. The second lowest bit is
    // extracted to later be used as an index to extract the required vector
    // element.
    assert(isa<LLVM::LLVMPointerType>(rmwPtr.getType()));
    auto intPtr = b.ptrtoint(i64_ty, rmwPtr);
    auto lowPtrBits = b.and_(intPtr, b.i64_val(3));
    auto elemIndex = b.trunc(i32_ty, b.lshr(lowPtrBits, b.i64_val(1)));
    auto alignPtr = b.inttoptr(rmwPtr.getType(), b.sub(intPtr, lowPtrBits));
    auto firstValInt = b.load(i32_ty, alignPtr, 4, false, false, false, false,
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
    auto origValVec = b.bitcast(origValInt, vec_ty(valueElemTy, 2));
    Value origVal = b.extract_element(origValVec, elemIndex);

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
    Value newValVec = b.insert_element(origValVec, newVal, elemIndex);
    Value newValInt = b.bitcast(newValVec, i32_ty);

    // Execute cmpxchg and loop back if it fails.
    auto successOrdering = LLVM::AtomicOrdering::acq_rel;
    auto failureOrdering = LLVM::AtomicOrdering::monotonic;
    auto cmpxchg = rewriter.create<LLVM::AtomicCmpXchgOp>(
        loc, alignPtr, origValInt, newValInt, successOrdering, failureOrdering);
    auto newLoaded = b.extract_val(cmpxchg, 0);
    auto done = b.extract_val(cmpxchg, 1);
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
    const intel::ModuleAxisInfoAnalysis &axisInfoAnalysis,
    PatternBenefit benefit, bool oneMatrixPerLoadForBT) {
  patterns.add<AtomicCASOpConversion, AtomicRMWOpConversion, StoreOpConversion,
               PrefetchOpConversion>(typeConverter, targetInfo,
                                     axisInfoAnalysis, benefit);
  patterns.add<LoadOpConversion>(typeConverter, targetInfo, axisInfoAnalysis,
                                 benefit, oneMatrixPerLoadForBT);
}
