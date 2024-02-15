#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_BASE_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_BASE_H

// TODO: refactor so that it doesn't fail if Allocation.h
// is included after utility.h (due to conflict in `store` macro
// and <atomic>
#include "triton/Analysis/Allocation.h"

#include "TypeConverter.h"
//
#include "Utility.h"
#include "mlir/IR/TypeUtilities.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Dialect/NVGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include <set>
#include <type_traits>

#define DEBUG_TYPE "ttgpu_to_llvm"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::LLVM::delinearize;
using ::mlir::LLVM::SharedMemoryObject;
using ::mlir::triton::gpu::BlockedEncodingAttr;
using ::mlir::triton::gpu::CTALayoutAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::NvidiaMmaEncodingAttr;
using ::mlir::triton::gpu::SliceEncodingAttr;
using ::mlir::triton::gpu::intel::DpasEncodingAttr;
namespace ttng = ::mlir::triton::nvidia_gpu;

typedef DenseMap<Operation *, triton::MakeTensorPtrOp> TensorPtrMapT;

class ConvertTritonGPUOpToLLVMPatternBase {
public:
  explicit ConvertTritonGPUOpToLLVMPatternBase(
      TritonGPUToLLVMTypeConverter &typeConverter, Target target)
      : converter(&typeConverter), target(target) {}

  TritonGPUToLLVMTypeConverter *getTypeConverter() const { return converter; }

  Value getClusterCTAId(ConversionPatternRewriter &rewriter,
                        Location loc) const {
    switch (target) {
    case triton::Target::NVVM:
      return rewriter.create<triton::nvgpu::ClusterCTAIdOp>(
          loc, rewriter.getI32Type());
    case triton::Target::ROCDL:
    case triton::Target::GENX:
      // Clusters of thread blocks aren't supported.
      return rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
    default:
      llvm_unreachable("Unexpected target");
    }
  }

  // -----------------------------------------------------------------------
  // Shared memory utilities
  // -----------------------------------------------------------------------

  DenseMap<unsigned, Value>
  getSwizzledSharedPtrs(Location loc, unsigned inVec, RankedTensorType srcTy,
                        triton::gpu::SharedEncodingAttr resSharedLayout,
                        Type resElemTy, SharedMemoryObject smemObj,
                        ConversionPatternRewriter &rewriter,
                        SmallVectorImpl<Value> &offsetVals,
                        SmallVectorImpl<Value> &srcStrides) const {
    // This utility computes the pointers for accessing the provided swizzled
    // shared memory layout `resSharedLayout`. More specifically, it computes,
    // for all indices (row, col) of `srcEncoding` such that idx % inVec = 0,
    // the pointer: ptr[(row, col)] = base + (rowOff * strides[ord[1]] +
    // colOff) where :
    //   phase = (row // perPhase) % maxPhase
    //   rowOff = row
    //   colOff = colOffSwizzled + colOffOrdered
    //     colOffSwizzled = ((col // outVec) ^ phase) * outVec
    //     colOffOrdered = (col % outVec) // minVec * minVec
    //
    // Note 1:
    // -------
    // Because swizzling happens at a granularity of outVec, we need to
    // decompose the offset into a swizzled factor and a non-swizzled
    // (ordered) factor
    //
    // Note 2:
    // -------
    // If we have x, y, z of the form:
    // x = 0b00000xxxx
    // y = 0byyyyy0000
    // z = 0b00000zzzz
    // then (x + y) XOR z = 0byyyyxxxx XOR 0b00000zzzz = (x XOR z) + y
    // This means that we can use some immediate offsets for shared memory
    // operations.
    auto dstPtrTy = ptr_ty(rewriter.getContext(), 3);
    auto dstOffset = dot(rewriter, loc, offsetVals, smemObj.strides);
    Value dstPtrBase = gep(dstPtrTy, getTypeConverter()->convertType(resElemTy),
                           smemObj.base, dstOffset);

    auto srcEncoding = srcTy.getEncoding();
    auto srcShape = srcTy.getShape();
    auto srcShapePerCTA = triton::gpu::getShapePerCTA(srcTy);
    unsigned numElems = triton::gpu::getTotalElemsPerThread(srcTy);
    // swizzling params as described in TritonGPUAttrDefs.td
    unsigned outVec = resSharedLayout.getVec();
    unsigned perPhase = resSharedLayout.getPerPhase();
    unsigned maxPhase = resSharedLayout.getMaxPhase();
    // Order
    auto inOrder = triton::gpu::getOrder(srcEncoding);
    auto outOrder = triton::gpu::getOrder(resSharedLayout);
    assert(maxPhase == 1 ||
           outVec * maxPhase <= srcShape[outOrder[0]] &&
               "Swizzling would generate out of bounds memory accesses");
    // Tensor indices held by the current thread, as LLVM values
    auto srcIndices = emitIndices(loc, rewriter, srcEncoding, srcTy, false);
    // Swizzling with leading offsets (e.g. Hopper GMMA)
    unsigned swizzlingByteWidth = 0;
    if (resSharedLayout.getHasLeadingOffset()) {
      if (perPhase == 4 && maxPhase == 2)
        swizzlingByteWidth = 32;
      else if (perPhase == 2 && maxPhase == 4)
        swizzlingByteWidth = 64;
      else if (perPhase == 1 && maxPhase == 8)
        swizzlingByteWidth = 128;
      else
        llvm::report_fatal_error("Unsupported shared layout.");
    }
    unsigned numElemsPerSwizzlingRow =
        swizzlingByteWidth * 8 / resElemTy.getIntOrFloatBitWidth();
    Value numElemsPerSwizzlingRowVal = i32_val(numElemsPerSwizzlingRow);
    unsigned leadingDimOffset;
    if (outOrder.size() == 2) {
      leadingDimOffset = numElemsPerSwizzlingRow * srcShapePerCTA[outOrder[1]];
    } else {
      leadingDimOffset = numElemsPerSwizzlingRow;
    }

    Value leadingDimOffsetVal = i32_val(leadingDimOffset);
    // Return values
    DenseMap<unsigned, Value> ret;
    // cache for non-immediate offsets
    DenseMap<unsigned, Value> cacheCol, cacheRow;
    unsigned minVec = std::min(outVec, inVec);
    for (unsigned elemIdx = 0; elemIdx < numElems; elemIdx += minVec) {
      Value offset = i32_val(0);
      // Extract multi dimensional index for current element
      auto idx = srcIndices[elemIdx];
      Value idxCol = idx[outOrder[0]]; // contiguous dimension
      Value idxRow, strideRow;
      if (outOrder.size() == 2) {
        idxRow = idx[outOrder[1]]; // discontiguous dimension
        strideRow = srcStrides[outOrder[1]];
      } else {
        idxRow = i32_val(0);
        strideRow = i32_val(0);
      }
      Value strideCol = srcStrides[outOrder[0]];
      // compute phase = (row // perPhase) % maxPhase
      Value phase = urem(udiv(idxRow, i32_val(perPhase)), i32_val(maxPhase));
      // extract dynamic/static offset for immediate offsetting
      unsigned immedateOffCol = 0;
      unsigned immedateOffRow = 0;
      if (leadingDimOffset) {
        // hopper
        offset =
            mul(udiv(idxCol, numElemsPerSwizzlingRowVal), leadingDimOffsetVal);
        // Shrink by swizzling blocks
        idxCol = urem(idxCol, numElemsPerSwizzlingRowVal);
        strideRow = numElemsPerSwizzlingRowVal;
      } else {
        if (auto add = dyn_cast_or_null<LLVM::AddOp>(idxCol.getDefiningOp()))
          if (auto _cst = dyn_cast_or_null<LLVM::ConstantOp>(
                  add.getRhs().getDefiningOp())) {
            unsigned cst =
                _cst.getValue().cast<IntegerAttr>().getValue().getSExtValue();
            unsigned key = cst % (outVec * maxPhase);
            cacheCol.insert({key, idxCol});
            idxCol = cacheCol[key];
            immedateOffCol = cst / (outVec * maxPhase) * (outVec * maxPhase);
          }
        if (auto add = dyn_cast_or_null<LLVM::AddOp>(idxRow.getDefiningOp()))
          if (auto _cst = dyn_cast_or_null<LLVM::ConstantOp>(
                  add.getRhs().getDefiningOp())) {
            unsigned cst =
                _cst.getValue().cast<IntegerAttr>().getValue().getSExtValue();
            unsigned key = cst % (perPhase * maxPhase);
            cacheRow.insert({key, idxRow});
            idxRow = cacheRow[key];
            immedateOffRow =
                cst / (perPhase * maxPhase) * (perPhase * maxPhase);
          }
      }
      // row offset is simply row index
      Value rowOff = mul(idxRow, strideRow);
      // because swizzling happens at a granularity of outVec, we need to
      // decompose the offset into a swizzled factor and a non-swizzled
      // (ordered) factor: colOffSwizzled = ((col // outVec) ^ phase) * outVec
      // colOffOrdered = (col % outVec) // minVec * minVec
      Value colOffSwizzled = xor_(udiv(idxCol, i32_val(outVec)), phase);
      colOffSwizzled = mul(colOffSwizzled, i32_val(outVec));
      Value colOffOrdered = urem(idxCol, i32_val(outVec));
      colOffOrdered = udiv(colOffOrdered, i32_val(minVec));
      colOffOrdered = mul(colOffOrdered, i32_val(minVec));
      Value colOff = add(colOffSwizzled, colOffOrdered);
      // compute non-immediate offset
      offset = add(offset, add(rowOff, mul(colOff, strideCol)));
      Value currPtr = gep(dstPtrTy, getTypeConverter()->convertType(resElemTy),
                          dstPtrBase, offset);
      // compute immediate offset
      Value immediateOff;
      if (outOrder.size() == 2) {
        immediateOff =
            add(mul(i32_val(immedateOffRow), srcStrides[outOrder[1]]),
                i32_val(immedateOffCol));
      } else {
        immediateOff = i32_val(immedateOffCol);
      }

      ret[elemIdx] = gep(dstPtrTy, getTypeConverter()->convertType(resElemTy),
                         currPtr, immediateOff);
    }
    return ret;
  }

  /*-------*/
  SmallVector<Value>
  loadSharedToDistributed(Value dst, ArrayRef<SmallVector<Value>> dstIndices,
                          Value src, SharedMemoryObject smemObj, Type elemTy,
                          Location loc,
                          ConversionPatternRewriter &rewriter) const {
    auto dstTy = dst.getType().cast<RankedTensorType>();
    auto dstShape = dstTy.getShape();
    assert(dstShape.size() == 2 &&
           "Unexpected rank of loadSharedToDistributed");
    auto srcTy = src.getType().cast<RankedTensorType>();
    auto dstDistributedLayout = dstTy.getEncoding();
    if (auto mmaLayout =
            dstDistributedLayout.dyn_cast<NvidiaMmaEncodingAttr>()) {
      assert((!mmaLayout.isVolta()) &&
             "ConvertLayout Shared->MMAv1 is not supported yet");
    }
    auto srcSharedLayout =
        srcTy.getEncoding().cast<triton::gpu::SharedEncodingAttr>();
    auto srcElemTy = srcTy.getElementType();
    auto dstElemTy = dstTy.getElementType();
    auto inOrd = triton::gpu::getOrder(srcSharedLayout);
    auto outOrd = triton::gpu::getOrder(dstDistributedLayout);
    unsigned outVec = inOrd == outOrd
                          ? triton::gpu::getUniqueContigPerThread(
                                dstDistributedLayout, dstShape)[outOrd[0]]
                          : 1;
    unsigned inVec = srcSharedLayout.getVec();
    unsigned minVec = std::min(outVec, inVec);
    unsigned outElems = triton::gpu::getTotalElemsPerThread(dstTy);
    SmallVector<Value> offsetVals = {i32_val(0), i32_val(0)};
    assert(outElems == dstIndices.size());

    DenseMap<unsigned, Value> sharedPtrs =
        getSwizzledSharedPtrs(loc, outVec, dstTy, srcSharedLayout, srcElemTy,
                              smemObj, rewriter, offsetVals, smemObj.strides);
    assert(outElems % minVec == 0 && "Unexpected number of elements");
    unsigned numVecs = outElems / minVec;
    auto wordTy = vec_ty(elemTy, minVec);
    SmallVector<Value> outVals(outElems);
    for (unsigned i = 0; i < numVecs; ++i) {
      Value smemAddr = sharedPtrs[i * minVec];
      smemAddr = bitcast(smemAddr, ptr_ty(rewriter.getContext(), 3));
      Value valVec = load(wordTy, smemAddr);
      for (unsigned v = 0; v < minVec; ++v) {
        Value currVal = extract_element(dstElemTy, valVec, i32_val(v));
        outVals[i * minVec + v] = currVal;
      }
    }
    return outVals;
  }

  void storeDistributedToShared(Value src, Value llSrc,
                                ArrayRef<Value> dstStrides,
                                ArrayRef<SmallVector<Value>> srcIndices,
                                Value dst, Value smemBase, Type elemTy,
                                Location loc,
                                ConversionPatternRewriter &rewriter) const {
    auto srcTy = src.getType().cast<RankedTensorType>();
    auto srcShape = srcTy.getShape();
    assert(srcShape.size() == 2 &&
           "Unexpected rank of storeDistributedToShared");
    auto dstTy = dst.getType().cast<RankedTensorType>();
    auto srcDistributedLayout = srcTy.getEncoding();
    if (auto mmaLayout =
            srcDistributedLayout.dyn_cast<NvidiaMmaEncodingAttr>()) {
      assert((!mmaLayout.isVolta()) &&
             "ConvertLayout MMAv1->Shared is not supported yet");
    }
    auto dstSharedLayout =
        dstTy.getEncoding().cast<triton::gpu::SharedEncodingAttr>();
    auto dstElemTy = dstTy.getElementType();
    auto inOrd = triton::gpu::getOrder(srcDistributedLayout);
    auto outOrd = dstSharedLayout.getOrder();
    unsigned inVec = inOrd == outOrd
                         ? triton::gpu::getUniqueContigPerThread(
                               srcDistributedLayout, srcShape)[inOrd[0]]
                         : 1;
    unsigned outVec = dstSharedLayout.getVec();
    unsigned minVec = std::min(outVec, inVec);
    unsigned numElems = triton::gpu::getTotalElemsPerThread(srcTy);
    assert(numElems == srcIndices.size());
    auto inVals = unpackLLElements(loc, llSrc, rewriter);
    auto wordTy = vec_ty(elemTy, minVec);
    Value word;

    SmallVector<Value> srcStrides = {dstStrides[0], dstStrides[1]};
    SmallVector<Value> offsetVals = {i32_val(0), i32_val(0)};
    SharedMemoryObject smemObj(smemBase, elemTy, srcStrides, offsetVals);

    DenseMap<unsigned, Value> sharedPtrs =
        getSwizzledSharedPtrs(loc, inVec, srcTy, dstSharedLayout, dstElemTy,
                              smemObj, rewriter, offsetVals, srcStrides);

    for (unsigned i = 0; i < numElems; ++i) {
      if (i % minVec == 0)
        word = undef(wordTy);
      word = insert_element(wordTy, word, inVals[i], i32_val(i % minVec));
      if (i % minVec == minVec - 1) {
        Value smemAddr = sharedPtrs[i / minVec * minVec];
        smemAddr = bitcast(smemAddr, ptr_ty(rewriter.getContext(), 3));
        store(word, smemAddr);
      }
    }
  }

  // -----------------------------------------------------------------------
  // Utilities
  // -----------------------------------------------------------------------
  Value getMask(Type valueTy, ConversionPatternRewriter &rewriter,
                Location loc) const {
    auto tensorTy = valueTy.dyn_cast<RankedTensorType>();
    Value mask = int_val(1, 1);
    auto tid = tid_val();
    auto clusterCTAId = getClusterCTAId(rewriter, loc);
    if (tensorTy) {
      auto layout = tensorTy.getEncoding();
      auto shape = tensorTy.getShape();
      unsigned rank = shape.size();
      auto sizePerThread = triton::gpu::getSizePerThread(layout);
      auto threadsPerWarp = triton::gpu::getThreadsPerWarp(layout);
      auto warpsPerCTA = triton::gpu::getWarpsPerCTA(layout);
      auto order = triton::gpu::getOrder(layout);
      auto shapePerCTATile = triton::gpu::getShapePerCTATile(layout, shape);
      Value warpSize = getModuleWarpSize(rewriter, loc);
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

  // -----------------------------------------------------------------------
  // Get offsets / indices for any layout
  // -----------------------------------------------------------------------

  SmallVector<Value> emitCTAOffsetForLayout(Location loc,
                                            ConversionPatternRewriter &rewriter,
                                            Attribute layout,
                                            ArrayRef<int64_t> shape) const {
    unsigned rank = shape.size();
    SmallVector<unsigned> CTAsPerCGA = triton::gpu::getCTAsPerCGA(layout);
    SmallVector<unsigned> CTASplitNum = triton::gpu::getCTASplitNum(layout);
    SmallVector<unsigned> CTAOrder = triton::gpu::getCTAOrder(layout);
    SmallVector<int64_t> shapePerCTA =
        triton::gpu::getShapePerCTA(CTASplitNum, shape);

    // Delinearize clusterCTAId
    Value clusterCTAId = getClusterCTAId(rewriter, loc);
    SmallVector<Value> multiDimClusterCTAId =
        delinearize(rewriter, loc, clusterCTAId, CTAsPerCGA, CTAOrder);

    // CTA Wrapping
    for (unsigned i = 0; i < rank; ++i) {
      // This wrapping rule must be consistent with getShapePerCTA
      unsigned splitNum = std::min<unsigned>(shape[i], CTASplitNum[i]);
      multiDimClusterCTAId[i] =
          urem(multiDimClusterCTAId[i], i32_val(splitNum));
    }

    SmallVector<Value> CTAOffset(rank);
    for (unsigned i = 0; i < rank; ++i)
      CTAOffset[i] = mul(multiDimClusterCTAId[i], i32_val(shapePerCTA[i]));

    return CTAOffset;
  }

  SmallVector<Value> emitBaseIndexForLayout(Location loc,
                                            ConversionPatternRewriter &rewriter,
                                            Attribute layout,
                                            RankedTensorType type,
                                            bool withCTAOffset) const {
    auto shape = type.getShape();

    SmallVector<Value> baseIndex;
    ConversionPatternRewriter::InsertionGuard guard(rewriter);
    SmallVector<Value> result;
    if (auto blockedLayout = layout.dyn_cast<BlockedEncodingAttr>()) {
      result = emitBaseIndexWithinCTAForBlockedLayout(loc, rewriter,
                                                      blockedLayout, type);
    } else if (auto mmaLayout = layout.dyn_cast<NvidiaMmaEncodingAttr>()) {
      if (mmaLayout.isVolta())
        result = emitBaseIndexWithinCTAForMmaLayoutV1(loc, rewriter, mmaLayout,
                                                      type);
      if (mmaLayout.isAmpere() || mmaLayout.isHopper())
        result = emitBaseIndexWithinCTAForMmaLayoutV2V3(loc, rewriter,
                                                        mmaLayout, type);
    } else if (auto dpasLayout = layout.dyn_cast<DpasEncodingAttr>()) {
      result = emitBaseIndexForDpasLayout(loc, rewriter, dpasLayout, type);
    } else if (auto sliceLayout = layout.dyn_cast<SliceEncodingAttr>()) {
      auto parentLayout = sliceLayout.getParent();
      auto parentShape = sliceLayout.paddedShape(type.getShape());
      RankedTensorType parentTy = RankedTensorType::get(
          parentShape, type.getElementType(), parentLayout);
      result = emitBaseIndexForLayout(loc, rewriter, parentLayout, parentTy,
                                      withCTAOffset);
      result.erase(result.begin() + sliceLayout.getDim());
      // CTAOffset has been added in emitBaseIndexForLayout of parentLayout
      return result;
    } else {
      llvm_unreachable("unsupported emitBaseIndexForLayout");
    }
    if (withCTAOffset) {
      auto CTAOffset = emitCTAOffsetForLayout(loc, rewriter, layout, shape);
      assert(CTAOffset.size() == result.size() && "Rank mismatch");
      for (unsigned k = 0; k < result.size(); ++k)
        result[k] = add(result[k], CTAOffset[k]);
    }
    return result;
  }

  // Emit indices calculation within each ConversionPattern, and returns a
  // [elemsPerThread X rank] index matrix.
  SmallVector<SmallVector<Value>>
  emitIndices(Location loc, ConversionPatternRewriter &rewriter,
              Attribute layout, RankedTensorType type,
              bool withCTAOffset) const {
    // step 1, delinearize threadId to get the base index
    auto multiDimBase =
        emitBaseIndexForLayout(loc, rewriter, layout, type, withCTAOffset);
    // step 2, get offset of each element
    auto offset = emitOffsetForLayout(layout, type);
    // step 3, add offset to base, and reorder the sequence
    // of indices to guarantee that elems in the same
    // sizePerThread are adjacent in order
    auto shape = type.getShape();
    unsigned rank = shape.size();
    unsigned elemsPerThread = offset.size();
    SmallVector<SmallVector<Value>> multiDimIdx(elemsPerThread,
                                                SmallVector<Value>(rank));
    for (unsigned n = 0; n < elemsPerThread; ++n)
      for (unsigned k = 0; k < rank; ++k)
        multiDimIdx[n][k] = add(multiDimBase[k], i32_val(offset[n][k]));
    return multiDimIdx;
  }

private:
  void restoreInsertionPointIfSet(OpBuilder::InsertPoint *insertPt,
                                  ConversionPatternRewriter &rewriter) const {
    if (insertPt->isSet()) {
      rewriter.restoreInsertionPoint(*insertPt);
    } else {
      auto func =
          rewriter.getInsertionPoint()->getParentOfType<LLVM::LLVMFuncOp>();
      rewriter.setInsertionPointToStart(&func.getBody().front());
    }
  }

  // -----------------------------------------------------------------------
  // Blocked layout indices
  // -----------------------------------------------------------------------

  // Get an index-base for each dimension for a \param blockedLayout.
  SmallVector<Value> emitBaseIndexWithinCTAForBlockedLayout(
      Location loc, ConversionPatternRewriter &rewriter,
      const BlockedEncodingAttr &blockedLayout, RankedTensorType type) const {
    auto shape = type.getShape();
    Value threadId = getThreadId(rewriter, loc);
    Value warpSize = getModuleWarpSize(rewriter, loc);
    Value laneId = urem(threadId, warpSize);
    Value warpId = udiv(threadId, warpSize);
    auto sizePerThread = blockedLayout.getSizePerThread();
    auto threadsPerWarp = blockedLayout.getThreadsPerWarp();
    auto warpsPerCTA = blockedLayout.getWarpsPerCTA();
    auto order = blockedLayout.getOrder();
    auto shapePerCTA = triton::gpu::getShapePerCTA(blockedLayout, shape);
    unsigned rank = shape.size();

    // delinearize threadId to get the base index
    SmallVector<Value> multiDimWarpId =
        delinearize(rewriter, loc, warpId, warpsPerCTA, order);
    SmallVector<Value> multiDimThreadId =
        delinearize(rewriter, loc, laneId, threadsPerWarp, order);

    SmallVector<Value> multiDimBase(rank);
    for (unsigned k = 0; k < rank; ++k) {
      // Wrap around multiDimWarpId/multiDimThreadId in case
      // shapePerCTATile[k] > shapePerCTA[k]
      auto maxWarps =
          ceil<unsigned>(shapePerCTA[k], sizePerThread[k] * threadsPerWarp[k]);
      auto maxThreads = ceil<unsigned>(shapePerCTA[k], sizePerThread[k]);
      multiDimWarpId[k] = urem(multiDimWarpId[k], i32_val(maxWarps));
      multiDimThreadId[k] = urem(multiDimThreadId[k], i32_val(maxThreads));
      // multiDimBase[k] = (multiDimThreadId[k] +
      //                    multiDimWarpId[k] * threadsPerWarp[k]) *
      //                   sizePerThread[k];
      Value threadsPerWarpK = i32_val(threadsPerWarp[k]);
      Value sizePerThreadK = i32_val(sizePerThread[k]);
      multiDimBase[k] =
          mul(sizePerThreadK, add(multiDimThreadId[k],
                                  mul(multiDimWarpId[k], threadsPerWarpK)));
    }
    return multiDimBase;
  }

protected:
  TritonGPUToLLVMTypeConverter *converter;
  Target target;
};

template <typename SourceOp>
class ConvertTritonGPUOpToLLVMPattern
    : public ConvertOpToLLVMPattern<SourceOp>,
      public ConvertTritonGPUOpToLLVMPatternBase {
public:
  using OpAdaptor = typename SourceOp::Adaptor;

  explicit ConvertTritonGPUOpToLLVMPattern(
      TritonGPUToLLVMTypeConverter &typeConverter, Target target,
      PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<SourceOp>(typeConverter, benefit),
        ConvertTritonGPUOpToLLVMPatternBase(typeConverter, target) {}

protected:
  TritonGPUToLLVMTypeConverter *getTypeConverter() const {
    LLVMTypeConverter *ret =
        ((ConvertTritonGPUOpToLLVMPatternBase *)this)->getTypeConverter();
    return (TritonGPUToLLVMTypeConverter *)ret;
  }
};

namespace mlir::triton {
class ReduceOp;
class ScanOp;
} // namespace mlir::triton

template <typename SourceOp>
class ConvertTritonGPUReduceScanToLLVMPattern
    : public ConvertTritonGPUOpToLLVMPattern<SourceOp> {
public:
  // Make sure the class is only instantiated with Reduce and Scan
  static_assert(std::is_same_v<SourceOp, ReduceOp> ||
                std::is_same_v<SourceOp, ScanOp>);

  using ConvertTritonGPUOpToLLVMPatternBase::getTypeConverter;
  using ConvertTritonGPUOpToLLVMPattern<
      SourceOp>::ConvertTritonGPUOpToLLVMPattern;

  // Return the pointee type of the shared memory pointer for operand i.
  Type getElementType(SourceOp op, int i) const {
    auto ty = op.getInputTypes()[i].getElementType();
    return getTypeConverter()->convertType(ty);
  }

  // Helper to compute the smem bases in both reductions and scans
  SmallVector<Value> getSmemBases(SourceOp op, unsigned elems,
                                  ConversionPatternRewriter &rewriter,
                                  Target target) const {
    auto loc = op.getLoc();
    // indices will store the index of the op operands in descending order
    // of their bitwidths
    std::vector<unsigned> indices(op.getNumOperands());
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), [&](unsigned i, unsigned j) {
      return op.getElementTypes()[i].getIntOrFloatBitWidth() >
             op.getElementTypes()[j].getIntOrFloatBitWidth();
    });
    // Assign base index to each operand in their order in indices
    std::map<unsigned, Value> indexToBase;
    indexToBase[indices[0]] =
        LLVM::getSharedMemoryBase(loc, rewriter, op.getOperation(), target);
    for (unsigned i = 1; i < op.getNumOperands(); ++i) {
      indexToBase[indices[i]] = gep(
          ptr_ty(rewriter.getContext(), 3), getElementType(op, indices[i - 1]),
          indexToBase[indices[i - 1]], i32_val(elems));
    }
    // smemBases[k] is the base pointer for the k-th operand
    SmallVector<Value> smemBases(op.getNumOperands());
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      smemBases[i] = indexToBase[i];
    }
    return smemBases;
  }
};

#endif
