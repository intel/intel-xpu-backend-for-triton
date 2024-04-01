#ifndef TRITON_CONVERSION_TRITONINTELGPU_TO_LLVM_UTILITY_H
#define TRITON_CONVERSION_TRITONINTELGPU_TO_LLVM_UTILITY_H

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGEN/IR/TritonGENDialect.h"
#include "triton/Dialect/TritonIntelGPU/IR/Dialect.h"

#define DEBUG_TYPE "ttgpu_to_llvm"

using namespace mlir;
using namespace mlir::triton;

#undef store
#define store(...) rewriter.create<LLVM::StoreOp>(loc, __VA_ARGS__)
#undef addrspacecast
#define addrspacecast(...)                                                     \
  rewriter.create<LLVM::AddrSpaceCastOp>(loc, __VA_ARGS__)

// Constants
#define f16_val(...) LLVM::Intel::createConstantF16(loc, rewriter, __VA_ARGS__)
#define i64_val(...) LLVM::Intel::createConstantI64(loc, rewriter, __VA_ARGS__)

namespace mlir {
namespace LLVM {

namespace Intel {

/// Create a predicated block, using \p cond as the condition and \p ops for the
/// values supplied by the conditional branch to the exit block. The \p
/// thenOpsFn function is used to inject operations in the 'then' branch:
///   cf.cond_br %cond, ^br1, ^br2(%ops)
///   ^br1:
///     %then_ops = `thenOpsFn()`
///     cf.br ^br2(%then_ops)
///   ^br2(%block_ops):
template <typename ThenOpsFn>
Block &createPredicatedBlock(ConversionPatternRewriter &rewriter, Location loc,
                             Value cond, ArrayRef<Value> ops,
                             ThenOpsFn &&thenOpsFn) {
  Block *insertionBlock = rewriter.getInsertionBlock();
  Block *thenBlock =
      rewriter.splitBlock(insertionBlock, rewriter.getInsertionPoint());
  Block *endBlock = rewriter.splitBlock(thenBlock, thenBlock->begin());

  rewriter.setInsertionPointToEnd(insertionBlock);
  rewriter.create<cf::CondBranchOp>(loc, cond, thenBlock, endBlock, ops);

  rewriter.setInsertionPointToStart(thenBlock);
  auto thenOps = thenOpsFn();
  assert(thenOps.size() == ops.size() && "Inconsistent size");
  assert(llvm::all_of(llvm::enumerate(ops, thenOps),
                      [](const auto &enumerator) {
                        auto [index, op, thenOp] = enumerator;
                        return op.getType() == thenOp.getType();
                      }) &&
         "type mismatch found");

  if (thenOps.empty())
    rewriter.create<cf::BranchOp>(loc, endBlock);
  else
    rewriter.create<cf::BranchOp>(loc, endBlock, thenOps);

  for (Value op : thenOps)
    endBlock->addArgument(op.getType(), op.getLoc());

  rewriter.setInsertionPointToStart(endBlock);
  return *endBlock;
}

/// Create a predicated block, using \p cond as the condition and \p thenOpsFn
/// to inject operations in the 'then' branch:
///   cf.cond_br %cond, ^br1, ^br2
///   ^br1:
///     `thenOpsFn()`
///     cf.br ^br2
///   ^br2:
template <typename ThenOpsFn>
Block &createPredicatedBlock(ConversionPatternRewriter &rewriter, Location loc,
                             Value cond, ThenOpsFn &&thenOpsFn) {
  return createPredicatedBlock(rewriter, loc, cond, {}, thenOpsFn);
}

/// Create a 64-bit integer constant.
Value createConstantI64(Location loc, OpBuilder &rewriter, int64_t v);

/// Create a 16-bit float constant.
Value createConstantF16(Location loc, OpBuilder &rewriter, float v);

Value storeShared(ConversionPatternRewriter &rewriter, Location loc, Value ptr,
                  Value val, Value pred);

Value loadShared(ConversionPatternRewriter &rewriter, Location loc, Value ptr,
                 Type elemTy, Value pred);

Value shflSync(Location loc, ConversionPatternRewriter &rewriter, Value val,
               int i);
Value shflUpSync(Location loc, ConversionPatternRewriter &rewriter, Value val,
                 int i);
Value shflIdxSync(Location loc, ConversionPatternRewriter &rewriter, Value val,
                  int i);
Value shflIdxSync(Location loc, ConversionPatternRewriter &rewriter, Value val,
                  Value i);

Value llGetPid(Location loc, ConversionPatternRewriter &rewriter,
               ModuleOp moduleOp, int axis);

Value addStringToModule(Location loc, ConversionPatternRewriter &rewriter,
                        StringRef key, StringRef content,
                        unsigned addressSpace);

static Value getStackPointer(PatternRewriter &rewriter,
                             FunctionOpInterface funcOp) {
  auto mod = funcOp->getParentOfType<ModuleOp>();
  LLVM::LLVMPointerType ptrTy = ptr_ty(
      rewriter.getContext(), TritonGEN::TritonGENMemorySpace::kWorkgroup);
  if (mod->getAttrOfType<IntegerAttr>("triton_gpu.shared").getInt() == 0)
    return rewriter.create<LLVM::UndefOp>(funcOp.getLoc(), ptrTy);
  return funcOp.getArgument(funcOp.getNumArguments() - 1);
}

static Value getSharedMemoryBase(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 Operation *op) {
  auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext(), 3);
  FunctionOpInterface func =
      op->template getParentOfType<FunctionOpInterface>();
  assert(op->hasAttr("allocation.offset"));
  size_t offset = op->getAttr("allocation.offset")
                      .cast<IntegerAttr>()
                      .getValue()
                      .getZExtValue();
  Value offVal = i32_val(offset);
  Value base =
      gep(ptrTy, i8_ty, LLVM::Intel::getStackPointer(rewriter, func), offVal);
  return base;
}

// Returns a Value for the format string, which you can reuse.
Value llPrintf(ConversionPatternRewriter &rewriter, StringRef msg,
               ValueRange args);

void llPrintf(ConversionPatternRewriter &rewriter, Value msg, ValueRange args);

static Value getModuleWarpSize(RewriterBase &rewriter, Location loc) {
  auto mod = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  return i32_val(triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod));
}

static Value getClusterCTAId(RewriterBase &rewriter, Location loc) {
  // Clusters of thread blocks aren't supported.
  return rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
}
} // namespace Intel
} // namespace LLVM

// -----------------------------------------------------------------------
// Shared memory utilities
// -----------------------------------------------------------------------
using ::mlir::triton::getMultiDimIndex;
using ::mlir::triton::gpu::BlockedEncodingAttr;
using ::mlir::triton::gpu::CTALayoutAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::NvidiaMmaEncodingAttr;
using ::mlir::triton::gpu::SliceEncodingAttr;
using ::mlir::triton::gpu::intel::DpasEncodingAttr;

static SmallVector<Value>
emitBaseIndexForDpasLayout(Location loc, RewriterBase &rewriter,
                           const DpasEncodingAttr &dpasLayout,
                           RankedTensorType type);

static void
emitOffsetForDpasLayoutPerCTA(const DpasEncodingAttr &dpasLayout,
                              SmallVector<SmallVector<unsigned>> &offsets,
                              unsigned ctaOffsetX, unsigned ctaOffsetY) {
  SmallVector<unsigned> sizePerThreads = getSizePerThread(dpasLayout);
  uint32_t elemsPerThreadPerGroup = product<unsigned>(sizePerThreads);
  uint32_t rowsPerWarp =
      dpasLayout.getSubGroupSize() / dpasLayout.getExecutionSize();
  SmallVector<unsigned> shapePerCTA =
      triton::gpu::getShapePerCTATile(dpasLayout);

  for (unsigned elem = 0; elem < elemsPerThreadPerGroup; elem++) {
    uint32_t elemRowIndex = (elem / sizePerThreads[1]) * rowsPerWarp;
    uint32_t elemColIndex = elem % sizePerThreads[1];
    offsets.push_back({ctaOffsetX + elemRowIndex, ctaOffsetY + elemColIndex});
  }
}

static SmallVector<SmallVector<unsigned>>
emitOffsetForDpasLayout(const DpasEncodingAttr &dpasLayout,
                        RankedTensorType type) {
  ArrayRef<int64_t> shape = type.getShape();
  SmallVector<SmallVector<unsigned>> offsets;
  SmallVector<unsigned> shapePerCTA = getShapePerCTATile(dpasLayout);

  for (unsigned i = 0; i < shape[0]; i += shapePerCTA[0]) {
    for (unsigned j = 0; j < shape[1]; j += shapePerCTA[1]) {
      emitOffsetForDpasLayoutPerCTA(dpasLayout, offsets, i, j);
    }
  }

  return offsets;
}

// -----------------------------------------------------------------------
// Dpas layout indices
// -----------------------------------------------------------------------

static SmallVector<Value>
emitBaseIndexForDpasLayout(Location loc, RewriterBase &rewriter,
                           const DpasEncodingAttr &dpasLayout,
                           RankedTensorType type) {
  Value threadId = getThreadId(rewriter, loc);
  Value warpSize = i32_val(triton::gpu::getWarpSize(dpasLayout));
  Value warpId = udiv(threadId, warpSize);
  Value laneId = urem(threadId, warpSize);

  auto warpsPerCTA = dpasLayout.getWarpsPerCTA();
  ArrayRef<int64_t> shape = type.getShape();

  auto order = triton::gpu::getOrder(dpasLayout);
  SmallVector<Value> multiDimWarpId =
      delinearize(rewriter, loc, warpId, warpsPerCTA, order);

  // Compute the 2-dim coordinates of the warp containing the tensor element
  // operated on by this thread.
  SmallVector<unsigned> warpShape = dpasLayout.getShapeC();
  Value rowWarpId =
      urem(multiDimWarpId[0], i32_val(std::ceil(shape[0] / warpShape[0])));
  Value colWarpId =
      urem(multiDimWarpId[1], i32_val(std::ceil(shape[1] / warpShape[1])));
  Value rowWarpOffset = mul(rowWarpId, i32_val(warpShape[0]));
  Value colWarpOffset = mul(colWarpId, i32_val(warpShape[1]));

  // Compute the 2-dim coordinates of the first element in the warp operated
  // on by this thread.
  SmallVector<unsigned> threadsPerWarp = getThreadsPerWarp(dpasLayout);
  SmallVector<Value> multiDimBase = {
      add(udiv(laneId, i32_val(threadsPerWarp[1])), rowWarpOffset),
      add(urem(laneId, i32_val(threadsPerWarp[1])), colWarpOffset)};
  return multiDimBase;
}

namespace triton {
namespace intel {

static SmallVector<SmallVector<unsigned>>
emitOffsetForLayout(Attribute layout, RankedTensorType type);

static SmallVector<SmallVector<unsigned>>
emitOffsetForSliceLayout(const SliceEncodingAttr &sliceLayout,
                         RankedTensorType type) {
  auto parentEncoding = sliceLayout.getParent();
  unsigned dim = sliceLayout.getDim();
  auto parentShape = sliceLayout.paddedShape(type.getShape());
  RankedTensorType parentTy =
      RankedTensorType::get(parentShape, type.getElementType(), parentEncoding);
  auto parentOffsets = ::intel::emitOffsetForLayout(parentEncoding, parentTy);

  unsigned numOffsets = parentOffsets.size();
  SmallVector<SmallVector<unsigned>> resultOffsets;
  std::set<SmallVector<unsigned>> uniqueOffsets;

  for (unsigned i = 0; i < numOffsets; ++i) {
    SmallVector<unsigned> offsets = parentOffsets[i];
    offsets.erase(offsets.begin() + dim);
    if (uniqueOffsets.find(offsets) == uniqueOffsets.end()) {
      resultOffsets.push_back(offsets);
      uniqueOffsets.insert(offsets);
    }
  }
  return resultOffsets;
}

//

// -----------------------------------------------------------------------
// Get offsets / indices for any layout
// -----------------------------------------------------------------------

static SmallVector<Value> emitCTAOffsetForLayout(Location loc,
                                                 RewriterBase &rewriter,
                                                 Attribute layout,
                                                 ArrayRef<int64_t> shape) {
  unsigned rank = shape.size();
  SmallVector<unsigned> CTAsPerCGA = triton::gpu::getCTAsPerCGA(layout);
  SmallVector<unsigned> CTASplitNum = triton::gpu::getCTASplitNum(layout);
  SmallVector<unsigned> CTAOrder = triton::gpu::getCTAOrder(layout);
  SmallVector<int64_t> shapePerCTA =
      triton::gpu::getShapePerCTA(CTASplitNum, shape);

  // Delinearize clusterCTAId
  Value clusterCTAId = LLVM::Intel::getClusterCTAId(rewriter, loc);
  SmallVector<Value> multiDimClusterCTAId =
      delinearize(rewriter, loc, clusterCTAId, CTAsPerCGA, CTAOrder);

  // CTA Wrapping
  for (unsigned i = 0; i < rank; ++i) {
    // This wrapping rule must be consistent with getShapePerCTA
    unsigned splitNum = std::min<unsigned>(shape[i], CTASplitNum[i]);
    multiDimClusterCTAId[i] = urem(multiDimClusterCTAId[i], i32_val(splitNum));
  }

  SmallVector<Value> CTAOffset(rank);
  for (unsigned i = 0; i < rank; ++i)
    CTAOffset[i] = mul(multiDimClusterCTAId[i], i32_val(shapePerCTA[i]));

  return CTAOffset;
}

static SmallVector<Value>
emitBaseIndexForLayout(Location loc, RewriterBase &rewriter, Attribute layout,
                       RankedTensorType type, bool withCTAOffset) {
  auto shape = type.getShape();

  SmallVector<Value> baseIndex;
  RewriterBase::InsertionGuard guard(rewriter);
  SmallVector<Value> result;
  if (auto blockedLayout = layout.dyn_cast<BlockedEncodingAttr>()) {
    result = emitBaseIndexWithinCTAForBlockedLayout(loc, rewriter,
                                                    blockedLayout, type);
  } else if (auto mmaLayout = layout.dyn_cast<NvidiaMmaEncodingAttr>()) {
    if (mmaLayout.isVolta())
      result =
          emitBaseIndexWithinCTAForMmaLayoutV1(loc, rewriter, mmaLayout, type);
    if (mmaLayout.isAmpere() || mmaLayout.isHopper())
      result = emitBaseIndexWithinCTAForMmaLayoutV2V3(loc, rewriter, mmaLayout,
                                                      type);
  } else if (auto dpasLayout = layout.dyn_cast<DpasEncodingAttr>()) {
    result = emitBaseIndexForDpasLayout(loc, rewriter, dpasLayout, type);
  } else if (auto sliceLayout = layout.dyn_cast<SliceEncodingAttr>()) {
    auto parentLayout = sliceLayout.getParent();
    auto parentShape = sliceLayout.paddedShape(type.getShape());
    RankedTensorType parentTy =
        RankedTensorType::get(parentShape, type.getElementType(), parentLayout);
    result = ::intel::emitBaseIndexForLayout(loc, rewriter, parentLayout,
                                             parentTy, withCTAOffset);
    result.erase(result.begin() + sliceLayout.getDim());
    // CTAOffset has been added in emitBaseIndexForLayout of parentLayout
    return result;
  } else {
    llvm_unreachable("unsupported emitBaseIndexForLayout");
  }
  if (withCTAOffset) {
    auto CTAOffset =
        ::intel::emitCTAOffsetForLayout(loc, rewriter, layout, shape);
    assert(CTAOffset.size() == result.size() && "Rank mismatch");
    for (unsigned k = 0; k < result.size(); ++k)
      result[k] = add(result[k], CTAOffset[k]);
  }
  return result;
}

static SmallVector<SmallVector<unsigned>>
emitOffsetForLayout(Attribute layout, RankedTensorType type) {
  if (auto blockedLayout = layout.dyn_cast<BlockedEncodingAttr>())
    return emitOffsetForBlockedLayout(blockedLayout, type);
  if (auto mmaLayout = layout.dyn_cast<NvidiaMmaEncodingAttr>()) {
    if (mmaLayout.isVolta())
      return emitOffsetForMmaLayoutV1(mmaLayout, type);
    if (mmaLayout.isAmpere())
      return emitOffsetForMmaLayoutV2(mmaLayout, type);
    if (mmaLayout.isHopper())
      return emitOffsetForMmaLayoutV3(mmaLayout, type);
  }
  if (auto mfmaLayout = layout.dyn_cast<AMDMfmaEncodingAttr>()) {
    return emitOffsetForMfmaLayout(mfmaLayout, type);
  }
  if (auto wmmaLayout = layout.dyn_cast<AMDWmmaEncodingAttr>()) {
    return emitOffsetForWmmaLayout(wmmaLayout, type);
  }
  if (auto dpasLayout = layout.dyn_cast<DpasEncodingAttr>()) {
    return emitOffsetForDpasLayout(dpasLayout, type);
  }
  if (auto sliceLayout = layout.dyn_cast<SliceEncodingAttr>())
    return ::intel::emitOffsetForSliceLayout(sliceLayout, type);
  llvm_unreachable("unsupported emitOffsetForLayout");
}

// Emit indices calculation within each ConversionPattern, and returns a
// [elemsPerThread X rank] index matrix.
static SmallVector<SmallVector<Value>>
emitIndices(Location loc, RewriterBase &rewriter, Attribute layout,
            RankedTensorType type, bool withCTAOffset) {
  // step 1, delinearize threadId to get the base index
  auto multiDimBase = ::intel::emitBaseIndexForLayout(loc, rewriter, layout,
                                                      type, withCTAOffset);
  // step 2, get offset of each element
  auto offset = intel::emitOffsetForLayout(layout, type);
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

/* ---------------- */
/* ---------------- */
DenseMap<unsigned, Value> static getSwizzledSharedPtrs(
    Location loc, unsigned inVec, RankedTensorType srcTy,
    triton::gpu::SharedEncodingAttr resSharedLayout, Type resElemTy,
    SharedMemoryObject smemObj, RewriterBase &rewriter,
    SmallVectorImpl<Value> &offsetVals, SmallVectorImpl<Value> &srcStrides) {
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
  Value dstPtrBase = gep(dstPtrTy, resElemTy, smemObj.base, dstOffset);

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
  auto srcIndices =
      ::intel::emitIndices(loc, rewriter, srcEncoding, srcTy, false);
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
  if (outOrder.size() >= 2) {
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
  Value strideRow = outOrder.size() >= 2 ? srcStrides[outOrder[1]] : i32_val(0);
  Value strideCol = srcStrides[outOrder[0]];
  LDBG("getSwizzledSharedPtrs: perPhase = "
       << perPhase << " maxPhase = " << maxPhase << " minVec = " << minVec
       << " inVec = " << inVec << " outVec = " << outVec << " strideRow "
       << strideRow << " strideCol " << strideCol);
  for (unsigned elemIdx = 0; elemIdx < numElems; elemIdx += minVec) {
    Value offset = i32_val(0);
    // Extract multi dimensional index for current element
    auto idx = srcIndices[elemIdx];
    Value idxCol = idx[outOrder[0]]; // contiguous dimension
    Value idxRow;
    if (outOrder.size() >= 2) {
      idxRow = idx[outOrder[1]]; // discontiguous dimension
    } else {
      idxRow = i32_val(0);
    }
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
    }
    if (auto add = dyn_cast_or_null<LLVM::AddOp>(idxCol.getDefiningOp())) {
      if (auto _cst = dyn_cast_or_null<LLVM::ConstantOp>(
              add.getRhs().getDefiningOp())) {
        unsigned cst =
            _cst.getValue().cast<IntegerAttr>().getValue().getSExtValue();
        unsigned key = cst % (outVec * maxPhase);
        cacheCol.insert({key, idxCol});
        idxCol = cacheCol[key];
        immedateOffCol = cst / (outVec * maxPhase) * (outVec * maxPhase);
      }
    }
    if (auto add = dyn_cast_or_null<LLVM::AddOp>(idxRow.getDefiningOp())) {
      if (auto _cst = dyn_cast_or_null<LLVM::ConstantOp>(
              add.getRhs().getDefiningOp())) {
        unsigned cst =
            _cst.getValue().cast<IntegerAttr>().getValue().getSExtValue();
        unsigned key = cst % (perPhase * maxPhase);
        cacheRow.insert({key, idxRow});
        idxRow = cacheRow[key];
        immedateOffRow = cst / (perPhase * maxPhase) * (perPhase * maxPhase);
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
    if (outOrder.size() == 3)
      offset = add(offset, mul(idx[outOrder[2]], srcStrides[outOrder[2]]));
    offset = add(offset, add(rowOff, mul(colOff, strideCol)));
    Value currPtr = gep(dstPtrTy, resElemTy, dstPtrBase, offset);
    // compute immediate offset
    Value immediateOff;
    if (outOrder.size() >= 2) {
      immediateOff =
          add(mul(i32_val(immedateOffRow), strideRow), i32_val(immedateOffCol));
    } else {
      immediateOff = i32_val(immedateOffCol);
    }

    ret[elemIdx] = gep(dstPtrTy, resElemTy, currPtr, immediateOff);
  }
  return ret;
}

static SmallVector<Value>
loadSharedToDistributed(Value dst, ArrayRef<SmallVector<Value>> dstIndices,
                        Value src, SharedMemoryObject smemObj, Type elemTy,
                        Location loc, ConversionPatternRewriter &rewriter) {
  auto dstTy = dst.getType().cast<RankedTensorType>();
  auto dstShape = dstTy.getShape();
  assert(dstShape.size() <= 2 && "Unexpected rank of loadSharedToDistributed");
  auto srcTy = src.getType().cast<MemDescType>();
  auto dstDistributedLayout = dstTy.getEncoding();
  if (auto mmaLayout = dstDistributedLayout.dyn_cast<NvidiaMmaEncodingAttr>()) {
    assert((!mmaLayout.isVolta()) &&
           "ConvertLayout Shared->MMAv1 is not supported yet");
  }
  auto srcSharedLayout =
      srcTy.getEncoding().cast<triton::gpu::SharedEncodingAttr>();
  auto srcElemTy = srcTy.getElementType();
  auto dstElemTy = dstTy.getElementType();
  LDBG("loadSharedToDistributed elemTy " << elemTy << " srcElemTy " << srcElemTy
                                         << " dstElemTy " << dstElemTy);
  auto inOrd = triton::gpu::getOrder(srcSharedLayout);
  auto outOrd = triton::gpu::getOrder(dstDistributedLayout);
  unsigned outVec = inOrd == outOrd
                        ? triton::gpu::getUniqueContigPerThread(
                              dstDistributedLayout, dstShape)[outOrd[0]]
                        : 1;

  // If the shmem layout is not swizzled, we can trivially vectorize loads
  // across the whole width of the most-minor dimension of the shape, because
  // Triton requires all the dims are powers of 2.
  unsigned inVec = srcSharedLayout.getMaxPhase() == 1
                       ? srcTy.getShape()[inOrd[0]]
                       : srcSharedLayout.getVec();
  unsigned minVec = std::min(outVec, inVec);
  unsigned outElems = triton::gpu::getTotalElemsPerThread(dstTy);
  SmallVector<Value> offsetVals = {smemObj.strides.size(), i32_val(0)};
  assert(outElems == dstIndices.size());

  DenseMap<unsigned, Value> sharedPtrs = ::intel::getSwizzledSharedPtrs(
      loc, outVec, dstTy, srcSharedLayout, elemTy, smemObj, rewriter,
      offsetVals, smemObj.strides);
  assert(outElems % minVec == 0 && "Unexpected number of elements");
  unsigned numVecs = outElems / minVec;
  auto wordTy = vec_ty(elemTy, minVec);
  SmallVector<Value> outVals(outElems);
  for (unsigned i = 0; i < numVecs; ++i) {
    Value smemAddr = sharedPtrs[i * minVec];
    smemAddr = bitcast(smemAddr, ptr_ty(rewriter.getContext(), 3));
    auto valVec = load(wordTy, smemAddr);
    valVec.setAlignment(minVec * elemTy.getIntOrFloatBitWidth() / 8);
    for (unsigned v = 0; v < minVec; ++v) {
      Value currVal = extract_element(elemTy, valVec, i32_val(v));
      outVals[i * minVec + v] = currVal;
    }
  }
  return outVals;
}

static void storeDistributedToShared(Value src, ArrayRef<Value> inVals,
                                     ArrayRef<Value> dstStrides,
                                     ArrayRef<SmallVector<Value>> srcIndices,
                                     Value dst, Value smemBase, Type elemTy,
                                     Location loc,
                                     ConversionPatternRewriter &rewriter) {
  auto srcTy = src.getType().cast<RankedTensorType>();
  auto srcShape = srcTy.getShape();
  auto rank = srcShape.size();
  assert(rank == 2 ||
         rank == 3 && "Unexpected rank of storeDistributedToShared");
  auto dstTy = dst.getType().cast<MemDescType>();
  auto srcDistributedLayout = srcTy.getEncoding();
  if (auto mmaLayout = srcDistributedLayout.dyn_cast<NvidiaMmaEncodingAttr>()) {
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
  // If the shmem layout is not swizzled, we can trivially vectorize stores
  // across the whole width of the most-minor dimension of the shape, because
  // Triton requires all the dims are powers of 2.
  unsigned outVec = dstSharedLayout.getMaxPhase() == 1
                        ? dstTy.getShape()[inOrd[0]]
                        : dstSharedLayout.getVec();
  unsigned minVec = std::min(outVec, inVec);
  unsigned numElems = triton::gpu::getTotalElemsPerThread(srcTy);
  assert(numElems == srcIndices.size());
  auto wordTy = vec_ty(elemTy, minVec);
  Value word;

  SmallVector<Value, 3> srcStrides(dstStrides);
  SmallVector<Value, 3> offsetVals(rank, i32_val(0));
  SharedMemoryObject smemObj(smemBase, elemTy, srcStrides, offsetVals);

  DenseMap<unsigned, Value> sharedPtrs =
      ::intel::getSwizzledSharedPtrs(loc, inVec, srcTy, dstSharedLayout, elemTy,
                                     smemObj, rewriter, offsetVals, srcStrides);
  LDBG("storeDistributedToShared: numElems = " << numElems << " minVec = "
                                               << minVec << " " << wordTy);
  for (unsigned i = 0; i < numElems; ++i) {
    if (i % minVec == 0)
      word = undef(wordTy);
    word = insert_element(wordTy, word, inVals[i], i32_val(i % minVec));
    if (i % minVec == minVec - 1) {
      Value smemAddr = sharedPtrs[i / minVec * minVec];
      smemAddr = bitcast(smemAddr, ptr_ty(rewriter.getContext(), 3));
      store(word, smemAddr)
          .setAlignment(minVec * elemTy.getIntOrFloatBitWidth() / 8);
    }
  }
}

} // namespace intel
} // namespace triton
} // namespace mlir

#endif
