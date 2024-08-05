#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "Utility.h"

#include "intel/include/Analysis/Utility.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/LinearLayoutConversions.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::LLVM::getStridesFromShapeAndOrder;
using ::mlir::LLVM::linearize;

using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getContigPerThread;
using ::mlir::triton::gpu::getOrder;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::getShapePerCTATile;
using ::mlir::triton::gpu::getSizePerThread;
using ::mlir::triton::gpu::getTotalElemsPerThread;
using ::mlir::triton::gpu::isaDistributedLayout;
using ::mlir::triton::gpu::SharedEncodingAttr;
using ::mlir::triton::gpu::intel::DpasEncodingAttr;

// Forward declarations
namespace SharedToDotOperandDPAS::intel {
Value convertLayout(int opIdx, ConversionPatternRewriter &rewriter,
                    Location loc, Value tensor,
                    DotOperandEncodingAttr bEncoding,
                    const SharedMemoryObject &smemObj,
                    const LLVMTypeConverter *typeConverter, Value thread);

} // namespace SharedToDotOperandDPAS::intel

namespace mlir::triton::gpu {
namespace {

// shared -> dot_operand if the result layout is dpas
Value lowerSharedToDotOperandDPAS(
    triton::gpu::LocalLoadOp op, triton::gpu::LocalLoadOpAdaptor adaptor,
    const LLVMTypeConverter *typeConverter, ConversionPatternRewriter &rewriter,
    const DpasEncodingAttr &dpasLayout,
    const DotOperandEncodingAttr &dotOperandLayout, bool isOuter) {
  auto loc = op.getLoc();
  auto src = op.getSrc();
  Value dst = op.getResult();

  auto llvmElemTy = typeConverter->convertType(src.getType().getElementType());

  auto smemObj = getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                 llvmElemTy, rewriter);
  Value res;
  if (!isOuter) {
    res = SharedToDotOperandDPAS::intel::convertLayout(
        dotOperandLayout.getOpIdx(), rewriter, loc, src, dotOperandLayout,
        smemObj, typeConverter, tid_val());
  } else {
    assert(false && "unsupported DPAS layout found");
  }
  return res;
}
// shared -> dpas_operand
LogicalResult lowerSharedToDotOperand(triton::gpu::LocalLoadOp op,
                                      triton::gpu::LocalLoadOpAdaptor adaptor,
                                      const LLVMTypeConverter *typeConverter,
                                      ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();
  auto dstEnc = cast<DotOperandEncodingAttr>(op.getType().getEncoding());
  auto sharedLayout =
      cast<SharedEncodingAttr>(op.getSrc().getType().getEncoding());

  int K;
  if (dstEnc.getOpIdx() == 0) // $a
    K = op.getType().getShape()[sharedLayout.getOrder()[0]];
  else // $b
    K = op.getType().getShape()[sharedLayout.getOrder()[1]];
  bool isOuter = K == 1;

  Value res;
  if (auto dpasLayout =
          dyn_cast_or_null<DpasEncodingAttr>(dstEnc.getParent())) {
    res = lowerSharedToDotOperandDPAS(op, adaptor, typeConverter, rewriter,
                                      dpasLayout, dstEnc, isOuter);
  } else if (auto blockedLayout =
                 dyn_cast_or_null<BlockedEncodingAttr>(dstEnc.getParent())) {
    auto thread = getThreadId(rewriter, loc);
    res = SharedToDotOperandFMA::convertLayout(
        dstEnc.getOpIdx(), op.getSrc(), adaptor.getSrc(), blockedLayout, thread,
        loc, typeConverter, rewriter);
  } else {
    assert(false && "Unsupported dot operand layout found");
  }

  rewriter.replaceOp(op, res);
  return success();
}

struct LocalLoadOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp> {
public:
  LocalLoadOpConversion(LLVMTypeConverter &typeConverter,
                        const TargetInfoBase &targetInfo,
                        PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(triton::gpu::LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemDescType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();
    if (isa<DotOperandEncodingAttr>(dstLayout))
      return lowerSharedToDotOperand(op, adaptor, getTypeConverter(), rewriter);
    if (isa<SharedEncodingAttr>(srcLayout) && isaDistributedLayout(dstLayout))
      return lowerSharedToDistributed(op, adaptor, getTypeConverter(),
                                      rewriter);
    return failure();
  }

private:
  LogicalResult
  lowerSharedToDistributed(triton::gpu::LocalLoadOp op,
                           triton::gpu::LocalLoadOpAdaptor adaptor,
                           const LLVMTypeConverter *typeConverter,
                           ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getResult().getType();
    auto dstShape = dstTy.getShape();
    assert(dstShape.size() <= 2 &&
           "Unexpected rank of ConvertLayout(shared->blocked)");
    auto srcSharedLayout = cast<SharedEncodingAttr>(srcTy.getEncoding());
    auto dstLayout = dstTy.getEncoding();
    auto inOrd = getOrder(srcSharedLayout);

    auto smemObj = getSharedMemoryObjectFromStruct(
        loc, adaptor.getSrc(),
        typeConverter->convertType(srcTy.getElementType()), rewriter);
    auto elemTy = typeConverter->convertType(dstTy.getElementType());

    auto srcStrides =
        getStridesFromShapeAndOrder(srcTy.getShape(), inOrd, loc, rewriter);
    auto dstIndices =
        ::intel::emitIndices(loc, rewriter, targetInfo, dstLayout, dstTy, true);

    SmallVector<Value> outVals =
        ::intel::loadSharedToDistributed(op.getResult(), op.getSrc(), smemObj,
                                         elemTy, loc, rewriter, targetInfo);

    Value result = packLLElements(loc, typeConverter, outVals, rewriter, dstTy);
    rewriter.replaceOp(op, result);

    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

struct ConvertLayoutOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::gpu::ConvertLayoutOp> {
public:
  ConvertLayoutOpConversion(const LLVMTypeConverter &typeConverter,
                            const triton::intel::TargetInfo &targetInfo,
                            PatternBenefit benefit = 1)
      : ConvertTritonGPUOpToLLVMPattern(typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();
    if (isaDistributedLayout(srcLayout) && isaDistributedLayout(dstLayout)) {
      return lowerDistributedToDistributed(op, adaptor, rewriter);
    }
    if (isa<DpasEncodingAttr>(srcLayout) &&
        isa<DotOperandEncodingAttr>(dstLayout)) {
      return lowerDpasToDotOperand(op, adaptor, rewriter);
    }

    return failure();
  }

private:
  SmallVector<Value>
  getMultiDimOffset(Attribute layout, Location loc,
                    ConversionPatternRewriter &rewriter, unsigned elemId,
                    RankedTensorType type,
                    ArrayRef<unsigned> multiDimCTAInRepId,
                    ArrayRef<unsigned> shapePerCTATile) const {
    auto shape = type.getShape();
    unsigned rank = shape.size();
    if (auto blockedLayout = dyn_cast<BlockedEncodingAttr>(layout)) {
      auto multiDimOffsetFirstElem = ::intel::emitBaseIndexForLayout(
          loc, rewriter, targetInfo, blockedLayout, type, false);
      SmallVector<Value> multiDimOffset(rank);
      SmallVector<unsigned> multiDimElemId = getMultiDimIndex<unsigned>(
          elemId, getSizePerThread(layout), getOrder(layout));
      for (unsigned d = 0; d < rank; ++d) {
        multiDimOffset[d] =
            add(multiDimOffsetFirstElem[d],
                i32_val(multiDimCTAInRepId[d] * shapePerCTATile[d] +
                        multiDimElemId[d]));
      }
      return multiDimOffset;
    }
    if (auto sliceLayout = dyn_cast<SliceEncodingAttr>(layout)) {
      unsigned dim = sliceLayout.getDim();
      auto parentEncoding = sliceLayout.getParent();
      auto parentSizePerThread = getSizePerThread(parentEncoding);
      auto parentShape = sliceLayout.paddedShape(shape);
      auto parentTy = RankedTensorType::get(parentShape, type.getElementType(),
                                            parentEncoding);
      auto offsets = ::intel::emitOffsetForLayout(layout, type);
      auto parentOffset =
          ::intel::emitOffsetForLayout(parentEncoding, parentTy);
      SmallVector<int> idxs;
      for (SmallVector<unsigned> off : offsets) {
        off.insert(off.begin() + dim, 0);
        auto it = std::find(parentOffset.begin(), parentOffset.end(), off);
        idxs.push_back(std::distance(parentOffset.begin(), it));
      }
      auto multiDimOffsetParent = getMultiDimOffset(
          parentEncoding, loc, rewriter, idxs[elemId], parentTy,
          sliceLayout.paddedShape(multiDimCTAInRepId),
          sliceLayout.paddedShape(shapePerCTATile));
      SmallVector<Value> multiDimOffset(rank);
      for (unsigned d = 0; d < rank + 1; ++d) {
        if (d == dim)
          continue;
        unsigned slicedD = d < dim ? d : (d - 1);
        multiDimOffset[slicedD] = multiDimOffsetParent[d];
      }
      return multiDimOffset;
    }
    if (auto dpasLayout = dyn_cast<DpasEncodingAttr>(layout)) {
      assert(rank == 2);
      auto multiDimBase = ::intel::emitBaseIndexForLayout(
          loc, rewriter, targetInfo, layout, type, false);
      SmallVector<SmallVector<unsigned>> offsets;
      ::emitOffsetForDpasLayoutPerCTA(
          dpasLayout, offsets, multiDimCTAInRepId[0] * shapePerCTATile[0],
          multiDimCTAInRepId[1] * shapePerCTATile[1]);

      SmallVector<Value> multiDimOffset = {
          add(multiDimBase[0], i32_val(offsets[elemId][0])),
          add(multiDimBase[1], i32_val(offsets[elemId][1]))};

      return multiDimOffset;
    }
    llvm_unreachable("unexpected layout in getMultiDimOffset");
  }

  SmallVector<Value>
  getWrappedMultiDimOffset(ConversionPatternRewriter &rewriter, Location loc,
                           ArrayRef<Value> multiDimOffset,
                           ArrayRef<unsigned> shape,
                           SmallVector<unsigned> shapePerCTATile,
                           SmallVector<int64_t> shapePerCTA) const {
    unsigned rank = shape.size();
    SmallVector<Value> multiDimOffsetWrapped(rank);
    for (unsigned d = 0; d < rank; ++d) {
      if (shapePerCTATile[d] > shapePerCTA[d])
        multiDimOffsetWrapped[d] = urem(multiDimOffset[d], i32_val(shape[d]));
      else
        multiDimOffsetWrapped[d] = multiDimOffset[d];
    }
    return multiDimOffsetWrapped;
  }

  // shared memory rd/st for blocked or dpas layout with data padding
  void processReplica(Location loc, ConversionPatternRewriter &rewriter,
                      bool stNotRd, RankedTensorType type,
                      ArrayRef<unsigned> numCTAsEachRep,
                      ArrayRef<unsigned> multiDimRepId, unsigned vec,
                      ArrayRef<unsigned> paddedRepShape,
                      ArrayRef<unsigned> origRepShape,
                      ArrayRef<unsigned> outOrd, SmallVector<Value> &vals,
                      Value smemBase) const {
    auto accumNumCTAsEachRep = product<unsigned>(numCTAsEachRep);
    auto layout = type.getEncoding();
    auto rank = type.getRank();
    auto sizePerThread = getSizePerThread(layout);
    auto accumSizePerThread = product<unsigned>(sizePerThread);
    SmallVector<unsigned> numCTATiles(rank);
    auto shapePerCTATile = getShapePerCTATile(layout);
    auto shapePerCTA = getShapePerCTA(layout, type.getShape());
    auto order = getOrder(layout);
    for (unsigned d = 0; d < rank; ++d) {
      numCTATiles[d] = ceil<unsigned>(shapePerCTA[d], shapePerCTATile[d]);
    }
    auto elemTy = type.getElementType();
    bool isInt1 = elemTy.isInteger(1);
    bool isPtr = isa<triton::PointerType>(elemTy);
    auto llvmElemTyOrig = getTypeConverter()->convertType(elemTy);
    if (isInt1)
      elemTy = IntegerType::get(elemTy.getContext(), 8);
    else if (isPtr)
      elemTy = IntegerType::get(elemTy.getContext(), 64);

    auto llvmElemTy = getTypeConverter()->convertType(elemTy);

    for (unsigned ctaId = 0; ctaId < accumNumCTAsEachRep; ++ctaId) {
      auto multiDimCTAInRepId =
          getMultiDimIndex<unsigned>(ctaId, numCTAsEachRep, order);
      SmallVector<unsigned> multiDimCTAId(rank);
      for (const auto &it : llvm::enumerate(multiDimCTAInRepId)) {
        auto d = it.index();
        multiDimCTAId[d] = multiDimRepId[d] * numCTAsEachRep[d] + it.value();
      }

      auto linearCTAId =
          getLinearIndex<unsigned>(multiDimCTAId, numCTATiles, order);
      // TODO: This is actually redundant index calculation, we should
      //       consider of caching the index calculation result in case
      //       of performance issue observed.
      for (unsigned elemId = 0; elemId < accumSizePerThread; elemId += vec) {
        SmallVector<Value> multiDimOffset =
            getMultiDimOffset(layout, loc, rewriter, elemId, type,
                              multiDimCTAInRepId, shapePerCTATile);
        SmallVector<Value> multiDimOffsetWrapped = getWrappedMultiDimOffset(
            rewriter, loc, multiDimOffset, origRepShape, shapePerCTATile,
            shapePerCTA);
        Value offset = linearize(rewriter, loc, multiDimOffsetWrapped,
                                 paddedRepShape, outOrd);
        auto elemPtrTy = ptr_ty(rewriter.getContext(), 3);
        Value ptr = gep(elemPtrTy, llvmElemTy, smemBase, offset);
        auto vecTy = vec_ty(llvmElemTy, vec);
        ptr = bitcast(ptr, ptr_ty(rewriter.getContext(), 3));
        if (stNotRd) {
          Value valVec = undef(vecTy);
          for (unsigned v = 0; v < vec; ++v) {
            auto currVal = vals[elemId + linearCTAId * accumSizePerThread + v];
            if (isInt1)
              currVal = zext(llvmElemTy, currVal);
            else if (isPtr)
              currVal = ptrtoint(llvmElemTy, currVal);
            valVec = insert_element(vecTy, valVec, currVal, i32_val(v));
          }
          store(valVec, ptr);
        } else {
          Value valVec = load(vecTy, ptr);
          for (unsigned v = 0; v < vec; ++v) {
            Value currVal = extract_element(llvmElemTy, valVec, i32_val(v));
            if (isInt1)
              currVal = icmp_ne(currVal,
                                rewriter.create<LLVM::ConstantOp>(
                                    loc, i8_ty, rewriter.getI8IntegerAttr(0)));
            else if (isPtr)
              currVal = inttoptr(llvmElemTyOrig, currVal);
            vals[elemId + linearCTAId * accumSizePerThread + v] = currVal;
          }
        }
      }
    }
  }

  // blocked/dpas -> blocked/dpas.
  // Data padding in shared memory to avoid bank conflict.
  LogicalResult
  lowerDistributedToDistributed(triton::gpu::ConvertLayoutOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto typeConverter = getTypeConverter();
    RankedTensorType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();

    Value smemBase =
        LLVM::intel::getSharedMemoryBase(loc, rewriter, op.getOperation());
    auto elemPtrTy = ptr_ty(rewriter.getContext(), 3);
    smemBase = bitcast(smemBase, elemPtrTy);
    auto shape = dstTy.getShape();
    unsigned rank = dstTy.getRank();
    SmallVector<unsigned> numReplicates(rank);
    SmallVector<unsigned> inNumCTAsEachRep(rank);
    SmallVector<unsigned> outNumCTAsEachRep(rank);
    SmallVector<unsigned> inNumCTAs(rank);
    SmallVector<unsigned> outNumCTAs(rank);
    auto srcShapePerCTATile = getShapePerCTATile(srcLayout, srcTy.getShape());
    auto dstShapePerCTATile = getShapePerCTATile(dstLayout, shape);
    auto shapePerCTA = getShapePerCTA(srcLayout, shape);

    for (unsigned d = 0; d < rank; ++d) {
      unsigned inPerCTA =
          std::min<unsigned>(shapePerCTA[d], srcShapePerCTATile[d]);
      unsigned outPerCTA =
          std::min<unsigned>(shapePerCTA[d], dstShapePerCTATile[d]);
      unsigned maxPerCTA = std::max(inPerCTA, outPerCTA);
      numReplicates[d] = ceil<unsigned>(shapePerCTA[d], maxPerCTA);
      inNumCTAsEachRep[d] = maxPerCTA / inPerCTA;
      outNumCTAsEachRep[d] = maxPerCTA / outPerCTA;
      assert(maxPerCTA % inPerCTA == 0 && maxPerCTA % outPerCTA == 0);
      inNumCTAs[d] = ceil<unsigned>(shapePerCTA[d], inPerCTA);
      outNumCTAs[d] = ceil<unsigned>(shapePerCTA[d], outPerCTA);
    }
    // Potentially we need to store for multiple CTAs in this replication
    auto accumNumReplicates = product<unsigned>(numReplicates);
    auto vals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    auto scratchConfig = getScratchConfigForCvt(srcTy, dstTy);
    unsigned inVec = scratchConfig.inVec;
    unsigned outVec = scratchConfig.outVec;
    auto paddedRepShape = scratchConfig.paddedRepShape;
    auto origRepShape = scratchConfig.repShape;
    if (isa<mlir::Float8E4M3B11FNUZType, mlir::Float8E4M3FNType>(
            getElementTypeOrSelf(op.getType()))) {
      assert(inVec % 4 == 0 && "conversion not supported for FP8E4M3B15");
      assert(outVec % 4 == 0 && "conversion not supported for FP8E4M3B15");
    }

    unsigned outElems = getTotalElemsPerThread(dstTy);
    auto outOrd = getOrder(dstLayout);
    SmallVector<Value> outVals(outElems);

    for (unsigned repId = 0; repId < accumNumReplicates; ++repId) {
      auto multiDimRepId =
          getMultiDimIndex<unsigned>(repId, numReplicates, outOrd);
      if (repId != 0) {
        barrier();
      }
      if (isa<BlockedEncodingAttr>(srcLayout) ||
          isa<SliceEncodingAttr>(srcLayout) ||
          isa<DpasEncodingAttr>(srcLayout)) {
        processReplica(loc, rewriter, /*stNotRd*/ true, srcTy, inNumCTAsEachRep,
                       multiDimRepId, inVec, paddedRepShape, origRepShape,
                       outOrd, vals, smemBase);
      } else {
        llvm::report_fatal_error(
            "ConvertLayout with input layout not implemented");
        return failure();
      }

      barrier();
      if (isa<BlockedEncodingAttr>(dstLayout) ||
          isa<SliceEncodingAttr>(dstLayout) ||
          isa<DpasEncodingAttr>(dstLayout)) {
        processReplica(loc, rewriter, /*stNotRd*/ false, dstTy,
                       outNumCTAsEachRep, multiDimRepId, outVec, paddedRepShape,
                       origRepShape, outOrd, outVals, smemBase);
      } else {
        llvm::report_fatal_error(
            "ConvertLayout with output layout not implemented");
        return failure();
      }
    }

    Value result =
        packLLElements(loc, getTypeConverter(), outVals, rewriter, dstTy);
    rewriter.replaceOp(op, result);

    return success();
  }

  using ValueTable = std::map<std::pair<unsigned, unsigned>, Value>;

  ValueTable getValuesFromDpasLayoutStruct(Location loc,
                                           ConversionPatternRewriter &rewriter,
                                           Value vals,
                                           RankedTensorType srcType) const {
    SmallVector<Value> elems = unpackLLElements(loc, vals, rewriter);
    auto dpasLayout = dyn_cast<DpasEncodingAttr>(srcType.getEncoding());

    size_t totalElems = elems.size();
    auto numElemsPerOperand =
        product<unsigned>(dpasLayout.getDPASInstShapeC()) /
        dpasLayout.getSubGroupSize();
    Type elemTy =
        this->getTypeConverter()->convertType(srcType.getElementType());
    VectorType dotOpTy = vec_ty(elemTy, numElemsPerOperand);
    SmallVector<int64_t> repetitions =
        dpasLayout.getDPASRepetitions(srcType.getShape(), 2 /*operand C*/);
    ArrayRef<unsigned> repCluster = dpasLayout.getRepCluster();

    int offset = 0;
    ValueTable result;
    for (int i = 0; i < repetitions[0]; ++i) {
      for (int j = 0; j < repetitions[1]; ++j) {
        for (int repOuter = 0; repOuter < repCluster[0]; ++repOuter) {
          for (int repInner = 0; repInner < repCluster[1]; ++repInner) {
            Value matVal = rewriter.create<LLVM::UndefOp>(loc, dotOpTy);
            for (int k = 0; k < numElemsPerOperand; ++k) {
              matVal =
                  insert_element(dotOpTy, matVal, elems[offset++], i32_val(k));
            }
            result[{i * repCluster[0] + repOuter,
                    j * repCluster[1] + repInner}] = matVal;
          }
        }
      }
    }

    return result;
  }

  Value composeValuesToDotOperandLayoutStruct(
      Location loc, ConversionPatternRewriter &rewriter, const ValueTable &vals,
      RankedTensorType dstType) const {
    auto dotLayout = dyn_cast<DotOperandEncodingAttr>(dstType.getEncoding());
    auto dpasLayout = dyn_cast<DpasEncodingAttr>(dotLayout.getParent());
    unsigned opIdx = dotLayout.getOpIdx();
    SmallVector<int64_t> repetitions =
        dpasLayout.getDPASRepetitions(dstType.getShape(), opIdx);
    ArrayRef<unsigned> repCluster = dpasLayout.getRepCluster();
    unsigned repOuter = 0u;
    unsigned repInner = 0u;
    unsigned repClusterOuter = 0u;
    if (opIdx == 0) {
      // operand A
      repOuter = repetitions[0];
      repInner = repetitions[1];
      repClusterOuter = repCluster[0];
    } else {
      // operand B
      repOuter = repetitions[1];
      repInner = repetitions[0];
      repClusterOuter = repCluster[1];
    }

    // TODO: Operands B requires extra steps to combine [8, 16] to [16, 16].
    SmallVector<Value> elems;
    for (int m = 0; m < repOuter; ++m) {
      for (int k = 0; k < repInner; ++k) {
        for (int repOuterIdx = 0; repOuterIdx < repClusterOuter;
             ++repOuterIdx) {
          unsigned offsetM = m * repClusterOuter + repOuterIdx;
          unsigned offsetN = k;
          Value matVal = vals.at({offsetM, offsetN});
          VectorType vecType = cast<mlir::VectorType>(matVal.getType());
          Type valTy = vecType.getElementType();
          for (int i = 0; i < vecType.getNumElements(); ++i) {
            Value val = extract_element(valTy, matVal, i32_val(i));
            elems.push_back(val);
          }
        }
      }
    }

    Type elemTy =
        this->getTypeConverter()->convertType(dstType.getElementType());
    Type structTy = LLVM::LLVMStructType::getLiteral(
        this->getContext(), SmallVector<Type>(elems.size(), elemTy));
    return packLLElements(loc, this->getTypeConverter(), elems, rewriter,
                          structTy);
  }

  // dpas -> dot_operand
  LogicalResult
  lowerDpasToDotOperand(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    RankedTensorType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();

    if (!intel::isDpasToDotShortcut(srcTy, dstTy))
      return failure();

    // reorder the elements to match the dot_operand layout.
    ValueTable values =
        getValuesFromDpasLayoutStruct(loc, rewriter, adaptor.getSrc(), srcTy);
    Value view =
        composeValuesToDotOperandLayoutStruct(loc, rewriter, values, dstTy);

    rewriter.replaceOp(op, view);
    return success();
  }

private:
  const triton::intel::TargetInfo &targetInfo;
};

struct ConvertLayoutOpUsingLinearLayoutsConversion
    : public ConvertOpToLLVMPattern<ConvertLayoutOp> {
  // Set benefit to 2 so that this pattern applies before other convert-layout
  // conversions.  TODO(jlebar): Eventually we want this to be the only pattern.
  explicit ConvertLayoutOpUsingLinearLayoutsConversion(
      LLVMTypeConverter &typeConverter, PatternBenefit benefit = 2)
      : ConvertOpToLLVMPattern(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *ctx = op.getContext();
    const auto &shape = op.getType().getShape();
    std::optional<LinearLayout> srcLayout;
    auto srcTy = op.getSrc().getType();

    if (auto dpasLayout = dyn_cast<DpasEncodingAttr>(srcTy.getEncoding())) {
      srcLayout = gpu::DPAStoLinearLayout(shape, dpasLayout);
    } else {
      srcLayout = gpu::toLinearLayout(shape, srcTy.getEncoding());
    }

    std::optional<LinearLayout> dstLayout;
    auto dstTy = op.getType();
    if (auto dpasLayout = dyn_cast<DpasEncodingAttr>(dstTy.getEncoding())) {
      dstLayout = gpu::DPAStoLinearLayout(shape, dpasLayout);
    } else {
      dstLayout = gpu::toLinearLayout(shape, dstTy.getEncoding());
    }

    if (!srcLayout.has_value() || !dstLayout.has_value()) {
      return failure();
    }

    // There are four cases to handle.
    //
    //  1. Transfer between values in the same thread, in which case we simply
    //     reorder the elements of adaptor.getSrc().
    //  2. Transfer between values in the same warp, in which case we try to
    //     move values using warp shuffles, though if the pattern is complicated
    //     enough we may fall back to using shared memory (case 3).
    //  3. Transfer between values in the same CTA, in which case we move values
    //     through shared memory.
    //  4. Transfer between values in different CTAs, in which case we move
    //     values through distributed shared memory.
    //
    // We can tell which case we're in by examining `conversion`.  If e.g. the
    // block -> block mapping is {1, 2, 4, ...} then there's no movement between
    // data in different CTAs and we know we're not in case 4.
    LinearLayout conversion = srcLayout->invertAndCompose(*dstLayout);

    int numLanes = conversion.getInDimSize(str_attr("lane"));
    int numWarps = conversion.getInDimSize(str_attr("warp"));
    int numBlocks = conversion.getInDimSize(str_attr("block"));

    StringAttr kLane = str_attr("lane");
    StringAttr kWarp = str_attr("warp");
    StringAttr kBlock = str_attr("block");

    // TODO(jlebar): These checks are overly-restrictive.  For example, we can
    // transfer by shuffling registers (case 1) if and only if all of the bases
    // for `register` have 0s for lane, warp, and block.  But the check below is
    // stronger than this, checking also that the choice of lane/warp/block does
    // not affect the permutation of registers.  If we allow different
    // lane/warp/blocks to have different permutations, we can generalize this.
    if (std::optional<LinearLayout> c = conversion.divideRight(
            LinearLayout::identity1D(numLanes, kLane, kLane) *
            LinearLayout::identity1D(numWarps, kWarp, kWarp) *
            LinearLayout::identity1D(numBlocks, kBlock, kBlock));
        c.has_value()) {
      return transferWithinThread(*c, op, adaptor, rewriter);
    }

    if (std::optional<LinearLayout> c = conversion.divideRight(
            LinearLayout::identity1D(numWarps, kWarp, kWarp) *
            LinearLayout::identity1D(numBlocks, kBlock, kBlock));
        c.has_value()) {
      return transferWithinLane(*c, op, adaptor, rewriter);
    }

    if (std::optional<LinearLayout> c = conversion.divideRight(
            LinearLayout::identity1D(numBlocks, kBlock, kBlock));
        c.has_value()) {
      return transferWithinBlock(*c, op, adaptor, rewriter);
    }

    return transferWithinBlockGroup(conversion, op, adaptor, rewriter);
  }

  LogicalResult
  transferWithinThread(const LinearLayout &conversion, ConvertLayoutOp op,
                       OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const {
    MLIRContext *ctx = op.getContext();
    auto loc = op.getLoc();
    StringAttr kRegister = str_attr("register");

    assert(!cvtNeedsSharedMemory(op.getSrc().getType(), op.getType()));
    assert(ArrayRef(to_vector(conversion.getInDimNames())) ==
           ArrayRef{kRegister});
    assert(ArrayRef(to_vector(conversion.getOutDimNames())) ==
           ArrayRef{kRegister});

    auto inVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    SmallVector<Value> outVals(conversion.getOutDimSize(kRegister));
    for (int i = 0; i < conversion.getInDimSize(kRegister); i++) {
      auto dstIdx = conversion.apply({{kRegister, i}});
      outVals[dstIdx.begin()->second] = inVals[i];
    }
    Value result = packLLElements(loc, getTypeConverter(), outVals, rewriter,
                                  op.getType());
    rewriter.replaceOp(op, result);
    return success();
  }

  LogicalResult transferWithinLane(const LinearLayout &conversion,
                                   ConvertLayoutOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
    // TODO(jlebar): Implement me.
    return failure();
  }

  LogicalResult transferWithinBlock(const LinearLayout &conversion,
                                    ConvertLayoutOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
    // TODO(jlebar): Implement me.
    return failure();
  }

  LogicalResult
  transferWithinBlockGroup(const LinearLayout &conversion, ConvertLayoutOp op,
                           OpAdaptor adaptor,
                           ConversionPatternRewriter &rewriter) const {
    // TODO(jlebar): Implement me.
    return failure();
  }
};

} // namespace
} // namespace mlir::triton::gpu

void mlir::triton::intel::populateConvertLayoutOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  // We prefer using the linear layout conversion, so it gets a higher benefit.
  // Eventually the LL conversion will subsume all of the others and be the only
  // one left.
  patterns.add<gpu::ConvertLayoutOpUsingLinearLayoutsConversion>(
      typeConverter, benefit.getBenefit() + 1);
  patterns.add<gpu::ConvertLayoutOpConversion>(typeConverter, targetInfo,
                                               benefit);
  patterns.add<gpu::LocalLoadOpConversion>(typeConverter, targetInfo, benefit);
}
