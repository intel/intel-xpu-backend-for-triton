#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "Utility.h"

#include "llvm/ADT/TypeSwitch.h"

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
using ::mlir::triton::gpu::SharedEncodingAttr;
using ::mlir::triton::gpu::intel::DpasEncodingAttr;

namespace mlir::triton::gpu {
namespace {

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
    if (isa<BlockedEncodingAttr, MmaEncodingTrait, SliceEncodingAttr>(
            srcLayout) &&
        isa<BlockedEncodingAttr, MmaEncodingTrait, SliceEncodingAttr>(
            dstLayout)) {
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
        SmallVector<Value> multiDimOffsetWrapped =
            mlir::LLVM::getWrappedMultiDimOffset(rewriter, loc, multiDimOffset,
                                                 origRepShape, shapePerCTATile,
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

    Value smemBase = LLVM::intel::getSharedMemoryBase(loc, rewriter, targetInfo,
                                                      op.getOperation());
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
    const auto &paddedRepShape = scratchConfig.paddedRepShape;
    const auto &origRepShape = scratchConfig.repShape;
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
  constexpr static unsigned minSubGroupTransposeWidth = 8;

  const TargetInfoBase &targetInfo;

  // Set benefit to 2 so that this pattern applies before other convert-layout
  // conversions.  TODO(jlebar): Eventually we want this to be the only pattern.
  ConvertLayoutOpUsingLinearLayoutsConversion(LLVMTypeConverter &typeConverter,
                                              const TargetInfoBase &targetInfo,
                                              PatternBenefit benefit = 2)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  bool isSubGroupTranspose(const LinearLayout &srcLayout,
                           const LinearLayout &dstLayout) const {
    MLIRContext *ctx = srcLayout.getInDimNames().begin()->getContext();
    StringAttr kRegister = str_attr("register");
    StringAttr kLane = str_attr("lane");
    StringAttr kWarp = str_attr("warp");
    StringAttr kBlock = str_attr("block");

    LinearLayout comp = dstLayout.invertAndCompose(srcLayout);
    std::optional<LinearLayout> conversion = comp.divideRight(
        LinearLayout::identity1D(comp.getInDimSize(kWarp), kWarp, kWarp) *
        LinearLayout::identity1D(comp.getInDimSize(kBlock), kBlock, kBlock));
    assert(conversion && "Expecting valid conversion");
    // Expected conversion is:
    // - register=1 -> (0, 1)
    // ...
    // - register=i -> (0, 2**(i-1))
    // ...
    // - register=N -> (0, 2**(N-1))
    // - lane=1 -> (0, 1)
    // ...
    // - lane=j -> (2**(j-1), 0)
    // ...
    //   lane=M -> (2**(M-1), 0)
    // where out dims are: [register (size 2**(N-1)), lane (size 2**(M-1))]
    //
    // With N = M.
    const auto buildBasis = [&](int32_t size, std::size_t index) {
      std::vector<std::vector<int32_t>> basis;
      std::vector<int32_t> curr(2);
      for (int32_t i = 1; i < size; i *= 2) {
        curr[index] = i;
        basis.push_back(curr);
      }
      return basis;
    };

    constexpr std::size_t laneIndex = 0;
    constexpr std::size_t registerIndex = 1;
    int32_t size = conversion->getInDimSize(kLane);
    std::array<std::pair<StringAttr, std::vector<std::vector<int32_t>>>, 2>
        bases{{{kRegister, buildBasis(size, registerIndex)},
               {kLane, buildBasis(size, laneIndex)}}};
    std::array<StringAttr, 2> outDimNames{kRegister, kLane};
    return conversion == LinearLayout(bases, outDimNames);
  }

  LogicalResult
  matchAndRewrite(ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *ctx = op.getContext();

    const auto &shape = op.getType().getShape();
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();
    std::optional<LinearLayout> srcLayout =
        toLinearLayout(shape, srcTy.getEncoding());
    std::optional<LinearLayout> dstLayout =
        toLinearLayout(shape, dstTy.getEncoding());
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
    // We can tell which case we're in by examining `conversion`.
    // For example, if the block -> block mapping is an identity layout: {1, 2,
    // 4, ...}, then there's no movement between data in different CTAs, and we
    // know we're not in case 4.
    if (cvtReordersRegisters(srcTy, dstTy)) { // Case 1.
      return transferWithinThread(op, *srcLayout, *dstLayout, adaptor,
                                  rewriter);
    }

    if (cvtNeedsWarpShuffle(srcTy, dstTy)) { // Case 2.
      return transferWithinLane(op, *srcLayout, *dstLayout, adaptor, rewriter);
    }

    // TODO: match transferWithinBlockOrGroup from
    // TritonGPUToLLVM/ConvertLayoutOpToLLVM.cpp
    return transferWithinBlockGroup(op, *srcLayout, *dstLayout, adaptor,
                                    rewriter);
  }

  LogicalResult
  transferWithinThread(ConvertLayoutOp op, const LinearLayout &srcLayout,
                       const LinearLayout &dstLayout, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const {
    MLIRContext *ctx = op.getContext();
    auto loc = op.getLoc();
    StringAttr kRegister = str_attr("register");
    StringAttr kLane = str_attr("lane");
    StringAttr kWarp = str_attr("warp");
    StringAttr kBlock = str_attr("block");

    // There are three possible cases:
    //
    // 1. `srcLayout` has the same number of registers as `dstLayout`.
    // 2. `srcLayout` has fewer registers than `dstLayout`.
    // 3. `srcLayout` has more registers than `dstLayout`.
    //
    // In the second case `srcLayout . dstLayout^-1` is not surjective
    // because not all destination registers are covered.
    // Since the goal is to cover all of the destination
    // registers, we can instead use `dstLayout . srcLayout^-1`.
    LinearLayout conversion = dstLayout.invertAndCompose(srcLayout);
    auto dstToSrc = conversion.divideRight(
        LinearLayout::identity1D(conversion.getInDimSize(kLane), kLane, kLane) *
        LinearLayout::identity1D(conversion.getInDimSize(kWarp), kWarp, kWarp) *
        LinearLayout::identity1D(conversion.getInDimSize(kBlock), kBlock,
                                 kBlock));

    assert(!cvtNeedsSharedMemory(op.getSrc().getType(), op.getType()));
    assert(ArrayRef(to_vector(dstToSrc->getInDimNames())) ==
           ArrayRef{kRegister});
    assert(ArrayRef(to_vector(dstToSrc->getOutDimNames())) ==
           ArrayRef{kRegister});

    auto inVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    SmallVector<Value> outVals;
    outVals.resize(dstToSrc->getInDimSize(kRegister));
    for (int i = 0; i < dstToSrc->getInDimSize(kRegister); i++) {
      auto srcIdx = dstToSrc->apply({{kRegister, i}});
      outVals[i] = inVals[srcIdx.begin()->second];
    }
    Value result = packLLElements(loc, getTypeConverter(), outVals, rewriter,
                                  op.getType());
    rewriter.replaceOp(op, result);
    return success();
  }

  bool isSupportedSubGroupTranspose(ConvertLayoutOp op,
                                    OpAdaptor adaptor) const {
    auto srcType = cast<LLVM::LLVMStructType>(adaptor.getSrc().getType());
    ArrayRef<Type> body = srcType.getBody();
    // TODO: Support more configurations.
    auto mod = op->getParentOfType<ModuleOp>();
    int threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    if (body.size() != threadsPerWarp)
      return false;
    return TypeSwitch<Type, bool>(body.front())
        .Case([this](FloatType floatTy) {
          // Support via bitcasting to integer type.
          return isValidTypeForSubGroupTranspose(
              IntegerType::get(floatTy.getContext(), floatTy.getWidth()));
        })
        .Case([this](IntegerType intTy) {
          // Support via extending to supported type.
          return isValidTypeForSubGroupTranspose(intTy) ||
                 intTy.getWidth() < minSubGroupTransposeWidth;
        })
        .Case([](LLVM::LLVMPointerType) {
          // Support via ptrtoint
          return true;
        })
        .Default(false);
  }

  LogicalResult transferWithinLane(ConvertLayoutOp op,
                                   const LinearLayout &srcLayout,
                                   const LinearLayout &dstLayout,
                                   OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
    // If the operation is a supported sub-group transposition, perform via SLM.
    if (isSubGroupTranspose(srcLayout, dstLayout) &&
        isSupportedSubGroupTranspose(op, adaptor)) {
      performSubGroupTranspose(op, srcLayout, dstLayout, adaptor, rewriter);
      return success();
    }
    // TODO(jlebar): Implement me.
    return failure();
  }

  bool isValidTypeForSubGroupTranspose(Type type) const {
    return TypeSwitch<Type, bool>(type)
        .Case([](IntegerType intTy) {
          unsigned width = intTy.getWidth();
          return width == 8 || width == 16 || width == 32 || width == 64;
        })
        .Default(false);
  }

  void performSubGroupTranspose(ConvertLayoutOp op,
                                const LinearLayout &srcLayout,
                                const LinearLayout &dstLayout,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    assert(isSubGroupTranspose(srcLayout, dstLayout) &&
           "Expecting sub-group transpose");
    assert(isSupportedSubGroupTranspose(op, adaptor) &&
           "Expecting supported sub-group transpose");

    Location loc = op.getLoc();

    SmallVector<Value> inVals =
        unpackLLElements(loc, adaptor.getSrc(), rewriter);

    auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
    Type origElemTy = inVals.front().getType();

    TypeSwitch<Type>(origElemTy)
        .Case([&](FloatType floatTy) {
          // TODO: Support FP4.
          Type dstType = int_ty(floatTy.getWidth());
          assert(isValidTypeForSubGroupTranspose(dstType) &&
                 "Expecting valid type");
          llvm::transform(inVals, std::begin(inVals), [&](Value val) -> Value {
            return bitcast(val, dstType);
          });
        })
        .Case([&](IntegerType intTy) {
          if (isValidTypeForSubGroupTranspose(intTy))
            return;
          assert(intTy.getWidth() < minSubGroupTransposeWidth &&
                 "Expecting type to extend to i8");
          Type dstType = i8_ty;
          llvm::transform(inVals, std::begin(inVals), [&](Value val) -> Value {
            return zext(dstType, val);
          });
        })
        .Case([&](LLVM::LLVMPointerType) {
          Type dstType = i64_ty;
          assert(isValidTypeForSubGroupTranspose(dstType) &&
                 "i64 type should be supported");
          llvm::transform(inVals, std::begin(inVals), [&](Value val) -> Value {
            return ptrtoint(dstType, val);
          });
        })
        .Default([](auto) { llvm_unreachable("Unsupported type"); });

    SmallVector<Value> outVals =
        performSubGroupTranspose(loc, inVals, rewriter);

    TypeSwitch<Type>(origElemTy)
        .Case([&](FloatType floatTy) {
          llvm::transform(
              outVals, std::begin(outVals),
              [&](Value val) -> Value { return bitcast(val, origElemTy); });
        })
        .Case([&](IntegerType intTy) {
          // Check whether conversion took place.
          if (intTy == outVals.front().getType())
            return;
          llvm::transform(
              outVals, std::begin(outVals),
              [&](Value val) -> Value { return trunc(origElemTy, val); });
        })
        .Case([&](LLVM::LLVMPointerType ptrTy) {
          llvm::transform(
              outVals, std::begin(outVals),
              [&](Value val) -> Value { return inttoptr(ptrTy, val); });
        })
        .Default([](auto) { llvm_unreachable("Unsupported type"); });

    Value result = packLLElements(loc, getTypeConverter(), outVals, rewriter,
                                  op.getType());
    rewriter.replaceOp(op, result);
  }

  VectorType
  getTypeForSubGroupTranspose(ArrayRef<Value> inVals,
                              ConversionPatternRewriter &rewriter) const {
    auto elementTy = cast<IntegerType>(inVals.front().getType());
    return elementTy.getWidth() <= 16 ? vec_ty(elementTy, 16)
                                      : vec_ty(elementTy, 8);
  }

  Value wrapInVector(Location loc, VectorType type, ArrayRef<Value> values,
                     ConversionPatternRewriter &rewriter) const {
    assert(type.getShape()[0] == values.size() && "Size mismatch");
    Value res = rewriter.create<LLVM::PoisonOp>(loc, type);
    for (auto [index, val] : llvm::enumerate(values))
      res = insert_element(res, val, i32_val(index));
    return res;
  }

  SmallVector<Value>
  unwrapFromVector(Location loc, Value vec,
                   ConversionPatternRewriter &rewriter) const {
    SmallVector<Value> res;
    for (unsigned i = 0, n = cast<VectorType>(vec.getType()).getShape()[0];
         i < n; ++i)
      res.push_back(extract_element(vec, i32_val(i)));
    return res;
  }

  SmallVector<Value>
  performSubGroupTranspose(Location loc, ArrayRef<Value> inVals,
                           ConversionPatternRewriter &rewriter) const {
    VectorType opType = getTypeForSubGroupTranspose(inVals, rewriter);
    auto mod = rewriter.getInsertionPoint()->getParentOfType<ModuleOp>();
    unsigned vecWidth = opType.getShape()[0];

    Value smemBase = LLVM::intel::getSharedMemoryBase(
        loc, rewriter, targetInfo, &*rewriter.getInsertionPoint());
    Type ptrType = smemBase.getType();

    int threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    int offset = threadsPerWarp;
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
    Value subGroupOffset = mul(sgStride, subGroupId);
    Type elementType = opType.getElementType();
    Value subGroupBasePtr = gep(ptrType, elementType, smemBase,
                                ValueRange{subGroupOffset}, /*inbounds=*/true);
    Value base = subGroupBasePtr;
    // Store in matrix, transposed
    for (ArrayRef<Value> vals = inVals; !vals.empty();
         vals = vals.drop_front(vecWidth)) {
      ArrayRef<Value> curr = vals.take_front(vecWidth);
      Value vec = wrapInVector(loc, opType, curr, rewriter);
      rewriter.create<TritonGEN::SIMDBlockWriteOp>(loc, base, vec);
      base = gep(base.getType(), opType, base, ArrayRef<LLVM::GEPArg>{offset},
                 /*inbounds=*/true);
    }

    // Load from matrix, non-trasposed.
    Value workItemOffset = mul(wiStride, subGroupLocalId);
    Value workItemBasePtr = gep(ptrType, elementType, subGroupBasePtr,
                                ValueRange{workItemOffset}, /*inbounds=*/true);
    Value transposedVec =
        load(vec_ty(opType.getElementType(), inVals.size()), workItemBasePtr);

    return unwrapFromVector(loc, transposedVec, rewriter);
  }

  LogicalResult
  transferWithinBlockGroup(ConvertLayoutOp op, const LinearLayout &srcLayout,
                           const LinearLayout &dstLayout, OpAdaptor adaptor,
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
      typeConverter, targetInfo, benefit.getBenefit() + 1);
  patterns.add<gpu::ConvertLayoutOpConversion>(typeConverter, targetInfo,
                                               benefit);
}
