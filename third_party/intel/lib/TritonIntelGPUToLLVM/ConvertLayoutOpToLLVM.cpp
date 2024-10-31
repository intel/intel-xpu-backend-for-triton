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
      assert(rank == 2 || rank == 3);
      auto multiDimBase = ::intel::emitBaseIndexForLayout(
          loc, rewriter, targetInfo, layout, type, false);
      SmallVector<SmallVector<unsigned>> offsets;
      ::emitOffsetForDpasLayoutPerCTA(
          dpasLayout, offsets,
          multiDimCTAInRepId[rank - 2] * shapePerCTATile[rank - 2],
          multiDimCTAInRepId[rank - 1] * shapePerCTATile[rank - 1]);

      SmallVector<Value> multiDimOffset(rank);
      if (rank == 3)
        multiDimOffset[0] = add(multiDimBase[0], i32_val(multiDimCTAInRepId[0] *
                                                         shapePerCTATile[0]));
      multiDimOffset[rank - 2] =
          add(multiDimBase[rank - 2], i32_val(offsets[elemId][rank - 2]));
      multiDimOffset[rank - 1] =
          add(multiDimBase[rank - 1], i32_val(offsets[elemId][rank - 1]));

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

  using ValueTable = std::map<std::array<unsigned, 3>, Value>;

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
    size_t rank = repCluster.size();
    size_t outerDim = rank - 2;
    size_t innerDim = rank - 1;

    int offset = 0;
    ValueTable result;
    for (unsigned b = 0; b < repetitions[0]; ++b) {
      for (int i = 0; i < repetitions[1]; ++i) {
        for (int j = 0; j < repetitions[2]; ++j) {
          for (int repOuter = 0; repOuter < repCluster[outerDim]; ++repOuter) {
            for (int repInner = 0; repInner < repCluster[innerDim];
                 ++repInner) {
              Value matVal = rewriter.create<LLVM::UndefOp>(loc, dotOpTy);
              for (int k = 0; k < numElemsPerOperand; ++k) {
                matVal = insert_element(dotOpTy, matVal, elems[offset++],
                                        i32_val(k));
              }
              result[{b, i * repCluster[outerDim] + repOuter,
                      j * repCluster[innerDim] + repInner}] = matVal;
            }
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
    size_t rank = repCluster.size();
    unsigned repBatch = repetitions[0];
    unsigned repOuter = 0u;
    unsigned repInner = 0u;
    unsigned repClusterOuter = 0u;
    if (opIdx == 0) {
      // operand A
      repOuter = repetitions[1];
      repInner = repetitions[2];
      repClusterOuter = repCluster[rank - 2];
    } else {
      // operand B
      repOuter = repetitions[2];
      repInner = repetitions[1];
      repClusterOuter = repCluster[rank - 1];
    }

    // TODO: Operands B requires extra steps to combine [8, 16] to [16, 16].
    SmallVector<Value> elems;
    for (unsigned b = 0; b < repBatch; ++b) {
      for (int m = 0; m < repOuter; ++m) {
        for (int k = 0; k < repInner; ++k) {
          for (int repOuterIdx = 0; repOuterIdx < repClusterOuter;
               ++repOuterIdx) {
            unsigned offsetM = m * repClusterOuter + repOuterIdx;
            unsigned offsetN = k;
            Value matVal = vals.at({b, offsetM, offsetN});
            VectorType vecType = cast<mlir::VectorType>(matVal.getType());
            Type valTy = vecType.getElementType();
            for (int i = 0; i < vecType.getNumElements(); ++i) {
              Value val = extract_element(valTy, matVal, i32_val(i));
              elems.push_back(val);
            }
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

  // Return a vector such as:
  // [[0, 1], [0, 2], [0, 4], ..., [0, laneSize / 2], [laneSize, 0], ...,
  // [registerSize / 2, 0]],
  // i.e., mapping registers to lanes till laneSize and performing an ID
  // conversion afterwards.
  static std::vector<std::vector<int32_t>>
  buildSubGroupTransposeRegisterBases(int32_t registerSize, int32_t laneSize) {
    std::vector<std::vector<int32_t>> bases;
    std::vector<int32_t> curr(2);
    for (int32_t i = 1; i < laneSize; i *= 2) {
      curr[1] = i;
      bases.push_back(curr);
    }
    curr[1] = 0;
    for (int32_t i = laneSize; i < registerSize; i *= 2) {
      curr[0] = i;
      bases.push_back(curr);
    }
    return bases;
  }

  // Return a vector such as:
  // [[0, 1], [0, 2], [0, 4], ..., [0, laneSize / 2], [1, 0], ...,
  // [registerSize / (2 * laneSize), 0]]
  // i.e., mapping registers to lanes till laneSize and repeating the pattern
  // afterwards.
  static std::vector<std::vector<int32_t>>
  buildSubGroupShuffleRegisterBases(int32_t registerSize, int32_t laneSize) {
    std::vector<std::vector<int32_t>> bases;
    std::vector<int32_t> curr(2);
    for (int32_t i = 1; i < laneSize; i *= 2) {
      curr[1] = i;
      bases.push_back(curr);
    }
    curr[1] = 0;
    for (int32_t i = laneSize, val = 1; i < registerSize; i *= 2, val *= 2) {
      curr[0] = val;
      bases.push_back(curr);
    }
    return bases;
  }

  // Return a vector such as:
  // [[1, 0], [2, 0], [4, 0], ..., [laneSize / 2, 0]],
  // i.e., mapping lanes to registers.
  static std::vector<std::vector<int32_t>>
  buildSubGroupTransposeLaneBases(int32_t laneSize) {
    std::vector<std::vector<int32_t>> bases;
    std::vector<int32_t> curr(2);
    for (int32_t i = 1; i < laneSize; i *= 2) {
      curr[0] = i;
      bases.push_back(curr);
    }
    return bases;
  }

  bool isSubGroupTranspose(const LinearLayout &srcLayout,
                           const LinearLayout &dstLayout) const {
    MLIRContext *ctx = srcLayout.getInDimNames().begin()->getContext();
    StringAttr kRegister = str_attr("register");
    StringAttr kLane = str_attr("lane");
    StringAttr kWarp = str_attr("warp");
    StringAttr kBlock = str_attr("block");

    LinearLayout comp = dstLayout.invertAndCompose(srcLayout);
    std::optional<LinearLayout> conversion =
        comp.quotient(kBlock)->quotient(kWarp);
    assert(conversion && "Expecting valid conversion");
    // Expected conversion is:
    // - register=1 -> (0, 1)
    // ...
    // - register=2**i -> (0, 2**i)
    // ...
    // - register=M -> (0, 2**M)
    // ...
    // - register=2**k -> (2**k, 0)
    // ...
    // - register=N -> (2**N, 0)
    // - lane=1 -> (0, 1)
    // ...
    // - lane=2**j -> (2**j, 0)
    // ...
    //   lane=2**M -> (2**M, 0)
    // where out dims are: [register (size 2**(N + 1)), lane (size 2**(M + 1))]
    //
    // With N >= M.
    int32_t registerInDimSize = conversion->getInDimSize(kRegister);
    int32_t laneInDimSize = conversion->getInDimSize(kLane);
    return conversion->getBases().lookup(kRegister) ==
               buildSubGroupTransposeRegisterBases(registerInDimSize,
                                                   laneInDimSize) &&
           conversion->getBases().lookup(kLane) ==
               buildSubGroupTransposeLaneBases(laneInDimSize);
  }

  LogicalResult
  matchAndRewrite(ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *ctx = op.getContext();

    const auto &shape = op.getType().getShape();
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();

    auto conversion = minimalCvtLayout(srcTy, dstTy);
    if (!conversion.has_value()) {
      return rewriter.notifyMatchFailure(
          op, "NYI. srcTy and/or dstTy don't implement LLs yet");
    }
    LinearLayout srcLayout =
        *toLinearLayout(srcTy.getShape(), srcTy.getEncoding());
    LinearLayout dstLayout =
        *toLinearLayout(dstTy.getShape(), dstTy.getEncoding());

    StringAttr kBlock = str_attr("block");
    StringAttr kWarp = str_attr("warp");
    StringAttr kLane = str_attr("lane");
    StringAttr kRegister = str_attr("register");

    assert(to_vector(conversion->getInDimNames()) ==
           to_vector(conversion->getOutDimNames()));
    auto dims = conversion->getInDimNames();
    if (llvm::is_contained(dims, str_attr("block"))) {
      // Case 1: Transfer between values in different CTAs.
      //          This requires moving values through distributed shared memory.
      return rewriter.notifyMatchFailure(
          op, "NYI: Transfer between different CTAs");
    } else if (llvm::is_contained(dims, str_attr("warp"))) {
      return rewriter.notifyMatchFailure(
          op, "NYI: Transfer between different warps");
    } else if (llvm::is_contained(dims, str_attr("lane"))) {
      // Case 2: Transfer between values in the same CTA, in which case we move
      //         values through shared memory.
      // If the operation is a supported sub-group shuffle, perform via shuffle
      // operations.
      if (isSubGroupShuffle(srcLayout, dstLayout) &&
          isSupportedSubGroupShuffle(op, adaptor)) {
        performSubGroupShuffle(op, srcLayout, dstLayout, adaptor, rewriter);
        return success();
      }
      // If the operation is a supported sub-group transposition, perform via
      // SLM.
      if (isSubGroupTranspose(srcLayout, dstLayout) &&
          isSupportedSubGroupTranspose(op, adaptor)) {
        performSubGroupTranspose(op, srcLayout, dstLayout, adaptor, rewriter);
        return success();
      }
      // TODO(jlebar): Implement me.
      return failure();
    } else if (llvm::is_contained(dims, str_attr("register"))) {
      // Case 4. Transfer between values in the same thread, in which case we
      //         simply reorder the elements of adaptor.getSrc().
      return transferWithinThread(
          op, dstLayout.getFreeVariableMasks()[kRegister],
          dstLayout.getInDimSize(kRegister), *conversion, adaptor, rewriter);
    } else {
      // The two layouts are equivalent. We should probably remove these in
      // RemoveLayoutConversion.
      rewriter.replaceOp(op, adaptor.getSrc());
      return success();
    }
  }

  LogicalResult
  transferWithinThread(ConvertLayoutOp op, int32_t regMasks, int32_t numRegs,
                       const LinearLayout &conversion, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const {
    MLIRContext *ctx = op.getContext();
    auto loc = op.getLoc();
    StringAttr kRegister = str_attr("register");
    assert(!cvtNeedsSharedMemory(op.getSrc().getType(), op.getType()));

    auto inVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    SmallVector<Value> outVals(numRegs);
    for (int i = 0; i < outVals.size(); i++) {
      // Remove free masks from the register index
      // For example, if idx = 0b00111, and masks = 0b00100, then we get
      // 0b00011. It means that register 7 (0b111) has the same value as
      // register 3 (0b011).
      auto idx = i & (~regMasks);
      auto srcIdx = conversion.hasInDim(kRegister)
                        ? conversion.apply({{kRegister, idx}}).begin()->second
                        : idx;
      outVals[i] = inVals[srcIdx];
    }
    Value result = packLLElements(loc, getTypeConverter(), outVals, rewriter,
                                  op.getType());
    rewriter.replaceOp(op, result);
    return success();
  }

  bool isSubGroupShuffle(const LinearLayout &srcLayout,
                         const LinearLayout &dstLayout) const {
    MLIRContext *ctx = srcLayout.getInDimNames().begin()->getContext();
    StringAttr kRegister = str_attr("register");
    StringAttr kLane = str_attr("lane");
    StringAttr kWarp = str_attr("warp");
    StringAttr kBlock = str_attr("block");

    LinearLayout comp = dstLayout.invertAndCompose(srcLayout);
    std::optional<LinearLayout> conversion =
        comp.quotient(kBlock)->quotient(kWarp);
    assert(conversion && "Expecting valid conversion");
    // TODO: Support more kind of shuffles.
    // Expected conversion is:
    // - register=1 -> (0, 1)
    // ...
    // - register=2**i -> (0, 2**i)
    // ...
    // - register=M -> (0, 2**M)
    // ...
    // - register=2**k -> (2**(k-M), 0)
    // ...
    // - register=2**N -> (2**(N-M), 0)
    // - lane=1 -> (0, 0)
    // ...
    // - lane=2**j -> (0, 0)
    // ...
    //   lane=2**M -> (0, 0)
    // where out dims are: [register (size 2**(N - M)), lane (size 2**(M + 1))]
    //
    // With N >= M.
    int32_t registerInDimSize = conversion->getInDimSize(kRegister);
    int32_t laneOutDimSize = conversion->getOutDimSize(kLane);
    return conversion->sublayoutIsZero({kLane}, {kRegister, kLane}) &&
           conversion->getBases().lookup(kRegister) ==
               buildSubGroupShuffleRegisterBases(registerInDimSize,
                                                 laneOutDimSize);
  }

  bool isSupportedSubGroupShuffle(ConvertLayoutOp, OpAdaptor) const {
    // TODO: Limit when sub-group shuffles get more complex.
    // We do not need to limit by type here as `gpu.shuffle` conversion will
    // fail for us.
    return true;
  }

  void performSubGroupShuffle(ConvertLayoutOp op, const LinearLayout &srcLayout,
                              const LinearLayout &dstLayout, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const {
    assert(isSubGroupShuffle(srcLayout, dstLayout) &&
           "Expecting sub-group shuffle");
    assert(isSupportedSubGroupShuffle(op, adaptor) &&
           "Expecting supported sub-group shuffle");

    MLIRContext *ctx = op->getContext();
    StringAttr kLane = str_attr("lane");
    StringAttr kWarp = str_attr("warp");
    StringAttr kBlock = str_attr("block");
    LinearLayout comp = dstLayout.invertAndCompose(srcLayout);
    LinearLayout conversion = *comp.quotient(kBlock)->quotient(kWarp);
    int32_t subGroupSize = conversion.getOutDimSize(kLane);

    Location loc = op.getLoc();

    SmallVector<Value> inVals =
        unpackLLElements(loc, adaptor.getSrc(), rewriter);

    // TODO: Drop 'BFloat16Type' and 'IntegerType' cases when supported at MLIR
    // upstream level. We are not enabling support for all types here as that
    // should be done upstream.
    Type origElemTy = inVals.front().getType();
    TypeSwitch<Type>(origElemTy)
        .Case([&](BFloat16Type) {
          auto intTy = i16_ty;
          llvm::transform(inVals, std::begin(inVals), [&](Value val) -> Value {
            return bitcast(val, intTy);
          });
        })
        .Case([&](IntegerType intTy) {
          constexpr unsigned minWidth = 8;
          if (intTy.getWidth() >= minWidth)
            return;
          auto dstTy = i8_ty;
          llvm::transform(inVals, std::begin(inVals),
                          [&](Value val) -> Value { return zext(dstTy, val); });
        })
        .Case([&](LLVM::LLVMPointerType) {
          Type dstType = i64_ty;
          llvm::transform(inVals, std::begin(inVals), [&](Value val) -> Value {
            return ptrtoint(dstType, val);
          });
        });

    SmallVector<Value> outVals =
        performSubGroupShuffle(loc, inVals, subGroupSize, rewriter);

    // TODO: Drop 'BFloat16Type' and 'IntegerType' cases when supported at MLIR
    // upstream level. We are not enabling support for all types here as that
    // should be done upstream.
    TypeSwitch<Type>(origElemTy)
        .Case([&](BFloat16Type) {
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
        });

    Value result = packLLElements(loc, getTypeConverter(), outVals, rewriter,
                                  op.getType());
    rewriter.replaceOp(op, result);
  }

  SmallVector<Value>
  performSubGroupShuffle(Location loc, ArrayRef<Value> inVals,
                         int32_t subGroupSize,
                         ConversionPatternRewriter &rewriter) const {
    SmallVector<Value> res;
    Value width = i32_val(subGroupSize);
    for (Value val : inVals) {
      for (int32_t i = 0; i < subGroupSize; ++i)
        res.push_back(
            rewriter
                .create<mlir::gpu::ShuffleOp>(loc, val, i32_val(i), width,
                                              mlir::gpu::ShuffleMode::IDX)
                .getShuffleResult());
    }
    return res;
  }

  bool isSupportedSubGroupTranspose(ConvertLayoutOp op,
                                    OpAdaptor adaptor) const {
    auto srcType = cast<LLVM::LLVMStructType>(adaptor.getSrc().getType());
    ArrayRef<Type> body = srcType.getBody();
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
  unwrapFromVectors(Location loc, ArrayRef<Value> vecs,
                    ConversionPatternRewriter &rewriter) const {
    SmallVector<Value> res;
    for (Value vec : vecs) {
      for (unsigned i = 0, n = cast<VectorType>(vec.getType()).getShape()[0];
           i < n; ++i)
        res.push_back(extract_element(vec, i32_val(i)));
    }
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

    int numElements = inVals.size();
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
        loc, offsetType, threadsPerWarp * numElements);
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
    // As per SIMD block semantics, we have stored the elements in a matrix of
    // `Nxsub_group_size` size, so we need to load back in blocks of
    // `sub_group_size` (`N/sub_group_size` loads).
    Value workItemOffset = mul(wiStride, subGroupLocalId);
    Value workItemBasePtr = gep(ptrType, elementType, subGroupBasePtr,
                                ValueRange{workItemOffset}, /*inbounds=*/true);
    SmallVector<Value> transposedVecs;
    Type loadTy = vec_ty(opType.getElementType(), threadsPerWarp);
    for (std::size_t i = 0, n = inVals.size(); i < n; i += threadsPerWarp) {
      transposedVecs.push_back(load(loadTy, workItemBasePtr));
      workItemBasePtr = gep(ptrType, loadTy, workItemBasePtr,
                            ArrayRef<LLVM::GEPArg>{offset}, /*inbounds=*/true);
    }
    return unwrapFromVectors(loc, transposedVecs, rewriter);
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
