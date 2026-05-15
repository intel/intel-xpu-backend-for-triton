#include "PatternTritonGPUOpToLLVM.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"

using ::mlir::triton::gpu::LocalAllocOp;
using ::mlir::triton::gpu::LocalLoadOp;
using ::mlir::triton::gpu::LocalStoreOp;
using ::mlir::triton::gpu::MemDescType;

namespace {

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

// Scalar select chain replaces extractelement on <N x ptr>, which
// Intel's SPIR-V toolchain rejects.
static Value selectPartitionedSmemBase(Location loc, RewriterBase &rewriter,
                                       ArrayRef<Value> smemBases,
                                       Value partitionIdx) {
  assert(!smemBases.empty() && "expected at least one smem base");
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value result = smemBases[0];
  for (size_t p = 1; p < smemBases.size(); ++p) {
    Value cmp = b.icmp_eq(partitionIdx, b.i32_val(p));
    result = b.select(cmp, smemBases[p], result);
  }
  return result;
}

// Mirror of upstream lowerLdSt (Utility.cpp) specialized for partitioned
// shared memory, with the base pointer picked via scalar select.
static SmallVector<Value> lowerLdStPartitionedIntel(
    Location loc, MLIRContext *ctx, LinearLayout cvt, ArrayRef<Value> valsArray,
    Type llvmElemTy, ArrayRef<Value> smemBases,
    ArrayRef<std::pair<unsigned, unsigned>> paddingShifts, Value affineOffset,
    uint64_t maskSpanAffineOffset, Value laneId, Value warpId,
    RewriterBase &rewriter, const TargetInfoBase &targetInfo,
    std::optional<int> maybeMaxVecElems, Operation *localLoadOp) {
  assert(smemBases.size() > 1 && "Intel partitioned path expects >1 bases");
  auto vals = to_vector(valsArray);
  bool isStore = !vals.empty();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto smemPtrTy = ptr_ty(ctx, targetInfo.getSharedAddressSpace());
  auto kReg = str_attr("register");
  auto kLane = str_attr("lane");
  auto kWarp = str_attr("warp");
  auto kBlock = str_attr("block");
  auto kOffset = str_attr("offset");
  auto kPartition = str_attr("partition");
  auto bitwidth = getIntOrFloatOrPtrBitWidth(llvmElemTy);

  assert(cvt.hasOutDim(kPartition) &&
         cvt.getOutDimSize(kPartition) == smemBases.size() &&
         "partition dimension size must match number of bases");

  auto inDimNames = to_vector(cvt.getInDimNames());
  LinearLayout partitionLayout = cvt.sublayout(inDimNames, {kPartition});

  // Query before stripping kPartition: isTrivialOver requires the dim to
  // exist as both input and output, which partitionLayout no longer does.
  bool partitionRegInvariant = cvt.sublayoutIsZero({kReg}, {kPartition});

  SmallVector<StringAttr> outDims = to_vector(cvt.getOutDimNames());
  llvm::erase(outDims, kPartition);
  cvt = cvt.sublayout(inDimNames, outDims);

  auto [elemsPerVec, permutation] =
      largestVectorisation(ctx, cvt, bitwidth, maybeMaxVecElems);

  cvt = permutation.apply(cvt);
  if (isStore) {
    vals = permutation.apply(vals);
  }

  auto tile = LinearLayout::identity1D(elemsPerVec, kReg, kOffset);
  auto quot = divideLeft(cvt, tile);
  assert(quot.has_value() && "cvt must be divisible by tile");
  LinearLayout reps = zerosLike(tile) * *quot;
  assert(reps.hasInDim(kBlock));
  LinearLayout addrLayout =
      LinearLayout({{kLane, reps.getBases().lookup(kLane)},
                    {kWarp, reps.getBases().lookup(kWarp)},
                    {kBlock, reps.getBases().lookup(kBlock)}},
                   reps.getOutDims(), false);
  auto [nAdditive, permStrides] = actionAdditiveStrides(
      reps, addrLayout, maskSpanAffineOffset, elemsPerVec);
  reps = permStrides.apply(reps);

  partitionLayout = permutation.apply(partitionLayout);
  partitionLayout = permStrides.apply(partitionLayout);
  if (isStore) {
    vals = permStrides.apply(vals);
  }

  auto i8Tile =
      LinearLayout::zeros1D(bitwidth / 8, kReg, kOffset, bitwidth / 8);
  auto i8AddrLayout = i8Tile * addrLayout;

  Value blockId = b.i32_val(0);
  bool useBlockId = !reps.isTrivialOver({kBlock});
  if (useBlockId) {
    blockId = targetInfo.getClusterCTAId(rewriter, loc);
  }

  auto baseI8AndCTA = applyLinearLayout(loc, rewriter, i8AddrLayout,
                                        {{kReg, b.i32_val(0)},
                                         {kLane, laneId},
                                         {kWarp, warpId},
                                         {kBlock, blockId}});
  auto regBaseI8 = baseI8AndCTA[0].second;
  Value targetCtaId;
  if (useBlockId) {
    targetCtaId = baseI8AndCTA[1].second;
  }

  auto affineOffsetI8 = b.mul(affineOffset, b.i32_val(bitwidth / 8));
  regBaseI8 = b.xor_(regBaseI8, affineOffsetI8);

  // Fast path: compute the base once, reuse for every tile.
  Value hoistedSmemBase;
  if (partitionRegInvariant) {
    auto partitionResult = applyLinearLayout(loc, rewriter, partitionLayout,
                                             {{kReg, b.i32_val(0)},
                                              {kLane, laneId},
                                              {kWarp, warpId},
                                              {kBlock, blockId}});
    Value partitionIdx = partitionResult[0].second;
    hoistedSmemBase =
        selectPartitionedSmemBase(loc, rewriter, smemBases, partitionIdx);
  }

  auto emitLdSt = [&](Location loc, ArrayRef<Value> vals, Value shmemAddr,
                      int idx, VectorType vecTy,
                      std::optional<Value> ctaId) -> SmallVector<Value> {
    auto length = vecTy.getNumElements();
    if (isStore) {
      Value valsVec =
          packLLVector(loc, ArrayRef<Value>(vals).slice(idx, length), rewriter);
      targetInfo.storeDShared(rewriter, loc, shmemAddr, ctaId, valsVec,
                              /*pred=*/b.true_val());
      return {};
    }
    assert(vals.empty());
    Value valsVec =
        targetInfo.loadDShared(rewriter, loc, shmemAddr, ctaId, vecTy,
                               /*pred=*/b.true_val(), localLoadOp);
    return unpackLLVector(loc, valsVec, rewriter);
  };

  SmallVector<Value> outVals;
  auto vecTy = vec_ty(llvmElemTy, elemsPerVec);
  for (int i = 0; i < cvt.getInDimSize(kReg); i += nAdditive) {
    auto idxAndBlock =
        reps.apply({{kReg, i}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
    auto regIdxI8 = idxAndBlock[0].second * (bitwidth / 8);
    Value offset = b.xor_(regBaseI8, b.i32_val(regIdxI8));
    Value ctaOffset = b.i32_val(0);
    if (useBlockId) {
      ctaOffset = b.xor_(targetCtaId, b.i32_val(idxAndBlock[1].second));
    }
    offset = applyPadding(loc, rewriter, offset, paddingShifts);
    for (int j = 0; j < nAdditive; j += elemsPerVec) {
      auto idxAndBlockAdd =
          reps.apply({{kReg, j}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
      auto regIdxAddI8 = idxAndBlockAdd[0].second * (bitwidth / 8);
      regIdxAddI8 = applyPadding(regIdxAddI8, paddingShifts);
      Value innerOffset = b.add(offset, b.i32_val(regIdxAddI8));

      Value smemBase;
      if (partitionRegInvariant) {
        smemBase = hoistedSmemBase;
      } else {
        auto partitionResult = applyLinearLayout(loc, rewriter, partitionLayout,
                                                 {{kReg, b.i32_val(i + j)},
                                                  {kLane, laneId},
                                                  {kWarp, warpId},
                                                  {kBlock, blockId}});
        Value partitionIdx = partitionResult[0].second;
        smemBase =
            selectPartitionedSmemBase(loc, rewriter, smemBases, partitionIdx);
      }

      std::optional<Value> innerCtaOffset;
      if (useBlockId) {
        innerCtaOffset = b.add(ctaOffset, b.i32_val(idxAndBlockAdd[1].second));
      }
      auto vecAddr = b.gep(smemPtrTy, i8_ty, smemBase, innerOffset,
                           LLVM::GEPNoWrapFlags::inbounds);
      llvm::append_range(
          outVals, emitLdSt(loc, vals, vecAddr, i + j, vecTy, innerCtaOffset));
    }
  }

  if (!isStore) {
    auto invPermStrides = permStrides.inverse();
    outVals = invPermStrides.apply(outVals);
    auto invPerm = permutation.inverse();
    outVals = invPerm.apply(outVals);
  }
  return outVals;
}

// Mirror of upstream lowerLocalLdSt (Utility.cpp): strip broadcasts,
// collect padding info, then invoke the partitioned tile loop.
static SmallVector<Value> lowerPartitionedLocalLdSt(
    Location loc, MLIRContext *ctx, LinearLayout cvt, ArrayRef<Value> valsArray,
    Type llvmElemTy, MemDescType srcTy, SharedMemoryObject smemObj,
    RewriterBase &rewriter, const TargetInfoBase &targetInfo,
    Operation *localLoadOp) {
  bool isStore = !valsArray.empty();
  auto removeBroadcastSrc = actionRemoveBroadcastedRegs(cvt);
  if (!removeBroadcastSrc.isIdentity()) {
    auto prmtCvt = removeBroadcastSrc.apply(cvt);
    auto inVals = to_vector(valsArray);
    if (isStore) {
      inVals = removeBroadcastSrc.apply(inVals);
    }
    auto outVals =
        lowerPartitionedLocalLdSt(loc, ctx, prmtCvt, inVals, llvmElemTy, srcTy,
                                  smemObj, rewriter, targetInfo, localLoadOp);
    if (!isStore) {
      outVals = broadcastAs(outVals, cvt);
    }
    return outVals;
  }

  std::optional<int> maybeMaxVecElems;
  SmallVector<std::pair<unsigned, unsigned>> paddingShifts;
  if (isPaddedEncoding(srcTy.getEncoding())) {
    maybeMaxVecElems = getMinInterval(srcTy.getEncoding());
    auto bitwidth = getIntOrFloatOrPtrBitWidth(llvmElemTy);
    paddingShifts = getPaddedSharedShifts(srcTy.getEncoding(), bitwidth,
                                          /*offsetInBytes=*/true);
  }

  SmallVector<Value> smemBases(smemObj.getBases().begin(),
                               smemObj.getBases().end());
  auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
  return lowerLdStPartitionedIntel(
      loc, ctx, cvt, valsArray, llvmElemTy, smemBases, paddingShifts,
      smemObj.getShmemOffset(loc, rewriter, srcTy),
      smemObj.getMaskSpanOffsets(srcTy), laneId, warpId, rewriter, targetInfo,
      maybeMaxVecElems, localLoadOp);
}

// Partitioned local_load. Non-partitioned falls through to upstream.
struct IntelLocalLoadOpConversion : public ConvertOpToLLVMPattern<LocalLoadOp> {
  IntelLocalLoadOpConversion(LLVMTypeConverter &typeConverter,
                             const TargetInfoBase &targetInfo,
                             PatternBenefit benefit)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto memDescTy = cast<MemDescType>(op.getSrc().getType());
    auto regTy = cast<RankedTensorType>(op.getResult().getType());
    const auto *typeConverter = getTypeConverter();

    auto llvmElemTy = typeConverter->convertType(memDescTy.getElementType());
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                         llvmElemTy, rewriter);

    if (smemObj.getBases().size() <= 1)
      return rewriter.notifyMatchFailure(op, "not partitioned shared memory");

    auto regLayout = toLinearLayout(regTy);
    auto sharedLayout = isPaddedEncoding(memDescTy.getEncoding())
                            ? paddedLinearLayout(memDescTy)
                            : toLinearLayout(memDescTy);
    auto cvt = regLayout.invertAndCompose(sharedLayout);

    auto kBlock = str_attr("block");
    if (!cvt.isTrivialOver({kBlock}))
      return rewriter.notifyMatchFailure(op, "non-trivial block dimension");

    SmallVector<Value> outVals =
        lowerPartitionedLocalLdSt(loc, ctx, cvt, /*valsArray=*/{}, llvmElemTy,
                                  memDescTy, smemObj, rewriter, targetInfo, op);

    Value result = packLLElements(loc, typeConverter, outVals, rewriter, regTy);
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

// Partitioned local_store. Non-partitioned falls through to upstream.
struct IntelLocalStoreOpConversion
    : public ConvertOpToLLVMPattern<LocalStoreOp> {
  IntelLocalStoreOpConversion(const LLVMTypeConverter &typeConverter,
                              const TargetInfoBase &targetInfo,
                              PatternBenefit benefit)
      : ConvertOpToLLVMPattern<LocalStoreOp>(typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(LocalStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto memDescTy = cast<MemDescType>(op.getDst().getType());
    auto regTy = cast<RankedTensorType>(op.getSrc().getType());
    const auto *typeConverter = getTypeConverter();

    auto llvmElemTy = typeConverter->convertType(memDescTy.getElementType());
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getDst(),
                                                         llvmElemTy, rewriter);

    if (smemObj.getBases().size() <= 1)
      return rewriter.notifyMatchFailure(op, "not partitioned shared memory");

    auto regLayout = toLinearLayout(regTy);
    auto sharedLayout = isPaddedEncoding(memDescTy.getEncoding())
                            ? paddedLinearLayout(memDescTy)
                            : toLinearLayout(memDescTy);
    auto cvt = regLayout.invertAndCompose(sharedLayout);

    auto kBlock = str_attr("block");
    if (!cvt.isTrivialOver({kBlock}))
      return rewriter.notifyMatchFailure(op, "non-trivial block dimension");

    auto inVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    (void)lowerPartitionedLocalLdSt(loc, ctx, cvt, inVals, llvmElemTy,
                                    memDescTy, smemObj, rewriter, targetInfo,
                                    /*localLoadOp=*/nullptr);

    rewriter.eraseOp(op);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

// Partitioned local_alloc with init value. Upstream LocalAllocOp lowering
// emits the init store via lowerLocalStore -> extractelement on <N x ptr>,
// so intercept here and route through the partitioned path.
struct IntelLocalAllocOpConversion
    : public ConvertOpToLLVMPattern<LocalAllocOp> {
  IntelLocalAllocOpConversion(const LLVMTypeConverter &typeConverter,
                              const TargetInfoBase &targetInfo,
                              PatternBenefit benefit)
      : ConvertOpToLLVMPattern<LocalAllocOp>(typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(LocalAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.isSharedMemoryAlloc())
      return rewriter.notifyMatchFailure(op, "not a shared-memory alloc");
    // Allocs without an init value have no store to intercept.
    if (!op.getSrc())
      return rewriter.notifyMatchFailure(op, "no initial value");

    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto memDescTy = cast<MemDescType>(op.getType());
    const auto *typeConverter = getTypeConverter();
    auto llvmElemTy = typeConverter->convertType(memDescTy.getElementType());

    SmallVector<Value> smemBases = LLVM::getSharedMemoryBases(
        loc, rewriter, targetInfo, op.getOperation());
    if (smemBases.size() <= 1)
      return rewriter.notifyMatchFailure(op, "not partitioned shared memory");

    auto regTy = cast<RankedTensorType>(op.getSrc().getType());
    auto regLayout = toLinearLayout(regTy);
    auto sharedLayout = isPaddedEncoding(memDescTy.getEncoding())
                            ? paddedLinearLayout(memDescTy)
                            : toLinearLayout(memDescTy);
    auto cvt = regLayout.invertAndCompose(sharedLayout);

    auto kBlock = str_attr("block");
    if (!cvt.isTrivialOver({kBlock}))
      return rewriter.notifyMatchFailure(op, "non-trivial block dimension");

    auto smemObj = SharedMemoryObject(smemBases, llvmElemTy,
                                      memDescTy.getRank(), loc, rewriter);
    auto inVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    (void)lowerPartitionedLocalLdSt(loc, ctx, cvt, inVals, llvmElemTy,
                                    memDescTy, smemObj, rewriter, targetInfo,
                                    /*localLoadOp=*/nullptr);

    Value retVal = getStructFromSharedMemoryObject(loc, smemObj, rewriter);
    rewriter.replaceOp(op, retVal);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

} // namespace

void mlir::triton::intel::populateMemoryOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfoBase &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<IntelLocalAllocOpConversion, IntelLocalLoadOpConversion,
               IntelLocalStoreOpConversion>(typeConverter, targetInfo, benefit);
}
