#include "PatternTritonGPUOpToLLVM.h"
#include "lib/Conversion/TritonGPUToLLVM/ReduceScanCommon.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Tools/LayoutUtils.h"

#include <numeric>

using namespace mlir;
using namespace mlir::triton;

using ::mlir::LLVM::linearize;
using ::mlir::triton::gpu::DistributedEncodingTrait;
using ::mlir::triton::gpu::getTotalElemsPerThread;

// Enable to A/B test cross-warp reduction logic.
//
// When enabled, we keep Intel step 1/2 (within-thread + within-warp) intact,
// but replace step 3 (cross-warp) with the common ReduceOpToLLVM.cpp logic:
// convert_layout through shared memory into a temporary layout, then perform up
// to two additional warp reductions until the reduction axis size becomes 1.
#ifndef TRITON_INTEL_REDUCE_USE_COMMON_CROSS_WARP
#define TRITON_INTEL_REDUCE_USE_COMMON_CROSS_WARP 1
#endif

namespace {
struct ReduceOpConversion
    : public ConvertTritonGPUReduceScanToLLVMPattern<triton::ReduceOp> {
public:
  ReduceOpConversion(LLVMTypeConverter &typeConverter,
                     const TargetInfoBase &targetInfo, PatternBenefit benefit)
      : ConvertTritonGPUReduceScanToLLVMPattern<triton::ReduceOp>(typeConverter,
                                                                  benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ReduceOpHelper helper(op);
    assert(helper.isReduceWithinCTA() &&
           "Unexpected srcLayout in ReduceOpConversion");
    Location loc = op->getLoc();

    auto srcValues = unpackInputs(loc, op, adaptor, rewriter);
    std::map<SmallVector<unsigned>, SmallVector<Value>> accs;
    std::map<SmallVector<unsigned>, SmallVector<Value>> indices;

    // Step 1: reduce all the values along axis within each thread.
    reduceWithinThreads(helper, srcValues, accs, indices, rewriter);

    // Step 2: reduce across threads within a warp.
    reduceWithinWarps(helper, accs, rewriter);

    if (helper.isWarpSynchronous()) {
      packResults(helper, accs, rewriter);
      return success();
    }

    // Step 3: reduce across warps.
#if TRITON_INTEL_REDUCE_USE_COMMON_CROSS_WARP
    if (failed(reduceAcrossWarpsLikeCommon(helper, accs, rewriter)))
      return failure();
#else
    reduceAcrossWarpsIntel(helper, accs, indices, rewriter);
#endif

    return success();
  }

private:
  const TargetInfoBase &targetInfo;

  void accumulate(Location loc, ConversionPatternRewriter &rewriter,
                  Region &combineOp, SmallVector<Value> &acc, ValueRange cur,
                  Value pred = {}) const {
    auto results = applyCombineOp(loc, rewriter, combineOp, acc, cur, pred);
    if (acc.size() < results.size()) {
      acc.resize(results.size());
    }
    for (unsigned i = 0; i < acc.size(); ++i) {
      acc[i] = results[i];
    }
  }

  SmallVector<SmallVector<Value>>
  unpackInputs(Location loc, triton::ReduceOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const {
    auto types = op.getInputTypes();
    auto operands = adaptor.getOperands();
    unsigned srcElems = getTotalElemsPerThread(types[0]);
    SmallVector<SmallVector<Value>> srcValues(srcElems);
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      auto values = unpackLLElements(loc, operands[i], rewriter);

      assert(values.size() == srcValues.size());
      for (unsigned j = 0; j < srcValues.size(); ++j) {
        srcValues[j].push_back(values[j]);
      }
    }
    return srcValues;
  }

  void sync(ConversionPatternRewriter &rewriter, Location loc,
            triton::ReduceOp op) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    b.barrier(triton::gpu::AddrSpace::Local);
  }

  // Reduce along op axis for elements that are in the same thread. The
  // accumulated value is stored in accs.
  void reduceWithinThreads(
      ReduceOpHelper &helper, SmallVector<SmallVector<Value>> &srcValues,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &indices,
      ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    RankedTensorType operandType = op.getInputTypes()[0];
    // Assumes offsets don't actually depend on type
    SmallVector<SmallVector<unsigned>> offsets =
        emitOffsetForLayout(helper.getSrcLayout(), operandType);

    // Thread X might hold the same input value in two registers.  Get the
    // indices in `offsets` that hold unique values, and only accumulate over
    // those.
    llvm::MapVector<ArrayRef<unsigned>, int> uniqueOffsets;
    for (int i = 0; i < offsets.size(); ++i) {
      uniqueOffsets.insert({offsets[i], i});
    }

    unsigned srcElems = getTotalElemsPerThread(operandType);
    auto *combineOp = &op.getCombineOp();
    auto srcIndices = emitIndices(op.getLoc(), rewriter, targetInfo,
                                  helper.getSrcLayout(), operandType, true);
    // reduce within threads
    for (const auto &[_, i] : uniqueOffsets) {
      SmallVector<unsigned> key = offsets[i];
      key[op.getAxis()] = 0;
      bool isFirst = accs.find(key) == accs.end();
      accumulate(op.getLoc(), rewriter, *combineOp, accs[key], srcValues[i]);
      if (isFirst)
        indices[key] = srcIndices[i];
    }
  }

  // Apply warp reduction across the given number of contiguous lanes using op
  // region and the accumulator values as source.
  void warpReduce(ConversionPatternRewriter &rewriter, Location loc,
                  SmallVector<Value> &acc, triton::ReduceOp op,
                  unsigned numLaneToReduce, unsigned interleave,
                  Value pred = {}) const {
    auto success = targetInfo.warpReduce(rewriter, loc, acc, op,
                                         numLaneToReduce, interleave);
    if (success)
      return;

    for (unsigned N = numLaneToReduce / 2; N > 0; N >>= 1) {
      SmallVector<Value> shfl(acc.size());
      for (unsigned i = 0; i < acc.size(); ++i) {
        shfl[i] = targetInfo.shuffleXor(rewriter, loc, acc[i], N * interleave);
      }
      accumulate(op.getLoc(), rewriter, op.getCombineOp(), acc, shfl, pred);
    }
  }

  // Reduce across threads within each warp.
  void
  reduceWithinWarps(ReduceOpHelper &helper,
                    std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
                    ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    unsigned sizeIntraWarps = helper.getIntraWarpSizeWithUniqueData();
    unsigned threadOffsetOnReductionAxis =
        helper.getThreadOffsetOnReductionAxis();
    for (auto it : accs) {
      const SmallVector<unsigned> &key = it.first;
      SmallVector<Value> &acc = accs[key];
      warpReduce(rewriter, op.getLoc(), acc, op, sizeIntraWarps,
                 threadOffsetOnReductionAxis);
    }
  }

  // Pack the accumulator values and replace the reduce op with the result.
  void packResults(ReduceOpHelper &helper,
                   std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
                   ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    Location loc = op.getLoc();
    unsigned axis = op.getAxis();
    SmallVector<Value> results(op.getNumOperands());
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      if (auto resultTy =
              dyn_cast<RankedTensorType>(op.getResult()[i].getType())) {
        auto resultLayout = cast<SliceEncodingAttr>(resultTy.getEncoding());
        unsigned resultElems = getTotalElemsPerThread(resultTy);
        SmallVector<SmallVector<unsigned>> resultOffset =
            emitOffsetForLayout(resultLayout, resultTy);
        SmallVector<Value> resultVals;
        for (int j = 0; j < resultElems; j++) {
          auto key = resultOffset[j];
          key.insert(key.begin() + axis, 0);
          resultVals.push_back(accs[key][i]);
        }
        results[i] = packLLElements(loc, getTypeConverter(), resultVals,
                                    rewriter, resultTy);
      } else
        results[i] = accs.begin()->second[i];
    }
    rewriter.replaceOp(op, results);
  }

  void storeWarpReduceToSharedMemory(
      ReduceOpHelper &helper,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &indices,
      SmallVector<Value> &smemBases,
      ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto srcLayout =
        mlir::cast<DistributedEncodingTrait>(helper.getSrcLayout());
    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
    unsigned axis = op.getAxis();
    auto smemShape = helper.getScratchRepShape();

    // Lezcano: We should move all the shared memory logic to use LLs natively
    auto srcShape = helper.getSrcShape();
    auto kLane = rewriter.getStringAttr("lane");
    auto [multiDimLaneId, isRepresentativeLane] =
        delinearize(rewriter, loc, srcLayout, srcShape, kLane, laneId);
    auto kWarp = rewriter.getStringAttr("warp");
    auto [multiDimWarpId, isRepresentativeWarp] =
        delinearize(rewriter, loc, srcLayout, srcShape, kWarp, warpId);

    Value laneIdAxis = multiDimLaneId[axis];
    Value laneZero = b.icmp_eq(laneIdAxis, b.i32_val(0));
    Value write =
        b.and_(b.and_(isRepresentativeLane, isRepresentativeWarp), laneZero);

    Value warpIdAxis = multiDimWarpId[axis];

    auto smemOrder = helper.getOrderWithAxisAtBeginning();
    for (auto it : accs) {
      const SmallVector<unsigned> &key = it.first;
      SmallVector<Value> &acc = it.second;

      SmallVector<Value> writeIdx = indices[key];
      writeIdx[axis] = warpIdAxis;
      Value writeOffset =
          linearize(rewriter, loc, writeIdx, smemShape, smemOrder);
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        auto elemTy = getElementType(op, i);
        Value writePtr =
            b.gep(smemBases[i].getType(), elemTy, smemBases[i], writeOffset);
        targetInfo.storeShared(rewriter, loc, writePtr, acc[i], write);
      }
    }
  }

  // Load the reduction of each warp and accumulate them to a final value and
  // store back to shared memory.
  void accumulatePartialReductions(ReduceOpHelper &helper,
                                   SmallVector<Value> &smemBases,
                                   ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    auto smemShape = helper.getScratchRepShape();
    unsigned elems = product<unsigned>(smemShape);
    unsigned sizeInterWarps = helper.getInterWarpSizeWithUniqueData();
    assert(((sizeInterWarps - 1) & sizeInterWarps) == 0 &&
           "sizeInterWarps must be 2^m.");
    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    auto mod = op->getParentOfType<ModuleOp>();
    unsigned numLanes = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    int numWarps = triton::gpu::lookupNumWarps(op);
    int numThreads = numLanes * numWarps;

    Value threadId = getThreadId(rewriter, loc);
    Value warpSize = b.i32_val(numLanes);
    Value laneId = b.urem(threadId, warpSize);
    Value zero = b.i32_val(0);

    // It is a batched reduce with the initial problem shape [elems /
    // sizeInterWarps, sizeInterWarps]. The numLanes is 2^n. The
    // sizeInterWarps is 2^m. With the horizontal warp reduction, the problem
    // size is [elems / sizeInterWarps, N] -> [elems / sizeInterWarps, ceil(N,
    // numLanes)] in each reduce iteration.
    unsigned problemBatchSize = elems / sizeInterWarps;
    for (unsigned problemSize = sizeInterWarps; problemSize > 1;
         problemSize = problemSize / numLanes) {
      unsigned reduceLaneNumber = std::min(problemSize, numLanes);
      unsigned totalProblemSizePerIter = problemSize * problemBatchSize;
      unsigned elemsPerThread =
          mlir::ceil<unsigned>(totalProblemSizePerIter, numThreads);

      // The problem stride in each iteration is [sizeInterWarps / problemSize]
      Value readOffset =
          b.mul(threadId, b.i32_val(sizeInterWarps / problemSize));

      for (unsigned round = 0; round < elemsPerThread; ++round) {
        Value threadIsNeeded = b.icmp_slt(readOffset, b.i32_val(elems));

        SmallVector<Value> acc(op.getNumOperands());
        for (unsigned i = 0; i < op.getNumOperands(); ++i) {
          auto elemTy = getElementType(op, i);
          Value readPtr =
              b.gep(smemBases[i].getType(), elemTy, smemBases[i], readOffset);
          acc[i] = targetInfo.loadShared(rewriter, loc, readPtr, elemTy,
                                         threadIsNeeded);
        }
        warpReduce(rewriter, loc, acc, op, reduceLaneNumber, 1 /* interleave */,
                   threadIsNeeded);
        // only the first thread in each sizeInterWarps is writing
        Value writeOffset = readOffset;
        SmallVector<Value> writePtrs(op.getNumOperands());
        for (unsigned i = 0; i < op.getNumOperands(); ++i) {
          auto elemTy = getElementType(op, i);
          writePtrs[i] =
              b.gep(smemBases[i].getType(), elemTy, smemBases[i], writeOffset);
        }

        // only the first thread in each reduceLaneNumber is writing
        Value threadIdModSizeReduceLanes =
            b.urem(threadId, b.i32_val(reduceLaneNumber));
        Value threadIdModSizeReduceLanesIsZero =
            b.icmp_eq(threadIdModSizeReduceLanes, zero);
        Value pred = b.and_(threadIsNeeded, threadIdModSizeReduceLanesIsZero);

        for (unsigned i = 0; i < op.getNumOperands(); ++i) {
          targetInfo.storeShared(rewriter, loc, writePtrs[i], acc[i], pred);
        }

        if (round != elemsPerThread - 1) {
          readOffset =
              b.add(readOffset,
                    b.i32_val(numThreads * (sizeInterWarps / problemSize)));
        }
      }

      if (problemSize > numLanes) {
        // More reduce iteration required. Synchronize here.
        sync(rewriter, loc, op);
      }
    }
  }

  // Load the final reduction from shared memory and replace the reduce result
  // with it.
  void loadReductionAndPackResult(ReduceOpHelper &helper,
                                  SmallVector<unsigned> smemShape,
                                  SmallVector<Value> &smemBases,
                                  ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto srcLayout = helper.getSrcLayout();
    auto axis = op.getAxis();
    auto smemOrder = helper.getOrderWithAxisAtBeginning();
    SmallVector<Value> results(op.getNumOperands());
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      auto elemTy = getElementType(op, i);
      if (auto resultTy =
              dyn_cast<RankedTensorType>(op.getResult()[i].getType())) {
        // nd-tensor where n >= 1
        auto resultLayout = cast<SliceEncodingAttr>(resultTy.getEncoding());
        unsigned resultElems = getTotalElemsPerThread(resultTy);
        auto resultIndices = emitIndices(loc, rewriter, targetInfo,
                                         resultLayout, resultTy, true);
        auto resultShape = resultTy.getShape();
        assert(resultIndices.size() == resultElems);

        SmallVector<Value> resultVals(resultElems);
        for (size_t j = 0; j < resultElems; ++j) {
          SmallVector<Value> readIdx = resultIndices[j];
          readIdx.insert(readIdx.begin() + op.getAxis(), b.i32_val(0));
          for (size_t resultIdx = 0, resultDim = resultShape.size();
               resultIdx < resultDim; ++resultIdx) {
            auto smemIdx = resultIdx < op.getAxis() ? resultIdx : resultIdx + 1;
            if (resultShape[resultIdx] > smemShape[smemIdx]) {
              // When srcShape smaller then src sizePerThread, only srcShape
              // elements is accumulated in smem. Modulo smemShape effectively
              // replicates srcShape elements to src sizePerThread.
              readIdx[smemIdx] =
                  b.urem(readIdx[smemIdx], b.i32_val(smemShape[smemIdx]));
            }
          }
          Value readOffset =
              linearize(rewriter, loc, readIdx, smemShape, smemOrder);
          Value readPtr =
              b.gep(smemBases[i].getType(), elemTy, smemBases[i], readOffset);
          resultVals[j] = b.load(elemTy, readPtr);
        }

        results[i] = packLLElements(loc, getTypeConverter(), resultVals,
                                    rewriter, resultTy);
      } else {
        // 0d-tensor -> scalar
        results[i] = b.load(elemTy, smemBases[i]);
      }
    }
    rewriter.replaceOp(op, results);
  }

  // === Step 3 implementations ===

  void reduceAcrossWarpsIntel(
      ReduceOpHelper &helper,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &indices,
      ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    Location loc = op.getLoc();

    // Compute a shared memory base per operand.
    auto smemShape = helper.getScratchRepShape();
    SmallVector<Value> smemBases =
        getSmemBases(op, product<unsigned>(smemShape), rewriter, targetInfo);

    storeWarpReduceToSharedMemory(helper, accs, indices, smemBases, rewriter);
    sync(rewriter, loc, op);

    accumulatePartialReductions(helper, smemBases, rewriter);

    // We could avoid this barrier in some of the layouts; not the general case.
    sync(rewriter, loc, op);

    // Set output values.
    loadReductionAndPackResult(helper, smemShape, smemBases, rewriter);
  }

  LogicalResult reduceAcrossWarpsLikeCommon(
      ReduceOpHelper &helper,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
      ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    // The common lowering logic works on LinearLayout + SSA vectors of values.
    // We reconstruct (layout, values) from the post-step-2 Intel state:
    //  - layout is the reducedRegLaneLayout for the src type.
    //  - values are taken from accs in the canonical per-thread register order.
    //
    // This keeps the numeric behavior of step 1/2 unchanged, and only changes
    // the cross-warp orchestration.
    unsigned axis = op.getAxis();

    LinearLayout regLl = triton::gpu::toLinearLayout(helper.getSrcTy());
    // After Intel step 1/2, we should be at the same point as common lowering
    // after its first two phases.
    regLl = ReduceOpHelper::reducedRegLaneLayout(helper.getSrcTy(), axis);

    // Materialize per-operand register values from the (key -> acc) map.
    // Keys are indices in the reduced layout with the reduction axis set to 0.
    SmallVector<SmallVector<Value>> llAccs(op.getNumOperands());

    // We need the per-thread register offsets for the reduced layout.
    // element type doesn’t matter for offsets.
    auto srcTy = cast<RankedTensorType>(op.getOperandTypes().front());
    auto srcEncoding = cast<DistributedEncodingTrait>(srcTy.getEncoding());
    auto regTensorTy = RankedTensorType::get(
        srcTy.getShape(), srcTy.getElementType(), srcTy.getEncoding());
    (void)regTensorTy;

    // Use emitIndices/emitOffsetForLayout on the *src* layout to define a
    // stable ordering of registers. Then map each register key to the reduced
    // key (axis=0). This matches how reduceWithinThreads builds accs.
    SmallVector<SmallVector<unsigned>> offsets =
        emitOffsetForLayout(helper.getSrcLayout(), srcTy);

    // offsets.size() can have duplicates due to broadcasting.
    llvm::MapVector<ArrayRef<unsigned>, int> uniqueOffsets;
    for (int i = 0; i < static_cast<int>(offsets.size()); ++i)
      uniqueOffsets.insert({offsets[i], i});

    // Reorder unique offsets by their original index to keep deterministic.
    SmallVector<int> uniqueIdx;
    uniqueIdx.reserve(uniqueOffsets.size());
    for (const auto &it : uniqueOffsets)
      uniqueIdx.push_back(it.second);
    llvm::sort(uniqueIdx);

    // For each unique register, compute the corresponding reduced key.
    for (int i : uniqueIdx) {
      SmallVector<unsigned> key = offsets[i];
      key[axis] = 0;
      auto it = accs.find(key);
      if (it == accs.end()) {
        // Should not happen; indicates mismatch between offset computation and
        // accumulator keys.
        return rewriter.notifyMatchFailure(op,
                                           "failed to find accumulator key");
      }
      for (unsigned opIdx = 0; opIdx < op.getNumOperands(); ++opIdx)
        llAccs[opIdx].push_back(it->second[opIdx]);
    }

    // Now mirror the common lowering’s cross-warp loop.
    auto kAxis = *(regLl.getOutDimNames().begin() + axis);
    auto kBlock = StringAttr::get(ctx, "block");
    bool lastCvtCrossesCTAs = false;
    int round = 0;

    while (regLl.getOutDimSize(kAxis) != 1) {
      LinearLayout tmpLl = ReduceOpHelper::getInterLayout(regLl, axis);

      if (round > 0)
        sync(rewriter, loc, op);

      llAccs = convertLayoutValues(loc, rewriter, op, regLl, tmpLl, llAccs);
      lastCvtCrossesCTAs = !mlir::isCvtDimSync(regLl, tmpLl, kBlock);

      (void)lastCvtCrossesCTAs;
      std::tie(regLl, llAccs) = reduceWithinWarpsLikeCommon(
          op, std::move(tmpLl), std::move(llAccs), rewriter);
      ++round;
    }

    // Common expects <= 2 rounds.
    if (round > 2) {
      return rewriter.notifyMatchFailure(
          op, "expected at most 2 cross-warp rounds");
    }

    // Remove axis dim (size 1) and pack results.
    regLl = removeStandardDim(regLl, axis);

    // Convert to output layout if needed.
    if (auto resultTy =
            dyn_cast<RankedTensorType>(op.getResult()[0].getType())) {
      auto outLl = triton::gpu::toLinearLayout(resultTy);
      if (regLl != outLl) {
        sync(rewriter, loc, op);
        llAccs = convertLayoutValues(loc, rewriter, op, regLl, outLl, llAccs);
      }
    }

    // Replace op with packed results.
    packResultsLikeCommon(op, llAccs, rewriter);
    return success();
  }

  // === Helpers borrowed from common reduce lowering (kept local to Intel file)

  std::pair<LinearLayout, SmallVector<SmallVector<Value>>>
  reduceWithinWarpsLikeCommon(triton::ReduceOp op, LinearLayout layout,
                              SmallVector<SmallVector<Value>> accs,
                              ConversionPatternRewriter &rewriter) const {
    auto *ctx = op.getContext();
    auto kLane = str_attr("lane");
    const auto &laneBases = layout.getBases().lookup(kLane);
    unsigned reduceLaneIdMask = 0;
    for (unsigned bit = 0; bit < laneBases.size(); ++bit) {
      if (laneBases[bit][op.getAxis()] != 0)
        reduceLaneIdMask |= 1u << bit;
    }
    if (reduceLaneIdMask == 0)
      return {std::move(layout), std::move(accs)};

    unsigned regs = accs.front().size();
    for (unsigned reg = 0; reg < regs; ++reg) {
      SmallVector<Value> acc(op.getNumOperands());
      for (unsigned i = 0; i < op.getNumOperands(); ++i)
        acc[i] = accs[i][reg];

      warpReduceLikeCommon(op, reduceLaneIdMask, acc, rewriter);

      for (unsigned i = 0; i < op.getNumOperands(); ++i)
        accs[i][reg] = acc[i];
    }

    layout = ReduceOpHelper::zeroBasesAlongDimAndReorder(layout, op.getAxis(),
                                                         kLane);
    return {std::move(layout), std::move(accs)};
  }

  void warpReduceLikeCommon(triton::ReduceOp op, unsigned reduceLaneIdMask,
                            SmallVector<Value> &acc,
                            ConversionPatternRewriter &rewriter) const {
    if (reduceLaneIdMask == 0)
      return;

    auto moduleOp = op->getParentOfType<ModuleOp>();
    unsigned warpSize =
        triton::gpu::TritonGPUDialect::getThreadsPerWarp(moduleOp);
    assert(reduceLaneIdMask < warpSize &&
           "expected reduce lane ID mask < warp size");

    if (targetInfo.warpReduce(rewriter, op.getLoc(), acc, op,
                              reduceLaneIdMask)) {
      return;
    }

    for (int bit = llvm::Log2_32(warpSize) - 1; bit >= 0; --bit) {
      unsigned mask = 1u << bit;
      if ((reduceLaneIdMask & mask) == 0)
        continue;
      SmallVector<Value> shfl(op.getNumOperands());
      for (unsigned i = 0; i < op.getNumOperands(); ++i)
        shfl[i] = targetInfo.shuffleXor(rewriter, op.getLoc(), acc[i], mask);

      accumulate(op.getLoc(), rewriter, op.getCombineOp(), acc, shfl);
    }
  }

  void packResultsLikeCommon(triton::ReduceOp op,
                             SmallVector<SmallVector<Value>> &accs,
                             ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    SmallVector<Value> results(op.getNumOperands());
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      if (auto resultTy =
              dyn_cast<RankedTensorType>(op.getResult()[i].getType())) {
        results[i] = packLLElements(loc, getTypeConverter(), accs[i], rewriter,
                                    resultTy);
      } else {
        results[i] = accs[i].front();
      }
    }
    rewriter.replaceOp(op, results);
  }

  SmallVector<SmallVector<Value>>
  convertLayoutValues(Location loc, ConversionPatternRewriter &rewriter,
                      triton::ReduceOp op, const LinearLayout &srcLayout,
                      const LinearLayout &dstLayout,
                      const SmallVector<SmallVector<Value>> &inVals) const {
    SmallVector<SmallVector<Value>> outVals(op.getNumOperands());
    auto *ctx = rewriter.getContext();
    SmallVector<int64_t> shape;
    for (auto dim : srcLayout.getOutDimNames()) {
      shape.push_back(srcLayout.getOutDimSize(dim));
    }
    auto srcEnc = triton::gpu::LinearEncodingAttr::get(ctx, srcLayout);
    auto dstEnc = triton::gpu::LinearEncodingAttr::get(ctx, dstLayout);
    auto baseOffsetAttr = op->getAttrOfType<IntegerAttr>("allocation.offset");
    assert(baseOffsetAttr && "expected allocation.offset on reduce op");
    int64_t baseOffset = baseOffsetAttr.getValue().getZExtValue();
    auto smemBaseOffsets = getSmemBaseOffsets(op, srcLayout, dstLayout);
    auto offsetTy = IntegerType::get(ctx, 32);
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      auto elemTy = op.getElementTypes()[i];
      auto srcTy = RankedTensorType::get(shape, elemTy, srcEnc);
      auto dstTy = RankedTensorType::get(shape, elemTy, dstEnc);
      Value packed =
          packLLElements(loc, getTypeConverter(), inVals[i], rewriter, srcTy);
      auto srcTensor =
          UnrealizedConversionCastOp::create(rewriter, loc, srcTy, packed)
              .getResult(0);
      auto cvt =
          triton::gpu::ConvertLayoutOp::create(rewriter, loc, dstTy, srcTensor);
      cvt->setAttr("allocation.offset",
                   IntegerAttr::get(offsetTy, baseOffset + smemBaseOffsets[i]));
      Type packedDstTy = getTypeConverter()->convertType(dstTy);
      auto packedDst = UnrealizedConversionCastOp::create(
                           rewriter, loc, packedDstTy, cvt.getResult())
                           .getResult(0);
      outVals[i] = unpackLLElements(loc, packedDst, rewriter);
    }
    return outVals;
  }

  Type getReduceMemElemTy(Type elemTy, MLIRContext *ctx) const {
    if (elemTy.isIntOrFloat() && elemTy.getIntOrFloatBitWidth() < 8)
      return IntegerType::get(ctx, 8);
    return elemTy;
  }

  SmallVector<int64_t> getSmemBaseOffsets(triton::ReduceOp op,
                                          const LinearLayout &srcLayout,
                                          const LinearLayout &dstLayout) const {
    // Hack:
    // Here we know that we are never going to use ldmatrix/stmatrix
    // instructions as by the time we go through shared memory, we have already
    // reduced all the registers As such, we can use
    // `getNumScratchElemsSwizzledCvt` which assumes ld.shared/st.shared
    // instructions
    // The proper way to lower reduce would be to lower it to:
    // reduce_threads / reduce_lanes / convert_layout
    // And let the AllocationAnalysis handle the shared memory allocation
    // and membar the barriers
    std::vector<unsigned> indices(op.getNumOperands());
    std::iota(indices.begin(), indices.end(), 0);
    auto *ctx = op.getContext();
    std::sort(indices.begin(), indices.end(), [&](unsigned i, unsigned j) {
      auto lhsTy = getReduceMemElemTy(op.getElementTypes()[i], ctx);
      auto rhsTy = getReduceMemElemTy(op.getElementTypes()[j], ctx);
      return getIntOrFloatOrPtrBitWidth(lhsTy) >
             getIntOrFloatOrPtrBitWidth(rhsTy);
    });
    SmallVector<int64_t> offsets(op.getNumOperands());
    int64_t offset = 0;
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      unsigned idx = indices[i];
      offsets[idx] = offset;
      auto inputTy = op.getInputTypes()[idx];
      auto bytes = getNumScratchElemsSwizzledCvt(srcLayout, dstLayout,
                                                 getBitwidth(inputTy)) *
                   (getBitwidth(inputTy) / 8);
      offset += bytes;
    }
    return offsets;
  }
};
} // namespace

void mlir::triton::intel::populateReduceOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<ReduceOpConversion>(typeConverter, targetInfo, benefit);
}
