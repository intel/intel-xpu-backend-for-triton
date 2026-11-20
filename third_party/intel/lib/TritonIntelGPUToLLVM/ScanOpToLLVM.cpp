// Intel lowering of `tt.scan` for blocked layouts.
//
// This is core's lowering from before triton-lang/triton#12184, which main
// used, with its in-warp phase on the hardware sub-group scan when the scan
// axis fills the sub-group. Core's newer lowering, which takes every layout,
// moves the segment totals between lanes and registers with layout
// conversions that are slow on Intel GPUs. Keep the operations of the old
// lowering and their order: IGC schedules even reordered independent
// operations differently, which costs prefix-sum kernels several percent.
// A reverse scan runs the same steps from the end of the axis, instead of
// reversing every register before and after a forward scan as the old
// lowering did. Core's lowering takes the other layouts.

#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"

#include "intel/include/Analysis/Utility.h"
#include "intel/include/TritonIntelGPUToLLVM/TargetInfo.h"
#include "lib/Conversion/TritonGPUToLLVM/ReduceScanCommon.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::triton;
using ::mlir::LLVM::delinearize;
using ::mlir::LLVM::linearize;
namespace ttgi = mlir::triton::gpu::intel;

namespace {
using Values = SmallVector<Value>;

class ScanLowering {
public:
  ScanLowering(triton::ScanOp op, const ttgi::BlockedScanLayout &s,
               ConversionPatternRewriter &rewriter,
               const intel::TargetInfo &targetInfo,
               const LLVMTypeConverter &typeConverter)
      : op(op), s(s), rewriter(rewriter), targetInfo(targetInfo),
        typeConverter(typeConverter), loc(op.getLoc()), b(loc, rewriter),
        warpSize(triton::gpu::TritonGPUDialect::getThreadsPerWarp(
            op->getParentOfType<ModuleOp>())),
        reverse(op.getReverse()) {
    unsigned numRegs = s.blocks * s.parallelBlocks * s.parallelElemsPerThread *
                       s.elemsPerThread;
    for (unsigned srcIndex = 0; srcIndex < numRegs; ++srcIndex) {
      if ((srcIndex / s.elemStride) % s.elemsPerThread != s.elemsPerThread - 1)
        continue;
      unsigned blockId = chunks.size() / s.parallelElemsPerThread;
      unsigned parallelBlockId =
          blockId % s.blockStride +
          ((blockId / s.blockStride) / s.blocks) * s.blockStride;
      chunks.push_back({srcIndex,
                        unsigned(chunks.size()) % s.parallelElemsPerThread +
                            parallelBlockId * s.parallelElemsPerThread,
                        (blockId / s.blockStride) % s.blocks});
    }
  }

  void run(triton::ScanOpAdaptor adaptor) {
    Value threadId = getThreadId(rewriter, loc);
    Value warpSizeValue = b.i32_val(warpSize);
    Value warpId = b.udiv(threadId, warpSizeValue);
    Value laneId = b.urem(threadId, warpSizeValue);
    auto [laneIdAxis, warpIdAxis, flatIdParallel, isRepresentative] =
        getDelinearizedIds(laneId, warpId);
    srcValues.resize(s.blocks * s.parallelBlocks * s.parallelElemsPerThread *
                     s.elemsPerThread);
    for (Value operand : adaptor.getOperands()) {
      Values elements = unpackUniqueTensorElements(loc, operand, rewriter);
      assert(elements.size() == srcValues.size());
      for (auto [i, element] : llvm::enumerate(elements))
        srcValues[i].push_back(element);
    }
    // A reverse scan walks the registers, lanes and warps from the end of the
    // axis; reversing the registers only renames values.
    if (reverse) {
      warpIdAxis = b.sub(b.i32_val(s.warps - 1), warpIdAxis);
      laneIdAxis = b.sub(b.i32_val(s.lanes - 1), laneIdAxis);
      std::reverse(srcValues.begin(), srcValues.end());
    }
    scanThreadContiguousElements();
    warpScan(laneIdAxis);
    if (s.warps > 1) {
      SmallVector<Value> smemBases =
          LLVM::intel::getSmemBases(op, op.getElementTypes(), s.scratchElems,
                                    typeConverter, rewriter, targetInfo);
      SmallVector<Type> smemTypes;
      for (Type ty : op.getElementTypes())
        smemTypes.push_back(typeConverter.convertType(ty));
      storeWarpAccumulator(smemBases, smemTypes, laneIdAxis, warpIdAxis,
                           flatIdParallel, isRepresentative);
      b.barrier(triton::gpu::AddrSpace::Local);
      addPartialReduce(smemBases, smemTypes, warpIdAxis, laneIdAxis,
                       flatIdParallel);
    } else if (srcValues.size() > 1) {
      auto multiDimLaneId = std::get<0>(delinearizeId("lane", laneId));
      multiDimLaneId[s.axis] = b.i32_val(reverse ? 0 : s.lanes - 1);
      Value laneIdLast = linearize(rewriter, loc, multiDimLaneId, s.encoding,
                                   rewriter.getStringAttr("lane"));
      addPartialReduceOneWarp(warpIdAxis, laneIdAxis, laneIdLast);
    }
    if (reverse)
      std::reverse(srcValues.begin(), srcValues.end());

    SmallVector<Value> results;
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      Values elements;
      for (const Values &values : srcValues)
        elements.push_back(values[i]);
      results.push_back(packUniqueTensorElements(loc, &typeConverter, elements,
                                                 rewriter,
                                                 op.getResult()[i].getType()));
    }
    rewriter.replaceOp(op, results);
  }

private:
  triton::ScanOp op;
  const ttgi::BlockedScanLayout &s;
  ConversionPatternRewriter &rewriter;
  const intel::TargetInfo &targetInfo;
  const LLVMTypeConverter &typeConverter;
  Location loc;
  TritonLLVMOpBuilder b;
  int warpSize;
  bool reverse;
  // srcValues[register][operand].
  SmallVector<Values> srcValues;
  // The chunks of contiguous axis elements, in register order: the register
  // of the last element, the accumulator and the block along the axis.
  struct Chunk {
    unsigned srcIndex, accumulatorIndex, axisBlockId;
  };
  SmallVector<Chunk> chunks;

  Values accumulate(ValueRange acc, ValueRange cur, Value pred = {}) {
    return applyCombineOp(loc, rewriter, op.getCombineOp(), acc, cur, pred);
  }

  // The value of the lane `delta` lane ids earlier in scan order.
  Value shuffleFromEarlierLane(Value value, int delta) {
    return reverse ? LLVM::intel::shuffleDown(loc, rewriter, value, delta)
                   : LLVM::intel::shuffleUp(loc, rewriter, value, delta);
  }

  std::tuple<Values, Value> delinearizeId(StringRef dim, Value id) {
    return delinearize(rewriter, loc, s.encoding, s.shape,
                       rewriter.getStringAttr(dim), id);
  }

  // The lane and warp positions along the axis, the index of the thread among
  // the threads of other scans, and whether it holds no broadcast copy.
  std::tuple<Value, Value, Value, Value> getDelinearizedIds(Value laneId,
                                                            Value warpId) {
    SmallVector<unsigned> threadsPerWarp = s.encoding.getThreadsPerWarp();
    SmallVector<unsigned> warpsPerCTA = s.encoding.getWarpsPerCTA();
    auto [multiDimLaneId, isRepresentativeLane] = delinearizeId("lane", laneId);
    auto [multiDimWarpId, isRepresentativeWarp] = delinearizeId("warp", warpId);
    Value laneIdAxis = multiDimLaneId[s.axis];
    Value warpIdAxis = multiDimWarpId[s.axis];
    multiDimLaneId[s.axis] = b.i32_val(0);
    threadsPerWarp[s.axis] = 1;
    Value laneIdParallel =
        linearize(rewriter, loc, multiDimLaneId, threadsPerWarp, s.order);
    multiDimWarpId[s.axis] = b.i32_val(0);
    warpsPerCTA[s.axis] = 1;
    Value warpIdParallel =
        linearize(rewriter, loc, multiDimWarpId, warpsPerCTA, s.order);
    Value flatIdParallel =
        b.add(laneIdParallel,
              b.mul(warpIdParallel, b.i32_val(s.parallelLanesPerWarp)));
    return {laneIdAxis, warpIdAxis, flatIdParallel,
            b.and_(isRepresentativeLane, isRepresentativeWarp)};
  }

  // 1. Each thread scans its contiguous chunks of axis elements.
  void scanThreadContiguousElements() {
    SmallVector<Values> accs(srcValues.size() / s.elemsPerThread);
    for (unsigned srcIndex = 0; srcIndex < srcValues.size(); ++srcIndex) {
      unsigned accIndex = srcIndex % s.elemStride + srcIndex / s.elemStride /
                                                        s.elemsPerThread *
                                                        s.elemStride;
      accs[accIndex] = accumulate(accs[accIndex], srcValues[srcIndex]);
      srcValues[srcIndex] = accs[accIndex];
    }
  }

  // 2. The axis lanes scan the chunk totals. The builtin scans in lane order,
  // so a reverse scan, which walks the lanes from the end, shuffles instead:
  // mirroring the lanes around the builtin costs more than it saves.
  void warpScan(Value laneIdAxis) {
    for (const Chunk &chunk : chunks) {
      unsigned srcIndex = chunk.srcIndex;
      Values acc = srcValues[srcIndex];
      if (!reverse &&
          targetInfo.warpScan(rewriter, loc, acc, op, s.lanes, s.laneStride)) {
        srcValues[srcIndex] = std::move(acc);
        continue;
      }
      for (unsigned i = 1; i <= s.lanes / 2; i <<= 1) {
        Values shfl;
        for (Value value : acc)
          shfl.push_back(shuffleFromEarlierLane(value, i * s.laneStride));
        Value mask = b.icmp_sge(laneIdAxis, b.i32_val(i));
        Values tempAcc = accumulate(shfl, acc, mask);
        for (unsigned j = 0; j < acc.size(); ++j)
          acc[j] = b.select(mask, tempAcc[j], acc[j]);
      }
      srcValues[srcIndex] = std::move(acc);
    }
  }

  // 3. When the axis spans warps, the lanes holding the warp totals store
  // them, per chunk, warp and parallel thread, to shared memory ...
  void storeWarpAccumulator(ArrayRef<Value> smemBases, ArrayRef<Type> smemTypes,
                            Value laneIdAxis, Value warpIdAxis,
                            Value flatIdParallel, Value isRepresentative) {
    for (auto [chunkId, chunk] : llvm::enumerate(chunks)) {
      Value mask = b.icmp_eq(laneIdAxis, b.i32_val(s.lanes - 1));
      mask = b.and_(mask, isRepresentative);
      Value index = b.add(flatIdParallel,
                          b.mul(warpIdAxis, b.i32_val(s.parallelThreads)));
      index = b.add(index, b.i32_val(chunkId * s.parallelThreads * s.warps));
      for (auto [i, value] : llvm::enumerate(srcValues[chunk.srcIndex])) {
        Value ptr =
            b.gep(smemBases[i].getType(), smemTypes[i], smemBases[i], index);
        targetInfo.storeShared(rewriter, loc, ptr, value, mask);
      }
    }
  }

  // ... and, after a barrier, each thread combines the totals of the warps
  // and blocks before its own into its chunks.
  void addPartialReduce(ArrayRef<Value> smemBases, ArrayRef<Type> smemTypes,
                        Value warpIdAxis, Value laneIdAxis,
                        Value flatIdParallel) {
    Value maskNotFirstWarp = b.icmp_ne(warpIdAxis, b.i32_val(0));
    Value maskNotFirstLane = b.icmp_ne(laneIdAxis, b.i32_val(0));
    Value maskNotFirstThread = b.or_(maskNotFirstWarp, maskNotFirstLane);
    struct Accumulator {
      Values acc, maskedAcc;
    };
    SmallVector<Accumulator> accumulators(s.parallelBlocks *
                                          s.parallelElemsPerThread);
    for (auto [chunkId, chunk] : llvm::enumerate(chunks)) {
      auto [srcIndex, accumulatorIndex, axisBlockId] = chunk;
      Accumulator &accumulator = accumulators[accumulatorIndex];
      for (unsigned i = 0; i < s.warps; ++i) {
        Value index = b.add(flatIdParallel, b.i32_val(s.parallelThreads *
                                                      (i + chunkId * s.warps)));
        Values partialReduce;
        for (auto [base, elemTy] : llvm::zip(smemBases, smemTypes)) {
          Value ptr = b.gep(base.getType(), elemTy, base, index);
          partialReduce.push_back(
              targetInfo.loadShared(rewriter, loc, ptr, elemTy, b.true_val()));
        }
        if (accumulator.acc.empty()) {
          accumulator.acc = partialReduce;
          accumulator.maskedAcc = partialReduce;
          continue;
        }
        Value mask = b.icmp_sge(warpIdAxis, b.i32_val(i + 1));
        accumulator.acc = accumulate(accumulator.acc, partialReduce);
        for (auto [maskedAcc, acc] :
             llvm::zip(accumulator.maskedAcc, accumulator.acc))
          maskedAcc = b.select(mask, acc, maskedAcc);
      }

      Value pred = axisBlockId == 0 ? maskNotFirstWarp : Value{};
      Values temp =
          accumulate(accumulator.maskedAcc, srcValues[srcIndex], pred);
      if (axisBlockId == 0)
        for (auto [value, original] : llvm::zip(temp, srcValues[srcIndex]))
          value = b.select(maskNotFirstWarp, value, original);
      srcValues[srcIndex] = temp;
      Values lastElement;
      for (auto [value, maskedAcc] : llvm::zip(temp, accumulator.maskedAcc))
        lastElement.push_back(
            b.select(maskNotFirstLane,
                     shuffleFromEarlierLane(value, s.laneStride), maskedAcc));
      for (unsigned i = 1; i < s.elemsPerThread; ++i) {
        Values &element = srcValues[srcIndex - i * s.elemStride];
        pred = axisBlockId == 0 ? maskNotFirstThread : Value{};
        Values laneValue = accumulate(lastElement, element, pred);
        if (axisBlockId == 0)
          for (auto [value, original] : llvm::zip(laneValue, element))
            value = b.select(maskNotFirstThread, value, original);
        element = std::move(laneValue);
      }
      accumulator.maskedAcc = accumulator.acc;
    }
  }

  // 3. When the axis stays within a warp, each block takes the total of the
  // blocks before it, broadcast from the last lane along the axis.
  void addPartialReduceOneWarp(Value warpIdAxis, Value laneIdAxis,
                               Value laneIdLast) {
    Value maskFirstWarp = b.icmp_eq(warpIdAxis, b.i32_val(0));
    Value maskFirstLane = b.icmp_eq(laneIdAxis, b.i32_val(0));
    Value maskFirstThread = b.and_(maskFirstWarp, maskFirstLane);
    SmallVector<Values> accumulators(s.parallelBlocks *
                                     s.parallelElemsPerThread);
    for (auto [srcIndex, accumulatorIndex, axisBlockId] : chunks) {
      Values &accumulator = accumulators[accumulatorIndex];
      if (axisBlockId == 0)
        accumulator = srcValues[srcIndex];
      else
        srcValues[srcIndex] = accumulate(accumulator, srcValues[srcIndex]);
      Values lastElement = srcValues[srcIndex];
      if (s.lanes > 1) {
        for (unsigned i = 0; i < op.getNumOperands(); ++i) {
          lastElement[i] =
              shuffleFromEarlierLane(srcValues[srcIndex][i], s.laneStride);
          lastElement[i] =
              b.select(maskFirstLane, accumulator[i], lastElement[i]);
          if (s.blocks > 1)
            accumulator[i] = targetInfo.shuffleIdx(
                rewriter, loc, srcValues[srcIndex][i], laneIdLast);
        }
      } else if (s.blocks > 1) {
        // The rest of the chunk needs the total of the blocks before it.
        lastElement = accumulator;
        accumulator = srcValues[srcIndex];
      }
      for (unsigned i = 1; i < s.elemsPerThread; ++i) {
        Values &element = srcValues[srcIndex - i * s.elemStride];
        Values laneValue = accumulate(lastElement, element);
        if (axisBlockId == 0)
          for (auto [value, original] : llvm::zip(laneValue, element))
            value = b.select(maskFirstThread, original, value);
        element = std::move(laneValue);
      }
    }
  }
};

struct ScanOpConversion : public ConvertOpToLLVMPattern<triton::ScanOp> {
  ScanOpConversion(LLVMTypeConverter &typeConverter,
                   const intel::TargetInfo &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::ScanOp>(typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::ScanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Core's lowering, registered with a lower benefit, takes other layouts.
    std::optional<ttgi::BlockedScanLayout> s = ttgi::BlockedScanLayout::get(op);
    if (!s)
      return failure();
    ScanLowering(op, *s, rewriter, targetInfo, *getTypeConverter())
        .run(adaptor);
    return success();
  }

  const intel::TargetInfo &targetInfo;
};

} // namespace

void mlir::triton::intel::populateScanOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfo &targetInfo, PatternBenefit benefit) {
  patterns.add<ScanOpConversion>(typeConverter, targetInfo, benefit);
}
