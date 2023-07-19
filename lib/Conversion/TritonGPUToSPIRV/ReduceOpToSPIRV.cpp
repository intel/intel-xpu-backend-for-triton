#include "ReduceOpToSPIRV.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::spirv::shflSync;
using ::mlir::spirv::storeShared;
using ::mlir::triton::gpu::getElemsPerThread;
using ::mlir::triton::gpu::getOrder;
using ::mlir::triton::gpu::getTotalElemsPerThread;

struct ReduceOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<triton::ReduceOp> {
public:
  using ConvertTritonGPUOpToSPIRVPattern<
      triton::ReduceOp>::ConvertTritonGPUOpToSPIRVPattern;

  LogicalResult
  matchAndRewrite(triton::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (ReduceOpHelper(op).isFastReduction())
      return matchAndRewriteFast(op, adaptor, rewriter);
    return matchAndRewriteBasic(op, adaptor, rewriter);
  }

private:
  void accumulate(ConversionPatternRewriter &rewriter, Region &combineOp,
                  llvm::SmallVectorImpl<Value> &acc, ValueRange cur,
                  bool isFirst) const {
    if (isFirst) {
      acc.resize(cur.size());
      for (unsigned i = 0; i < cur.size(); ++i) {
        acc[i] = cur[i];
      }
      return;
    }

    // Create a new copy of the reduce block, and inline it
    Block *currentBlock = rewriter.getBlock();
    Region &parent = *currentBlock->getParent();
    rewriter.cloneRegionBefore(combineOp, &parent.front());
    auto &newReduce = parent.front();
    auto returnOp = dyn_cast<triton::ReduceReturnOp>(newReduce.getTerminator());

    llvm::SmallVector<Value> combineArgs(2 * acc.size());
    for (unsigned i = 0; i < acc.size(); ++i) {
      combineArgs[i] = acc[i];
      combineArgs[acc.size() + i] = cur[i];
    }

    rewriter.inlineBlockBefore(&newReduce, &*rewriter.getInsertionPoint(),
                               combineArgs);

    auto results = returnOp.getResult();
    for (unsigned i = 0; i < acc.size(); ++i) {
      acc[i] = results[i];
    }

    // Delete the terminator, which is no longer used
    rewriter.eraseOp(returnOp);
  }

  SmallVector<SmallVector<Value>>
  unpackInputs(Location loc, triton::ReduceOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const {
    auto types = op.getInputTypes();
    auto operands = adaptor.getOperands();
    unsigned srcElems = getTotalElemsPerThread(types[0]);
    SmallVector<SmallVector<Value>> srcValues(srcElems);
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      auto values = getTypeConverter()->unpackLLElements(loc, operands[i],
                                                         rewriter, types[i]);

      assert(values.size() == srcValues.size());
      for (unsigned j = 0; j < srcValues.size(); ++j) {
        srcValues[j].push_back(values[j]);
      }
    }
    return srcValues;
  }

  // Calculates the write index in the shared memory where we would be writing
  // the within-thread accumulations before we start doing across-threads
  // accumulations. `index` is the index of the within-thread accumulations in
  // the full tensor, whereas `writeIdx` is the mapped-to index in the shared
  // memory
  void getWriteIndexBasic(ConversionPatternRewriter &rewriter, Location loc,
                          Attribute layout, SmallVector<Value> &index,
                          SmallVector<Value> &writeIdx,
                          std::map<int, Value> &ints, unsigned axis) const {
    if (auto sliceLayout = layout.dyn_cast<SliceEncodingAttr>()) {
      auto dim = sliceLayout.getDim();
      assert(dim != axis && "Reduction axis cannot be sliced");
      auto parentLayout = sliceLayout.getParent();
      getWriteIndexBasic(rewriter, loc, parentLayout, index, writeIdx, ints,
                         axis);
      return;
    }

    writeIdx = index;
    auto sizePerThread = triton::gpu::getSizePerThread(layout);
    Value axisSizePerThread = ints[sizePerThread[axis]];
    Value _8 = ints[8];
    Value _16 = ints[16];
    if (layout.isa<BlockedEncodingAttr>()) {
      // A single thread owns axisSizePerThread contiguous values
      // on the reduction axis. After within thread reduction,
      // we would have a single accumulation every `axisSizePerThread`
      // contiguous values in the original tensor, so we would need
      // to map every `axisSizePerThread` to 1 value in smem as:
      // writeIdx[axis] = index[axis] / axisSizePerThread
      writeIdx[axis] = udiv(index[axis], axisSizePerThread);
    } else if (auto mmaLayout = layout.dyn_cast<MmaEncodingAttr>()) {
      if (!mmaLayout.isAmpere()) {
        llvm::report_fatal_error("Unsupported layout");
      }
      if (axis == 0) {
        // Because warpTileSize = [16, 8] and threadsPerWarp = [8, 4], each 8
        // rows in smem would correspond to a warp. The mapping
        // is: (warp_index) x 8 + (row index within warp)
        writeIdx[axis] =
            add(mul(udiv(index[axis], _16), _8), urem(index[axis], _8));
      } else {
        // Same as BlockedEncodingAttr case
        writeIdx[axis] = udiv(index[axis], axisSizePerThread);
      }
    } else {
      llvm::report_fatal_error("Unsupported layout");
    }
  }

  // Use shared memory for reduction within warps and across warps
  LogicalResult
  matchAndRewriteBasic(triton::ReduceOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const {
    ReduceOpHelper helper(op);
    Location loc = op.getLoc();
    unsigned axis = op.getAxis();

    auto srcTys = op.getInputTypes();
    auto srcLayout = helper.getSrcLayout();
    if (!helper.isSupportedLayout()) {
      assert(false && "Unexpected srcLayout in ReduceOpConversion");
    }
    // The order of the axes for the the threads within the warp
    auto srcOrd = triton::gpu::getOrder(srcLayout);
    auto sizePerThread = triton::gpu::getSizePerThread(srcLayout);
    auto srcShape = helper.getSrcShape();

    SmallVector<Type> elemPtrTys(srcTys.size());
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      auto ty = srcTys[i].getElementType();
      auto spirvElemTy = getTypeConverter()->convertType(ty);
      elemPtrTys[i] =
          spirv::PointerType::get(spirvElemTy, spirv::StorageClass::Workgroup);
    }
    auto spirvIndexTy = getTypeConverter()->getIndexType();
    auto indexPtrTy =
        spirv::PointerType::get(spirvIndexTy, spirv::StorageClass::Workgroup);

    auto smemShape = helper.getScratchConfigBasic();
    unsigned elems = product<unsigned>(smemShape);

    SmallVector<Value> smemBases(op.getNumOperands());
    smemBases[0] = bitcast(
        getSharedMemoryBase(loc, rewriter, op.getOperation()), elemPtrTys[0]);
    for (unsigned i = 1; i < op.getNumOperands(); ++i) {
      smemBases[i] =
          bitcast(gep(elemPtrTys[i - 1], smemBases[i - 1], i32_val(elems)),
                  elemPtrTys[i]);
    }

    unsigned srcElems = getTotalElemsPerThread(srcTys[0]);
    // Emits indices of the original tensor that each thread
    // would own
    auto srcIndices = emitIndices(loc, rewriter, srcLayout, srcTys[0]);
    auto srcValues = unpackInputs(loc, op, adaptor, rewriter);

    // Emits offsets (the offset from the base index)
    // of the original tensor that each thread would own
    // NOTE: Assumes offsets don't actually depend on type
    SmallVector<SmallVector<unsigned>> offset =
        emitOffsetForLayout(srcLayout, srcTys[0]);

    // Keep track of accumulations and their indices
    std::map<SmallVector<unsigned>, SmallVector<Value>> accs;
    std::map<SmallVector<unsigned>, SmallVector<Value>> indices;

    Region *combineOp = &op.getCombineOp();

    // reduce within threads
    for (unsigned i = 0; i < srcElems; ++i) {
      SmallVector<unsigned> key = offset[i];
      key[axis] = 0;
      bool isFirst = accs.find(key) == accs.end();
      accumulate(rewriter, *combineOp, accs[key], srcValues[i], isFirst);
      if (isFirst)
        indices[key] = srcIndices[i];
    }

    // cached int32 constants
    std::map<int, Value> ints;
    ints[0] = i32_val(0);
    for (int N = smemShape[axis] / 2; N > 0; N >>= 1)
      ints[N] = i32_val(N);
    ints[sizePerThread[axis]] = i32_val(sizePerThread[axis]);
    ints[8] = i32_val(8);
    ints[16] = i32_val(16);

    // reduce across threads
    for (auto it : accs) {
      const SmallVector<unsigned> &key = it.first;
      auto &acc = it.second;
      // get the writeIdx at which to write in smem
      SmallVector<Value> writeIdx;
      getWriteIndexBasic(rewriter, loc, srcLayout, indices[key], writeIdx, ints,
                         axis);

      // calculate the offset in smem for that writeIdx
      Value writeOffset = linearize(rewriter, loc, writeIdx, smemShape, srcOrd);
      SmallVector<Value> writePtrs(op.getNumOperands());
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        // Store the within-thread accumulated value into shared memory
        writePtrs[i] = gep(elemPtrTys[i], smemBases[i], writeOffset);
        store(acc[i], writePtrs[i]);
      }

      SmallVector<Value> readIdx(writeIdx.size(), ints[0]);
      // Perform parallel reduction with sequential addressing
      // E.g. We reduce `smemShape[axis]` elements into `smemShape[axis]/2`
      // elements using `smemShape[axis]/2` threads where each thread
      // would accumalte values that are `smemShape[axis]/2` apart
      // to avoid bank conflicts. Then we repeat with `smemShape[axis]/4`
      // threads, .. etc.
      for (int N = smemShape[axis] / 2; N > 0; N >>= 1) {
        // The readIdx will be N elements away on the reduction axis
        readIdx[axis] = ints[N];
        // If the writeIdx is greater or equal to N, do nothing
        Value readMask = icmp_ult(writeIdx[axis], ints[N]);
        // Calculate the readOffset, if readMask is False, readOffset=0
        // meaning we reduce the value at writeIdx with itself
        Value readOffset = select(
            readMask, linearize(rewriter, loc, readIdx, smemShape, srcOrd),
            ints[0]);
        SmallVector<Value> readPtrs(op.getNumOperands());
        for (unsigned i = 0; i < op.getNumOperands(); ++i) {
          // The readPtr is readOffset away from writePtr
          readPtrs[i] = gep(elemPtrTys[i], writePtrs[i], readOffset);
        }

        barrier();
        // Combine accumulator value from another thread
        SmallVector<Value> cur(op.getNumOperands());
        for (unsigned i = 0; i < op.getNumOperands(); ++i) {
          cur[i] = load(readPtrs[i]);
        }
        accumulate(rewriter, *combineOp, acc, cur, false);

        barrier();
        // Publish our new accumulator value to shared memory
        for (unsigned i = 0; i < op.getNumOperands(); ++i) {
          store(acc[i], writePtrs[i]);
        }
      }
    }

    barrier();

    // set output values
    SmallVector<Value> results(op.getNumOperands());
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      if (auto resultTy =
              op.getResult()[i].getType().dyn_cast<RankedTensorType>()) {
        // nd-tensor where n >= 1

        auto resultLayout = resultTy.getEncoding();

        unsigned resultElems = getTotalElemsPerThread(resultTy);
        auto resultIndices = emitIndices(loc, rewriter, resultLayout, resultTy);
        assert(resultIndices.size() == resultElems);

        SmallVector<Value> resultVals(resultElems);
        for (unsigned j = 0; j < resultElems; ++j) {
          SmallVector<Value> readIdx = resultIndices[j];
          readIdx.insert(readIdx.begin() + axis, ints[0]);
          Value readOffset =
              linearize(rewriter, loc, readIdx, smemShape, srcOrd);
          Value readPtr = gep(elemPtrTys[i], smemBases[i], readOffset);
          resultVals[j] = load(readPtr);
        }
        results[i] = getTypeConverter()->packLLElements(loc, resultVals,
                                                        rewriter, resultTy);
      } else {
        // 0d-tensor -> scalar
        results[i] = load(smemBases[i]);
      }
    }

    auto parentBlock = op.getOperation()->getBlock();
    rewriter.replaceOp(op, results);
    return success();
  }

  // Use warp shuffle for reduction within warps and shared memory for data
  // exchange across warps
  LogicalResult matchAndRewriteFast(triton::ReduceOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
    ReduceOpHelper helper(op);
    Location loc = op->getLoc();
    unsigned axis = adaptor.getAxis();

    auto srcTys = op.getInputTypes();
    auto srcLayout = helper.getSrcLayout();
    if (!helper.isSupportedLayout()) {
      assert(false && "Unexpected srcLayout in ReduceOpConversion");
    }
    auto srcOrd = triton::gpu::getOrder(srcLayout);
    auto srcShape = helper.getSrcShape();

    SmallVector<Type> elemPtrTys(srcTys.size());
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      auto ty = srcTys[i].getElementType();
      auto spirvElemTy = getTypeConverter()->convertType(ty);
      elemPtrTys[i] =
          spirv::PointerType::get(spirvElemTy, spirv::StorageClass::Workgroup);
    }
    auto spirvIndexTy = getTypeConverter()->getIndexType();
    auto indexPtrTy =
        spirv::PointerType::get(spirvIndexTy, spirv::StorageClass::Workgroup);

    auto smemShapes = helper.getScratchConfigsFast();
    unsigned elems = product<unsigned>(smemShapes[0]);
    unsigned maxElems = std::max(elems, product<unsigned>(smemShapes[1]));

    unsigned sizeIntraWarps = helper.getIntraWarpSizeWithUniqueData();
    unsigned sizeInterWarps = helper.getInterWarpSizeWithUniqueData();

    SmallVector<Value> smemBases(op.getNumOperands());
    if (sizeInterWarps > 1) {
      smemBases[0] = bitcast(
          getSharedMemoryBase(loc, rewriter, op.getOperation()), elemPtrTys[0]);
      for (unsigned i = 1; i < op.getNumOperands(); ++i) {
        smemBases[i] =
            bitcast(gep(elemPtrTys[i - 1], smemBases[i - 1], i32_val(maxElems)),
                    elemPtrTys[i]);
      }
    }
    unsigned srcElems = getTotalElemsPerThread(srcTys[0]);
    auto srcIndices = emitIndices(loc, rewriter, srcLayout, srcTys[0]);
    auto srcValues = unpackInputs(loc, op, adaptor, rewriter);

    std::map<SmallVector<unsigned>, SmallVector<Value>> accs;
    std::map<SmallVector<unsigned>, SmallVector<Value>> indices;

    // Assumes offsets don't actually depend on type
    SmallVector<SmallVector<unsigned>> offset =
        emitOffsetForLayout(srcLayout, srcTys[0]);

    auto *combineOp = &op.getCombineOp();

    // reduce within threads
    for (unsigned i = 0; i < srcElems; ++i) {
      SmallVector<unsigned> key = offset[i];
      key[axis] = 0;
      bool isFirst = accs.find(key) == accs.end();
      accumulate(rewriter, *combineOp, accs[key], srcValues[i], isFirst);
      if (isFirst)
        indices[key] = srcIndices[i];
    }

    Value threadId = getThreadId(rewriter, loc);
    Value warpSize = i32_val(32);
    Value warpId = udiv(threadId, warpSize);
    Value laneId = urem(threadId, warpSize);

    auto threadsPerWarp = triton::gpu::getThreadsPerWarp(srcLayout);
    auto warpsPerCTA = triton::gpu::getWarpsPerCTA(srcLayout);
    auto order = getOrder(srcLayout);
    SmallVector<Value> multiDimLaneId =
        delinearize(rewriter, loc, laneId, threadsPerWarp, order);
    SmallVector<Value> multiDimWarpId =
        delinearize(rewriter, loc, warpId, warpsPerCTA, order);

    Value laneIdAxis = multiDimLaneId[axis];
    Value warpIdAxis = multiDimWarpId[axis];

    Value zero = i32_val(0);
    Value laneZero = icmp_eq(laneIdAxis, zero);

    std::map<SmallVector<unsigned>, SmallVector<Value>> finalAccs;
    for (auto it : accs) {
      const SmallVector<unsigned> &key = it.first;
      SmallVector<Value> acc = it.second;

      // Reduce within warps
      for (unsigned N = sizeIntraWarps / 2; N > 0; N >>= 1) {
        SmallVector<Value> shfl(op.getNumOperands());
        for (unsigned i = 0; i < op.getNumOperands(); ++i) {
          shfl[i] = shflSync(loc, rewriter, acc[i], N);
        }
        accumulate(rewriter, *combineOp, acc, shfl, false);
      }

      if (sizeInterWarps == 1) {
        finalAccs[key] = acc;
        continue;
      }

      SmallVector<Value> writeIdx = indices[key];
      writeIdx[axis] = warpIdAxis;
      Value writeOffset =
          linearize(rewriter, loc, writeIdx, smemShapes[0], order);
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        Value writePtr = gep(elemPtrTys[i], smemBases[i], writeOffset);
        storeShared(rewriter, loc, writePtr, acc[i], laneZero);
      }
    }

    if (sizeInterWarps == 1) {
      SmallVector<Value> results(op.getNumOperands());
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        if (auto resultTy =
                op.getResult()[i].getType().dyn_cast<RankedTensorType>()) {
          auto resultLayout = resultTy.getEncoding().cast<SliceEncodingAttr>();
          unsigned resultElems = getTotalElemsPerThread(resultTy);
          SmallVector<SmallVector<unsigned>> resultOffset =
              emitOffsetForLayout(resultLayout, resultTy);
          SmallVector<Value> resultVals;
          for (int j = 0; j < resultElems; j++) {
            auto key = resultOffset[j];
            key.insert(key.begin() + axis, 0);
            resultVals.push_back(finalAccs[key][i]);
          }
          results[i] = getTypeConverter()->packLLElements(loc, resultVals,
                                                          rewriter, resultTy);
        } else
          results[i] = finalAccs.begin()->second[i];
      }
      rewriter.replaceOp(op, results);
      return success();
    }

    barrier();

    // The second round of shuffle reduction
    //   now the problem size: sizeInterWarps, s1, s2, .. , sn
    //   where sizeInterWarps is 2^m
    //
    // Each thread needs to process:
    //   elemsPerThread = sizeInterWarps * s1 * s2 .. Sn / numThreads
    unsigned numThreads =
        product<unsigned>(triton::gpu::getWarpsPerCTA(srcLayout)) * 32;
    unsigned elemsPerThread = std::max<unsigned>(elems / numThreads, 1);
    Value readOffset = threadId;
    for (unsigned round = 0; round < elemsPerThread; ++round) {
      // FIXME(Qingyi): need predicate icmp_slt(threadId,
      // i32_val(sizeInerWarps))
      SmallVector<Value> acc(op.getNumOperands());
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        Value readPtr = gep(elemPtrTys[i], smemBases[i], readOffset);
        acc[i] = load(readPtr);
      }

      for (unsigned N = sizeInterWarps / 2; N > 0; N >>= 1) {
        SmallVector<Value> shfl(op.getNumOperands());
        for (unsigned i = 0; i < op.getNumOperands(); ++i) {
          shfl[i] = shflSync(loc, rewriter, acc[i], N);
        }
        accumulate(rewriter, *combineOp, acc, shfl, false);
      }

      // only the first thread in each sizeInterWarps is writing
      Value writeOffset = readOffset;
      SmallVector<Value> writePtrs(op.getNumOperands());
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        writePtrs[i] = gep(elemPtrTys[i], smemBases[i], writeOffset);
      }
      Value threadIsNeeded = icmp_ult(threadId, i32_val(elems));
      Value laneIdModSizeInterWarps = urem(laneId, i32_val(sizeInterWarps));
      Value laneIdModSizeInterWarpsIsZero =
          icmp_eq(laneIdModSizeInterWarps, zero);
      Value pred = logic_and(threadIsNeeded, laneIdModSizeInterWarpsIsZero);

      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        storeShared(rewriter, loc, writePtrs[i], acc[i], pred);
      }

      if (round != elemsPerThread - 1) {
        readOffset = add(readOffset, i32_val(numThreads));
      }
    }

    barrier();

    // set output values
    SmallVector<Value> results(op.getNumOperands());
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      if (auto resultTy =
              op.getResult()[i].getType().dyn_cast<RankedTensorType>()) {
        // nd-tensor where n >= 1
        auto resultLayout = resultTy.getEncoding().cast<SliceEncodingAttr>();
        unsigned resultElems = getTotalElemsPerThread(resultTy);
        auto resultIndices = emitIndices(loc, rewriter, resultLayout, resultTy);
        assert(resultIndices.size() == resultElems);

        SmallVector<Value> resultVals(resultElems);
        for (size_t j = 0; j < resultElems; ++j) {
          SmallVector<Value> readIdx = resultIndices[j];
          readIdx.insert(readIdx.begin() + axis, i32_val(0));
          readOffset = linearize(rewriter, loc, readIdx, smemShapes[0], order);
          Value readPtr = gep(elemPtrTys[i], smemBases[i], readOffset);
          resultVals[j] = load(readPtr);
        }

        results[i] = getTypeConverter()->packLLElements(loc, resultVals,
                                                        rewriter, resultTy);
      } else {
        // 0d-tensor -> scalar
        results[i] = load(smemBases[i]);
      }
    }
    rewriter.replaceOp(op, results);

    return success();
  }
};

void populateReduceOpToSPIRVPatterns(
    TritonGPUToSPIRVTypeConverter &typeConverter, mlir::MLIRContext *context,
    RewritePatternSet &patterns, int numWarps,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, ModuleAllocation &allocation,
    ConvertTritonGPUOpToSPIRVPatternBase::IndexCacheInfo &indexCacheInfo,
    PatternBenefit benefit) {
  patterns.add<ReduceOpSPIRVConversion>(typeConverter, context, allocation,
                                        indexCacheInfo, benefit);
}
