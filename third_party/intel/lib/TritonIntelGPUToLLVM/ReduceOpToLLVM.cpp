#include "lib/Conversion/TritonGPUToLLVM/ReduceScanCommon.h"

#include <memory>
#include <tuple>
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/Support/MathExtras.h"

#include "PatternTritonGPUOpToLLVM.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::LLVM::linearize;
using ::mlir::triton::gpu::DistributedEncodingTrait;
using ::mlir::triton::gpu::getTotalElemsPerThread;

// FIXME: Remove the Intel workaround to align the ReduceOp lowering logic same
// to the upstream. Enable to A/B test cross-warp reduction logic.
//
// When enabled, we keep reduce step 1/2 (within-thread + within-warp) intact,
// but replace step 3 (cross-warp) with the common ReduceOpToLLVM.cpp logic:
// convert_layout through shared memory into a temporary layout, then perform up
// to two additional warp reductions until the reduction axis size becomes 1.
#ifndef TRITON_INTEL_REDUCE_USE_COMMON_CROSS_WARP
#define TRITON_INTEL_REDUCE_USE_COMMON_CROSS_WARP 0
#endif

#ifndef TRITON_INTEL_REDUCE_USE_LEFT_FOLD_THREAD_REDUCE
#define TRITON_INTEL_REDUCE_USE_LEFT_FOLD_THREAD_REDUCE 1
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
    Location loc = op->getLoc();
    auto accs = unpackInputs(loc, op, adaptor, rewriter);
    unsigned axis = op.getAxis();

    auto *ctx = op.getContext();

    // Remove block as we don't currently support it
    LinearLayout regLl = triton::gpu::toLinearLayout(helper.getSrcTy());
    // Remove broadcasting in registers as SliceLayout removes them
    auto removeBroadcast = actionRemoveBroadcastedRegs(regLl);
    if (!removeBroadcast.isIdentity()) {
      regLl = removeBroadcast.apply(regLl);
      for (auto &vals : accs) {
        vals = removeBroadcast.apply(vals);
      }
    }

    // First reduce all the values along axis within each thread.
    std::tie(regLl, accs) =
        reduceWithinThreads(op, std::move(regLl), std::move(accs), rewriter);

    // Then reduce across threads within a warp.
    std::tie(regLl, accs) =
        reduceWithinWarps(op, std::move(regLl), std::move(accs), rewriter);

    // reducedRegLaneLayout is used in the AllocationAnalysis to get the size
    // of the scratch space.
    assert(regLl ==
           ReduceOpHelper::reducedRegLaneLayout(helper.getSrcTy(), axis));

    // Step 3: reduce across warps.
#if TRITON_INTEL_REDUCE_USE_COMMON_CROSS_WARP
    // If we still need to reduce along warps / blocks:
    // Create temporary layout for reduction within warps.
    // By construction of tmpLl, we will iterate at most 2 times, as the maximum
    // number of warp / block bases is 64 * 16 = 32 * 32
    // That is, they fit in 2 rounds of warp reductions
    // Even more, if we do two rounds, getInterLayout will make sure that the
    // first one does not cross CTAs
    auto kAxis = *(regLl.getOutDimNames().begin() + axis);
    auto kBlock = StringAttr::get(ctx, "block");
    bool lastCvtCrossesCTAs = false;
    int i = 0;
    while (regLl.getOutDimSize(kAxis) != 1) {
      LinearLayout tmpLl = ReduceOpHelper::getInterLayout(regLl, axis);

      // Emit a barrier if we are reusing the shmem
      if (i > 0) {
        sync(rewriter, loc, lastCvtCrossesCTAs);
      }
      accs = convertLayoutValues(loc, rewriter, op, regLl, tmpLl, accs);
      lastCvtCrossesCTAs = !mlir::isCvtDimSync(regLl, tmpLl, kBlock);

      std::tie(regLl, accs) =
          reduceWithinWarps(op, std::move(tmpLl), std::move(accs), rewriter);
      ++i;
    }
    assert(i <= 2 && "expected at most 2 rounds of warp reductions");
    // Remove the axis dimension, which at this point is of size 1
    regLl = removeStandardDim(regLl, axis);

    // Convert to output layout if we didn't fit the warp bases within zero
    // bases in the tmpLl
    if (auto resultTy =
            dyn_cast<RankedTensorType>(op.getResult()[0].getType())) {
      auto outputLayout = triton::gpu::toLinearLayout(resultTy);
      if (regLl != outputLayout) {
        // Reuse the shmem
        sync(rewriter, loc, lastCvtCrossesCTAs);
        accs =
            convertLayoutValues(loc, rewriter, op, regLl, outputLayout, accs);
      }
    }

    packResults(op, accs, rewriter);
#else

    if (!helper.isReduceWithinCTA())
      return rewriter.notifyMatchFailure(op,
                                         "cross-CTA reduction not supported");

    std::map<SmallVector<unsigned>, SmallVector<Value>> intelAccs;
    std::map<SmallVector<unsigned>, SmallVector<Value>> indices;

    // Scatter to Intel’s canonical maps.
    if (failed(scatterReducedRegsToAccsAndIndices(
            op, helper, regLl, accs, intelAccs, indices, rewriter)))
      return failure();

    if (helper.isWarpSynchronous()) {
      packResultsIntel(op, helper, intelAccs, rewriter);
      return success();
    }

    reduceAcrossWarpsIntel(op, helper, intelAccs, indices, rewriter, regLl);
#endif

    return success();
  }

private:
  const TargetInfoBase &targetInfo;

  // Reduce values using a tree of the given arity. Arity=3 generates
  // combine(combine(a, b), c) groups that LLVM folds into ternary
  // instructions (e.g. v_maximum3_f32 on AMD).
  SmallVector<Value> treeReduce(Location loc,
                                ConversionPatternRewriter &rewriter,
                                Region &combineOp,
                                SmallVector<SmallVector<Value>> values,
                                unsigned arity) const {
    assert(!values.empty() && arity >= 2);
    while (values.size() > 1) {
      SmallVector<SmallVector<Value>> next;
      for (size_t i = 0; i < values.size(); i += arity) {
        size_t remaining = values.size() - i;
        size_t groupSize = std::min(static_cast<size_t>(arity), remaining);
        if (groupSize == 1) {
          next.push_back(std::move(values[i]));
        } else {
          SmallVector<Value> acc = std::move(values[i]);
          for (size_t j = 1; j < groupSize; ++j)
            accumulate(loc, rewriter, combineOp, acc, values[i + j]);
          next.push_back(std::move(acc));
        }
      }
      values = std::move(next);
    }
    return values.front();
  }

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
    auto operands = adaptor.getOperands();
    SmallVector<SmallVector<Value>> srcValues(op.getNumOperands());
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      srcValues[i] = unpackLLElements(loc, operands[i], rewriter);
    }
    return srcValues;
  }

  void sync(ConversionPatternRewriter &rewriter, Location loc,
            bool crossCTA) const {
    if (crossCTA) {
      targetInfo.clusterBarrier(loc, rewriter);
    } else {
      targetInfo.barrier(loc, rewriter, triton::gpu::AddrSpace::Local);
    }
  }

  void packVectorized(SmallVector<SmallVector<Value>> &accs,
                      ConversionPatternRewriter &rewriter) const {
    auto loc = accs.front().front().getLoc();
    for (auto &acc : accs) {
      SmallVector<Value> packedAcc;
      for (unsigned reg = 0; reg < acc.size(); reg += 2) {
        auto vector = packLLVector(loc, {acc[reg], acc[reg + 1]}, rewriter);
        packedAcc.emplace_back(std::move(vector));
      }
      acc = std::move(packedAcc);
    }
  }

  std::unique_ptr<Region> createVectorCombineRegion(
      Location loc, Type elemTy,
      ReduceOpHelper::InThreadVectorizeOpKind vectorizeKind,
      ConversionPatternRewriter &rewriter) const {
    if (vectorizeKind == ReduceOpHelper::InThreadVectorizeOpKind::None)
      return nullptr;
    MLIRContext *ctx = rewriter.getContext();
    auto vecTy = vec_ty(elemTy, 2);

    auto storage = std::make_unique<Region>();
    auto *block = new Block();
    storage->push_back(block);
    block->addArgument(vecTy, loc);
    block->addArgument(vecTy, loc);

    OpBuilder builder(ctx);
    builder.setInsertionPointToStart(block);
    Value result = ReduceOpHelper::createInThreadVectorizedCombineOp(
        builder, loc, vectorizeKind, block->getArgument(0),
        block->getArgument(1));
    triton::ReduceReturnOp::create(builder, loc, ValueRange{result});
    return storage;
  }

  void unpackVectorized(Location loc, SmallVector<SmallVector<Value>> &accs,
                        ConversionPatternRewriter &rewriter,
                        Region *reduction) const {
    for (auto &acc : accs) {
      SmallVector<Value> unpacked;
      for (Value val : acc) {
        auto elems = unpackLLVector(loc, val, rewriter);
        assert(elems.size() == 2 && "expected a 2-lane packed vector");
        if (reduction) {
          SmallVector<Value> cur = {elems[0]};
          accumulate(loc, rewriter, *reduction, cur, {elems[1]});
          unpacked.emplace_back(cur[0]);
        } else {
          unpacked.emplace_back(elems[0]);
          unpacked.emplace_back(elems[1]);
        }
      }
      acc = std::move(unpacked);
    }
  }

  // Reduce along op axis for elements that are in the same thread. The
  // accumulated value is stored in accs.
  std::pair<LinearLayout, SmallVector<SmallVector<Value>>>
  reduceWithinThreads(triton::ReduceOp op, LinearLayout layout,
                      SmallVector<SmallVector<Value>> accs,
                      ConversionPatternRewriter &rewriter) const {
    auto *ctx = op.getContext();
    auto loc = op.getLoc();
    unsigned axis = op.getAxis();
    auto kReg = str_attr("register");
    auto linearAttr = triton::gpu::LinearEncodingAttr::get(ctx, layout);
    auto basesPerDim = linearAttr.basesPerDim(kReg, /*skipBroadcast=*/true);
    unsigned axisPack = basesPerDim[axis];
    if (axisPack == 1) {
      return {std::move(layout), std::move(accs)};
    }

    ReduceOpHelper helper(op);
    auto vectorizeKind = helper.getInThreadVectorizeOpKind(
        axisPack, targetInfo.supportBitwidth16Elementwise(),
        targetInfo.supportBitwidth32Elementwise());
    bool vectorize =
        vectorizeKind != ReduceOpHelper::InThreadVectorizeOpKind::None;

    // Bring the registers that move the axis to the front
    auto perm = ReduceOpHelper::moveAxisBasesToFront(layout, axis, vectorize);
    if (!perm.isIdentity()) {
      layout = perm.apply(layout);
      for (auto &vals : accs) {
        vals = perm.apply(vals);
      }
    }

    // Pack the inputs into vector values
    if (vectorize)
      packVectorized(accs, rewriter);

    // If we pack along the reduction axis we need to process half the registers
    const auto &regBases = layout.getBases().lookup(kReg);
    bool packAlongAxis = vectorize && regBases.front()[axis] != 0;
    if (packAlongAxis)
      axisPack /= 2;

    // Create the vectorized region if needed
    auto elemTy =
        cast<RankedTensorType>(op.getOperandTypes().front()).getElementType();
    std::unique_ptr<Region> vectorCombineRegion =
        createVectorCombineRegion(loc, elemTy, vectorizeKind, rewriter);
    Region &combineRegion =
        vectorCombineRegion ? *vectorCombineRegion : op.getCombineOp();

    Operation &combinerOp = combineRegion.front().front();
    unsigned arity = targetInfo.getReductionTreeArity(&combinerOp);

    // Perform a tree reduction
    unsigned numOperands = accs.size();
    SmallVector<SmallVector<Value>> reduced(numOperands);
    unsigned regs = accs.front().size();
    for (unsigned regBase = 0; regBase < regs; regBase += axisPack) {
      // Transpose from [opIdx][reg] into [reg][opIdx]
      SmallVector<SmallVector<Value>> vals;
      for (unsigned i = 0; i < axisPack; ++i) {
        SmallVector<Value> cur(numOperands);
        for (unsigned opIdx = 0; opIdx < numOperands; ++opIdx) {
          cur[opIdx] = accs[opIdx][regBase + i];
        }
        vals.push_back(std::move(cur));
      }

#ifdef TRITON_INTEL_REDUCE_USE_LEFT_FOLD_THREAD_REDUCE
      // Use a deterministic left fold to avoid extra reassociation error in
      // low-precision reductions.
      SmallVector<Value> acc = vals.front();
      for (unsigned i = 1; i < vals.size(); ++i) {
        accumulate(loc, rewriter, combineRegion, acc, vals[i]);
      }
#else
      auto acc =
          treeReduce(loc, rewriter, combineRegion, std::move(vals), arity);
#endif
      for (unsigned opIdx = 0; opIdx < numOperands; ++opIdx) {
        reduced[opIdx].push_back(acc[opIdx]);
      }
    }
    accs = std::move(reduced);

    // Unpack the vector values into the accumulator values
    // Reduce one last time via the scalar combine op if we packed along the
    // axis
    if (vectorize) {
      Region *reduceAfterUnpacking =
          packAlongAxis ? &op.getCombineOp() : nullptr;
      unpackVectorized(loc, accs, rewriter, reduceAfterUnpacking);
    }

    // Update layout killing the axis bases along registers
    layout = ReduceOpHelper::zeroBasesAlongDimAndReorder(layout, axis, kReg);
    layout = actionRemoveBroadcastedRegs(layout).apply(layout);
    return {std::move(layout), std::move(accs)};
  }

  // Reduce across threads within each warp.
  std::pair<LinearLayout, SmallVector<SmallVector<Value>>>
  reduceWithinWarps(triton::ReduceOp op, LinearLayout layout,
                    SmallVector<SmallVector<Value>> accs,
                    ConversionPatternRewriter &rewriter) const {
    auto *ctx = op.getContext();
    auto kLane = str_attr("lane");
    const auto &laneBases = layout.getBases().lookup(kLane);
    unsigned reduceLaneIdMask = 0;
    for (unsigned bit = 0; bit < laneBases.size(); ++bit) {
      if (laneBases[bit][op.getAxis()] != 0) {
        reduceLaneIdMask |= 1u << bit;
      }
    }
    if (reduceLaneIdMask == 0) {
      return {std::move(layout), std::move(accs)};
    }

    unsigned regs = accs.front().size();
    for (unsigned reg = 0; reg < regs; ++reg) {
      SmallVector<Value> acc(op.getNumOperands());
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        acc[i] = accs[i][reg];
      }
      warpReduce(op, reduceLaneIdMask, acc, rewriter);
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        accs[i][reg] = acc[i];
      }
    }

    layout = ReduceOpHelper::zeroBasesAlongDimAndReorder(layout, op.getAxis(),
                                                         kLane);
    return {std::move(layout), std::move(accs)};
  }

  void warpReduce(triton::ReduceOp op, unsigned reduceLaneIdMask,
                  SmallVector<Value> &acc,
                  ConversionPatternRewriter &rewriter) const {
    // No reduction to do
    if (reduceLaneIdMask == 0)
      return;
    auto moduleOp = op->getParentOfType<ModuleOp>();
    unsigned warpSize =
        triton::gpu::TritonGPUDialect::getThreadsPerWarp(moduleOp);
    assert(reduceLaneIdMask < warpSize &&
           "expected reduce lane ID mask to be strictly less than warp size");
    // Try to use the redux op if it is supported by the target
    if (targetInfo.warpReduce(rewriter, op.getLoc(), acc, op,
                              reduceLaneIdMask)) {
      return;
    }
    // Not that it matters a lot, but a more reasonable iteration order would be
    // from bit 0 to bit llvm::Log2_32(warpSize) - 1. Changing this breaks a ton
    // of bitwise comparisons so we stick with the legacy inverse order
    for (int bit = llvm::Log2_32(warpSize) - 1; bit >= 0; --bit) {
      unsigned mask = 1u << bit;
      if ((reduceLaneIdMask & mask) == 0)
        continue;
      SmallVector<Value> shfl(op.getNumOperands());
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        shfl[i] = targetInfo.shuffleXor(rewriter, op.getLoc(), acc[i], mask);
      }
      accumulate(op.getLoc(), rewriter, op.getCombineOp(), acc, shfl);
    }
  }

  // Pack the accumulator values and replace the reduce op with the result.
  void packResults(triton::ReduceOp op, SmallVector<SmallVector<Value>> &accs,
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

  /// Convert the post-step1 reduced reg layout into Intel’s map key space.
  ///
  /// Keys are the logical tensor offsets for the reduced tensor (axis=0).
  /// This function also reconstructs `indices` in the same key order.
  LogicalResult scatterReducedRegsToAccsAndIndices(
      triton::ReduceOp op, ReduceOpHelper &helper, const LinearLayout &regLl,
      const SmallVector<SmallVector<Value>> &reducedAccs,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &indices,
      ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    MLIRContext *ctx = op.getContext();
    unsigned axis = op.getAxis();

    auto srcTy = cast<RankedTensorType>(op.getOperandTypes().front());
    auto reducedShape = llvm::to_vector(srcTy.getShape());
    reducedShape[axis] = 1;

    auto reducedEnc = triton::gpu::LinearEncodingAttr::get(ctx, regLl);
    auto reducedTensorTy =
        RankedTensorType::get(reducedShape, srcTy.getElementType(), reducedEnc);

    SmallVector<SmallVector<unsigned>> offsets =
        emitOffsetForLayout(reducedEnc, reducedTensorTy);
    if (offsets.size() != reducedAccs.front().size())
      return rewriter.notifyMatchFailure(op, "unexpected reduced reg count");

    // Generate indices for the reduced layout, matching Intel’s expectation.
    auto reducedIndices = emitIndices(loc, rewriter, targetInfo, reducedEnc,
                                      reducedTensorTy, /*withCTAOffset=*/true);
    if (reducedIndices.size() != offsets.size())
      return rewriter.notifyMatchFailure(op, "unexpected reduced index count");

    for (unsigned reg = 0; reg < offsets.size(); ++reg) {
      SmallVector<unsigned> key = offsets[reg];
      key[axis] = 0;

      SmallVector<Value> tuple(op.getNumOperands());
      for (unsigned opIdx = 0; opIdx < op.getNumOperands(); ++opIdx)
        tuple[opIdx] = reducedAccs[opIdx][reg];
      accs[SmallVector<unsigned>(key)] = std::move(tuple);

      SmallVector<Value> idx = reducedIndices[reg];
      idx[axis] = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
      indices[std::move(key)] = std::move(idx);
    }

    return success();
  }

  static unsigned computeReduceLaneIdMask(unsigned numLaneToReduce,
                                          unsigned interleave) {
    assert(numLaneToReduce && "numLaneToReduce cannot be 0");
    assert(llvm::isPowerOf2_32(numLaneToReduce) &&
           "numLaneToReduce must be a power of 2");
    assert(interleave && "interleave cannot be 0");
    assert(llvm::isPowerOf2_32(interleave) &&
           "interleave must be a power of 2");

    unsigned numBits = llvm::Log2_32(numLaneToReduce);
    unsigned baseMask = (1u << numBits) - 1u;
    numBits = llvm::Log2_32(interleave);
    return baseMask << numBits;
  }

  // Apply warp reduction across the given number of contiguous lanes using op
  // region and the accumulator values as source.
  void warpReduceIntel(ConversionPatternRewriter &rewriter, Location loc,
                       SmallVector<Value> &acc, triton::ReduceOp op,
                       unsigned numLaneToReduce, unsigned interleave,
                       Value pred = {}) const {
    auto success = targetInfo.warpReduce(
        rewriter, loc, acc, op,
        computeReduceLaneIdMask(numLaneToReduce, interleave));
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

  // Pack the accumulator values and replace the reduce op with the result.
  void
  packResultsIntel(triton::ReduceOp op, ReduceOpHelper &helper,
                   std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
                   ConversionPatternRewriter &rewriter) const {
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
      triton::ReduceOp op, ReduceOpHelper &helper,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &indices,
      SmallVector<Value> &smemBases, ConversionPatternRewriter &rewriter,
      LinearLayout &regLl) const {
    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
    unsigned axis = op.getAxis();
    auto smemShape = helper.getScratchRepShape();
    SmallVector<int64_t> smemShapeI64(smemShape.begin(), smemShape.end());

    auto reducedEnc =
        triton::gpu::LinearEncodingAttr::get(rewriter.getContext(), regLl);
    // Lezcano: We should move all the shared memory logic to use LLs natively
    auto kLane = rewriter.getStringAttr("lane");
    auto [multiDimLaneId, isRepresentativeLane] =
        delinearize(rewriter, loc, reducedEnc, smemShapeI64, kLane, laneId);
    auto kWarp = rewriter.getStringAttr("warp");
    auto [multiDimWarpId, isRepresentativeWarp] =
        delinearize(rewriter, loc, reducedEnc, smemShapeI64, kWarp, warpId);

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
  void accumulatePartialReductions(triton::ReduceOp op, ReduceOpHelper &helper,
                                   SmallVector<Value> &smemBases,
                                   ConversionPatternRewriter &rewriter) const {
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
        warpReduceIntel(rewriter, loc, acc, op, reduceLaneNumber,
                        1 /* interleave */, threadIsNeeded);
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
        sync(rewriter, loc, /*crossCTA=*/false);
      }
    }
  }

  // Load the final reduction from shared memory and replace the reduce result
  // with it.
  void loadReductionAndPackResult(triton::ReduceOp op, ReduceOpHelper &helper,
                                  SmallVector<unsigned> smemShape,
                                  SmallVector<Value> &smemBases,
                                  ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
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

  void reduceAcrossWarpsIntel(
      triton::ReduceOp op, ReduceOpHelper &helper,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &indices,
      ConversionPatternRewriter &rewriter, LinearLayout &regLl) const {
    Location loc = op.getLoc();

    // Compute a shared memory base per operand.
    auto smemShape = helper.getScratchRepShape();
    SmallVector<Value> smemBases =
        getSmemBases(op, product<unsigned>(smemShape), rewriter, targetInfo);

    storeWarpReduceToSharedMemory(op, helper, accs, indices, smemBases,
                                  rewriter, regLl);
    sync(rewriter, loc, /*crossCTA=*/false);

    accumulatePartialReductions(op, helper, smemBases, rewriter);

    // We could avoid this barrier in some of the layouts; not the general case.
    sync(rewriter, loc, /*crossCTA=*/false);

    // Set output values.
    loadReductionAndPackResult(op, helper, smemShape, smemBases, rewriter);
  }
};
} // namespace

void mlir::triton::intel::populateReduceOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<ReduceOpConversion>(typeConverter, targetInfo, benefit);
}
