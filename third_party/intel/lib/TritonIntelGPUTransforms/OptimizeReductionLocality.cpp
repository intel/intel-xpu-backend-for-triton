//===- OptimizeReductionLocality.cpp ------------------------------------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// This file implements the `tritonintelgpu-optimize-reduction-locality` pass.
//===----------------------------------------------------------------------===//

#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#define DEBUG_TYPE "tritonintelgpu-optimize-reduction-locality"

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONINTELGPUOPTIMIZEREDUCTIONLOCALITY
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"

namespace {
static CTALayoutAttr getIdentityCTALayoutAttr(PatternRewriter &rewriter,
                                              size_t rank) {
  SmallVector<unsigned> ctasPerCGA(rank, 1);
  SmallVector<unsigned> ctaSplitNum(rank, 1);
  SmallVector<unsigned> ctaOrder(rank);
  std::iota(std::rbegin(ctaOrder), std::rend(ctaOrder), 0);
  return rewriter.getAttr<CTALayoutAttr>(ctasPerCGA, ctaSplitNum, ctaOrder);
}

// clang-format off
  /// Optimize reduction with DPAS-encoded input.
  ///
  /// This optimization reshapes and converts input tensor layouts to split the
  /// reduction in three equivalent ones.
  ///
  /// This only works if the number of items for a given thread across dimension
  /// 0 and the execution size are equal to the sub-group size.
  ///
  /// We first want to reshape the input tensor to obtain a tensor with an
  /// equivalent encoding in terms of how elements are distributed across the
  /// device, but with more dimensions across the reduction axis. This way, we
  /// will be able to split the reduction in three steps:
  ///
  /// 1. Reduce within the work-item
  /// 2. Convert layout for better locality
  /// 3. Reduce within the sub-group and work-group
  ///
  /// Step 1 may involve more than one dimension depending on the input encoding
  /// (2 in this case). After step 1, each thread will hold a single element
  /// across the reduction axis dimension, so step 2 will be cheaper.
  ///
  /// For step 1, we first go from a DPAS layout to an equivalent blocked layout
  /// as follows:
  ///
  /// DPAS:
  /// ```
  ///                                                                  warpsPerCTA[1]
  ///                                   <-------------------------------------------------------------------------------->
  ///                                             repCluster[1]
  ///                                   <----------------------------------->
  ///                                      execution size
  ///                                   <---------------->
  ///                  ^             ^  t0 t1 t2 t3 ... tn t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn tn1 tn2 tn3 tn4 ... tnn ^
  ///                  |             |  t0 t1 t2 t3 ... tn t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn tn1 tn2 tn3 tn4 ... tnn |
  ///                  | repeatCount |  t0 t1 t2 t3 ... tn t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn tn1 tn2 tn3 tn4 ... tnn |
  ///                  |             |  t0 t1 t2 t3 ... tn t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn tn1 tn2 tn3 tn4 ... tnn |
  ///                  |             v  t0 t1 t2 t3 ... tn t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn tn1 tn2 tn3 tn4 ... tnn |
  ///                  |                t0 t1 t2 t3 ... tn t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn tn1 tn2 tn3 tn4 ... tnn | warpsPerCTA[0]
  ///                  |                t0 t1 t2 t3 ... tn t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn tn1 tn2 tn3 tn4 ... tnn |
  ///    repCluster[0] |                t0 t1 t2 t3 ... tn t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn tn1 tn2 tn3 tn4 ... tnn |
  ///                  |                t0 t1 t2 t3 ... tn t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn tn1 tn2 tn3 tn4 ... tnn |
  ///                  v                t0 t1 t2 t3 ... tn t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn tn1 tn2 tn3 tn4 ... tnn |
  /// ```
  /// Blocked (#triton_gpu.blocked<{sizePerThread = [executionSize, 1, 1, 1, 1, 1, 1], threadsPerWarp = [1, 1, 1, executionSize, 1, 1, 1], warpsPerCTA = [1, 1, warpsPerCTA[0], 1, 1, warpsPerCTA[1], 1], order = [3, 4, 5, 6, 0, 1, 2]}>):
  /// ```
  ///                                                                  warpsPerCTA[5]
  ///                                    <------------------------------------------------------------------------------->
  ///                                              getShape()[4]
  ///                                    <---------------------------------->
  ///                                     threadsPerWarp[3]
  ///                                    <---------------->
  ///               ^                  ^ t0 t1 t2 t3 ... tn t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn tn1 tn2 tn3 tn4 ... tnn ^
  ///               |                  | t0 t1 t2 t3 ... tn t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn tn1 tn2 tn3 tn4 ... tnn |
  ///               | sizePerThread[0] | t0 t1 t2 t3 ... tn t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn tn1 tn2 tn3 tn4 ... tnn |
  ///               |                  | t0 t1 t2 t3 ... tn t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn tn1 tn2 tn3 tn4 ... tnn |
  ///               |                  v t0 t1 t2 t3 ... tn t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn tn1 tn2 tn3 tn4 ... tnn |
  ///               |                    ..................................................................................|
  ///               |                    t0 t1 t2 t3 ... tn t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn tn1 tn2 tn3 tn4 ... tnn | warpsPerCTA[2]
  ///               |                    t0 t1 t2 t3 ... tn t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn tn1 tn2 tn3 tn4 ... tnn |
  /// getShape()[1] |                    t0 t1 t2 t3 ... tn t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn tn1 tn2 tn3 tn4 ... tnn |
  ///               |                    t0 t1 t2 t3 ... tn t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn tn1 tn2 tn3 tn4 ... tnn |
  ///               v                    t0 t1 t2 t3 ... tn t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn tn1 tn2 tn3 tn4 ... tnn |
  /// ```
  /// So we can reduce on dimensions 6 and 4 to get to:
  /// ```
  ///                                                 warpsPerCTA[3]
  ///                                    <------------------------------------->
  ///                                     threadsPerWarp[3]
  ///                                    <---------------->
  ///               ^                  ^ t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn ^
  ///               |                  | t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn |
  ///               | sizePerThread[0] | t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn |
  ///               |                  | t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn |
  ///               |                  v t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn |
  ///               |                    .......................................|
  ///               |                    t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn | warpsPerCTA[2]
  ///               |                    t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn |
  /// getShape()[1] |                    t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn |
  ///               |                    t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn |
  ///               v                    t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn |
  /// ```
  ///
  /// Now on with step 2: After reshaping and layout conversion, we can get to
  /// the actual layout optimization we wanted to achieve:
  /// Blocked (#triton_gpu.blocked<{sizePerThread = [1, 1, 1, executionSize], threadsPerWarp = [executionSize, 1, 1, 1], warpsPerCTA = [1, 1, warpsPerCTA[0], warpsPerCTA[1]], order = [3, 0, 1, 2]}>):
  /// ```
  ///                               warpsPerCTA[3]
  ///                    <------------------------------------>
  ///                     sizePerThread[3]
  ///                    <------------------>
  ///                   ^ t0 t0 t0 t0 ... t0 tn1 tn1 tn1 ... tn1 ^
  ///                   | t1 t1 t1 t1 ... t1 tn2 tn2 tn2 ... tn2 |
  /// threadsPerWarp[0] | t2 t2 t2 t2 ... t2 tn3 tn3 tn3 ... tn3 | warpsPerCTA[2]
  ///                   | t3 t3 t3 t3 ... t3 tn4 tn4 tn4 ... tn4 |
  /// ```
  /// And on with step 3, after reducing on dimension 3, we'd get:
  /// Blocked (#triton_gpu.blocked<{sizePerThread = [1, 1, 1, executionSize], threadsPerWarp = [executionSize, 1, 1, 1], warpsPerCTA = [1, 1, warpsPerCTA[0], warpsPerCTA[1]], order = [3, 0, 1, 2]}>):
  /// Sliced (#triton_gpu.sliced<{dim = 3, parent = #blocked}>)
  /// ```
  ///                   ^ t0 ^
  ///                   | t1 |
  /// threadsPerWarp[0] | t2 | warpsPerCTA[2]
  ///                   | t3 |
  /// ```
  /// Reshaping from this layout to the final state would not work, as we would
  /// end up modifying the number of elements per work-item (not allowed in
  /// `reshape`).
  ///
  /// In order to avoid that, we can just convert the layout to a sliced layout
  /// equivalent to the end product we want to achieve:
  /// Blocked (#triton_gpu.blocked<{sizePerThread = [1, 1, 1, executionSize], threadsPerWarp = [executionSize, 1, 1, 1], warpsPerCTA = [1, 1, warpsPerCTA[0], warpsPerCTA[1]], order = [3, 0, 1, 2]}>)
  /// Sliced (#triton_gpu.sliced<{dim = 3, parent = #blocked}>)
  /// ```
  ///                   ^ t0 ^
  ///                   | t0 |
  /// threadsPerWarp[0] | t0 | warpsPerCTA[2]
  ///                   | t0 |
  /// ```
  /// And just reshape to the final type using a NOP `reshape`.
// clang-format on
struct DpasOperandPattern final : OpRewritePattern<ReduceOp> {
  using OpRewritePattern<ReduceOp>::OpRewritePattern;

  static constexpr int preferredNonReductionAxis = 0;
  static constexpr int finalReductionAxis = 3;
  static constexpr int preferredReductionAxis = 1;
  static constexpr int repCountReshapedAxis = 4;
  static constexpr int withinWarpXAxisReshapedAxis = 6;

  LogicalResult matchAndRewrite(ReduceOp op,
                                PatternRewriter &rewriter) const final {
    ValueRange operands = op.getOperands();
    // Allowing single operand for now
    if (operands.size() != 1)
      return failure();
    // Check this is has `triton_intel_gpu.dpas` encoding.
    Value operand = operands.front();
    auto type = cast<RankedTensorType>(operand.getType());
    // Only support reduction after 2D-dot for now.
    if (type.getRank() != 2)
      return failure();
    auto encoding =
        llvm::dyn_cast_or_null<DpasEncodingAttr>(type.getEncoding());
    if (!encoding)
      return failure();

    // Axis 1 will lead to within-warp reduction.
    if (op.getAxis() != preferredReductionAxis)
      return failure();

    // We want to transpose matrices of N*threads_per_warpxthreads_per_warp
    // shape.
    if ( // X axis condition
        encoding.getExecutionSize() != encoding.getSubGroupSize() ||
        // Y axis conditions
        (encoding.getRepeatCount() * encoding.getRepCluster()[0]) %
                encoding.getSubGroupSize() !=
            0)
      return failure();

    LLVM_DEBUG(llvm::dbgs() << "Optimizing reduction: " << op << "\n");

    operand = reshapeForElementWiseReduction(op, rewriter);

    LLVM_DEBUG(llvm::dbgs()
               << "Reshaped for elementwise reduction: " << operand << "\n");

    operand = performElementWiseReductionAcrossRepCounts(op, rewriter, operand);

    LLVM_DEBUG(llvm::dbgs()
               << "Performed elementwise reduction across repCount: " << operand
               << "\n");

    operand = performElementWiseReductionWithinRepCount(op, rewriter, operand);

    LLVM_DEBUG(llvm::dbgs()
               << "Performed elementwise reduction within repCount: " << operand
               << "\n");

    operand = convertLayoutForFinalReduction(op, rewriter, operand);

    LLVM_DEBUG(llvm::dbgs()
               << "Converted layout for final reduction: " << operand << "\n");

    operand = reshapeForFinalReduction(op, rewriter, operand);

    LLVM_DEBUG(llvm::dbgs()
               << "Reshaped for final reduction: " << operand << "\n");

    operand = performFinalReduction(op, rewriter, operand);

    LLVM_DEBUG(llvm::dbgs()
               << "Final reduction performed: " << operand << "\n");

    operand = convertLayoutToOriginalType(op, rewriter, operand);

    LLVM_DEBUG(llvm::dbgs()
               << "Converted layout to original type: " << operand << "\n");

    operand = reshapeToOriginalType(op, rewriter, operand);

    LLVM_DEBUG(llvm::dbgs()
               << "Reshaped to original type: " << operand << "\n");

    rewriter.replaceOp(op, operand);

    return success();
  }

private:
  Value reshapeForElementWiseReduction(ReduceOp op,
                                       PatternRewriter &rewriter) const {
    assert(op.getOperands().size() == 1 && "Expecting a single operand");

    Value val = op.getOperands().front();
    auto oldType = cast<RankedTensorType>(val.getType());
    ArrayRef<int64_t> oldShape = oldType.getShape();
    auto oldEncoding = cast<DpasEncodingAttr>(oldType.getEncoding());

    constexpr size_t rank = 7;
    std::array<int64_t, rank> shape{
        // Y axis contiguous elements handled by a single thread.
        oldEncoding.getExecutionSize(),
        // Y axis contiguous elements handled by a single thread.
        // Needs to be split from previous dimension to perform transpose.
        (oldEncoding.getRepeatCount() * oldEncoding.getRepCluster()[0]) /
            oldEncoding.getExecutionSize(),
        // Y axis rest.
        oldShape[0] /
            (oldEncoding.getRepeatCount() * oldEncoding.getRepCluster()[0]),
        // X axis contiguous elements distributed within individual threads in a
        // warp.
        oldEncoding.getExecutionSize(),
        // X axis contiguous elements distributed within a warp.
        oldEncoding.getRepCluster()[1],
        // X axis number of warps.
        oldEncoding.getWarpsPerCTA()[1],
        // X axis rest.
        oldShape[1] /
            (oldEncoding.getExecutionSize() * oldEncoding.getRepCluster()[1] *
             oldEncoding.getWarpsPerCTA()[1])};
    std::array<unsigned, rank> sizePerThread{
        oldEncoding.getExecutionSize(), 1, 1, 1, 1, 1, 1};
    std::array<unsigned, rank> threadsPerWarp{
        1, 1, 1, oldEncoding.getExecutionSize(), 1, 1, 1};
    std::array<unsigned, rank> warpsPerCTA{
        1, 1, oldEncoding.getWarpsPerCTA()[0],
        1, 1, oldEncoding.getWarpsPerCTA()[1],
        1};
    std::array<unsigned, rank> order{3, 4, 5, 6, 0, 1, 2};
    CTALayoutAttr ctaLayout = getIdentityCTALayoutAttr(rewriter, rank);

    auto encoding = rewriter.getAttr<BlockedEncodingAttr>(
        sizePerThread, threadsPerWarp, warpsPerCTA, order, ctaLayout);

    RankedTensorType type =
        RankedTensorType::get(shape, oldType.getElementType(), encoding);

    // Although this is a NOP, we have to pass allow_reorder=true as static
    // analysis will fail to infer it.
    return rewriter.create<ReshapeOp>(op.getLoc(), type, val,
                                      /*allow_reorder=*/true,
                                      /*efficient_layout=*/true);
  }

  Value performReduction(ReduceOp op, PatternRewriter &rewriter, Value val,
                         int axis) const {
    assert(axis >= 0 && "Expecting positive axis");

    auto newOp = rewriter.create<ReduceOp>(op.getLoc(), val, /*axis=*/axis);
    auto &newCombineOp = newOp.getCombineOp();
    rewriter.cloneRegionBefore(op.getCombineOp(), newCombineOp,
                               newCombineOp.end());
    assert(newOp.getResult().size() == 1 && "Expecting single result");
    return newOp.getResult().front();
  }

  Value performElementWiseReductionWithinRepCount(ReduceOp op,
                                                  PatternRewriter &rewriter,
                                                  Value val) const {
    return performReduction(op, rewriter, val, /*axis=*/repCountReshapedAxis);
  }

  Value performElementWiseReductionAcrossRepCounts(ReduceOp op,
                                                   PatternRewriter &rewriter,
                                                   Value val) const {
    return performReduction(op, rewriter, val,
                            /*axis=*/withinWarpXAxisReshapedAxis);
  }

  Value convertLayoutForFinalReduction(ReduceOp op, PatternRewriter &rewriter,
                                       Value val) const {
    assert(op.getOperands().size() == 1 && "Expecting a single operand");

    auto oldType = cast<RankedTensorType>(val.getType());
    auto dpasEncoding = cast<DpasEncodingAttr>(
        cast<RankedTensorType>(op.getOperands().front().getType())
            .getEncoding());

    constexpr size_t rank = 5;
    ArrayRef<int64_t> shape = oldType.getShape();
    std::array<unsigned, rank> sizePerThread{
        1, 1, 1, dpasEncoding.getExecutionSize(), 1};
    std::array<unsigned, rank> threadsPerWarp{dpasEncoding.getExecutionSize(),
                                              1, 1, 1, 1};
    std::array<unsigned, rank> warpsPerCTA{1, 1,
                                           dpasEncoding.getWarpsPerCTA()[0], 1,
                                           dpasEncoding.getWarpsPerCTA()[1]};
    std::array<unsigned, rank> order{3, 4, 0, 1, 2};
    CTALayoutAttr ctaLayout = getIdentityCTALayoutAttr(rewriter, rank);

    auto encoding = rewriter.getAttr<BlockedEncodingAttr>(
        sizePerThread, threadsPerWarp, warpsPerCTA, order, ctaLayout);

    RankedTensorType type =
        RankedTensorType::get(shape, oldType.getElementType(), encoding);

    return rewriter.create<ConvertLayoutOp>(op.getLoc(), type, val);
  }

  Value reshapeForFinalReduction(ReduceOp op, PatternRewriter &rewriter,
                                 Value val) const {
    auto oldType = cast<RankedTensorType>(val.getType());
    ArrayRef<int64_t> oldShape = oldType.getShape();
    auto oldEncoding = cast<BlockedEncodingAttr>(oldType.getEncoding());

    constexpr size_t rank = 4;
    std::array<int64_t, rank> shape{oldShape[0], oldShape[1], oldShape[2],
                                    oldShape[3] * oldShape[4]};
    std::array<unsigned, rank> sizePerThread{1, 1, 1,
                                             oldEncoding.getSizePerThread()[3]};
    std::array<unsigned, rank> threadsPerWarp{
        oldEncoding.getThreadsPerWarp()[0], 1, 1, 1};
    std::array<unsigned, rank> warpsPerCTA{
        1, 1, oldEncoding.getWarpsPerCTA()[2], oldEncoding.getWarpsPerCTA()[4]};
    std::array<unsigned, rank> order{3, 0, 1, 2};
    CTALayoutAttr ctaLayout = getIdentityCTALayoutAttr(rewriter, rank);

    auto encoding = rewriter.getAttr<BlockedEncodingAttr>(
        sizePerThread, threadsPerWarp, warpsPerCTA, order, ctaLayout);

    RankedTensorType type =
        RankedTensorType::get(shape, oldType.getElementType(), encoding);

    // Although this is a NOP, we have to pass allow_reorder=true as static
    // analysis will fail to infer it.
    return rewriter.create<ReshapeOp>(op.getLoc(), type, val,
                                      /*allow_reorder=*/true,
                                      /*efficient_layout=*/true);
  }

  Value performFinalReduction(ReduceOp op, PatternRewriter &rewriter,
                              Value val) const {
    return performReduction(op, rewriter, val, /*axis=*/finalReductionAxis);
  }

  Value convertLayoutToOriginalType(ReduceOp op, PatternRewriter &rewriter,
                                    Value val) const {
    auto oldType = cast<RankedTensorType>(val.getType());
    auto dpasEncoding = cast<DpasEncodingAttr>(
        cast<RankedTensorType>(op.getOperands().front().getType())
            .getEncoding());

    // Only Y axis (X axis has already been reduced)
    constexpr size_t rankBeforeLastReduction = 4;
    ArrayRef<int64_t> shape = oldType.getShape();
    std::array<unsigned, rankBeforeLastReduction> sizePerThread{
        dpasEncoding.getExecutionSize(), 1, 1, 1};
    std::array<unsigned, rankBeforeLastReduction> threadsPerWarp{
        1, 1, 1, dpasEncoding.getExecutionSize()};
    std::array<unsigned, rankBeforeLastReduction> warpsPerCTA{
        1, 1, dpasEncoding.getWarpsPerCTA()[0],
        dpasEncoding.getWarpsPerCTA()[1]};
    std::array<unsigned, rankBeforeLastReduction> order{3, 0, 1, 2};
    CTALayoutAttr ctaLayout =
        getIdentityCTALayoutAttr(rewriter, rankBeforeLastReduction);

    auto blockedEncoding = rewriter.getAttr<BlockedEncodingAttr>(
        sizePerThread, threadsPerWarp, warpsPerCTA, order, ctaLayout);
    auto encoding = rewriter.getAttr<SliceEncodingAttr>(finalReductionAxis,
                                                        blockedEncoding);

    RankedTensorType type =
        RankedTensorType::get(shape, oldType.getElementType(), encoding);

    return rewriter.create<ConvertLayoutOp>(op.getLoc(), type, val);
  }

  Value reshapeToOriginalType(ReduceOp op, PatternRewriter &rewriter,
                              Value val) const {
    return rewriter.create<ReshapeOp>(op.getLoc(),
                                      op.getResult().front().getType(), val,
                                      /*allow_reorder=*/true,
                                      /*efficient_layout=*/true);
  }
};

struct TritonIntelGPUOptimizeReductionLocality final
    : impl::TritonIntelGPUOptimizeReductionLocalityBase<
          TritonIntelGPUOptimizeReductionLocality> {
  using impl::TritonIntelGPUOptimizeReductionLocalityBase<
      TritonIntelGPUOptimizeReductionLocality>::
      TritonIntelGPUOptimizeReductionLocalityBase;

  void runOnOperation() final {
    Operation *op = getOperation();
    MLIRContext *ctx = op->getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<DpasOperandPattern>(ctx);
    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
} // namespace mlir::triton::gpu::intel
