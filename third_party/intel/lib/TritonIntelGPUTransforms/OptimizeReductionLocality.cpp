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

static Value createReshapeForReduction(PatternRewriter &rewriter, Location loc,
                                       Type type, Value val) {
  auto reshapeOp =
      rewriter.create<ReshapeOp>(loc, type, val, /*allow_reorder=*/true);
  reshapeOp.setEfficientLayout(true);
  return reshapeOp;
}

// clang-format off
  /// Optimize reduction with DPAS-encoded input.
  ///
  /// This optimization reshapes and converts input tensor layouts to split the
  /// reduction in three equivalent ones:
  ///
  /// This only works if the number of items for a given thread across dimension
  /// 0 and the execution size are equal to the sub-group size.
  ///
  /// First, we go from a DPAS layout to an equivalent blocked layout as follows:
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
  /// Blocked (#triton_gpu.blocked<{sizePerThread = [repCluster[0]*repeatCount, 1, 1, 1, 1], threadsPerWarp = [1, executionSize, 1, 1, 1], warpsPerCTA = [warpsPerCTA[0], 1, 1, warpsPerCTA[1], 1], order = [4, 0, 1, 2, 3]}>):
  /// ```
  ///                                                    warpsPerCTA[3]
  ///                    <------------------------------------------------------------------------------->
  ///                                 size[2]
  ///                    <---------------------------------->
  ///                     threadsPerWarp[1]
  ///                    <---------------->
  ///                  ^ t0 t1 t2 t3 ... tn t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn tn1 tn2 tn3 tn4 ... tnn ^
  ///                  | t0 t1 t2 t3 ... tn t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn tn1 tn2 tn3 tn4 ... tnn |
  ///                  | t0 t1 t2 t3 ... tn t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn tn1 tn2 tn3 tn4 ... tnn |
  ///                  | t0 t1 t2 t3 ... tn t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn tn1 tn2 tn3 tn4 ... tnn |
  ///                  | t0 t1 t2 t3 ... tn t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn tn1 tn2 tn3 tn4 ... tnn |
  ///                  | t0 t1 t2 t3 ... tn t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn tn1 tn2 tn3 tn4 ... tnn | warpsPerCTA[0]
  ///                  | t0 t1 t2 t3 ... tn t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn tn1 tn2 tn3 tn4 ... tnn |
  /// sizePerThread[0] | t0 t1 t2 t3 ... tn t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn tn1 tn2 tn3 tn4 ... tnn |
  ///                  | t0 t1 t2 t3 ... tn t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn tn1 tn2 tn3 tn4 ... tnn |
  ///                  v t0 t1 t2 t3 ... tn t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn tn1 tn2 tn3 tn4 ... tnn |
  /// ```
  /// So we can reduce on dimensions 4 and 2 to get to:
  /// ```
  ///                               warpsPerCTA[2]
  ///                    <------------------------------------>
  ///                     threadsPerWarp[1]
  ///                    <------------------>
  ///                  ^ t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn ^
  ///                  | t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn |
  ///                  | t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn |
  ///                  | t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn |
  ///                  | t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn |
  ///                  | t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn | warpsPerCTA[0]
  ///                  | t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn |
  /// sizePerThread[0] | t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn |
  ///                  | t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn |
  ///                  v t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn |
  /// ```
  /// After reshaping and layout conversion, we can get to the actual layout
  /// optimization we wanted to achieve:
  /// Blocked (#triton_gpu.blocked<{sizePerThread = [1, repCluster[0]*repeatCount], threadsPerWarp = [executionSize, 1], warpsPerCTA = [warpsPerCTA[0], warpsPerCTA[1]], order = [1, 0]}>):
  /// ```
  ///                               warpsPerCTA[1]
  ///                    <------------------------------------>
  ///                     sizePerThread[1]
  ///                    <------------------>
  ///                  ^ t0 t0 t0 t0 ... t0 tn1 tn1 tn1 ... tn1 ^
  ///                  | t1 t1 t1 t1 ... t1 tn2 tn2 tn2 ... tn2 |
  /// sizePerThread[0] | t2 t2 t2 t2 ... t2 tn3 tn3 tn3 ... tn3 | warpsPerCTA[0]
  ///                  | t3 t3 t3 t3 ... t3 tn4 tn4 tn4 ... tn4 |
  /// ```
  /// And reducing on dimension 1 and converting the layout to the original one
  /// leads to the same output as the original operation.
// clang-format on
struct DpasOperandPattern final : OpRewritePattern<ReduceOp> {
  using OpRewritePattern<ReduceOp>::OpRewritePattern;

  static constexpr int preferredNonReductionAxis = 0;
  static constexpr int preferredReductionAxis = 1;
  static constexpr int repCountReshapedAxis = 2;
  static constexpr int withinWarpXAxisReshapedAxis = 4;

  LogicalResult matchAndRewrite(ReduceOp op,
                                PatternRewriter &rewriter) const final {
    ValueRange operands = op.getOperands();
    // Allowing single operand for now
    if (operands.size() != 1)
      return failure();
    // Check this is has `triton_intel_gpu.dpas` encoding.
    Value operand = operands.front();
    auto type = cast<RankedTensorType>(operand.getType());
    auto encoding =
        llvm::dyn_cast_or_null<DpasEncodingAttr>(type.getEncoding());
    if (!encoding)
      return failure();

    // Axis 1 will lead to within-warp reduction.
    assert(type.getRank() == 2 && "Expecting 2D tensor");
    if (op.getAxis() != preferredReductionAxis)
      return failure();

    // We want to transpose matrices of (threads_per_warp)^2 shape for now.
    if ( // X axis condition
        encoding.getExecutionSize() != encoding.getSubGroupSize() ||
        // Y axis condition
        encoding.getRepeatCount() * encoding.getRepCluster()[0] !=
            encoding.getSubGroupSize())
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

    operand = convertToOriginalType(op, rewriter, operand);

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

    constexpr size_t rank = 5;
    std::array<int64_t, rank> shape{
        // Y axis
        oldShape[0],
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
    std::array<unsigned, rank> sizePerThread{oldEncoding.getRepeatCount() *
                                                 oldEncoding.getRepCluster()[0],
                                             1, 1, 1, 1};
    std::array<unsigned, rank> threadsPerWarp{1, oldEncoding.getExecutionSize(),
                                              1, 1, 1};
    std::array<unsigned, rank> warpsPerCTA{oldEncoding.getWarpsPerCTA()[0], 1,
                                           1, oldEncoding.getWarpsPerCTA()[1],
                                           1};
    std::array<unsigned, rank> order{4, 0, 1, 2, 3};
    CTALayoutAttr ctaLayout = getIdentityCTALayoutAttr(rewriter, rank);

    auto encoding = rewriter.getAttr<BlockedEncodingAttr>(
        sizePerThread, threadsPerWarp, warpsPerCTA, order, ctaLayout);

    RankedTensorType type =
        RankedTensorType::get(shape, oldType.getElementType(), encoding);

    // Although this is a NOP, we have to pass allow_reorder=true as static
    // analysis will fail to infer it.
    return createReshapeForReduction(rewriter, op.getLoc(), type, val);
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

    constexpr size_t rank = 3;
    ArrayRef<int64_t> shape = oldType.getShape();
    std::array<unsigned, rank> sizePerThread{1, dpasEncoding.getExecutionSize(),
                                             1};
    std::array<unsigned, rank> threadsPerWarp{dpasEncoding.getExecutionSize(),
                                              1, 1};
    std::array<unsigned, rank> warpsPerCTA{dpasEncoding.getWarpsPerCTA()[0], 1,
                                           dpasEncoding.getWarpsPerCTA()[1]};
    std::array<unsigned, rank> order{2, 0, 1};
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

    constexpr size_t rank = 2;
    std::array<int64_t, rank> shape{oldShape[0], oldShape[1] * oldShape[2]};
    std::array<unsigned, rank> sizePerThread{1,
                                             oldEncoding.getSizePerThread()[1]};
    std::array<unsigned, rank> threadsPerWarp{
        oldEncoding.getThreadsPerWarp()[0], 1};
    std::array<unsigned, rank> warpsPerCTA{oldEncoding.getWarpsPerCTA()[0],
                                           oldEncoding.getWarpsPerCTA()[2]};
    std::array<unsigned, rank> order{1, 0};
    CTALayoutAttr ctaLayout = getIdentityCTALayoutAttr(rewriter, rank);

    auto encoding = rewriter.getAttr<BlockedEncodingAttr>(
        sizePerThread, threadsPerWarp, warpsPerCTA, order, ctaLayout);

    RankedTensorType type =
        RankedTensorType::get(shape, oldType.getElementType(), encoding);

    // Although this is a NOP, we have to pass allow_reorder=true as static
    // analysis will fail to infer it.
    return createReshapeForReduction(rewriter, op.getLoc(), type, val);
  }

  Value performFinalReduction(ReduceOp op, PatternRewriter &rewriter,
                              Value val) const {
    return performReduction(op, rewriter, val, /*axis=*/preferredReductionAxis);
  }

  Value convertToOriginalType(ReduceOp op, PatternRewriter &rewriter,
                              Value val) const {
    return rewriter.create<ConvertLayoutOp>(
        op.getLoc(), op.getResult().front().getType(), val);
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
