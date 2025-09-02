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
#include "triton/Dialect/Triton/IR/Utility.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#define DEBUG_TYPE "tritonintelgpu-optimize-reduction-locality"

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONINTELGPUOPTIMIZEREDUCTIONLOCALITY
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"

namespace {
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
  ///                                      executionSize
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
  /// - Shape: [executionSize,
  ///           repeatCount,
  ///           repCluster[1],
  ///           repCluster[0],
  ///           warpsPerCTA[1],
  ///           oldShape[1] / (executionSize * repCluster[1] * warpsPerCTA[1]),
  ///           warpsPerCTA[0]]
  /// - Encoding: `#ttg.blocked<{
  ///                 sizePerThread = [1, repeatCount, repCluster[1], repCluster[0], 1, oldShape[1] / (executionSize * repCluster[1] * warpsPerCTA[1]), 1],
  ///                 threadsPerWarp = [executionSize, 1, 1, 1, 1, 1, 1],
  ///                 warpsPerCTA = [1, 1, 1, 1, warpsPerCTA[1], 1, warpsPerCTA[0]],
  ///                 order = [0, 1, 2, 3, 4, 5, 6]}>`.
  ///
  /// Notes:
  /// - The implicit [1, 0] order translates to taking elements from the
  ///   original encoding referring to X and Y dimension alternatively when
  ///   building the block layout.
  /// - Dimensions 1, 3 and 6 refer to the original dimension 0
  /// - Dimensions 0, 2, 4 and 5 refer to the original dimension 1
  /// - Order is preserved
  /// - We enforce repeatCount * repCluster[0] * warpsPerCTA[0] = oldShape[0]
  /// ```
  ///                                                                     sizePerThread[5]
  ///                                       <----------------------------------------------------------------------------------
  ///                                                                     warpsPerCTA[4]
  ///                                       <------------------------------------------------------------------------------->
  ///                                                 sizePerThread[2]
  ///                                       <---------------------------------->
  ///                                        threadsPerWarp[0]
  ///                                       <---------------->
  ///                  ^                  ^ t0 t1 t2 t3 ... tn t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn tn1 tn2 tn3 tn4 ... tnn ^
  ///                  |                  | t0 t1 t2 t3 ... tn t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn tn1 tn2 tn3 tn4 ... tnn |
  ///                  | sizePerThread[1] | t0 t1 t2 t3 ... tn t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn tn1 tn2 tn3 tn4 ... tnn |
  ///                  |                  | t0 t1 t2 t3 ... tn t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn tn1 tn2 tn3 tn4 ... tnn |
  ///                  |                  v t0 t1 t2 t3 ... tn t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn tn1 tn2 tn3 tn4 ... tnn |
  ///                  |                    ..................................................................................|
  ///                  |                    t0 t1 t2 t3 ... tn t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn tn1 tn2 tn3 tn4 ... tnn | warpsPerCTA[6]
  ///                  |                    t0 t1 t2 t3 ... tn t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn tn1 tn2 tn3 tn4 ... tnn |
  /// sizePerThread[3] |                    t0 t1 t2 t3 ... tn t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn tn1 tn2 tn3 tn4 ... tnn |
  ///                  |                    t0 t1 t2 t3 ... tn t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn tn1 tn2 tn3 tn4 ... tnn |
  ///                  v                    t0 t1 t2 t3 ... tn t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn tn1 tn2 tn3 tn4 ... tnn |
  /// ```
  /// So we can reduce on dimensions 2 and 4 (5 - 1 as we have already squashed
  /// dimension 2) to get to:
  /// ```
  ///                                                    warpsPerCTA[3]
  ///                                       <------------------------------------->
  ///                                        threadsPerWarp[0]
  ///                                       <---------------->
  ///                  ^                  ^ t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn ^
  ///                  |                  | t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn |
  ///                  | sizePerThread[1] | t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn |
  ///                  |                  | t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn |
  ///                  |                  v t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn |
  ///                  |                    .......................................|
  ///                  |                    t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn | warpsPerCTA[4]
  ///                  |                    t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn |
  /// sizePerThread[2] |                    t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn |
  ///                  |                    t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn |
  ///                  v                    t0 t1 t2 t3 ... tn tn1 tn2 tn3 ... tnn |
  /// ```
  ///
  /// Now on with step 2: After reshaping and layout conversion, we can get to
  /// the actual layout optimization we wanted to achieve by a simple layout
  /// conversion to:
  /// - Shape (unchanged): [executionSize,
  ///                       repeatCount,
  ///                       repCluster[0],
  ///                       warpsPerCTA[1],
  ///                       warpsPerCTA[0]]
  /// - Encoding: `#ttg.blocked<{
  ///                 sizePerThread = [executionSize, repeatCount * repCluster[0] / executionSize, 1, 1, 1],
  ///                 threadsPerWarp = [1, executionSize / repCluster[0], repCluster[0], 1, 1],
  ///                 warpsPerCTA = [1, 1, 1, warpsPerCTA[1], warpsPerCTA[0]],
  ///                 order = [0, 1, 2, 3, 4]}>`.
  /// Notes:
  /// - The layout conversion performs a sub-group transpose by setting
  ///   sizePerThread[0] to executionSize
  /// - sizePerThread[2] = 1 as we know
  ///   executionSize <= repeatCount * repCluster[0] (pattern application
  ///   condition: repeatCount * repCluster[0] % executionSize == 0).
  ///   We could say elements in dimension 2 are moved to dimension 1 to
  ///   simplify handling.
  /// - sizePerThread[1] value is set to keep size per thread
  /// - Dimensions 1, 2 and 4 refer to the original dimension 0
  /// - Dimensions 0, and 3 refer to the original dimension 1
  /// - Order is preserved
  ///
  /// Note at this point the transpose has already taken place. We just need a
  /// reshape to be an anchor for this (see layout conversion elimination pass):
  /// - Shape (unchanged): [executionSize,
  ///                       repeatCount * repCluster[0],
  ///                       warpsPerCTA[1],
  ///                       warpsPerCTA[0]]
  /// - Encoding: `#ttg.blocked<{
  ///                 sizePerThread = [executionSize, repeatCount * repCluster[0] / executionSize, 1, 1],
  ///                 threadsPerWarp = [1, executionSize, 1, 1],
  ///                 warpsPerCTA = [1, 1, warpsPerCTA[1], warpsPerCTA[0]],
  ///                 order = [0, 1, 2, 3]}>`.
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
  /// Notes:
  /// - The reshape simplifies the tensor and provides a layout anchor
  /// - We can get shape, sizePerThread, threadsPerWarp and warpsPerCTA for
  ///   dimension 1 by multiplying such values from dimensions 1 and 2 in the
  ///   old tensor.
  /// - Dimensions 1 and 3 refer to the original dimension 0
  /// - Dimensions 0, and 2 refer to the original dimension 1
  /// - Order is preserved
  /// And on with step 3, after reducing on dimensions 0 and 1 (2 - 1 as 0 is
  /// squashed), we'd get:
  /// ```
  ///                   ^ t0 ^
  ///                   | t1 |
  /// threadsPerWarp[0] | t2 | warpsPerCTA[1]
  ///                   | t3 |
  /// ```
  /// Now we can reshape to provide an anchor and go back to the original
  /// result shape (back to a 1D tensor):
  /// ```
  ///                   ^ t0 ^
  ///                   | t1 |
  /// threadsPerWarp[0] | t2 | warpsPerCTA[0]
  ///                   | t3 |
  /// ```
  /// And untranspose with a layout conversion to the original layout.
// clang-format on
struct DpasOperandPattern final : OpRewritePattern<ReduceOp> {
  using OpRewritePattern<ReduceOp>::OpRewritePattern;

  // Original reduction
  static constexpr int preferredNonReductionAxis = 0;
  static constexpr int preferredReductionAxis = 1;

  // Intermediate reductions
  static constexpr int finalElementwiseReductionAxis = 0;
  static constexpr int finalWarpsReductionAxis = 1;
  static constexpr int innerElementwiseReductionAxis = 2;
  static constexpr int outerElementwiseReductionAxis = 4;

  LogicalResult matchAndRewrite(ReduceOp op,
                                PatternRewriter &rewriter) const final {
    ValueRange operands = op.getOperands();
    // Allowing single operand for now
    if (operands.size() != 1)
      return failure();
    // Check this is has `ttig.dpas` encoding.
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
    unsigned threadsPerWarp = encoding.getThreadsPerWarp();
    if ( // X axis condition
        encoding.getExecutionSize() != threadsPerWarp ||
        // Y axis conditions
        (encoding.getRepeatCount() * encoding.getRepCluster()[0]) %
                threadsPerWarp !=
            0)
      return failure();

    // The X axis must contain enough elements to be fully covered
    // by the encoding.
    // i.e., the number of elements per warp allows
    // the elementwise reshape to happen.
    if (type.getShape()[1] <
        (encoding.getExecutionSize() * encoding.getRepCluster()[1] *
         encoding.getWarpsPerCTA()[1]))
      return failure();

    // The encoding should cover the Y axis.
    if (encoding.getRepeatCount() * encoding.getRepCluster()[0] *
            encoding.getWarpsPerCTA()[0] !=
        type.getShape()[0])
      return failure();

    LLVM_DEBUG(llvm::dbgs() << "Optimizing reduction: " << op << "\n");

    operand = reshapeForElementWiseReduction(op, rewriter, encoding);

    LLVM_DEBUG(llvm::dbgs()
               << "Reshaped for elementwise reduction: " << operand << "\n");

    operand = performInitialElementWiseReductions(op, rewriter, operand);

    LLVM_DEBUG(llvm::dbgs() << "Performed initial elementwise reductions: "
                            << operand << "\n");

    operand = convertLayoutForFinalReduction(op, rewriter, operand, encoding);

    LLVM_DEBUG(llvm::dbgs()
               << "Converted layout for final reduction: " << operand << "\n");

    operand = reshapeForFinalReduction(op, rewriter, operand, encoding);

    LLVM_DEBUG(llvm::dbgs()
               << "Reshaped for final reduction: " << operand << "\n");

    operand = performFinalElementwiseReduction(op, rewriter, operand);

    LLVM_DEBUG(llvm::dbgs()
               << "Final elementwise reduction performed: " << operand << "\n");

    operand = performFinalAcrossWarpsReduction(op, rewriter, operand);

    LLVM_DEBUG(llvm::dbgs() << "Final across-warps reduction performed: "
                            << operand << "\n");

    operand = reshapeToOriginalType(op, rewriter, operand, encoding);

    LLVM_DEBUG(llvm::dbgs()
               << "Reshaped to original type: " << operand << "\n");

    operand = convertLayoutToOriginalType(op, rewriter, operand);

    LLVM_DEBUG(llvm::dbgs()
               << "Converted layout to original type: " << operand << "\n");

    rewriter.replaceOp(op, operand);

    return success();
  }

private:
  Value reshapeForElementWiseReduction(ReduceOp op, PatternRewriter &rewriter,
                                       DpasEncodingAttr dpasEncoding) const {
    assert(op.getOperands().size() == 1 && "Expecting a single operand");

    Value val = op.getOperands().front();
    auto oldType = cast<RankedTensorType>(val.getType());
    ArrayRef<int64_t> oldShape = oldType.getShape();

    constexpr size_t rank = 7;
    std::array<int64_t, rank> shape{
        dpasEncoding.getExecutionSize(),
        dpasEncoding.getRepeatCount(),
        dpasEncoding.getRepCluster()[1],
        dpasEncoding.getRepCluster()[0],
        dpasEncoding.getWarpsPerCTA()[1],
        oldShape[1] /
            (dpasEncoding.getExecutionSize() * dpasEncoding.getRepCluster()[1] *
             dpasEncoding.getWarpsPerCTA()[1]),
        dpasEncoding.getWarpsPerCTA()[0]};
    std::array<unsigned, rank> sizePerThread{
        1,
        dpasEncoding.getRepeatCount(),
        dpasEncoding.getRepCluster()[1],
        dpasEncoding.getRepCluster()[0],
        1,
        static_cast<unsigned>(oldShape[1]) /
            (dpasEncoding.getExecutionSize() * dpasEncoding.getRepCluster()[1] *
             dpasEncoding.getWarpsPerCTA()[1]),
        1};
    std::array<unsigned, rank> threadsPerWarp{
        dpasEncoding.getExecutionSize(), 1, 1, 1, 1, 1, 1};
    std::array<unsigned, rank> warpsPerCTA{1,
                                           1,
                                           1,
                                           1,
                                           dpasEncoding.getWarpsPerCTA()[1],
                                           1,
                                           dpasEncoding.getWarpsPerCTA()[0]};
    constexpr std::array<unsigned, rank> order{0, 1, 2, 3, 4, 5, 6};
    CTALayoutAttr ctaLayout = CTALayoutAttr::getDefault(getContext(), rank);

    auto encoding = rewriter.getAttr<BlockedEncodingAttr>(
        sizePerThread, threadsPerWarp, warpsPerCTA, order, ctaLayout);

    RankedTensorType::Builder type(oldType);
    type.setShape(shape);
    type.setEncoding(encoding);

    // Although this is a NOP, we have to pass allow_reorder=true as static
    // analysis will fail to infer it.
    return rewriter.create<ReshapeOp>(op.getLoc(),
                                      static_cast<RankedTensorType>(type), val,
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

  Value performInitialElementWiseReductions(ReduceOp op,
                                            PatternRewriter &rewriter,
                                            Value val) const {
    return performReduction(
        op, rewriter,
        performReduction(op, rewriter, val,
                         /*axis=*/innerElementwiseReductionAxis),
        outerElementwiseReductionAxis);
  }

  Value convertLayoutForFinalReduction(ReduceOp op, PatternRewriter &rewriter,
                                       Value val,
                                       DpasEncodingAttr dpasEncoding) const {
    auto oldType = cast<RankedTensorType>(val.getType());
    RankedTensorType::Builder type(oldType);

    constexpr size_t rank = 5;
    std::array<unsigned, rank> sizePerThread{
        dpasEncoding.getExecutionSize(),
        dpasEncoding.getRepeatCount() * dpasEncoding.getRepCluster()[0] /
            dpasEncoding.getExecutionSize(),
        1, 1, 1};
    std::array<unsigned, rank> threadsPerWarp{
        1, dpasEncoding.getExecutionSize() / dpasEncoding.getRepCluster()[0],
        dpasEncoding.getRepCluster()[0], 1, 1};
    std::array<unsigned, rank> warpsPerCTA{1, 1, 1,
                                           dpasEncoding.getWarpsPerCTA()[1],
                                           dpasEncoding.getWarpsPerCTA()[0]};
    constexpr std::array<unsigned, rank> order{0, 1, 2, 3, 4};
    CTALayoutAttr ctaLayout = CTALayoutAttr::getDefault(getContext(), rank);

    auto encoding = rewriter.getAttr<BlockedEncodingAttr>(
        sizePerThread, threadsPerWarp, warpsPerCTA, order, ctaLayout);

    type.setEncoding(encoding);

    return rewriter.create<ConvertLayoutOp>(
        op.getLoc(), static_cast<RankedTensorType>(type), val);
  }

  Value reshapeForFinalReduction(ReduceOp op, PatternRewriter &rewriter,
                                 Value val,
                                 DpasEncodingAttr dpasEncoding) const {
    auto oldType = cast<RankedTensorType>(val.getType());
    ArrayRef<int64_t> oldShape = oldType.getShape();

    constexpr size_t rank = 4;
    std::array<int64_t, rank> shape{
        dpasEncoding.getExecutionSize(),
        dpasEncoding.getRepeatCount() * dpasEncoding.getRepCluster()[0],
        dpasEncoding.getWarpsPerCTA()[1], dpasEncoding.getWarpsPerCTA()[0]};
    std::array<unsigned, rank> sizePerThread{
        dpasEncoding.getExecutionSize(),
        dpasEncoding.getRepeatCount() * dpasEncoding.getRepCluster()[0] /
            dpasEncoding.getExecutionSize(),
        1, 1};
    std::array<unsigned, rank> threadsPerWarp{
        1, dpasEncoding.getExecutionSize(), 1, 1};
    std::array<unsigned, rank> warpsPerCTA{1, 1,
                                           dpasEncoding.getWarpsPerCTA()[1],
                                           dpasEncoding.getWarpsPerCTA()[0]};
    constexpr std::array<unsigned, rank> order{0, 1, 2, 3};
    CTALayoutAttr ctaLayout = CTALayoutAttr::getDefault(getContext(), rank);

    auto encoding = rewriter.getAttr<BlockedEncodingAttr>(
        sizePerThread, threadsPerWarp, warpsPerCTA, order, ctaLayout);

    RankedTensorType::Builder type(oldType);
    type.setShape(shape);
    type.setEncoding(encoding);

    // Although this is a NOP, we have to pass allow_reorder=true as static
    // analysis will fail to infer it.
    return rewriter.create<ReshapeOp>(op.getLoc(),
                                      static_cast<RankedTensorType>(type), val,
                                      /*allow_reorder=*/true,
                                      /*efficient_layout=*/true);
  }

  Value performFinalElementwiseReduction(ReduceOp op, PatternRewriter &rewriter,
                                         Value val) const {
    return performReduction(op, rewriter, val,
                            /*axis=*/finalElementwiseReductionAxis);
  }

  Value performFinalAcrossWarpsReduction(ReduceOp op, PatternRewriter &rewriter,
                                         Value val) const {
    return performReduction(op, rewriter, val,
                            /*axis=*/finalWarpsReductionAxis);
  }

  Value reshapeToOriginalType(ReduceOp op, PatternRewriter &rewriter, Value val,
                              DpasEncodingAttr dpasEncoding) const {
    RankedTensorType::Builder type(
        cast<RankedTensorType>(op.getResult().front().getType()));

    constexpr size_t rank = 2;
    std::array<unsigned, rank> sizePerThread{
        1, dpasEncoding.getRepeatCount() * dpasEncoding.getRepCluster()[0] /
               dpasEncoding.getExecutionSize()};
    std::array<unsigned, rank> threadsPerWarp{1,
                                              dpasEncoding.getExecutionSize()};
    std::array<unsigned, rank> warpsPerCTA{dpasEncoding.getWarpsPerCTA()[1],
                                           dpasEncoding.getWarpsPerCTA()[0]};
    constexpr std::array<unsigned, rank> order{0, 1};
    CTALayoutAttr ctaLayout = CTALayoutAttr::getDefault(getContext(), rank);

    auto parentEncoding = rewriter.getAttr<BlockedEncodingAttr>(
        sizePerThread, threadsPerWarp, warpsPerCTA, order, ctaLayout);

    type.setEncoding(SliceEncodingAttr::get(getContext(), 0, parentEncoding));

    return rewriter.create<ReshapeOp>(op.getLoc(),
                                      static_cast<RankedTensorType>(type), val,
                                      /*allow_reorder=*/true,
                                      /*efficient_layout=*/true);
  }

  Value convertLayoutToOriginalType(ReduceOp op, PatternRewriter &rewriter,
                                    Value val) const {
    return rewriter.create<ConvertLayoutOp>(
        op.getLoc(), op.getResult().front().getType(), val);
  }
};

struct TritonIntelGPUOptimizeReductionLocality final
    : impl::TritonIntelGPUOptimizeReductionLocalityBase<
          TritonIntelGPUOptimizeReductionLocality> {
  using Base::Base;

  void runOnOperation() final {
    Operation *op = getOperation();
    MLIRContext *ctx = op->getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<DpasOperandPattern>(ctx);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
} // namespace mlir::triton::gpu::intel
