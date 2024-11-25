//===- OptimizeElementwiseParallelism.cpp -------------------------------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// This file implements the `tritonintelgpu-optimize-elementwise-parallelism`
/// pass.
//===----------------------------------------------------------------------===//

#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#define DEBUG_TYPE "tritonintelgpu-optimize-elementwise-parallelism"

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONINTELGPUOPTIMIZEELEMENTWISEPARALLELISM
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"

namespace {
bool isMultiWarpValidLayoutForUnbroadcast(const LinearLayout &linearLayout,
                                          int32_t numWorkGroupPos,
                                          PatternRewriter &rewriter) {
  StringAttr kLane = rewriter.getStringAttr("lane");
  StringAttr kWarp = rewriter.getStringAttr("warp");
  int32_t subGroupSize = linearLayout.getInDimSize(kLane);
  ArrayRef<int32_t> numContiguousPerWarp = linearLayout.getBasis(kWarp, 0);
  // Check the warp dimension hasn't been sliced away and we have n *
  // sub_group_size contiguous elements per warp.
  if (numContiguousPerWarp == ArrayRef<int32_t>{0} ||
      numContiguousPerWarp[0] % subGroupSize != 0)
    return false;
  int32_t expectedValue = numContiguousPerWarp[0] * 2;
  for (int32_t pos = 1; pos < numWorkGroupPos; ++pos) {
    if (linearLayout.getBasis(kWarp, pos) != ArrayRef<int32_t>{expectedValue})
      return false;
    expectedValue *= 2;
  }
  return true;
}

/// Return whether the input linear layout can be unbroadcasted.
///
/// A layout is valid for being "unbroadcasted" along its lanes if:
/// - The 'lane' input dimension is zero: this means the lane dimension has been
/// sliced.
/// - The size of the input 'block' dimension is 1. This is true for XPU
/// backend.
/// - The size of the input 'warp' dimension is 1 or there are n*sub_group_size
/// contiguous elements per warp.
///
/// Broadcasted layouts are layouts with sliced lane, warp or block (not
/// possible for XPU backend) dimensions, i.e., the same data is owned by
/// different threads.
bool isValidLayoutForUnbroadcast(const LinearLayout &linearLayout,
                                 PatternRewriter &rewriter) {
  StringAttr kLane = rewriter.getStringAttr("lane");
  StringAttr kWarp = rewriter.getStringAttr("warp");
  StringAttr kBlock = rewriter.getStringAttr("block");
  StringAttr kDim0 = rewriter.getStringAttr("dim0");
  // 'lane' dimension must have been sliced away completely.
  if (!linearLayout.sublayoutIsZero(kLane, kDim0))
    return false;
  // Only single block for now.
  if (linearLayout.getInDimSize(kBlock) != 1)
    return false;
  // 'warp' dimension hasn't been sliced away and there are n*sub_group_size
  // contiguous elements in each warp (or there is a single warp).
  int32_t numWorkGroupPos = linearLayout.getInDimSizeLog2(kWarp);
  return numWorkGroupPos == 0 || isMultiWarpValidLayoutForUnbroadcast(
                                     linearLayout, numWorkGroupPos, rewriter);
}

/// Get optimized unbroadcasted tensor type.
///
/// Get optimized ranked tensor type after unbroadcasting. As we only support 1D
/// tensors, this is as simple as getting an "unboradcasted" blocked-encoded 1D
/// tensor type.
RankedTensorType getOptimizedType(RankedTensorType type,
                                  const LinearLayout &linearLayout,
                                  PatternRewriter &rewriter) {
  StringAttr kWarp = rewriter.getStringAttr("warp");

  auto encoding = cast<DistributedEncodingTrait>(type.getEncoding());
  unsigned threadsPerWarp = product(encoding.getThreadsPerWarp());
  unsigned warpsPerCTA = product(encoding.getWarpsPerCTA());
  [[maybe_unused]] unsigned ctaSplitNum = product(encoding.getCTASplitNum());
  assert(ctaSplitNum == 1 && "Expecting single CTA");

  RankedTensorType::Builder builder(type);
  int32_t numWorkGroupPos = linearLayout.getInDimSizeLog2(kWarp);
  unsigned sizePerThread =
      numWorkGroupPos == 0
          ? 1
          : linearLayout.getBasis(kWarp, 0)[0] / threadsPerWarp;
  CTALayoutAttr ctaLayout = CTALayoutAttr::getDefault(rewriter.getContext(), 1);
  auto newEncoding = rewriter.getAttr<BlockedEncodingAttr>(
      sizePerThread, threadsPerWarp, warpsPerCTA, /*order=*/0, ctaLayout);
  builder.setEncoding(newEncoding);
  return builder;
}

struct ElementwiseOptPattern final
    : OpTraitRewritePattern<OpTrait::Elementwise> {
  using OpTraitRewritePattern<OpTrait::Elementwise>::OpTraitRewritePattern;

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const final {
    // Rely on this for a simpler pass.
    if (!op->hasTrait<OpTrait::SameOperandsAndResultType>() ||
        op->getNumResults() != 1)
      return failure();

    // Skip complex operations.
    if (op->hasSuccessors() || op->getNumRegions() != 0)
      return failure();

    // Layout optimizations only apply to tensors.
    auto type = dyn_cast<RankedTensorType>(op->getResultTypes().front());
    if (!type)
      return failure();

    // Check if the layout is actually bad and can be optimized using our
    // approach. We only support 1D tensors for now as these are easier to
    // handle.
    Attribute layout = type.getEncoding();
    if (!layout || type.getRank() != 1)
      return failure();
    std::optional<LinearLayout> linearLayout =
        toLinearLayout(type.getShape(), layout);
    if (!linearLayout || !isValidLayoutForUnbroadcast(*linearLayout, rewriter))
      return failure();

    // Check the operands are not used by other operations. This will prevent
    // register pressure increase:
    if (!llvm::all_of(op->getOperands(),
                      [](Value val) { return val.hasOneUse(); }))
      return failure();

    // As we are dealing with 1D tensors, we can do a simple transform to obtain
    // a more optimized operation.
    Location loc = op->getLoc();
    RankedTensorType newType = getOptimizedType(type, *linearLayout, rewriter);
    SmallVector<Value> newOperands(op->getNumOperands());
    llvm::transform(op->getOperands(), std::begin(newOperands),
                    [&rewriter, loc, newType](Value operand) {
                      return rewriter.create<ConvertLayoutOp>(loc, newType,
                                                              operand);
                    });

    // Now we create the optimized operation:
    StringAttr opName = op->getName().getIdentifier();
    ArrayRef<NamedAttribute> attributes = op->getAttrs();
    Operation *newElementwiseOp =
        rewriter.create(loc, opName, newOperands, newType, attributes);
    assert(newElementwiseOp->getNumResults() == 1 &&
           "Expecting single result operation");

    // Convert to unoptimized encoding for further use.
    Value newValue = newElementwiseOp->getResult(0);
    rewriter.replaceOpWithNewOp<ConvertLayoutOp>(op, type, newValue);

    return success();
  }
};

struct TritonIntelGPUOptimizeElementwiseParallelism final
    : impl::TritonIntelGPUOptimizeElementwiseParallelismBase<
          TritonIntelGPUOptimizeElementwiseParallelism> {
  using Base::Base;

  void runOnOperation() final {
    Operation *op = getOperation();
    MLIRContext *ctx = op->getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<ElementwiseOptPattern>(ctx);
    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
} // namespace mlir::triton::gpu::intel
