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
                                          Builder &builder) {
  StringAttr kLane = builder.getStringAttr("lane");
  StringAttr kWarp = builder.getStringAttr("warp");
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
                                 Builder &builder) {
  StringAttr kLane = builder.getStringAttr("lane");
  StringAttr kWarp = builder.getStringAttr("warp");
  StringAttr kBlock = builder.getStringAttr("block");
  StringAttr kDim0 = builder.getStringAttr("dim0");
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
                                     linearLayout, numWorkGroupPos, builder);
}

/// Generic checks for the operation not looking at the tensor type.
bool isCandidateOp(Operation *op) {
  // Rely on this for a simpler pass.
  if (!op->hasTrait<OpTrait::SameOperandsAndResultType>() ||
      op->getNumResults() != 1)
    return false;

  // Skip complex operations.
  if (op->hasSuccessors() || op->getNumRegions() != 0)
    return false;

  return true;
}

bool optimizationDoesNotWorsenRegisterPressure(
    Value value, RankedTensorType newType, SmallPtrSetImpl<Value> &visited) {
  if (!visited.insert(value).second)
    return true;
  // All users must be operations we will optimize too or layout conversions we
  // will introduce later.
  return llvm::all_of(value.getUses(), [&visited, newType](OpOperand &operand) {
    Operation *owner = operand.getOwner();

    // We will be introducing just this operation later.
    if (auto convertLayout = dyn_cast<ConvertLayoutOp>(owner))
      return convertLayout.getResult().getType() == newType;

    // Broadcasted in source.
    if (isa<ExpandDimsOp>(owner))
      return true;

    // Allow for loop optimizations.
    if (isa<scf::YieldOp>(owner))
      return true;

    // Only allow candidates. Check only operation constraints. We do not have
    // to check the type as we did already.
    if (!owner->hasTrait<OpTrait::Elementwise>() || !isCandidateOp(owner))
      return false;

    // Check other operands fit the constraints.
    return llvm::all_of(owner->getOperands(),
                        [&visited, newType](Value operand) {
                          return optimizationDoesNotWorsenRegisterPressure(
                              operand, newType, visited);
                        });
  });
}

/// Get optimized unbroadcasted tensor type.
///
/// Get optimized ranked tensor type after unbroadcasting. As we only support 1D
/// tensors, this is as simple as getting an "unboradcasted" blocked-encoded 1D
/// tensor type.
RankedTensorType getOptimizedType(RankedTensorType type,
                                  const LinearLayout &linearLayout,
                                  Builder &builder) {
  StringAttr kWarp = builder.getStringAttr("warp");

  auto encoding = cast<DistributedEncodingTrait>(type.getEncoding());
  unsigned threadsPerWarp = product(encoding.getThreadsPerWarp());
  unsigned warpsPerCTA = product(encoding.getWarpsPerCTA());
  [[maybe_unused]] unsigned ctaSplitNum = product(encoding.getCTASplitNum());
  assert(ctaSplitNum == 1 && "Expecting single CTA");

  RankedTensorType::Builder typeBuilder(type);
  int32_t numWorkGroupPos = linearLayout.getInDimSizeLog2(kWarp);
  unsigned sizePerThread =
      numWorkGroupPos == 0
          ? 1
          : linearLayout.getBasis(kWarp, 0)[0] / threadsPerWarp;
  CTALayoutAttr ctaLayout = CTALayoutAttr::getDefault(rewriter.getContext(), 1);
  auto newEncoding = rewriter.getAttr<BlockedEncodingAttr>(
      sizePerThread, threadsPerWarp, warpsPerCTA, /*order=*/0, ctaLayout);
  typeBuilder.setEncoding(newEncoding);
  return typeBuilder;
}

bool isCandidateTypeForOptimization(RankedTensorType type) {
  if (!type)
    return false;

  // Check if the layout is actually bad and can be optimized using our
  // approach. We only support 1D tensors for now as these are easier to
  // handle.
  Attribute layout = type.getEncoding();
  if (!layout || type.getRank() != 1)
    return false;
  std::optional<LinearLayout> linearLayout =
      toLinearLayout(type.getShape(), layout);

  LLVM_DEBUG(llvm::dbgs() << "Checking linear layout:\n"
                          << linearLayout << "\n");

  Builder builder(type.getContext());
  if (!linearLayout || !isValidLayoutForUnbroadcast(*linearLayout, builder))
    return false;

  // As we are dealing with 1D tensors, we can do a simple transform to obtain
  // a more optimized operation.
  RankedTensorType newType = getOptimizedType(type, *linearLayout, builder);

  LLVM_DEBUG(llvm::dbgs() << "Would convert to type:\n" << newType << "\n");

  return true;
}

bool canLoopInductionVarBeOptimized(Value initArg, Value regionIterArg,
                                    Value yieldedVal) {
  LLVM_DEBUG(llvm::dbgs() << "Checking loop vars:\n"
                          << initArg << "\n"
                          << regionIterArg << "\n"
                          << yieldedVal << "\n");

  // Check the induction variable is a candidate for this optimization based on
  // its type.
  auto type = dyn_cast<RankedTensorType>(initArg.getType());
  if (!isCandidateTypeForOptimization(type))
    return false;
  assert(type && "Expecting ranked tensor type");

  // We want to check all the variables involve in the optimization can be
  // replaced by ones with a more efficient layout without affecting register
  // pressure.

  LinearLayout linearLayout =
      *toLinearLayout(type.getShape(), type.getEncoding());
  Builder builder(type.getContext());
  RankedTensorType newType = getOptimizedType(type, linearLayout, builder);

  SmallPtrSet<Value, 2> visited;
  // Only allow initArgs with a single use for now.
  return initArg.hasOneUse() &&
         optimizationDoesNotWorsenRegisterPressure(yieldedVal, newType,
                                                   visited) &&
         optimizationDoesNotWorsenRegisterPressure(regionIterArg, newType,
                                                   visited);
}

struct ElementwiseOptPattern final
    : OpTraitRewritePattern<OpTrait::Elementwise> {
  using OpTraitRewritePattern<OpTrait::Elementwise>::OpTraitRewritePattern;

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const final {
    LLVM_DEBUG(llvm::dbgs() << "Checking operation:\n" << *op << "\n");

    // Rely on this for a simpler pass.
    if (!isCandidateOp(op))
      return failure();

    // Check the operation is a candidate for this optimization based on its
    // type.
    auto type = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!isCandidateTypeForOptimization(type))
      return failure();
    assert(type && "Expecting ranked tensor type");

    LinearLayout linearLayout =
        *toLinearLayout(type.getShape(), type.getEncoding());

    // As we are dealing with 1D tensors, we can do a simple transform to obtain
    // a more optimized operation.
    Location loc = op->getLoc();
    RankedTensorType newType = getOptimizedType(type, linearLayout, rewriter);

    // Check the operands are not used by other operations. This will prevent
    // register pressure increase:
    if (SmallPtrSet<Value, 2> visited;
        !llvm::all_of(op->getOperands(), [&visited, newType](Value operand) {
          return optimizationDoesNotWorsenRegisterPressure(operand, newType,
                                                           visited);
        }))
      return failure();

    // Obtain converted operands.
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

    LLVM_DEBUG(llvm::dbgs() << "Conversion took place.\n");

    return success();
  }
};

struct ForOptPattern final : OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp loop,
                                PatternRewriter &rewriter) const final {
    LLVM_DEBUG(llvm::dbgs() << "Checking operation:\n" << loop << "\n");

    // Tuples of <initArg, blockArg, yieldedVal>
    SmallVector<std::tuple<Value, BlockArgument, Value>> toOptimize;
    llvm::copy_if(llvm::zip_equal(loop.getInitArgs(), loop.getRegionIterArgs(),
                                  loop.getYieldedValues()),
                  std::back_inserter(toOptimize), [](auto entry) {
                    auto [initArg, regionIterArg, yieldedVal] = entry;
                    return canLoopInductionVarBeOptimized(
                        initArg, regionIterArg, yieldedVal);
                  });
    if (toOptimize.empty())
      return failure();

    constexpr auto getRealArgNumber = [](BlockArgument blockArg) {
      return blockArg.getArgNumber() - 1;
    };

    LLVM_DEBUG({
      llvm::dbgs() << "Selected for optimization:\n";
      for (auto [initArg, regionIterArg, yieldedVal] : toOptimize)
        llvm::dbgs() << "Value:\n- init_arg: " << initArg
                     << "\n- index: " << getRealArgNumber(regionIterArg)
                     << "\n- yielded_value: " << yieldedVal << "\n";
    });

    Location loc = loop.getLoc();
    Value lowerBound = loop.getLowerBound();
    Value upperBound = loop.getUpperBound();
    Value step = loop.getStep();

    // Convert candidate init args:
    SmallVector<Value> newInitArgs(loop.getInitArgs());
    for (auto [initArg, regionIterArg, yieldedVal] : toOptimize) {
      unsigned index = getRealArgNumber(regionIterArg);
      auto type = cast<RankedTensorType>(initArg.getType());
      LinearLayout linearLayout =
          *toLinearLayout(type.getShape(), type.getEncoding());
      RankedTensorType newType = getOptimizedType(type, linearLayout, rewriter);
      newInitArgs[index] =
          rewriter.create<ConvertLayoutOp>(loc, newType, initArg);
    }

    // Create new for loop.
    // We provide a custom loop body builder that will clone the original body,
    // but adding layout conversions for the optimized block arguments.
    auto loopBodyBuilder = [&](OpBuilder &builder, Location loc,
                               Value inductionVar, ValueRange regionIterArgs) {
      // Add mapping for the cloning.
      // We need to convert the operations back to the unoptimized layout in the
      // loop body.
      SmallVector<Value> argValues{inductionVar};
      llvm::append_range(argValues, regionIterArgs);
      for (auto [initArg, regionIterArg, yieldedVal] : toOptimize) {
        unsigned index = getRealArgNumber(regionIterArg);
        auto type = cast<RankedTensorType>(initArg.getType());
        Value backToOriginalLayout = builder.create<ConvertLayoutOp>(
            loc, regionIterArg.getType(), regionIterArgs[index]);
        argValues[regionIterArg.getArgNumber()] = backToOriginalLayout;
      }

      rewriter.mergeBlocks(&loop.getRegion().front(), builder.getBlock(),
                           argValues);

      // Modify yield operation with updated values.
      auto yieldOp = cast<scf::YieldOp>(builder.getBlock()->getTerminator());
      builder.setInsertionPoint(yieldOp);
      for (auto [initArg, regionIterArg, yieldedVal] : toOptimize) {
        unsigned index = getRealArgNumber(regionIterArg);
        Type type = regionIterArgs[index].getType();
        yieldOp.getResultsMutable()[index].assign(
            builder.create<ConvertLayoutOp>(
                loc, type, yieldOp.getResultsMutable()[index].get()));
      }
    };
    auto newForOp = rewriter.create<scf::ForOp>(
        loc, lowerBound, upperBound, step, newInitArgs, loopBodyBuilder);

    LLVM_DEBUG(llvm::dbgs() << "New loop:\n" << newForOp << "\n");

    // Convert for loop results back to their original types:
    SmallVector<Value> newVals(newForOp.getResults().size());
    llvm::transform(llvm::zip_equal(newForOp.getResults(), loop.getResults()),
                    std::begin(newVals), [&](auto entry) -> Value {
                      auto [newRes, origRes] = entry;
                      if (newRes.getType() == origRes.getType())
                        return newRes;
                      return rewriter.create<ConvertLayoutOp>(
                          loc, origRes.getType(), newRes);
                    });
    rewriter.replaceOp(loop, newVals);
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
    patterns.add<ElementwiseOptPattern, ForOptPattern>(ctx);
    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
} // namespace mlir::triton::gpu::intel
