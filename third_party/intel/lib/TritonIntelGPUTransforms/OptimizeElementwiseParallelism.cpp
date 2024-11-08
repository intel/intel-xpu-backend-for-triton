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
#include "llvm/ADT/TypeSwitch.h"

#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#define DEBUG_TYPE "tritonintelgpu-optimize-elementwise-parallelism"

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONINTELGPUOPTIMIZEELEMENTWISEPARALLELISM
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"

namespace {
bool isBadLayoutForElementwise(Attribute layout) {
  // We only support 'triton_gpu.slice' for now.
  auto slicedLayout = dyn_cast<SliceEncodingAttr>(layout);
  if (!slicedLayout)
    return false;

  // Check the parent layout is squeezed across a dimension with more than one
  // warp per CTA or thread per warp, i.e., there is data duplication across
  // threads along that dimension.
  unsigned dim = slicedLayout.getDim();
  auto parentLayout = cast<DistributedEncodingTrait>(slicedLayout.getParent());
  return parentLayout.getWarpsPerCTA()[dim] != 1 ||
         parentLayout.getThreadsPerWarp()[dim] != 1;
}

Value convertToCheaperLayout(Location loc, Value val, RankedTensorType type,
                             PatternRewriter &rewriter) {
  return TypeSwitch<Operation *, Value>(val.getDefiningOp())
      .Case([loc, type, val, &rewriter](SplatOp splat) {
        // This is a cost <= 0 conversion as:
        // - If the splat is used by other operation, we just don't use all the
        // duplicated elements in our elementwise operation.
        // - If the splat is not used by other operations, we reduce data
        // duplication and possibly even calculation for this data.
        return rewriter.create<SplatOp>(loc, type, splat.getSrc());
      })
      .Case([](ConvertLayoutOp convertLayout) {
        // This is a cost = 0 conversion as we ensured no other op is using the
        // layout conversion result.
        return convertLayout.getSrc();
      })
      .Case([](ReshapeOp reshape) {
        // We only allow `tt.reshape` ops with a `triton_gpu.convert_layout`
        // `src`.
        auto convertLayout = reshape.getSrc().getDefiningOp<ConvertLayoutOp>();
        assert(convertLayout && "Expecting convert layout src");
        return convertLayout.getSrc();
      });
}

/// Class encoding source types of convert_layout operations or
/// convert_layout-reshape operation chains.
///
/// `reshape(convert_layout(src))` chains are common patterns arising from
/// `-intel-triton-optimize-reduction-locality`, so it is worth matching
/// them.
class ConvertToOriginalAcc {
public:
  static ConvertToOriginalAcc id() {
    return ConvertToOriginalAcc(std::nullopt, std::nullopt);
  }

  static ConvertToOriginalAcc error() {
    return ConvertToOriginalAcc(RankedTensorType(), RankedTensorType());
  }

  ConvertToOriginalAcc() = default;
  explicit ConvertToOriginalAcc(ConvertLayoutOp convertLayout)
      : convertLayoutSrcType(convertLayout.getSrc().getType()) {}
  explicit ConvertToOriginalAcc(ReshapeOp reshape)
      : reshapeSrcType(reshape.getSrc().getType()) {}

  friend bool operator==(ConvertToOriginalAcc lhs, ConvertToOriginalAcc rhs) {
    return lhs.convertLayoutSrcType == rhs.convertLayoutSrcType &&
           lhs.reshapeSrcType == rhs.reshapeSrcType;
  }

  friend bool operator!=(ConvertToOriginalAcc lhs, ConvertToOriginalAcc rhs) {
    return !(lhs == rhs);
  }

  /// ConvertToOriginalAcc accumulator checking lhs and rhs are one of the
  /// identities or encode the same types.
  friend ConvertToOriginalAcc operator+(const ConvertToOriginalAcc &lhs,
                                        const ConvertToOriginalAcc &rhs) {
    if (lhs == error() || rhs == error())
      return error();
    if (lhs == id())
      return rhs;
    if (rhs == id())
      return lhs;
    if (lhs != rhs)
      return error();
    return lhs;
  }

  /// ConvertToOriginalAcc merger under certain conditions.
  friend ConvertToOriginalAcc operator*(const ConvertToOriginalAcc &lhs,
                                        const ConvertToOriginalAcc &rhs) {
    if (lhs == error() || rhs == error())
      return error();
    if (lhs == id())
      return rhs;
    if (rhs == id())
      return lhs;
    std::optional<RankedTensorType> convertLayoutSrcType =
        merge(lhs.convertLayoutSrcType, rhs.convertLayoutSrcType);
    if (convertLayoutSrcType && !*convertLayoutSrcType)
      return error();
    std::optional<RankedTensorType> reshapeSrcType =
        merge(lhs.reshapeSrcType, rhs.reshapeSrcType);
    if (reshapeSrcType && !*reshapeSrcType)
      return error();
    return {convertLayoutSrcType, reshapeSrcType};
  }

  RankedTensorType getConvertLayoutSrcType() const {
    return *convertLayoutSrcType;
  }

  std::optional<RankedTensorType> getReshapeSrcType() const {
    return reshapeSrcType;
  }

private:
  ConvertToOriginalAcc(std::optional<RankedTensorType> convertLayoutSrcType,
                       std::optional<RankedTensorType> reshapeSrcType)
      : convertLayoutSrcType(convertLayoutSrcType),
        reshapeSrcType(reshapeSrcType) {}

  static std::optional<RankedTensorType>
  merge(std::optional<RankedTensorType> lhs,
        std::optional<RankedTensorType> rhs) {
    if (!lhs)
      return rhs;
    if (!rhs)
      return lhs;
    if (lhs != rhs)
      return RankedTensorType{};
    return lhs;
  }

  std::optional<RankedTensorType> convertLayoutSrcType;
  std::optional<RankedTensorType> reshapeSrcType;
};

Value convertToOriginalLayout(Location loc, Value val,
                              RankedTensorType originalType,
                              const ConvertToOriginalAcc &convertRecipe,
                              PatternRewriter &rewriter) {
  if (std::optional<RankedTensorType> type =
          convertRecipe.getReshapeSrcType()) {
    // If we read a reshape operation before, we need to perform the same
    // conversion again.
    val = rewriter.create<ConvertLayoutOp>(loc, *type, val);
    return rewriter.create<ReshapeOp>(loc, originalType, val,
                                      /*allow_reorder=*/true,
                                      /*efficient_layout=*/true);
  }
  return rewriter.create<ConvertLayoutOp>(loc, originalType, val);
}

ConvertToOriginalAcc getConvertToOriginalFromVal(Value val) {
  Operation *definingOp = val.getDefiningOp();
  if (!definingOp)
    return ConvertToOriginalAcc::error();
  return TypeSwitch<Operation *, ConvertToOriginalAcc>(definingOp)
      .Case([](SplatOp) { return ConvertToOriginalAcc::id(); })
      .Case([](ConvertLayoutOp convertLayout) {
        if (!convertLayout->hasOneUse())
          return ConvertToOriginalAcc::error();
        return ConvertToOriginalAcc(convertLayout);
      })
      .Case([](ReshapeOp reshape) {
        if (!reshape->hasOneUse())
          return ConvertToOriginalAcc::error();
        auto convertLayout = reshape.getSrc().getDefiningOp<ConvertLayoutOp>();
        if (!convertLayout || !convertLayout->hasOneUse())
          return ConvertToOriginalAcc::error();
        return ConvertToOriginalAcc(reshape) *
               ConvertToOriginalAcc(convertLayout);
      })
      .Default(ConvertToOriginalAcc::error());
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

    // Layout optimizations only apply to tensors.
    auto type = dyn_cast<RankedTensorType>(op->getResultTypes().front());
    if (!type)
      return failure();

    // Skip complex operations.
    if (op->hasSuccessors() || op->getNumRegions() != 0)
      return failure();

    // Check if the layout is actually bad.
    Attribute layout = type.getEncoding();
    if (!layout || !isBadLayoutForElementwise(layout))
      return failure();

    // Check if we can convert the operands to a common optimal layout while
    // getting the "recipe" to convert back to the original tensor type.
    ConvertToOriginalAcc convertRecipe = std::transform_reduce(
        op->operand_begin(), op->operand_end(), ConvertToOriginalAcc::id(),
        std::plus<>{}, getConvertToOriginalFromVal);
    if (convertRecipe == ConvertToOriginalAcc::error() ||
        convertRecipe == ConvertToOriginalAcc::id())
      return failure();

    // Check the new layout is good for elementwise operations.
    // TODO: Provide heuristics to check it's *better* than the original one
    // instead.
    RankedTensorType newType = convertRecipe.getConvertLayoutSrcType();
    if (isBadLayoutForElementwise(newType.getEncoding()))
      return failure();

    // Replace operation with new operation taking operands with a more optimal
    // layout.
    Location loc = op->getLoc();
    StringAttr opName = op->getName().getIdentifier();
    SmallVector<Value> newOperands(op->getNumOperands());
    llvm::transform(op->getOperands(), std::begin(newOperands),
                    [loc, newType, &rewriter](Value val) {
                      return convertToCheaperLayout(loc, val, newType,
                                                    rewriter);
                    });
    ArrayRef<NamedAttribute> attributes = op->getAttrs();
    Operation *newElementwiseOp =
        rewriter.create(loc, opName, newOperands, newType, attributes);
    assert(newElementwiseOp->getNumResults() == 1 &&
           "Expecting single result operation");

    LLVM_DEBUG({
      for (OpOperand &operand : newElementwiseOp->getOpOperands()) {
        llvm::dbgs() << "Converted operand #" << operand.getOperandNumber()
                     << ":\n"
                     << operand.get() << "\n";
      }
      llvm::dbgs() << "Optimized elementwise operation created:\n"
                   << newElementwiseOp->getResult(0) << "\n";
    });

    // Convert the result back to the original layout for type consistency.
    // Check if we can convert the operands to a common optimal layout.
    Value newOp = convertToOriginalLayout(loc, newElementwiseOp->getResult(0),
                                          type, convertRecipe, rewriter);

    rewriter.replaceOp(op, newOp);
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
