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

Value convertToCheaperLayout(Location loc, Value val, Attribute newLayout,
                             PatternRewriter &rewriter) {
  assert(newLayout && "Expecting valid layout");
  return TypeSwitch<Operation *, Value>(val.getDefiningOp())
      .Case([loc, newLayout, val, &rewriter](SplatOp splat) {
        // This is a cost <= 0 conversion as:
        // - If the splat is used by other operation, we just don't use all the
        // duplicated elements in our elementwise operation.
        // - If the splat is not used by other operations, we reduce data
        // duplication and possibly even calculation for this data.
        RankedTensorType type =
            RankedTensorType::Builder(splat.getResult().getType())
                .setEncoding(newLayout);
        return rewriter.create<SplatOp>(loc, type, splat.getSrc());
      })
      .Case([](ConvertLayoutOp convertLayout) {
        // This is a cost = 0 conversion as we ensured no other op is using the
        // layout conversion result.
        return convertLayout.getSrc();
      });
}

Value convertToOriginalLayout(Location loc, Value val, Attribute layout,
                              PatternRewriter &rewriter) {
  RankedTensorType type =
      RankedTensorType::Builder(cast<RankedTensorType>(val.getType()))
          .setEncoding(layout);
  return rewriter.create<ConvertLayoutOp>(loc, type, val);
}

class AttributeAcc {
public:
  static AttributeAcc id() { return AttributeAcc(std::nullopt); }
  static AttributeAcc error() { return AttributeAcc(Attribute()); }

  AttributeAcc() = default;
  AttributeAcc(Attribute value) : value(value) {}

  friend bool operator==(AttributeAcc lhs, AttributeAcc rhs) {
    return lhs.value == rhs.value;
  }

  friend bool operator!=(AttributeAcc lhs, AttributeAcc rhs) {
    return !(lhs == rhs);
  }

  friend AttributeAcc operator+(AttributeAcc lhs, AttributeAcc rhs) {
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

  Attribute operator*() const {
    assert(*this != id() && *this != error() && "Expecting valid layout");
    return *value;
  }

private:
  AttributeAcc(std::optional<Attribute> value) : value(value) {}

  std::optional<Attribute> value;
};

AttributeAcc getCheapLayoutToConvertTo(Value value) {
  Operation *op = value.getDefiningOp();
  if (!op)
    return AttributeAcc::error();
  return TypeSwitch<Operation *, AttributeAcc>(op)
      .Case([](SplatOp splat) {
        // Do not support tensor splats, just scalar splats.
        return isa<RankedTensorType>(splat.getSrc().getType())
                   ? AttributeAcc::error()
                   : AttributeAcc::id();
      })
      .Case([](ConvertLayoutOp convertLayout) -> AttributeAcc {
        // If the layout conversion has more than one user, this may worsen
        // register pressure, as data would need to coexist in both layouts at
        // the same time in registers.
        // TODO: Extend with heuristics to check this is cheap to do.
        if (!convertLayout->hasOneUse())
          return AttributeAcc::error();
        return convertLayout.getSrc().getType().getEncoding();
      })
      .Default(AttributeAcc::error());
}

AttributeAcc accumulateCheapLayoutToConvertTo(AttributeAcc acc, Value val) {
  return acc + getCheapLayoutToConvertTo(val);
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

    // Check if we can convert the operands to a common optimal layout.
    AttributeAcc layoutAcc =
        std::accumulate(op->operand_begin(), op->operand_end(),
                        AttributeAcc::id(), accumulateCheapLayoutToConvertTo);
    if (layoutAcc == AttributeAcc::error() || layoutAcc == AttributeAcc::id())
      return failure();

    // Check the new layout is good for elementwise operations.
    // TODO: Provide heuristics to check it's *better* than the original one
    // instead.
    Attribute newLayout = *layoutAcc;
    assert(newLayout && "Expecting valid layout");
    if (isBadLayoutForElementwise(newLayout))
      return failure();

    // Replace operation with new operation taking operands with a more optimal
    // layout.
    Location loc = op->getLoc();
    StringAttr opName = op->getName().getIdentifier();
    SmallVector<Value> newOperands(op->getNumOperands());
    llvm::transform(op->getOperands(), std::begin(newOperands),
                    [loc, newLayout, &rewriter](Value val) {
                      return convertToCheaperLayout(loc, val, newLayout,
                                                    rewriter);
                    });
    Type newType = newOperands.front().getType();
    ArrayRef<NamedAttribute> attributes = op->getAttrs();
    Operation *newElementwiseOp =
        rewriter.create(loc, opName, newOperands, newType, attributes);
    assert(newElementwiseOp->getNumResults() == 1 &&
           "Expecting single result operation");

    // Convert the result back to the original layout for type consistency.
    Value newOp = convertToOriginalLayout(loc, newElementwiseOp->getResult(0),
                                          layout, rewriter);

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
