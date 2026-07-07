#include <Dialect/TritonIntelGPU/Transforms/Utility.h>

#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace ttg = mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;

namespace mlir::triton::gpu::intel {

#define GEN_PASS_DEF_TRITONINTELGPUREWRITEDESCRIPTORGATHERSCATTER
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"

namespace {

SmallVector<NamedAttribute> filterSegmentSizes(ArrayRef<NamedAttribute> attrs) {
  SmallVector<NamedAttribute> result;
  llvm::copy_if(attrs, std::back_inserter(result), [](const NamedAttribute &a) {
    return a.getName().getValue() != "operandSegmentSizes";
  });
  return result;
}

/// Builds converted x-offsets using source encoding inferred from `value`.
/// This is shared by descriptor gather/scatter rewrites.
Value buildConvertedXOffsets(PatternRewriter &rewriter, Location loc,
                             Value xOffsets, RankedTensorType valueType) {
  Attribute newEncoding = ttg::SliceEncodingAttr::get(
      rewriter.getContext(), 1,
      cast<ttg::DistributedEncodingTrait>(valueType.getEncoding()));
  auto xOffsetsType = cast<RankedTensorType>(xOffsets.getType());
  RankedTensorType newXOffsetsType =
      xOffsetsType.cloneWithEncoding(newEncoding);
  return ConvertLayoutOp::create(rewriter, loc, newXOffsetsType, xOffsets);
}

struct RewriteDescriptorGatherPattern
    : OpRewritePattern<mlir::triton::DescriptorGatherOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::triton::DescriptorGatherOp op,
                                PatternRewriter &rewriter) const override {

    auto resultType = cast<RankedTensorType>(op->getResults()[0].getType());
    Value newIndex = buildConvertedXOffsets(rewriter, op.getLoc(),
                                            op.getXOffsets(), resultType);
    auto newOp = ttgi::DescriptorGatherOp::create(
        rewriter, op.getLoc(), op.getResult().getType(), op.getDesc(), newIndex,
        op.getYOffset());
    newOp->setAttrs(filterSegmentSizes(op->getAttrs()));
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct RewriteDescriptorScatterPattern
    : OpRewritePattern<mlir::triton::DescriptorScatterOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::triton::DescriptorScatterOp op,
                                PatternRewriter &rewriter) const override {
    auto srcType = cast<RankedTensorType>(op.getSrc().getType());
    Value newXOffsets = buildConvertedXOffsets(rewriter, op.getLoc(),
                                               op.getXOffsets(), srcType);
    auto newOp = ttgi::DescriptorScatterOp::create(
        rewriter, op.getLoc(), op.getDesc(), newXOffsets, op.getYOffset(),
        op.getSrc());
    newOp->setAttrs(filterSegmentSizes(op->getAttrs()));
    rewriter.eraseOp(op);
    return success();
  }
};

class TritonIntelGPURewriteDescriptorGatherScatterPass
    : public impl::TritonIntelGPURewriteDescriptorGatherScatterBase<
          TritonIntelGPURewriteDescriptorGatherScatterPass> {
public:
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns
        .add<RewriteDescriptorGatherPattern, RewriteDescriptorScatterPattern>(
            ctx);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::triton::gpu::intel
