#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include <memory>

namespace {

using namespace mlir;
using namespace triton;
using namespace triton::gpu;

// Given
//   convert(trans(convert(src) #shared)) #dot_operand,
// change the encoding of the inner convert to a special, swizzled shared
// encoding.
class SwizzleShmemConvert : public OpRewritePattern<ConvertLayoutOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvertLayoutOp outerCvt,
                                PatternRewriter &rewriter) const override {
    // Match outerCvt(trans(innerCvt(x))).
    auto trans = outerCvt.getSrc().getDefiningOp<TransOp>();
    if (!trans || trans.getOrder() != ArrayRef<int32_t>{1, 0})
      return failure();
    auto innerCvt = trans.getSrc().getDefiningOp<ConvertLayoutOp>();
    if (!innerCvt)
      return failure();

    auto srcTy = innerCvt.getSrc().getType().cast<RankedTensorType>();
    auto innerCvtTy = innerCvt.getType().cast<RankedTensorType>();
    auto outerCvtTy = outerCvt.getType().cast<RankedTensorType>();

    auto innerCvtEnc = innerCvtTy.getEncoding().dyn_cast<SharedEncodingAttr>();
    auto outerCvtEnc =
        outerCvtTy.getEncoding().dyn_cast<DotOperandEncodingAttr>();
    if (!innerCvtEnc || !outerCvtEnc)
      return failure();

    // TODO(Qingyi): need to check whether the CTALayout of innerCvtEnc should
    // be used here. For tests where numCTAs = 1, this is not a problem since
    // all CTALayouts are the same.
    //
    // Set needTrans to true here. newInnerCvtEnc is computed based on
    // argEncoding which is before the transpose. Without needTrans we will
    // compute vec and maxPhase based on incorrect m, n and k size of mma. The
    // type inference of TransOp simply swap the order but doesn't fix the vec
    // and maxPhase for the YType, hence it would causing incorrect swizzling
    // code.
    auto newInnerCvtEnc = SharedEncodingAttr::get(
        getContext(), outerCvtEnc, innerCvtTy.getShape(),
        /*order=*/getOrder(srcTy.getEncoding()), innerCvtEnc.getCTALayout(),
        innerCvtTy.getElementType(), /*needTrans=*/true);
    if (newInnerCvtEnc == innerCvtEnc)
      return failure();

    auto newInnerCvt = rewriter.create<ConvertLayoutOp>(
        innerCvt.getLoc(),
        RankedTensorType::get(innerCvtTy.getShape(),
                              innerCvtTy.getElementType(), newInnerCvtEnc),
        innerCvt.getSrc());
    auto newTrans = rewriter.create<TransOp>(trans.getLoc(), newInnerCvt,
                                             ArrayRef<int32_t>({1, 0}));
    rewriter.replaceOpWithNewOp<ConvertLayoutOp>(outerCvt, outerCvtTy,
                                                 newTrans);
    return success();
  }
};

// Move convert-to-dot-operand "up" past elementwise ops:
//
//  convert(elementwise(x)) #dot_operand ->
//  elementwise(convert(x, #dot_operand)).
//
// The goal is to put the convert right next to the originating load.  If we can
// accomplish this, then we can save a shmem round-trip:
//
//   Before:
//
//     - Load from global into shmem using an async copy.
//     - Load from shmem into a #blocked layout.
//     - Do elementwise ops over #blocked layout.
//     - Convert to #dot_operand (round-trip through shmem).
//     - Do dot.
//
//   After:
//
//     - Load from global into shmem using an async copy (same as before).
//     - Load from shmem into a #dot_operand layout.
//     - Do elementwise ops over #dot_operand layout.
//     - Do dot.
//
// Eliminating the shmem round-trip is such a big win, we're willing to do it
// even if this duplicates work because some of the elementwise ops have uses
// that don't flow into the dot.  On the other hand, we only want to do this if
// we can in fact reduce shmem round-trips: For example, simply moving a convert
// up above e.g. an `add` now means we have *two* converts.  That's worse,
// unless we can continue moving the converts upwards and eventually merge them.
// So we try to check that this will be beneficial before making any changes.
class HoistLayoutConversion : public OpRewritePattern<ConvertLayoutOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvertLayoutOp cvt,
                                PatternRewriter &rewriter) const override {
    // Only consider conversions to dot operand.
    auto cvtTy = cvt.getType().cast<RankedTensorType>();
    if (!cvtTy.getEncoding().isa<DotOperandEncodingAttr>())
      return failure();

    auto src = cvt.getSrc().getDefiningOp();
    if (!src || src->getNumOperands() == 0 || src->getNumResults() != 1)
      return failure();

    auto srcTy = src->getResult(0).getType().dyn_cast<RankedTensorType>();
    if (!srcTy)
      return failure();

    if (!all_of(src->getOperandTypes(),
                [](Type ty) { return ty.isa<RankedTensorType>(); }))
      return failure();

    // Only consider custom conversions or arith ops.
    // TODO(jlebar): Is this too restrictive?
    if (!isa<FpToFpOp, BitcastOp>(src) &&
        src->getDialect()->getTypeID() != TypeID::get<arith::ArithDialect>())
      return failure();

    // Currently, these instructions are not supported during lowering of
    // shared -> dot_operand layout. Not all types and type conversions are
    // supported.
    if (isa<arith::TruncIOp, arith::TruncFOp, arith::SelectOp>(src))
      return failure();

    // Check that the conversion is transitively dependent on a load, and all
    // operations between the load and the conversion are layout preserving.
    //
    // TODO(jlebar): This is accidentally quadratic; we iterate over the whole
    // slice but then at the end we only modify one op!
    SetVector<Operation *> slice;
    BackwardSliceOptions opt;
    opt.omitBlockArguments = true;
    // TODO(jlebar): Is this filter redundant with omitBlockArguments == true?
    // That is, is it possible to get into a different region without going
    // through a block argument?
    opt.filter = [&](Operation *op) {
      return op->getParentRegion() == cvt->getParentRegion();
    };
    getBackwardSlice(cvt.getOperation(), &slice, opt);

    // TODO(jlebar): This is too conservative when there are multiple loads in
    // the chain (e.g. cvt(load(x) + load(y))).  The intent is to check that all
    // of the ops between the loads and the convert are elementwise.  But
    // actually we set foundLoad = true once we see the first load, and so we
    // will reject the chain if the *second* load we encounter uses a
    // non-elementwise op to calculate its pointers.
    bool foundLoad = false;
    for (Operation *currOp : slice) {
      if (isa<LoadOp>(currOp)) {
        foundLoad = true;
      } else if (foundLoad) {
        // Bail out if there exists an op after Load that is not FpToFp,
        // Bitcast, or Arith.
        if (!isa<FpToFpOp, BitcastOp>(currOp) &&
            currOp->getDialect()->getTypeID() !=
                TypeID::get<arith::ArithDialect>())
          return failure();
      }
    }
    if (!foundLoad)
      return failure();

    SmallVector<ConvertLayoutOp> newOperands;
    for (auto operand : src->getOperands()) {
      // We checked earlier that all operands are ranked tensors.
      auto operandTy = operand.getType().cast<RankedTensorType>();
      Type newCvtTy = RankedTensorType::get(
          srcTy.getShape(), operandTy.getElementType(), cvtTy.getEncoding());
      newOperands.push_back(
          rewriter.create<ConvertLayoutOp>(cvt.getLoc(), newCvtTy, operand));
    }
    auto newRet = rewriter.clone(*src);
    for (int i = 0; i < newOperands.size(); i++)
      newRet->setOperand(i, newOperands[i]);
    newRet->getResult(0).setType(RankedTensorType::get(
        srcTy.getShape(), srcTy.getElementType(), cvtTy.getEncoding()));

    rewriter.replaceOp(cvt, newRet->getResults());
    return success();
  }
};

// Rewrite
//
//   dot(convert(trans(convert(src) #shared)) #shared1) ->
//   dot(trans(convert(src) #shared2))
//
// if dot is an MMAv3 (because MMAv3 allows us to fold transposes).
class FuseTransHopper : public OpRewritePattern<ConvertLayoutOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvertLayoutOp outerCvt,
                                PatternRewriter &rewriter) const override {
    if (!outerCvt->hasOneUse() ||
        !isa<DotOp, nvidia_gpu::DotAsyncOp>(*outerCvt->getUsers().begin()))
      return failure();

    auto dot = *outerCvt->getUsers().begin();
    auto dotEnc = dot->getResult(0)
                      .getType()
                      .cast<RankedTensorType>()
                      .getEncoding()
                      .dyn_cast<NvidiaMmaEncodingAttr>();
    if (!dotEnc || dotEnc.getVersionMajor() != 3)
      return failure();

    // Match outerCvt(trans(innerCvt(x))).
    auto trans = outerCvt.getSrc().getDefiningOp<TransOp>();
    if (!trans || trans.getOrder() != ArrayRef<int32_t>({1, 0}))
      return failure();
    auto innerCvt = trans.getSrc().getDefiningOp<ConvertLayoutOp>();
    if (!innerCvt)
      return failure();

    RankedTensorType srcTy = innerCvt.getSrc().getType();
    RankedTensorType innerCvtTy = innerCvt.getType();
    RankedTensorType outerCvtTy = outerCvt.getType();

    auto innerCvtEnc = innerCvtTy.getEncoding().dyn_cast<SharedEncodingAttr>();
    auto outerCvtEnc = outerCvtTy.getEncoding().dyn_cast<SharedEncodingAttr>();
    if (!innerCvtEnc || !outerCvtEnc)
      return failure();

    // MMAv3 with transpose only supports f16 and bf16.  Fall back to MMAv3
    // without transpose for other data types.
    auto newInnerCvtOrder = getOrder(srcTy.getEncoding());
    auto srcElemTy = innerCvtTy.getElementType();
    if (!srcElemTy.isF16() && !srcElemTy.isBF16()) {
      if (outerCvt.getResult() == dot->getOperand(0)) {
        newInnerCvtOrder = {0, 1};
      } else if (outerCvt.getResult() == dot->getOperand(1)) {
        newInnerCvtOrder = {1, 0};
      }
    }

    // TODO(Qingyi): need to check whether the CTALayout of innerCvtEnc should
    // be used here. For tests where numCTAs = 1, this is not a problem since
    // all CTALayouts are the same.
    auto newInnerCvtEnc = SharedEncodingAttr::get(
        getContext(), innerCvtTy.getShape(), newInnerCvtOrder,
        innerCvtEnc.getCTALayout(), innerCvtTy.getElementType());
    auto newInnerCvtTy = RankedTensorType::get(
        innerCvtTy.getShape(), innerCvtTy.getElementType(), newInnerCvtEnc);

    auto newInnerCvt = rewriter.create<ConvertLayoutOp>(
        innerCvt.getLoc(), newInnerCvtTy, innerCvt.getSrc());
    rewriter.replaceOpWithNewOp<TransOp>(outerCvt, newInnerCvt,
                                         ArrayRef<int32_t>({1, 0}));
    return success();
  }
};

// Rewrite
//   dot(convert(lhs #mma) #shared, rhs) #mma ->
//   dot(convert(lhs #mma) #dot_operand, rhs) #mma,
// for fp16 or bf16 MMAv3 dots.
struct MMAV3UseRegOperand : public OpRewritePattern<DotOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DotOp dotOp,
                                PatternRewriter &rewriter) const override {
    auto convertLhs = dotOp.getOperand(0).getDefiningOp<ConvertLayoutOp>();
    if (!convertLhs)
      return failure();

    auto getEncoding = [](Value v) {
      return v.getType().cast<RankedTensorType>().getEncoding();
    };

    if (!getEncoding(dotOp.getOperand(0)).isa<SharedEncodingAttr>())
      return failure();
    auto srcEnc =
        getEncoding(convertLhs.getSrc()).dyn_cast<NvidiaMmaEncodingAttr>();
    auto dstEnc =
        getEncoding(dotOp.getResult()).dyn_cast<NvidiaMmaEncodingAttr>();
    if (!srcEnc || srcEnc.getVersionMajor() != 3 || !dstEnc ||
        dstEnc.getVersionMajor() != 3)
      return failure();

    // We currently only support convert from f16 and bf16 mma to f16 and bf16
    // dot operand, as the other types require shuffling data across threads.
    // TODO: extend it to more types.
    auto srcTy = convertLhs.getSrc().getType().cast<RankedTensorType>();
    if (!(srcTy.getElementType().isF16() || srcTy.getElementType().isBF16()))
      return failure();

    auto dotOperandEnc = DotOperandEncodingAttr::get(
        dotOp.getContext(), /*opIdx=*/0, srcEnc, /*kWidth=*/0);
    auto newTy = RankedTensorType::get(srcTy.getShape(), srcTy.getElementType(),
                                       dotOperandEnc);
    Value newOperand = rewriter.create<ConvertLayoutOp>(dotOp.getLoc(), newTy,
                                                        convertLhs.getSrc());
    rewriter.modifyOpInPlace(dotOp, [&]() { dotOp.setOperand(0, newOperand); });
    return success();
  }
};

} // namespace

// TODO(jlebar): These autogenerated headers (and the passes they declare)
// should be included within a namespace, but we have to do it consistently.
#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class TritonGPUOptimizeDotOperandsPass
    : public TritonGPUOptimizeDotOperandsBase<
          TritonGPUOptimizeDotOperandsPass> {
public:
  TritonGPUOptimizeDotOperandsPass() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::PassManager pm(m.getContext());
    pm.addPass(mlir::createCanonicalizerPass());
    auto ret = pm.run(m);

    mlir::RewritePatternSet patterns(context);
    patterns.add<SwizzleShmemConvert>(context);
    if (triton::gpu::TritonGPUDialect::getComputeCapability(m) >= 80)
      patterns.add<HoistLayoutConversion>(context);
    patterns.add<FuseTransHopper>(context);
    patterns.add<MMAV3UseRegOperand>(context);
    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};

std::unique_ptr<Pass> mlir::triton::gpu::createOptimizeDotOperandsPass() {
  return std::make_unique<TritonGPUOptimizeDotOperandsPass>();
}
