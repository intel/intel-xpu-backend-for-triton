#include "intel/include/Dialect/Triton/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "triton-intel-reassociate-dot-scale"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = mlir::triton;

namespace mlir::triton::intel {
#define GEN_PASS_DEF_TRITONINTELREASSOCIATEDOTSCALE
#include "intel/include/Dialect/Triton/Transforms/Passes.h.inc"
} // namespace mlir::triton::intel

namespace {

// Carries the unpacked scalar scale and its element type.
struct ScaleInfo {
  Value scalarValue;
  FloatType elemTy;
  bool isConstantSplat = false;
  APFloat constantValue{0.0};
};

struct ReassociateDotScalePattern : public OpRewritePattern<arith::MulFOp> {
public:
  ReassociateDotScalePattern(MLIRContext *context, int benefit = 1)
      : OpRewritePattern<arith::MulFOp>(context, benefit) {}

  LogicalResult matchAndRewrite(arith::MulFOp mulOp,
                                PatternRewriter &rewriter) const override {
    tt::DotOp dotOp;
    Value scaleVal;
    if (failed(findDotAndScale(mulOp, rewriter, dotOp, scaleVal)))
      return failure();

    if (failed(validateDot(dotOp, rewriter)))
      return failure();

    ScaleInfo scale;
    if (failed(extractScaleScalar(mulOp, scaleVal, rewriter, scale)))
      return failure();

    Value target;
    unsigned opIdx;
    if (failed(selectTargetOperand(dotOp, scale, rewriter, target, opIdx)))
      return failure();

    return rewriteWithScaledOperand(mulOp, dotOp, target, opIdx, scale,
                                    rewriter);
  }

private:
  // Step 1: Identify which mulOp operand is the dot result and which is the
  // scale. Fails if neither operand is a DotOp result, or if both are.
  LogicalResult findDotAndScale(arith::MulFOp mulOp, PatternRewriter &rewriter,
                                tt::DotOp &dotOp, Value &scaleVal) const {
    for (unsigned i = 0; i < 2; ++i) {
      if (auto defDotOp = mulOp.getOperand(i).getDefiningOp<tt::DotOp>()) {
        if (mulOp.getOperand(1 - i).getDefiningOp<tt::DotOp>())
          return rewriter.notifyMatchFailure(mulOp,
                                             "both operands are DotOp results");
        dotOp = defDotOp;
        scaleVal = mulOp.getOperand(1 - i);
        return success();
      }
    }
    return rewriter.notifyMatchFailure(mulOp,
                                       "neither operand is a DotOp result");
  }

  // Step 2: Validate the dot: single use and zero accumulator.
  LogicalResult validateDot(tt::DotOp dotOp, PatternRewriter &rewriter) const {
    if (!dotOp->hasOneUse())
      return rewriter.notifyMatchFailure(dotOp, "dot result has multiple uses");
    if (!matchPattern(dotOp.getC(), m_AnyZeroFloat()))
      return rewriter.notifyMatchFailure(dotOp, "accumulator is not zero");
    return success();
  }

  // Step 3: Verify the scale is a uniform float scalar broadcast and unpack it.
  // Accepts tt.splat of a float scalar, or arith.constant dense float splat.
  // Integer dots are intentionally out of scope: the reassociation is
  // algebraically valid for integers, but scaling a narrow DPAS operand
  // (i8/u8/i4) overflows, and widening it defeats the integer DPAS lowering
  // this rewrite exists to preserve. Integer scales also reach the dot via
  // muli/sitofp, not the mulf pattern this rewrite matches.
  LogicalResult extractScaleScalar(arith::MulFOp mulOp, Value scaleVal,
                                   PatternRewriter &rewriter,
                                   ScaleInfo &info) const {
    if (auto splat = scaleVal.getDefiningOp<tt::SplatOp>()) {
      info.scalarValue = splat.getSrc();
      // The float guard also protects the cast<FloatType> below.
      if (!isa<FloatType>(info.scalarValue.getType()))
        return rewriter.notifyMatchFailure(
            mulOp, "splat source is not a float scalar");
      info.elemTy = cast<FloatType>(info.scalarValue.getType());
      return success();
    }

    if (auto constOp = scaleVal.getDefiningOp<arith::ConstantOp>()) {
      auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue());
      if (!denseAttr)
        return rewriter.notifyMatchFailure(mulOp, "constant is not dense");
      if (!denseAttr.isSplat())
        return rewriter.notifyMatchFailure(mulOp, "constant is not a splat");
      auto floatAttr =
          dyn_cast<FloatAttr>(denseAttr.getSplatValue<Attribute>());
      if (!floatAttr)
        return rewriter.notifyMatchFailure(mulOp,
                                           "splat constant is not float-typed");
      info.elemTy = cast<FloatType>(floatAttr.getType());
      info.constantValue = floatAttr.getValue();
      info.isConstantSplat = true;
      return success();
    }

    return rewriter.notifyMatchFailure(
        mulOp, "scale is not a splat or constant splat");
  }

  // Step 4: Pick the loop-invariant operand to absorb the scale.  The scale
  // itself must also be loop-invariant, and the other operand loop-variant
  // (otherwise the whole dot is already hoistable and the rewrite adds no
  // value).
  LogicalResult selectTargetOperand(tt::DotOp dotOp, const ScaleInfo &scale,
                                    PatternRewriter &rewriter, Value &target,
                                    unsigned &opIdx) const {
    auto loop = dotOp->getParentOfType<LoopLikeOpInterface>();
    if (!loop)
      return rewriter.notifyMatchFailure(dotOp, "dot is not inside a loop");

    // Constant splats are always loop-invariant.
    bool scaleInv =
        scale.isConstantSplat || loop.isDefinedOutsideOfLoop(scale.scalarValue);
    if (!scaleInv)
      return rewriter.notifyMatchFailure(dotOp, "scale is not loop-invariant");

    bool aInv = loop.isDefinedOutsideOfLoop(dotOp.getA());
    bool bInv = loop.isDefinedOutsideOfLoop(dotOp.getB());

    if (aInv && bInv)
      return rewriter.notifyMatchFailure(
          dotOp, "both operands invariant (dot already hoistable)");

    if (aInv && !bInv) {
      target = dotOp.getA();
      opIdx = 0;
      LDBG("Targeting operand A for scaling");
      return success();
    }
    if (bInv && !aInv) {
      target = dotOp.getB();
      opIdx = 1;
      LDBG("Targeting operand B for scaling");
      return success();
    }
    return rewriter.notifyMatchFailure(
        dotOp, "both operands variant (no hoisting opportunity)");
  }

  // Step 5: Perform the rewrite. The scale multiply is performed in the wider
  // of the scale and target element types so it is always lossless, then the
  // result is converted back to the target type.
  LogicalResult rewriteWithScaledOperand(arith::MulFOp mulOp, tt::DotOp dotOp,
                                         Value target, unsigned opIdx,
                                         const ScaleInfo &scale,
                                         PatternRewriter &rewriter) const {
    LDBG("Applying reassociation on " << *mulOp);
    Location loc = mulOp.getLoc();

    auto targetTy = cast<RankedTensorType>(target.getType());
    auto targetElemTy = cast<FloatType>(targetTy.getElementType());
    FloatType computeElemTy = scale.elemTy.getIntOrFloatBitWidth() >=
                                      targetElemTy.getIntOrFloatBitWidth()
                                  ? scale.elemTy
                                  : targetElemTy;

    // Materialize the scalar for constant splats.
    Value scalarValue = scale.scalarValue;
    if (scale.isConstantSplat)
      scalarValue = arith::ConstantOp::create(
          rewriter, loc, FloatAttr::get(scale.elemTy, scale.constantValue));

    // Extend the scale scalar to the compute type if it is narrower (lossless).
    // Extending before the splat keeps the conversion on a scalar.
    if (scale.elemTy != computeElemTy)
      scalarValue =
          arith::ExtFOp::create(rewriter, loc, computeElemTy, scalarValue);

    RankedTensorType splatTy = RankedTensorType::get(
        targetTy.getShape(), computeElemTy, targetTy.getEncoding());
    Value scaleSplat = tt::SplatOp::create(rewriter, loc, splatTy, scalarValue);

    // Widen the target operand if the compute type is wider.
    Value targetExt = target;
    if (targetElemTy != computeElemTy)
      targetExt = arith::ExtFOp::create(rewriter, loc, splatTy, target);

    auto fmf = arith::FastMathFlagsAttr::get(rewriter.getContext(),
                                             arith::FastMathFlags::reassoc);
    Value scaledExt =
        arith::MulFOp::create(rewriter, loc, targetExt, scaleSplat, fmf);

    // Truncate the product back to the target type if we widened.
    Value scaled = scaledExt;
    if (targetElemTy != computeElemTy)
      scaled = arith::TruncFOp::create(rewriter, loc, targetTy, scaledExt);

    Value newA = (opIdx == 0) ? scaled : dotOp.getA();
    Value newB = (opIdx == 1) ? scaled : dotOp.getB();
    auto resultTy = cast<RankedTensorType>(dotOp.getResult().getType());
    auto newDot = tt::DotOp::create(rewriter, loc, resultTy, newA, newB,
                                    dotOp.getC(), dotOp.getInputPrecision(),
                                    dotOp.getMaxNumImpreciseAcc());
    LDBG("Created new dot: " << *newDot);

    rewriter.replaceOp(mulOp, newDot.getResult());
    rewriter.eraseOp(dotOp);
    return success();
  }
};

struct ReassociateDotScale
    : public mlir::triton::intel::impl::TritonIntelReassociateDotScaleBase<
          ReassociateDotScale> {
public:
  using Base::Base;

  void runOnOperation() override {
    if (!fastMath) {
      markAllAnalysesPreserved();
      return;
    }

    ModuleOp mod = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<ReassociateDotScalePattern>(&getContext());

    if (failed(applyPatternsGreedily(mod, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
