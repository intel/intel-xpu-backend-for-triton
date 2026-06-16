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

struct ReassociateDotScalePattern : public OpRewritePattern<arith::MulFOp> {
public:
  ReassociateDotScalePattern(MLIRContext *context, int benefit = 1)
      : OpRewritePattern<arith::MulFOp>(context, benefit) {}

  LogicalResult matchAndRewrite(arith::MulFOp mulOp,
                                PatternRewriter &rewriter) const override {
    Location loc = mulOp.getLoc();

    // Step 1: Identify dot operand vs scale operand
    Value dotOperand, scaleVal;
    tt::DotOp dotOp;
    for (unsigned i = 0; i < 2; ++i) {
      Value operand = mulOp.getOperand(i);
      if (auto defDotOp = operand.getDefiningOp<tt::DotOp>()) {
        dotOp = defDotOp;
        dotOperand = operand;
        scaleVal = mulOp.getOperand(1 - i);
        break;
      }
    }

    if (!dotOp)
      return rewriter.notifyMatchFailure(mulOp,
                                         "neither operand is a DotOp result");

    // Short-circuit: if both operands are dots (unlikely but possible), fail
    if (mulOp.getOperand(0).getDefiningOp<tt::DotOp>() &&
        mulOp.getOperand(1).getDefiningOp<tt::DotOp>())
      return rewriter.notifyMatchFailure(mulOp,
                                         "both operands are DotOp results");

    // Step 2: Dot result must have exactly one use
    if (!dotOp->hasOneUse())
      return rewriter.notifyMatchFailure(dotOp, "dot result has multiple uses");

    // Step 3: Accumulator must be zero
    if (!matchPattern(dotOp.getC(), m_AnyZeroFloat()))
      return rewriter.notifyMatchFailure(dotOp, "accumulator is not zero");

    // Step 4: Scale must be a uniform scalar broadcast
    Value scalarValue;
    FloatType scaleElemTy;
    bool isConstantSplat = false;
    APFloat constantSplatValue(0.0);

    if (auto splat = scaleVal.getDefiningOp<tt::SplatOp>()) {
      scalarValue = splat.getSrc();
      if (!isa<FloatType>(scalarValue.getType()))
        return rewriter.notifyMatchFailure(
            mulOp, "splat source is not a float scalar");
      scaleElemTy = cast<FloatType>(scalarValue.getType());
    } else if (auto constOp = scaleVal.getDefiningOp<arith::ConstantOp>()) {
      if (auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue())) {
        if (!denseAttr.isSplat())
          return rewriter.notifyMatchFailure(mulOp, "constant is not a splat");
        auto splatAttr = denseAttr.getSplatValue<Attribute>();
        if (auto floatAttr = dyn_cast<FloatAttr>(splatAttr)) {
          constantSplatValue = floatAttr.getValue();
          scaleElemTy = cast<FloatType>(floatAttr.getType());
          isConstantSplat = true;
        } else {
          return rewriter.notifyMatchFailure(
              mulOp, "splat constant is not float-typed");
        }
      } else {
        return rewriter.notifyMatchFailure(mulOp, "constant is not dense");
      }
    } else {
      return rewriter.notifyMatchFailure(
          mulOp, "scale is not a splat or constant splat");
    }

    // Step 5: Heuristic (LICM-opportunity test)
    auto loop = dotOp->getParentOfType<LoopLikeOpInterface>();
    if (!loop)
      return rewriter.notifyMatchFailure(dotOp, "dot is not inside a loop");

    bool aInv = loop.isDefinedOutsideOfLoop(dotOp.getA());
    bool bInv = loop.isDefinedOutsideOfLoop(dotOp.getB());
    // For the scale, check the underlying scalar if splat, otherwise the splat
    // constant is invariant
    bool scaleInv = isConstantSplat || loop.isDefinedOutsideOfLoop(scalarValue);

    if (!scaleInv)
      return rewriter.notifyMatchFailure(mulOp, "scale is not loop-invariant");

    if (aInv && bInv)
      return rewriter.notifyMatchFailure(
          dotOp, "both operands invariant (dot already hoistable)");

    Value target;
    unsigned opIdx;
    if (aInv && !bInv) {
      target = dotOp.getA();
      opIdx = 0;
      LDBG("Targeting operand A for scaling");
    } else if (bInv && !aInv) {
      target = dotOp.getB();
      opIdx = 1;
      LDBG("Targeting operand B for scaling");
    } else {
      return rewriter.notifyMatchFailure(
          dotOp, "both operands variant (no hoisting opportunity)");
    }

    // Step 6: Compute-type safety
    auto targetTy = cast<RankedTensorType>(target.getType());
    auto targetElemTy = cast<FloatType>(targetTy.getElementType());
    if (scaleElemTy.getIntOrFloatBitWidth() <
        targetElemTy.getIntOrFloatBitWidth())
      return rewriter.notifyMatchFailure(
          mulOp, "scale element type is narrower than target");

    // REWRITE
    LDBG("Applying reassociation on " << *mulOp);

    // Materialize scalar if needed
    if (isConstantSplat) {
      scalarValue = arith::ConstantOp::create(
          rewriter, loc, FloatAttr::get(scaleElemTy, constantSplatValue));
    }

    // Build splat to target shape in compute type
    RankedTensorType splatTy = RankedTensorType::get(
        targetTy.getShape(), scaleElemTy, targetTy.getEncoding());
    Value scaleSplat = tt::SplatOp::create(rewriter, loc, splatTy, scalarValue);

    // Widen target if needed
    Value targetExt = target;
    if (targetElemTy != scaleElemTy) {
      targetExt = arith::ExtFOp::create(rewriter, loc, splatTy, target);
    }

    // Multiply with reassoc fastmath flag
    auto fmf = arith::FastMathFlagsAttr::get(rewriter.getContext(),
                                             arith::FastMathFlags::reassoc);
    Value scaledExt =
        arith::MulFOp::create(rewriter, loc, targetExt, scaleSplat, fmf);

    // Truncate back if widened
    Value scaled = scaledExt;
    if (targetElemTy != scaleElemTy) {
      scaled = arith::TruncFOp::create(rewriter, loc, targetTy, scaledExt);
    }

    // Build new dot in original operand positions
    Value newA = (opIdx == 0) ? scaled : dotOp.getA();
    Value newB = (opIdx == 1) ? scaled : dotOp.getB();
    RankedTensorType resultTy =
        cast<RankedTensorType>(dotOp.getResult().getType());

    auto newDot = tt::DotOp::create(rewriter, loc, resultTy, newA, newB,
                                    dotOp.getC(), dotOp.getInputPrecision(),
                                    dotOp.getMaxNumImpreciseAcc());

    LDBG("Created new dot: " << *newDot);

    // Replace mulOp with new dot and erase the now-dead old dot
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
