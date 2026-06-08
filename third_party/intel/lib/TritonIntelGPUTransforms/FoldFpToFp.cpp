#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.h"
#include "llvm/ADT/APFloat.h"

using namespace mlir;
namespace tt = mlir::triton;

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONINTELGPUFOLDFPTOFP
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

namespace {

// Returns true iff every value of float element type `from` is exactly
// representable in float element type `to` (i.e. `from` is a subset of `to`):
// `to` must have at least the mantissa precision and contain the exponent range
// of `from`. Both args must be FloatType.
static bool isLosslessFpCast(Type from, Type to) {
  auto fromF = dyn_cast<FloatType>(from);
  auto toF = dyn_cast<FloatType>(to);
  if (!fromF || !toF)
    return false;
  const llvm::fltSemantics &fromSem = fromF.getFloatSemantics();
  const llvm::fltSemantics &toSem = toF.getFloatSemantics();
  return llvm::APFloat::semanticsPrecision(fromSem) <=
             llvm::APFloat::semanticsPrecision(toSem) &&
         llvm::APFloat::semanticsMaxExponent(fromSem) <=
             llvm::APFloat::semanticsMaxExponent(toSem) &&
         llvm::APFloat::semanticsMinExponent(fromSem) >=
             llvm::APFloat::semanticsMinExponent(toSem);
}

// Fold `inner: A -> B` feeding `outer: B -> C` into a single `A -> C` when the
// outer cast `B -> C` is lossless (B is a subset of C). This drops a redundant
// narrow intermediate; the result is more accurate than the double cast, so it
// is NOT value-preserving (gated by TRITON_INTEL_FOLD_LOSSY_FPCAST, default
// on).
class FoldNarrowFpToFpIntermediate : public OpRewritePattern<tt::FpToFpOp> {
public:
  using OpRewritePattern<tt::FpToFpOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tt::FpToFpOp outer,
                                PatternRewriter &rewriter) const override {
    auto inner = outer.getSrc().getDefiningOp<tt::FpToFpOp>();
    if (!inner)
      return rewriter.notifyMatchFailure(outer, "src is not an fp_to_fp");

    Type aTy = getElementTypeOrSelf(inner.getSrc().getType());
    Type bTy = getElementTypeOrSelf(inner.getType());
    Type cTy = getElementTypeOrSelf(outer.getType());

    // Only fold when the outer widen B -> C loses nothing.
    if (!isLosslessFpCast(bTy, cTy))
      return rewriter.notifyMatchFailure(outer, "outer cast is not lossless");

    // Determine rounding for the merged A -> C op. A downcast requires a
    // rounding mode (see FpToFpOp::verify); a lossless cast does not need one.
    // getRoundingAttr() returns a null RoundingModeAttr when absent.
    triton::RoundingModeAttr rounding;
    if (!isLosslessFpCast(aTy, cTy)) {
      // A -> C is a downcast: rounding required. Prefer outer's rounding
      // (applied at the final type width) over inner's if both are present.
      if (outer.getRoundingAttr())
        rounding = outer.getRoundingAttr();
      else if (inner.getRoundingAttr())
        rounding = inner.getRoundingAttr();
      else
        return rewriter.notifyMatchFailure(
            outer, "downcast result needs a rounding mode but none present");
    }

    // A null `rounding` selects the no-rounding overload (optional attr).
    rewriter.replaceOpWithNewOp<tt::FpToFpOp>(outer, outer.getType(),
                                              inner.getSrc(), rounding);
    return success();
  }
};

class TritonIntelGPUFoldFpToFpPass
    : public triton::gpu::intel::impl::TritonIntelGPUFoldFpToFpBase<
          TritonIntelGPUFoldFpToFpPass> {
public:
  void runOnOperation() override {
    // Drop redundant narrow fp intermediates (default on; opt out by setting
    // TRITON_INTEL_FOLD_LOSSY_FPCAST to a false value). Changes fp8 numerics.
    bool foldNarrowIntermediate =
        tt::tools::isEnvValueBool(
            tt::tools::getStrEnv("TRITON_INTEL_FOLD_LOSSY_FPCAST"))
            .value_or(true);
    if (!foldNarrowIntermediate)
      return;

    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    RewritePatternSet patterns(context);
    patterns.add<FoldNarrowFpToFpIntermediate>(context);
    if (applyPatternsGreedily(mod, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace
