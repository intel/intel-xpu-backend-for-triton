#include "Dialect/TritonIntelGPU/Transforms/DecomposeScaledBlocked.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/DecomposeScaledBlocked.h"

using namespace mlir;
using namespace mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;

namespace {

// Intel specialization of upstream's `tt.dot_scaled` decomposition. The scale
// handling is upstream's: a native f8E8M0FNU scale is bitcast to i8 and decoded
// on the integer path, and with fastMath off `fma(s, 0, s)` turns the +inf that
// byte 0xFF decodes to into NaN. Only the dot itself differs:
//   - results that already carry a DPAS layout, or no layout, are left alone,
//   - the `tt.dot` is created with TF32 input precision,
//   - the chained-dot-kind attribute is carried over.
class DecomposeScaledBlockedIntel : public ttg::DecomposeScaledBlocked {
public:
  using ttg::DecomposeScaledBlocked::DecomposeScaledBlocked;

  LogicalResult matchAndRewrite(DotScaledOp op,
                                PatternRewriter &rewriter) const override {
    Attribute enc = op.getType().getEncoding();
    if (!enc || isa<ttgi::DpasEncodingAttr>(enc))
      return failure();

    FloatType computeType =
        getComputeType(op.getAElemType(), op.getBElemType(), rewriter);
    auto a =
        cvtDotOperand(rewriter, op, 0, scaleArg(rewriter, op, 0, computeType));
    auto b =
        cvtDotOperand(rewriter, op, 1, scaleArg(rewriter, op, 1, computeType));
    auto dot = DotOp::create(rewriter, op.getLoc(), a, b, op.getC(),
                             InputPrecision::TF32, 0);
    StringRef kindName =
        ttgi::TritonIntelGPUDialect::getChainedDotKindAttrName();
    if (Attribute kind = op->getAttr(kindName))
      dot->setAttr(kindName, kind);

    rewriter.replaceOpWithNewOp<ttg::ConvertLayoutOp>(op, op.getType(), dot);
    return success();
  }
};

} // namespace

namespace mlir::triton::gpu::intel {

void populateDecomposeScaledBlockedPatterns(RewritePatternSet &patterns,
                                            int benefit) {
  patterns.add<DecomposeScaledBlockedIntel>(patterns.getContext(), benefit);
}

} // namespace mlir::triton::gpu::intel
