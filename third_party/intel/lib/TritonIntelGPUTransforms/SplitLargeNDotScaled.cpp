//===- SplitLargeNDotScaled.cpp - Pre-layout N-split for fp4 matmul -------===//
//
// Splits tt.dot_scaled ops with fp4 B operand along N at the TTIR level
// (before layout assignment) when the static GRF estimate predicts register
// spill on BMG-class hardware without hardware BDPAS.
//
// Root cause: DecomposeScaledBlocked emits a scale-multiply chain that IGC
// lowers to 64 scalar f32 B values per thread simultaneously (from
// ConvertBF16ToFINTEL calls). Combined with the 64-scalar f32 accumulator this
// reaches 256 GRF — exactly the SIMD-16 budget — and any misc register causes
// the accumulator to spill every K-iteration (10,688 B, ~2.8 TFLOPS vs ~7).
//
// Fix: split the DotScaledOp into two half-N ops before layout assignment.
// At TTIR level tensors have no encoding, so tt.split's sizePerThread>=2
// constraint (gated on `if (srcEnc)`) is bypassed.  Each half-N decomposition
// sees only 32 B f32 values, keeping peak GRF ~196.
//
// The split produces:
//   d0 = dot_scaled(A, B[:,0:N/2],   C[:,0:N/2],   scaleA, scaleB[0:N/2,:])
//   d1 = dot_scaled(A, B[:,N/2:N],   C[:,N/2:N],   scaleA, scaleB[N/2:N,:])
//   d  = tt.cat(d0, d1)   -> [M, N]
// scaleA is NOT split (A is reused identically for both N-halves).
//===----------------------------------------------------------------------===//

#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONINTELGPUSPLITLARGENDOTSCALED
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

namespace {

// Returns true when processing the full B fp4 tile for this DotScaledOp would
// overflow the 256-GRF budget on a SIMD-16 BMG target.
//
// IGC materialises bF32PerThread scalar f32 B values (from ConvertBF16ToFINTEL)
// plus accumF32PerThread scalar f32 accumulator phi nodes simultaneously.
// Describes which dimension to split and which operand is the fp4 bottleneck.
enum class SplitKind { None, SplitN, SplitM };

// Returns the split kind (SplitN for fp4 B overflow, SplitM for fp4 A
// overflow, None when no split is needed).
//
// At the B × scale_B (or A × scale_A) multiply step, IGC materialises
// opF32PerThread scalar f32 values simultaneously with accumF32PerThread
// accumulator scalars.  In SIMD-16, each scalar f32 per work-item = 2 GRFs.
static SplitKind splitKind(tt::DotScaledOp op, int numWarps,
                           int threadsPerWarp) {
  // Only fire on targets without hardware BDPAS (e.g. BMG Xe2).
  auto mod = op->getParentOfType<ModuleOp>();
  if (mod->hasAttr(ttg::intel::TritonIntelGPUDialect::
                       getSupportBlockScaleDPASAttrName()))
    return SplitKind::None;

  int numThreads = numWarps * threadsPerWarp;
  if (numThreads <= 0)
    return SplitKind::None;

  int64_t accumF32PerThread =
      cast<RankedTensorType>(op.getD().getType()).getNumElements() / numThreads;

  // B=fp4: bottleneck at B × scale_B; split along N.
  if (op.getBElemType() == tt::ScaleDotElemType::E2M1) {
    // fp4_to_fp doubles element count (2 bf16 per packed i8 byte).
    int64_t bF32PerThread =
        2 * cast<RankedTensorType>(op.getB().getType()).getNumElements() /
        numThreads;
    if ((bF32PerThread + accumF32PerThread) * 2 >= 256)
      return SplitKind::SplitN;
  }

  // A=fp4: bottleneck at A × scale_A; split along M.
  if (op.getAElemType() == tt::ScaleDotElemType::E2M1) {
    int64_t aF32PerThread =
        2 * cast<RankedTensorType>(op.getA().getType()).getNumElements() /
        numThreads;
    if ((aF32PerThread + accumF32PerThread) * 2 >= 256)
      return SplitKind::SplitM;
  }

  return SplitKind::None;
}

// Split a rank-2 tensor [D0, N] along its last dimension into two [D0, N/2]
// tensors covering the contiguous first and second N-halves respectively.
//
// Uses the idiom:
//   reshape [D0,N] → [D0,2,N/2]    (element [i,j] → [i, j//(N/2), j%(N/2)])
//   trans   [D0,2,N/2] → [D0,N/2,2]  (put the size-2 "half indicator" last)
//   split   [D0,N/2,2] → {[D0,N/2], [D0,N/2]}
//
// At TTIR level tensors have no encoding; all three ops are unconstrained.
static std::pair<Value, Value> splitAlongLastDim(OpBuilder &b, Location loc,
                                                 Value v) {
  auto ty = cast<RankedTensorType>(v.getType());
  assert(ty.getRank() == 2 && "expected rank-2 tensor");
  int64_t d0 = ty.getShape()[0], N = ty.getShape()[1];
  assert(N % 2 == 0 && "N must be even to split in half");

  Value reshaped =
      tt::ReshapeOp::create(b, loc, SmallVector<int64_t>{d0, 2, N / 2}, v);
  Value transposed =
      tt::TransOp::create(b, loc, reshaped, SmallVector<int32_t>{0, 2, 1});
  auto splitOp = tt::SplitOp::create(b, loc, transposed);
  return {splitOp.getOutLHS(), splitOp.getOutRHS()};
}

// Split a rank-2 tensor [N, Ksf] along its first dimension into two [N/2, Ksf]
// tensors covering the contiguous first and second N-halves.  Used for scaleB.
//
//   reshape [N,Ksf] → [2,N/2,Ksf]
//   trans   [2,N/2,Ksf] → [N/2,Ksf,2]
//   split   [N/2,Ksf,2] → {[N/2,Ksf], [N/2,Ksf]}
static std::pair<Value, Value> splitAlongFirstDim(OpBuilder &b, Location loc,
                                                  Value v) {
  auto ty = cast<RankedTensorType>(v.getType());
  assert(ty.getRank() == 2 && "expected rank-2 tensor");
  int64_t N = ty.getShape()[0], Ksf = ty.getShape()[1];
  assert(N % 2 == 0 && "N must be even to split in half");

  Value reshaped =
      tt::ReshapeOp::create(b, loc, SmallVector<int64_t>{2, N / 2, Ksf}, v);
  Value transposed =
      tt::TransOp::create(b, loc, reshaped, SmallVector<int32_t>{1, 2, 0});
  auto splitOp = tt::SplitOp::create(b, loc, transposed);
  return {splitOp.getOutLHS(), splitOp.getOutRHS()};
}

struct SplitPattern : public OpRewritePattern<tt::DotScaledOp> {
  SplitPattern(MLIRContext *ctx, int numWarps, int threadsPerWarp)
      : OpRewritePattern<tt::DotScaledOp>(ctx), numWarps(numWarps),
        threadsPerWarp(threadsPerWarp) {}

  LogicalResult matchAndRewrite(tt::DotScaledOp op,
                                PatternRewriter &rewriter) const override {
    SplitKind kind = splitKind(op, numWarps, threadsPerWarp);
    if (kind == SplitKind::None)
      return failure();

    auto loc = op.getLoc();

    Value aH1, aH2, bH1, bH2, sAH1, sAH2, sBH1, sBH2, cH1, cH2;

    if (kind == SplitKind::SplitN) {
      // B=fp4 overflow: split B [K/2,N] and C [M,N] along N (last dim).
      std::tie(bH1, bH2) = splitAlongLastDim(rewriter, loc, op.getB());
      if (op.getBScale())
        std::tie(sBH1, sBH2) =
            splitAlongFirstDim(rewriter, loc, op.getBScale());
      std::tie(cH1, cH2) = splitAlongLastDim(rewriter, loc, op.getC());
      aH1 = aH2 = op.getA();        // A unchanged
      sAH1 = sAH2 = op.getAScale(); // scaleA unchanged
    } else {
      // A=fp4 overflow (SplitM): split A [M,K/2] and C [M,N] along M (first
      // dim).
      std::tie(aH1, aH2) = splitAlongFirstDim(rewriter, loc, op.getA());
      if (op.getAScale())
        std::tie(sAH1, sAH2) =
            splitAlongFirstDim(rewriter, loc, op.getAScale());
      std::tie(cH1, cH2) = splitAlongFirstDim(rewriter, loc, op.getC());
      bH1 = bH2 = op.getB();        // B unchanged
      sBH1 = sBH2 = op.getBScale(); // scaleB unchanged
    }

    auto cHalfTy = cast<RankedTensorType>(cH1.getType());

    // Emit two half-dimension DotScaledOps.
    auto d0 = tt::DotScaledOp::create(
        rewriter, loc, cHalfTy, aH1, bH1, cH1, sAH1, sBH1,
        op.getAElemTypeAttr(), op.getBElemTypeAttr(), op.getFastMathAttr(),
        op.getLhsKPackAttr(), op.getRhsKPackAttr());
    auto d1 = tt::DotScaledOp::create(
        rewriter, loc, cHalfTy, aH2, bH2, cH2, sAH2, sBH2,
        op.getAElemTypeAttr(), op.getBElemTypeAttr(), op.getFastMathAttr(),
        op.getLhsKPackAttr(), op.getRhsKPackAttr());

    // Reassemble using the exact inverse of the split.
    // SplitN: join+trans+reshape along N (last dim).
    // SplitM: join+trans+reshape along M (first dim).
    auto halfTy = cast<RankedTensorType>(d0.getType());
    int64_t Mh = halfTy.getShape()[0], Nh = halfTy.getShape()[1];
    Value joined = tt::JoinOp::create(rewriter, loc, d0, d1); // [Mh,Nh,2]

    Value result;
    if (kind == SplitKind::SplitN) {
      // Inverse of splitAlongLastDim: [Mh,Nh,2] → trans[0,2,1] → [Mh,2,Nh]
      //   → reshape → [Mh, Nh*2]
      Value trans = tt::TransOp::create(rewriter, loc, joined,
                                        SmallVector<int32_t>{0, 2, 1});
      result = tt::ReshapeOp::create(rewriter, loc,
                                     SmallVector<int64_t>{Mh, Nh * 2}, trans);
    } else {
      // Inverse of splitAlongFirstDim: [Mh,Nh,2] → trans[2,0,1] → [2,Mh,Nh]
      //   → reshape → [Mh*2, Nh]
      Value trans = tt::TransOp::create(rewriter, loc, joined,
                                        SmallVector<int32_t>{2, 0, 1});
      result = tt::ReshapeOp::create(rewriter, loc,
                                     SmallVector<int64_t>{Mh * 2, Nh}, trans);
    }
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  int numWarps;
  int threadsPerWarp;
};

struct SplitLargeNDotScaledPass
    : mlir::triton::gpu::intel::impl::TritonIntelGPUSplitLargeNDotScaledBase<
          SplitLargeNDotScaledPass> {
  using mlir::triton::gpu::intel::impl::TritonIntelGPUSplitLargeNDotScaledBase<
      SplitLargeNDotScaledPass>::TritonIntelGPUSplitLargeNDotScaledBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    int tpw = ttg::TritonGPUDialect::getThreadsPerWarp(mod);
    if (tpw <= 0)
      tpw = 16; // BMG default subgroup size
    RewritePatternSet patterns(&getContext());
    patterns.add<SplitPattern>(&getContext(), numWarps, tpw);
    (void)applyPatternsGreedily(mod, std::move(patterns));
  }
};

} // namespace
