#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/Debug.h"

#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONINTELGPUSTAGELARGEFMADOTSVIASLM
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;

#define DEBUG_TYPE "tritonintelgpu-stage-large-fma-dots-via-slm"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

// Per-thread live-bytes threshold above which the pass fires. The 4 KB
// figure approximates the worst-case GRF pressure for one thread holding
// fully-unrolled K accumulations on a non-DPAS Intel GPU.
constexpr unsigned kPressureThresholdBytes = 4096;

// Conservative target post-decompose live-bytes per K-tile per thread. Used
// to derive K_TILE adaptively rather than picking from a hardcoded set.
constexpr unsigned kTargetBytesPerOp = 1024;

// SLM safety margin (bytes). Leaves headroom for other passes (RDD,
// RemoveLayoutConversions) that also stage through SLM. ARL-S iGPU has
// 64 KB; allowing up to 56 KB for our staging keeps 8 KB free.
constexpr unsigned kSlmCapBytes = 56u * 1024u;

// Estimate per-thread live operand bytes if FMA unrolls K fully.
// All arithmetic is done in 64-bit: shapes come from getShape() as int64_t,
// and byte products of large tensors can exceed the 32-bit range.
static uint64_t estimatePerThreadBytes(tt::DotOp dotOp) {
  auto aType = cast<RankedTensorType>(dotOp.getA().getType());
  auto resultType = dotOp.getType();
  auto enc = dyn_cast<ttg::BlockedEncodingAttr>(resultType.getEncoding());
  if (!enc)
    return 0;

  unsigned rank = aType.getRank();
  int64_t M = resultType.getShape()[rank - 2];
  int64_t N = resultType.getShape()[rank - 1];
  int64_t K = aType.getShape()[rank - 1];
  int64_t elemBytes = std::max<int64_t>(1, aType.getElementTypeBitWidth() / 8);
  int64_t accBytes = resultType.getElementTypeBitWidth() / 8;

  auto spt = enc.getSizePerThread();
  auto tpw = enc.getThreadsPerWarp();
  auto wpc = enc.getWarpsPerCTA();
  int64_t mSpt = spt[rank - 2], nSpt = spt[rank - 1];
  int64_t ctaTileM = mSpt * tpw[rank - 2] * wpc[rank - 2];
  int64_t ctaTileN = nSpt * tpw[rank - 1] * wpc[rank - 1];
  int64_t mReps = ctaTileM ? M / ctaTileM : 1;
  int64_t nReps = ctaTileN ? N / ctaTileN : 1;

  uint64_t aBytes = uint64_t(mReps) * mSpt * K * elemBytes;
  uint64_t bBytes = uint64_t(K) * nReps * nSpt * elemBytes;
  uint64_t cBytes = uint64_t(mReps) * mSpt * nReps * nSpt * accBytes;
  return aBytes + bBytes + cBytes;
}

// Total SLM bytes required to stage operands A and B in their full shape.
// Uses the source-side (pre-fp_to_fp) element type if present, since that's
// what we actually allocate. Falls back to the dot operand element type.
static uint64_t slmStagingBytes(Value aSrc, Value bSrc) {
  auto bytes = [](Value v) -> uint64_t {
    auto t = cast<RankedTensorType>(v.getType());
    int64_t eb = std::max<int64_t>(1, t.getElementTypeBitWidth() / 8);
    int64_t n = 1;
    for (int64_t d : t.getShape())
      n *= d;
    return uint64_t(n) * eb;
  };
  return bytes(aSrc) + bytes(bSrc);
}

// Pick a K_TILE that's a power of 2, divides K, and fits the per-op byte
// budget. Returns 0 if no viable tile.
//
// The minimum tile is 16 (matches the smallest kWidth supported across
// our DPAS/non-DPAS layouts). We deliberately do not require alignment
// with the parent BlockedEncoding's perCTAK: ttg.local_load produces a
// new dot_op tensor whose layout is recomputed from the tile shape, and
// downstream verification handles whether the partial-K dot_op tensor is
// a valid input to tt.dot.
static uint64_t selectKTile(tt::DotOp dotOp) {
  auto aType = cast<RankedTensorType>(dotOp.getA().getType());
  auto resultType = dotOp.getType();
  auto enc = cast<ttg::BlockedEncodingAttr>(resultType.getEncoding());
  unsigned rank = aType.getRank();
  int64_t K = aType.getShape()[rank - 1]; // shape dim, always >= 0
  int64_t M = resultType.getShape()[rank - 2];
  int64_t elemBytes = std::max<int64_t>(1, aType.getElementTypeBitWidth() / 8);
  int64_t mSpt = enc.getSizePerThread()[rank - 2];
  int64_t denom =
      mSpt * enc.getThreadsPerWarp()[rank - 2] * enc.getWarpsPerCTA()[rank - 2];
  int64_t mReps = denom ? M / denom : 1;
  if (mReps == 0)
    mReps = 1;

  uint64_t perKByte = uint64_t(mReps) * mSpt * elemBytes;
  uint64_t maxKTile = perKByte ? kTargetBytesPerOp / perKByte : 32;
  if (maxKTile == 0)
    maxKTile = 16;

  uint64_t k = uint64_t(K);
  uint64_t tile = llvm::bit_floor(maxKTile | 1u);
  while (tile > 16 && (tile >= k || k % tile != 0))
    tile >>= 1;
  if (tile < 16 || tile >= k || k % tile != 0)
    return 0;
  return tile;
}

// If `v` is the result of a ttg.local_load, return the memdesc it loads
// from (so we can reuse already-staged SLM). Otherwise return nullptr.
static Value findStagedSmem(Value v) {
  if (auto load = v.getDefiningOp<ttg::LocalLoadOp>())
    return load.getSrc();
  return Value();
}

// Per-operand chain we found between `ttg.convert_layout` (the SLM-staging
// boundary) and `tt.dot`. After AccelerateMatmul, this is typically:
//   tt.load -> ttg.convert_layout (to dot_op) -> tt.fp_to_fp (f16->f32) -> dot
// We stage the pre-fp_to_fp f16 value (smaller in SLM) and replay
// `fp_to_fp` per K-tile after `local_load`.
struct OperandChain {
  Value smemSource;                    // pre-staging tensor in BlockedEnc
  ttg::DotOperandEncodingAttr loadEnc; // dot_op encoding for partial-K
                                       // local_load result (post-staging
                                       // tensor element type)
  Type postLoadElemTy;                 // element type after replaying chain
  // Ops to replay on the partial-K loaded value, innermost first. Each is
  // either tt.fp_to_fp or arith.extf with a single tensor input.
  SmallVector<Operation *> replay;
};

// Walk back from the dot operand to the smallest pre-promotion boundary.
// Two real shapes seen in the wild after AccelerateMatmul:
//   (a) tt.load -> ttg.convert_layout (to dot_op) -> tt.fp_to_fp -> dot
//   (b) tt.load -> tt.fp_to_fp -> ttg.convert_layout (to dot_op) -> dot
// We always stage at the smallest representation: hop the convert_layout to
// reach dot_op encoding, then keep walking through fp_to_fp/extf to find
// the pre-promotion source. The replay ops are applied per K-tile after the
// partial-K local_load.
static std::optional<OperandChain> findOperandChain(Value dotOperand) {
  SmallVector<Operation *> replay;
  Value v = dotOperand;
  Type postLoadElemTy =
      cast<RankedTensorType>(dotOperand.getType()).getElementType();
  ttg::DotOperandEncodingAttr dotOpEnc;
  bool sawBoundary = false;

  while (true) {
    Operation *def = v.getDefiningOp();
    if (!def)
      return std::nullopt;

    if (!sawBoundary) {
      // Boundary forms:
      //   (1) ttg.convert_layout (to dot_op) — typical post-AccelerateMatmul
      //   (2) ttg.local_load (from a memdesc) — SLM already staged by RDD
      //       or software pipelining
      if (auto cvt = dyn_cast<ttg::ConvertLayoutOp>(def)) {
        dotOpEnc = dyn_cast<ttg::DotOperandEncodingAttr>(
            cast<RankedTensorType>(cvt.getResult().getType()).getEncoding());
        if (!dotOpEnc)
          return std::nullopt;
        sawBoundary = true;
        v = cvt.getSrc();
        continue;
      }
      if (auto load = dyn_cast<ttg::LocalLoadOp>(def)) {
        dotOpEnc = dyn_cast<ttg::DotOperandEncodingAttr>(
            cast<RankedTensorType>(load.getResult().getType()).getEncoding());
        if (!dotOpEnc)
          return std::nullopt;
        // We don't walk further: the boundary is the local_load itself.
        // stageDot's findStagedSmem will pick up the same local_load and
        // reuse its source memdesc.
        OperandChain chain;
        chain.smemSource = dotOperand; // marker: stageDot reuses memdesc
        chain.loadEnc = dotOpEnc;
        chain.postLoadElemTy = postLoadElemTy;
        // No replay ops in the local_load case.
        return chain;
      }
      if (isa<tt::FpToFpOp, arith::ExtFOp>(def)) {
        replay.push_back(def);
        v = def->getOperand(0);
        continue;
      }
      return std::nullopt;
    }

    // Past the convert_layout: continue walking through promotion casts
    // to reach the smallest pre-promotion source for SLM staging.
    if (isa<tt::FpToFpOp, arith::ExtFOp>(def)) {
      replay.push_back(def);
      v = def->getOperand(0);
      continue;
    }
    // Stop here. `v` is the SLM staging source (could be a tt.load result,
    // or any op producing a tensor with the source-side BlockedEncoding).
    break;
  }

  if (!sawBoundary)
    return std::nullopt;

  // replay was collected innermost-first as we walked back. To apply on a
  // partial-K loaded value (which has the pre-promotion element type) we
  // need to apply outermost-cast last, i.e. iterate replay in reverse.
  std::reverse(replay.begin(), replay.end());

  OperandChain chain;
  chain.smemSource = v;
  chain.loadEnc = dotOpEnc;
  chain.postLoadElemTy = postLoadElemTy;
  chain.replay = std::move(replay);
  return chain;
}

// Replay each op in `chain` on `v`, adjusting the result type to match
// the new (partial-K) shape.  The op's existing result element type is
// preserved; only the shape changes.
static Value replayChain(OpBuilder &builder, Location loc, Value v,
                         ArrayRef<Operation *> chain) {
  for (Operation *op : chain) {
    auto newInTy = cast<RankedTensorType>(v.getType());
    auto origOutTy = cast<RankedTensorType>(op->getResult(0).getType());
    auto newOutTy = RankedTensorType::get(
        newInTy.getShape(), origOutTy.getElementType(), newInTy.getEncoding());
    if (auto fp = dyn_cast<tt::FpToFpOp>(op)) {
      v = tt::FpToFpOp::create(builder, loc, newOutTy, v, fp.getRoundingAttr());
    } else if (auto ex = dyn_cast<arith::ExtFOp>(op)) {
      v = arith::ExtFOp::create(builder, loc, newOutTy, v);
    }
  }
  return v;
}

// Build a SwizzledShared memdesc type for `srcTensor` matched to the dot
// operand encoding `dstDotOp` (so the partial-K local_load is well-formed).
static ttg::MemDescType buildShared(MLIRContext *ctx, RankedTensorType srcTy,
                                    ttg::DotOperandEncodingAttr dstDotOp) {
  auto srcOrder = ttg::getOrder(srcTy);
  SmallVector<unsigned> sharedOrder(srcOrder.begin(), srcOrder.end());
  auto smemSpace = ttg::SharedMemorySpaceAttr::get(ctx);
  auto sharedEnc = ttg::SwizzledSharedEncodingAttr::get(
      ctx, dstDotOp, srcTy.getShape(), sharedOrder,
      ttg::getCGALayout(srcTy.getEncoding()), srcTy.getElementType());
  return ttg::MemDescType::get(srcTy.getShape(), srcTy.getElementType(),
                               sharedEnc, smemSpace);
}

// Drop the K-dim of a memdesc to `kTile`, keeping all other dims.
static ttg::MemDescType sliceK(ttg::MemDescType src, unsigned kDim,
                               int64_t kTile) {
  SmallVector<int64_t> newShape(src.getShape());
  newShape[kDim] = kTile;
  return ttg::MemDescType::get(newShape, src.getElementType(),
                               src.getEncoding(), src.getMemorySpace(),
                               src.getMutableMemory(), src.getAllocShape());
}

// Drop the K-dim of a tensor type to `kTile`, keeping its dot_op encoding.
static RankedTensorType sliceTensorK(RankedTensorType src, unsigned kDim,
                                     int64_t kTile) {
  SmallVector<int64_t> newShape(src.getShape());
  newShape[kDim] = kTile;
  return RankedTensorType::get(newShape, src.getElementType(),
                               src.getEncoding());
}

static LogicalResult stageDot(tt::DotOp dotOp, const OperandChain &aChain,
                              const OperandChain &bChain) {
  Location loc = dotOp.getLoc();
  OpBuilder builder(dotOp);

  auto aType = cast<RankedTensorType>(dotOp.getA().getType());
  unsigned rank = aType.getRank();
  unsigned kDimA = rank - 1;
  unsigned kDimB = rank - 2;
  int64_t K = aType.getShape()[kDimA];
  uint64_t kTile = selectKTile(dotOp);
  if (kTile == 0) {
    LDBG("no viable kTile for dot " << dotOp);
    return failure();
  }
  uint64_t numTiles = uint64_t(K) / kTile;
  LDBG("staging dot K=" << K << " kTile=" << kTile << " numTiles=" << numTiles);

  auto aSrcTy = cast<RankedTensorType>(aChain.smemSource.getType());
  auto bSrcTy = cast<RankedTensorType>(bChain.smemSource.getType());

  MLIRContext *ctx = dotOp.getContext();

  // Reuse SLM if a prior pass already staged it; otherwise allocate at the
  // pre-promotion (smaller) element type.
  Value aSmem = findStagedSmem(dotOp.getA());
  if (!aSmem)
    aSmem = ttg::LocalAllocOp::create(builder, loc,
                                      buildShared(ctx, aSrcTy, aChain.loadEnc),
                                      aChain.smemSource);
  Value bSmem = findStagedSmem(dotOp.getB());
  if (!bSmem)
    bSmem = ttg::LocalAllocOp::create(builder, loc,
                                      buildShared(ctx, bSrcTy, bChain.loadEnc),
                                      bChain.smemSource);

  auto aSmemTy = cast<ttg::MemDescType>(aSmem.getType());
  auto bSmemTy = cast<ttg::MemDescType>(bSmem.getType());

  Value acc = dotOp.getC();
  auto aSubTy = sliceK(aSmemTy, kDimA, kTile);
  auto bSubTy = sliceK(bSmemTy, kDimB, kTile);
  // The local_load result tensor uses the SLM element type (pre-promotion)
  // and the dot operand encoding from the chain.
  auto aLoadTy = RankedTensorType::get(aSubTy.getShape(),
                                       aSrcTy.getElementType(), aChain.loadEnc);
  auto bLoadTy = RankedTensorType::get(bSubTy.getShape(),
                                       bSrcTy.getElementType(), bChain.loadEnc);

  for (uint64_t i = 0; i < numTiles; ++i) {
    // Offset into the memdesc along K. MemDescSubsliceOp takes a
    // DenseI32ArrayAttr, so the offset is int32 by the op's ABI; it is
    // bounded by K, which fits int32 for any realizable tensor.
    int32_t kOffset = static_cast<int32_t>(i * kTile);
    SmallVector<int32_t> aOffset(rank, 0);
    aOffset[kDimA] = kOffset;
    SmallVector<int32_t> bOffset(rank, 0);
    bOffset[kDimB] = kOffset;

    Value aSub =
        ttg::MemDescSubsliceOp::create(builder, loc, aSubTy, aSmem, aOffset);
    Value bSub =
        ttg::MemDescSubsliceOp::create(builder, loc, bSubTy, bSmem, bOffset);
    Value aPart = ttg::LocalLoadOp::create(builder, loc, aLoadTy, aSub);
    Value bPart = ttg::LocalLoadOp::create(builder, loc, bLoadTy, bSub);
    aPart = replayChain(builder, loc, aPart, aChain.replay);
    bPart = replayChain(builder, loc, bPart, bChain.replay);
    acc = tt::DotOp::create(builder, loc, dotOp.getType(), aPart, bPart, acc,
                            dotOp.getInputPrecisionAttr(),
                            dotOp.getMaxNumImpreciseAccAttr());
  }

  // Collect the orphaned defining ops from the original chain so we can
  // erase them after RAUW. The dot's original operands feed only into this
  // dot; once we replace the dot, they're dead.
  Value oldA = dotOp.getA();
  Value oldB = dotOp.getB();
  dotOp.replaceAllUsesWith(acc);
  dotOp.erase();

  // Walk back from oldA/oldB and erase fp_to_fp/extf/convert_layout ops
  // that have no remaining users.
  auto eraseDeadChain = [](Value v) {
    while (Operation *def = v.getDefiningOp()) {
      if (!isa<tt::FpToFpOp, arith::ExtFOp, ttg::ConvertLayoutOp>(def))
        break;
      if (!def->use_empty())
        break;
      Value next = def->getOperand(0);
      def->erase();
      v = next;
    }
  };
  eraseDeadChain(oldA);
  eraseDeadChain(oldB);
  return success();
}

struct StageLargeFMADotsViaSLMPass
    : public ttgi::impl::TritonIntelGPUStageLargeFMADotsViaSLMBase<
          StageLargeFMADotsViaSLMPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();

    // The pass fires on any tt.dot whose result still has BlockedEncoding.
    // On DPAS hardware (PVC, BMG, ARL-H Xe2), AccelerateMatmul rewrites
    // DPAS-lowerable dots to DpasEncoding so the BlockedEncoding check
    // excludes them automatically. f32 dots with `inputPrecision = ieee`
    // are not DPAS-lowerable (DPAS only handles f32 via TF32 truncation),
    // so they retain BlockedEncoding even on DPAS hardware and lower via
    // FMA — exactly the case this pass needs to fire on.
    SmallVector<std::tuple<tt::DotOp, OperandChain, OperandChain>> targets;
    mod.walk([&](tt::DotOp dotOp) {
      auto resTy = dotOp.getType();
      if (!isa<ttg::BlockedEncodingAttr>(resTy.getEncoding()))
        return;
      uint64_t bytes = estimatePerThreadBytes(dotOp);
      if (bytes <= kPressureThresholdBytes) {
        LDBG("dot under pressure threshold (" << bytes << " B), skipping");
        return;
      }
      auto aChain = findOperandChain(dotOp.getA());
      auto bChain = findOperandChain(dotOp.getB());
      if (!aChain || !bChain) {
        LDBG("dot operand chain not at SLM boundary, skipping");
        return;
      }
      uint64_t smem = slmStagingBytes(aChain->smemSource, bChain->smemSource);
      if (smem > kSlmCapBytes) {
        LDBG("dot SLM-staging cost " << smem << " B exceeds cap "
                                     << kSlmCapBytes << " B, skipping");
        return;
      }
      if (selectKTile(dotOp) == 0) {
        LDBG("no viable kTile for dot " << dotOp);
        return;
      }
      targets.emplace_back(dotOp, std::move(*aChain), std::move(*bChain));
    });

    for (auto &[dotOp, aChain, bChain] : targets)
      (void)stageDot(dotOp, aChain, bChain);
  }
};

} // namespace
