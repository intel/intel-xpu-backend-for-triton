#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/TypeUtilities.h"
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

// Smallest K tile when an operand is staged decoded (see stageDot). The dot is
// lowered to FMA, which has no kWidth constraint.
constexpr uint64_t kMinDecodedKTile = 8;

// SLM safety margin (bytes). Leaves headroom for other passes (RDD,
// RemoveLayoutConversions) that also stage through SLM. The staging may use
// the target's SLM minus this margin: 56 KB of the 64 KB on ARL-S, which is
// also assumed when the SLM size is unknown.
constexpr unsigned kSlmMarginBytes = 8u * 1024u;
constexpr unsigned kDefaultSlmBytes = 64u * 1024u;

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

// SLM bytes required to stage `v` in its full shape, at its element type.
static uint64_t slmStagingBytes(Value v) {
  auto t = cast<RankedTensorType>(v.getType());
  int64_t eb = std::max<int64_t>(1, t.getElementTypeBitWidth() / 8);
  int64_t n = 1;
  for (int64_t d : t.getShape())
    n *= d;
  return uint64_t(n) * eb;
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
//
// When the operand was decoded from fp8 before the convert_layout
// (AccelerateMatmul does this so the multi-instruction decode runs once per
// element rather than once per replicated copy), staging the fp8 value would
// replay the decode per copy again. The chain then also records the
// alternative of staging the decoded value, the convert_layout source, at
// twice the SLM; `useDecoded` selects it.
struct OperandChain {
  Value smemSource;                    // pre-staging tensor in BlockedEnc
  ttg::DotOperandEncodingAttr loadEnc; // dot_op encoding for partial-K
                                       // local_load result (post-staging
                                       // tensor element type)
  Type postLoadElemTy;                 // element type after replaying chain
  // Ops to replay on the partial-K loaded value, in the order they apply (the
  // cast nearest the staged value first). Each is either tt.fp_to_fp or
  // arith.extf with a single tensor input.
  SmallVector<Operation *> replay;
  // The fp8 value decoded before the convert_layout, and the casts after the
  // convert_layout to replay on it. Null if there is no such decode.
  Value decodedSource;
  SmallVector<Operation *> decodedReplay;
  bool useDecoded = false;

  Value stagedSource() const { return useDecoded ? decodedSource : smemSource; }
  ArrayRef<Operation *> stagedReplay() const {
    return useDecoded ? decodedReplay : replay;
  }
};

// Whether the only cast before the convert_layout is an fp8 decode to a 16-bit
// type, so that the decoded value can be staged instead of the fp8 one.
// `preCvtCasts` is in the order the casts apply, the one nearest the load
// first. The size check matters: with fp8 -> fp16 -> fp32, front() is the
// decode, and staging the conversion's source would stage fp32. With native
// fp8 conversion, fp8 -> fp16 is a single builtin and replaying it per K tile
// is cheaper than staging twice the bytes.
static bool isDecodeBeforeConversion(ArrayRef<Operation *> preCvtCasts,
                                     ModuleOp mod) {
  if (preCvtCasts.empty())
    return false;
  auto fp = dyn_cast<tt::FpToFpOp>(preCvtCasts.front());
  if (!fp)
    return false;
  Type srcTy = getElementTypeOrSelf(fp.getSrc());
  Type dstTy = getElementTypeOrSelf(fp.getResult());
  if (!isa<FloatType>(srcTy) || srcTy.getIntOrFloatBitWidth() != 8 ||
      !(dstTy.isF16() || dstTy.isBF16()))
    return false;
  // Only stage the decoded value if every cast before the convert_layout is
  // the decode itself: a further widening (e.g. to fp32) would be staged too.
  if (preCvtCasts.size() != 1)
    return false;
  return !(dstTy.isF16() &&
           mod->hasAttr(
               ttgi::TritonIntelGPUDialect::getSupportF8ConversionAttrName()));
}

// Walk back from the dot operand to the smallest pre-promotion boundary.
// Two real shapes seen in the wild after AccelerateMatmul:
//   (a) tt.load -> ttg.convert_layout (to dot_op) -> tt.fp_to_fp -> dot
//   (b) tt.load -> tt.fp_to_fp -> ttg.convert_layout (to dot_op) -> dot
// The chain records the smallest representation: hop the convert_layout to
// reach dot_op encoding, then keep walking through fp_to_fp/extf to find
// the pre-promotion source. The replay ops are applied per K-tile after the
// partial-K local_load. In shape (b), when the fp_to_fp is an fp8 decode, it
// also records the convert_layout source as the decoded alternative.
static std::optional<OperandChain> findOperandChain(Value dotOperand) {
  SmallVector<Operation *> replay;
  Value v = dotOperand;
  Type postLoadElemTy =
      cast<RankedTensorType>(dotOperand.getType()).getElementType();
  ttg::DotOperandEncodingAttr dotOpEnc;
  bool sawBoundary = false;
  Value cvtSrc;
  size_t numPostCvtCasts = 0;

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
        cvtSrc = cvt.getSrc();
        numPostCvtCasts = replay.size();
        v = cvtSrc;
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
  // After the reverse, the casts before the convert_layout come first.
  ArrayRef<Operation *> casts(replay);
  size_t numPreCvtCasts = casts.size() - numPostCvtCasts;
  if (isDecodeBeforeConversion(
          casts.take_front(numPreCvtCasts),
          dotOperand.getParentRegion()->getParentOfType<ModuleOp>())) {
    chain.decodedSource = cvtSrc;
    chain.decodedReplay.assign(casts.begin() + numPreCvtCasts, casts.end());
  }
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
      v = arith::ExtFOp::create(builder, loc, newOutTy, v,
                                arith::FastMathFlagsAttr{});
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
  // A decoded operand is read back from SLM as 16-bit values instead of fp8,
  // which needs more registers per K tile. On BMG that made IGC spill more on
  // some tiles, enough to make the dot slower than staging fp8. With the K
  // tile halved, those spills went away and the decoded staging was no slower
  // than staging fp8 on any tile measured.
  if ((aChain.useDecoded || bChain.useDecoded) &&
      kTile / 2 >= kMinDecodedKTile && uint64_t(K) % (kTile / 2) == 0)
    kTile /= 2;
  uint64_t numTiles = uint64_t(K) / kTile;
  LDBG("staging dot K=" << K << " kTile=" << kTile << " numTiles=" << numTiles);

  auto aSrcTy = cast<RankedTensorType>(aChain.stagedSource().getType());
  auto bSrcTy = cast<RankedTensorType>(bChain.stagedSource().getType());

  MLIRContext *ctx = dotOp.getContext();

  // Reuse SLM if a prior pass already staged it; otherwise allocate at the
  // element type of the value chosen for staging (the pre-promotion value, or
  // the decoded one).
  Value aSmem = findStagedSmem(dotOp.getA());
  if (!aSmem)
    aSmem = ttg::LocalAllocOp::create(builder, loc,
                                      buildShared(ctx, aSrcTy, aChain.loadEnc),
                                      aChain.stagedSource());
  Value bSmem = findStagedSmem(dotOp.getB());
  if (!bSmem)
    bSmem = ttg::LocalAllocOp::create(builder, loc,
                                      buildShared(ctx, bSrcTy, bChain.loadEnc),
                                      bChain.stagedSource());

  auto aSmemTy = cast<ttg::MemDescType>(aSmem.getType());
  auto bSmemTy = cast<ttg::MemDescType>(bSmem.getType());

  Value acc = dotOp.getC();
  auto aSubTy = sliceK(aSmemTy, kDimA, kTile);
  auto bSubTy = sliceK(bSmemTy, kDimB, kTile);
  // The local_load result tensor uses the SLM element type and the dot
  // operand encoding from the chain.
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
    aPart = replayChain(builder, loc, aPart, aChain.stagedReplay());
    bPart = replayChain(builder, loc, bPart, bChain.stagedReplay());
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
  using Base::Base;

  // A way to stage a dot: which operands to stage decoded, and the SLM bytes
  // that takes.
  struct StagingOption {
    bool decodeA = false;
    bool decodeB = false;
    uint64_t bytes = 0;
  };

  // The ways to stage `dotOp`, most preferred first: both operands decoded,
  // then only the operand with more copies in total decoded, then the other,
  // then both as they are. Staging decoded values runs the fp8 decode once per
  // element instead of once per copy on every K tile, at twice the SLM for that
  // operand, so the operand with more copies in total (elements per thread)
  // saves more decodes. If an option does not fit, the next one is tried. The
  // last option, both as they are, is always there.
  SmallVector<StagingOption, 4> getStagingOptions(tt::DotOp dotOp,
                                                  const OperandChain &aChain,
                                                  const OperandChain &bChain) {
    auto bytes = [](const OperandChain &chain, bool decoded) {
      return slmStagingBytes(decoded ? chain.decodedSource : chain.smemSource);
    };
    bool canA = stageDecoded && aChain.decodedSource;
    bool canB = stageDecoded && bChain.decodedSource;
    SmallVector<std::pair<bool, bool>, 4> choices;
    if (canA && canB)
      choices.push_back({true, true});
    std::pair<bool, bool> onlyA{true, false}, onlyB{false, true};
    if (ttg::getTotalElemsPerThread(dotOp.getA().getType()) <
        ttg::getTotalElemsPerThread(dotOp.getB().getType()))
      std::swap(onlyA, onlyB);
    for (auto choice : {onlyA, onlyB})
      if (choice.first ? canA : canB)
        choices.push_back(choice);
    choices.push_back({false, false});

    SmallVector<StagingOption, 4> options;
    for (auto [decodeA, decodeB] : choices)
      options.push_back(
          {decodeA, decodeB, bytes(aChain, decodeA) + bytes(bChain, decodeB)});
    return options;
  }

  // A candidate dot, its staging options, and the option chosen for it.
  struct DotPlan {
    tt::DotOp dotOp;
    SmallVector<StagingOption, 4> options;
    // Index into `options`, or -1 if the dot is not staged.
    int chosen = -1;
  };

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    unsigned targetSlm = slmSize ? unsigned(slmSize) : kDefaultSlmBytes;
    uint64_t slmCap =
        targetSlm > kSlmMarginBytes ? targetSlm - kSlmMarginBytes : 0;
    // The per-dot limit for staging both operands as they are, which the pass
    // has always used. It is kept as a floor: every dot that fits it is staged
    // as before, whatever the target.
    constexpr uint64_t asIsCap = kDefaultSlmBytes - kSlmMarginBytes;

    // The pass fires on any tt.dot whose result still has BlockedEncoding.
    // On DPAS hardware (PVC, BMG, ARL-H Xe2), AccelerateMatmul rewrites
    // DPAS-lowerable dots to DpasEncoding so the BlockedEncoding check
    // excludes them automatically. f32 dots with `inputPrecision = ieee`
    // are not DPAS-lowerable (DPAS only handles f32 via TF32 truncation),
    // so they retain BlockedEncoding even on DPAS hardware and lower via
    // FMA — exactly the case this pass needs to fire on.
    SmallVector<DotPlan> plans;
    mod.walk([&](tt::DotOp dotOp) {
      if (!isa<ttg::BlockedEncodingAttr>(dotOp.getType().getEncoding()))
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
      if (selectKTile(dotOp) == 0) {
        LDBG("no viable kTile for dot " << dotOp);
        return;
      }
      plans.push_back({dotOp, getStagingOptions(dotOp, *aChain, *bChain)});
    });

    // First stage what fits the per-dot limit as is, as the pass always did.
    // Then upgrade dots, in program order, to a preferred option (a decoded
    // operand, or staging as is beyond the per-dot limit) while all the dots
    // staged in the function still fit in `slmCap` together: a later pass
    // (ReorderInstructions) can move each local_alloc up to its source, so the
    // buffers of different dots may be live at the same time.
    uint64_t total = 0;
    for (DotPlan &plan : plans) {
      const StagingOption &asIs = plan.options.back();
      if (asIs.bytes <= asIsCap) {
        plan.chosen = static_cast<int>(plan.options.size()) - 1;
        total += asIs.bytes;
      }
    }
    for (DotPlan &plan : plans) {
      int current =
          plan.chosen < 0 ? static_cast<int>(plan.options.size()) : plan.chosen;
      uint64_t currentBytes =
          plan.chosen < 0 ? 0 : plan.options[plan.chosen].bytes;
      for (int i = 0; i < current; ++i) {
        uint64_t newTotal = total - currentBytes + plan.options[i].bytes;
        if (newTotal > slmCap) {
          LDBG("staging A " << (plan.options[i].decodeA ? "decoded" : "as is")
                            << ", B "
                            << (plan.options[i].decodeB ? "decoded" : "as is")
                            << " would stage " << newTotal << " B in total");
          continue;
        }
        plan.chosen = i;
        total = newTotal;
        break;
      }
      if (plan.chosen < 0)
        LDBG("dot SLM-staging cost exceeds the limits, skipping");
    }

    // Stage the chosen dots one at a time: staging a dot replaces it, so the
    // operand chain of a later dot that uses its result (e.g. a chain of dots)
    // must be found again afterwards.
    for (DotPlan &plan : plans) {
      if (plan.chosen < 0)
        continue;
      auto aChain = findOperandChain(plan.dotOp.getA());
      auto bChain = findOperandChain(plan.dotOp.getB());
      if (!aChain || !bChain)
        continue;
      const StagingOption &option = plan.options[plan.chosen];
      aChain->useDecoded = option.decodeA && aChain->decodedSource;
      bChain->useDecoded = option.decodeB && bChain->decodedSource;
      (void)stageDot(plan.dotOp, *aChain, *bChain);
    }
  }
};

} // namespace
