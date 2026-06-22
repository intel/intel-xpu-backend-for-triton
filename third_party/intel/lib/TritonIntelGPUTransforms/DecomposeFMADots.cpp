#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/Support/Debug.h"

#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONINTELGPUDECOMPOSEFMADOTS
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;

#define DEBUG_TYPE "tritonintelgpu-decompose-fma-dots"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

constexpr unsigned K_PRESSURE_THRESHOLD_BYTES = 4096;

/// Estimate per-thread register pressure from a dot with BlockedEncoding.
/// Returns the approximate bytes each thread needs live for the K-loop.
static unsigned estimatePerThreadBytes(tt::DotOp dotOp) {
  auto aType = cast<RankedTensorType>(dotOp.getA().getType());
  auto bType = cast<RankedTensorType>(dotOp.getB().getType());
  auto resultType = dotOp.getType();

  auto enc = dyn_cast<ttg::BlockedEncodingAttr>(resultType.getEncoding());
  if (!enc)
    return 0;

  unsigned rank = aType.getRank();
  unsigned kDimA = rank - 1;
  unsigned kDimB = rank - 2;

  int64_t K = aType.getShape()[kDimA];
  unsigned elemBits = aType.getElementTypeBitWidth();
  unsigned elemBytes = std::max(1u, elemBits / 8);

  auto sizePerThread = enc.getSizePerThread();
  unsigned mSpt = sizePerThread[rank - 2];
  unsigned nSpt = sizePerThread[rank - 1];

  return (mSpt * K + K * nSpt) * elemBytes;
}

/// Select a K tile size that is compatible with the encoding.
/// Returns 0 if tiling is not possible.
static unsigned selectKTile(unsigned K, ttg::BlockedEncodingAttr enc,
                            unsigned kDim) {
  auto sizePerThread = enc.getSizePerThread();
  auto threadsPerWarp = enc.getThreadsPerWarp();
  auto warpsPerCTA = enc.getWarpsPerCTA();

  unsigned perCTAK =
      sizePerThread[kDim] * threadsPerWarp[kDim] * warpsPerCTA[kDim];

  for (unsigned tile : {32u, 64u, 16u}) {
    if (tile < K && K % tile == 0 && tile % perCTAK == 0)
      return tile;
  }
  return 0;
}

/// Trace a dot operand back to its defining tt.load, through optional
/// intermediate ops (arith.extf, ttg.convert_layout). Returns nullptr if
/// the operand cannot be traced to a load.
static tt::LoadOp traceToLoad(Value operand) {
  Value current = operand;
  for (unsigned depth = 0; depth < 8; ++depth) {
    if (!current.getDefiningOp())
      return nullptr;
    if (auto loadOp = dyn_cast<tt::LoadOp>(current.getDefiningOp()))
      return loadOp;
    if (auto extOp = dyn_cast<arith::ExtFOp>(current.getDefiningOp())) {
      current = extOp.getIn();
      continue;
    }
    if (auto cvtOp = dyn_cast<ttg::ConvertLayoutOp>(current.getDefiningOp())) {
      current = cvtOp.getSrc();
      continue;
    }
    if (auto truncOp = dyn_cast<arith::TruncFOp>(current.getDefiningOp())) {
      current = truncOp.getIn();
      continue;
    }
    return nullptr;
  }
  return nullptr;
}

/// Collect ops between a load and the dot operand (the intermediate chain).
/// Returns them in order from load output to dot input.
static SmallVector<Operation *> collectIntermediateOps(Value dotOperand,
                                                      tt::LoadOp loadOp) {
  SmallVector<Operation *> chain;
  Value current = dotOperand;
  while (current.getDefiningOp() != loadOp.getOperation()) {
    chain.push_back(current.getDefiningOp());
    if (auto extOp = dyn_cast<arith::ExtFOp>(current.getDefiningOp()))
      current = extOp.getIn();
    else if (auto cvtOp =
                 dyn_cast<ttg::ConvertLayoutOp>(current.getDefiningOp()))
      current = cvtOp.getSrc();
    else if (auto truncOp = dyn_cast<arith::TruncFOp>(current.getDefiningOp()))
      current = truncOp.getIn();
    else
      break;
  }
  std::reverse(chain.begin(), chain.end());
  return chain;
}

/// Adjust tensor types in a shape by replacing one dimension.
static RankedTensorType adjustTensorType(RankedTensorType origType,
                                         unsigned dim, int64_t newSize) {
  SmallVector<int64_t> newShape(origType.getShape());
  newShape[dim] = newSize;
  return RankedTensorType::get(newShape, origType.getElementType(),
                               origType.getEncoding());
}

/// Clone the pointer subgraph for a load, adjusting K dimension from full K
/// to K_TILE. Returns the new pointer value for the tiled load.
/// Also produces an "advance" value that can be added per iteration.
static Value clonePtrForTile(OpBuilder &builder, tt::LoadOp loadOp,
                             unsigned kDim, unsigned kTile,
                             IRMapping &mapping) {
  Value origPtr = loadOp.getPtr();
  auto ptrType = cast<RankedTensorType>(origPtr.getType());
  RankedTensorType tiledPtrType = adjustTensorType(ptrType, kDim, kTile);

  Operation *ptrDef = origPtr.getDefiningOp();
  if (!ptrDef) {
    return nullptr;
  }

  // Collect the backward slice of ops that define the pointer
  SmallVector<Operation *> ptrOps;
  SmallVector<Operation *> worklist = {ptrDef};
  DenseSet<Operation *> visited;
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (!op || visited.contains(op))
      continue;
    visited.insert(op);
    ptrOps.push_back(op);
    for (Value operand : op->getOperands()) {
      if (Operation *defOp = operand.getDefiningOp()) {
        if (isa<tt::MakeRangeOp, tt::SplatOp, tt::ExpandDimsOp,
                tt::BroadcastOp, tt::AddPtrOp, arith::MulIOp, arith::AddIOp,
                arith::ConstantOp>(defOp)) {
          worklist.push_back(defOp);
        }
      }
    }
  }

  // Sort by topological order (reverse of collection order, approximately)
  std::reverse(ptrOps.begin(), ptrOps.end());

  // Clone each op, adjusting shapes and make_range as needed
  for (Operation *op : ptrOps) {
    if (auto makeRange = dyn_cast<tt::MakeRangeOp>(op)) {
      auto rangeType = makeRange.getType();
      int64_t end = makeRange.getEnd();
      // If this make_range produces the K-dimension range, tile it
      if (end > static_cast<int64_t>(kTile) &&
          rangeType.getShape()[0] > static_cast<int64_t>(kTile)) {
        auto newType = RankedTensorType::get({kTile}, rangeType.getElementType(),
                                             rangeType.getEncoding());
        auto newRange = tt::MakeRangeOp::create(builder, makeRange.getLoc(),
                                                newType, 0, kTile);
        mapping.map(makeRange.getResult(), newRange);
        continue;
      }
    }

    // Clone op with remapped operands and adjust result types
    Operation *cloned = builder.clone(*op, mapping);

    // Adjust result types for ops that produce pointer/tensor types
    for (unsigned i = 0; i < cloned->getNumResults(); ++i) {
      auto origResultType =
          dyn_cast<RankedTensorType>(op->getResult(i).getType());
      if (!origResultType)
        continue;
      auto shape = origResultType.getShape();
      if (kDim < shape.size() &&
          shape[kDim] > static_cast<int64_t>(kTile)) {
        auto newType = adjustTensorType(origResultType, kDim, kTile);
        cloned->getResult(i).setType(newType);
      }
    }
    mapping.map(op->getResults(), cloned->getResults());
  }

  if (mapping.contains(origPtr))
    return mapping.lookup(origPtr);
  return nullptr;
}

/// Decompose a single dot op into a K-tiled scf.for loop.
static LogicalResult decomposeDot(tt::DotOp dotOp, unsigned kTile) {
  auto resultType = dotOp.getType();
  auto aType = cast<RankedTensorType>(dotOp.getA().getType());
  auto bType = cast<RankedTensorType>(dotOp.getB().getType());
  unsigned rank = aType.getRank();
  unsigned kDimA = rank - 1; // K is last dim of A
  unsigned kDimB = rank - 2; // K is second-to-last dim of B
  int64_t K = aType.getShape()[kDimA];
  unsigned numTiles = K / kTile;

  // Trace operands to loads
  tt::LoadOp aLoad = traceToLoad(dotOp.getA());
  tt::LoadOp bLoad = traceToLoad(dotOp.getB());
  if (!aLoad || !bLoad) {
    LDBG("Cannot trace dot operands to loads, skipping");
    return failure();
  }

  // Collect intermediate op chains
  SmallVector<Operation *> aChain = collectIntermediateOps(dotOp.getA(), aLoad);
  SmallVector<Operation *> bChain = collectIntermediateOps(dotOp.getB(), bLoad);

  Location loc = dotOp.getLoc();
  OpBuilder builder(dotOp);

  // Create loop bounds
  auto indexType = builder.getIndexType();
  Value zero = arith::ConstantOp::create(builder, loc, builder.getIndexAttr(0));
  Value numTilesVal = arith::ConstantOp::create(
      builder, loc, builder.getIndexAttr(numTiles));
  Value one = arith::ConstantOp::create(builder, loc, builder.getIndexAttr(1));

  // Build tiled pointer for A (initial tile)
  IRMapping aMappingInit;
  Value aTiledPtrInit = clonePtrForTile(builder, aLoad, kDimA, kTile, aMappingInit);
  if (!aTiledPtrInit) {
    LDBG("Failed to clone A pointer chain");
    return failure();
  }

  // Build tiled pointer for B (initial tile)
  IRMapping bMappingInit;
  Value bTiledPtrInit = clonePtrForTile(builder, bLoad, kDimB, kTile, bMappingInit);
  if (!bTiledPtrInit) {
    LDBG("Failed to clone B pointer chain");
    return failure();
  }

  // Compute stride advance tensors for A and B
  // A advances by kTile elements along kDimA
  auto aPtrType = cast<RankedTensorType>(aTiledPtrInit.getType());
  auto i32Type = builder.getI32Type();
  auto aStrideType = RankedTensorType::get(
      aPtrType.getShape(), i32Type, aPtrType.getEncoding());
  Value aKTileConst = arith::ConstantOp::create(
      builder, loc,
      DenseElementsAttr::get(aStrideType, builder.getI32IntegerAttr(kTile)));

  auto bPtrType = cast<RankedTensorType>(bTiledPtrInit.getType());
  auto bStrideType = RankedTensorType::get(
      bPtrType.getShape(), i32Type, bPtrType.getEncoding());
  // For B, K is along rows (dim kDimB), stride = kTile * N for row-stepping
  // But since pointer tensors already have element-level addressing,
  // each B pointer element at row k points to B[k, n], so advancing K
  // means adding kTile * N to each pointer element... Actually, for
  // contiguous pointers generated via make_range, the stride is already
  // baked into the pointer values. We need to add kTile * b_stride_k to
  // each pointer element.
  // Simpler: advance by kTile * number_of_columns for B's pointer tensor
  int64_t bN = bType.getShape()[rank - 1]; // N dimension of B
  Value bKTileConst = arith::ConstantOp::create(
      builder, loc,
      DenseElementsAttr::get(bStrideType,
                             builder.getI32IntegerAttr(kTile * bN)));

  // The accumulator is the dot's C operand
  Value initAcc = dotOp.getC();

  // Create the scf.for loop
  SmallVector<Value> iterArgs = {initAcc, aTiledPtrInit, bTiledPtrInit};
  auto forOp =
      scf::ForOp::create(builder, loc, zero, numTilesVal, one, iterArgs);

  // Build loop body
  builder.setInsertionPointToStart(forOp.getBody());
  Value loopAcc = forOp.getRegionIterArg(0);
  Value loopAPtr = forOp.getRegionIterArg(1);
  Value loopBPtr = forOp.getRegionIterArg(2);

  // Load A tile
  auto aTileType = adjustTensorType(aType, kDimA, kTile);
  Value aTile = tt::LoadOp::create(builder, loc, aTileType, loopAPtr);

  // Load B tile
  auto bTileType = adjustTensorType(bType, kDimB, kTile);
  Value bTile = tt::LoadOp::create(builder, loc, bTileType, loopBPtr);

  // Apply intermediate ops on A (extf, convert_layout, etc.)
  Value aPrepared = aTile;
  for (Operation *op : aChain) {
    IRMapping tileMapping;
    tileMapping.map(op->getOperand(0), aPrepared);
    Operation *cloned = builder.clone(*op, tileMapping);
    // Adjust result type for K dimension
    for (unsigned i = 0; i < cloned->getNumResults(); ++i) {
      if (auto rtt = dyn_cast<RankedTensorType>(cloned->getResult(i).getType())) {
        if (kDimA < rtt.getShape().size() &&
            rtt.getShape()[kDimA] != static_cast<int64_t>(kTile)) {
          cloned->getResult(i).setType(adjustTensorType(rtt, kDimA, kTile));
        }
      }
    }
    aPrepared = cloned->getResult(0);
  }

  // Apply intermediate ops on B
  Value bPrepared = bTile;
  for (Operation *op : bChain) {
    IRMapping tileMapping;
    tileMapping.map(op->getOperand(0), bPrepared);
    Operation *cloned = builder.clone(*op, tileMapping);
    for (unsigned i = 0; i < cloned->getNumResults(); ++i) {
      if (auto rtt = dyn_cast<RankedTensorType>(cloned->getResult(i).getType())) {
        if (kDimB < rtt.getShape().size() &&
            rtt.getShape()[kDimB] != static_cast<int64_t>(kTile)) {
          cloned->getResult(i).setType(adjustTensorType(rtt, kDimB, kTile));
        }
      }
    }
    bPrepared = cloned->getResult(0);
  }

  // Create tiled dot
  auto tiledDot = tt::DotOp::create(builder, loc, resultType, aPrepared,
                                    bPrepared, loopAcc,
                                    dotOp.getInputPrecision(),
                                    dotOp.getMaxNumImpreciseAcc());

  // Advance pointers
  Value aNextPtr = tt::AddPtrOp::create(builder, loc, aPtrType, loopAPtr,
                                        aKTileConst);
  Value bNextPtr = tt::AddPtrOp::create(builder, loc, bPtrType, loopBPtr,
                                        bKTileConst);

  // Yield
  scf::YieldOp::create(builder, loc,
                        ValueRange{tiledDot.getResult(), aNextPtr, bNextPtr});

  // Replace original dot with loop result
  dotOp.replaceAllUsesWith(forOp.getResult(0));
  dotOp.erase();

  // Clean up dead original loads if they have no other users
  if (aLoad->use_empty())
    aLoad.erase();
  if (bLoad->use_empty())
    bLoad.erase();

  LDBG("Decomposed dot [K=" << K << "] into " << numTiles << " tiles of "
                             << kTile);
  return success();
}

struct DecomposeFMADotsPass
    : public ttgi::impl::TritonIntelGPUDecomposeFMADotsBase<
          DecomposeFMADotsPass> {
  using ttgi::impl::TritonIntelGPUDecomposeFMADotsBase<
      DecomposeFMADotsPass>::TritonIntelGPUDecomposeFMADotsBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    // Only run on non-DPAS hardware
    if (mod->hasAttr(ttgi::TritonIntelGPUDialect::getSupportDPASAttrName()))
      return;

    SmallVector<tt::DotOp> dotsToDecompose;
    mod.walk([&](tt::DotOp dotOp) {
      auto resultType = dotOp.getType();
      auto enc = dyn_cast<ttg::BlockedEncodingAttr>(resultType.getEncoding());
      if (!enc)
        return;

      unsigned perThreadBytes = estimatePerThreadBytes(dotOp);
      if (perThreadBytes <= K_PRESSURE_THRESHOLD_BYTES)
        return;

      auto aType = cast<RankedTensorType>(dotOp.getA().getType());
      unsigned rank = aType.getRank();
      unsigned kDimA = rank - 1;
      int64_t K = aType.getShape()[kDimA];

      unsigned kTile = selectKTile(K, enc, kDimA);
      if (kTile == 0)
        return;

      LDBG("Found decomposable dot: K=" << K << " → kTile=" << kTile
                                         << " perThreadBytes="
                                         << perThreadBytes);
      dotsToDecompose.push_back(dotOp);
    });

    for (tt::DotOp dotOp : dotsToDecompose) {
      auto aType = cast<RankedTensorType>(dotOp.getA().getType());
      auto enc =
          cast<ttg::BlockedEncodingAttr>(dotOp.getType().getEncoding());
      unsigned rank = aType.getRank();
      unsigned kDimA = rank - 1;
      int64_t K = aType.getShape()[kDimA];
      unsigned kTile = selectKTile(K, enc, kDimA);

      if (failed(decomposeDot(dotOp, kTile))) {
        LDBG("Failed to decompose dot, skipping");
      }
    }
  }
};

} // namespace
