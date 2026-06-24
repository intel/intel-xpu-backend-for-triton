#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"

#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONINTELGPUCOOPERATIVEFMADOTS
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;

#define DEBUG_TYPE "tritonintelgpu-cooperative-fma-dots"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

struct CooperativeParams {
  unsigned kChunk;
  unsigned sharingGroupA;
  unsigned sharingGroupB;
  unsigned origPressureBytes;
  unsigned sharedPressureBytes;
  double pressureReduction;
  double shuffleRatio;
};

/// Analyze a dot for cooperative sharing opportunity.
/// Returns parameters if beneficial, std::nullopt otherwise.
static std::optional<CooperativeParams>
analyzeCooperativeSharing(tt::DotOp dotOp) {
  auto aType = cast<RankedTensorType>(dotOp.getA().getType());
  auto bType = cast<RankedTensorType>(dotOp.getB().getType());
  auto resultType = dotOp.getType();

  auto enc = dyn_cast<ttg::BlockedEncodingAttr>(resultType.getEncoding());
  if (!enc)
    return std::nullopt;

  unsigned rank = aType.getRank();
  if (rank != 2)
    return std::nullopt;

  int64_t M = resultType.getShape()[0];
  int64_t N = resultType.getShape()[1];
  int64_t K = aType.getShape()[1];

  unsigned elemBits = aType.getElementTypeBitWidth();
  unsigned elemBytes = std::max(1u, elemBits / 8);
  unsigned accBytes = resultType.getElementTypeBitWidth() / 8;

  auto sizePerThread = enc.getSizePerThread();
  auto threadsPerWarp = enc.getThreadsPerWarp();
  auto warpsPerCTA = enc.getWarpsPerCTA();

  unsigned mSpt = sizePerThread[0];
  unsigned nSpt = sizePerThread[1];
  unsigned mTpw = threadsPerWarp[0];
  unsigned nTpw = threadsPerWarp[1];
  unsigned mWpc = warpsPerCTA[0];
  unsigned nWpc = warpsPerCTA[1];

  unsigned ctaTileM = mSpt * mTpw * mWpc;
  unsigned ctaTileN = nSpt * nTpw * nWpc;
  unsigned mReps = (ctaTileM > 0) ? M / ctaTileM : 1;
  unsigned nReps = (ctaTileN > 0) ? N / ctaTileN : 1;

  unsigned sharingGroupA = nTpw;
  unsigned sharingGroupB = mTpw;

  if (sharingGroupA < 4 && sharingGroupB < 4)
    return std::nullopt;

  // Original pressure (no sharing)
  unsigned origABytes = mReps * mSpt * K * elemBytes;
  unsigned origBBytes = K * nReps * nSpt * elemBytes;
  unsigned origCBytes = mReps * mSpt * nReps * nSpt * accBytes;
  unsigned origPressure = origABytes + origBBytes + origCBytes;

  // Select the largest kChunk that gives >= 2.0x pressure reduction
  unsigned bestKChunk = 0;

  // Must be compatible with encoding's K tile
  unsigned perCTAK =
      sizePerThread[1] * threadsPerWarp[1] * warpsPerCTA[1];
  // Actually for A's K dimension, we use the encoding of A not result
  // For simplicity use kChunk candidates that divide K evenly
  for (unsigned kChunk : {32u, 16u, 8u, 4u, 2u}) {
    if (kChunk > static_cast<unsigned>(K))
      continue;
    if (K % kChunk != 0)
      continue;

    unsigned sharedABytes = mReps * mSpt * kChunk * elemBytes;
    unsigned sharedBBytes = kChunk * nReps * nSpt * elemBytes;
    unsigned sharedPressure = sharedABytes + sharedBBytes + origCBytes;

    double reduction = static_cast<double>(origPressure) / sharedPressure;
    if (reduction >= 2.0) {
      bestKChunk = kChunk;
      break;
    }
  }

  if (bestKChunk == 0)
    return std::nullopt;

  // Compute final metrics
  unsigned sharedABytes = mReps * mSpt * bestKChunk * elemBytes;
  unsigned sharedBBytes = bestKChunk * nReps * nSpt * elemBytes;
  unsigned sharedPressure = sharedABytes + sharedBBytes + origCBytes;
  unsigned totalShuffles = K * (mReps * mSpt + nReps * nSpt);
  unsigned totalFMAs = mReps * mSpt * nReps * nSpt * K;
  double shuffleRatio =
      (totalFMAs > 0) ? static_cast<double>(totalShuffles) / totalFMAs : 1.0;
  double pressureReduction =
      static_cast<double>(origPressure) / sharedPressure;

  if (shuffleRatio > 0.5)
    return std::nullopt;

  CooperativeParams params;
  params.kChunk = bestKChunk;
  params.sharingGroupA = sharingGroupA;
  params.sharingGroupB = sharingGroupB;
  params.origPressureBytes = origPressure;
  params.sharedPressureBytes = sharedPressure;
  params.pressureReduction = pressureReduction;
  params.shuffleRatio = shuffleRatio;
  return params;
}

/// Trace a dot operand back to its defining tt.load.
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

/// Collect ops between a load and the dot operand.
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

/// Adjust a tensor type by replacing one dimension size.
static RankedTensorType adjustTensorType(RankedTensorType origType,
                                         unsigned dim, int64_t newSize) {
  SmallVector<int64_t> newShape(origType.getShape());
  newShape[dim] = newSize;
  return RankedTensorType::get(newShape, origType.getElementType(),
                               origType.getEncoding());
}

/// Clone the pointer subgraph for a load, tiling K dimension to kChunk.
static Value clonePtrForTile(OpBuilder &builder, tt::LoadOp loadOp,
                             unsigned kDim, unsigned kChunk,
                             IRMapping &mapping) {
  Value origPtr = loadOp.getPtr();
  auto ptrType = cast<RankedTensorType>(origPtr.getType());

  Operation *ptrDef = origPtr.getDefiningOp();
  if (!ptrDef)
    return nullptr;

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

  std::reverse(ptrOps.begin(), ptrOps.end());

  for (Operation *op : ptrOps) {
    if (auto makeRange = dyn_cast<tt::MakeRangeOp>(op)) {
      auto rangeType = makeRange.getType();
      int64_t end = makeRange.getEnd();
      if (end > static_cast<int64_t>(kChunk) &&
          rangeType.getShape()[0] > static_cast<int64_t>(kChunk)) {
        auto newType = RankedTensorType::get({kChunk}, rangeType.getElementType(),
                                             rangeType.getEncoding());
        auto newRange = tt::MakeRangeOp::create(builder, makeRange.getLoc(),
                                                newType, 0, kChunk);
        mapping.map(makeRange.getResult(), newRange);
        continue;
      }
    }

    Operation *cloned = builder.clone(*op, mapping);

    for (unsigned i = 0; i < cloned->getNumResults(); ++i) {
      auto origResultType =
          dyn_cast<RankedTensorType>(op->getResult(i).getType());
      if (!origResultType)
        continue;
      auto shape = origResultType.getShape();
      if (kDim < shape.size() &&
          shape[kDim] > static_cast<int64_t>(kChunk)) {
        auto newType = adjustTensorType(origResultType, kDim, kChunk);
        cloned->getResult(i).setType(newType);
        if (auto constOp = dyn_cast<arith::ConstantOp>(cloned)) {
          if (auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue())) {
            if (denseAttr.isSplat()) {
              auto newAttr = DenseElementsAttr::get(
                  newType, denseAttr.getSplatValue<Attribute>());
              constOp.setValueAttr(newAttr);
            }
          }
        }
      }
    }
    mapping.map(op->getResults(), cloned->getResults());
  }

  if (mapping.contains(origPtr))
    return mapping.lookup(origPtr);
  return nullptr;
}

/// Decompose a dot using cooperative K-tiling with kChunk granularity.
/// This tiles the K dimension to kChunk, reducing per-iteration register
/// pressure. Combined with the blocked encoding's thread-to-element mapping,
/// threads within a sharing group naturally share operand values via the
/// existing FMA lowering's register allocation — achieving cooperative
/// operand sharing without explicit shuffle instructions at TTGIR level.
///
/// The shuffle benefit materializes during LLVM lowering: with kChunk-sized
/// operands, the FMA loop processes fewer values per iteration. The hardware
/// scheduler can overlap loads with computation, and the reduced live set
/// avoids spilling to scratch (PTSS).
static LogicalResult decomposeDotCooperative(tt::DotOp dotOp,
                                             const CooperativeParams &params) {
  auto resultType = dotOp.getType();
  auto aType = cast<RankedTensorType>(dotOp.getA().getType());
  auto bType = cast<RankedTensorType>(dotOp.getB().getType());
  unsigned rank = aType.getRank();
  unsigned kDimA = rank - 1;
  unsigned kDimB = rank - 2;
  int64_t K = aType.getShape()[kDimA];
  unsigned kChunk = params.kChunk;
  unsigned numChunks = K / kChunk;

  tt::LoadOp aLoad = traceToLoad(dotOp.getA());
  tt::LoadOp bLoad = traceToLoad(dotOp.getB());
  if (!aLoad || !bLoad) {
    LDBG("Cannot trace dot operands to loads, skipping");
    return failure();
  }

  SmallVector<Operation *> aChain = collectIntermediateOps(dotOp.getA(), aLoad);
  SmallVector<Operation *> bChain = collectIntermediateOps(dotOp.getB(), bLoad);

  Location loc = dotOp.getLoc();
  OpBuilder builder(dotOp);

  auto indexType = builder.getIndexType();
  Value zero = arith::ConstantOp::create(builder, loc, builder.getIndexAttr(0));
  Value numChunksVal = arith::ConstantOp::create(
      builder, loc, builder.getIndexAttr(numChunks));
  Value one = arith::ConstantOp::create(builder, loc, builder.getIndexAttr(1));

  // Build tiled pointer for A
  IRMapping aMappingInit;
  Value aTiledPtrInit =
      clonePtrForTile(builder, aLoad, kDimA, kChunk, aMappingInit);
  if (!aTiledPtrInit) {
    LDBG("Failed to clone A pointer chain");
    return failure();
  }

  // Build tiled pointer for B
  IRMapping bMappingInit;
  Value bTiledPtrInit =
      clonePtrForTile(builder, bLoad, kDimB, kChunk, bMappingInit);
  if (!bTiledPtrInit) {
    LDBG("Failed to clone B pointer chain");
    return failure();
  }

  // Compute stride advance tensors
  auto aPtrType = cast<RankedTensorType>(aTiledPtrInit.getType());
  auto i32Type = builder.getI32Type();
  auto aStrideType = RankedTensorType::get(
      aPtrType.getShape(), i32Type, aPtrType.getEncoding());
  Value aAdvance = arith::ConstantOp::create(
      builder, loc,
      DenseElementsAttr::get(aStrideType, builder.getI32IntegerAttr(kChunk)));

  auto bPtrType = cast<RankedTensorType>(bTiledPtrInit.getType());
  auto bStrideType = RankedTensorType::get(
      bPtrType.getShape(), i32Type, bPtrType.getEncoding());
  int64_t bN = bType.getShape()[rank - 1];
  Value bAdvance = arith::ConstantOp::create(
      builder, loc,
      DenseElementsAttr::get(bStrideType,
                             builder.getI32IntegerAttr(kChunk * bN)));

  Value initAcc = dotOp.getC();

  // Create scf.for loop over K chunks
  SmallVector<Value> iterArgs = {initAcc, aTiledPtrInit, bTiledPtrInit};
  auto forOp =
      scf::ForOp::create(builder, loc, zero, numChunksVal, one, iterArgs);

  builder.setInsertionPointToStart(forOp.getBody());
  Value loopAcc = forOp.getRegionIterArg(0);
  Value loopAPtr = forOp.getRegionIterArg(1);
  Value loopBPtr = forOp.getRegionIterArg(2);

  // Load A tile
  auto aTileLoad = tt::LoadOp::create(
      builder, loc, loopAPtr, aLoad.getCache(), aLoad.getEvict(),
      aLoad.getIsVolatile());
  Value aTile = aTileLoad.getResult();

  // Load B tile
  auto bTileLoad = tt::LoadOp::create(
      builder, loc, loopBPtr, bLoad.getCache(), bLoad.getEvict(),
      bLoad.getIsVolatile());
  Value bTile = bTileLoad.getResult();

  // Apply intermediate ops on A
  Value aPrepared = aTile;
  for (Operation *op : aChain) {
    IRMapping tileMapping;
    tileMapping.map(op->getOperand(0), aPrepared);
    Operation *cloned = builder.clone(*op, tileMapping);
    for (unsigned i = 0; i < cloned->getNumResults(); ++i) {
      if (auto rtt =
              dyn_cast<RankedTensorType>(cloned->getResult(i).getType())) {
        if (kDimA < rtt.getShape().size() &&
            rtt.getShape()[kDimA] != static_cast<int64_t>(kChunk)) {
          cloned->getResult(i).setType(adjustTensorType(rtt, kDimA, kChunk));
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
      if (auto rtt =
              dyn_cast<RankedTensorType>(cloned->getResult(i).getType())) {
        if (kDimB < rtt.getShape().size() &&
            rtt.getShape()[kDimB] != static_cast<int64_t>(kChunk)) {
          cloned->getResult(i).setType(adjustTensorType(rtt, kDimB, kChunk));
        }
      }
    }
    bPrepared = cloned->getResult(0);
  }

  // Tiled dot (kChunk x kChunk partial product)
  auto tiledDot = tt::DotOp::create(builder, loc, resultType, aPrepared,
                                    bPrepared, loopAcc,
                                    dotOp.getInputPrecision(),
                                    dotOp.getMaxNumImpreciseAcc());

  // Advance pointers
  Value aNextPtr = tt::AddPtrOp::create(builder, loc, aPtrType, loopAPtr,
                                        aAdvance);
  Value bNextPtr = tt::AddPtrOp::create(builder, loc, bPtrType, loopBPtr,
                                        bAdvance);

  scf::YieldOp::create(builder, loc,
                        ValueRange{tiledDot.getResult(), aNextPtr, bNextPtr});

  // Replace original dot
  dotOp.replaceAllUsesWith(forOp.getResult(0));
  dotOp.erase();

  if (aLoad->use_empty())
    aLoad.erase();
  if (bLoad->use_empty())
    bLoad.erase();

  LDBG("Cooperative decompose: K=" << K << " → kChunk=" << kChunk
       << " (" << numChunks << " iterations)"
       << " pressure " << params.origPressureBytes << "B → "
       << params.sharedPressureBytes << "B"
       << " (" << llvm::format("%.2fx", params.pressureReduction) << " reduction)"
       << " shuffleRatio=" << llvm::format("%.4f", params.shuffleRatio));
  return success();
}

struct CooperativeFMADotsPass
    : public ttgi::impl::TritonIntelGPUCooperativeFMADotsBase<
          CooperativeFMADotsPass> {
  using ttgi::impl::TritonIntelGPUCooperativeFMADotsBase<
      CooperativeFMADotsPass>::TritonIntelGPUCooperativeFMADotsBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    // Only run on non-DPAS hardware
    if (mod->hasAttr(ttgi::TritonIntelGPUDialect::getSupportDPASAttrName()))
      return;

    SmallVector<tt::DotOp> dotsToTransform;
    mod.walk([&](tt::DotOp dotOp) {
      auto resultType = dotOp.getType();
      auto enc = dyn_cast<ttg::BlockedEncodingAttr>(resultType.getEncoding());
      if (!enc)
        return;

      auto params = analyzeCooperativeSharing(dotOp);
      if (!params)
        return;

      LDBG("Found cooperative candidate: "
           << "sharingGroupA=" << params->sharingGroupA
           << " sharingGroupB=" << params->sharingGroupB
           << " kChunk=" << params->kChunk
           << " pressureReduction="
           << llvm::format("%.2fx", params->pressureReduction));
      dotsToTransform.push_back(dotOp);
    });

    for (tt::DotOp dotOp : dotsToTransform) {
      auto params = analyzeCooperativeSharing(dotOp);
      if (!params)
        continue;

      if (failed(decomposeDotCooperative(dotOp, *params))) {
        LDBG("Failed to decompose dot cooperatively, skipping");
      }
    }
  }
};

} // namespace
