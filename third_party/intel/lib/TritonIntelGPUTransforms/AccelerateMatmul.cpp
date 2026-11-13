#include "Dialect/TritonIntelGPU/IR/Attributes.h"
#include "Dialect/TritonIntelGPU/Transforms/DecomposeScaledBlocked.h"
#include "Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "intel/include/Analysis/DPAS.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"

#include "Dialect/TritonIntelGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <optional>
#include <utility>

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONINTELGPUACCELERATEMATMUL
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"

} // namespace mlir::triton::gpu::intel

namespace {

#define DEBUG_TYPE "tritonintelgpu-accelerate-matmul"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << X << "\n")

// FIXME: Remove once IGC can split large 2D block loads.
static void setAttrOnBOperand(Operation *op, StringRef attrName,
                              Attribute attr) {
  Operation *defOp;
  llvm::TypeSwitch<Operation *>(op).Case<tt::DotOp, tt::DotScaledOp>(
      [&](auto op) { defOp = op.getB().getDefiningOp(); });
  while (auto convOp = dyn_cast_or_null<ttg::ConvertLayoutOp>(defOp))
    defOp = convOp.getSrc().getDefiningOp();
  if (auto transOp = dyn_cast_or_null<tt::TransOp>(defOp))
    defOp = transOp.getOperand().getDefiningOp();
  if (auto loadOp = dyn_cast_or_null<tt::LoadOp>(defOp))
    loadOp->setAttr(attrName, attr);
  if (auto descLoadOp = dyn_cast_or_null<tt::DescriptorLoadOp>(defOp))
    descLoadOp->setAttr(attrName, attr);
}

unsigned getOpsPerChannel(Type elemType, ModuleOp m) {
  assert((elemType.isIntOrFloat() || isa<tt::PackedFloat4E2M1Type>(elemType)) &&
         "unsupported type for DpasEncodingAttr");

  unsigned dpasElemBitWidths = getIntOrFloatOrPtrBitWidth(elemType);
  bool supportsFP8 =
      m->hasAttr(ttgi::TritonIntelGPUDialect::getSupportDPASWithBF8AttrName());
  if (!supportsFP8 && llvm::isa<Float8E5M2Type, Float8E4M3FNType>(elemType))
    dpasElemBitWidths *= 2; // We are upcasting FP8 to FP16.

  return ttgi::DpasEncodingAttr::DPASCapability::opsChanBitWidths /
         dpasElemBitWidths;
}

SetVector<Operation *> getChainedDotOps(Value operand) {
  auto filter = [&operand](Operation *op) {
    return op->getParentRegion() == operand.getParentRegion();
  };
  SetVector<Operation *> slice;
  if (getBackwardSlice(operand, &slice, {filter}).succeeded()) {
    SetVector<Operation *> dotOps;
    for (Operation *op : slice) {
      if (isa<tt::DotOp, tt::DotScaledOp>(op)) {
        dotOps.insert(op);
      }
    }
    return dotOps;
  }
  LDBG("getBackwardSlice failed for operand: " << operand);
  return {};
}

std::pair<Value, Value> getDotOperands(Operation *dotOp) {
  return llvm::TypeSwitch<Operation *, std::pair<Value, Value>>(dotOp)
      .Case<tt::DotOp, tt::DotScaledOp>(
          [](auto op) { return std::make_pair(op.getA(), op.getB()); })
      .Default([](Operation *) { return std::pair<Value, Value>{}; });
}

enum class ChainedDotKind {
  // Chained dots are classified per-dot to bias warp placement:
  // - ChainedAlongA: prioritize M warps to reduce A-path convert_layout costs.
  // - ChainedAlongB: prioritize N warps to reduce B-path convert_layout costs.
  // - ChainedMixedAB: dot participates on both A/B chains; use default tiling.
  NoChained,
  ChainedAlongA,
  ChainedAlongB,
  ChainedMixedAB,
};

bool mergeChainedDotFlags(ChainedDotKind chainedDotKind, bool &chainedDotA,
                          bool &chainedDotB) {
  switch (chainedDotKind) {
  case ChainedDotKind::ChainedMixedAB:
    return true;
  case ChainedDotKind::ChainedAlongA:
    chainedDotA = true;
    break;
  case ChainedDotKind::ChainedAlongB:
    chainedDotB = true;
    break;
  case ChainedDotKind::NoChained:
    break;
  }
  return false;
}

static unsigned clampToPowerOfTwo(unsigned value) {
  // DPAS layout construction expects power-of-two warp counts per dimension.
  return value == 0 ? 1 : llvm::bit_floor(value);
}

using ChainedDotKindMap = llvm::DenseMap<Operation *, ChainedDotKind>;

ChainedDotKind computeChainedDotKindFromSlices(Operation *dotOp,
                                               ChainedDotKindMap &cache);

ChainedDotKind computeChainedDotKind(Operation *dotOp,
                                     ChainedDotKindMap &cache);

void setChainedDotKindAttr(Operation *op, ChainedDotKind kind) {
  op->setAttr(ttgi::TritonIntelGPUDialect::getChainedDotKindAttrName(),
              IntegerAttr::get(IntegerType::get(op->getContext(), 32),
                               static_cast<int32_t>(kind)));
}

std::optional<ChainedDotKind> getChainedDotKindAttr(Operation *op) {
  auto kindAttr = dyn_cast_or_null<IntegerAttr>(
      op->getAttr(ttgi::TritonIntelGPUDialect::getChainedDotKindAttrName()));
  if (!kindAttr)
    return std::nullopt;
  int64_t raw = kindAttr.getInt();
  if (raw < static_cast<int64_t>(ChainedDotKind::NoChained) ||
      raw > static_cast<int64_t>(ChainedDotKind::ChainedMixedAB))
    return std::nullopt;
  return static_cast<ChainedDotKind>(raw);
}

ChainedDotKind computeChainedDotKind(Operation *dotOp,
                                     ChainedDotKindMap &cache) {
  if (!dotOp || !isa<tt::DotOp, tt::DotScaledOp>(dotOp))
    return ChainedDotKind::NoChained;
  if (auto it = cache.find(dotOp); it != cache.end())
    return it->second;

  ChainedDotKind dotChainedType = computeChainedDotKindFromSlices(dotOp, cache);
  // Recursion computeChainedDotKind -> computeChainedDotKindFromSlices ->
  // computeChainedDotKind is bounded by the backward-slice. No re-entering of
  // DotOp. DotOp output cannot be used by same DotOp.
  cache.try_emplace(dotOp, dotChainedType);
  return dotChainedType;
}

ChainedDotKind computeChainedDotKindFromSlices(Operation *dotOp,
                                               ChainedDotKindMap &cache) {
  auto filter = [&dotOp](Operation *op) {
    return op->getParentRegion() == dotOp->getParentRegion();
  };

  auto [operandA, operandB] = getDotOperands(dotOp);

  auto chainedDotOpsA = getChainedDotOps(operandA);
  bool chainedDotA = !chainedDotOpsA.empty();
  auto chainedDotOpsB = getChainedDotOps(operandB);
  bool chainedDotB = !chainedDotOpsB.empty();
  if (!(chainedDotA && chainedDotB)) {
    // Not mixed yet: merge chained kinds from backward-slice DotOps.
    // Termination relies on backward-slice acyclicity.
    auto updateChainedDotFlags = [&](auto &chainedDotOps) {
      for (Operation *op : chainedDotOps) {
        if (mergeChainedDotFlags(computeChainedDotKind(op, cache), chainedDotA,
                                 chainedDotB))
          return true;
      }
      return false;
    };

    if (updateChainedDotFlags(chainedDotOpsA) ||
        updateChainedDotFlags(chainedDotOpsB))
      return ChainedDotKind::ChainedMixedAB;

    SetVector<Operation *> forwardSlice;
    getForwardSlice(dotOp, &forwardSlice, {filter});
    for (Operation *forwardOp : forwardSlice) {
      if (!isa<tt::DotOp, tt::DotScaledOp>(forwardOp))
        continue;

      auto [forwardOperandA, forwardOperandB] = getDotOperands(forwardOp);
      auto forwardChainedDotOpsA = getChainedDotOps(forwardOperandA);
      chainedDotA |= forwardChainedDotOpsA.contains(dotOp);
      auto forwardChainedDotOpsB = getChainedDotOps(forwardOperandB);
      chainedDotB |= forwardChainedDotOpsB.contains(dotOp);
      if (chainedDotA && chainedDotB)
        break;
    }
  }

  if (chainedDotA && chainedDotB)
    return ChainedDotKind::ChainedMixedAB;
  if (chainedDotA)
    return ChainedDotKind::ChainedAlongA;
  if (chainedDotB)
    return ChainedDotKind::ChainedAlongB;
  return ChainedDotKind::NoChained;
}

void annotateChainedDotKinds(ModuleOp mod) {
  ChainedDotKindMap chainedDotKinds;
  mod.walk([&](Operation *op) {
    if (!isa<tt::DotOp, tt::DotScaledOp>(op))
      return;
    setChainedDotKindAttr(op, computeChainedDotKind(op, chainedDotKinds));
  });
}

void clearChainedDotKindAttrs(ModuleOp mod) {
  StringRef attrName = ttgi::TritonIntelGPUDialect::getChainedDotKindAttrName();
  mod.walk([&](Operation *op) {
    if (!isa<tt::DotOp, tt::DotScaledOp>(op))
      return;
    op->removeAttr(attrName);
  });
}

/// GRF entries per hardware thread the kernel can end up with.
static unsigned getMaxGRFEntries(ModuleOp mod, StringRef grfMode,
                                 unsigned numWarps) {
  unsigned entries = 0;
  if (!grfMode.getAsInteger(10, entries))
    return entries;
  // "default" and "auto" can escalate up to ttig.max_grf_mode, but a larger
  // mode cannot launch more than 32 warps.
  if (numWarps > 32)
    return 128;
  if (auto maxGRFMode = mod->getAttrOfType<StringAttr>(
          ttgi::TritonIntelGPUDialect::getMaxGRFModeAttrName()))
    if (!maxGRFMode.getValue().getAsInteger(10, entries))
      return entries;
  return 256;
}

/// Whether a tt.dot with result shape `retShape` and reduction size `k` fits
/// DPAS: every dimension a power of 2, N >= executionSize, and K a multiple
/// of the native K.
static bool isDPASDotShape(ArrayRef<int64_t> retShape, int64_t k,
                           const ttgi::DpasEncodingAttr::DPASCapability &cap,
                           unsigned opsPerChan) {
  bool resultShapeSupported = llvm::all_of(retShape, [](int64_t dimension) {
    return dimension > 0 && llvm::isPowerOf2_64(dimension);
  });
  bool kLayoutSupported = k > 0 && llvm::isPowerOf2_64(k);
  const int64_t nativeK = static_cast<int64_t>(cap.systolicDepth) *
                          static_cast<int64_t>(opsPerChan);
  bool kInstructionSupported = k >= nativeK && k % nativeK == 0;
  bool nSupported = retShape.back() >= static_cast<int64_t>(cap.executionSize);
  return resultShapeSupported && kLayoutSupported && kInstructionSupported &&
         nSupported;
}

template <typename OpTy, typename = std::enable_if_t<llvm::is_one_of<
                             OpTy, tt::DotOp, tt::DotScaledOp>::value>>
class BlockedToDPAS : public OpRewritePattern<OpTy> {
  using TensorValue = TypedValue<RankedTensorType>;

public:
  BlockedToDPAS(MLIRContext *context, int benefit, StringRef grfMode)
      : OpRewritePattern<OpTy>(context, benefit), grfMode(grfMode) {}

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    RankedTensorType oldRetType = op.getType();
    if (!oldRetType.getEncoding() ||
        isa<ttgi::DpasEncodingAttr>(oldRetType.getEncoding()))
      return failure();

    auto funcOp = op->template getParentOfType<FunctionOpInterface>();
    ModuleOp mod = funcOp->template getParentOfType<ModuleOp>();
    auto dpasAnalysis = ttgi::DPASAnalysisFactory::createDPASAnalysis(mod);

    // Ensure function-wide DPAS applicability.
    if (ttgi::DPASAnalysisFactory::canUseDPAS(funcOp, dpasAnalysis) !=
        ttgi::DPASAnalysisResult::True)
      return failure();

    // Ensure operation-level DPAS applicability.
    if (ttgi::DPASAnalysisFactory::canUseDPAS(op, dpasAnalysis) !=
        ttgi::DPASAnalysisResult::True)
      return failure();

    // Get DPAS capability from module.
    std::optional<ttgi::DpasEncodingAttr::DPASCapability> dpasCap =
        ttgi::DpasEncodingAttr::getDPASCapability(mod);
    if (!dpasCap.has_value())
      return failure();

    // Create a DPAS encoding for the requested number of warps.
    ArrayRef<int64_t> retShape = oldRetType.getShape();
    unsigned numWarps = ttg::lookupNumWarps(funcOp);

    TensorValue a = op.getA();
    TensorValue b = op.getB();
    auto oldAType = cast<RankedTensorType>(a.getType());
    auto oldBType = cast<RankedTensorType>(b.getType());

    Type elemType = oldAType.getElementType();
    unsigned opsPerChan = getOpsPerChannel(elemType, mod);

    // Verify shape for DPAS dot. Does not apply to tt::DotScaledOp.
    if constexpr (std::is_same_v<OpTy, tt::DotOp>) {
      if (!isDPASDotShape(retShape, oldAType.getShape().back(), *dpasCap,
                          opsPerChan))
        return rewriter.notifyMatchFailure(
            op, "dot shape is not eligible for DPAS");
    }

    unsigned grfEntries = getMaxGRFEntries(mod, grfMode, numWarps);
    SmallVector<unsigned> warpsPerTile =
        getWarpsPerTile(op, *dpasCap, retShape, numWarps, grfEntries);
    unsigned threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(mod);

    size_t rank = retShape.size();
    SmallVector<unsigned> repCluster = ttgi::calculateRepCluster(
        *dpasCap, opsPerChan, retShape, threadsPerWarp,
        getIntOrFloatOrPtrBitWidth(oldAType.getElementType()),
        isa<Float8E5M2Type, Float8E4M3FNType>(oldAType.getElementType()),
        oldAType.getShape(), oldBType.getShape(), warpsPerTile);

    unsigned repeatCount = std::min(
        dpasCap->repeatCount, static_cast<unsigned>(retShape[rank - 2] /*M*/));
    unsigned numElemsPerRowForA =
        opsPerChan == 1
            ? dpasCap->systolicDepth
            : dpasCap->systolicDepth * 2; // A is packed to i16 or i32.
    unsigned minM = mlir::ceil<unsigned>(threadsPerWarp, numElemsPerRowForA);
    repeatCount = std::max(repeatCount, minM);

    ttgi::DPASEngineTypeVariant dpasType =
        ttgi::DPASAnalysisFactory::getDPASType(op, dpasAnalysis);

    auto dpasEnc = ttgi::DpasEncodingAttr::get(
        oldRetType.getContext(), repeatCount, dpasCap->systolicDepth,
        dpasCap->executionSize, opsPerChan, warpsPerTile, repCluster,
        threadsPerWarp,
        std::holds_alternative<ttgi::DPASEngineTypeXe3P>(dpasType) &&
                std::get<ttgi::DPASEngineTypeXe3P>(dpasType) ==
                    ttgi::DPASEngineTypeXe3P::FP32_FP32_FP4_FP4
            ? std::make_optional(2)
            : std::nullopt);

    RankedTensorType newRetType = oldRetType.cloneWithEncoding(dpasEnc);

    // Convert accumulator.
    TensorValue oldAcc = op.getC();
    auto newAcc = ttg::ConvertLayoutOp::create(rewriter, oldAcc.getLoc(),
                                               newRetType, oldAcc);
    // Operand A is packed to i16 for scalar types < 16 bits.
    // Operand B is packed to i32.
    auto newAEncoding = ttg::DotOperandEncodingAttr::get(
        oldAType.getContext(), 0, newRetType.getEncoding(),
        opsPerChan == 1 ? opsPerChan : opsPerChan / 2);
    auto newBEncoding = ttg::DotOperandEncodingAttr::get(
        oldBType.getContext(), 1, newRetType.getEncoding(), opsPerChan);

    RankedTensorType newAType = oldAType.cloneWithEncoding(newAEncoding);
    RankedTensorType newBType = oldBType.cloneWithEncoding(newBEncoding);

    a = ttg::ConvertLayoutOp::create(rewriter, a.getLoc(), newAType, a);
    b = ttg::ConvertLayoutOp::create(rewriter, b.getLoc(), newBType, b);

    Value res = nullptr;
    if constexpr (std::is_same<OpTy, tt::DotScaledOp>::value) {
      MLIRContext *ctx = rewriter.getContext();
      TensorValue scaleA = op.getAScale();
      if (scaleA) {
        tt::LinearLayout scaleALayout = BlockScaledDPAStoLinearLayout(
            scaleA.getType().getShape(), dpasEnc, 3);
        RankedTensorType newScaleAType = scaleA.getType().cloneWithEncoding(
            ttg::LinearEncodingAttr::get(ctx, scaleALayout));
        scaleA = ttg::ConvertLayoutOp::create(rewriter, scaleA.getLoc(),
                                              newScaleAType, scaleA);
      }
      TensorValue scaleB = op.getBScale();
      if (scaleB) {
        tt::LinearLayout scaleBLayout = BlockScaledDPAStoLinearLayout(
            scaleB.getType().getShape(), dpasEnc, 4);
        RankedTensorType newScaleBType = scaleB.getType().cloneWithEncoding(
            ttg::LinearEncodingAttr::get(ctx, scaleBLayout));
        scaleB = ttg::ConvertLayoutOp::create(rewriter, scaleB.getLoc(),
                                              newScaleBType, scaleB);
      }
      auto newOp = tt::DotScaledOp::create(
          rewriter, op.getLoc(), newRetType, a, b, newAcc, scaleA, scaleB,
          op.getAElemType(), op.getBElemType(), op.getFastMath());
      res = newOp.getResult();
    } else if constexpr (std::is_same<OpTy, tt::DotOp>::value) {
      auto newOp =
          tt::DotOp::create(rewriter, op.getLoc(), newRetType, a, b, newAcc,
                            op.getInputPrecision(), op.getMaxNumImpreciseAcc());
      res = newOp.getResult();
    }
    assert(res && "Expecting a valid value");

    rewriter.replaceOpWithNewOp<ttg::ConvertLayoutOp>(op, oldRetType, res);

    return success();
  }

private:
  std::string grfMode;

  /// True if B plus the accumulator of the dot do not fit in \p grfEntries.
  static bool exceedsRegistersWhenDuplicated(
      Operation *dotOp, ArrayRef<int64_t> shape,
      const ttgi::DpasEncodingAttr::DPASCapability &dpasCap,
      unsigned grfEntries) {
    Value operandB = getDotOperands(dotOp).second;
    auto bType = dyn_cast<RankedTensorType>(operandB.getType());
    auto cType = dyn_cast<RankedTensorType>(dotOp->getResult(0).getType());
    if (!bType || !cType)
      return false;
    unsigned rank = shape.size();
    int64_t m = shape[rank - 2], n = shape[rank - 1];
    int64_t k = bType.getShape()[rank - 2];
    int64_t bBytes =
        k * n * getIntOrFloatOrPtrBitWidth(bType.getElementType()) / 8;
    int64_t accBytes = m * n * cType.getElementTypeBitWidth() / 8;
    // A GRF entry holds one 32-bit value per lane of the DPAS sub-group.
    int64_t grfBytes = int64_t(grfEntries) * dpasCap.executionSize * 4;
    return bBytes + accBytes > grfBytes;
  }

  /// Dots whose A operand is computed from this dot.
  static SmallVector<Operation *> getConsumersAlongA(Operation *dotOp) {
    auto filter = [&dotOp](Operation *op) {
      return op->getParentRegion() == dotOp->getParentRegion();
    };
    SetVector<Operation *> forwardSlice;
    getForwardSlice(dotOp, &forwardSlice, {filter});
    SmallVector<Operation *> consumers;
    for (Operation *op : forwardSlice)
      if (isa<tt::DotOp, tt::DotScaledOp>(op) &&
          getChainedDotOps(getDotOperands(op).first).contains(dotOp))
        consumers.push_back(op);
    return consumers;
  }

  /// Whether the warps should go along N instead of being duplicated along M.
  /// A producer moves only together with an oversized consumer.
  static bool
  shouldDistributeAlongN(Operation *dotOp, ArrayRef<int64_t> shape,
                         const ttgi::DpasEncodingAttr::DPASCapability &dpasCap,
                         unsigned grfEntries) {
    if (!exceedsRegistersWhenDuplicated(dotOp, shape, dpasCap, grfEntries))
      return false;
    if (!getChainedDotOps(getDotOperands(dotOp).first).empty())
      return true;
    return llvm::any_of(getConsumersAlongA(dotOp), [&](Operation *consumer) {
      auto type = cast<RankedTensorType>(consumer->getResult(0).getType());
      return exceedsRegistersWhenDuplicated(consumer, type.getShape(), dpasCap,
                                            grfEntries);
    });
  }

  SmallVector<unsigned>
  getWarpsPerTile(Operation *dotOp,
                  const ttgi::DpasEncodingAttr::DPASCapability &dpasCap,
                  const ArrayRef<int64_t> shape, unsigned numWarps,
                  unsigned grfEntries) const {
    auto filter = [&dotOp](Operation *op) {
      return op->getParentRegion() == dotOp->getParentRegion();
    };

    SetVector<Operation *> slices = getSlice(dotOp, {filter});
    for (Operation *op : slices) {
      if (isa<tt::DotOp, tt::DotScaledOp>(op) && (op != dotOp)) {
        if (auto forOp = op->getParentOfType<scf::ForOp>()) {
          // FIXME: Remove once IGC can split large 2D block loads.
          MLIRContext *ctx = forOp->getContext();
          StringRef attrName =
              ttgi::TritonIntelGPUDialect::getOneMatrixPerLoadAttrName();
          setAttrOnBOperand(dotOp, attrName, UnitAttr::get(ctx));
          setAttrOnBOperand(op, attrName, UnitAttr::get(ctx));
        }
      }
    }

    ChainedDotKind chainedDotKind = ChainedDotKind::NoChained;
    if (auto kind = getChainedDotKindAttr(dotOp)) {
      chainedDotKind = *kind;
    }

    if (chainedDotKind == ChainedDotKind::ChainedAlongA ||
        chainedDotKind == ChainedDotKind::ChainedAlongB) {
      unsigned rank = shape.size();
      assert((rank == 2 || rank == 3) && "expecting a 2D or 3D dot operation");
      SmallVector<unsigned> ret(rank, 1);
      unsigned maxNumWarpsAlongM = clampToPowerOfTwo(
          mlir::ceil<unsigned>(shape[rank - 2], dpasCap.repeatCount));
      unsigned maxNumWarpsAlongN = clampToPowerOfTwo(
          mlir::ceil<unsigned>(shape[rank - 1], dpasCap.executionSize));

      // Distribute \p warps over the batch dimension (if any) and return the
      // number of warps left to distribute over the M and N dimensions.
      auto assignBatchWarps = [&](unsigned warps) {
        if (rank == 2)
          return warps;
        unsigned batch = static_cast<unsigned>(std::max<int64_t>(shape[0], 1));
        ret[0] = std::min(clampToPowerOfTwo(batch), warps);
        return mlir::ceil<unsigned>(warps, ret[0]);
      };

      // When the chain runs through operand A, N is the K dimension of the
      // consumer dot, so distributing warps along N forces the intermediate
      // result to be exchanged between warps through shared memory on every
      // iteration of the enclosing loop. If M cannot hold more than one warp
      // (e.g. M == 1 for attention decode) there is nothing to trade: keep the
      // warps on the batch and M dimensions, because duplicating the
      // computation is cheaper than that exchange.
      //
      // The mirrored case (a chain through operand B, whose M is the K
      // dimension of the consumer dot) has the same defect but is deliberately
      // left alone: there the degenerate dimension is N, and a chain with a
      // narrow N can still have a large M. Moving the warps onto N would
      // duplicate all but one of them and forfeit that M parallelism to save a
      // per-iteration exchange -- measured 1.6x to 7.6x slower for N == 16,
      // M == 128, growing with the warp count. Trading the exchange against the
      // parallelism it costs needs a cost model rather than this guard.
      //
      // Duplicating along M only pays off while the copy fits in registers.
      if (chainedDotKind == ChainedDotKind::ChainedAlongA &&
          maxNumWarpsAlongM == 1) {
        if (shouldDistributeAlongN(dotOp, shape, dpasCap, grfEntries)) {
          ret[rank - 1] = assignBatchWarps(numWarps);
          return ret;
        }
        ret[rank - 2] = assignBatchWarps(numWarps);
        return ret;
      }

      if (chainedDotKind == ChainedDotKind::ChainedAlongA) {
        ret[rank - 2] = std::min(maxNumWarpsAlongM, numWarps);
        ret[rank - 1] = std::min(maxNumWarpsAlongN,
                                 mlir::ceil<unsigned>(numWarps, ret[rank - 2]));
      } else {
        ret[rank - 1] = std::min(maxNumWarpsAlongN, numWarps);
        ret[rank - 2] = std::min(maxNumWarpsAlongM,
                                 mlir::ceil<unsigned>(numWarps, ret[rank - 1]));
      }

      unsigned numWarpsUsed = ret[rank - 1] * ret[rank - 2];
      if (numWarpsUsed < numWarps) {
        unsigned remainingWarps =
            assignBatchWarps(mlir::ceil<unsigned>(numWarps, numWarpsUsed));

        // Put remaining parallelism on the non-chained dot dimension.
        if (chainedDotKind == ChainedDotKind::ChainedAlongA) {
          ret[rank - 1] *= remainingWarps;
        } else {
          ret[rank - 2] *= remainingWarps;
        }
      }

      return ret;
    }

    SmallVector<unsigned> ret = ttgi::calculateWarpsPerTile(
        dpasCap.repeatCount, dpasCap.executionSize, shape, numWarps);
    return ret;
  }
};

class UpcastScaledBlocked : public OpRewritePattern<tt::DotScaledOp> {

public:
  using OpRewritePattern<tt::DotScaledOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tt::DotScaledOp scaledDotOp,
                                PatternRewriter &rewriter) const override {
    RankedTensorType oldRetType = scaledDotOp.getType();
    if (!oldRetType.getEncoding() ||
        isa<ttgi::DpasEncodingAttr>(oldRetType.getEncoding()))
      return failure();

    tt::ScaleDotElemType precA;
    tt::ScaleDotElemType precB;
    if (!(scaledDotOp.getRhsKPack() && scaledDotOp.getLhsKPack())) {
      // BDPAS only supports the fp4 which is packed along K for the A and B
      // matrices. Upcast A and B to unpack the FP4 which is packed on non-k
      // dim. FP16 is chosen here because:
      // 1. It can represent all FP4 values exactly (without rounding)
      // 2. Sufficient range for the FP4.
      // 3. The Fp4ToFp only supports upcast to fp16/bf16 for now.
      precA = precB = tt::ScaleDotElemType::FP16;
    } else {
      // Upcast A and B for mixed types.
      std::optional<std::tuple<tt::ScaleDotElemType, tt::ScaleDotElemType>>
          computeType = getComputeType(scaledDotOp.getAElemType(),
                                       scaledDotOp.getBElemType(), rewriter);
      if (!computeType)
        return failure();
      precA = std::get<0>(*computeType);
      precB = std::get<1>(*computeType);
    }

    TypedValue<RankedTensorType> A =
        upcastMatrix(rewriter, scaledDotOp, 0, precA);
    TypedValue<RankedTensorType> B =
        upcastMatrix(rewriter, scaledDotOp, 1, precB);
    auto newDot = tt::DotScaledOp::create(
        rewriter, scaledDotOp.getLoc(), scaledDotOp->getResultTypes(), A, B,
        scaledDotOp.getC(), scaledDotOp.getAScale(), scaledDotOp.getBScale(),
        precA, precB, scaledDotOp.getFastMath(), scaledDotOp.getLhsKPack(),
        scaledDotOp.getRhsKPack());
    if (auto chainedDotKind = getChainedDotKindAttr(scaledDotOp))
      setChainedDotKindAttr(newDot, *chainedDotKind);

    rewriter.replaceOp(scaledDotOp, newDot);
    return success();
  }

private:
  static std::optional<unsigned>
  getScaleDotElemTypeBitWidth(tt::ScaleDotElemType type) {
    switch (type) {
    case tt::ScaleDotElemType::E2M1:
      return 4;
    case tt::ScaleDotElemType::E4M3:
    case tt::ScaleDotElemType::E5M2:
      return 8;
    case tt::ScaleDotElemType::BF16:
    case tt::ScaleDotElemType::FP16:
      return 16;
    default:
      // For other unsupported float types.
      return std::nullopt;
    }
  };

  // Retrieve the precision type of matrix A and B supported by bdpas for mixed
  // tt.dot_scaled operations.
  std::optional<std::tuple<tt::ScaleDotElemType, tt::ScaleDotElemType>>
  getComputeType(tt::ScaleDotElemType aType, tt::ScaleDotElemType bType,
                 PatternRewriter &rewriter) const {
    if (aType == bType) // Skip dot_scaled when it is not mixed precision.
      return std::nullopt;
    std::optional<unsigned> aBitWidth = getScaleDotElemTypeBitWidth(aType);
    std::optional<unsigned> bBitWidth = getScaleDotElemTypeBitWidth(bType);
    if (!aBitWidth || !bBitWidth) // unsupported type.
      return std::nullopt;
    unsigned minBitWidth = std::min(*aBitWidth, *bBitWidth);
    unsigned maxBitWidth = std::max(*aBitWidth, *bBitWidth);
    if (minBitWidth < maxBitWidth) {
      // align to the larger bit width type.
      if (minBitWidth == 4) {
        // There is limitation in Fp4ToFpOp that it only supports to upcast to
        // fp16/bf16.
        if (aType == tt::ScaleDotElemType::FP16 ||
            bType == tt::ScaleDotElemType::FP16)
          return std::make_tuple(tt::ScaleDotElemType::FP16,
                                 tt::ScaleDotElemType::FP16);
        else
          return std::make_tuple(tt::ScaleDotElemType::BF16,
                                 tt::ScaleDotElemType::BF16);
      }

      if (aBitWidth > bBitWidth) {
        return std::make_tuple(aType, aType);
      } else {
        return std::make_tuple(bType, bType);
      }
    } else {
      // align to the type with larger range.
      assert(minBitWidth != 4 &&
             "invalid packed dot_scaled with different fp4");

      if (minBitWidth == 8) {
        // BDPAS support mixed fp8 natively.
        return std::nullopt;
      }

      if (minBitWidth == 16) {
        return std::make_tuple(tt::ScaleDotElemType::BF16,
                               tt::ScaleDotElemType::BF16);
      }
    }
    return std::nullopt;
  }

  // Upcast the matrix A or B of the tt.dot_scaled.
  TypedValue<RankedTensorType>
  upcastMatrix(PatternRewriter &rewriter, tt::DotScaledOp scaledDotOp,
               int opIdx, tt::ScaleDotElemType computeType) const {
    TypedValue<RankedTensorType> v =
        opIdx == 0 ? scaledDotOp.getA() : scaledDotOp.getB();
    bool isFp4 =
        tt::ScaleDotElemType::E2M1 ==
        (opIdx == 0 ? scaledDotOp.getAElemType() : scaledDotOp.getBElemType());

    Location loc = v.getLoc();
    int64_t rank = v.getType().getRank();
    int64_t kDim = opIdx == 0 ? rank - 1 : rank - 2;

    // Upcast value to computeType (fp16/bf16)
    if (isFp4) {
      bool kPack =
          opIdx == 0 ? scaledDotOp.getLhsKPack() : scaledDotOp.getRhsKPack();
      int64_t packDim = kPack ? kDim : (opIdx == 0 ? rank - 2 : rank - 1);
      v = ttg::Fp4ToFpOp::create(rewriter, loc, v,
                                 getScalarType(rewriter, computeType), packDim);
    } else {
      RankedTensorType vType =
          v.getType().clone(getScalarType(rewriter, computeType));
      tt::FpToFpOp op =
          tt::FpToFpOp::create(rewriter, loc, vType, v, tt::RoundingModeAttr{});
      v = cast<TypedValue<RankedTensorType>>(op.getResult());
    }
    return v;
  }

  FloatType getScalarType(PatternRewriter &rewriter,
                          tt::ScaleDotElemType computeType) const {
    mlir::MLIRContext *ctx = rewriter.getContext();
    switch (computeType) {
    case tt::ScaleDotElemType::BF16:
      return rewriter.getBF16Type();
    case tt::ScaleDotElemType::FP16:
      return rewriter.getF16Type();
    case tt::ScaleDotElemType::E5M2:
      return mlir::Float8E5M2Type::get(ctx);
    case tt::ScaleDotElemType::E4M3:
      return mlir::Float8E4M3FNType::get(ctx);
    case tt::ScaleDotElemType::E2M1:
      return mlir::Float4E2M1FNType::get(ctx);
    default:
      assert(false && "unsupported precision type");
    }
    return {};
  }
};

} // namespace

static Value promoteOperand(OpBuilder &builder, Location loc, Value operand,
                            Type promotedType) {
  auto tensorPromotedType = cast<RankedTensorType>(operand.getType())
                                .cloneWith(std::nullopt, promotedType);
  Type elemType = tensorPromotedType.getElementType();

  return llvm::TypeSwitch<Type, Value>(elemType)
      .Case<FloatType>([&](auto) {
        return tt::FpToFpOp::create(builder, loc, tensorPromotedType, operand,
                                    tt::RoundingModeAttr{});
      })
      .Case<IntegerType>([&](auto) {
        unsigned tgtBitWidth = elemType.getIntOrFloatBitWidth(),
                 valBitWidth = cast<RankedTensorType>(operand.getType())
                                   .getElementTypeBitWidth();
        Operation *castOp =
            (valBitWidth <= tgtBitWidth)
                ? arith::ExtSIOp::create(builder, loc, tensorPromotedType,
                                         operand)
                : arith::TruncIOp::create(builder, loc, tensorPromotedType,
                                          operand);
        return castOp->getResult(0);
      });
}

// Replaces \p dotOp, which has a bf16 accumulator, with a dot that accumulates
// in fp32 followed by a single RTNE rounding of the result to bf16.
static tt::DotOp accumulateInF32(tt::DotOp dotOp) {
  OpBuilder builder(dotOp);
  Location loc = dotOp.getLoc();
  RankedTensorType DType = dotOp.getD().getType();
  RankedTensorType f32Type = DType.clone(builder.getF32Type());
  auto acc = tt::FpToFpOp::create(builder, loc, f32Type, dotOp.getC(),
                                  tt::RoundingModeAttr{});
  auto f32Dot = tt::DotOp::create(builder, loc, f32Type, dotOp.getA(),
                                  dotOp.getB(), acc, dotOp.getInputPrecision(),
                                  dotOp.getMaxNumImpreciseAcc());
  auto result = tt::FpToFpOp::create(
      builder, loc, DType, f32Dot.getD(),
      tt::RoundingModeAttr::get(builder.getContext(), tt::RoundingMode::RTNE));
  dotOp.replaceAllUsesWith(result.getResult());
  dotOp.erase();
  return f32Dot;
}

// Returns how many threads hold each element of \p dotOperand on average: its
// elements per thread over an even split of the tensor across the CTA.
static double getDotOperandReplication(Value dotOperand) {
  auto type = cast<RankedTensorType>(dotOperand.getType());
  auto mod = dotOperand.getParentRegion()->getParentOfType<ModuleOp>();
  int64_t numThreads =
      int64_t(ttg::lookupNumWarps(dotOperand.getParentRegion())) *
      ttg::TritonGPUDialect::getThreadsPerWarp(mod);
  return double(ttg::getTotalElemsPerThread(type)) * numThreads /
         double(product(type.getShape()));
}

// Promotes \p operand like promoteOperand. When \p operand is an fp8 layout
// conversion into a dot operand layout, which replicates each element across
// threads, the fp8 values can be decoded before the conversion instead, once
// per element rather than once per copy. They are decoded only to a 16-bit
// type (fp16 for an fp32 promotion, which is exact for fp8), and a wider
// promotion is finished in the dot operand layout, where it is a single cheap
// instruction: moving 32-bit values through the layout conversion costs more
// registers than it saves. The decode is only moved while it takes several
// instructions: with native fp8 conversion, fp8 -> fp16 is a single builtin,
// as cheap as the casts of other types (e.g. int8, fp16), which stay in the
// dot operand layout, while fp8 -> bf16 is still emulated. An fp16 decode of
// an operand with little replication keeps the cast in the dot operand layout
// too.
// The decode is placed right after the source, so it runs once and is hoisted
// with it (e.g. out of a loop), at the cost of keeping the decoded 16-bit
// value live from there; the new conversion is placed at the builder's
// insertion point (the dot), so the replicated value is created where it is
// used.
static Value promoteOperandBeforeLayoutConversion(OpBuilder &builder,
                                                  Location loc, Value operand,
                                                  Type promotedType) {
  Type elemType = getElementTypeOrSelf(operand);
  bool isFP8 =
      isa<FloatType>(elemType) && elemType.getIntOrFloatBitWidth() == 8;
  auto cvtOp = operand.getDefiningOp<ttg::ConvertLayoutOp>();
  if (!isFP8 || !cvtOp ||
      isa<ttg::DotOperandEncodingAttr>(cvtOp.getSrc().getType().getEncoding()))
    return promoteOperand(builder, loc, operand, promotedType);

  // fp8 -> bf16 is always emulated, so it is moved for any replication. An
  // fp16 decode can be cheap (a shift for e5m2, a builtin with native fp8
  // conversion): on BMG, moving it paid off only for operands replicated at
  // least 16 times.
  constexpr double minReplication = 16;
  Type decodedType = promotedType.getIntOrFloatBitWidth() > 16
                         ? builder.getF16Type()
                         : promotedType;
  if (!decodedType.isBF16()) {
    auto mod = operand.getParentRegion()->getParentOfType<ModuleOp>();
    if (mod->hasAttr(
            ttgi::TritonIntelGPUDialect::getSupportF8ConversionAttrName()) ||
        getDotOperandReplication(operand) < minReplication)
      return promoteOperand(builder, loc, operand, promotedType);
  }

  Value decodedSrc;
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfterValue(cvtOp.getSrc());
    decodedSrc = promoteOperand(builder, loc, cvtOp.getSrc(), decodedType);
  }
  auto decodedTy = cast<RankedTensorType>(operand.getType())
                       .cloneWith(std::nullopt, decodedType);
  Value decoded = ttg::ConvertLayoutOp::create(builder, cvtOp.getLoc(),
                                               decodedTy, decodedSrc);
  if (decodedType == promotedType)
    return decoded;
  return promoteOperand(builder, loc, decoded, promotedType);
}

// promote operands of dot op if the existing combination is not natively
// supported.
static void decomposeMixedModeDotOp(ModuleOp mod) {
  mod.walk([](tt::DotOp dotOp) -> void {
    auto D = dotOp.getD();
    OpBuilder builder(dotOp);
    Type AElType = dotOp.getA().getType().getElementType();
    auto dpasLayout =
        dyn_cast<ttgi::DpasEncodingAttr>(D.getType().getEncoding());

    Type promoteType;
    if (dpasLayout) {
      // FP8 is not always natively supported by the DPAS instruction.
      // Promote it when needed.
      bool isNativeFP8 = isa<Float8E5M2Type, Float8E4M3FNType>(AElType);
      auto mod = dotOp->getParentOfType<ModuleOp>();
      bool supportsFP8 = mod->hasAttr(
          ttgi::TritonIntelGPUDialect::getSupportDPASWithBF8AttrName());
      if (supportsFP8 || !isNativeFP8)
        return;
      // Extend to fp16, which is exact for fp8. DPAS has a bf16 accumulator
      // only for bf16 operands, so a dot with a bf16 accumulator accumulates
      // in fp32 and rounds the result to bf16 once.
      if (D.getType().getElementType().isBF16()) {
        dotOp = accumulateInF32(dotOp);
        builder.setInsertionPoint(dotOp);
      }
      promoteType = builder.getF16Type();
    } else {
      // FMA case.
      Type DElType = D.getType().getElementType();
      if (DElType.isBF16()) {
        // FMA rounds the accumulator after every multiply-add, and IGC lowers
        // a bf16 multiply-add to a `mad` with a bf16 destination, which rounds
        // toward zero. Accumulate in fp32 and round the result once instead,
        // as bf16 DPAS does per instruction. This covers bf16 and fp8 dots
        // that cannot use DPAS (e.g. fp8 without fp8 DPAS, or mixed fp8).
        // fp16 accumulators are intentionally left as is: an fp16 multiply-add
        // rounds to nearest.
        dotOp = accumulateInF32(dotOp);
        DElType = builder.getF32Type();
      }
      if (AElType == DElType)
        return;
      promoteType = DElType;
      builder.setInsertionPoint(dotOp);
    }

    Location loc = dotOp.getLoc();
    auto promote =
        dpasLayout ? promoteOperand : promoteOperandBeforeLayoutConversion;
    Value oldA = dotOp.getA(), oldB = dotOp.getB();
    dotOp.setOperand(0, promote(builder, loc, oldA, promoteType));
    dotOp.setOperand(1, promote(builder, loc, oldB, promoteType));
    // Drop layout conversions that the promotion left unused.
    if (auto cvtOp = oldA.getDefiningOp<ttg::ConvertLayoutOp>();
        cvtOp && cvtOp->use_empty())
      cvtOp->erase();
    if (oldB != oldA)
      if (auto cvtOp = oldB.getDefiningOp<ttg::ConvertLayoutOp>();
          cvtOp && cvtOp->use_empty())
        cvtOp->erase();
  });
}

static void updateValueType(Value v, Attribute encoding,
                            ArrayRef<int64_t> shape) {
  auto tensorType = cast<RankedTensorType>(v.getType());
  auto newType =
      RankedTensorType::get(shape, tensorType.getElementType(), encoding);
  v.setType(newType);
}

static std::optional<tt::TransOp>
updateUsers(Value result, const SetVector<Operation *> &slice) {
  if (llvm::any_of(result.getUsers(),
                   [&](Operation *user) { return slice.count(user) == 0; })) {
    OpBuilder builder(result.getContext());
    builder.setInsertionPointAfterValue(result);
    auto transOp =
        tt::TransOp::create(builder, result.getLoc(), result, ArrayRef({1, 0}));
    result.replaceUsesWithIf(transOp.getResult(), [&](OpOperand &operand) {
      return operand.getOwner() != transOp.getOperation() &&
             slice.count(operand.getOwner()) == 0;
    });
    return transOp;
  }
  return std::nullopt;
}

// TODO: Sync the transpose in the IR, this is done to avoid generating convert
// layout when we have a transpose right after a dot as mma layout cannot be
// propagated through transpose op. Once we have layouts that can represent
// transposed MMA we can remove this transformation.
static void sinkTransposeOp(tt::TransOp input) {
  SmallVector<tt::TransOp> queue = {input};
  while (!queue.empty()) {
    tt::TransOp transOp = queue.back();
    Value currentValue = transOp.getResult();
    queue.pop_back();
    mlir::ForwardSliceOptions options;
    options.filter = [](Operation *op) {
      if (op->hasTrait<OpTrait::Elementwise>() && op->getNumOperands() == 1)
        return true;
      if (isa<scf::YieldOp>(op))
        return isa<scf::ForOp>(op->getParentOp());
      return isa<ttg::ConvertLayoutOp>(op);
    };
    SetVector<Operation *> slice;
    mlir::getForwardSlice(currentValue, &slice, options);
    for (Operation *op : slice) {
      if (op->hasTrait<OpTrait::Elementwise>()) {
        // Update users of transpose op.
        if (op->getOperand(0) == transOp.getResult())
          op->setOperand(0, transOp.getOperand());
        // Update the type of the result.
        for (Value result : op->getResults()) {
          auto srcType = cast<RankedTensorType>(op->getOperand(0).getType());
          updateValueType(result, srcType.getEncoding(), srcType.getShape());
          updateUsers(result, slice);
        }
        continue;
      }
      if (auto cvtOp = dyn_cast<ttg::ConvertLayoutOp>(op)) {
        // Update users of transpose op.
        if (op->getOperand(0) == transOp.getResult())
          op->setOperand(0, transOp.getOperand());
        auto resultEncoding = cvtOp.getType().getEncoding();
        auto newDstEncoding = ttgi::inferSrcEncoding(transOp, resultEncoding);
        assert(newDstEncoding && "Expecting valid result encoding");
        auto srcType = cast<RankedTensorType>(cvtOp.getOperand().getType());
        updateValueType(cvtOp.getResult(), newDstEncoding, srcType.getShape());
        updateUsers(cvtOp.getResult(), slice);
        continue;
      }
      assert(isa<scf::YieldOp>(op) &&
             "Transpose forward slice should contain "
             "only elementwise, convert layout and yield ops.");
      auto forOp = cast<scf::ForOp>(op->getParentOp());
      for (OpOperand &operand : op->getOpOperands()) {
        Operation *def = operand.get().getDefiningOp();
        if (def && (slice.count(def)) || def == transOp.getOperation()) {
          if (def == transOp.getOperation())
            operand.set(transOp.getOperand());
          Type newType = operand.get().getType();
          forOp.getResult(operand.getOperandNumber()).setType(newType);
          std::optional<tt::TransOp> retTrans =
              updateUsers(forOp.getResult(operand.getOperandNumber()), slice);
          // Recursively try to propagate the new transpose inserted.
          if (retTrans.has_value())
            queue.push_back(retTrans.value());
          forOp.getRegionIterArg(operand.getOperandNumber()).setType(newType);
          std::optional<tt::TransOp> argTrans = updateUsers(
              forOp.getRegionIterArg(operand.getOperandNumber()), slice);
          if (argTrans.has_value())
            queue.push_back(argTrans.value());
          OpBuilder builder(forOp);
          OpOperand &init = forOp.getInitsMutable()[operand.getOperandNumber()];
          auto initTranspose = tt::TransOp::create(
              builder, forOp.getLoc(), init.get(), ArrayRef({1, 0}));
          init.set(initTranspose);
        }
      }
    }
  }
}

static tt::TransOp transposeDotScaledOp(tt::DotScaledOp dotOp) {
  assert(dotOp.getAScale() == nullptr && dotOp.getBScale() != nullptr &&
         "Transpose DotOp expects scale on RHS");
  OpBuilder builder(dotOp);
  Value lhs = dotOp.getA();
  std::array<int, 2> transOrder = {1, 0};
  auto lhsTransposed =
      tt::TransOp::create(builder, lhs.getLoc(), lhs, transOrder);
  Value rhs = dotOp.getB();
  auto rhsTransposed =
      tt::TransOp::create(builder, rhs.getLoc(), rhs, transOrder);
  Value c = dotOp.getC();
  auto cTransposed = tt::TransOp::create(builder, c.getLoc(), c, transOrder);
  auto result = tt::DotScaledOp::create(
      builder, dotOp.getLoc(), cTransposed.getType(), rhsTransposed,
      lhsTransposed, cTransposed, dotOp.getBScale(), dotOp.getAScale(),
      dotOp.getBElemType(), dotOp.getAElemType(), dotOp.getFastMath(),
      /*lhs_k_pack=*/dotOp.getRhsKPack(),
      /*rhs_k_pack=*/dotOp.getLhsKPack());
  auto transOp =
      tt::TransOp::create(builder, result.getLoc(), result, transOrder);
  dotOp.replaceAllUsesWith(transOp.getOperation());
  dotOp.erase();
  return transOp;
}

static void transposeDotScaledOp(ModuleOp m) {
  SmallVector<tt::DotScaledOp> toTranspose;
  m.walk([&](tt::DotScaledOp dotOp) -> void {
    // The rewrite uses rank-2 transposes; batched dots are decomposed as-is.
    if (dotOp.getAScale() == nullptr && dotOp.getBScale() != nullptr &&
        dotOp.getType().getRank() == 2)
      toTranspose.push_back(dotOp);
  });
  SmallVector<tt::TransOp> transposes;
  for (tt::DotScaledOp &dotOp : toTranspose) {
    tt::TransOp transpose = transposeDotScaledOp(dotOp);
    transposes.push_back(transpose);
  }

  for (tt::TransOp transpose : transposes) {
    sinkTransposeOp(transpose);
  }
}

// Fold a tt.trans of a computed tile into its producers:
//
//   %s  = tt.dot %a, %b, %c
//   %p  = f(%s, broadcast(expand_dims(%x, 1)), ...)
//   %pT = tt.trans %p {order = [1, 0]} -> [truncf] -> cvt -> tt.dot %pT (as A)
// becomes
//   %sT = tt.dot trans(%b), trans(%a), trans(%c)
//   %pT = f(%sT, broadcast(expand_dims(%x, 0)), ...)
//
// The tt.trans gives the A operand of the consuming dot a layout with the
// warps along dim 1, while DPAS A needs them along dim 0, so the operand goes
// through shared memory. Computing the tile transposed keeps it in registers.
//
// The rewritten region is the backward slice of all such transposes in one
// function or loop body. Its ops are rank-2 elementwise ops, broadcast,
// convert_layout and tt.dot, all in that block, and none of its values may be
// used outside it. It ends at leaves that are cheap to transpose: splats and
// splat constants, expand_dims of a rank-1 value (the axis swaps), tt.trans
// (it cancels), descriptor loads (when RemoveLayoutConversions later gives each
// use its own load, OptimizeDotOperands makes trans(load) a column-major load;
// otherwise the transpose stays), select conditions (one tt.trans of the mask)
// and values defined outside the block (transposed once there). Every new value
// takes the type tt.trans infers for the old one, so the types seen by the
// users of the transposes do not change.
namespace {

constexpr std::array<int32_t, 2> transposeOrder = {1, 0};

bool isTranspose2D(tt::TransOp trans) {
  return trans.getOrder() == ArrayRef<int32_t>(transposeOrder);
}

RankedTensorType getTransposedType(RankedTensorType type) {
  ArrayRef<int64_t> shape = type.getShape();
  Attribute encoding = type.getEncoding();
  Attribute transEncoding;
  if (encoding &&
      failed(cast<tt::DialectInferLayoutInterface>(&encoding.getDialect())
                 ->inferTransOpEncoding(encoding, shape, transposeOrder,
                                        transEncoding, std::nullopt)))
    return {};
  return RankedTensorType::get({shape[1], shape[0]}, type.getElementType(),
                               transEncoding);
}

// The value a dot operand is converted from: dot operands are a
// convert_layout to #dot_op at this point of the pipeline.
Value getDotOperandSource(Value operand,
                          SmallVectorImpl<Operation *> *cvts = nullptr) {
  while (auto cvt = operand.getDefiningOp<ttg::ConvertLayoutOp>()) {
    if (cvts)
      cvts->push_back(cvt);
    operand = cvt.getSrc();
  }
  return operand;
}

// The tt.dot that the single-use chain from `trans` reaches as operand A, or
// null.
tt::DotOp getDotFedAsA(tt::TransOp trans) {
  Value value = trans.getResult();
  while (value.hasOneUse()) {
    Operation *user = *value.getUsers().begin();
    if (auto dot = dyn_cast<tt::DotOp>(user))
      return dot.getA() == value ? dot : tt::DotOp();
    if (!isa<arith::TruncFOp, tt::FpToFpOp, ttg::ConvertLayoutOp>(user))
      return {};
    value = user->getResult(0);
  }
  return {};
}

// The shape check of BlockedToDPAS for `dot` with result shape `shape`.
bool fitsDPASShape(tt::DotOp dot, ArrayRef<int64_t> shape,
                   const ttgi::DpasEncodingAttr::DPASCapability &dpasCap,
                   ModuleOp mod) {
  RankedTensorType aType = dot.getA().getType();
  return isDPASDotShape(shape, aType.getShape().back(), dpasCap,
                        getOpsPerChannel(aType.getElementType(), mod));
}

bool isFreeLeaf(Operation *def) {
  if (auto cst = dyn_cast<arith::ConstantOp>(def))
    return isa<SplatElementsAttr>(cst.getValue());
  if (auto expandDims = dyn_cast<tt::ExpandDimsOp>(def))
    return expandDims.getSrc().getType().getRank() == 1;
  if (auto trans = dyn_cast<tt::TransOp>(def))
    return isTranspose2D(trans);
  return isa<tt::SplatOp, tt::DescriptorLoadOp>(def);
}

// A select's i1 condition is transposed once as a whole, as in
// tl.trans(mask). Pulling its compares into the region would build the mask in
// the dot layout, which keeps more values live across the dots. It is decided
// per value, so a mask with other uses is still a leaf.
bool isSelectCondition(Value value) {
  return llvm::any_of(value.getUsers(), [&](Operation *user) {
    auto select = dyn_cast<arith::SelectOp>(user);
    return select && select.getCondition() == value;
  });
}

bool isInterior(Operation *op) {
  if (op->getNumResults() != 1)
    return false;
  // A dot that has no DPAS layout yet; transposeDot rebuilds its operand
  // encodings on a blocked parent.
  if (auto dot = dyn_cast<tt::DotOp>(op))
    return llvm::all_of(ValueRange{dot.getA(), dot.getB()}, [](Value v) {
      auto encoding = dyn_cast_or_null<ttg::DotOperandEncodingAttr>(
          cast<RankedTensorType>(v.getType()).getEncoding());
      return encoding && isa<ttg::BlockedEncodingAttr>(encoding.getParent());
    });
  if (isa<tt::BroadcastOp, ttg::ConvertLayoutOp>(op))
    return true;
  return op->hasTrait<OpTrait::Elementwise>() && isMemoryEffectFree(op);
}

struct TransposeRegion {
  SmallVector<tt::TransOp> roots;
  // Interior ops, in program order.
  SmallVector<Operation *> ops;
  SetVector<Value> leaves;
  // The convert_layout ops between a dot of the region and its A and B
  // sources; they are replaced along with the dot.
  SmallVector<Operation *> operandCvts;
};

std::optional<TransposeRegion>
collectRegion(ArrayRef<tt::TransOp> roots,
              const ttgi::DpasEncodingAttr::DPASCapability &dpasCap,
              ModuleOp mod) {
  // Bounds the compile time of the walk; not a cost model.
  constexpr unsigned maxOps = 256;
  Block *block = roots.front()->getBlock();
  TransposeRegion region;
  region.roots.assign(roots.begin(), roots.end());
  SetVector<Operation *> ops;
  SmallVector<Value> worklist;
  for (tt::TransOp root : roots) {
    // The rewrite gives the users of a root the transposed source, so the
    // root's type must be the one getTransposedType builds.
    auto srcType = dyn_cast<RankedTensorType>(root.getSrc().getType());
    if (!srcType || srcType.getRank() != 2 ||
        root.getType() != getTransposedType(srcType))
      return std::nullopt;
    worklist.push_back(root.getSrc());
  }

  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    auto type = dyn_cast<RankedTensorType>(value.getType());
    // Scalar operands of elementwise ops stay as they are.
    if (!type)
      continue;
    if (type.getRank() != 2 || !getTransposedType(type))
      return std::nullopt;
    Operation *def = value.getDefiningOp();
    // Block arguments, e.g. loop-carried values, are not handled.
    if (!def)
      return std::nullopt;
    if (isFreeLeaf(def) || def->getBlock() != block ||
        isSelectCondition(value)) {
      region.leaves.insert(value);
      continue;
    }
    if (!isInterior(def))
      return std::nullopt;
    if (!ops.insert(def))
      continue;
    if (ops.size() > maxOps)
      return std::nullopt;
    if (auto dot = dyn_cast<tt::DotOp>(def)) {
      worklist.push_back(getDotOperandSource(dot.getA(), &region.operandCvts));
      worklist.push_back(getDotOperandSource(dot.getB(), &region.operandCvts));
      worklist.push_back(dot.getC());
      continue;
    }
    llvm::append_range(worklist, def->getOperands());
  }

  // Without a dot the transposes would only move.
  if (llvm::none_of(ops, [](Operation *op) { return isa<tt::DotOp>(op); }))
    return std::nullopt;

  // The flipped dot swaps M and N. A dot whose shape fits DPAS only unflipped,
  // e.g. M < executionSize <= N, would fall back to FMA.
  for (Operation *op : ops) {
    auto dot = dyn_cast<tt::DotOp>(op);
    if (!dot)
      continue;
    ArrayRef<int64_t> shape = dot.getType().getShape();
    if (fitsDPASShape(dot, shape, dpasCap, mod) &&
        !fitsDPASShape(dot, {shape[1], shape[0]}, dpasCap, mod))
      return std::nullopt;
  }

  // A tt.trans leaf of an interior value, e.g. a root reached from another
  // root, would be mapped to that old value, which the rewrite erases.
  for (Value leaf : region.leaves)
    if (auto trans = leaf.getDefiningOp<tt::TransOp>();
        trans && ops.contains(trans.getSrc().getDefiningOp()))
      return std::nullopt;

  // Every value of the region must be used only inside it, by a root, or by
  // a dot operand convert of the region; otherwise it would be needed in both
  // orientations.
  DenseSet<Operation *> allowedUsers(ops.begin(), ops.end());
  for (tt::TransOp root : roots)
    allowedUsers.insert(root);
  allowedUsers.insert(region.operandCvts.begin(), region.operandCvts.end());
  for (Operation *op : ops)
    for (Operation *user : op->getUsers())
      if (!allowedUsers.contains(user))
        return std::nullopt;
  for (Operation *cvt : region.operandCvts)
    for (Operation *user : cvt->getUsers())
      if (!allowedUsers.contains(user))
        return std::nullopt;

  region.ops.assign(ops.begin(), ops.end());
  llvm::sort(region.ops,
             [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
  return region;
}

Value transposeLeaf(OpBuilder &builder, Value leaf) {
  auto newType = getTransposedType(cast<RankedTensorType>(leaf.getType()));
  Operation *def = leaf.getDefiningOp();
  Location loc = leaf.getLoc();
  if (auto trans = dyn_cast<tt::TransOp>(def);
      trans && isTranspose2D(trans) && trans.getSrc().getType() == newType)
    return trans.getSrc();
  if (auto splat = dyn_cast<tt::SplatOp>(def))
    return tt::SplatOp::create(builder, loc, newType, splat.getSrc());
  if (auto expandDims = dyn_cast<tt::ExpandDimsOp>(def)) {
    // The source becomes a slice of the transposed parent along the other
    // axis.
    int32_t newAxis = 1 - static_cast<int32_t>(expandDims.getAxis());
    Value src = expandDims.getSrc();
    Operation *cvt = nullptr;
    if (auto parent = dyn_cast_or_null<ttg::DistributedEncodingTrait>(
            newType.getEncoding())) {
      auto srcType = cast<RankedTensorType>(src.getType());
      auto sliceEncoding =
          ttg::SliceEncodingAttr::get(builder.getContext(), newAxis, parent);
      if (srcType.getEncoding() != sliceEncoding) {
        cvt = ttg::ConvertLayoutOp::create(
            builder, loc, srcType.cloneWithEncoding(sliceEncoding), src);
        src = cvt->getResult(0);
      }
    }
    auto newExpandDims = tt::ExpandDimsOp::create(builder, loc, src, newAxis);
    if (newExpandDims.getType() == newType)
      return newExpandDims;
    newExpandDims.erase();
    if (cvt)
      cvt->erase();
  }
  // An i1 tile is transposed as i32, the form tl.trans(mask.to(tl.int32))
  // takes: a direct i1 transpose lets the layout passes rebuild the mask's
  // producers in the dot layout.
  auto leafType = cast<RankedTensorType>(leaf.getType());
  if (leafType.getElementType().isInteger(1) &&
      !matchPattern(leaf, m_Constant())) {
    Value ext = arith::ExtUIOp::create(
        builder, loc, leafType.clone(builder.getI32Type()), leaf);
    auto trans = tt::TransOp::create(builder, loc, ext, transposeOrder);
    Value zero = arith::ConstantOp::create(
        builder, loc, builder.getZeroAttr(trans.getType()));
    return arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ne, trans,
                                 zero);
  }
  // Splat constants fold; other values get a real transpose.
  return builder.createOrFold<tt::TransOp>(loc, leaf, transposeOrder);
}

// trans(dot(a, b, c)) = dot(trans(b), trans(a), trans(c)).
Value transposeDot(OpBuilder &builder, tt::DotOp dot, IRMapping &mapping,
                   RankedTensorType newType) {
  auto convertOperand = [&](Value oldOperand, unsigned opIdx) -> Value {
    auto oldEncoding = cast<ttg::DotOperandEncodingAttr>(
        cast<RankedTensorType>(oldOperand.getType()).getEncoding());
    Value src = mapping.lookup(getDotOperandSource(oldOperand));
    auto encoding = ttg::DotOperandEncodingAttr::get(
        builder.getContext(), opIdx, newType.getEncoding(),
        oldEncoding.getKWidth());
    return ttg::ConvertLayoutOp::create(
        builder, dot.getLoc(),
        cast<RankedTensorType>(src.getType()).cloneWithEncoding(encoding), src);
  };
  Value a = convertOperand(dot.getB(), 0);
  Value b = convertOperand(dot.getA(), 1);
  auto newDot = tt::DotOp::create(
      builder, dot.getLoc(), newType, a, b, mapping.lookup(dot.getC()),
      dot.getInputPrecision(), dot.getMaxNumImpreciseAcc());
  newDot->setDiscardableAttrs(dot->getDiscardableAttrDictionary());
  return newDot;
}

void rewriteRegion(TransposeRegion &region) {
  assert(llvm::none_of(region.leaves,
                       [&](Value leaf) {
                         return llvm::is_contained(region.ops,
                                                   leaf.getDefiningOp());
                       }) &&
         "a value of the region is both a leaf and interior");
  OpBuilder builder(region.roots.front());
  IRMapping mapping;
  for (Value leaf : region.leaves) {
    builder.setInsertionPointAfterValue(leaf);
    mapping.map(leaf, transposeLeaf(builder, leaf));
  }
  for (Operation *op : region.ops) {
    builder.setInsertionPoint(op);
    auto newType =
        getTransposedType(cast<RankedTensorType>(op->getResult(0).getType()));
    Value newValue;
    if (auto dot = dyn_cast<tt::DotOp>(op)) {
      newValue = transposeDot(builder, dot, mapping, newType);
    } else if (auto broadcast = dyn_cast<tt::BroadcastOp>(op)) {
      newValue = tt::BroadcastOp::create(builder, broadcast.getLoc(), newType,
                                         mapping.lookup(broadcast.getSrc()));
    } else {
      // Same op on the transposed operands, in the same operand order.
      Operation *clone = builder.clone(*op, mapping);
      clone->getResult(0).setType(newType);
      newValue = clone->getResult(0);
    }
    mapping.map(op->getResult(0), newValue);
  }
  for (tt::TransOp root : region.roots) {
    root.replaceAllUsesWith(mapping.lookup(root.getSrc()));
    root.erase();
  }
  SetVector<Operation *> dead(region.ops.begin(), region.ops.end());
  dead.insert(region.operandCvts.begin(), region.operandCvts.end());
  assert(llvm::all_of(dead,
                      [&](Operation *op) {
                        return llvm::all_of(op->getUsers(),
                                            [&](Operation *user) {
                                              return dead.contains(user);
                                            });
                      }) &&
         "an op of the region is still used outside it");
  for (Operation *op : dead)
    op->dropAllUses();
  for (Operation *op : dead)
    op->erase();
}

void foldTransposeIntoProducers(ModuleOp mod) {
  // A descriptor load leaf is cheap only as a column-major 2D block load.
  if (!mod->hasAttr(ttgi::TritonIntelGPUDialect::getSupport2DBlockIOAttrName()))
    return;
  std::optional<ttgi::DpasEncodingAttr::DPASCapability> dpasCap =
      ttgi::DpasEncodingAttr::getDPASCapability(mod);
  if (!dpasCap)
    return;
  auto dpasAnalysis = ttgi::DPASAnalysisFactory::createDPASAnalysis(mod);
  mod.walk([&](tt::FuncOp func) {
    // The premise, DPAS A wanting the warps along dim 0, holds only where
    // BlockedToDPAS applies: its function-wide check here, its shape check
    // on the dot a transpose feeds below and on the region's dots in
    // collectRegion.
    if (ttgi::DPASAnalysisFactory::canUseDPAS(
            cast<FunctionOpInterface>(func.getOperation()), dpasAnalysis) !=
        ttgi::DPASAnalysisResult::True)
      return;
    llvm::MapVector<Block *, SmallVector<tt::TransOp>> rootsPerBlock;
    func.walk([&](tt::TransOp trans) {
      // In another body, e.g. of an scf.if, the computed tiles of the
      // enclosing block would become leaves, each with its own transpose that
      // runs whether or not the body does.
      if (!isTranspose2D(trans) ||
          !isa<tt::FuncOp, scf::ForOp>(trans->getParentOp()))
        return;
      tt::DotOp dot = getDotFedAsA(trans);
      if (dot && fitsDPASShape(dot, dot.getType().getShape(), *dpasCap, mod))
        rootsPerBlock[trans->getBlock()].push_back(trans);
    });
    for (auto &[block, roots] : rootsPerBlock) {
      std::optional<TransposeRegion> region =
          collectRegion(roots, *dpasCap, mod);
      if (!region) {
        LDBG("fold-trans: region rejected in block of " << *roots.front());
        continue;
      }
      LDBG("fold-trans: rewriting " << region->ops.size() << " ops for "
                                    << region->roots.size() << " transposes");
      rewriteRegion(*region);
    }
  });
}

} // namespace

class TritonIntelGPUAccelerateMatmulPass
    : public ttgi::impl::TritonIntelGPUAccelerateMatmulBase<
          TritonIntelGPUAccelerateMatmulPass> {
public:
  using ttgi::impl::TritonIntelGPUAccelerateMatmulBase<
      TritonIntelGPUAccelerateMatmulPass>::TritonIntelGPUAccelerateMatmulBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    // Transpose dot scale operations that have a scale on the RHS.
    bool supportBlockScaleDPAS = mod->hasAttr(
        ttgi::TritonIntelGPUDialect::getSupportBlockScaleDPASAttrName());
    if (!supportBlockScaleDPAS)
      transposeDotScaledOp(mod);

    // Before the chained-dot analysis and the DPAS layout choice, so that the
    // warps are chosen for the rewritten dots.
    if (foldTrans)
      foldTransposeIntoProducers(mod);

    annotateChainedDotKinds(mod);

    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    constexpr int benefitDefault = 1;
    patterns.add<BlockedToDPAS<tt::DotOp>>(context, benefitDefault + 1,
                                           grfMode);
    if (supportBlockScaleDPAS) {
      patterns.add<BlockedToDPAS<tt::DotScaledOp>>(context, benefitDefault + 1,
                                                   grfMode);
      patterns.add<UpcastScaledBlocked>(context, benefitDefault + 1);
    }
    ttgi::populateDecomposeScaledBlockedPatterns(patterns, benefitDefault);
    if (applyPatternsGreedily(mod, std::move(patterns)).failed())
      signalPassFailure();

    // Chained-dot kind is temporary analysis metadata and should not leak.
    clearChainedDotKindAttrs(mod);

    decomposeMixedModeDotOp(mod);
  }
};
