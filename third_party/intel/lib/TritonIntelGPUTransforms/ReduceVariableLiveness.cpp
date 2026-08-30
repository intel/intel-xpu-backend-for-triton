#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#include "Dialect/TritonIntelGPU/IR/Attributes.h"
#include "Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "intel/include/Analysis/DPAS.h"
#include "intel/include/Analysis/RegisterPressure.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include <algorithm>
#include <array>
#include <optional>

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONINTELGPUREDUCEVARIABLELIVENESS
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;

using TensorValue = TypedValue<RankedTensorType>;

#define DEBUG_TYPE "tritonintelgpu-reduce-variable-liveness"

namespace {

// A load is worth sinking (moving closer to its use, with a cheap prefetch
// left behind) when the loop it would otherwise stay live across is under
// enough register pressure to make trading a redundant-but-cheap 2D block
// load for freed registers a real win. `RegisterPressureAnalysis::
// liveInPressure` reports bytes **per lane** (see RegisterPressure.h), so the
// floor below must be expressed in the same per-lane unit rather than the
// per-*hardware-thread* unit `getGRFBytesPerThread` returns on its own --
// see `getPerLaneGRFBudgetInBytes`, which does that conversion using the
// module's actual threads-per-warp rather than an assumed constant.
constexpr uint32_t LIVE_IN_PRESSURE_GRF_BUDGET_MULTIPLIER = 2; // 200%

/// The set of `grf-mode` values `getGRFBytesPerThread` assigns a real,
/// mode-specific budget to. Anything else silently collapses to its
/// "default" fallback (4096 bytes/thread) inside that function, so an
/// unrecognized value is caught and warned about here instead.
constexpr std::array<StringRef, 5> VALID_GRF_MODES = {"default", "auto", "128",
                                                      "256", "512"};

/// Convert the per-hardware-thread GRF budget for \p grfMode into a per-lane
/// figure, dividing by the module's actual threads-per-warp so the result is
/// in the same unit `RegisterPressureAnalysis::liveInPressure` reports.
/// Falls back to the unscaled per-thread budget (with a diagnostic) if
/// threads-per-warp is missing or non-positive, rather than dividing by zero.
unsigned getPerLaneGRFBudgetInBytes(StringRef grfMode, ModuleOp mod) {
  if (!llvm::is_contained(VALID_GRF_MODES, grfMode))
    mod.emitWarning("unrecognized grf-mode '" + grfMode +
                    "' for tritonintelgpu-reduce-variable-liveness; "
                    "falling back to the 'default' GRF budget");
  unsigned grfBudget =
      ttg::intel::RegisterPressureAnalysis::getGRFBytesPerThread(grfMode);
  int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(mod);
  if (threadsPerWarp <= 0) {
    mod.emitWarning(
        "ttg.threads-per-warp is missing or non-positive; cannot convert "
        "the per-hardware-thread GRF budget to a per-lane figure, falling "
        "back to the unscaled per-thread budget for tritonintelgpu-"
        "reduce-variable-liveness");
    return grfBudget;
  }
  return grfBudget / static_cast<unsigned>(threadsPerWarp);
}

/// Return true if the lifespan of the \p v value is considered long.
bool isLongLifeSpanVariable(
    Value v, const ttg::intel::RegisterPressureAnalysis &analysis,
    Block *dotBlock, unsigned perLaneGRFBudget) {
  // The variable is considered as a long life span elected for being moved if
  // it is a 2D tensor, it is genuinely live-in to the block (defined outside,
  // used inside -- i.e. it would otherwise stay resident across the whole
  // loop), and the loop is under enough measured register pressure that
  // shortening this value's live range is worth the redundant reload.
  TensorValue tensorV = dyn_cast<TensorValue>(v);
  if (!tensorV)
    return false;

  auto tensorType = cast<RankedTensorType>(tensorV.getType());
  auto tensorOrder = ttg::getOrder(tensorType);
  unsigned liveInSizeInBytes = analysis.liveInPressure(dotBlock);
  return ((tensorOrder.size() == 2) &&
          (liveInSizeInBytes >=
           perLaneGRFBudget * LIVE_IN_PRESSURE_GRF_BUDGET_MULTIPLIER) &&
          analysis.isLiveIn(dotBlock, v));
}

/// Return true if the \p loadOp is suitable to be moved.
/// \p expectedElementType is the element type expected for the load to be a
/// candidate,
/// \p forOp operation to which we want to move the loadOp
bool isLoadCandidate(tt::DescriptorLoadOp loadOp, Type expectedElementType,
                     Operation *forOp) {
  Value loadSource = loadOp.getDesc();
  auto loadType = cast<RankedTensorType>(loadOp.getResult().getType());
  Type loadElType = loadType.getElementType();
  // Types mismatch => Skip this case to avoid inserting too
  // many addtional operations in the loop.
  if (expectedElementType != loadElType)
    return false;
  Attribute blockIOAttr =
      loadOp->getAttr(ttgi::TritonIntelGPUDialect::getBlockIOAttrName());
  if (!blockIOAttr)
    return false;
  // Only tensor with rank = 2 are considered to be moved
  if (loadType.getShape().size() != 2)
    return false;
  // Only loadOp out of the for loop body are considered to be moved
  if (loadOp->getParentOp() == forOp)
    return false;
  // Multiple users
  if (any_of(loadOp->getUsers(), [&](Operation *user) {
        return ((user->getBlock() == forOp->getBlock()) &&
                user->isBeforeInBlock(forOp));
      }))
    return false;
  // We skip the load if the defining op is not is the same region.
  // To avoid prefetching this data in another region
  // (as the prefetch is added after the defining op).
  if (!loadSource.getDefiningOp())
    return false;
  return true;
}

/// Create a prefetch operation for the given load operation.
void createPrefetchOp(tt::DescriptorLoadOp loadOp) {
  OpBuilder builder(loadOp);
  auto prefetchOp = ttgi::DescriptorPrefetchOp::create(
      builder, loadOp->getLoc(), loadOp.getDesc(), loadOp.getIndices(),
      loadOp.getCache(), loadOp.getEvict());

  // inherit attributes from the load operation
  auto attrs = loadOp->getAttrDictionary();
  prefetchOp->setAttrs(attrs);
}

/// Investigate opportunities for the reducing register pressure by moving DotOp
/// operands.
/// Returns `true` if at least one operand has been moved.
bool optimizeDotOperands(scf::ForOp forOp, SmallVector<Value> &prefetchedValue,
                         ttg::intel::RegisterPressureAnalysis &analysis,
                         unsigned perLaneGRFBudget) {
  Block *loop = forOp.getBody();
  bool opMoved = false;

  // Returns the DescriptorLoadOp that produces the value v, walking back
  // through ConvertLayoutOps. Returns nullptr if no DescriptorLoadOp is found.
  auto getLoad = [](Value v) -> tt::DescriptorLoadOp {
    Operation *op = v.getDefiningOp();
    while (op) {
      if (auto load = dyn_cast<tt::DescriptorLoadOp>(op))
        return load;
      if (!isa<ttg::ConvertLayoutOp>(op))
        break;
      op = op->getOperand(0).getDefiningOp();
    }
    return nullptr;
  };

  // Prefetch the dotOp operand and move it closer to dotOp.
  auto moveOperand = [&prefetchedValue, &opMoved](uint8_t opId, tt::DotOp dotOp,
                                                  tt::DescriptorLoadOp loadOp) {
    assert(opId < 2 && "opId must be 0 or 1");
    OpBuilder b(dotOp);
    TensorValue tensorV = opId == 0 ? dotOp.getA() : dotOp.getB();
    auto tensorType = cast<RankedTensorType>(tensorV.getType());
    Operation *insertBeforeOp = dotOp;
    SmallVector<Operation *> usesInSameLoop;
    // Other use(s) in the same loop
    for (Operation *user : loadOp->getUsers()) {
      if (user == dotOp)
        continue;
      if (user->getParentOp() == dotOp->getParentOp()) {
        usesInSameLoop.push_back(user);
        if (user->isBeforeInBlock(insertBeforeOp))
          insertBeforeOp = user;
      }
    }

    Value prefetchKey = loadOp.getDesc();
    if (std::find(prefetchedValue.begin(), prefetchedValue.end(),
                  prefetchKey) == prefetchedValue.end()) {
      createPrefetchOp(loadOp);
      prefetchedValue.push_back(prefetchKey);
    }
    b.setInsertionPoint(insertBeforeOp);
    auto *newLoad = b.clone(*loadOp);
    auto newCvt = ttg::ConvertLayoutOp::create(b, tensorV.getLoc(), tensorType,
                                               newLoad->getResult(0));
    dotOp.setOperand(opId, newCvt.getResult());

    // Update other user in the same loop if any
    for (Operation *user : usesInSameLoop)
      user->replaceUsesOfWith(loadOp->getResult(0), newLoad->getResult(0));

    // Multiple users: rematerialize the load after the loop for the users that
    // such a copy would dominate, so that the original load dies before the
    // loop instead of staying live across it.
    //
    // Only users the copy dominates may be rewired. A user nested in a region
    // that precedes the loop -- e.g. the body of an earlier sibling loop, as in
    // causal attention where one Q load feeds a dot in each of two loops -- is
    // not dominated by a definition placed after this loop, and must keep using
    // the original load. `isLoadCandidate` only rejects users sitting directly
    // in the loop's own block before the loop, so nested users reach here.
    if (!loadOp->use_empty()) {
      Operation *loopOp = dotOp->getParentOp();
      Block *loopBlock = loopOp->getBlock();
      // A use is dominated by the copy iff its ancestor in the loop's block
      // comes after the loop. Uses with no ancestor there live in an unrelated
      // region and conservatively keep the original load.
      auto dominatedByCopy = [&](OpOperand &use) {
        Operation *ancestor = loopBlock->findAncestorOpInBlock(*use.getOwner());
        return ancestor && loopOp->isBeforeInBlock(ancestor);
      };
      if (any_of(loadOp->getResult(0).getUses(), dominatedByCopy)) {
        b.setInsertionPointAfter(loopOp);
        auto *copyLoad = b.clone(*loadOp);
        loadOp->getResult(0).replaceUsesWithIf(copyLoad->getResult(0),
                                               dominatedByCopy);
      }
    }
    opMoved = true;
  };

  // Try to match and move a dot operand sourced from a descriptor load.
  auto tryMoveOperand = [&](uint8_t opId, tt::DotOp dot, Value operand,
                            Operation *forOp) {
    tt::DescriptorLoadOp loadOp = getLoad(operand);
    if (!loadOp)
      return;
    Block *dotBlock = dot->getBlock();
    // Check liveness on the load's result, not the dot operand, because the
    // dot operand may be a ConvertLayoutOp result (possibly inside the loop)
    // while the load result is the truly long-lived value defined outside.
    Value loadResult = loadOp->getResult(0);
    if (!isLongLifeSpanVariable(loadResult, analysis, dotBlock,
                                perLaneGRFBudget))
      return;
    auto tensorType = cast<RankedTensorType>(operand.getType());
    Type elTy = tensorType.getElementType();
    if (isLoadCandidate(loadOp, elTy, forOp))
      moveOperand(opId, dot, loadOp);
  };

  SmallVector<tt::DotOp> dotsInFor;
  for (Operation &op : *loop)
    if (auto dotOp = dyn_cast<tt::DotOp>(op)) {
      // Only accepts dotOps encoded as DPAS MMA
      if (!ttgi::hasDpasEncoding(dotOp.getResult().getType()))
        // Don't rewrite if any other type is found.
        return false;
      dotsInFor.push_back(dotOp);
    }

  if (dotsInFor.empty())
    return false;

  for (tt::DotOp dot : dotsInFor) {
    tryMoveOperand(0, dot, dot.getA(), forOp);
    tryMoveOperand(1, dot, dot.getB(), forOp);
  }
  return opMoved;
}

class ReduceVariableLivenessPass
    : public triton::gpu::intel::impl::TritonIntelGPUReduceVariableLivenessBase<
          ReduceVariableLivenessPass> {
public:
  using triton::gpu::intel::impl::TritonIntelGPUReduceVariableLivenessBase<
      ReduceVariableLivenessPass>::TritonIntelGPUReduceVariableLivenessBase;

  void runOnOperation() override {
    // Canonicalize convert ops to make the pattern matching easier.
    SmallVector<Value> prefetchedValue;
    RewritePatternSet cleanUpPatterns(&getContext());
    ttg::ConvertLayoutOp::getCanonicalizationPatterns(cleanUpPatterns,
                                                      &getContext());
    if (mlir::applyPatternsGreedily(getOperation(), std::move(cleanUpPatterns))
            .failed()) {
      signalPassFailure();
    }

    Operation *rootOperation = getOperation();
    ModuleOp mod = getOperation();
    unsigned perLaneGRFBudget = getPerLaneGRFBudgetInBytes(grfMode, mod);
    ttg::intel::RegisterPressureAnalysis analysis(rootOperation);
    // TODO: extend the pass to handle `while` loops.
    rootOperation->walk([&](scf::ForOp forOp) {
      if (optimizeDotOperands(forOp, prefetchedValue, analysis,
                              perLaneGRFBudget)) {
        // The register pressure analysis must be re-performed before the
        // processing of each "for loop" given that the liveness of variables
        // may have changed as a result of the code, and specifically `LoadOps`,
        // being modified by the pass.
        analysis = ttg::intel::RegisterPressureAnalysis(rootOperation);
        return;
      }
    });
  }
};

} // namespace
