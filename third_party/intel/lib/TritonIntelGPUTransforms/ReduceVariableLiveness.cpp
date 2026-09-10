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
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

/// Convert the per-hardware-thread GRF budget for \p grfMode into a per-lane
/// budget by dividing by threads-per-warp.
///
/// The two quantities are expressed in different units:
/// `getGRFBytesPerThread` reports the register file of a whole *hardware
/// thread*, which backs an entire sub-group, while
/// `RegisterPressureAnalysis` weighs every live value by
/// `getTotalElemsPerThread`, a per-*work-item* (per-lane) count. Comparing
/// them without this conversion overstates the budget by a factor of
/// threads-per-warp. getThreadsPerWarp() falls back to 32 when the module
/// attribute is absent, so the division is always well-defined.
unsigned getPerLaneGRFBudgetInBytes(StringRef grfMode, ModuleOp mod) {
  unsigned grfBudget =
      ttg::intel::RegisterPressureAnalysis::getGRFBytesPerThread(grfMode);
  int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(mod);
  assert(threadsPerWarp > 0 && "threads-per-warp must be positive");
  return grfBudget / static_cast<unsigned>(threadsPerWarp);
}

/// Return true if \p v is a 2D tensor that is live-in to \p loopBody (defined
/// outside the loop and used inside it), i.e. a value that would otherwise
/// occupy registers for the whole duration of the loop.
bool isLiveIn2DTensor(Value v,
                      const ttg::intel::RegisterPressureAnalysis &analysis,
                      Block *loopBody) {
  auto tensorType = dyn_cast<RankedTensorType>(v.getType());
  return tensorType && tensorType.getRank() == 2 &&
         analysis.isLiveIn(loopBody, v);
}

/// A prefetch is identified by the descriptor *and* the indices it is read at:
/// the same descriptor read at two different offsets denotes two different
/// tiles, each of which needs its own prefetch.
using PrefetchKey = SmallVector<Value, 3>;

PrefetchKey getPrefetchKey(tt::DescriptorLoadOp loadOp) {
  PrefetchKey key{loadOp.getDesc()};
  llvm::append_range(key, loadOp.getIndices());
  return key;
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
  // A user nested in a region *inside* the loop (e.g. an `scf.if` body) cannot
  // be redirected to the sunk copy: `moveOperand` only rewires uses sitting
  // directly in the loop body block, and the after-loop copy does not dominate
  // it either. Such a use keeps the original load live across the whole loop,
  // so sinking would add a redundant load and a prefetch while relieving no
  // register pressure at all.
  if (any_of(loadOp->getUsers(), [&](Operation *user) {
        return user->getParentOp() != forOp && forOp->isAncestor(user);
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
bool optimizeDotOperands(scf::ForOp forOp,
                         SmallVector<PrefetchKey> &prefetchedTiles,
                         ttg::intel::RegisterPressureAnalysis &analysis,
                         unsigned perLaneGRFBudget) {
  Block *loop = forOp.getBody();

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

  // The in-loop copy created for a given load, so that a load feeding several
  // dot operands is cloned once instead of once per operand. Cloning it per
  // operand would emit several identical 2D block loads per iteration -- the
  // opposite of what this pass is for.
  DenseMap<Operation *, Operation *> sunkLoads;

  // Prefetch the dotOp operand and move it closer to dotOp.
  auto moveOperand = [&](uint8_t opId, tt::DotOp dotOp,
                         tt::DescriptorLoadOp loadOp) {
    assert(opId < 2 && "opId must be 0 or 1");
    OpBuilder b(dotOp);
    TensorValue tensorV = opId == 0 ? dotOp.getA() : dotOp.getB();
    auto tensorType = cast<RankedTensorType>(tensorV.getType());

    // Already sunk for another operand: reuse the copy. It was inserted before
    // the earliest in-loop user of the load, so it dominates this dot.
    if (Operation *sunkLoad = sunkLoads.lookup(loadOp)) {
      // Nothing to do when this operand already reads from the copy: making the
      // copy rewires every in-loop user of the load, which may well include the
      // op feeding this operand.
      if (getLoad(tensorV).getOperation() == sunkLoad)
        return;
      Value operand = sunkLoad->getResult(0);
      if (operand.getType() != tensorType)
        operand = ttg::ConvertLayoutOp::create(b, tensorV.getLoc(), tensorType,
                                               operand)
                      .getResult();
      dotOp.setOperand(opId, operand);
      return;
    }

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

    PrefetchKey prefetchKey = getPrefetchKey(loadOp);
    if (!llvm::is_contained(prefetchedTiles, prefetchKey)) {
      createPrefetchOp(loadOp);
      prefetchedTiles.push_back(prefetchKey);
    }
    b.setInsertionPoint(insertBeforeOp);
    auto *newLoad = b.clone(*loadOp);
    sunkLoads.try_emplace(loadOp, newLoad);
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
  };

  // One entry per dot operand that could take its value from an in-loop copy of
  // a load defined outside the loop. Two entries may name the same load; it is
  // `moveOperand` that keeps the load itself to a single copy (see
  // `sunkLoads`), because every operand still needs its own layout conversion.
  struct Candidate {
    uint8_t opId;
    tt::DotOp dot;
    tt::DescriptorLoadOp loadOp;
  };
  SmallVector<Candidate> candidates;

  auto collectOperand = [&](uint8_t opId, tt::DotOp dot, Value operand) {
    tt::DescriptorLoadOp loadOp = getLoad(operand);
    if (!loadOp)
      return;
    // Check liveness on the load's result, not on the dot operand: the operand
    // may be a ConvertLayoutOp result defined inside the loop, while the load
    // result is the value that is actually live across the whole loop.
    if (!isLiveIn2DTensor(loadOp.getResult(), analysis, loop))
      return;
    auto tensorType = cast<RankedTensorType>(operand.getType());
    if (!isLoadCandidate(loadOp, tensorType.getElementType(), forOp))
      return;
    candidates.push_back({opId, dot, loadOp});
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
    collectOperand(0, dot, dot.getA());
    collectOperand(1, dot, dot.getB());
  }

  if (candidates.empty())
    return false;

  // Gate on the *peak* pressure of the loop body rather than on its live-in
  // pressure: live-in pressure is computed from `LivenessBlockInfo::in()`,
  // which by construction excludes the block arguments, so it does not see the
  // loop-carried values -- including the DPAS accumulator, frequently the
  // largest live tensor in the loop. Peak pressure is what determines whether
  // the register allocator has to spill, which is the condition under which
  // trading a redundant (but prefetched and cached) 2D block load for a
  // shorter live range pays off.
  //
  // The decision is per loop: sink every eligible operand or none. Choosing a
  // subset would need a model of how much a given sink actually lowers the
  // peak, which depends on where the peak sits relative to each live range.
  // Nothing here measures that, so a partial choice would be arbitrary rather
  // than selective.
  unsigned peakPressurePerLane = analysis.peakPressure(loop);
  if (peakPressurePerLane < perLaneGRFBudget) {
    LDBG("Keeping " << candidates.size()
                    << " dot operand(s) in place: peak pressure "
                    << peakPressurePerLane << " B/lane is within the "
                    << perLaneGRFBudget << " B/lane GRF budget");
    return false;
  }

  LDBG("Sinking " << candidates.size() << " dot operand(s): peak pressure "
                  << peakPressurePerLane << " B/lane is at or above the "
                  << perLaneGRFBudget << " B/lane GRF budget");
  for (Candidate &c : candidates)
    moveOperand(c.opId, c.dot, c.loadOp);

  return true;
}

class ReduceVariableLivenessPass
    : public triton::gpu::intel::impl::TritonIntelGPUReduceVariableLivenessBase<
          ReduceVariableLivenessPass> {
public:
  using triton::gpu::intel::impl::TritonIntelGPUReduceVariableLivenessBase<
      ReduceVariableLivenessPass>::TritonIntelGPUReduceVariableLivenessBase;

  void runOnOperation() override {
    // Canonicalize convert ops to make the pattern matching easier.
    SmallVector<PrefetchKey> prefetchedTiles;
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
      if (optimizeDotOperands(forOp, prefetchedTiles, analysis,
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
