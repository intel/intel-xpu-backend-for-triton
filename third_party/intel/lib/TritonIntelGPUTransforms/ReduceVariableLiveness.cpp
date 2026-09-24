#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
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

/// Return true if \p op may write to a resource that a global-memory 2D block
/// load can alias, or if its effects are unknown.
///
/// A write to shared memory (`ttg.local_alloc`/`ttg.local_store`, which the SLM
/// round trip of a `tt.trans` is lowered through) or to the L2 cache (a
/// prefetch, which declares a write only to keep CSE/DCE from removing it)
/// cannot change what a global load reads, so neither blocks moving such a load
/// past it.
bool mayWriteMemoryAliasingGlobalLoad(Operation *op) {
  std::optional<SmallVector<MemoryEffects::EffectInstance>> effects =
      getEffectsRecursively(op);
  if (!effects)
    return true; // conservative: unknown effects -> assume an aliasing write
  return llvm::any_of(
      *effects, [](const MemoryEffects::EffectInstance &effect) {
        if (!isa<MemoryEffects::Write>(effect.getEffect()))
          return false;
        SideEffects::Resource *resource = effect.getResource();
        if (isa<ttg::SharedMemory>(resource))
          return false;
        if (resource->getResourceID() == ttgi::L2Cache::getResourceID())
          return false;
        return true;
      });
}

/// Return true if any operation strictly between \p start and \p end may write
/// memory that a global load reads. \p start and \p end must be in the same
/// block, with \p start before \p end.
bool crossesAliasingWrite(Operation *start, Operation *end) {
  assert(start->getBlock() == end->getBlock() && "expecting the same block");
  for (Operation *op = start->getNextNode(); op && op != end;
       op = op->getNextNode())
    if (mayWriteMemoryAliasingGlobalLoad(op))
      return true;
  return false;
}

/// Sink 2D dot-operand loads that already sit in \p forOp's body down to just
/// before the first operation that uses them. Returns `true` if any load moved.
///
/// Intel lowers DPAS A/B operands straight from a 2D block load into registers,
/// so an operand tile loaded near the top of a loop body but consumed by a dot
/// near the bottom holds its full per-lane footprint for the whole iteration.
/// A B operand is the expensive case: with `warpsPerCTA[N] == 1` it is
/// replicated in every warp, so its live range costs `K * N * elemBytes /
/// threadsPerWarp` bytes per lane with no `num_warps` divisor.
///
/// Upstream `ReorderInstructions` shortens exactly this kind of live range, but
/// only for `ttg.local_load`/`ttg.convert_layout` -- the shared-memory staging
/// that other backends route dot operands through and that Intel does not --
/// so it never sees these loads.
///
/// Unlike `moveOperand`, this leaves nothing behind at the original position,
/// so a load that is sunk without need has its latency exposed rather than
/// overlapped. Each load is therefore checked individually against a necessary
/// condition for the sink to relieve any spilling at all: shortening a live
/// range can only lower the pressure at program points *inside* the range it
/// removes, so unless some point between the load and its first use is already
/// at or above the GRF budget, the loop's over-budget region lies entirely
/// outside what the sink shortens and the move can only cost latency.
///
/// \p analysis describes the loop as it was on entry, so once a load has moved
/// the pressure reported for the loads examined after it is stale. It is stale
/// in the safe direction only: a sink can only lower the pressure over the
/// interval it vacates, so a later load can be sunk on the strength of pressure
/// an earlier sink has already relieved, but never kept in place because of
/// pressure that is no longer there.
bool sinkInLoopDotOperandLoads(
    scf::ForOp forOp, const ttg::intel::RegisterPressureAnalysis &analysis,
    unsigned perLaneGRFBudget) {
  Block *loop = forOp.getBody();
  bool changed = false;

  SmallVector<tt::DescriptorLoadOp> loads;
  for (Operation &op : *loop)
    if (auto loadOp = dyn_cast<tt::DescriptorLoadOp>(&op))
      loads.push_back(loadOp);

  for (tt::DescriptorLoadOp loadOp : loads) {
    auto tensorType = dyn_cast<RankedTensorType>(loadOp.getResult().getType());
    if (!tensorType || tensorType.getRank() != 2)
      continue;
    if (!isa<ttg::DotOperandEncodingAttr>(tensorType.getEncoding()))
      continue;

    // The earliest user, as seen from the loop body block. A user with no
    // ancestor there lives in an unrelated region: bail rather than guess.
    Operation *firstUse = nullptr;
    bool hasOutOfBlockUser = false;
    for (Operation *user : loadOp->getUsers()) {
      Operation *ancestor = loop->findAncestorOpInBlock(*user);
      if (!ancestor) {
        hasOutOfBlockUser = true;
        break;
      }
      if (!firstUse || ancestor->isBeforeInBlock(firstUse))
        firstUse = ancestor;
    }
    if (hasOutOfBlockUser || !firstUse || firstUse == loadOp->getNextNode())
      continue;
    if (crossesAliasingWrite(loadOp, firstUse))
      continue;

    // Highest pressure over the live range this sink would remove.
    unsigned rangePressure = 0;
    for (Operation *op = loadOp; op && op != firstUse; op = op->getNextNode())
      rangePressure = std::max(rangePressure, analysis.pressureAt(op));
    if (rangePressure < perLaneGRFBudget) {
      LDBG("Keeping in-loop dot operand load in place: its live range peaks at "
           << rangePressure << " B/lane, within the " << perLaneGRFBudget
           << " B/lane GRF budget: " << *loadOp);
      continue;
    }

    LDBG("Sinking in-loop dot operand load to its first use (live range peaks "
         "at "
         << rangePressure << " B/lane, at or above the " << perLaneGRFBudget
         << " B/lane GRF budget): " << *loadOp);
    loadOp->moveBefore(firstUse);
    changed = true;
  }

  return changed;
}

/// Identifies the tile a descriptor load reads: the descriptor together with
/// the indices it is read at. Two loads of the same descriptor at different
/// indices touch different memory, so each one needs its own prefetch; keying
/// the bookkeeping on the descriptor alone would drop every prefetch but the
/// first.
using PrefetchKey = SmallVector<Value, 3>;

/// Return the prefetch key of \p loadOp.
PrefetchKey getPrefetchKey(tt::DescriptorLoadOp loadOp) {
  PrefetchKey key{loadOp.getDesc()};
  llvm::append_range(key, loadOp.getIndices());
  return key;
}

/// Create a prefetch operation for the given load operation.
void createPrefetchOp(tt::DescriptorLoadOp loadOp) {
  OpBuilder builder(loadOp);
  ttgi::CachePolicy cachePolicy =
      ttgi::getCachePolicy(loadOp.getCachePolicyAttr());
  auto prefetchOp = ttgi::DescriptorPrefetchOp::create(
      builder, loadOp->getLoc(), loadOp.getDesc(), loadOp.getIndices(),
      cachePolicy.cacheModifier, cachePolicy.evictionPolicy);

  // inherit attributes from the load operation
  auto attrs = loadOp->getAttrDictionary();
  prefetchOp->setAttrs(attrs);
}

/// Investigate opportunities for the reducing register pressure by moving DotOp
/// operands.
/// Loads already inside the loop are left in place when \p disableInLoopSink.
/// Returns `true` if at least one operand has been moved.
bool optimizeDotOperands(scf::ForOp forOp,
                         SmallVector<PrefetchKey> &prefetchedTiles,
                         ttg::intel::RegisterPressureAnalysis &analysis,
                         unsigned perLaneGRFBudget, bool disableInLoopSink) {
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

  // Gate on the *peak* pressure of the loop body rather than on its live-in
  // pressure: live-in pressure is computed from `LivenessBlockInfo::in()`,
  // which by construction excludes the block arguments, so it does not see the
  // loop-carried values -- including the DPAS accumulator, frequently the
  // largest live tensor in the loop. Peak pressure is what determines whether
  // the register allocator has to spill, which is the condition under which
  // trading a redundant (but prefetched and cached) 2D block load for a
  // shorter live range pays off.
  //
  // For the candidates handled by `moveOperand` the decision is per loop: sink
  // every eligible operand or none. Choosing a subset would need a model of how
  // much a given sink actually lowers the peak, which depends on where the peak
  // sits relative to each live range. Nothing here measures that, so a partial
  // choice would be arbitrary rather than selective -- and it would buy little,
  // since `moveOperand` leaves a prefetch behind and so costs almost nothing
  // when it sinks an operand that did not need sinking. The in-loop sink below
  // has no such fallback and is therefore gated per load instead; see
  // `sinkInLoopDotOperandLoads`.
  unsigned peakPressurePerLane = analysis.peakPressure(loop);
  if (peakPressurePerLane < perLaneGRFBudget) {
    LDBG("Keeping " << candidates.size()
                    << " dot operand(s) in place: peak pressure "
                    << peakPressurePerLane << " B/lane is within the "
                    << perLaneGRFBudget << " B/lane GRF budget");
    return false;
  }

  LDBG("Shortening dot operand live ranges ("
       << candidates.size() << " operand(s) to sink into the loop): peak "
       << "pressure " << peakPressurePerLane << " B/lane is at or above the "
       << perLaneGRFBudget << " B/lane GRF budget");
  // Operands already loaded inside the loop are not reached by `moveOperand`,
  // but their live range within one iteration is just as expensive; shorten
  // those too, per load (see `sinkInLoopDotOperandLoads`). Done first, while
  // `analysis` still describes the IR exactly: `moveOperand` inserts loads and
  // prefetches the analysis has no pressure information for.
  bool sunkInLoop =
      !disableInLoopSink &&
      sinkInLoopDotOperandLoads(forOp, analysis, perLaneGRFBudget);

  for (Candidate &c : candidates)
    moveOperand(c.opId, c.dot, c.loadOp);

  return !candidates.empty() || sunkInLoop;
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
    // Sinking is gated on the budget as a *threshold* to act, not a ceiling,
    // so an unknown ("default"/"auto") GRF size must assume the largest the
    // device supports -- see UnknownGRFSizeAssumption's documentation.
    unsigned perLaneGRFBudget =
        ttgi::RegisterPressureAnalysis::getPerLaneGRFBudgetInBytes(
            grfMode, mod,
            ttgi::RegisterPressureAnalysis::UnknownGRFSizeAssumption::Largest);
    ttg::intel::RegisterPressureAnalysis analysis(rootOperation);
    // TODO: extend the pass to handle `while` loops.
    rootOperation->walk([&](scf::ForOp forOp) {
      if (optimizeDotOperands(forOp, prefetchedTiles, analysis,
                              perLaneGRFBudget, disableInLoopSink)) {
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
