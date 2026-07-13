#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONINTELGPUCODESINKING
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

using namespace mlir;

#define DEBUG_TYPE "tritonintelgpu-code-sinking"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

// NOTE: mirrors the file-local helper in
// lib/Dialect/TritonGPU/Transforms/ReorderInstructions.cpp; kept local to
// avoid an upstream-divergence edit.
// DIVERGENCE: upstream returns false when the effects are unknown
// (`getEffectsRecursively` == nullopt). Here we return true so that an op with
// unknown effects is treated as a write. That is the conservative (safe)
// direction for both uses of this helper: as a candidate purity test (unknown
// effects -> not a candidate, do not sink) and inside crossWriteSideEffectingOp
// (unknown intervening op -> assume it may write, do not sink across it).
//
// Writes to the `L2Cache` resource are NOT treated as memory writes: cache-fill
// prefetches (`ttig.prefetch`, `ttig.descriptor_prefetch`) declare
// `MemWrite<L2Cache>` only to keep CSE/DCE from removing them; they do not
// mutate observable memory and so cannot alias a sunk load. This mirrors the
// exclusion in AnnotateCacheControl, and is essential because the software
// pipeliner emits prefetches between the bias loads and the loop in the
// motivating SwiGLU kernel (issue #7250).
static bool hasWriteSideEffect(Operation *op) {
  std::optional<SmallVector<MemoryEffects::EffectInstance>> effects =
      getEffectsRecursively(op);
  if (!effects)
    return true; // conservative: unknown effects -> assume a write
  return llvm::any_of(*effects, [](MemoryEffects::EffectInstance effect) {
    if (isa<MemoryEffects::Read, MemoryEffects::Allocate, MemoryEffects::Free>(
            effect.getEffect()))
      return false;
    // A write to the L2 cache (a prefetch) does not mutate observable memory.
    if (effect.getResource()->getResourceID() ==
        triton::gpu::intel::L2Cache::getResourceID())
      return false;
    return true;
  });
}

// NOTE: mirrors the file-local helper in
// lib/Dialect/TritonGPU/Transforms/ReorderInstructions.cpp; kept local to
// avoid an upstream-divergence edit.
// Return true if there is a write side effect on any path between start and
// end ops. This assumes start dominates end.
static bool crossWriteSideEffectingOp(Operation *start, Operation *end) {
  Operation *ancestor = start->getBlock()->findAncestorOpInBlock(*end);
  // Couldn't find an ancestor in the same block, conservatively assume true.
  if (!ancestor)
    return true;
  Operation *nextOp = start->getNextNode();
  while (nextOp) {
    if (hasWriteSideEffect(nextOp))
      return true;
    if (nextOp == ancestor)
      return false;
    nextOp = nextOp->getNextNode();
  }
  assert(false && "op doesn't dominate other");
  return true;
}

/// Returns the insertion point (the first use after the region-bearing op) for
/// sinking `op`, or nullptr if `op` should not be sunk. `op` can be sunk only
/// when:
/// - It has no users nested inside any of `regionOp`'s regions (e.g. loop
///   body, then/else branches, while before/after regions).
/// - It has no user at or before `regionOp` in the parent block. (A user that
///   is the `regionOp` itself covers values flowing in as operands to the
///   barrier, e.g. `iter_args` init operands to an `scf.for`, or the condition
///   of an `scf.if`. Such values cannot be sunk "after" the barrier.)
/// - It has at least one user (nullptr users -> leave for DCE).
///
/// Operand dominance is preserved by construction: the op is sunk *downward*
/// within its own block, so any value that dominated it at its original
/// (earlier) position still dominates every later position. We never hoist, so
/// the op's operands need no separate dominance check.
static Operation *getSinkPoint(Operation *op, Operation *regionOp) {
  Block *parentBlock = regionOp->getBlock();
  Operation *firstUseAfter = nullptr;

  for (Operation *user : op->getUsers()) {
    // Bail if the user is nested inside any region of the barrier op.
    if (regionOp->isAncestor(user))
      return nullptr;

    // Find the ancestor of this user in the parent block.
    Operation *userAnc = parentBlock->findAncestorOpInBlock(*user);
    if (!userAnc)
      return nullptr; // conservatively bail

    // Bail if this user is at or before the barrier (moving breaks dominance).
    if (userAnc == regionOp || userAnc->isBeforeInBlock(regionOp))
      return nullptr;

    // Track the first use after the barrier.
    if (!firstUseAfter || userAnc->isBeforeInBlock(firstUseAfter))
      firstUseAfter = userAnc;
  }

  // No use at all -> leave for DCE.
  return firstUseAfter;
}

// Sink any pure operation defined before a region-bearing op (e.g. `scf.for`,
// `scf.while`, `scf.if`) whose results are unused inside the region down to
// just before their first use after the region. This shortens the live range of
// a value that would otherwise be live across the entire region, reducing
// register pressure that competes with DPAS accumulators and A/B operand tiles.
// Loads (e.g. the SwiGLU bias loads of issue #7250) are the motivating case,
// but the transform applies to any read-only / effect-free op (constants,
// splats, elementwise, etc.).
class TritonIntelGPUCodeSinkingPass
    : public triton::gpu::intel::impl::TritonIntelGPUCodeSinkingBase<
          TritonIntelGPUCodeSinkingPass> {
public:
  using triton::gpu::intel::impl::TritonIntelGPUCodeSinkingBase<
      TritonIntelGPUCodeSinkingPass>::TritonIntelGPUCodeSinkingBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    // Iterate to a fixpoint. Sinking one op can expose another as loop-dead:
    // e.g. a `load -> convert_layout -> (use after loop)` chain. In the first
    // sweep only the `convert_layout` is sinkable (the load's sole user, the
    // convert, is still before the loop); once the convert moves below the
    // loop the load becomes loop-dead and the next sweep sinks it too. A single
    // sweep would leave the load (and its live range) straddling the loop.
    bool changed = true;
    while (changed)
      changed = sinkPureOpsPastRegions(moduleOp);
  }

private:
  // Run one sweep over all region-bearing "barrier" ops, sinking every pure op
  // that is dead across the barrier to its first use after it. Returns true if
  // any op was moved.
  static bool sinkPureOpsPastRegions(ModuleOp moduleOp) {
    bool changed = false;

    moduleOp.walk([&](Operation *regionOp) {
      // A "barrier" is any op that contains a region of code we may want to
      // sink an unused value past. Skip the function/module containers and any
      // op with no regions.
      if (regionOp->getNumRegions() == 0)
        return;
      if (isa<ModuleOp>(regionOp) || isa<FunctionOpInterface>(regionOp))
        return;

      Block *parentBlock = regionOp->getBlock();

      // Collect {op, insertBefore} pairs, then move after the walk over this
      // block to avoid invalidating the iteration. Moves never erase ops.
      SmallVector<std::pair<Operation *, Operation *>> toMove;

      for (Operation &siblingOp : *parentBlock) {
        // Only consider ops before the barrier.
        if (!siblingOp.isBeforeInBlock(regionOp))
          break;

        // Need at least one result to have a "use after the barrier".
        if (siblingOp.getNumResults() == 0)
          continue;

        // Must be pure (read-only or effect-free) to be moved. This excludes
        // volatile loads and stores automatically.
        if (hasWriteSideEffect(&siblingOp))
          continue;

        Operation *sinkPoint = getSinkPoint(&siblingOp, regionOp);
        if (!sinkPoint) {
          LDBG("skip (no sink point): " << siblingOp);
          continue;
        }

        // No aliasing write on the path op -> sinkPoint. `regionOp` is one of
        // the intervening siblings and getEffectsRecursively recurses into its
        // regions, so an in-region store is detected here.
        if (crossWriteSideEffectingOp(&siblingOp, sinkPoint)) {
          LDBG("skip (intervening write): " << siblingOp);
          continue;
        }

        LDBG("sink: " << siblingOp << "\n  before: " << *sinkPoint);
        toMove.push_back({&siblingOp, sinkPoint});
      }

      for (auto &[op, insertBefore] : toMove)
        op->moveBefore(insertBefore);
      if (!toMove.empty())
        changed = true;
    });

    return changed;
  }
};

} // namespace
