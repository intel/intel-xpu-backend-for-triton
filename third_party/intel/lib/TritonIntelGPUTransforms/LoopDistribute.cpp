#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONINTELGPULOOPDISTRIBUTE
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

using namespace mlir;
namespace tt = mlir::triton;

#define DEBUG_TYPE "tritonintelgpu-loop-distribute"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

STATISTIC(NumLoopsDistributed, "Number of loops distributed");

namespace {

/// Computes the backward slice of `root` -- i.e. the ops it transitively
/// depends on, including its own defining op -- following values captured
/// from inside nested regions (e.g. an `scf.if` that reads a value defined
/// earlier in the loop body via a plain SSA reference into its region,
/// rather than as an explicit operand of the `scf.if` itself). Every op in
/// the raw slice is mapped to its ancestor that is a direct child of
/// `loopBody` (an op nested inside e.g. an `scf.if` maps to that `scf.if`
/// itself, since that is what actually gets cloned at the top level). Ops
/// defined above the loop are dropped -- there is nothing to clone for them.
void collectBackwardSlice(Value root, Block *loopBody,
                          DenseSet<Operation *> &slice) {
  SetVector<Operation *> rawSlice;
  BackwardSliceOptions options;
  // The induction variable and iter_args are handled explicitly by the
  // caller, not by walking into them here.
  options.omitBlockArguments = true;
  // Follow values captured from above by nested regions (e.g. an scf.if
  // that reads a loop-body value without passing it as a region operand).
  options.omitUsesFromAbove = false;
  // Include root's own defining op, not just its transitive defs.
  options.inclusive = true;
  (void)getBackwardSlice(root, &rawSlice, options);

  for (Operation *op : rawSlice) {
    if (Operation *anchor = loopBody->findAncestorOpInBlock(*op))
      slice.insert(anchor);
  }
}

/// Returns true if `op` may be executed more than once without changing
/// program behavior (i.e., it is safe to clone into both distributed loops).
/// An op whose only memory effects are reads qualifies -- repeating a read is
/// observationally equivalent -- and this is checked recursively, so a region
/// op (e.g. an `scf.if`) that merely loads is replicable even though it is not
/// pure. Writes, allocations and unknown effects are not replicable. Note that
/// a volatile `tt.load` declares a Write effect (see `LoadOp::getEffects`)
/// precisely so that it is neither duplicated nor reordered, so it is rejected
/// here.
bool isReplicable(Operation *op) {
  if (isPure(op))
    return true;
  std::optional<SmallVector<MemoryEffects::EffectInstance>> effects =
      getEffectsRecursively(op);
  if (!effects)
    return false;
  return llvm::all_of(*effects,
                      [](const MemoryEffects::EffectInstance &effect) {
                        return isa<MemoryEffects::Read>(effect.getEffect());
                      });
}

/// Returns true if any op in `slice` -- including inside nested regions --
/// has `val` among its operands.
bool sliceUsesValue(const DenseSet<Operation *> &slice, Value val) {
  return llvm::any_of(slice, [&](Operation *op) {
    bool found = false;
    op->walk([&](Operation *nested) {
      if (llvm::is_contained(nested->getOperands(), val)) {
        found = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return found;
  });
}

/// Returns true if the chain feeding a loop-carried `iter_arg` reads `arg`:
/// either the yielded value *is* `arg`, or some op in the chain's backward
/// `slice` -- including inside nested regions -- has `arg` as an operand.
///
/// The first disjunct is load bearing twice over. A slot that yields a block
/// argument directly has no defining op in the loop body, so
/// `collectBackwardSlice` returns nothing for it: the value-rooted
/// `getBackwardSlice` re-roots at the block argument's parent op (i.e. the
/// `scf.for` itself) and every op it reaches is then dropped as being outside
/// the loop body. A slice-only test would see such a chain as dependency free.
/// It is also what catches a slot that aliases another carried slot by yielding
/// its block argument unchanged.
bool chainReadsArg(Value carriedVal, const DenseSet<Operation *> &slice,
                   BlockArgument arg) {
  return carriedVal == arg || sliceUsesValue(slice, arg);
}

/// Which distributed loop may compute a loop-carried `iter_arg`. `Both` means
/// the slot is computed identically in the two new loops; `Dot0`/`Dot1` mean
/// only the loop computing that dot can compute it, because the chain feeding
/// the slot reads that dot's accumulator -- which is live in that loop alone.
enum class Owner { Both, Dot0, Dot1 };

/// Try to distribute a for loop with exactly two dot operations into two
/// separate loops. Returns true if the transformation was applied.
bool tryDistributeLoop(scf::ForOp forOp) {
  Block *body = forOp.getBody();

  // Collect all dot operations in the loop body (top-level only).
  SmallVector<tt::DotOp> dots;
  for (Operation &op : *body) {
    if (auto dot = dyn_cast<tt::DotOp>(op))
      dots.push_back(dot);
  }

  // Only handle exactly 2 dots.
  if (dots.size() != 2) {
    LDBG("Skipping loop: does not have exactly 2 dots (has " << dots.size()
                                                             << ")");
    return false;
  }

  tt::DotOp dot0 = dots[0];
  tt::DotOp dot1 = dots[1];

  // Each dot must consume an iter_arg as its accumulator (operand C) and
  // yield back to the same iter_arg position.
  auto yieldOp = cast<scf::YieldOp>(body->getTerminator());

  // Find which iter_arg index each dot accumulates into.
  // The accumulator (C operand) of each dot should be a block argument
  // (iter_arg), and the dot result should be yielded back.
  auto getAccIterArgIndex = [&](tt::DotOp dot) -> std::optional<unsigned> {
    Value accum = dot.getC();
    // The accumulator should be either an iter_arg directly or produced by
    // an op chain from an iter_arg. For simplicity, require it to be a
    // direct block argument.
    auto blockArg = dyn_cast<BlockArgument>(accum);
    if (!blockArg || blockArg.getOwner() != body)
      return std::nullopt;
    // iter_args start at index 1 (index 0 is the induction variable).
    unsigned iterArgIdx = blockArg.getArgNumber() - 1;

    // Verify the dot result is yielded back to this position.
    Value dotResult = dot.getResult();
    // The yield operand at this index should be the dot result (possibly
    // through the same value).
    if (yieldOp.getOperand(iterArgIdx) != dotResult)
      return std::nullopt;

    return iterArgIdx;
  };

  auto idx0 = getAccIterArgIndex(dot0);
  auto idx1 = getAccIterArgIndex(dot1);
  if (!idx0 || !idx1) {
    LDBG("Skipping loop: dot accumulators are not direct iter_args");
    return false;
  }
  if (*idx0 == *idx1) {
    LDBG("Skipping loop: both dots accumulate into the same iter_arg");
    return false;
  }

  LDBG("Found distributable loop with dots at iter_arg indices "
       << *idx0 << " and " << *idx1);

  // Compute the backward slice for each dot (ops that feed A/B operands).
  DenseSet<Operation *> slice0, slice1;
  collectBackwardSlice(dot0.getA(), body, slice0);
  collectBackwardSlice(dot0.getB(), body, slice0);
  collectBackwardSlice(dot1.getA(), body, slice1);
  collectBackwardSlice(dot1.getB(), body, slice1);

  // Check that the two slices don't have conflicting dependencies
  // (i.e., one dot's result feeds into the other dot's inputs).
  // The slices may share ops (e.g., a shared load for operand A).
  if (slice0.contains(dot1.getOperation()) ||
      slice1.contains(dot0.getOperation())) {
    LDBG("Skipping loop: dots have inter-dependencies");
    return false;
  }

  BlockArgument accArg0 = forOp.getRegionIterArgs()[*idx0];
  BlockArgument accArg1 = forOp.getRegionIterArgs()[*idx1];

  // A dot's operand slice is rooted *at* its A/B operands, so the dot itself
  // is never a member of that slice -- check its operands directly too. Its C
  // operand is its own accumulator, which is never frozen in its own loop, so
  // scanning all three operands cannot over-reject.
  auto dotReadsArg = [](tt::DotOp dot, const DenseSet<Operation *> &slice,
                        BlockArgument arg) {
    return llvm::is_contained(dot->getOperands(), arg) ||
           sliceUsesValue(slice, arg);
  };

  // Classify each iter_arg by which new loop may compute it. Each accumulator
  // is owned by its own dot's loop by construction.
  SmallVector<Owner> owners(forOp.getNumRegionIterArgs(), Owner::Both);
  owners[*idx0] = Owner::Dot0;
  owners[*idx1] = Owner::Dot1;

  // Every other (non-accumulator) iter_arg is a true pass-through (yielded
  // unchanged), a chain safe to replicate into both new loops, or a chain that
  // reads exactly one dot's accumulator and therefore belongs to that dot's
  // loop only. A chain that depends on a dot's *result*, or that reads *both*
  // accumulators, fits in neither loop, so reject the whole loop.
  //
  // Ownership is seeded from the two accumulators and deliberately *not*
  // propagated transitively through other iter_args: this is one-hop ownership
  // plus the frozen-slot rejection below, not a general ownership analysis.
  // A fixpoint would distribute strictly more loops, but it would also
  // invalidate the overlap invariant below -- an op in a `Both` slice would no
  // longer be provably accumulator free, so an op shared between a `Dot0`- and
  // a `Dot1`-owned chain would become a new hazard needing a new check. That is
  // a soundness rewrite, not a refactor. `owners` is structured so a worklist
  // could later produce it without disturbing anything downstream.
  //
  // Overlap between chains is benign. Backward slices are transitively closed
  // within the body, so if any op in a chain's slice transitively reads an
  // accumulator, the op that reads it directly is *also* in that slice -- as
  // itself, or as the top-level anchor (e.g. the `scf.if`) that
  // `sliceUsesValue` descends into. Hence every op reached through a `Both`
  // chain is provably accumulator free, and two chains sharing an op agree on
  // ownership unless the accumulator read comes from a non-shared op -- in
  // which case the shared op is safe in both loops.
  //
  // The slices are kept per index rather than merged: the frozen-slot check
  // below is per chain, so a union would not be enough.
  SmallVector<DenseSet<Operation *>> carriedSlices(
      forOp.getNumRegionIterArgs());
  for (unsigned i = 0, e = forOp.getNumRegionIterArgs(); i != e; ++i) {
    if (i == *idx0 || i == *idx1)
      continue;

    Value carriedVal = yieldOp.getOperand(i);
    if (carriedVal == forOp.getRegionIterArgs()[i])
      continue; // True pass-through: nothing to clone, no slice needed.

    DenseSet<Operation *> &carriedSlice = carriedSlices[i];
    collectBackwardSlice(carriedVal, body, carriedSlice);

    if (carriedSlice.contains(dot0.getOperation()) ||
        carriedSlice.contains(dot1.getOperation())) {
      LDBG("Skipping loop: carried iter_arg " << i
                                              << " depends on a dot result");
      return false;
    }

    bool readsAcc0 = chainReadsArg(carriedVal, carriedSlice, accArg0);
    bool readsAcc1 = chainReadsArg(carriedVal, carriedSlice, accArg1);
    if (readsAcc0 && readsAcc1) {
      LDBG("Skipping loop: carried iter_arg "
           << i
           << " depends on both dot accumulators, so neither new loop "
              "can compute it");
      return false;
    }
    // A slot that *is* an accumulator block argument is owned too, not
    // rejected: the original semantics are `iter_arg(k+1) = acc(k)`, a
    // one-iteration lag that the owner loop reproduces bit for bit, since the
    // accumulator evolves there through exactly the original sequence.
    if (readsAcc0)
      owners[i] = Owner::Dot0;
    else if (readsAcc1)
      owners[i] = Owner::Dot1;
  }

  // A slot is frozen in a new loop exactly when the *other* loop owns it: this
  // loop yields it unchanged, so it holds the loop-invariant init value here.
  auto isFrozenIn = [&owners](unsigned i, Owner loopOwner) {
    return owners[i] != Owner::Both && owners[i] != loopOwner;
  };

  // Nothing placed in a new loop may read a slot that is frozen there: it would
  // see the init value instead of the value the original fused loop computed.
  // This covers the dot itself, its operand slice, and every carried chain
  // cloned into that loop -- including a chain that merely yields the frozen
  // block argument unchanged, which has an empty slice and so is invisible to
  // any scan of op operands. It subsumes the old foreign-accumulator check,
  // since each accumulator is frozen in the other dot's loop.
  for (unsigned j = 0, e = forOp.getNumRegionIterArgs(); j != e; ++j) {
    if (owners[j] == Owner::Both)
      continue;
    bool readerIsDot0 = owners[j] == Owner::Dot1;
    Owner reader = readerIsDot0 ? Owner::Dot0 : Owner::Dot1;
    BlockArgument arg = forOp.getRegionIterArgs()[j];

    if (dotReadsArg(readerIsDot0 ? dot0 : dot1, readerIsDot0 ? slice0 : slice1,
                    arg)) {
      LDBG("Skipping loop: dot " << (readerIsDot0 ? 0 : 1)
                                 << " depends on iter_arg " << j
                                 << ", which is frozen in its loop");
      return false;
    }
    for (unsigned i = 0; i != e; ++i) {
      if (isFrozenIn(i, reader))
        continue;
      if (chainReadsArg(yieldOp.getOperand(i), carriedSlices[i], arg)) {
        LDBG("Skipping loop: carried iter_arg "
             << i << " depends on iter_arg " << j
             << ", which is frozen in a loop that must compute it");
        return false;
      }
    }
  }

  // An owned chain is cloned only into its owner's loop; a `Both` chain is
  // replicated into both.
  DenseSet<Operation *> carriedForDot0, carriedForDot1;
  for (unsigned i = 0, e = forOp.getNumRegionIterArgs(); i != e; ++i) {
    if (owners[i] != Owner::Dot1)
      carriedForDot0.insert(carriedSlices[i].begin(), carriedSlices[i].end());
    if (owners[i] != Owner::Dot0)
      carriedForDot1.insert(carriedSlices[i].begin(), carriedSlices[i].end());
  }

  // Every top-level op must be classified: it is either one of the two
  // dots, a member of a dot-operand slice, or a member of a carried chain.
  // The canonicalizer already ran, so no dead ops should exist here -- an
  // unclassified op means we cannot prove it is safe to drop, so reject.
  // Every carried chain lands in at least one of the two clone sets, so their
  // union has the same contents as a single merged set over all chains.
  DenseSet<Operation *> allSlices;
  allSlices.insert(slice0.begin(), slice0.end());
  allSlices.insert(slice1.begin(), slice1.end());
  allSlices.insert(carriedForDot0.begin(), carriedForDot0.end());
  allSlices.insert(carriedForDot1.begin(), carriedForDot1.end());

  for (Operation &op : *body) {
    if (&op == body->getTerminator() || &op == dot0.getOperation() ||
        &op == dot1.getOperation())
      continue;
    if (!allSlices.contains(&op)) {
      LDBG("Skipping loop: unclassified op (neither a dot-operand slice nor "
           "a carried chain member): "
           << op);
      return false;
    }
  }

  // Every slice member must be safe to clone into both new loops. This applies
  // to an owned chain as well, even though it is cloned into one loop only:
  // duplication was never the sole hazard, because distribution also *reorders*
  // the op against every memory operation of the other loop and this pass has
  // no alias analysis. That is why a volatile `tt.load` -- which declares a
  // Write effect precisely to prevent reordering -- must be rejected regardless
  // of how many loops it lands in.
  for (Operation *op : allSlices) {
    if (!isReplicable(op)) {
      LDBG("Skipping loop: slice member is not safely replicable: " << *op);
      return false;
    }
  }

  OpBuilder builder(forOp);

  // Each distributed loop keeps all original iter_args but only computes its
  // target dot and the slots it owns, yielding every slot owned by the other
  // loop unchanged. Carried (non-accumulator) chains owned by neither loop are
  // replicated identically into both.

  auto buildDistributedLoop = [&](tt::DotOp targetDot, unsigned targetIdx,
                                  const DenseSet<Operation *> &targetSlice,
                                  const DenseSet<Operation *> &carriedSet,
                                  ValueRange initArgs) {
    Owner loopOwner = owners[targetIdx];
    // Use the ForOp builder callback to construct the body inline.
    // This avoids issues with empty blocks and missing terminators.
    scf::ForOp newForOp;
    IRMapping mapping;

    newForOp = scf::ForOp::create(
        builder, forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), initArgs,
        [&](OpBuilder &b, Location loc, Value iv, ValueRange iterArgs) {
          // Map old block args to new block args.
          mapping.map(forOp.getInductionVar(), iv);
          for (auto [oldArg, newArg] :
               llvm::zip(forOp.getRegionIterArgs(), iterArgs)) {
            mapping.map(oldArg, newArg);
          }

          // Clone ops in original order: this loop's own dot-operand slice
          // plus the carried-chain ops it must compute (chains owned by
          // neither loop are replicated identically into both), plus the
          // target dot.
          for (Operation &op : *body) {
            if (&op == body->getTerminator())
              continue;
            if (targetSlice.contains(&op) || carriedSet.contains(&op) ||
                &op == targetDot.getOperation()) {
              b.clone(op, mapping);
            }
          }

          // Build yield: for the target iter_arg, yield the dot result; for a
          // slot frozen here because the other loop owns it, pass it through
          // unchanged (its real value comes from that loop, this loop's copy
          // is dead); for a true pass-through carried iter_arg, yield it
          // unchanged; otherwise resolve the carried chain's cloned result
          // through the mapping.
          SmallVector<Value> yieldOperands;
          for (unsigned i = 0; i < forOp.getNumRegionIterArgs(); ++i) {
            if (i == targetIdx) {
              yieldOperands.push_back(mapping.lookup(targetDot.getResult()));
            } else if (isFrozenIn(i, loopOwner)) {
              // Must be tested *before* the mapping fallback below: an owned
              // chain is not cloned into the loop that does not own it, so
              // `mapping` has no entry for its yielded value and
              // `lookupOrDefault` would hand back the original value, defined
              // inside the loop we are about to erase.
              yieldOperands.push_back(iterArgs[i]);
            } else if (yieldOp.getOperand(i) == forOp.getRegionIterArgs()[i]) {
              yieldOperands.push_back(iterArgs[i]);
            } else {
              yieldOperands.push_back(
                  mapping.lookupOrDefault(yieldOp.getOperand(i)));
            }
          }
          scf::YieldOp::create(b, loc, yieldOperands);
        });

    // Both new loops have the same trip count as the original, so the same
    // stage/unroll-factor intent (and any other discardable attributes)
    // applies to each.
    newForOp->setDiscardableAttrs(forOp->getDiscardableAttrDictionary());

    builder.setInsertionPointAfter(newForOp);
    return newForOp;
  };

  // Build loop 1 (for dot0).
  scf::ForOp loop1 = buildDistributedLoop(dot0, *idx0, slice0, carriedForDot0,
                                          forOp.getInitArgs());

  // Build loop 2 (for dot1).
  scf::ForOp loop2 = buildDistributedLoop(dot1, *idx1, slice1, carriedForDot1,
                                          forOp.getInitArgs());

  // Each original result comes from the loop that owns its slot. A `Both` slot
  // is computed identically in the two loops, so take it from the first. Note
  // `loop1` computes dot0 and `loop2` computes dot1 -- the names are off by one
  // from the `Owner` values.
  for (unsigned i = 0; i < forOp.getNumResults(); ++i) {
    scf::ForOp ownerLoop = owners[i] == Owner::Dot1 ? loop2 : loop1;
    forOp.getResult(i).replaceAllUsesWith(ownerLoop.getResult(i));
  }

  // Erase the original loop.
  forOp.erase();

  LDBG("Successfully distributed loop into two loops");
  ++NumLoopsDistributed;
  return true;
}

class LoopDistributePass
    : public triton::gpu::intel::impl::TritonIntelGPULoopDistributeBase<
          LoopDistributePass> {
public:
  using triton::gpu::intel::impl::TritonIntelGPULoopDistributeBase<
      LoopDistributePass>::TritonIntelGPULoopDistributeBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    // Collect loops first to avoid modifying while walking. Post-order
    // (the default, made explicit here) visits inner loops before outer
    // loops, so a nested loop is recorded before any outer loop that
    // contains it gets erased -- avoiding stale handles into an erased
    // outer loop's body.
    SmallVector<scf::ForOp> loops;
    mod.walk<WalkOrder::PostOrder>(
        [&](scf::ForOp forOp) { loops.push_back(forOp); });

    for (scf::ForOp forOp : loops) {
      tryDistributeLoop(forOp);
    }
  }
};

} // namespace
