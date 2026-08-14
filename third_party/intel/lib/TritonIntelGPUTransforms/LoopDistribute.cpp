#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
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
bool isReplicable(Operation *op) {
  if (isPure(op))
    return true;
  if (auto load = dyn_cast<tt::LoadOp>(op))
    return !load.getIsVolatile();
  if (isa<tt::DescriptorLoadOp, tt::DescriptorGatherOp>(op))
    return true;
  return false;
}

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

  // Every other (non-accumulator) iter_arg is either a true pass-through
  // (yielded unchanged) or a chain that must be safely replicable into both
  // new loops. Chains that depend on either dot's result, or that read a
  // dot's accumulator block argument directly, cannot be replicated
  // correctly (each new loop only computes one accumulator), so reject the
  // whole loop in that case.
  BlockArgument accArg0 = forOp.getRegionIterArgs()[*idx0];
  BlockArgument accArg1 = forOp.getRegionIterArgs()[*idx1];

  DenseSet<Operation *> carriedUnion;
  for (unsigned i = 0, e = forOp.getNumRegionIterArgs(); i < e; ++i) {
    if (i == *idx0 || i == *idx1)
      continue;

    Value carriedVal = yieldOp.getOperand(i);
    if (carriedVal == forOp.getRegionIterArgs()[i]) {
      // True pass-through: nothing to clone, no slice needed.
      continue;
    }

    if (carriedVal == accArg0 || carriedVal == accArg1) {
      LDBG("Skipping loop: carried iter_arg "
           << i << " is itself a dot accumulator block argument");
      return false;
    }

    DenseSet<Operation *> carriedSlice;
    collectBackwardSlice(carriedVal, body, carriedSlice);

    if (carriedSlice.contains(dot0.getOperation()) ||
        carriedSlice.contains(dot1.getOperation())) {
      LDBG("Skipping loop: carried iter_arg " << i
                                              << " depends on a dot result");
      return false;
    }

    // Walk into nested regions too: a region-holding op (e.g. `scf.if`) may
    // read the accumulator as a captured value inside its region without
    // passing it as one of the op's own top-level operands.
    bool usesAcc = llvm::any_of(carriedSlice, [&](Operation *op) {
      bool found = false;
      op->walk([&](Operation *nested) {
        if (llvm::is_contained(nested->getOperands(), accArg0) ||
            llvm::is_contained(nested->getOperands(), accArg1)) {
          found = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      return found;
    });
    if (usesAcc) {
      LDBG("Skipping loop: carried iter_arg "
           << i << " depends on a dot accumulator block argument");
      return false;
    }

    carriedUnion.insert(carriedSlice.begin(), carriedSlice.end());
  }

  // Every top-level op must be classified: it is either one of the two
  // dots, a member of a dot-operand slice, or a member of a carried chain.
  // The canonicalizer already ran, so no dead ops should exist here -- an
  // unclassified op means we cannot prove it is safe to drop, so reject.
  DenseSet<Operation *> allSlices;
  allSlices.insert(slice0.begin(), slice0.end());
  allSlices.insert(slice1.begin(), slice1.end());
  allSlices.insert(carriedUnion.begin(), carriedUnion.end());

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

  // Every slice member must be safe to clone into both new loops.
  for (Operation *op : allSlices) {
    if (!isReplicable(op)) {
      LDBG("Skipping loop: slice member is not safely replicable: " << *op);
      return false;
    }
  }

  OpBuilder builder(forOp);

  // Each distributed loop keeps all original iter_args but only computes
  // its target dot, yielding the iter_arg unchanged for the other position.
  // Carried (non-accumulator) chains are replicated identically into both
  // loops.

  auto buildDistributedLoop = [&](tt::DotOp targetDot, unsigned targetIdx,
                                  unsigned otherIdx,
                                  const DenseSet<Operation *> &targetSlice,
                                  ValueRange initArgs) {
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
          // plus every carried-chain op (carried chains are replicated
          // identically into both distributed loops), plus the target dot.
          for (Operation &op : *body) {
            if (&op == body->getTerminator())
              continue;
            if (targetSlice.contains(&op) || carriedUnion.contains(&op) ||
                &op == targetDot.getOperation()) {
              b.clone(op, mapping);
            }
          }

          // Build yield: for the target iter_arg, yield the dot result; for
          // the other dot's accumulator, pass it through unchanged (its
          // real value comes from the other distributed loop, this loop's
          // copy is dead); for a true pass-through carried iter_arg, yield
          // it unchanged; otherwise resolve the carried chain's cloned
          // result through the mapping.
          SmallVector<Value> yieldOperands;
          for (unsigned i = 0; i < forOp.getNumRegionIterArgs(); ++i) {
            if (i == targetIdx) {
              yieldOperands.push_back(mapping.lookup(targetDot.getResult()));
            } else if (i == otherIdx) {
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
  scf::ForOp loop1 =
      buildDistributedLoop(dot0, *idx0, *idx1, slice0, forOp.getInitArgs());

  // Build loop 2 (for dot1).
  scf::ForOp loop2 =
      buildDistributedLoop(dot1, *idx1, *idx0, slice1, forOp.getInitArgs());

  // Replace the original loop's results: idx0 from loop1, idx1 from loop2,
  // others from either (they're pass-through in both).
  for (unsigned i = 0; i < forOp.getNumResults(); ++i) {
    Value replacement;
    if (i == *idx0)
      replacement = loop1.getResult(i);
    else if (i == *idx1)
      replacement = loop2.getResult(i);
    else
      replacement = loop1.getResult(i); // pass-through, same in both
    forOp.getResult(i).replaceAllUsesWith(replacement);
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
