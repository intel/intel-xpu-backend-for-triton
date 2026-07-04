#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/Support/Debug.h"

#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONINTELGPULOOPDISTRIBUTE
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

#define DEBUG_TYPE "tritonintelgpu-loop-distribute"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

/// Walk backwards from a dot's operands to find all ops in the loop body
/// that feed into it (excluding the induction variable and iter_args).
void collectBackwardSlice(tt::DotOp dot, Block *loopBody,
                          DenseSet<Operation *> &slice) {
  SmallVector<Value> worklist;
  // Add A and B operands (not C — that's the accumulator/iter_arg).
  worklist.push_back(dot.getA());
  worklist.push_back(dot.getB());

  while (!worklist.empty()) {
    Value val = worklist.pop_back_val();
    Operation *defOp = val.getDefiningOp();
    if (!defOp || defOp->getBlock() != loopBody)
      continue;
    if (!slice.insert(defOp).second)
      continue;
    for (Value operand : defOp->getOperands())
      worklist.push_back(operand);
  }
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

  LDBG("Found distributable loop with dots at iter_arg indices "
       << *idx0 << " and " << *idx1);

  // Compute the backward slice for each dot (ops that feed A/B operands).
  DenseSet<Operation *> slice0, slice1;
  collectBackwardSlice(dot0, body, slice0);
  collectBackwardSlice(dot1, body, slice1);

  // Check that the two slices don't have conflicting dependencies
  // (i.e., one dot's result feeds into the other dot's inputs).
  // The slices may share ops (e.g., a shared load for operand A).
  if (slice0.contains(dot1.getOperation()) ||
      slice1.contains(dot0.getOperation())) {
    LDBG("Skipping loop: dots have inter-dependencies");
    return false;
  }

  // Verify there are no other ops with side effects that we can't classify.
  for (Operation &op : *body) {
    if (&op == body->getTerminator())
      continue;
    if (isa<tt::DotOp>(op))
      continue;
    if (slice0.contains(&op) || slice1.contains(&op))
      continue;
    if (!isPure(&op)) {
      LDBG("Skipping loop: contains unclassified side-effecting op: " << op);
      return false;
    }
  }

  OpBuilder builder(forOp);

  // Each distributed loop keeps all original iter_args but only computes
  // its target dot, yielding the iter_arg unchanged for the other position.

  auto buildDistributedLoop = [&](tt::DotOp targetDot, unsigned targetIdx,
                                  unsigned otherIdx,
                                  DenseSet<Operation *> &targetSlice,
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

          // Clone ops in original order, but only those in the target slice
          // or the target dot.
          for (Operation &op : *body) {
            if (&op == body->getTerminator())
              continue;
            if (targetSlice.contains(&op) || &op == targetDot.getOperation()) {
              b.clone(op, mapping);
            }
          }

          // Build yield: for the target iter_arg, yield the dot result;
          // for all others, yield the iter_arg itself (pass-through).
          SmallVector<Value> yieldOperands;
          for (unsigned i = 0; i < forOp.getNumRegionIterArgs(); ++i) {
            if (i == targetIdx) {
              yieldOperands.push_back(mapping.lookup(targetDot.getResult()));
            } else {
              yieldOperands.push_back(iterArgs[i]);
            }
          }
          scf::YieldOp::create(b, loc, yieldOperands);
        });

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

    // Collect loops first to avoid modifying while walking.
    SmallVector<scf::ForOp> loops;
    mod.walk([&](scf::ForOp forOp) { loops.push_back(forOp); });

    for (scf::ForOp forOp : loops) {
      tryDistributeLoop(forOp);
    }
  }
};

} // namespace
