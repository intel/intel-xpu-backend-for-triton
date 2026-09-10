#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"

#include "intel/include/Analysis/RegisterPressure.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Debug.h"
#include <algorithm>

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONINTELGPUHOISTLAYOUTCONVERSIONS
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

using namespace mlir;
namespace ttg = mlir::triton::gpu;

#define DEBUG_TYPE "tritonintelgpu-hoist-layout-conversions"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

STATISTIC(NumConsidered,
          "Number of convert_layout ops considered for hoisting");
STATISTIC(NumHoisted, "Number of convert_layout ops hoisted out of loops");
STATISTIC(NumRejectedPressure,
          "Number of convert_layout ops rejected due to register pressure");
STATISTIC(NumSkippedOther,
          "Number of convert_layout ops skipped (not eligible)");

namespace {

/// Returns true if hoisting \p cvtOp out of \p forOp retires \p cvtOp's source
/// from the loop, i.e. once \p cvtOp has moved out, nothing remaining inside
/// the loop reads the source *and* nothing after the loop does either, so it
/// stops being live-in to the loop body.
///
/// Both halves of the question are asked of the *current* IR rather than of the
/// (immutable) liveness analysis, because hoists this pass has already
/// performed have moved their conversions and so must count as moved:
///
/// * In-loop: a source shared by several conversions in one loop is only
///   credited to the last of them to leave, which is exactly when it stops
///   crossing the loop.
/// * After the loop: a source read below the loop occupies a register for the
///   loop's whole duration no matter where the conversion sits, so hoisting
///   frees nothing and crediting it would under-count the real occupancy. But
///   "below the loop" has to mean below it *now*: when the reader below is
///   itself a conversion that this pass has already hoisted to above the loop,
///   the source no longer crosses the loop and the credit is real. Asking the
///   frozen analysis instead made the answer depend on which of two loops
///   sharing a source the pass happened to reach first (see
///   `runOnOperation`, which visits loops last-to-first so this view is final).
static bool hoistRetiresSource(ttg::ConvertLayoutOp cvtOp, scf::ForOp forOp) {
  Block *loopBlock = forOp->getBlock();
  for (Operation *user : cvtOp.getSrc().getUsers()) {
    if (user == cvtOp.getOperation())
      continue;
    if (forOp->isProperAncestor(user))
      return false;
    // Locate the user in the loop's own block to compare positions. A user
    // that cannot be placed there (a different region altogether) might run
    // after the loop, so refuse the credit.
    Operation *ancestor = loopBlock->findAncestorOpInBlock(*user);
    if (!ancestor)
      return false;
    if (forOp->isBeforeInBlock(ancestor))
      return false;
  }
  return true;
}

/// Returns the `scf.for` loop \p cvtOp could be hoisted out of, or a null
/// `scf::ForOp` when \p cvtOp is not a hoisting candidate at all (wrong
/// destination encoding, not directly in a loop body, or a loop-variant
/// source). Bumps the "considered" and "skipped" statistics.
static scf::ForOp getHoistCandidateLoop(ttg::ConvertLayoutOp cvtOp) {
  ++NumConsidered;
  // Check the destination has DotOperandEncodingAttr.
  auto rtType = dyn_cast<RankedTensorType>(cvtOp.getType());
  if (!rtType) {
    ++NumSkippedOther;
    return {};
  }
  Attribute encoding = rtType.getEncoding();
  if (!encoding || !isa<ttg::DotOperandEncodingAttr>(encoding)) {
    ++NumSkippedOther;
    return {};
  }

  // Find the enclosing scf.for loop.
  auto parentForOp = cvtOp->getParentOfType<scf::ForOp>();
  if (!parentForOp) {
    ++NumSkippedOther;
    return {};
  }

  // Only hoist if the cvtOp is directly in the ForOp's body, not nested
  // inside a conditional (e.g., scf.if with a loop-variant condition).
  if (cvtOp->getParentRegion() != &parentForOp.getRegion()) {
    ++NumSkippedOther;
    return {};
  }

  // Check the source is loop-invariant (defined outside the loop).
  // isDefinedOutsideOfLoop correctly rejects iter_args and induction vars.
  if (!parentForOp.isDefinedOutsideOfLoop(cvtOp.getSrc())) {
    ++NumSkippedOther;
    return {};
  }

  return parentForOp;
}

/// Returns the signed change, in per-lane bytes, that hoisting \p cvtOp out of
/// \p forOp would make to the loop body's live-in pressure.
///
/// Hoisting is a *substitution*, not an addition: the conversion's result
/// starts crossing the loop, and when nothing left inside the loop (and nothing
/// below it) reads the conversion's source, that source stops crossing it.
/// Modelling only the arrival systematically overestimates the cost, and
/// rejects hoists that would in fact have lowered pressure -- most importantly
/// when the source layout is per-lane fatter than the dot-operand layout it
/// feeds, which is the common case for an oversized blocked layout on a narrow
/// tensor. See
/// https://github.com/intel/intel-xpu-backend-for-triton/issues/7993.
///
/// The credit has to come from the analysis rather than from the source's type,
/// because `liveInPressure` filters out rematerializable values: a constant
/// source never occupied the bytes its type suggests.
static int
hoistDeltaBytes(ttg::ConvertLayoutOp cvtOp, scf::ForOp forOp,
                const ttg::intel::RegisterPressureAnalysis &analysis) {
  unsigned hoistBytes =
      ttg::intel::RegisterPressureAnalysis::getPerThreadSizeInBytes(
          cvtOp.getType());
  unsigned retiredBytes =
      hoistRetiresSource(cvtOp, forOp)
          ? analysis.liveInContribution(forOp.getBody(), cvtOp.getSrc())
          : 0;
  return static_cast<int>(hoistBytes) - static_cast<int>(retiredBytes);
}

/// Decide, for every hoisting candidate of a single \p forOp, whether to hoist
/// it out of the loop or to reject it on register pressure grounds.
///
/// The candidates are *not* processed in program order. Because a hoist can
/// lower the projected pressure as well as raise it, the running total is no
/// longer monotonically non-decreasing, and a rejection is only safely
/// conservative if every pressure-reducing candidate has already been credited
/// when it is made -- a rejection is irrevocable, since it stamps `tt.no_licm`
/// and the later generic LICM pass then never revisits the conversion. So the
/// candidate with the smallest (most negative) projected delta is always
/// decided first, which makes the outcome depend on the candidates' measured
/// costs rather than on their syntactic order inside the loop.
///
/// The delta is re-measured before each decision rather than sorted once up
/// front, because a candidate sharing its source with a sibling only retires
/// that source once the sibling has left.
///
/// \param forOp      The loop the candidates live in.
/// \param candidates Hoisting candidates directly inside \p forOp's body, in
///                   program order.
/// \param analysis   Module-level register pressure analysis.
/// \param grfBudget  Per-lane GRF budget in bytes for the current mode (see
///                   `RegisterPressureAnalysis::getPerLaneGRFBudgetInBytes`).
static void hoistCvtDotOpsOutOfLoop(
    scf::ForOp forOp, ArrayRef<ttg::ConvertLayoutOp> candidates,
    const ttg::intel::RegisterPressureAnalysis &analysis, unsigned grfBudget) {
  // `liveInBytes` comes from an analysis built once at pass entry, so it does
  // not reflect hoists this pass has already performed. `netBytes` carries
  // their accumulated effect (which may be negative) forward instead.
  unsigned liveInBytes = analysis.liveInPressure(forOp.getBody());
  int netBytes = 0;

  // Only hoist if the projected live-in pressure stays within 80% of the GRF
  // budget. The 20% headroom accounts for scalars, temporaries, and
  // loop-internal values not tracked by live-in liveness. Use integer
  // arithmetic (4/5) to avoid float-to-unsigned truncation.
  int threshold = static_cast<int>(grfBudget * 4 / 5);

  SmallVector<ttg::ConvertLayoutOp> pending(candidates);
  while (!pending.empty()) {
    // Pick the cheapest remaining candidate, breaking ties in program order.
    // Ties are outcome-neutral: equal deltas project the same values whichever
    // of the two is decided first.
    unsigned bestIdx = 0;
    int bestDelta = hoistDeltaBytes(pending[0], forOp, analysis);
    for (unsigned i = 1, e = pending.size(); i != e; ++i) {
      int delta = hoistDeltaBytes(pending[i], forOp, analysis);
      if (delta < bestDelta) {
        bestDelta = delta;
        bestIdx = i;
      }
    }
    ttg::ConvertLayoutOp cvtOp = pending[bestIdx];
    pending.erase(pending.begin() + bestIdx);

    // Estimate what the loop body's live-in pressure would be after the hoist
    // and compare that against the budget. A source is credited at most once
    // and only for bytes it contributed to `liveInBytes`, so the projection
    // cannot go negative.
    int projectedBytes = static_cast<int>(liveInBytes) + netBytes + bestDelta;
    assert(projectedBytes >= 0 && "over-credited a hoist's retired source");

    // The budget only vetoes a hoist that spends it. A hoist whose net effect
    // on the loop body's live-in pressure is zero or negative leaves the loop
    // exactly as far over (or under) budget as it already was, so rejecting it
    // buys back no register and only forgoes the loop-invariant conversion --
    // and the rejection is irrevocable, since it stamps `tt.no_licm`. Gating
    // such a hoist on a figure it does not move is what made this pass reject
    // six of _attn_bwd's eight loop-invariant dot-operand conversions at
    // HEAD_DIM=128, where all eight are measurably free (see the pass
    // description) and rejecting them cost 21.7% geomean on flash-attn-bwd.
    if (bestDelta > 0 && projectedBytes >= threshold) {
      LDBG("Skipping hoist: liveIn="
           << liveInBytes << " + alreadyHoisted=" << netBytes
           << " + thisHoist=" << bestDelta << " = " << projectedBytes
           << " B/lane exceeds 80% of budget=" << grfBudget << " B/lane");
      ++NumRejectedPressure;
      cvtOp->setAttr("tt.no_licm", UnitAttr::get(cvtOp.getContext()));
      continue;
    }

    LDBG("Hoisting convert_layout out of loop: liveIn="
         << liveInBytes << " + alreadyHoisted=" << netBytes
         << " + thisHoist=" << bestDelta << " = " << projectedBytes
         << " B/lane budget=" << grfBudget << " B/lane");
    // Hoist the conversion out of the loop.
    Operation *srcDefOp = cvtOp.getSrc().getDefiningOp();
    if (srcDefOp)
      cvtOp->moveAfter(srcDefOp);
    else
      cvtOp->moveBefore(forOp);

    ++NumHoisted;
    netBytes += bestDelta;
  }
}

class TritonIntelGPUHoistLayoutConversionsPass
    : public ttg::intel::impl::TritonIntelGPUHoistLayoutConversionsBase<
          TritonIntelGPUHoistLayoutConversionsPass> {

  using ttg::intel::impl::TritonIntelGPUHoistLayoutConversionsBase<
      TritonIntelGPUHoistLayoutConversionsPass>::
      TritonIntelGPUHoistLayoutConversionsBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    unsigned grfBudget =
        ttg::intel::RegisterPressureAnalysis::getPerLaneGRFBudgetInBytes(
            grfMode, mod);
    ttg::intel::RegisterPressureAnalysis analysis(mod);

    // Group the candidates by the loop they would leave, so that each loop's
    // budget can be spent on its candidates as a set rather than in whatever
    // order the walk happens to reach them. Insertion order (program order) is
    // preserved to keep the tie-break in `hoistCvtDotOpsOutOfLoop`
    // deterministic. Inner loops are visited before their enclosing loop's
    // later candidates, but the two never share a body block, so their budgets
    // are independent.
    llvm::MapVector<scf::ForOp, SmallVector<ttg::ConvertLayoutOp>> candidates;
    mod.walk([&](ttg::ConvertLayoutOp cvtOp) {
      if (scf::ForOp forOp = getHoistCandidateLoop(cvtOp))
        candidates[forOp].push_back(cvtOp);
    });

    // Decide the loops in reverse walk order, i.e. last to first. A hoist only
    // ever moves a conversion *earlier* in its block, so once every loop that
    // follows `forOp` has been decided, no later hoist can add or remove a use
    // of a source below `forOp`, and `hoistRetiresSource`'s after-loop question
    // has a final answer. Deciding first-to-last instead denied the credit to
    // the *earlier* of two loops sharing a source, purely because the later
    // loop's conversion had not moved out from under it yet -- the same
    // order-dependence that `hoistCvtDotOpsOutOfLoop` removes within one loop.
    for (auto &[forOp, loopCandidates] : llvm::reverse(candidates))
      hoistCvtDotOpsOutOfLoop(forOp, loopCandidates, analysis, grfBudget);

    if (mlir::triton::tools::getBoolEnv("TRITON_INTEL_HLC_STATS")) {
      llvm::errs() << "[HoistLayoutConversions] considered=" << NumConsidered
                   << " hoisted=" << NumHoisted
                   << " rejected_pressure=" << NumRejectedPressure
                   << " skipped_other=" << NumSkippedOther << "\n";
    }
  }
};

} // namespace
