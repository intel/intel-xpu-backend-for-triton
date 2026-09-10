#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"

#include "intel/include/Analysis/RegisterPressure.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.h"
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
/// the loop reads the source, so it stops being live-in to the loop body.
///
/// This inspects the *current* IR rather than the (immutable) liveness
/// analysis, because earlier hoists performed by this same pass have already
/// moved their conversions out of the loop and so must count as gone. A source
/// shared by several conversions in one loop is therefore only credited to the
/// last of them to leave, which is exactly when it stops crossing the loop.
static bool hoistRetiresSource(ttg::ConvertLayoutOp cvtOp, scf::ForOp forOp) {
  for (Operation *user : cvtOp.getSrc().getUsers()) {
    if (user == cvtOp.getOperation())
      continue;
    if (forOp->isProperAncestor(user))
      return false;
  }
  return true;
}

/// Hoist a convert_layout with DotOperandEncodingAttr destination out of its
/// parent scf.for loop when the source is loop-invariant and the resulting
/// register pressure stays within the GRF budget.
///
/// \param cvtOp      The convert_layout operation to consider for hoisting.
/// \param analysis   Module-level register pressure analysis.
/// \param grfBudget  Per-lane GRF budget in bytes for the current mode (see
///                   `RegisterPressureAnalysis::getPerLaneGRFBudgetInBytes`).
/// \param netHoistBytes  Per-loop running total of the pressure change caused
///                   by hoists this pass has already performed in that loop.
static void
hoistCvtDotOpOutOfLoop(ttg::ConvertLayoutOp cvtOp,
                       const ttg::intel::RegisterPressureAnalysis &analysis,
                       unsigned grfBudget,
                       DenseMap<Operation *, int> &netHoistBytes) {
  ++NumConsidered;
  // Check the destination has DotOperandEncodingAttr.
  auto rtType = dyn_cast<RankedTensorType>(cvtOp.getType());
  if (!rtType) {
    ++NumSkippedOther;
    return;
  }
  Attribute encoding = rtType.getEncoding();
  if (!encoding || !isa<ttg::DotOperandEncodingAttr>(encoding)) {
    ++NumSkippedOther;
    return;
  }

  // Find the enclosing scf.for loop.
  auto parentForOp = cvtOp->getParentOfType<scf::ForOp>();
  if (!parentForOp) {
    ++NumSkippedOther;
    return;
  }

  // Only hoist if the cvtOp is directly in the ForOp's body, not nested
  // inside a conditional (e.g., scf.if with a loop-variant condition).
  if (cvtOp->getParentRegion() != &parentForOp.getRegion()) {
    ++NumSkippedOther;
    return;
  }

  // Check the source is loop-invariant (defined outside the loop).
  // isDefinedOutsideOfLoop correctly rejects iter_args and induction vars.
  if (!parentForOp.isDefinedOutsideOfLoop(cvtOp.getSrc())) {
    ++NumSkippedOther;
    return;
  }

  // Register pressure check. Estimate what the loop body's live-in pressure
  // would be after the hoist and compare that against the budget.
  //
  // Hoisting is a *substitution*, not an addition: the conversion's result
  // starts crossing the loop, and when nothing left inside the loop reads the
  // conversion's source, that source stops crossing it. Modelling only the
  // arrival systematically overestimates the cost, and rejects hoists that
  // would in fact have lowered pressure -- most importantly when the source
  // layout is per-lane fatter than the dot-operand layout it feeds, which is
  // the common case for an oversized blocked layout on a narrow tensor.
  // See https://github.com/intel/intel-xpu-backend-for-triton/issues/7993.
  Block *bodyBlock = parentForOp.getBody();
  unsigned liveInBytes = analysis.liveInPressure(bodyBlock);
  unsigned hoistBytes =
      ttg::intel::RegisterPressureAnalysis::getPerThreadSizeInBytes(rtType);
  unsigned retiredBytes =
      hoistRetiresSource(cvtOp, parentForOp)
          ? analysis.liveInContribution(bodyBlock, cvtOp.getSrc())
          : 0;

  // `liveInBytes` comes from an analysis built once at pass entry, so it does
  // not reflect hoists this pass has already performed. `netHoistBytes` carries
  // their accumulated effect (which may be negative) forward instead.
  int alreadyHoisted = netHoistBytes.lookup(parentForOp);
  int thisHoist = static_cast<int>(hoistBytes) - static_cast<int>(retiredBytes);
  int projectedBytes =
      std::max(0, static_cast<int>(liveInBytes) + alreadyHoisted + thisHoist);

  // Only hoist if the projected live-in pressure stays within 80% of the GRF
  // budget. The 20% headroom accounts for scalars, temporaries, and
  // loop-internal values not tracked by live-in liveness. Use integer
  // arithmetic (4/5) to avoid float-to-unsigned truncation.
  if (projectedBytes >= static_cast<int>(grfBudget * 4 / 5)) {
    LDBG("Skipping hoist: liveIn="
         << liveInBytes << " + alreadyHoisted=" << alreadyHoisted
         << " + hoistBytes=" << hoistBytes << " - retiredBytes=" << retiredBytes
         << " = " << projectedBytes << " exceeds 80% of budget=" << grfBudget);
    ++NumRejectedPressure;
    cvtOp->setAttr("tt.no_licm", UnitAttr::get(cvtOp.getContext()));
    return;
  }

  LDBG("Hoisting convert_layout out of loop: liveIn="
       << liveInBytes << " + alreadyHoisted=" << alreadyHoisted
       << " + hoistBytes=" << hoistBytes << " - retiredBytes=" << retiredBytes
       << " = " << projectedBytes << " budget=" << grfBudget);
  // Hoist the conversion out of the loop.
  Operation *srcDefOp = cvtOp.getSrc().getDefiningOp();
  if (srcDefOp)
    cvtOp->moveAfter(srcDefOp);
  else
    cvtOp->moveBefore(parentForOp);

  ++NumHoisted;
  netHoistBytes[parentForOp] += thisHoist;
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

    SmallVector<ttg::ConvertLayoutOp> cvtOps;
    mod.walk([&](ttg::ConvertLayoutOp cvtOp) { cvtOps.push_back(cvtOp); });

    DenseMap<Operation *, int> netHoistBytes;
    for (auto cvtOp : cvtOps)
      hoistCvtDotOpOutOfLoop(cvtOp, analysis, grfBudget, netHoistBytes);

    if (mlir::triton::tools::getBoolEnv("TRITON_INTEL_HLC_STATS")) {
      llvm::errs() << "[HoistLayoutConversions] considered=" << NumConsidered
                   << " hoisted=" << NumHoisted
                   << " rejected_pressure=" << NumRejectedPressure
                   << " skipped_other=" << NumSkippedOther << "\n";
    }
  }
};

} // namespace
