#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"

#include "intel/include/Analysis/RegisterPressure.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Debug.h"

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

/// Hoist a convert_layout with DotOperandEncodingAttr destination out of its
/// parent scf.for loop when the source is loop-invariant and the resulting
/// register pressure stays within the GRF budget.
///
/// \param cvtOp      The convert_layout operation to consider for hoisting.
/// \param analysis   Module-level register pressure analysis.
/// \param grfBudget  The GRF budget in bytes per thread for the current mode.
static void
hoistCvtDotOpOutOfLoop(ttg::ConvertLayoutOp cvtOp,
                       const ttg::intel::RegisterPressureAnalysis &analysis,
                       unsigned grfBudget,
                       DenseMap<Operation *, unsigned> &cumulativeHoistBytes) {
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

  // Register pressure check.
  Block *bodyBlock = parentForOp.getBody();
  unsigned liveInBytes = analysis.liveInPressure(bodyBlock);
  unsigned hoistBytes =
      ttg::intel::RegisterPressureAnalysis::getPerThreadSizeInBytes(rtType);
  // Only hoist if the additional register pressure from the hoisted tensor
  // stays within 80% of the GRF budget. The 20% headroom accounts for
  // scalars, temporaries, and loop-internal values not tracked by liveness.
  // Use integer arithmetic (4/5) to avoid float-to-unsigned truncation.
  unsigned alreadyHoisted = cumulativeHoistBytes.lookup(parentForOp);
  if ((liveInBytes + alreadyHoisted + hoistBytes) >= grfBudget * 4 / 5) {
    LDBG("Skipping hoist: liveIn=" << liveInBytes
                                   << " + alreadyHoisted=" << alreadyHoisted
                                   << " + hoistBytes=" << hoistBytes
                                   << " exceeds 80% of budget=" << grfBudget);
    ++NumRejectedPressure;
    cvtOp->setAttr("tt.no_licm", UnitAttr::get(cvtOp.getContext()));
    return;
  }

  LDBG("Hoisting convert_layout out of loop: liveIn="
       << liveInBytes << " hoistBytes=" << hoistBytes
       << " budget=" << grfBudget);
  // Hoist the conversion out of the loop.
  Operation *srcDefOp = cvtOp.getSrc().getDefiningOp();
  if (srcDefOp)
    cvtOp->moveAfter(srcDefOp);
  else
    cvtOp->moveBefore(parentForOp);

  ++NumHoisted;
  cumulativeHoistBytes[parentForOp] += hoistBytes;
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
        ttg::intel::RegisterPressureAnalysis::getGRFBytesPerThread(grfMode);
    ttg::intel::RegisterPressureAnalysis analysis(mod);

    SmallVector<ttg::ConvertLayoutOp> cvtOps;
    mod.walk([&](ttg::ConvertLayoutOp cvtOp) { cvtOps.push_back(cvtOp); });

    DenseMap<Operation *, unsigned> cumulativeHoistBytes;
    for (auto cvtOp : cvtOps)
      hoistCvtDotOpOutOfLoop(cvtOp, analysis, grfBudget, cumulativeHoistBytes);

    if (mlir::triton::tools::getBoolEnv("TRITON_INTEL_HLC_STATS")) {
      llvm::errs() << "[HoistLayoutConversions] considered=" << NumConsidered
                   << " hoisted=" << NumHoisted
                   << " rejected_pressure=" << NumRejectedPressure
                   << " skipped_other=" << NumSkippedOther << "\n";
    }
  }
};

} // namespace
