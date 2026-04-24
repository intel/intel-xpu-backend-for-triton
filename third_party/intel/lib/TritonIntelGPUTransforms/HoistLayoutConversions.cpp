#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
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

namespace {

/// Return the GRF budget in bytes per thread for the given GRF mode.
///
/// Explicit sizes ("128", "256", "512") map to the exact per-thread budget.
/// For "default" and "auto" the compiler chooses the GRF size at JIT time,
/// so the pass conservatively assumes the smallest (128-register) budget to
/// avoid hoisting past the hardware limit on configurations that ultimately
/// compile with fewer registers.
///
/// \param grfMode  The GRF mode string ("default", "128", "256", "512", or
///                 "auto").
/// \return         The GRF budget in bytes per hardware thread.
unsigned getGRFBytesPerThread(StringRef grfMode) {
  return llvm::StringSwitch<unsigned>(grfMode)
      .Case("128", 4096)
      .Case("256", 8192)
      .Case("512", 16384)
      .Default(4096);
}

/// Return the per-thread size in bytes for \p type, using the encoding's
/// element distribution. Returns 0 if the element type is not int or float.
///
/// \param type  A ranked tensor type with a distributed encoding.
/// \return      The number of bytes each thread holds for this tensor.
unsigned getPerThreadSizeInBytes(RankedTensorType type) {
  Type elType = type.getElementType();
  if (!elType.isIntOrFloat())
    return 0;
  unsigned elemsPerThread = ttg::getTotalElemsPerThread(type);
  return elemsPerThread * (elType.getIntOrFloatBitWidth() / 8);
}

/// Return the total live-in size in bytes for \p livenessBlockInfo.
/// Accounts for both tensor values (using per-thread element distribution)
/// and scalar int/float values.
///
/// \param livenessBlockInfo  Liveness information for a single basic block.
/// \return                   The aggregate per-thread byte size of all live-in
///                           values.
unsigned getBlockLiveInSizeInBytes(const LivenessBlockInfo *livenessBlockInfo) {
  unsigned blockInSize = 0;
  for (Value liveVal : livenessBlockInfo->in()) {
    Type liveValTy = liveVal.getType();
    if (auto tensorType = dyn_cast<RankedTensorType>(liveValTy)) {
      blockInSize += getPerThreadSizeInBytes(tensorType);
    } else if (liveValTy.isIntOrFloat()) {
      blockInSize += liveValTy.getIntOrFloatBitWidth() / 8;
    }
  }
  return blockInSize;
}

/// Hoist a convert_layout with DotOperandEncodingAttr destination out of its
/// parent scf.for loop when the source is loop-invariant and the resulting
/// register pressure stays within the GRF budget.
///
/// \param cvtOp      The convert_layout operation to consider for hoisting.
/// \param liveness   Module-level liveness analysis results.
/// \param grfBudget  The GRF budget in bytes per thread for the current mode.
static void
hoistCvtDotOpOutOfLoop(ttg::ConvertLayoutOp cvtOp, Liveness &liveness,
                       unsigned grfBudget,
                       DenseMap<Operation *, unsigned> &cumulativeHoistBytes) {
  // Check the destination has DotOperandEncodingAttr.
  auto rtType = dyn_cast<RankedTensorType>(cvtOp.getType());
  if (!rtType)
    return;
  Attribute encoding = rtType.getEncoding();
  if (!encoding || !isa<ttg::DotOperandEncodingAttr>(encoding))
    return;

  // Find the enclosing scf.for loop.
  auto parentForOp = cvtOp->getParentOfType<scf::ForOp>();
  if (!parentForOp)
    return;

  // Only hoist if the cvtOp is directly in the ForOp's body, not nested
  // inside a conditional (e.g., scf.if with a loop-variant condition).
  if (cvtOp->getParentRegion() != &parentForOp.getRegion())
    return;

  // Check the source is loop-invariant (defined outside the loop).
  // isDefinedOutsideOfLoop correctly rejects iter_args and induction vars.
  if (!parentForOp.isDefinedOutsideOfLoop(cvtOp.getSrc()))
    return;

  // Liveness check.
  Block *bodyBlock = parentForOp.getBody();
  const LivenessBlockInfo *blockInfo = liveness.getLiveness(bodyBlock);
  // getLiveness() returns nullptr for unreachable blocks; defensive check.
  if (!blockInfo)
    return;

  unsigned liveInBytes = getBlockLiveInSizeInBytes(blockInfo);
  unsigned hoistBytes = getPerThreadSizeInBytes(rtType);
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
    unsigned grfBudget = getGRFBytesPerThread(grfMode);
    Liveness liveness(mod);

    SmallVector<ttg::ConvertLayoutOp> cvtOps;
    mod.walk([&](ttg::ConvertLayoutOp cvtOp) { cvtOps.push_back(cvtOp); });

    DenseMap<Operation *, unsigned> cumulativeHoistBytes;
    for (auto cvtOp : cvtOps)
      hoistCvtDotOpOutOfLoop(cvtOp, liveness, grfBudget, cumulativeHoistBytes);
  }
};

} // namespace
