#include "intel/include/Analysis/RegisterPressure.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <optional>

using namespace mlir;
using namespace mlir::triton;

namespace mlir::triton::gpu::intel {

RegisterPressureAnalysis::RegisterPressureAnalysis(Operation *op,
                                                   RegisterPressureOptions opts)
    : liveness(op), options(opts) {}

unsigned RegisterPressureAnalysis::getPerThreadSizeInBytes(Type type) {
  // Handle RankedTensorType with distributed encoding
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    Type elType = tensorType.getElementType();
    if (!elType.isIntOrFloat())
      return 0;
    unsigned elemsPerThread = gpu::getTotalElemsPerThread(tensorType);
    // Round up to whole bytes so sub-byte types (fp8/fp4/i1) are not counted
    // as zero pressure, which would systematically under-count FP8 kernels.
    unsigned bytesPerElem = (elType.getIntOrFloatBitWidth() + 7) / 8;
    return elemsPerThread * bytesPerElem;
  }

  // Handle scalar int/float types
  if (type.isIntOrFloat())
    return (type.getIntOrFloatBitWidth() + 7) / 8;

  // All other types contribute zero pressure
  return 0;
}

/// Maps an explicit GRF mode string ("128"/"256"/"512") to its exact
/// per-hardware-thread budget in bytes (one hardware thread executes a whole
/// subgroup/warp of lanes sharing one register file). Returns std::nullopt
/// for anything else ("default", "auto", empty, or an unrecognized value),
/// so callers can distinguish "not an explicit mode" from "unrecognized
/// explicit mode" and choose their own fallback rather than one of the two
/// silently degrading into the other.
static std::optional<unsigned> explicitGRFModeToBytes(StringRef grfMode) {
  if (grfMode == "128")
    return 4096;
  if (grfMode == "256")
    return 8192;
  if (grfMode == "512")
    return 16384;
  return std::nullopt;
}

unsigned RegisterPressureAnalysis::getGRFBytesPerHardwareThread(
    StringRef grfMode, ModuleOp mod,
    UnknownGRFSizeAssumption unknownAssumption) {
  if (auto explicitBytes = explicitGRFModeToBytes(grfMode))
    return *explicitBytes;
  // "default" and "auto": the compiler chooses the GRF size at JIT time, so
  // the true value isn't known here. Which bound is safe depends on the
  // caller; see UnknownGRFSizeAssumption's documentation.
  if (unknownAssumption == UnknownGRFSizeAssumption::Smallest)
    return 4096;
  // On the `grf_mode='default'` path, `make_zebin`'s automatic-escalation
  // retry -- the only path that would ever realize `ttig.max_grf_mode` --
  // is itself skipped outright once `num_warps > 32`: a larger GRF mode
  // halves the launchable work-group size, so escalating would produce an
  // unlaunchable kernel (see `make_zebin`'s own guard in compiler.py).
  // Above that bound the realizable ceiling is the same as `Smallest`,
  // regardless of what `ttig.max_grf_mode` says. `grf_mode='auto'` is not
  // included: its escalation happens inside IGC, not through this retry, and
  // is not itself gated on `num_warps`.
  if (grfMode == "default" && lookupNumWarps(mod) > 32)
    return 4096;
  // Largest: the true ceiling is per-target, mirrored onto the module via the
  // ttig.max_grf_mode attribute (see UnknownGRFSizeAssumption::Largest's
  // documentation). Reuse the same explicit-mode table above so a value other
  // than exactly "256"/"512"/"128" (a typo, a future mode, or the attribute
  // being absent) cannot silently resolve to the wrong budget: it falls
  // through to the behaviour-preserving 512-register-mode default below.
  if (auto maxGRFMode = mod->getAttrOfType<StringAttr>(
          TritonIntelGPUDialect::getMaxGRFModeAttrName()))
    if (auto explicitBytes = explicitGRFModeToBytes(maxGRFMode.getValue()))
      return *explicitBytes;
  // Absence resolves to 512-register mode here, not `driver.c`'s "256" for a
  // missing `load_binary` argument: each side preserves its own pre-existing
  // behaviour on absence (this one hardcoded 16384 before `ttig.max_grf_mode`
  // existed; `driver.c` already resolved a missing arg to "unknown", which
  // already selected 256). Not a bug -- but if either fallback's rationale
  // ever changes, check whether the other one still makes sense.
  return 16384;
}

unsigned RegisterPressureAnalysis::getPerLaneGRFBudgetInBytes(
    StringRef grfMode, ModuleOp mod,
    UnknownGRFSizeAssumption unknownAssumption) {
  unsigned grfBudget =
      getGRFBytesPerHardwareThread(grfMode, mod, unknownAssumption);
  int threadsPerWarp = TritonGPUDialect::getThreadsPerWarp(mod);
  return grfBudget / static_cast<unsigned>(threadsPerWarp);
}

bool RegisterPressureAnalysis::isRematerializable(Value value) const {
  Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return false; // Block arguments are not rematerializable

  // Check for constant-like operations that are cheap to regenerate.
  if (isa<arith::ConstantOp>(defOp))
    return true;

  // make_range is always cheap to regenerate (no inputs).
  if (isa<triton::MakeRangeOp>(defOp))
    return true;

  // A splat is only free to rematerialize if its scalar source is itself
  // rematerializable. A splat of a loop-variant scalar is NOT free: it would
  // require the scalar to be live (or recomputed) at the point of use.
  if (auto splatOp = dyn_cast<triton::SplatOp>(defOp))
    return isRematerializable(splatOp.getSrc());

  // Check for constant splat patterns using MLIR pattern matchers.
  Attribute constVal;
  if (matchPattern(defOp, m_Constant(&constVal)))
    return true;

  return false;
}

unsigned RegisterPressureAnalysis::pressureContribution(Value value) const {
  // A value nothing reads never needs to occupy a register. This also keeps the
  // reported figures independent of which operation happens to come first in a
  // block; see the header for why that matters. Live-in based figures are
  // unaffected: a value is live-in to a block precisely because something below
  // reads it.
  //
  // It does lower `peakPressure(loop)` for a body that defines an unread value,
  // which `ReduceVariableLiveness` gates its sink on -- only ever downward, so
  // that gate can keep an operand it would have sunk, never the reverse.
  if (value.use_empty())
    return 0;
  if (options.excludeRematerializable && isRematerializable(value))
    return 0;
  return getPerThreadSizeInBytes(value.getType());
}

unsigned RegisterPressureAnalysis::pressureAt(Operation *op) const {
  unsigned pressure = 0;

  // Query the base mlir::Liveness block info directly (rather than the Intel
  // LivenessAnalysis wrapper, which asserts the op is a direct child of the
  // analysis root). This works for ops nested in any region, e.g. an scf.for
  // body. currentlyLiveValues() takes an expansive view: a value defined by or
  // consumed by `op` is counted, so a value is counted at its defining op and
  // loop-carried iter args (block live-in) are included.
  const LivenessBlockInfo *blockInfo = liveness.getLiveness(op->getBlock());
  if (!blockInfo)
    return 0;
  Liveness::ValueSetT liveValues = blockInfo->currentlyLiveValues(op);

  for (Value liveVal : liveValues)
    pressure += pressureContribution(liveVal);

  return pressure;
}

RegisterPressureAnalysis::PressureAtPoint
RegisterPressureAnalysis::pressureAt(Operation *op, Value value) const {
  PressureAtPoint result;

  const LivenessBlockInfo *blockInfo = liveness.getLiveness(op->getBlock());
  if (!blockInfo)
    return result;
  Liveness::ValueSetT liveValues = blockInfo->currentlyLiveValues(op);

  for (Value liveVal : liveValues) {
    result.pressure += pressureContribution(liveVal);
    if (liveVal == value)
      result.valueLive = true;
  }

  return result;
}

unsigned RegisterPressureAnalysis::pressureBefore(Operation *op) const {
  const LivenessBlockInfo *blockInfo = liveness.getLiveness(op->getBlock());
  if (!blockInfo)
    return 0;

  unsigned pressure = 0;
  for (Value liveVal : blockInfo->currentlyLiveValues(op)) {
    // `pressureAt` counts a value at its defining op, which above `op` has not
    // run yet. Everything else live at `op` is also live immediately above it.
    if (liveVal.getDefiningOp() == op)
      continue;
    pressure += pressureContribution(liveVal);
  }

  return pressure;
}

unsigned RegisterPressureAnalysis::peakPressure(Block *block) const {
  unsigned peak = 0;

  // Iterate over all operations in the block and track the maximum pressure
  for (Operation &op : block->getOperations()) {
    unsigned pressure = pressureAt(&op);
    peak = std::max(peak, pressure);
  }

  return peak;
}

unsigned
RegisterPressureAnalysis::peakPressure(LoopLikeOpInterface loop) const {
  unsigned peak = 0;

  // A loop may expose multiple body regions, each with multiple blocks; check
  // all blocks across all loop regions.
  for (Region *region : loop.getLoopRegions()) {
    if (!region)
      continue;
    for (Block &block : *region) {
      unsigned blockPeak = peakPressure(&block);
      peak = std::max(peak, blockPeak);
    }
  }

  return peak;
}

unsigned
RegisterPressureAnalysis::peakPressureOverNestedBlocks(Operation *root) const {
  unsigned peak = 0;
  root->walk([&](Block *block) { peak = std::max(peak, peakPressure(block)); });
  return peak;
}

unsigned
RegisterPressureAnalysis::peakPressure(FunctionOpInterface func) const {
  return peakPressureOverNestedBlocks(func);
}

unsigned RegisterPressureAnalysis::liveInPressure(Block *block) const {
  const LivenessBlockInfo *blockInfo = liveness.getLiveness(block);
  if (!blockInfo)
    return 0;
  unsigned pressure = 0;
  for (Value liveVal : blockInfo->in())
    pressure += pressureContribution(liveVal);
  return pressure;
}

bool RegisterPressureAnalysis::isLiveIn(Block *block, Value value) const {
  const LivenessBlockInfo *blockInfo = liveness.getLiveness(block);
  return blockInfo && blockInfo->isLiveIn(value);
}

unsigned RegisterPressureAnalysis::liveInContribution(Block *block,
                                                      Value value) const {
  // Mirror liveInPressure's per-value accounting exactly, so that subtracting
  // this result from liveInPressure(block) yields the pressure the block would
  // report if `value` stopped being live-in.
  const LivenessBlockInfo *blockInfo = liveness.getLiveness(block);
  if (!blockInfo || !blockInfo->isLiveIn(value))
    return 0;
  return pressureContribution(value);
}

void RegisterPressureAnalysis::print(raw_ostream &os) const {
  Operation *rootOp = liveness.getOperation();
  if (!rootOp)
    return;

  os << "Register Pressure Analysis (per-thread bytes):\n";

  // The maximum over every block below, not the maximum in any one of them,
  // which is what a per-kernel allocation must cover. Reported first so that
  // adding it shifts the per-block lines below by exactly one line.
  os << "  Peak over all blocks in " << rootOp->getName() << ": "
     << peakPressureOverNestedBlocks(rootOp) << " bytes\n";

  // Walk all regions and blocks to report peak pressure. Qualify each block by
  // its parent op name so blocks in different regions (all named ^bb0) are
  // distinguishable in the output.
  rootOp->walk([&](Block *block) {
    unsigned peak = peakPressure(block);
    os << "  Block ";
    block->printAsOperand(os);
    os << " in " << block->getParentOp()->getName() << ": peak = " << peak
       << " bytes, live-in = " << liveInPressure(block) << " bytes\n";
  });
}

} // namespace mlir::triton::gpu::intel
