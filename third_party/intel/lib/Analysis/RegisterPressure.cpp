#include "intel/include/Analysis/RegisterPressure.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

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

/// Per-hardware-thread GRF register width in bytes (256 bits); see
/// hardware-reference.md's GRF Register Specifications.
static constexpr unsigned GRFRegisterSizeBytes = 32;

/// Per-hardware-thread budget of the smallest ("128") and largest ("512")
/// explicit GRF modes, in bytes. Named so the `Smallest`/`Largest`
/// fallbacks below express themselves in the same unit as
/// `explicitGRFModeToBytes` rather than repeating the arithmetic as bare
/// literals.
static constexpr unsigned SmallestGRFModeBytes = 128 * GRFRegisterSizeBytes;
static constexpr unsigned LargestGRFModeBytes = 512 * GRFRegisterSizeBytes;

/// Maps an explicit GRF mode string ("128"/"192"/"256"/"512") to its exact
/// per-hardware-thread budget in bytes (one hardware thread executes a whole
/// subgroup/warp of lanes sharing one register file). Returns 0 for anything
/// else ("default", "auto", empty, or an unrecognized value): 0 bytes is
/// never a valid budget, so it is an unambiguous "not an explicit mode"
/// signal callers can test directly, without needing std::optional.
static unsigned explicitGRFModeToBytes(StringRef grfMode) {
  unsigned mode = 0;
  if (grfMode.getAsInteger(10, mode) ||
      (mode != 128 && mode != 192 && mode != 256 && mode != 512))
    return 0;
  return mode * GRFRegisterSizeBytes;
}

unsigned RegisterPressureAnalysis::getGRFBytesPerHardwareThread(
    StringRef grfMode, ModuleOp mod,
    UnknownGRFSizeAssumption unknownAssumption) {
  if (unsigned explicitBytes = explicitGRFModeToBytes(grfMode))
    return explicitBytes;
  assert((grfMode == "default" || grfMode == "auto") &&
         "grfMode must be an explicit mode (\"128\"/\"256\"/\"512\"), "
         "\"default\", or \"auto\"");
  // "default" and "auto": the compiler chooses the GRF size at JIT time, so
  // the true value isn't known here. Which bound is safe depends on the
  // caller; see UnknownGRFSizeAssumption's documentation.
  if (unknownAssumption == UnknownGRFSizeAssumption::Smallest)
    return SmallestGRFModeBytes;
  // A larger GRF mode reduces the maximum launchable work-group size, so a
  // num_warps > 32 kernel can never actually run at a larger mode, no matter
  // which mechanism would have picked it: the cap applies to both
  // `'default'` and `'auto'`.
  if (lookupNumWarps(mod) > 32)
    return SmallestGRFModeBytes;
  // Largest: the true ceiling is per-target, mirrored onto the module via the
  // ttig.max_grf_mode attribute (see UnknownGRFSizeAssumption::Largest's
  // documentation), and applies to both `'default'` and `'auto'`. Reuse the
  // same explicit-mode table above so a value other than exactly
  // "256"/"512"/"128" (a typo, a future mode, or the attribute being absent)
  // cannot silently resolve to the wrong budget: it falls through to the
  // behaviour-preserving 512-register-mode default below.
  if (auto maxGRFMode = mod->getAttrOfType<StringAttr>(
          TritonIntelGPUDialect::getMaxGRFModeAttrName()))
    if (unsigned explicitBytes = explicitGRFModeToBytes(maxGRFMode.getValue()))
      return explicitBytes;
  // Absence resolves to 512-register mode here, not `driver.c`'s "256" for a
  // missing `load_binary` argument: each side preserves its own pre-existing
  // behaviour on absence (this one hardcoded 16384 before `ttig.max_grf_mode`
  // existed; `driver.c` already resolved a missing arg to "unknown", which
  // already selected 256). Not a bug, but if either fallback's rationale
  // ever changes, check whether the other one still makes sense.
  return LargestGRFModeBytes;
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
  auto [it, inserted] = typeSizeCache.try_emplace(value.getType(), 0);
  if (inserted)
    it->second = getPerThreadSizeInBytes(value.getType());
  return it->second;
}

void RegisterPressureAnalysis::claimCache(QueryCache &cache) const {
  assert((!cache.owner || cache.owner == this) &&
         "QueryCache shared between two RegisterPressureAnalysis instances");
  cache.owner = this;
}

const Liveness::ValueSetT &
RegisterPressureAnalysis::getRawLiveValuesAt(Operation *point,
                                             LiveValuesCache &cache) const {
  // Memoize the unfiltered set: whether the point's own results count depends
  // on the query, so one key must not stand for two answers.
  auto [it, inserted] = cache.try_emplace(point);
  if (inserted) {
    // mlir::Liveness directly: the Intel wrapper asserts a direct child of the
    // root.
    if (const LivenessBlockInfo *blockInfo =
            liveness.getLiveness(point->getBlock()))
      it->second = blockInfo->currentlyLiveValues(point);
  }
  return it->second;
}

/// True if \p use is a loop init operand, which the body's own liveness counts.
static bool isLoopInitUse(OpOperand &use) {
  auto loopLike = dyn_cast<LoopLikeOpInterface>(use.getOwner());
  if (!loopLike)
    return false;
  for (OpOperand &init : loopLike.getInitsMutable())
    if (&init == &use)
      return true;
  return false;
}

/// Returns true if \p value has a use by or inside \p ancestor and every such
/// use is one of \p ancestor's own loop init operands. A use nested inside
/// \p ancestor never qualifies, even as an inner loop's init: the inner loop
/// re-reads it on every iteration of \p ancestor.
static bool isFullyForwardedThrough(Value value, Operation *ancestor) {
  bool hasRelevantUse = false;
  for (OpOperand &use : value.getUses()) {
    bool isAncestorsOwnOperand = use.getOwner() == ancestor;
    bool isInsideRegions = ancestor->isProperAncestor(use.getOwner());
    if (!isAncestorsOwnOperand && !isInsideRegions)
      continue;
    hasRelevantUse = true;
    if (!isAncestorsOwnOperand || !isLoopInitUse(use))
      return false;
  }
  return hasRelevantUse;
}

const Liveness::ValueSetT &RegisterPressureAnalysis::getLiveThroughAncestorSet(
    Operation *ancestor, LiveValuesCache &liveCache,
    AncestorLiveThroughCache &ancestorCache, Operation *rootOp) const {
  if (auto it = ancestorCache.find(ancestor); it != ancestorCache.end())
    return it->second;

  // Build locally: the recursive call below may rehash `ancestorCache`.
  Liveness::ValueSetT set;

  // A copy: the `getRawLiveValuesAt(ancestor, ...)` miss below may rehash
  // `liveCache`.
  Liveness::ValueSetT liveAfterAncestor;
  if (Operation *next = ancestor->getNextNode())
    liveAfterAncestor = getRawLiveValuesAt(next, liveCache);

  bool ancestorIsLoop = isa<LoopLikeOpInterface>(ancestor);

  for (Value liveVal : getRawLiveValuesAt(ancestor, liveCache)) {
    // `ancestor`'s results are not live while its region runs. The values
    // yielded into them are counted by the nested block.
    if (liveVal.getDefiningOp() == ancestor)
      continue;

    if (ancestorIsLoop) {
      // A back edge keeps any value read in the body live for the whole loop,
      // which per-block liveness misses. Skip only a value that is just this
      // loop's init operand and dead after it.
      if (isFullyForwardedThrough(liveVal, ancestor) &&
          !liveAfterAncestor.contains(liveVal))
        continue;
    } else {
      // No back edge: a value read only inside dies where block liveness
      // says. Charge it here only if it is also live after the region.
      if (!liveAfterAncestor.contains(liveVal))
        continue;
    }

    set.insert(liveVal);
  }

  if (Operation *parent = ancestor->getParentOp();
      parent && parent != rootOp && parent->getBlock()) {
    const Liveness::ValueSetT &parentSet =
        getLiveThroughAncestorSet(parent, liveCache, ancestorCache, rootOp);
    set.insert(parentSet.begin(), parentSet.end());
  }

  return ancestorCache.try_emplace(ancestor, std::move(set)).first->second;
}

Liveness::ValueSetT RegisterPressureAnalysis::computeLiveValues(
    Operation *op, LiveValuesCache &liveCache,
    AncestorLiveThroughCache &ancestorCache) const {
  Liveness::ValueSetT liveValues;
  for (Value liveVal : getRawLiveValuesAt(op, liveCache))
    liveValues.insert(liveVal);

  Operation *rootOp = liveness.getOperation();
  if (Operation *parent = op->getParentOp();
      parent && parent != rootOp && parent->getBlock()) {
    const Liveness::ValueSetT &liveThrough =
        getLiveThroughAncestorSet(parent, liveCache, ancestorCache, rootOp);
    liveValues.insert(liveThrough.begin(), liveThrough.end());
  }

  return liveValues;
}

unsigned RegisterPressureAnalysis::pressureAt(Operation *op) const {
  QueryCache cache;
  return pressureAt(op, cache);
}

unsigned RegisterPressureAnalysis::pressureAt(Operation *op,
                                              QueryCache &cache) const {
  claimCache(cache);
  // Sum over a set: a value live at several nesting levels counts once.
  unsigned pressure = 0;
  for (Value liveVal :
       computeLiveValues(op, cache.liveCache, cache.ancestorCache))
    pressure += pressureContribution(liveVal);
  return pressure;
}

RegisterPressureAnalysis::PressureAtPoint
RegisterPressureAnalysis::pressureAt(Operation *op, Value value) const {
  QueryCache cache;
  return pressureAt(op, value, cache);
}

RegisterPressureAnalysis::PressureAtPoint
RegisterPressureAnalysis::pressureAt(Operation *op, Value value,
                                     QueryCache &cache) const {
  claimCache(cache);
  PressureAtPoint result;
  for (Value liveVal :
       computeLiveValues(op, cache.liveCache, cache.ancestorCache)) {
    result.pressure += pressureContribution(liveVal);
    if (liveVal == value)
      result.valueLive = true;
  }
  return result;
}

bool RegisterPressureAnalysis::isLiveAt(Operation *op, Value value,
                                        QueryCache &cache) const {
  claimCache(cache);
  if (getRawLiveValuesAt(op, cache.liveCache).contains(value))
    return true;
  Operation *rootOp = liveness.getOperation();
  Operation *parent = op->getParentOp();
  if (!parent || parent == rootOp || !parent->getBlock())
    return false;
  const Liveness::ValueSetT &liveThrough = getLiveThroughAncestorSet(
      parent, cache.liveCache, cache.ancestorCache, rootOp);
  return liveThrough.contains(value);
}

unsigned RegisterPressureAnalysis::pressureBefore(Operation *op) const {
  QueryCache cache;

  unsigned pressure = 0;
  for (Value liveVal :
       computeLiveValues(op, cache.liveCache, cache.ancestorCache)) {
    // `pressureAt` counts a value at its defining op, which above `op` has not
    // run yet. Everything else live at `op` is also live immediately above it.
    if (liveVal.getDefiningOp() == op)
      continue;
    pressure += pressureContribution(liveVal);
  }

  return pressure;
}

unsigned RegisterPressureAnalysis::peakPressure(Block *block) const {
  QueryCache cache;
  return peakPressure(block, cache);
}

unsigned RegisterPressureAnalysis::peakPressure(Block *block,
                                                QueryCache &cache) const {
  claimCache(cache);
  if (auto it = cache.blockPeakCache.find(block);
      it != cache.blockPeakCache.end())
    return it->second;

  unsigned peak = 0;

  block->walk(
      [&](Operation *op) { peak = std::max(peak, pressureAt(op, cache)); });

  cache.blockPeakCache.try_emplace(block, peak);
  return peak;
}

unsigned
RegisterPressureAnalysis::peakPressure(LoopLikeOpInterface loop) const {
  unsigned peak = 0;
  QueryCache cache;

  // A loop may expose multiple body regions, each with multiple blocks; check
  // all blocks across all loop regions.
  for (Region *region : loop.getLoopRegions()) {
    if (!region)
      continue;
    for (Block &block : *region)
      peak = std::max(peak, peakPressure(&block, cache));
  }

  return peak;
}

unsigned
RegisterPressureAnalysis::peakPressure(FunctionOpInterface func) const {
  QueryCache cache;
  return peakPressure(func, cache);
}

unsigned RegisterPressureAnalysis::peakPressure(FunctionOpInterface func,
                                                QueryCache &cache) const {
  // A block's peak covers its nested blocks, so top-level blocks suffice.
  unsigned peak = 0;
  for (Region &region : func->getRegions())
    for (Block &block : region)
      peak = std::max(peak, peakPressure(&block, cache));
  return peak;
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

  QueryCache cache;

  // Buffer the lines so the overall peak prints first without a second walk.
  unsigned overallPeak = 0;
  SmallVector<std::string> blockLines;
  rootOp->walk([&](Block *block) {
    unsigned peak = peakPressure(block, cache);
    overallPeak = std::max(overallPeak, peak);
    std::string line;
    llvm::raw_string_ostream lineOs(line);
    lineOs << "  Block ";
    block->printAsOperand(lineOs);
    // Qualify each block by its parent op name so blocks in different regions
    // (all named ^bb0) are distinguishable in the output.
    lineOs << " in " << block->getParentOp()->getName() << ": peak = " << peak
           << " bytes, live-in = " << liveInPressure(block) << " bytes\n";
    blockLines.push_back(std::move(line));
  });

  // The maximum over every block, not the maximum in any one of them, which
  // is what a per-kernel allocation must cover.
  os << "  Peak over all blocks in " << rootOp->getName() << ": " << overallPeak
     << " bytes\n";
  for (const std::string &line : blockLines)
    os << line;
}

} // namespace mlir::triton::gpu::intel
