#include "intel/include/Analysis/RegisterPressure.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
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

unsigned RegisterPressureAnalysis::getGRFBytesPerHardwareThread(
    StringRef grfMode, UnknownGRFSizeAssumption unknownAssumption) {
  // Explicit GRF modes map to exact per-hardware-thread budgets (one hardware
  // thread executes a whole subgroup/warp of lanes sharing one register file).
  if (grfMode == "128")
    return 4096;
  if (grfMode == "256")
    return 8192;
  if (grfMode == "512")
    return 16384;
  // "default" and "auto": the compiler chooses the GRF size at JIT time, so
  // the true value isn't known here. Which bound is safe depends on the
  // caller; see UnknownGRFSizeAssumption's documentation.
  //
  // FIXME(#8074): Largest's 16384 is not per-target; see the enum's doc.
  return unknownAssumption == UnknownGRFSizeAssumption::Smallest ? 4096 : 16384;
}

unsigned RegisterPressureAnalysis::getPerLaneGRFBudgetInBytes(
    StringRef grfMode, ModuleOp mod,
    UnknownGRFSizeAssumption unknownAssumption) {
  unsigned grfBudget = getGRFBytesPerHardwareThread(grfMode, unknownAssumption);
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

const Liveness::ValueSetT &
RegisterPressureAnalysis::getRawLiveValuesAt(Operation *point,
                                             LiveValuesCache &cache) const {
  // currentlyLiveValues() is expensive and MLIR does not cache it, so memoize
  // it. What is memoized is the *unfiltered* result for `point`: whether the
  // point's own results are dropped depends on the role `point` plays in a
  // given query, not on `point` itself, so filtering before storing would let
  // one key stand for two different answers. A point with no liveness info
  // memoizes as the empty set, which is the answer it would recompute.
  auto [it, inserted] = cache.try_emplace(point);
  if (inserted) {
    if (const LivenessBlockInfo *blockInfo =
            liveness.getLiveness(point->getBlock()))
      it->second = blockInfo->currentlyLiveValues(point);
  }
  return it->second;
}

/// Returns true if \p use is exactly one of its owner's own "init" operands
/// -- an `scf.for`/`scf.while`-style operand forwarded, unread, straight into
/// a region entry block argument. Such a use is a pure conduit: the value
/// dies the instant the region starts executing, renamed to that block
/// argument, which the nested block's own liveness already counts.
static bool isLoopInitUse(OpOperand &use) {
  auto loopLike = dyn_cast<LoopLikeOpInterface>(use.getOwner());
  if (!loopLike)
    return false;
  for (OpOperand &init : loopLike.getInitsMutable())
    if (&init == &use)
      return true;
  return false;
}

/// Returns true if \p value has at least one "relevant" use with respect to
/// \p ancestor (either \p ancestor's own operand, or a use strictly inside
/// one of \p ancestor's regions), and *every* such relevant use is \p
/// ancestor's own operand *and* a pure forwarding conduit into \p ancestor's
/// own init list (`isLoopInitUse`) -- i.e. \p value's only role is to be
/// handed down into \p ancestor's own loop-carried state, never read
/// directly. Such a value is fully superseded: the block argument \p
/// ancestor's own entry rebinds it to is already counted by that block's own
/// raw liveness.
///
/// A relevant use nested *inside* one of \p ancestor's regions is never a
/// conduit at \p ancestor's own level, even if that use happens to be some
/// descendant loop's own init operand: the descendant re-evaluates that init
/// fresh every time \p ancestor's own back edge loops around, so \p value
/// must still be charged for \p ancestor's whole duration.
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

  // Build the result in a local set first, and only insert it into the cache
  // once fully built (rather than inserting a placeholder and mutating it
  // in place): the recursive call below may itself insert into `ancestorCache`
  // and rehash it, which would invalidate a reference obtained before the
  // recursive call.
  Liveness::ValueSetT set;

  // The "live after `ancestor`" set is fetched into a local copy *before* the
  // loop below, rather than queried inside it: `getRawLiveValuesAt` inserts
  // into `liveCache` on a miss, and doing that while the loop is iterating
  // over a `const &` into that same map (the loop's own range) risks a
  // rehash that invalidates the reference the loop is iterating over -- the
  // same hazard the comment above this function's cache guards against.
  Liveness::ValueSetT liveAfterAncestor;
  if (Operation *next = ancestor->getNextNode())
    liveAfterAncestor = getRawLiveValuesAt(next, liveCache);

  bool ancestorIsLoop = isa<LoopLikeOpInterface>(ancestor);

  for (Value liveVal : getRawLiveValuesAt(ancestor, liveCache)) {
    // `ancestor`'s own results don't hold a register yet while its region is
    // still executing -- the register they will end up in is the one holding
    // the value yielded out of the region, which the nested block's own
    // liveness already accounts for.
    if (liveVal.getDefiningOp() == ancestor)
      continue;

    if (ancestorIsLoop) {
      // Loops have a back edge: MLIR's own per-block liveness is blind to
      // the region re-executing, so a value merely *referenced* inside the
      // body -- even once, even indirectly through a nested loop's own
      // region -- must still be charged for the loop's whole duration,
      // unless it is genuinely superseded by a block argument at whatever
      // depth actually carries it (`isFullyForwardedThrough`, which folds in
      // the direct "ancestor's own init operand" case too) and is not itself
      // needed again after this loop closes.
      if (isFullyForwardedThrough(liveVal, ancestor) &&
          !liveAfterAncestor.contains(liveVal))
        continue;
    } else {
      // Single-execution region (`scf.if`, ...): no back edge, so a value
      // used only inside dies exactly where MLIR's own per-block liveness
      // already says it dies -- the nested block's own raw liveness covers
      // it up to that point. This level only needs to charge it if it is
      // *also* needed again after the region closes entirely.
      if (!liveAfterAncestor.contains(liveVal))
        continue;
    }

    set.insert(liveVal);
  }

  // Recurse into `ancestor`'s own parent chain, reusing (not recomputing) its
  // memoized entry if another query already built it.
  if (Operation *parent = ancestor->getParentOp();
      parent && parent != rootOp && parent->getBlock()) {
    const Liveness::ValueSetT &parentSet =
        getLiveThroughAncestorSet(parent, liveCache, ancestorCache, rootOp);
    set.insert(parentSet.begin(), parentSet.end());
  }

  return ancestorCache.try_emplace(ancestor, std::move(set)).first->second;
}

/// Returns the full region-aware live set at \p op -- its own block-local
/// raw live values unioned with whatever lives through every enclosing
/// region-holding op -- the one computation every accessor below shares, so
/// they cannot disagree about what is live at a nested point (see the class
/// doc comment).
Liveness::ValueSetT RegisterPressureAnalysis::computeLiveValues(
    Operation *op, LiveValuesCache &liveCache,
    AncestorLiveThroughCache &ancestorCache) const {
  Liveness::ValueSetT liveValues;
  for (Value liveVal : getRawLiveValuesAt(op, liveCache))
    liveValues.insert(liveVal);

  // A block's liveness info only knows about values defined or used within
  // that block's region. A value defined before an enclosing region-holding
  // op and used after it occupies a register for the whole duration of that
  // op, yet is neither defined nor used inside the nested block, so it is
  // absent from the nested block's live set. Union in the live-through
  // contribution of `op`'s immediate parent, which already transitively
  // covers every op enclosing it up to the analysis root.
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
  // Standalone query: there is no enclosing traversal to share liveness or
  // ancestor-union results with, so both caches live exactly as long as this
  // call.
  LiveValuesCache liveCache;
  AncestorLiveThroughCache ancestorCache;
  return pressureAt(op, liveCache, ancestorCache);
}

unsigned RegisterPressureAnalysis::pressureAt(
    Operation *op, LiveValuesCache &liveCache,
    AncestorLiveThroughCache &ancestorCache) const {
  // Accumulate the values live at `op` into a set rather than summing sizes as
  // they are found: the same value can be live at several nesting levels, and
  // summing per level would count it once per level.
  unsigned pressure = 0;
  for (Value liveVal : computeLiveValues(op, liveCache, ancestorCache))
    pressure += pressureContribution(liveVal);
  return pressure;
}

unsigned RegisterPressureAnalysis::pressureAt(Operation *op,
                                              QueryCache &cache) const {
  return pressureAt(op, cache.liveCache, cache.ancestorCache);
}

RegisterPressureAnalysis::PressureAtPoint
RegisterPressureAnalysis::pressureAt(Operation *op, Value value) const {
  // Standalone query: there is no enclosing traversal to share liveness or
  // ancestor-union results with, so both caches live exactly as long as this
  // call -- same rationale as the other single-op standalone overloads.
  LiveValuesCache liveCache;
  AncestorLiveThroughCache ancestorCache;

  PressureAtPoint result;
  for (Value liveVal : computeLiveValues(op, liveCache, ancestorCache)) {
    result.pressure += pressureContribution(liveVal);
    if (liveVal == value)
      result.valueLive = true;
  }
  return result;
}

unsigned RegisterPressureAnalysis::pressureBefore(Operation *op) const {
  LiveValuesCache liveCache;
  AncestorLiveThroughCache ancestorCache;

  unsigned pressure = 0;
  for (Value liveVal : computeLiveValues(op, liveCache, ancestorCache)) {
    // `pressureAt` counts a value at its defining op, which above `op` has not
    // run yet. Everything else live at `op` is also live immediately above it.
    if (liveVal.getDefiningOp() == op)
      continue;
    pressure += pressureContribution(liveVal);
  }

  return pressure;
}

unsigned RegisterPressureAnalysis::peakPressure(Block *block) const {
  // Standalone query: both caches live exactly as long as this call.
  LiveValuesCache liveCache;
  AncestorLiveThroughCache ancestorCache;
  return peakPressure(block, liveCache, ancestorCache);
}

unsigned RegisterPressureAnalysis::peakPressure(
    Block *block, LiveValuesCache &liveCache,
    AncestorLiveThroughCache &ancestorCache) const {
  unsigned peak = 0;

  // Walk the block, descending into the regions of any nested op, and track the
  // maximum pressure. A block whose ops are individually cheap can still reach
  // its peak inside a nested region (an scf.if body, an inner scf.for body,
  // ...); iterating only this block's own operations would report such a block
  // as low pressure.
  block->walk([&](Operation *op) {
    peak = std::max(peak, pressureAt(op, liveCache, ancestorCache));
  });

  return peak;
}

unsigned
RegisterPressureAnalysis::peakPressure(LoopLikeOpInterface loop) const {
  unsigned peak = 0;

  // One pair of caches for every block examined below. The loop's blocks share
  // their enclosing ops with one another, and a block nested in one of them
  // shares them too, so each enclosing op's live set (and live-through union)
  // is computed once for the whole loop rather than once per block.
  LiveValuesCache liveCache;
  AncestorLiveThroughCache ancestorCache;

  // A loop may expose multiple body regions, each with multiple blocks; check
  // all blocks across all loop regions.
  for (Region *region : loop.getLoopRegions()) {
    if (!region)
      continue;
    for (Block &block : *region)
      peak = std::max(peak, peakPressure(&block, liveCache, ancestorCache));
  }

  return peak;
}

unsigned
RegisterPressureAnalysis::peakPressureOverNestedBlocks(Operation *root) const {
  // One shared cache pair for every block visited below, matching every other
  // multi-block traversal in this file (`peakPressure(LoopLikeOpInterface)`,
  // `print()`'s per-block walk). The public, cache-less `peakPressure(Block*)`
  // would allocate a fresh pair per block here, getting no benefit at all
  // from the caches `peakPressure(Block*)` itself now needs to walk nested
  // regions cheaply.
  LiveValuesCache liveCache;
  AncestorLiveThroughCache ancestorCache;

  // A block's peak already covers every block nested inside it (see
  // `peakPressure(Block*)`'s doc comment), so the maximum over *every* block
  // in `root` is provably equal to the maximum over `root`'s own top-level
  // blocks -- no need to `walk` into nested regions a second time here.
  unsigned peak = 0;
  for (Region &region : root->getRegions())
    for (Block &block : region)
      peak = std::max(peak, peakPressure(&block, liveCache, ancestorCache));
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

  // One pair of caches for the whole traversal below, which visits every
  // block: a nested block is walked once on its own account and once on behalf
  // of every block enclosing it, so without sharing, the liveness of the
  // innermost ops (and the live-through union of every enclosing op) would be
  // recomputed once per nesting level.
  LiveValuesCache liveCache;
  AncestorLiveThroughCache ancestorCache;

  // Collect every block's line first, tracking the running maximum, so the
  // summary line below can be printed *before* the per-block breakdown
  // without a second traversal to compute it: a block's peak already covers
  // every block nested inside it, so the maximum over every line collected
  // here is exactly the whole-root peak.
  unsigned overallPeak = 0;
  SmallVector<std::string> blockLines;
  rootOp->walk([&](Block *block) {
    unsigned peak = peakPressure(block, liveCache, ancestorCache);
    overallPeak = std::max(overallPeak, peak);
    std::string line;
    llvm::raw_string_ostream lineOs(line);
    lineOs << "  Block ";
    block->printAsOperand(lineOs);
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
