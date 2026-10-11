#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"

#include "intel/include/Analysis/RegisterPressure.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <cstdint>
#include <optional>

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
STATISTIC(NumRejectedFunctionPeakExact,
          "Number of convert_layout ops rejected because the exactly projected "
          "whole-function peak pressure would rise past the ceiling");
STATISTIC(
    NumRejectedFunctionPeakFallback,
    "Number of convert_layout ops rejected on a conservatively charged "
    "whole-function peak projection (a cap or a structure that could not be "
    "walked)");
STATISTIC(NumSkippedOther,
          "Number of convert_layout ops skipped (not eligible)");

namespace {

/// Where one operation sits relative to another. `isBeforeInBlock` alone is the
/// wrong question: it sees neither of two ops as after the other when one
/// contains the other, and `findAncestorOpInBlock` maps a nested operation onto
/// the containing one, which also looks like "not after".
enum class SubtreeOrder {
  Same,    ///< The very same operation.
  Inside,  ///< Nested somewhere inside the reference operation.
  Before,  ///< Runs strictly before, compared in the reference's own block.
  After,   ///< Runs strictly after, compared in the reference's own block.
  Unknown, ///< Cannot be placed relative to the reference at all.
};

/// Why a candidate was refused. Refusals are recorded, not applied, so
/// `reconsiderSharedSources` can overturn them.
enum class RefusalReason {
  Pressure,
  FunctionPeakExact,
  FunctionPeakFallback,
};

struct Refusal {
  ttg::ConvertLayoutOp cvtOp;
  scf::ForOp forOp;
  RefusalReason reason;
  /// Set once a joint hoist overturns this refusal.
  bool overturned = false;
  /// Set once this refusal's loop is charged for its merge into a hoist (see
  /// Twin merges), so neither `mergeFits` nor `reconsiderSharedSources`
  /// prices that merge again.
  bool mergeCharged = false;
};

/// Returns where \p subject sits relative to \p reference. `Unknown` means
/// \p subject lives in a region that does not nest under \p reference's block
/// (including a region *containing* it), so callers must take the conservative
/// branch rather than guess.
static SubtreeOrder subtreeOrder(Operation *subject, Operation *reference) {
  if (subject == reference)
    return SubtreeOrder::Same;
  if (reference->isProperAncestor(subject))
    return SubtreeOrder::Inside;
  Operation *ancestor = reference->getBlock()->findAncestorOpInBlock(*subject);
  if (!ancestor)
    return SubtreeOrder::Unknown;
  return reference->isBeforeInBlock(ancestor) ? SubtreeOrder::After
                                              : SubtreeOrder::Before;
}

// Twin merges. A twin of a candidate is a refused conversion of the same
// source to the same type. Once the candidate is hoisted,
// remove_layout_conversions replaces every twin the hoisted conversion
// properly dominates with its result, except for a dot operand with a blocked
// parent. The twin's loop then ends up as if the twin had been hoisted, so:
//  - hoistRetiresSource may treat such a twin as not reading the source,
//    subject to mergeFits;
//  - the twin's loop is charged the twin's hoistDeltaBytes once, whichever of
//    the hoist and the refusal is decided first (chargeMergedTwins,
//    chargeMergeIntoPriorHoist), and each such refusal is marked
//    mergeCharged.
// A twin in the hoisting loop itself is marked but not charged: that loop's
// delta already charged the result, and errs high by at most the source.

/// Returns true if \p twin merges into \p cvtOp's result once \p cvtOp leaves
/// \p forOp. Below a defining op, the hoisted conversion dominates every use
/// of the source. For a block-argument source it lands just above \p forOp,
/// so it dominates only twins inside or after \p forOp.
static bool mergesIntoHoist(ttg::ConvertLayoutOp cvtOp, Operation *twin,
                            scf::ForOp forOp) {
  auto rtType = dyn_cast<RankedTensorType>(cvtOp.getType());
  auto dotEnc =
      rtType
          ? dyn_cast_or_null<ttg::DotOperandEncodingAttr>(rtType.getEncoding())
          : ttg::DotOperandEncodingAttr();
  if (!dotEnc || isa<ttg::BlockedEncodingAttr>(dotEnc.getParent()))
    return false;
  if (cvtOp.getSrc().getDefiningOp())
    return true;
  SubtreeOrder order = subtreeOrder(twin, forOp);
  return order == SubtreeOrder::Inside || order == SubtreeOrder::After;
}

/// Appends to \p twinIdxs the index of every non-overturned refusal that
/// merges into \p cvtOp's result once it is hoisted. Unlike
/// `hoistRetiresSource`, ignores `mergeFits`: the merge happens whether or not
/// the credit relied on it.
static void collectMergingTwins(ttg::ConvertLayoutOp cvtOp, scf::ForOp forOp,
                                ArrayRef<Refusal> refusals,
                                SmallVectorImpl<unsigned> &twinIdxs) {
  for (auto [idx, r] : llvm::enumerate(refusals)) {
    ttg::ConvertLayoutOp otherCvt = r.cvtOp;
    if (r.overturned || otherCvt.getSrc() != cvtOp.getSrc() ||
        otherCvt.getType() != cvtOp.getType())
      continue;
    if (mergesIntoHoist(cvtOp, otherCvt.getOperation(), forOp))
      twinIdxs.push_back(idx);
  }
}

/// Returns true if hoisting \p cvtOp out of \p forOp retires \p cvtOp's source
/// from the loop, i.e. once \p cvtOp has moved out, nothing remaining inside
/// the loop reads the source *and* nothing after the loop does either, so it
/// stops being live-in to the loop body.
///
/// Both halves are asked of the *current* IR rather than of the (immutable)
/// liveness analysis, because hoists this pass has already performed must count
/// as moved. A source read below the loop occupies a register for the loop's
/// whole duration wherever the conversion sits, so hoisting frees nothing --
/// but "below the loop" must mean below it *now*: when the reader below is
/// itself a conversion this pass already hoisted above the loop, the credit is
/// real. Asking the frozen analysis made the answer depend on which of two
/// loops sharing a source the pass reached first (`runOnOperation` visits loops
/// last-to-first so this view is final).
///
/// \p priorRefusals excuses a reader that is a refused twin (see Twin merges).
/// With \p mergeFits set, a twin in another loop that is not yet
/// `mergeCharged` is excused only if \p mergeFits accepts the merge for that
/// loop, so the credit never pushes another loop past its gate.
static bool
hoistRetiresSource(ttg::ConvertLayoutOp cvtOp, scf::ForOp forOp,
                   ArrayRef<Refusal> priorRefusals = {},
                   llvm::function_ref<bool(const Refusal &)> mergeFits = {}) {
  auto isUnenforceableTwin = [&](Operation *user) {
    auto otherCvt = dyn_cast<ttg::ConvertLayoutOp>(user);
    if (!otherCvt || otherCvt.getSrc() != cvtOp.getSrc() ||
        otherCvt.getType() != cvtOp.getType() ||
        !mergesIntoHoist(cvtOp, user, forOp))
      return false;
    for (const Refusal &r : priorRefusals) {
      if (r.overturned || r.cvtOp != otherCvt)
        continue;
      return !mergeFits || r.mergeCharged || r.forOp == forOp || mergeFits(r);
    }
    return false;
  };
  for (Operation *user : cvtOp.getSrc().getUsers()) {
    if (user == cvtOp.getOperation() || isUnenforceableTwin(user))
      continue;
    switch (subtreeOrder(user, forOp)) {
    case SubtreeOrder::Inside:
      return false;
    case SubtreeOrder::After:
      return false;
    case SubtreeOrder::Unknown:
      // A user that cannot be placed in the loop's own block (a different
      // region altogether) might run after the loop, so refuse the credit.
      return false;
    case SubtreeOrder::Same:
      // The loop itself reads the source (an `iter_args` initializer or a
      // bound). That read happens before the body runs, so the source does not
      // cross the loop.
      break;
    case SubtreeOrder::Before:
      break;
    }
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
/// below it) reads the source, that source stops crossing it. Modelling only
/// the arrival overestimates the cost and rejects hoists that would have
/// *lowered* pressure -- notably when the source layout is per-lane fatter than
/// the dot-operand layout it feeds. See
/// https://github.com/intel/intel-xpu-backend-for-triton/issues/7993.
///
/// The credit comes from the analysis rather than the source's type because
/// `liveInPressure` filters out rematerializable values: a constant source
/// never occupied the bytes its type suggests.
///
/// \p priorRefusals and \p mergeFits are as for `hoistRetiresSource`.
static int
hoistDeltaBytes(ttg::ConvertLayoutOp cvtOp, scf::ForOp forOp,
                const ttg::intel::RegisterPressureAnalysis &analysis,
                ArrayRef<Refusal> priorRefusals = {},
                llvm::function_ref<bool(const Refusal &)> mergeFits = {}) {
  unsigned hoistBytes =
      ttg::intel::RegisterPressureAnalysis::getPerThreadSizeInBytes(
          cvtOp.getType());
  unsigned retiredBytes =
      hoistRetiresSource(cvtOp, forOp, priorRefusals, mergeFits)
          ? analysis.liveInContribution(forOp.getBody(), cvtOp.getSrc())
          : 0;
  return static_cast<int>(hoistBytes) - static_cast<int>(retiredBytes);
}

/// Returns the operation a hoisted \p cvtOp lands immediately *after*, or null
/// when it lands at the start of \p forOp's block (nothing precedes the loop
/// and the source has no defining operation to sit below).
///
/// The move itself is derived from this helper, so the point the projection
/// measures and the point the conversion lands at cannot drift apart.
static Operation *hoistAnchor(ttg::ConvertLayoutOp cvtOp, scf::ForOp forOp) {
  if (Operation *srcDefOp = cvtOp.getSrc().getDefiningOp())
    return srcDefOp;
  return forOp->getPrevNode();
}

static void moveToHoistAnchor(ttg::ConvertLayoutOp cvtOp, scf::ForOp forOp) {
  if (Operation *anchor = hoistAnchor(cvtOp, forOp))
    cvtOp->moveAfter(anchor);
  else
    cvtOp->moveBefore(forOp);
}

/// Returns the outermost loop-like ancestor of \p op, or null when \p op is not
/// in a loop. Anything nested inside that loop re-executes each iteration, so
/// its lexical position relative to \p op says nothing about whether it also
/// runs *after* \p op.
static Operation *outermostLoopAncestor(Operation *op) {
  Operation *outermost = nullptr;
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp())
    if (isa<LoopLikeOpInterface>(parent))
      outermost = parent;
  return outermost;
}

/// Returns true if no region enclosing \p op can contain a CFG cycle, i.e.
/// every one holds a single block whose terminator has no successors.
///
/// `outermostLoopAncestor` keys on `LoopLikeOpInterface`, which misses a cycle
/// built from `cf.br` back edges, and nothing in the candidate predicate rules
/// one out (`test/Analysis/test-membar.mlir` has an irreducible one). Testing
/// \p op's own block for multiple predecessors would not do: a cycle's header
/// can carry the extra predecessor while \p op sits in a body block with one.
static bool hasNoBranchCycles(Operation *op) {
  for (Region *region = op->getParentRegion(); region;
       region = region->getParentRegion()) {
    if (!region->hasOneBlock())
      return false;
    // A region whose block has no terminator (a graph region such as a
    // `ModuleOp` body) has no CFG edges either.
    Block &block = region->front();
    if (block.mightHaveTerminator() &&
        block.getTerminator()->getNumSuccessors() != 0)
      return false;
  }
  return true;
}

/// The position-independent half of "is the conversion's source dead at this
/// corridor operation once the conversion has moved above it?", computed once
/// per candidate instead of once per corridor operation.
struct SourceLiveness {
  /// The latest operation in the loop's own block a remaining user of the
  /// source maps onto, or null when there is none. Only the latest matters:
  /// "some remaining user runs at or after this point" is "the last one does".
  Operation *lastUser = nullptr;
  /// False when no corridor operation may be credited for the source at all.
  bool creditable = false;
  /// False when reaching that answer took a conservative shortcut, so a
  /// rejection based on it is not evidence of a real pressure rise.
  bool exact = true;
};

/// Classifies the remaining users of \p cvtOp's source once for the whole
/// corridor. \p srcUsers is the source's deduplicated user list (which
/// includes \p cvtOp).
static SourceLiveness classifySource(ttg::ConvertLayoutOp cvtOp,
                                     scf::ForOp forOp,
                                     ArrayRef<Operation *> srcUsers) {
  SourceLiveness result;

  // Withholding the credit is often the *correct* answer rather than a
  // conservative one: with a reader below the loop the source really is
  // unrelieved at every corridor operation, and in the common shape -- one
  // source read by a conversion in each of two adjacent loops -- the corridor
  // term's exactness, not a credit, keeps the hoist alive. But a body reader
  // *above* the conversion lets the source die mid-corridor, so every later
  // corridor operation is overpriced, and an unorderable reader is not known
  // either way. Those two are fallbacks, not evidence of a pressure rise.
  if (!hoistRetiresSource(cvtOp, forOp)) {
    Block *cvtBlock = cvtOp->getBlock();
    for (Operation *user : srcUsers) {
      if (user == cvtOp.getOperation())
        continue;
      SubtreeOrder order = subtreeOrder(user, forOp);
      if (order == SubtreeOrder::Unknown) {
        result.exact = false;
        return result;
      }
      if (order != SubtreeOrder::Inside)
        continue;
      Operation *mapped = cvtBlock->findAncestorOpInBlock(*user);
      if (!mapped || mapped->isBeforeInBlock(cvtOp)) {
        result.exact = false;
        return result;
      }
    }
    return result;
  }

  if (!hasNoBranchCycles(cvtOp)) {
    result.exact = false;
    return result;
  }

  Block *loopBlock = forOp->getBlock();
  Operation *enclosingLoop = outermostLoopAncestor(forOp);
  for (Operation *user : srcUsers) {
    if (user == cvtOp.getOperation())
      continue;
    if (enclosingLoop && enclosingLoop->isProperAncestor(user)) {
      // An enclosing loop's back edge re-reads the source after the corridor.
      // That is not modelled: price as if the source stayed live, and mark the
      // result inexact.
      result.exact = false;
      return result;
    }
    Operation *mapped = loopBlock->findAncestorOpInBlock(*user);
    if (!mapped) {
      // Unreachable while `hoistRetiresSource` holds -- it already refuses the
      // credit for a user it cannot place in this block. Kept so the two cannot
      // drift apart into an unsound credit.
      result.exact = false;
      return result;
    }
    if (!result.lastUser || result.lastUser->isBeforeInBlock(mapped))
      result.lastUser = mapped;
  }

  result.creditable = true;
  return result;
}

/// Returns true if the conversion's source is dead at corridor operation \p op
/// once the conversion has moved above it. Callers must have established that
/// `SourceLiveness::creditable` holds.
static bool srcDeadAfterMoveAt(const SourceLiveness &srcLiveness, Operation *op,
                               scf::ForOp forOp) {
  if (!srcLiveness.lastUser)
    return true;
  // A corridor operation in the loop's own block compares directly against
  // `lastUser`, which lives there too.
  if (op->getBlock() == forOp->getBlock())
    return srcLiveness.lastUser->isBeforeInBlock(op);
  // For a body operation, no remaining user is inside the loop
  // (`hoistRetiresSource`), so only a user before the loop or the loop itself
  // is left -- and the loop's own read (an `iter_args` initializer or a bound)
  // also happens before the body runs. Either way the source is dead in the
  // body.
  return srcLiveness.lastUser == forOp.getOperation() ||
         srcLiveness.lastUser->isBeforeInBlock(forOp);
}

/// Returns the last op in \p body that uses \p cvtOp's result, mapping a
/// nested use to its top-level op in \p body, or null if there is none.
static Operation *lastBodyUser(ttg::ConvertLayoutOp cvtOp, Block *body) {
  Operation *last = nullptr;
  for (Operation *user : cvtOp.getResult().getUsers()) {
    Operation *mapped = body->findAncestorOpInBlock(*user);
    if (!mapped)
      continue;
    if (!last || last->isBeforeInBlock(mapped))
      last = mapped;
  }
  return last;
}

/// Returns the last user of \p cvtOp's result, or null if it has none or its
/// users are in different blocks, which have no order.
static Operation *realLastUse(ttg::ConvertLayoutOp cvtOp) {
  Operation *last = nullptr;
  for (Operation *user : cvtOp.getResult().getUsers()) {
    if (!last) {
      last = user;
      continue;
    }
    if (last->getBlock() != user->getBlock())
      return nullptr;
    if (last->isBeforeInBlock(user))
      last = user;
  }
  return last;
}

/// Collects into \p corridor the ops at which a hoisted \p cvtOp's result
/// becomes newly live: the tail of \p anchor's block through \p forOp, the
/// body ops above \p cvtOp, and the body ops after its last body use, since
/// the back edge keeps a loop-invariant result live through the whole body.
/// Ops from \p cvtOp through that last use are skipped: the result is already
/// live there. Region-holding entries are not descended into;
/// `projectedFunctionPeak` prices their internal peak.
///
/// Returns false when the corridor cannot be walked, or is longer than
/// \p corridorOpCap; the caller must then fall back to a conservative charge.
static bool collectCorridor(ttg::ConvertLayoutOp cvtOp, scf::ForOp forOp,
                            Operation *anchor, unsigned corridorOpCap,
                            SmallVectorImpl<Operation *> &corridor) {
  Block *loopBlock = forOp->getBlock();
  // The candidate predicate allows a source defined in an enclosing block (the
  // loop nested in an `scf.if`, say), but its corridor crosses region
  // boundaries this walk does not model.
  if (anchor && anchor->getBlock() != loopBlock)
    return false;
  if (!forOp.getRegion().hasOneBlock())
    return false;
  if (cvtOp->getBlock() != forOp.getBody())
    return false;

  // The tail of the loop's own block, from just below the anchor up to and
  // including the loop: the result is live across each of these once the
  // conversion sits above them.
  Operation *op = anchor ? anchor->getNextNode() : &loopBlock->front();
  while (op) {
    if (corridor.size() >= corridorOpCap)
      return false;
    corridor.push_back(op);
    if (op == forOp.getOperation())
      break;
    op = op->getNextNode();
  }
  if (!op)
    return false; // The loop does not follow the anchor in this block.

  // The loop body operations the conversion is being lifted over.
  for (Operation *bodyOp = &forOp.getBody()->front();
       bodyOp != cvtOp.getOperation(); bodyOp = bodyOp->getNextNode()) {
    if (corridor.size() >= corridorOpCap)
      return false;
    corridor.push_back(bodyOp);
  }

  Operation *lastUse = lastBodyUser(cvtOp, forOp.getBody());
  Operation *start =
      lastUse ? lastUse->getNextNode() : cvtOp.getOperation()->getNextNode();
  for (Operation *bodyOp = start; bodyOp; bodyOp = bodyOp->getNextNode()) {
    if (corridor.size() >= corridorOpCap)
      return false;
    corridor.push_back(bodyOp);
  }

  return true;
}

/// Returns the peak pressure over the blocks of \p op's regions, memoized in
/// \p regionPeakCache. Values live across \p op are already in it, so callers
/// add only the arriving result.
static uint64_t regionPeakThroughOp(
    Operation *op, const ttg::intel::RegisterPressureAnalysis &analysis,
    ttg::intel::RegisterPressureAnalysis::QueryCache &queryCache,
    DenseMap<Operation *, uint64_t> &regionPeakCache) {
  auto [it, inserted] = regionPeakCache.try_emplace(op, 0);
  if (!inserted)
    return it->second;

  uint64_t peak = 0;
  for (Region &region : op->getRegions())
    for (Block &block : region)
      peak = std::max(peak, static_cast<uint64_t>(
                                analysis.peakPressure(&block, queryCache)));
  it->second = peak;
  return peak;
}

/// Prices each op after \p start in its block the way the corridor prices a
/// body op after the last use, including any region-holding op's internal
/// peak. \p dstBytes is added only where \p cvtResult is not already live,
/// since an enclosing inner loop's back edge may keep it live. Returns false,
/// with \p peak unspecified, past \p corridorOpCap ops.
static bool priceTailAfter(
    Operation *start, Value src, uint64_t srcBytes, Value cvtResult,
    uint64_t dstBytes, const ttg::intel::RegisterPressureAnalysis &analysis,
    const SourceLiveness &srcLiveness, scf::ForOp forOp, unsigned corridorOpCap,
    ttg::intel::RegisterPressureAnalysis::QueryCache &queryCache,
    DenseMap<Operation *, uint64_t> &regionPeakCache, uint64_t &peak) {
  peak = 0;
  unsigned walked = 0;
  for (Operation *op = start->getNextNode(); op; op = op->getNextNode()) {
    if (++walked > corridorOpCap)
      return false;

    auto point = analysis.pressureAt(op, src, queryCache);
    bool resultAlreadyLive = analysis.isLiveAt(op, cvtResult, queryCache);
    uint64_t priced = point.pressure + (resultAlreadyLive ? 0 : dstBytes);
    bool credit = srcLiveness.creditable && point.valueLive &&
                  srcDeadAfterMoveAt(srcLiveness, op, forOp);
    if (credit)
      priced -= srcBytes;
    peak = std::max(peak, priced);

    if (op->getNumRegions() > 0) {
      uint64_t regionPriced =
          regionPeakThroughOp(op, analysis, queryCache, regionPeakCache) +
          (resultAlreadyLive ? 0 : dstBytes);
      if (credit)
        regionPriced -= srcBytes;
      peak = std::max(peak, regionPriced);
    }
  }
  return true;
}

/// Returns the peak over the blocks of \p op's regions other than
/// \p excluded, priced as in `priceTailAfter`. Once hoisted, the result is
/// live through every block of the loop body, including branches that do not
/// use it.
static uint64_t siblingBlocksPeak(
    Operation *op, Block *excluded, Value src, uint64_t srcBytes,
    Value cvtResult, uint64_t dstBytes,
    const ttg::intel::RegisterPressureAnalysis &analysis,
    const SourceLiveness &srcLiveness, scf::ForOp forOp,
    ttg::intel::RegisterPressureAnalysis::QueryCache &queryCache) {
  uint64_t peak = 0;
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      if (&block == excluded || block.empty())
        continue;
      // Asking at the first op is exact here: no sibling block uses the result
      // (`realLastUse`) and no other in-loop op reads `src` (`creditable`), so
      // either value's liveness comes only from the ancestor set, which is the
      // same for every op in the block.
      Operation *rep = &block.front();
      bool resultAlreadyLive =
          analysis.pressureAt(rep, cvtResult, queryCache).valueLive;
      uint64_t priced = analysis.peakPressure(&block, queryCache) +
                        (resultAlreadyLive ? 0 : dstBytes);
      bool credit = srcLiveness.creditable &&
                    analysis.pressureAt(rep, src, queryCache).valueLive &&
                    srcDeadAfterMoveAt(srcLiveness, rep, forOp);
      if (credit)
        priced -= srcBytes;
      peak = std::max(peak, priced);
    }
  }
  return peak;
}

/// The terms of the whole-function peak projection, kept apart so the debug log
/// can name the one that dominated.
struct PeakProjection {
  /// The peak over blocks the hoist does not touch, which it cannot lower. The
  /// peak commonly lives in such a block; without this term the projection
  /// could come out *below* the measured post-hoist peak.
  uint64_t prePeak = 0;
  /// The peak over the corridor, with the conversion's result charged and its
  /// source credited where the move really retires it.
  uint64_t corridorTerm = 0;
  /// The pressure at the conversion's new program point.
  uint64_t newPointTerm = 0;
  /// False when any term had to be charged conservatively.
  bool exact = true;
  /// Corridor length actually walked, for the debug log.
  unsigned corridorLength = 0;

  uint64_t total() const {
    return std::max({prePeak, corridorTerm, newPointTerm});
  }
};

/// Returns an upper bound on the whole-function peak pressure -- as
/// `RegisterPressureAnalysis` reports it -- that hoisting \p cvtOp out of
/// \p forOp would leave behind, without mutating any IR. \p analysis must
/// describe the function as it stands now and \p prePeak must be its measured
/// whole-function peak.
///
/// This bounds the *reported* metric, not physical allocator demand: a real
/// allocator could still do better (rematerializing, spilling around a
/// narrow peak) than the flat per-thread-bytes figure this analysis reports,
/// same as for the loop-level gate. The arithmetic is widened to 64 bits to
/// keep the added terms from wrapping.
static PeakProjection projectedFunctionPeak(
    ttg::ConvertLayoutOp cvtOp, scf::ForOp forOp,
    const ttg::intel::RegisterPressureAnalysis &analysis, uint64_t prePeak,
    ArrayRef<Operation *> srcUsers, unsigned corridorOpCap,
    ttg::intel::RegisterPressureAnalysis::QueryCache &queryCache,
    DenseMap<Operation *, uint64_t> &regionPeakCache) {
  PeakProjection projection;
  projection.prePeak = prePeak;

  // Ask the analysis what the result weighs rather than deriving it from the
  // type: this must bound the figure the analysis *reports*, and it charges
  // nothing for a value nothing reads. Charging from the type would be sound
  // but would veto a hoist that provably cannot move the reported peak.
  uint64_t dstBytes = analysis.pressureContribution(cvtOp.getResult());
  Operation *anchor = hoistAnchor(cvtOp, forOp);

  // The new program point, named by the operation that will follow it: what is
  // live above that operation, plus the arriving result (the source is already
  // in that set). `pressureAt` would be wrong here -- it counts a value at its
  // defining operation, charging results that have not run yet.
  Operation *below =
      anchor ? anchor->getNextNode() : &forOp->getBlock()->front();
  if (below) {
    projection.newPointTerm = analysis.pressureBefore(below) + dstBytes;
  } else {
    // Nothing follows the anchor in its block, so there is no program point
    // between two operations to measure.
    projection.exact = false;
    projection.newPointTerm = prePeak + dstBytes;
  }

  SmallVector<Operation *> corridor;
  if (!collectCorridor(cvtOp, forOp, anchor, corridorOpCap, corridor)) {
    // Charge every unwalked point as if the result were newly live there and
    // nothing retired. `prePeak` alone would be unsound: a point already at
    // `prePeak` becomes `prePeak + dstBytes` once the result is live across it.
    projection.exact = false;
    projection.corridorTerm = prePeak + dstBytes;
    return projection;
  }
  projection.corridorLength = corridor.size();

  SourceLiveness srcLiveness = classifySource(cvtOp, forOp, srcUsers);
  projection.exact &= srcLiveness.exact;

  Value src = cvtOp.getSrc();
  uint64_t srcBytes = analysis.pressureContribution(src);
  for (Operation *op : corridor) {
    // The result is newly live at every corridor op: above its old position
    // because its live range now starts earlier, after its last use because
    // the back edge keeps it live through the body. A result yielded out of
    // the loop is a different value, so nothing after the loop needs pricing.
    auto point = analysis.pressureAt(op, src, queryCache);
    uint64_t priced = point.pressure + dstBytes;
    // Only subtract what the analysis actually counted here, and only where the
    // move really does retire it.
    if (srcLiveness.creditable && point.valueLive &&
        srcDeadAfterMoveAt(srcLiveness, op, forOp))
      priced -= srcBytes;
    projection.corridorTerm = std::max(projection.corridorTerm, priced);

    // A region-holding entry can peak inside its region. `forOp` is excluded:
    // its body is priced op by op in this corridor.
    if (op->getNumRegions() > 0 && op != forOp.getOperation()) {
      uint64_t regionPriced =
          regionPeakThroughOp(op, analysis, queryCache, regionPeakCache) +
          dstBytes;
      // The same credit holds inside the region: `srcLiveness` already rules
      // out any reader of `src` at or after `op` other than `cvtOp`.
      if (srcLiveness.creditable && point.valueLive &&
          srcDeadAfterMoveAt(srcLiveness, op, forOp))
        regionPriced -= srcBytes;
      projection.corridorTerm = std::max(projection.corridorTerm, regionPriced);
    }
  }

  // If the last body use maps to a region-holding op, the ops after the
  // nested use inside it also see the result newly live, but the corridor
  // skipped that op. The tail walk covers only the ops after the nested use in
  // its block, so it covers the whole tail only when that block is directly in
  // the op and the use holds no region itself. Otherwise charge the op's
  // whole-region peak and mark the result inexact.
  if (Operation *lastUse = lastBodyUser(cvtOp, forOp.getBody());
      lastUse && lastUse->getNumRegions() > 0) {
    Operation *nestedLast = realLastUse(cvtOp);
    uint64_t tailPeak = 0;
    if (nestedLast && nestedLast->getBlock()->getParentOp() == lastUse &&
        nestedLast->getNumRegions() == 0 &&
        priceTailAfter(nestedLast, src, srcBytes, cvtOp.getResult(), dstBytes,
                       analysis, srcLiveness, forOp, corridorOpCap, queryCache,
                       regionPeakCache, tailPeak)) {
      projection.corridorTerm = std::max(projection.corridorTerm, tailPeak);

      // The mapped op's other blocks, such as an `else` branch.
      uint64_t siblingPeak = siblingBlocksPeak(
          lastUse, nestedLast->getBlock(), src, srcBytes, cvtOp.getResult(),
          dstBytes, analysis, srcLiveness, forOp, queryCache);
      projection.corridorTerm = std::max(projection.corridorTerm, siblingPeak);
    } else {
      uint64_t regionPriced =
          regionPeakThroughOp(lastUse, analysis, queryCache, regionPeakCache) +
          dstBytes;
      projection.corridorTerm = std::max(projection.corridorTerm, regionPriced);
      projection.exact = false;
    }
  }

  return projection;
}

/// What the whole-function peak gate says about one candidate.
enum class PeakVerdict {
  Accept,
  /// The projection is exact and really does cross the ceiling.
  RejectExact,
  /// The projection was charged conservatively; the real peak may well fit.
  RejectFallback,
};

/// Owns the whole-function pressure analysis the second veto is measured
/// against, plus the bookkeeping that keeps it honest while hoists mutate the
/// IR.
///
/// The analysis is rebuilt after each accepted hoist rather than corrected by
/// an accumulated term. Accumulating `dstBytes` is self-defeating: over budget
/// the ceiling is exactly `prePeak`, so the first accepted hoist would
/// guarantee the rejection of every later candidate. (`ReduceVariableLiveness`
/// rebuilds liveness after moving operations for the same reason.)
class FunctionPeakGate {
public:
  FunctionPeakGate(FunctionOpInterface func, uint64_t threshold,
                   unsigned corridorOpCap, unsigned rebuildCap)
      : func(func), threshold(threshold), corridorOpCap(corridorOpCap),
        rebuildCap(rebuildCap) {}

  /// Decides whether hoisting \p cvtOp out of \p forOp may raise the function's
  /// reported peak pressure past `max(prePeak, threshold)`. On `Accept` the
  /// caller must perform the hoist and then call `noteHoisted`.
  PeakVerdict decide(ttg::ConvertLayoutOp cvtOp, scf::ForOp forOp) {
    if (!refreshAnalysis()) {
      LDBG("Skipping hoist: whole-function peak analysis rebuild cap ("
           << rebuildCap << ") reached; a stale analysis bounds nothing");
      return PeakVerdict::RejectFallback;
    }

    lastProjection = projectedFunctionPeak(
        cvtOp, forOp, *analysis, prePeak, sourceUsers(cvtOp.getSrc()),
        corridorOpCap, *queryCache, regionPeakCache);
    uint64_t projected = lastProjection.total();
    // No ratchet. Over budget the ceiling *is* the current peak, so no sequence
    // of hoists can raise it; under budget, only as far as the threshold.
    uint64_t ceiling = std::max(prePeak, threshold);

    if (projected > ceiling) {
      LDBG("Skipping hoist: projected function peak "
           << projected << " B/lane (max of prePeak=" << lastProjection.prePeak
           << " corridor=" << lastProjection.corridorTerm << " over "
           << lastProjection.corridorLength
           << " ops, newPoint=" << lastProjection.newPointTerm
           << ") exceeds ceiling " << ceiling << " B/lane"
           << (lastProjection.exact ? "" : " (conservatively charged)"));
      return lastProjection.exact ? PeakVerdict::RejectExact
                                  : PeakVerdict::RejectFallback;
    }

    LDBG("Function peak allows hoist: projected "
         << projected << " B/lane (max of prePeak=" << lastProjection.prePeak
         << " corridor=" << lastProjection.corridorTerm << " over "
         << lastProjection.corridorLength
         << " ops, newPoint=" << lastProjection.newPointTerm
         << ") within ceiling " << ceiling << " B/lane");
    return PeakVerdict::Accept;
  }

  /// Readies the gate for a *trial*: a change the caller is about to apply to
  /// the IR and have measured by `endTrial`, rather than projected by `decide`.
  /// Used for a joint hoist of several conversions, whose combined effect the
  /// single-candidate projection cannot express. Brings the analysis up to date
  /// so the ceiling is the peak of the IR the trial starts from. Returns false,
  /// and the caller must not apply the trial, once the rebuild cap leaves no
  /// room for the trial's own measurement.
  bool beginTrial() {
    if (!refreshAnalysis() || rebuilds >= rebuildCap) {
      LDBG("Skipping joint hoist: whole-function peak analysis rebuild cap ("
           << rebuildCap << ") reached");
      return false;
    }
    trialCeiling = std::max(prePeak, threshold);
    return true;
  }

  /// Measures the peak of the IR the caller has changed since `beginTrial`.
  /// The measurement is exact, not a projection, so a refusal is always
  /// `RejectExact`. On a refusal the caller must restore the IR exactly: the
  /// gate's own analysis still describes the pre-trial IR and is kept. On
  /// `Accept` the measured analysis becomes the current one, so an accepted
  /// trial spends one rebuild.
  PeakVerdict endTrial() {
    ++rebuilds;
    ttg::intel::RegisterPressureAnalysis trial(func);
    uint64_t trialPeak = trial.peakPressure(func);
    if (trialPeak > trialCeiling) {
      LDBG("Skipping joint hoist: measured function peak "
           << trialPeak << " B/lane exceeds ceiling " << trialCeiling
           << " B/lane");
      return PeakVerdict::RejectExact;
    }
    LDBG("Function peak allows joint hoist: measured "
         << trialPeak << " B/lane within ceiling " << trialCeiling
         << " B/lane");
    analysis.emplace(std::move(trial));
    prePeak = trialPeak;
    dirty = false;
    resetPerGenerationCaches();
    return PeakVerdict::Accept;
  }

  /// Records that the hoist `decide` just accepted has been performed, so the
  /// analysis now describes stale IR.
  void noteHoisted() {
    dirty = true;
    pendingProjection = lastProjection.total();
  }

  /// Re-measures the peak once after the function's last candidate, so the
  /// upper-bound invariant is checked even for a hoist no later candidate
  /// forces a rebuild for. Assertions-only, and outside the rebuild cap: it
  /// cannot change a verdict, and counting it would retire the invariant on
  /// exactly the functions that hit the cap.
  void flushInvariantCheck() {
#ifndef NDEBUG
    if (!dirty)
      return;
    ttg::intel::RegisterPressureAnalysis fresh(func);
    uint64_t freshPeak = fresh.peakPressure(func);
    assert(freshPeak <= pendingProjection &&
           "hoist raised the reported function peak above its projection");
    (void)freshPeak;
    dirty = false;
#endif
  }

private:
  /// Brings `analysis`/`prePeak` up to date, and checks the upper-bound
  /// invariant at the only point where both the freshly measured peak and the
  /// projection that predicted it exist. (A rejected candidate is never
  /// applied, so it cannot be checked.)
  ///
  /// Returns false once the rebuild cap is spent. Reusing the last analysis
  /// would be unsound, not merely loose: after a hoist it describes pre-move
  /// IR, so its `prePeak` is neither a bound on the current peak nor the
  /// ceiling to meet.
  bool refreshAnalysis() {
    if (!analysis) {
      // The first build is not a rebuild and does not spend the cap.
      analysis.emplace(func);
      resetPerGenerationCaches();
      prePeak = analysis->peakPressure(func, *queryCache);
      return true;
    }
    if (!dirty)
      return true;
    if (rebuilds >= rebuildCap)
      return false;
    ++rebuilds;
    analysis.emplace(func);
    resetPerGenerationCaches();
    prePeak = analysis->peakPressure(func, *queryCache);
    assert(prePeak <= pendingProjection &&
           "hoist raised the reported function peak above its projection");
    dirty = false;
    return true;
  }

  /// Both caches describe the IR `analysis` was built on. Reset them whenever
  /// `analysis` changes.
  void resetPerGenerationCaches() {
    queryCache.emplace();
    regionPeakCache.clear();
  }

  /// Returns \p src's users, deduplicated, so sibling candidates sharing one
  /// source share one scan of it. No generation key is needed: a hoist only
  /// *moves* a conversion, so this list is invariant across rebuilds. The
  /// position-dependent part is recomputed per candidate in `classifySource`.
  ArrayRef<Operation *> sourceUsers(Value src) {
    auto [it, inserted] = userCache.try_emplace(src);
    if (!inserted)
      return it->second;
    SmallPtrSet<Operation *, 8> seen;
    for (Operation *user : src.getUsers())
      if (seen.insert(user).second)
        it->second.push_back(user);
    return it->second;
  }

  FunctionOpInterface func;
  uint64_t threshold;
  unsigned corridorOpCap;
  unsigned rebuildCap;

  std::optional<ttg::intel::RegisterPressureAnalysis> analysis;
  uint64_t prePeak = 0;
  PeakProjection lastProjection;
  uint64_t pendingProjection = 0;
  uint64_t trialCeiling = 0;
  bool dirty = false;
  unsigned rebuilds = 0;
  DenseMap<Value, SmallVector<Operation *>> userCache;
  // Scoped to one `analysis` generation; see `resetPerGenerationCaches`.
  std::optional<ttg::intel::RegisterPressureAnalysis::QueryCache> queryCache;
  DenseMap<Operation *, uint64_t> regionPeakCache;
};

/// Decision state for one function, shared between the one-at-a-time phase
/// (`hoistCvtDotOpsOutOfLoop`) and the shared-source phase
/// (`reconsiderSharedSources`), and settled by `finalizeRefusals`.
///
/// A refusal is only *recorded* when made; `tt.no_licm` and the rejection
/// statistics are applied once, when the function's last phase has run. That
/// keeps every candidate counted exactly once however many times it is
/// weighed: as hoisted if any phase takes it, else under the reason the
/// one-at-a-time phase refused it for.
struct FunctionHoistState {
  /// Per loop: the accumulated effect of the hoists already taken on the
  /// loop-level gate's projected live-in, in per-lane bytes (may be negative).
  /// Keyed by loop so the shared-source phase continues each loop's figure
  /// where the one-at-a-time phase left it.
  DenseMap<Operation *, int> netBytes;
  /// In the order the refusals were made.
  SmallVector<Refusal> refusals;
  /// Sources at least one of whose conversions has been hoisted.
  llvm::SmallPtrSet<Value, 8> hoistedSources;
  /// Every conversion the one-at-a-time phase hoisted, with the loop it left,
  /// so a refusal recorded after it can still be matched against it (see
  /// `chargeMergeIntoPriorHoist`). `hoistedSources` is too coarse for that:
  /// a merge needs the same result type too, and a block-argument source's
  /// dominance depends on which loop the conversion left.
  SmallVector<std::pair<ttg::ConvertLayoutOp, scf::ForOp>> hoisted;
};

/// Returns the threshold both gates use: 80% of the per-lane GRF budget
/// \p grfBudget. The 20% headroom accounts for scalars, temporaries, and
/// loop-internal values not tracked by live-in liveness. Integer arithmetic
/// (4/5) avoids float-to-unsigned truncation.
static int liveInThreshold(unsigned grfBudget) {
  return static_cast<int>(grfBudget * 4 / 5);
}

/// Charges the loop of each twin in \p twinIdxs for its merge into the hoist
/// out of \p forOp (see Twin merges): the twin's `hoistDeltaBytes`, once per
/// loop, since several twins in one loop merge into one result. Twins already
/// `mergeCharged` are skipped. Without this charge, a merge whose result is
/// larger than the source leaves the loop's figure below its real live-in,
/// and a later shared-source group passes on that figure.
static void
chargeMergedTwins(ArrayRef<unsigned> twinIdxs, scf::ForOp forOp,
                  const ttg::intel::RegisterPressureAnalysis &analysis,
                  FunctionHoistState &state) {
  llvm::SmallPtrSet<Operation *, 4> chargedLoops;
  for (unsigned idx : twinIdxs) {
    Refusal &twin = state.refusals[idx];
    if (twin.mergeCharged)
      continue;
    twin.mergeCharged = true;
    Operation *twinLoop = twin.forOp.getOperation();
    if (twinLoop == forOp.getOperation() ||
        !chargedLoops.insert(twinLoop).second)
      continue;
    // A twin's loop has been, or is being, decided, so its entry exists.
    // Looked up rather than indexed: the caller holds a reference into
    // `netBytes` that an insertion's rehash would invalidate.
    auto it = state.netBytes.find(twinLoop);
    assert(it != state.netBytes.end() && "twin's loop never decided");
    if (it == state.netBytes.end())
      continue;
    it->second +=
        hoistDeltaBytes(twin.cvtOp, twin.forOp, analysis, state.refusals);
  }
}

/// Charges the loop of the refusal just recorded if it merges into a
/// conversion hoisted earlier, which `collectMergingTwins` could not see at
/// that time. Only one-at-a-time hoists are matched: the shared-source phase
/// records no refusals.
static void
chargeMergeIntoPriorHoist(const ttg::intel::RegisterPressureAnalysis &analysis,
                          FunctionHoistState &state) {
  unsigned idx = state.refusals.size() - 1;
  ttg::ConvertLayoutOp refusedCvt = state.refusals[idx].cvtOp;
  for (auto [hoistedCvt, hoistedForOp] : state.hoisted) {
    if (hoistedCvt.getSrc() != refusedCvt.getSrc() ||
        hoistedCvt.getType() != refusedCvt.getType() ||
        !mergesIntoHoist(hoistedCvt, refusedCvt.getOperation(), hoistedForOp))
      continue;
    chargeMergedTwins(ArrayRef<unsigned>(idx), hoistedForOp, analysis, state);
    // For a hoist out of the refusal's own loop, `chargeMergedTwins` only
    // flags the refusal, so there is no charge to report.
    Operation *refusedLoop = state.refusals[idx].forOp.getOperation();
    if (refusedLoop != hoistedForOp.getOperation())
      LDBG("Refused conversion merges into an earlier-decided loop's hoist: "
           "its loop is now at alreadyHoisted="
           << state.netBytes.lookup(refusedLoop) << " B/lane");
    return;
  }
}

/// Decide, for every hoisting candidate of a single \p forOp, whether to hoist
/// it out of the loop or to reject it on register pressure grounds.
///
/// The candidates are *not* processed in program order. A hoist can lower the
/// projected pressure as well as raise it, so the running total is not
/// monotonic and a rejection is only safely conservative once every
/// pressure-reducing candidate has been credited. A rejection this phase makes
/// is final unless `reconsiderSharedSources` overturns it, since it ends up
/// stamping `tt.no_licm` and the later generic LICM pass never revisits the
/// conversion. Deciding the smallest (most negative) projected delta first
/// makes the outcome depend on measured costs rather than syntactic order. The
/// delta is re-measured per decision rather than sorted up front, because a
/// candidate sharing its source with a sibling only retires that source once
/// the sibling has left.
///
/// A candidate clearing the loop-level gate faces a second, independent veto on
/// the whole *function's* peak. Hoisting relocates the point where source and
/// result are simultaneously live from inside the loop into the straight-line
/// code above it, which the live-in figure cannot see -- issue #7993's gap (b).
/// The two vetoes stay separate: they bound different quantities, and the
/// loop-level one keeps its frozen analysis and `netBytes` accounting
/// unchanged.
///
/// \p candidates are the candidates directly inside \p forOp's body, in program
/// order; \p grfBudget is per-lane bytes (see
/// `RegisterPressureAnalysis::getPerLaneGRFBudgetInBytes`). Refusals are
/// recorded in \p state, not applied; see `FunctionHoistState`.
static void hoistCvtDotOpsOutOfLoop(
    scf::ForOp forOp, ArrayRef<ttg::ConvertLayoutOp> candidates,
    const ttg::intel::RegisterPressureAnalysis &analysis, unsigned grfBudget,
    FunctionPeakGate &peakGate, FunctionHoistState &state) {
  // `liveInBytes` comes from an analysis built once at pass entry, so it does
  // not reflect hoists this pass has already performed. `netBytes` carries
  // their accumulated effect (which may be negative) forward instead.
  unsigned liveInBytes = analysis.liveInPressure(forOp.getBody());
  int &netBytes = state.netBytes[forOp.getOperation()];

  // Only hoist if the projected live-in pressure stays within 80% of the GRF
  // budget.
  int threshold = liveInThreshold(grfBudget);

  // Whether the merge a twin credit relies on keeps the twin's loop within
  // this loop-level gate, at that loop's current figure plus the merge's
  // delta. The peak gate is not rerun for the twin's loop. The twin's delta is
  // asked without this predicate, so the check cannot recurse.
  auto mergeFits = [&](const Refusal &twin) {
    scf::ForOp twinForOp = twin.forOp;
    Operation *twinLoop = twinForOp.getOperation();
    auto it = state.netBytes.find(twinLoop);
    if (it == state.netBytes.end())
      return false;
    int delta =
        hoistDeltaBytes(twin.cvtOp, twinForOp, analysis, state.refusals);
    int projected =
        static_cast<int>(analysis.liveInPressure(twinForOp.getBody())) +
        it->second + delta;
    if (delta <= 0 || projected < threshold)
      return true;
    LDBG("No twin credit: merging would leave its loop at liveIn="
         << analysis.liveInPressure(twinForOp.getBody()) << " + alreadyHoisted="
         << it->second << " + thisHoist=" << delta << " = " << projected
         << " B/lane, past 80% of budget=" << grfBudget << " B/lane");
    return false;
  };

  SmallVector<ttg::ConvertLayoutOp> pending(candidates);
  while (!pending.empty()) {
    // Pick the cheapest remaining candidate, breaking ties in program order.
    // Ties are outcome-neutral: equal deltas project the same values whichever
    // of the two is decided first.
    unsigned bestIdx = 0;
    int bestDelta =
        hoistDeltaBytes(pending[0], forOp, analysis, state.refusals, mergeFits);
    for (unsigned i = 1, e = pending.size(); i != e; ++i) {
      int delta = hoistDeltaBytes(pending[i], forOp, analysis, state.refusals,
                                  mergeFits);
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
    // exactly as far over (or under) budget as it was, so rejecting it buys
    // back no register and only forgoes the loop-invariant conversion. Gating
    // such a hoist on a figure it does not move made this pass reject six of
    // _attn_bwd's eight loop-invariant dot-operand conversions at HEAD_DIM=128,
    // where all eight are measurably free (see the pass description),
    // costing 21.7% geomean on flash-attn-bwd.
    if (bestDelta > 0 && projectedBytes >= threshold) {
      LDBG("Skipping hoist: liveIn="
           << liveInBytes << " + alreadyHoisted=" << netBytes
           << " + thisHoist=" << bestDelta << " = " << projectedBytes
           << " B/lane exceeds 80% of budget=" << grfBudget << " B/lane");
      state.refusals.push_back({cvtOp, forOp, RefusalReason::Pressure});
      chargeMergeIntoPriorHoist(analysis, state);
      continue;
    }

    // Second veto: the whole-function peak, which the live-in figure above does
    // not measure.
    PeakVerdict verdict = peakGate.decide(cvtOp, forOp);
    if (verdict != PeakVerdict::Accept) {
      state.refusals.push_back({cvtOp, forOp,
                                verdict == PeakVerdict::RejectExact
                                    ? RefusalReason::FunctionPeakExact
                                    : RefusalReason::FunctionPeakFallback});
      chargeMergeIntoPriorHoist(analysis, state);
      continue;
    }

    LDBG("Hoisting convert_layout out of loop: liveIn="
         << liveInBytes << " + alreadyHoisted=" << netBytes
         << " + thisHoist=" << bestDelta << " = " << projectedBytes
         << " B/lane budget=" << grfBudget << " B/lane");
    // Every refusal that merges into this hoist, not only those its credit
    // relied on. `mergesIntoHoist` does not depend on `cvtOp`'s position, so
    // this can run before the move.
    SmallVector<unsigned> mergingTwins;
    collectMergingTwins(cvtOp, forOp, state.refusals, mergingTwins);
    // Hoist the conversion out of the loop, to the program point the projection
    // above was measured at.
    moveToHoistAnchor(cvtOp, forOp);

    ++NumHoisted;
    netBytes += bestDelta;
    state.hoistedSources.insert(cvtOp.getSrc());
    state.hoisted.push_back({cvtOp, forOp});
    peakGate.noteHoisted();
    chargeMergedTwins(mergingTwins, forOp, analysis, state);
  }
}

/// Second phase: retry, as one move, the refused conversions of each source
/// shared by two or more refusals, or by a refusal and a hoist. Priced one at
/// a time, each conversion sees the others keep the source live, so none earns
/// the credit. Moved together, the source dies right after them.
///
/// A group is applied and measured: it must pass the loop-level gate once per
/// loop it leaves and the peak gate on the moved IR, or it is restored
/// exactly. Subsets are not tried. Groups are tried once each, in
/// first-refusal order, and each trial spends one unit of `peak-rebuild-cap`.
/// A group of only `mergeCharged` refusals is tried too: its move changes the
/// IR later trials are measured on.
static void
reconsiderSharedSources(FunctionHoistState &state,
                        const ttg::intel::RegisterPressureAnalysis &analysis,
                        unsigned grfBudget, FunctionPeakGate &peakGate) {
  llvm::MapVector<Value, SmallVector<Refusal *>> bySource;
  for (Refusal &refusal : state.refusals)
    bySource[refusal.cvtOp.getSrc()].push_back(&refusal);

  int threshold = liveInThreshold(grfBudget);
  for (auto &[src, group] : bySource) {
    if (group.size() < 2 && !state.hoistedSources.contains(src))
      continue;
    if (!peakGate.beginTrial())
      return;

    // Apply the group, remembering where each conversion came from. The body
    // terminator follows every candidate, so a next node always exists.
    SmallVector<std::pair<ttg::ConvertLayoutOp, Operation *>> undo;
    for (Refusal *refusal : group) {
      undo.push_back({refusal->cvtOp, refusal->cvtOp->getNextNode()});
      moveToHoistAnchor(refusal->cvtOp, refusal->forOp);
    }
    auto restore = [&]() {
      // In reverse, so a conversion whose old successor was another member of
      // the group finds that member back in place first.
      for (auto &[cvtOp, next] : llvm::reverse(undo))
        cvtOp->moveBefore(next);
    };

    // The loop-level gate, per loop the group leaves, on the moved IR. With the
    // whole group moved, `hoistRetiresSource` gives the same answer for every
    // member of one loop, so the first one stands for the loop and the source
    // is credited once per loop.
    llvm::MapVector<Operation *, int> deltas;
    llvm::MapVector<Operation *, Refusal *> firstInLoop;
    for (Refusal *refusal : group) {
      // Already charged by the hoist it merges into. Other members in the
      // same loop are still priced.
      if (refusal->mergeCharged)
        continue;
      Operation *loop = refusal->forOp.getOperation();
      deltas[loop] += static_cast<int>(
          ttg::intel::RegisterPressureAnalysis::getPerThreadSizeInBytes(
              refusal->cvtOp.getType()));
      firstInLoop.insert({loop, refusal});
    }
    bool fits = true;
    for (auto &[loop, refusal] : firstInLoop) {
      scf::ForOp forOp = refusal->forOp;
      int &delta = deltas[loop];
      if (hoistRetiresSource(refusal->cvtOp, forOp, state.refusals))
        delta -=
            static_cast<int>(analysis.liveInContribution(forOp.getBody(), src));
      int projectedBytes =
          static_cast<int>(analysis.liveInPressure(forOp.getBody())) +
          state.netBytes[forOp.getOperation()] + delta;
      assert(projectedBytes >= 0 && "over-credited a joint hoist's source");
      if (delta > 0 && projectedBytes >= threshold) {
        LDBG("Skipping joint hoist of "
             << group.size() << " conversion(s): loop liveIn="
             << analysis.liveInPressure(forOp.getBody())
             << " + alreadyHoisted=" << state.netBytes[forOp.getOperation()]
             << " + thisHoist=" << delta << " = " << projectedBytes
             << " B/lane exceeds 80% of budget=" << grfBudget << " B/lane");
        fits = false;
        break;
      }
    }
    if (!fits) {
      restore();
      continue;
    }

    if (peakGate.endTrial() != PeakVerdict::Accept) {
      restore();
      continue;
    }

    LDBG("Hoisting jointly a group of " << group.size()
                                        << " conversion(s) sharing a source");
    for (Refusal *refusal : group) {
      refusal->overturned = true;
      ++NumHoisted;
    }
    for (auto &[loop, delta] : deltas)
      state.netBytes[loop] += delta;
    state.hoistedSources.insert(src);
  }
}

/// Applies every refusal no phase overturned: stamps `tt.no_licm`, so the later
/// generic LICM pass does not hoist the conversion anyway, and counts it under
/// the reason it was first refused for.
static void finalizeRefusals(FunctionHoistState &state) {
  for (Refusal &refusal : state.refusals) {
    if (refusal.overturned)
      continue;
    switch (refusal.reason) {
    case RefusalReason::Pressure:
      ++NumRejectedPressure;
      break;
    case RefusalReason::FunctionPeakExact:
      ++NumRejectedFunctionPeakExact;
      break;
    case RefusalReason::FunctionPeakFallback:
      ++NumRejectedFunctionPeakFallback;
      break;
    }
    refusal.cvtOp->setAttr("tt.no_licm",
                           UnitAttr::get(refusal.cvtOp.getContext()));
  }
  state = FunctionHoistState();
}

class TritonIntelGPUHoistLayoutConversionsPass
    : public ttg::intel::impl::TritonIntelGPUHoistLayoutConversionsBase<
          TritonIntelGPUHoistLayoutConversionsPass> {

  using ttg::intel::impl::TritonIntelGPUHoistLayoutConversionsBase<
      TritonIntelGPUHoistLayoutConversionsPass>::
      TritonIntelGPUHoistLayoutConversionsBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    // Hoisting is gated on the budget as a *ceiling* on what may be added,
    // so an unknown ("default"/"auto") GRF size must assume the smallest the
    // device supports -- see UnknownGRFSizeAssumption's documentation.
    unsigned grfBudget =
        ttg::intel::RegisterPressureAnalysis::getPerLaneGRFBudgetInBytes(
            grfMode, mod,
            ttg::intel::RegisterPressureAnalysis::UnknownGRFSizeAssumption::
                Smallest);
    ttg::intel::RegisterPressureAnalysis analysis(mod);

    // Group the candidates by the loop they would leave, so each loop's budget
    // is spent on its candidates as a set rather than in whatever order the
    // walk reaches them. Insertion order (program order) is preserved to keep
    // the tie-break in `hoistCvtDotOpsOutOfLoop` deterministic. Inner loops are
    // visited before their enclosing loop's later candidates, but the two never
    // share a body block, so their budgets are independent.
    llvm::MapVector<scf::ForOp, SmallVector<ttg::ConvertLayoutOp>> candidates;
    mod.walk([&](ttg::ConvertLayoutOp cvtOp) {
      if (scf::ForOp forOp = getHoistCandidateLoop(cvtOp))
        candidates[forOp].push_back(cvtOp);
    });

    // The whole-function peak gate owns one analysis per function, rebuilt as
    // hoists invalidate it. `mod.walk` groups a function's candidates together
    // and reversing preserves that grouping, so a single slot suffices.
    // Reaching one function twice would only rebuild its gate, not borrow
    // another function's peak: the slot is re-emplaced whenever the function
    // changes.
    uint64_t threshold = liveInThreshold(grfBudget);
    std::optional<FunctionPeakGate> peakGate;
    FunctionOpInterface gatedFunc;
    // Scoped to `gatedFunc` like `peakGate`: a function's refusals are settled
    // (jointly reconsidered, then stamped and counted) before the next
    // function's loops are decided.
    FunctionHoistState state;
    auto settleFunction = [&]() {
      if (!peakGate)
        return;
      reconsiderSharedSources(state, analysis, grfBudget, *peakGate);
      finalizeRefusals(state);
      peakGate->flushInvariantCheck();
    };

    // Decide the loops last to first. A hoist only moves a conversion *earlier*
    // in its block, so once every loop after `forOp` has been decided, no later
    // hoist can add or remove a use of a source below `forOp` and
    // `hoistRetiresSource`'s after-loop question has a final answer. Deciding
    // first-to-last instead denied the credit to the *earlier* of two loops
    // sharing a source, purely because the later loop's conversion had not
    // moved out from under it yet -- the same order-dependence
    // `hoistCvtDotOpsOutOfLoop` removes within one loop.
    for (auto &[forOp, loopCandidates] : llvm::reverse(candidates)) {
      auto func = forOp->getParentOfType<FunctionOpInterface>();
      // A null parent would compare equal to the default-constructed
      // `gatedFunc` and leave `peakGate` disengaged, so refuse rather than
      // dereference it.
      assert(func && "candidate loop outside a function");
      if (!func)
        continue;
      if (func != gatedFunc) {
        settleFunction();
        gatedFunc = func;
        peakGate.emplace(func, threshold, corridorOpCap, peakRebuildCap);
      }
      hoistCvtDotOpsOutOfLoop(forOp, loopCandidates, analysis, grfBudget,
                              *peakGate, state);
    }
    settleFunction();

    if (mlir::triton::tools::getBoolEnv("TRITON_INTEL_HLC_STATS")) {
      llvm::errs() << "[HoistLayoutConversions] considered=" << NumConsidered
                   << " hoisted=" << NumHoisted
                   << " rejected_pressure=" << NumRejectedPressure
                   << " rejected_function_peak_exact="
                   << NumRejectedFunctionPeakExact
                   << " rejected_function_peak_fallback="
                   << NumRejectedFunctionPeakFallback
                   << " skipped_other=" << NumSkippedOther << "\n";
    }
  }
};

} // namespace
