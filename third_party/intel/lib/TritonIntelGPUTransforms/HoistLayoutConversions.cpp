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

/// Why a candidate was refused, recorded rather than acted on at once so the
/// shared-source phase (`reconsiderSharedSources`) can still overturn it; see
/// `FunctionHoistState`.
///
/// Declared here (rather than where it is first used, in
/// `reconsiderSharedSources`) so that `hoistRetiresSource` below can see
/// `Refusal`'s definition and consult a function's refusals so far when
/// computing credit; see its doc comment.
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
  /// Set once an accepted hoist structurally merges this refusal away (see
  /// `collectMergingTwins`, or `chargeMergeIntoPriorHoist` when the hoist
  /// came first) and this refusal's own loop was charged, in
  /// `netBytes`, for that merge (see `chargeMergedTwins`) -- whether or not
  /// the hoist's own delta actually relied on excusing this refusal as an
  /// unenforceable twin (see `hoistRetiresSource`). From then on the loop's
  /// figure already reflects the merge, so `reconsiderSharedSources` must not
  /// price this refusal's move again. It may still excuse a later hoist's
  /// credit: the merge retires the source for every hoisting loop.
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

/// Returns true if `remove_layout_conversions` will replace \p twin -- a
/// conversion of the same source to the same type as \p cvtOp -- with \p
/// cvtOp's result once \p cvtOp is hoisted out of \p forOp. That pass's
/// backward rematerialization reuses an earlier conversion of the same (value,
/// encoding) pair only if it properly dominates the later one, and never for a
/// dot operand whose parent is a blocked layout: it returns for those before
/// consulting its remat cache at all (an FMA dot's operands).
///
/// Dominance follows from where `hoistAnchor` puts \p cvtOp. Below a defining
/// operation it dominates every user of the source, \p twin included. A block
/// argument has none, so \p cvtOp lands just above \p forOp instead and
/// dominates only what nests inside \p forOp or follows it in its block: a
/// twin in a loop after an `scf.if` holding \p forOp, say, is not replaced.
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

/// Appends to \p twinIdxs the index, within \p refusals, of every
/// non-overturned refusal that `remove_layout_conversions` will replace with
/// \p cvtOp's result once it is hoisted out of \p forOp: same source, same
/// result type, `mergesIntoHoist` holding.
///
/// Unlike `hoistRetiresSource`'s own twin match, this does not consult
/// \p cvtOp's credit decision at all -- not `mergeFits`, and not whether
/// `hoistRetiresSource` judged \p cvtOp's own hoist to retire the source.
/// Those two questions are about whether *this* hoist's delta may treat the
/// twin as absent; this one is about which twins the hoist makes
/// `remove_layout_conversions` merge away regardless, which is unconditional
/// on the structural match alone. A twin `mergeFits` refuses, or one left
/// out because `hoistRetiresSource` returned false for an unrelated reader,
/// still merges once \p cvtOp is actually hoisted -- see `chargeMergedTwins`,
/// which this collects for.
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
/// \p priorRefusals additionally excuses a user that is itself a *refused*
/// hoist candidate for the exact same (source, result type) pair as \p cvtOp,
/// provided `mergesIntoHoist` holds for it. Such a twin, though still
/// physically sitting in its own loop right now, does not survive the
/// pipeline once \p cvtOp is hoisted: `remove_layout_conversions`'s
/// backward-rematerialization cache (`getRematValue` + a dominance check)
/// replaces it with the hoisted \p cvtOp. Counting the twin as a real reader
/// double-charges this hoist for a read that will not remain, which is what
/// denies a sibling loop's hoist its own credit in a shared-source cascade --
/// confirmed empirically on `_attn_bwd` via the compiled TTGIR, where exactly
/// this merge is what keeps the twin from being duplicated. A twin
/// `mergesIntoHoist` rejects
/// stays in its loop and keeps reading the source there, so it is a real
/// reader like any other.
///
/// The merge also changes the twin's *own* loop, which this credit says
/// nothing about: that loop stops reading the source and reads \p cvtOp's
/// hoisted result instead, exactly as if the twin itself had been hoisted.
/// When \p mergeFits is set, a twin in a loop other than \p forOp that is not
/// yet `mergeCharged` is excused only if \p mergeFits accepts that change for
/// the twin's loop; a twin it refuses is a real reader too, so the credit is
/// never what pushes another loop past a gate the pass enforces (see
/// `hoistCvtDotOpsOutOfLoop`). This governs only whether *this* hoist's own
/// delta may treat the twin as absent, not whether the twin's own loop gets
/// charged for the merge -- that is `collectMergingTwins`' job, asked
/// unconditionally of every accepted hoist regardless of what this function
/// returns, and `chargeMergeIntoPriorHoist`'s for a twin refused only after
/// the hoist (see `chargeMergedTwins`).
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
/// \p priorRefusals and \p mergeFits are forwarded to `hoistRetiresSource` so
/// a sibling loop's already-refused, equivalent conversion does not withhold
/// this credit; see that function's doc comment.
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
      // An enclosing loop's back edge re-reads the source after the corridor,
      // so physically the credit must go. This is not a blind spot in what
      // `RegisterPressureAnalysis` itself reports -- it is back-edge aware
      // throughout -- it is a deliberate choice not to reason about the
      // enclosing loop's own iteration structure at all, so the caller prices
      // as if the departing source were still fully live. That can leave the
      // projection above the very peak it gates (measured: case 23 projects
      // 1376 against a post-hoist 1248).
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

/// Returns the last operation in \p body, in block order, that uses \p
/// cvtOp's result -- or null if it has none there. A use inside a nested
/// region is mapped onto the top-level operation in \p body that contains it,
/// same as `classifySource` does for the conversion's source.
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

/// Returns \p cvtOp's real last use, in its own block's own program order --
/// unlike `lastBodyUser`, this does *not* map a nested use up to any enclosing
/// top-level op. Returns null if \p cvtOp's uses don't all share one
/// immediate containing block (spread across sibling branches of an
/// `scf.if`, say, or across different nesting depths): comparing positions
/// across different blocks has no defined order, so the caller must fall
/// back to a fully conservative charge in that case. Also returns null for a
/// userless conversion, which cannot happen for a hoist candidate but is
/// handled the same conservative way for safety.
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

/// Collects into \p corridor the operations at which a hoisted \p cvtOp's
/// result becomes newly live: the tail of \p anchor's block down through the
/// loop, the body operations \p cvtOp is being lifted over, and the body
/// operations strictly after \p cvtOp's own last use in the body. \p cvtOp
/// itself is excluded -- it is erased by the hoist and priced separately as
/// the new program point -- and so is every body operation from \p cvtOp up
/// to and including that last use: the un-hoisted result is already locally
/// live there, at the same cost hoisting it would add, so pricing them again
/// would double count.
///
/// An operation strictly after that last use looks, in the body's own
/// unhoisted liveness, like it runs once the result is already dead; hoisting
/// changes that: the loop's back edge makes the result live through the
/// *whole* loop once it no longer lives inside the body (see
/// `RegisterPressureAnalysis::getLiveThroughAncestorSet`'s loop branch), so
/// such an operation sees a genuinely new charge.
///
/// Does **not** recurse into nested regions: a sibling loop the corridor
/// steps over is walked as the single operation that loop op is, not op by
/// op. `pressureAt`/`pressureBefore` are region-aware (they include whatever
/// lives through that loop, per `RegisterPressureAnalysis`'s own doc), so the
/// region-holding op's own point already reflects anything live through it --
/// but not the (possibly higher) peak at some op strictly inside it. The
/// caller (`projectedFunctionPeak`) separately checks that peak for any
/// corridor entry holding a region, so this not recursing is only about how
/// the walk is *structured*, not a gap in what gets priced.
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

  // Body operations strictly after `cvtOp`'s own last use -- see the doc
  // comment above for why these need pricing and the ones in between do not.
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

/// Returns the peak per-thread pressure `analysis` reports over every block
/// nested in \p op's own regions, or 0 if \p op holds none. Used to price a
/// corridor entry that is itself a region-holding op (a sibling loop the
/// corridor steps over without descending into, per `collectCorridor`'s doc
/// comment): `pressureAt(op, ...)` only reports what is live at \p op's own
/// program point, which is silent about a higher peak reached *inside* that
/// region -- exactly the gap `RegisterPressureAnalysis`'s own class doc warns
/// `pressureAt` does not cover (only `peakPressure` descends into regions).
/// Any value live across \p op -- defined before it, still needed after it
/// closes -- is already included in this figure by the ancestor
/// live-through rule, for a loop and a single-execution region (an
/// `scf.if`) alike: nothing here is excluded just because \p op itself has
/// no relevant use of it. The caller's own `+ dstBytes` on top of this
/// figure models only the newly arriving result, not anything already
/// counted here; that is exactly what the sibling-region term in
/// `projectedFunctionPeak` and the per-op corridor loop above it both rely
/// on being true (nothing else could be contributing a departing source's
/// bytes inside a region it does not itself read).
///
/// Memoized in \p regionPeakCache, keyed by \p op: many candidates in the same
/// loop (or in sibling loops) can step over the same region-holding op, and
/// without this, each one re-walks it -- and, transitively through
/// `peakPressure`, every op and block nested inside it -- from scratch, an
/// O(candidates x region size) cost. \p regionPeakCache and \p queryCache
/// must both be reset whenever the analysis they describe is rebuilt (see
/// `FunctionPeakGate`); within one such generation the IR they describe never
/// changes, so \p op unambiguously identifies the same answer for as long as
/// the entry survives.
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

/// Prices every operation strictly after \p start, in \p start's own block,
/// the same way `projectedFunctionPeak`'s own corridor loop prices a body
/// operation strictly after `cvtOp`'s last use: \p src is credited via the
/// exact same rule used everywhere else in this file, and any op that itself
/// holds a region is additionally checked via `regionPeakThroughOp`. Used to
/// price the tail of a nested branch after \p cvtOp's own *real* (not
/// top-level-mapped) last use; see `projectedFunctionPeak`'s call site for
/// why that tail needs its own pass rather than folding into
/// `regionPeakThroughOp`'s whole-region charge.
///
/// \p cvtResult's own weight, \p dstBytes, is charged at an op only when it is
/// not already live there in the current, pre-hoist IR. It usually is not --
/// that is the whole reason this tail needs pricing at all, the un-hoisted
/// result is dead past its own last use inside a single-execution region
/// (see `collectCorridor`'s doc comment). But if \p start's block nests
/// inside a *loop* (an inner `scf.for` between \p start and `forOp`, say),
/// that loop's own back edge already keeps \p cvtResult live for its whole
/// duration -- the same rule that makes the hoisted result live through
/// `forOp` itself -- so charging it again here would double it.
///
/// Every per-op charge here is either a result charge decided by a fully
/// determined fact (already live or not) or a credit decided the same way
/// (nothing to credit, a real remaining reader, or `srcLiveness.creditable`,
/// whose own uncertainty the caller already propagates via
/// `srcLiveness.exact`), so this walk never discovers a *new* uncertainty of
/// its own and has no exactness of its own to report.
///
/// Returns false (leaving \p peak unspecified) if the walk exceeds
/// \p corridorOpCap, so the caller falls back to a fully conservative charge
/// instead of an unbounded one.
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
    bool resultAlreadyLive =
        analysis.pressureAt(op, cvtResult, queryCache).valueLive;
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
      // `credit` is false here either because nothing is live to credit, or
      // because a real remaining reader provably keeps `src` alive, or
      // because `srcLiveness.creditable` is false -- and that last case is
      // already reflected in `srcLiveness.exact`, which the caller ANDs into
      // `projection.exact` once for the whole function. None of the three is
      // a *local* uncertainty this loop discovers, so there is nothing to
      // clear `exact` for here.
      if (credit)
        regionPriced -= srcBytes;
      peak = std::max(peak, regionPriced);
    }
  }
  return true;
}

/// Returns the peak per-thread pressure over every block in \p op's own
/// regions except \p excluded, crediting \p src the same way everywhere else
/// in this file and skipping \p dstBytes wherever \p cvtResult is already
/// live there in the current, pre-hoist IR.
///
/// Used to price a sibling of the one block `priceTailAfter` already prices
/// precisely (an `else` branch when `cvtOp`'s only use is in `then`, say):
/// \p op (`lastUse`) is never itself a corridor entry once its real nested
/// last use is found, so nothing else in this function ever looks at its
/// other blocks -- but once hoisted, `cvtResult` becomes loop-invariant and,
/// by the same loop back-edge rule as everywhere else in this file, live
/// through *every* block of the loop body it now sits above, sibling blocks
/// included.
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
      // Any op in the block answers "is cvtResult/src already live here" the
      // same way, *given this function's own callers*: `realLastUse` already
      // guarantees no sibling block holds a use of `cvtResult` (all of
      // `cvtOp`'s uses share one block, checked before this is ever called),
      // and `srcLiveness.creditable` guarantees no other in-loop reader of
      // `src` -- so within a sibling block, liveness for either value can only
      // come from the ancestor live-through set, which `computeLiveValues`
      // unions in identically (keyed on `op`, this block's parent) for every
      // op in the block, making membership independent of which op is asked.
      // This does not hold unconditionally: absent that precondition, a value
      // can die partway through a block (a real, local last use inside it),
      // and asking at `block.front()` instead of at the actual die point would
      // wrongly answer "live" past where it no longer is.
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
    // One live-value set build answers both questions; that set is uncached and
    // is the dominant cost of this walk.
    //
    // The result is charged unconditionally because it is newly live at every
    // corridor operation. For a point in the anchor's own block (up to and
    // including the loop) or a body operation *above* the conversion's old
    // position, that is because the result's live range now begins earlier
    // than it used to. For a body operation strictly *after* the conversion's
    // own last use, it is because the loop's back edge makes the hoisted
    // result live through the whole loop, where the un-hoisted value would
    // have already been dead (see `collectCorridor`'s doc comment). Either
    // way, a result yielded out of the loop stays confined to the body region
    // -- the yield creates a loop *result*, a different value -- so nothing
    // outside the corridor needs pricing on the result's account.
    auto point = analysis.pressureAt(op, src, queryCache);
    uint64_t priced = point.pressure + dstBytes;
    // Only subtract what the analysis actually counted here, and only where the
    // move really does retire it.
    if (srcLiveness.creditable && point.valueLive &&
        srcDeadAfterMoveAt(srcLiveness, op, forOp))
      priced -= srcBytes;
    projection.corridorTerm = std::max(projection.corridorTerm, priced);

    // `op` may itself hold a region the corridor steps over without
    // descending into (a *sibling* loop between the anchor and `forOp`, say).
    // `pressureAt` above only reports what is live at `op`'s own program
    // point, not the peak reached inside it, so also check that peak
    // directly.
    //
    // `forOp` itself is excluded: it is not a region being stepped over, it
    // is the loop `cvtOp` is being hoisted *out of*, and its own body is
    // already priced op by op elsewhere in this same corridor (both the
    // ops above `cvtOp` and the ones after its last use). Reusing
    // `regionPeakThroughOp` for `forOp` would price its own pre-hoist
    // internal peak plus `dstBytes` a second time, uncredited, on top of
    // that per-op accounting.
    if (op->getNumRegions() > 0 && op != forOp.getOperation()) {
      uint64_t regionPriced =
          regionPeakThroughOp(op, analysis, queryCache, regionPeakCache) +
          dstBytes;
      // Reuse the exact same credit the flat charge above just applied. It
      // already proves that -- apart from `cvtOp` itself -- nothing at or
      // after `op` reads `src`, and `op`'s own region is a subset of "at or
      // after `op`": whenever `src` is counted inside that region at all
      // (via the ancestor live-through rule, because something past the
      // region still needs it), that something can only be `cvtOp`, since
      // every other real consumer is already accounted for in `srcLiveness`.
      // So wherever this credit applies, the subtraction is exact, not
      // merely conservative, here just as it is above. And wherever it does
      // not apply, that is either because nothing here is live to credit or
      // because a real remaining reader provably keeps `src` alive, or
      // because `srcLiveness.creditable` is false -- already reflected in
      // `srcLiveness.exact` above, not a fresh local uncertainty -- so there
      // is nothing to clear `exact` for on this branch either.
      if (srcLiveness.creditable && point.valueLive &&
          srcDeadAfterMoveAt(srcLiveness, op, forOp))
        regionPriced -= srcBytes;
      projection.corridorTerm = std::max(projection.corridorTerm, regionPriced);
    }
  }

  // `lastBodyUser` maps a use nested inside a sub-region (an `scf.if` branch,
  // say) onto the top-level op in the body that contains it, and
  // `collectCorridor` deliberately excludes that mapped op -- along with
  // everything from `cvtOp` up to it -- from the corridor, to avoid
  // double-counting the portion already locally live before the real,
  // nested last use. That exclusion is only sound up to the actual nested
  // use point: if a higher-pressure operation follows later in the *same*
  // branch, still inside the mapped op's region, it runs strictly after
  // `cvtOp`'s real last use and needs the same new-charge treatment as any
  // other post-last-use corridor entry -- but because the mapped op itself
  // was never added to the corridor, neither the per-op loop above nor its
  // region-peak fallback ever prices it.
  if (Operation *lastUse = lastBodyUser(cvtOp, forOp.getBody());
      lastUse && lastUse->getNumRegions() > 0) {
    // Try to price only the tail after the real nested use, crediting `src`
    // the same exact way the rest of this function does. This is only sound
    // one nesting level below `lastUse`: `realLastUse` needs every use of
    // `cvtOp`'s result to share one immediate block (unrelated to how deep
    // that block sits), and `priceTailAfter`'s own single-block walk covers
    // everything after that use only up to the end of *that* block -- if the
    // block sits deeper than directly inside `lastUse`, whatever lies between
    // the end of that inner block and the end of `lastUse`'s own region would
    // go unpriced. Both conditions hold for the common case this fixes (a
    // single use early in one `scf.if` branch, followed by a filler chain in
    // that same branch), so falling back conservatively for anything more
    // exotic costs precision only where the precise answer would take
    // real extra work to get right, not correctness.
    //
    // A further precondition, beyond the two above: `nestedLast` itself must
    // hold no regions of its own. `priceTailAfter`'s walk starts at
    // `nestedLast->getNextNode()` -- it prices every op *after* `nestedLast`
    // in its own block, checking each one for regions via
    // `regionPeakThroughOp`, but it never applies that same check to
    // `nestedLast` itself. If `cvtOp`'s result is used only as, say, an inner
    // `scf.for`'s own `iter_args` init (so `nestedLast` is that inner loop),
    // the inner loop's own internal peak would go unpriced by both this path
    // and `siblingBlocksPeak` (which only covers `lastUse`'s *other* blocks,
    // not `nestedLast`'s own nested ones) -- exactly the same class of gap
    // this function exists to close, one level deeper. Falling back to the
    // conservative whole-region charge below is safe and simple; a precise
    // charge for this case would need its own `regionPeakThroughOp` call on
    // `nestedLast`, not attempted here since the fallback already covers it.
    Operation *nestedLast = realLastUse(cvtOp);
    uint64_t tailPeak = 0;
    if (nestedLast && nestedLast->getBlock()->getParentOp() == lastUse &&
        nestedLast->getNumRegions() == 0 &&
        priceTailAfter(nestedLast, src, srcBytes, cvtOp.getResult(), dstBytes,
                       analysis, srcLiveness, forOp, corridorOpCap, queryCache,
                       regionPeakCache, tailPeak)) {
      projection.corridorTerm = std::max(projection.corridorTerm, tailPeak);

      // `lastUse` may hold other blocks besides `nestedLast`'s own (an
      // `else` branch, say) that the precise tail walk above never visits.
      // Price them too; see `siblingBlocksPeak`'s doc for why they matter.
      uint64_t siblingPeak = siblingBlocksPeak(
          lastUse, nestedLast->getBlock(), src, srcBytes, cvtOp.getResult(),
          dstBytes, analysis, srcLiveness, forOp, queryCache);
      projection.corridorTerm = std::max(projection.corridorTerm, siblingPeak);
    } else {
      // Conservatively price the mapped op's own whole-region peak instead:
      // this may overstate the true peak (it covers the whole region, not
      // just the tail after the real use, and applies no source credit),
      // which is why it also clears `exact`, but it closes the gap rather
      // than silently pricing it as zero.
      uint64_t regionPriced =
          regionPeakThroughOp(lastUse, analysis, queryCache, regionPeakCache) +
          dstBytes;
      projection.corridorTerm = std::max(projection.corridorTerm, regionPriced);
      projection.exact = false;
    }
  }

  // No operation *outside the loop and after the anchor* needs pricing beyond
  // what the corridor above already covers: the result stays confined to the
  // anchor's block, the loop op, and the loop body -- see `collectCorridor`'s
  // doc comment for why the body's coverage now extends past the conversion's
  // old position too.
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
  /// `Accept` the trial's IR stands and the next `decide`/`beginTrial` rebuilds
  /// against it, checking it against the figure measured here.
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
    dirty = true;
    pendingProjection = trialPeak;
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
  /// projection that predicted it exist. (A candidate `decide` rejects cannot
  /// be checked this way without speculatively mutating IR, which only a
  /// trial -- see `beginTrial` -- does.)
  ///
  /// Returns false once the rebuild cap is spent. Reusing the last analysis
  /// would be unsound, not merely loose: after a hoist it describes pre-move
  /// IR, so its `prePeak` is neither a bound on the current peak nor the
  /// ceiling to meet.
  bool refreshAnalysis() {
    if (!analysis) {
      // The first build is not a rebuild and does not spend the cap.
      analysis.emplace(func);
      prePeak = analysis->peakPressure(func);
      resetPerGenerationCaches();
      return true;
    }
    if (!dirty)
      return true;
    if (rebuilds >= rebuildCap)
      return false;
    ++rebuilds;
    analysis.emplace(func);
    prePeak = analysis->peakPressure(func);
    assert(prePeak <= pendingProjection &&
           "hoist raised the reported function peak above its projection");
    dirty = false;
    resetPerGenerationCaches();
    return true;
  }

  /// `queryCache`/`regionPeakCache` memoize answers about the IR `analysis`
  /// describes. Both must be wiped in lockstep with every `analysis` rebuild
  /// above -- an entry from a stale generation would silently describe IR
  /// that a since-accepted hoist has already moved.
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

/// Returns the loop-level gate's threshold: 80% of the per-lane GRF budget
/// \p grfBudget. The 20% headroom accounts for scalars, temporaries, and
/// loop-internal values not tracked by live-in liveness. Integer arithmetic
/// (4/5) avoids float-to-unsigned truncation.
static int liveInThreshold(unsigned grfBudget) {
  return static_cast<int>(grfBudget * 4 / 5);
}

/// Charges the loop of each twin in \p twinIdxs (a refusal that an accepted
/// hoist out of \p forOp makes `remove_layout_conversions` merge away) for
/// the merge that pass performs unconditionally once that hoist exists. Called
/// right after the hoist is accepted, for the twins already refused by then,
/// and by `chargeMergeIntoPriorHoist` for a twin refused only after it.
///
/// `remove_layout_conversions` replaces every twin `mergesIntoHoist` allows
/// with the hoisted result (see `collectMergingTwins`, which gathers \p
/// twinIdxs), whether or not `hoistRetiresSource` excused it from this
/// hoist's own delta, and whether or not `reconsiderSharedSources` later
/// moves the twin, so the twin's loop ends up exactly as if the twin had been
/// hoisted: the result becomes live-in, and the source -- which the merge
/// proves nothing outside the twins reads at or after \p forOp -- stops
/// being. That is `hoistDeltaBytes` of the twin, the same substitution
/// phase 2 prices when it moves a group, charged here once per loop and
/// flagged `mergeCharged` so phase 2 does not price it a second time whatever
/// its own verdict. Several twins in one loop merge into the one result, so
/// only the first is charged. A twin in \p forOp itself is flagged but not
/// charged: \p forOp's own delta already charged the result it merges into.
/// For a twin already refused when that delta was priced, the delta also
/// retired the source; for one refused only after it, the twin still read
/// the source then, so the figure errs high by at most that source. A twin
/// already charged for an earlier hoist of the same pair is skipped; its
/// loop's merge is already priced.
///
/// At hoist time, \p twinIdxs is every refusal `collectMergingTwins`
/// structurally matches to \p forOp's hoist, not only the ones this hoist's
/// own credit relied on: a sibling hoisting on its own uncredited merits
/// still merges an unexcused twin away. A same-pair conversion refused only
/// *after* this hoist, in a loop decided later, is no recorded refusal yet
/// and so not among them; `chargeMergeIntoPriorHoist` charges it through
/// this same function, with \p twinIdxs that one refusal, once its refusal
/// is recorded. Between the two, a merge of a refusal into a hoist is
/// charged whichever of the two was decided first. That matters
/// when the merging result is bigger per lane than the retired source (dst >
/// src, the replicated dot_b shape the twin credit targets): an uncharged
/// merge leaves the loop's figure below its post-merge live-in, and a later
/// shared-source group gated on that loop passes on the lower figure (see
/// `test/TritonIntelGPU/hoist-layout-conversions-uncharged-merge.mlir` and
/// `hoist-layout-conversions-late-refusal-merge.mlir`, one repro per order).
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

/// Charges the loop of the refusal just recorded, the last in \p state's
/// refusals, for its merge into a conversion the one-at-a-time phase already
/// hoisted out of a loop decided earlier (one physically later, since loops
/// are decided last to first).
///
/// `collectMergingTwins` could not see this pair when that hoist was
/// accepted: the conversion refused now was no refusal yet. Its refusal does
/// not stop the merge, though. `remove_layout_conversions` still replaces it
/// with the hoisted result wherever `mergesIntoHoist` holds, so its loop ends
/// up as if it had been hoisted after all. Charging that loop here, through
/// `chargeMergedTwins` (same delta, same `mergeCharged` flag), lets the
/// candidates this loop decides from now on, and the shared-source phase,
/// see the figure the IR will actually have.
///
/// A refusal in the hoisting loop itself is flagged but not charged, as
/// `chargeMergedTwins` does. There the hoist was priced with this
/// conversion still reading the source, so its loop's figure already holds
/// the merged result and errs high by at most the source it did not retire.
///
/// Only one-at-a-time hoists are matched: the shared-source phase records no
/// refusals, and moves every refusal of a source in one group.
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
/// pressure-reducing candidate has been credited -- and a rejection this phase
/// makes is final unless `reconsiderSharedSources` overturns it, since it ends
/// up stamping `tt.no_licm` and the later generic LICM pass never revisits the
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

  // Whether a refused twin's downstream merge, which a credit past it relies
  // on, keeps the twin's loop within this same loop-level gate. This does not
  // look at why the twin was refused (`Refusal::reason` is never read here):
  // it simply reruns the loop-level gate for the twin's own loop, at that
  // loop's current figure plus the delta the merge would add. A twin the
  // loop-level gate itself refused will, by construction, still fail this
  // same gate run again unless a merge charge has since lowered its loop's
  // figure: its own refusal already described the merged loop exactly, and
  // the credit must not be what overrides it. A twin only the peak gate
  // refused was never checked against the loop-level gate as failing, so it
  // commonly keeps its credit here -- but that is not because the merged
  // state is known to be no worse: this function never re-examines the peak
  // gate's own concern, the whole-function peak, for the twin's loop at all.
  // Measured on the kernel this asymmetry targets (_attn_bwd's forced config
  // at grf256), the merge is not safe by that measure: the peak gate refused
  // the K twin at a projected 1876, but after HLC and the downstream merge
  // both run, the real function peak is 1940, over both that refused figure
  // and HLC's 1812 ceiling. The asymmetry is kept anyway, backed by hardware
  // evidence rather than by this projection: at that config n_spills=0,
  // matching main, so the reported peak overstates physical demand there.
  // The twin's delta is asked without this predicate, so the check cannot
  // recurse.
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
    // Every refusal `remove_layout_conversions` will merge into `cvtOp`'s
    // result, not just the ones this hoist's own credit relied on:
    // `mergesIntoHoist` only looks at `forOp`, the twin, and the source's
    // defining op, none of which `cvtOp`'s own position affects, so this can
    // run before or after the move with the same answer.
    SmallVector<unsigned> mergingTwins;
    collectMergingTwins(cvtOp, forOp, state.refusals, mergingTwins);
    // Hoist the conversion out of the loop, to the program point the projection
    // above was measured at.
    if (Operation *anchor = hoistAnchor(cvtOp, forOp))
      cvtOp->moveAfter(anchor);
    else
      cvtOp->moveBefore(forOp);

    ++NumHoisted;
    netBytes += bestDelta;
    state.hoistedSources.insert(cvtOp.getSrc());
    state.hoisted.push_back({cvtOp, forOp});
    peakGate.noteHoisted();
    chargeMergedTwins(mergingTwins, forOp, analysis, state);
  }
}

/// Moves \p cvtOp out of \p forOp to the point `hoistAnchor` names -- the same
/// point `hoistCvtDotOpsOutOfLoop` moves an accepted candidate to.
static void moveToHoistAnchor(ttg::ConvertLayoutOp cvtOp, scf::ForOp forOp) {
  if (Operation *anchor = hoistAnchor(cvtOp, forOp))
    cvtOp->moveAfter(anchor);
  else
    cvtOp->moveBefore(forOp);
}

/// Second phase: re-decide, *jointly*, the refused conversions of each source
/// that is shared -- by two or more refused conversions, or by a refused one
/// and one already hoisted.
///
/// Deciding one conversion at a time cannot see what a group sharing a source
/// is worth together. Each is weighed while the others still read the source,
/// so none earns the credit for retiring it, and the whole-function projection
/// charges each arriving result on top of a source that stays live. Case 16 in
/// hoist-layout-conversions.mlir is the shape: one source feeding a conversion
/// in each of two sibling loops. Alone, the later loop's hoist lands above the
/// earlier loop, whose own conversion still holds the source live through it,
/// so the peak rises; and the earlier loop's hoist earns its credit only
/// because the later loop's refused conversion is an unenforceable twin (see
/// `hoistRetiresSource`). Together, the source dies right after the two
/// conversions and the function's peak falls. Re-queueing a refusal once a
/// sibling is hoisted is not enough: without a twin to credit past -- the
/// result types differ, as in case 50 -- no sibling is ever hoisted alone at
/// 128-GRF, so there is no such moment.
///
/// A group made only of `mergeCharged` refusals still gets a trial, though it
/// merges downstream whatever the verdict: an accepted trial moves it out of
/// its loops, and every later trial in this phase is measured against, and
/// takes its ceiling from, the IR that leaves. Its trial spends the rebuild cap
/// like any other, and a refused one leaves its members counted as refusals
/// although none survives the pipeline.
///
/// A group is decided by applying it and measuring, since the single-candidate
/// projection has no way to express several moves at once. It must pass both
/// vetoes, as one change:
///   - the loop-level gate, once per loop the group leaves, with that loop's
///     conversions' results charged and the source credited when the move
///     retires it from that loop -- the same substitution as a single hoist,
///     asked of the IR with the whole group moved;
///   - the whole-function peak gate, measured on the moved IR against the same
///     `max(prePeak, threshold)` ceiling.
/// A group either moves as a whole or is restored exactly; subsets are not
/// tried. Groups are decided once each, in the order their first refusal was
/// made.
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
      // Its loop already carries this move's effect: an earlier hoist already
      // made this refusal's merge inevitable and charged its loop for it (see
      // `chargeMergedTwins`), whether or not that hoist's own credit relied
      // on this refusal specifically. Skipping it here avoids pricing that
      // same merge a second time. This does not drop the whole loop out of
      // the gate: a different member of this group sharing the loop but not
      // this refusal's (source, type) pair, or one not yet merge-charged, is
      // still priced below.
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
    uint64_t threshold = grfBudget * 4 / 5;
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
