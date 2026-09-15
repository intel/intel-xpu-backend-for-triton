#include "intel/include/Dialect/Triton/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-intel-optimize-load-masks"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = mlir::triton;

namespace mlir::triton::intel {
#define GEN_PASS_DEF_TRITONINTELOPTIMIZELOADMASKS
#include "intel/include/Dialect/Triton/Transforms/Passes.h.inc"
} // namespace mlir::triton::intel

namespace {

/// The predicates assumed constant on the lanes reaching the value being
/// simplified. `{%c, true}` reads "`%c` holds in every lane this value
/// decides".
using Facts = SmallVector<std::pair<Value, bool>, 4>;

/// Descending into a select whose condition is not already assumed splits the
/// search in two, so bound both the nesting and the total work.
constexpr unsigned MaxDepth = 12;
constexpr unsigned MaxSteps = 512;

/// How far forward from a load the search for its consumers' conditions walks.
constexpr unsigned MaxUseChain = 16;

/// How many operations may be moved above a load to make a condition narrowing
/// its mask available there. Everything moved has its live range stretched
/// across the load, and the load cannot issue until it has retired, neither of
/// which is weighed against the lanes narrowing saves -- so bound the breadth
/// the way `MaxDepth` bounds the depth.
constexpr unsigned MaxHoist = 4;

static std::optional<bool> lookupFact(const Facts &facts, Value v) {
  for (auto [pred, holds] : facts)
    if (pred == v)
      return holds;
  return std::nullopt;
}

/// Returns the constant boolean \p v holds, uniformly across lanes. Only an
/// `i1` answers: a wider integer constant is not a mask, and reading one as
/// though it were would turn a bitwise `arith.andi` into an identity.
static std::optional<bool> getConstantMask(Value v) {
  Attribute attr;
  if (!matchPattern(v, m_Constant(&attr)))
    return std::nullopt;
  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    if (!intAttr.getType().isInteger(1))
      return std::nullopt;
    return intAttr.getValue().isOne();
  }
  if (auto dense = dyn_cast<DenseIntElementsAttr>(attr))
    if (dense.isSplat() && dense.getElementType().isInteger(1))
      return dense.getSplatValue<APInt>().isOne();
  return std::nullopt;
}

/// Returns the attribute for a uniform boolean of \p type, which may be a
/// tensor of `i1` or a plain `i1`.
static TypedAttr boolAttr(Type type, bool value, OpBuilder &builder) {
  if (auto shaped = dyn_cast<ShapedType>(type))
    return DenseElementsAttr::get(shaped, value);
  return builder.getBoolAttr(value);
}

/// Returns true if \p op maps lane `i` of its result from lane `i` of its
/// operands only, and has no side effects.
///
/// Only such an operation may be rewritten under an assumption carried down a
/// select arm: the assumption holds on the lanes that took the arm and on no
/// others, so an operation that mixes lanes (a reduction, a broadcast, a
/// shuffle) would be computing a lane of its result from lanes whose assumption
/// does not hold. `tt.load` qualifies as well but is not pure, so it is matched
/// separately.
static bool isLaneWise(Operation *op) {
  // `tt.elementwise_inline_asm` carries the trait but is not lane-wise in this
  // sense: it hands the asm `packed_element` lanes at a time and leaves the
  // grouping to it, so a lane of its result may be computed from a neighbouring
  // lane's input.
  if (isa<tt::ElementwiseInlineAsmOp>(op))
    return false;
  return isPure(op) && op->hasTrait<OpTrait::Elementwise>();
}

/// Returns true if \p v is absent or a zero splat, i.e. denotes the value a
/// `tt.load` yields on its masked-off lanes when no `other` is given.
///
/// An omitted `other` is not "don't care": the Intel lowering materializes an
/// explicit zero for those lanes (`LoadStoreOpToLLVM.cpp`, the `otherElems`
/// empty case), and the 2D block path relies on the hardware zero-filling them.
/// So an absent `other` and an explicit zero one are interchangeable.
static bool isZeroOther(Value v) {
  if (!v)
    return true;
  auto isZero = [](Attribute attr) {
    if (auto intAttr = dyn_cast<IntegerAttr>(attr))
      return intAttr.getValue().isZero();
    if (auto floatAttr = dyn_cast<FloatAttr>(attr))
      return floatAttr.getValue().isZero() &&
             !floatAttr.getValue().isNegative();
    return false;
  };
  Attribute attr;
  if (!matchPattern(v, m_Constant(&attr)))
    return false;
  if (auto dense = dyn_cast<DenseElementsAttr>(attr))
    return dense.isSplat() && isZero(dense.getSplatValue<Attribute>());
  return isZero(attr);
}

/// Returns true if two `tt.load`s with the same mask yield the same values on
/// their masked-off lanes, given their respective `other` operands.
static bool sameMaskedOffValue(Value lhs, Value rhs) {
  return lhs == rhs || (isZeroOther(lhs) && isZeroOther(rhs));
}

/// Returns true if \p op might write to memory, or might have an effect this
/// pass cannot reason about.
static bool mayWriteMemory(Operation *op) {
  if (isMemoryEffectFree(op))
    return false;
  // A region holds operations whose effects `getEffects` does not report.
  if (op->getNumRegions() != 0)
    return true;
  auto memOp = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memOp)
    return true;
  SmallVector<MemoryEffects::EffectInstance> effects;
  memOp.getEffects(effects);
  return llvm::any_of(effects, [](const MemoryEffects::EffectInstance &effect) {
    return !isa<MemoryEffects::Read>(effect.getEffect());
  });
}

/// Returns true if an operation between \p from and \p to, both in the same
/// block, might write memory. Used to check that two loads of the same address
/// observe the same contents.
static bool writesBetween(Operation *from, Operation *to) {
  for (Operation *op = from->getNextNode(); op != to; op = op->getNextNode()) {
    assert(op && "`to` follows `from` in the same block");
    if (mayWriteMemory(op))
      return true;
  }
  return false;
}

/// Flattens the `i1` conjunction rooted at \p v into \p conjuncts. A mask built
/// as `%a & %b & %c` contributes all three, because the mask is false wherever
/// any single one of them is.
static void collectConjuncts(Value v, SmallVectorImpl<Value> &conjuncts,
                             unsigned depth = 0) {
  auto andOp = v.getDefiningOp<arith::AndIOp>();
  if (depth > MaxDepth || !andOp ||
      !getElementTypeOrSelf(v.getType()).isInteger(1)) {
    conjuncts.push_back(v);
    return;
  }
  collectConjuncts(andOp.getLhs(), conjuncts, depth + 1);
  collectConjuncts(andOp.getRhs(), conjuncts, depth + 1);
}

/// Collects into \p facts the conditions under which the data \p v carries is
/// observed. Reaching the true arm of `select %c` means every lane of \p v that
/// anything downstream ever reads is a lane where `%c` holds.
///
/// This is the same relation `simplify` consumes, gathered in the opposite
/// direction: `simplify` is handed the conditions of the arm it was called for
/// and walks towards the definitions, this walks from a definition towards its
/// consumers to discover them.
///
/// Returns false if \p v flows somewhere this walk cannot describe, or if
/// nothing was learned; in either case \p facts must not be used.
static bool collectUseFacts(Value v, Facts &facts) {
  for (unsigned step = 0; step != MaxUseChain; ++step) {
    // Nothing observes this at all. `simplify` leaves the arm it replaced
    // behind for the canonicalizer, so a chain that dead-ends is exactly a load
    // it has already retired; strengthening that load's mask would be pure
    // waste.
    if (v.use_empty())
      return false;

    // With more than one consumer the value is observed under the *disjunction*
    // of their conditions, which this does not represent. The facts gathered so
    // far still hold -- every one of those consumers is downstream of them --
    // so stop here rather than give up.
    if (!v.hasOneUse())
      return !facts.empty();

    OpOperand &use = *v.getUses().begin();
    Operation *user = use.getOwner();

    if (auto select = dyn_cast<arith::SelectOp>(user)) {
      // Operand 0 is the condition. A value used there is not a datum whose
      // lanes the select picks between, so it carries no condition.
      if (use.getOperandNumber() == 0)
        return false;
      facts.emplace_back(select.getCondition(), use.getOperandNumber() == 1);
      v = select.getResult();
      continue;
    }

    // Anything that mixes lanes reads the very lanes the conditions would let
    // us drop, so it ends the walk -- for the same reason `simplify` refuses to
    // rewrite through one.
    if (!isLaneWise(user) || user->getNumResults() != 1)
      return !facts.empty();
    v = user->getResult(0);
  }
  return !facts.empty();
}

/// Returns true if every lane on which a mask with conjuncts \p maskConjuncts
/// is true is a lane on which \p pred is true. Established structurally: each
/// conjunct of \p pred is also a conjunct of the mask.
///
/// Both sides are flattened, because a predicate is routinely a conjunction
/// that the mask carries as its own conjuncts rather than as that one value:
/// the mask
/// `(%a & %b) & %w` implies `%a & %b`, but neither `%a & %b` nor its parts
/// appear in the mask's leaves as a single term to compare against.
static bool maskImplies(ArrayRef<Value> maskConjuncts, Value pred) {
  SmallVector<Value> predConjuncts;
  collectConjuncts(pred, predConjuncts);
  return llvm::all_of(predConjuncts, [&](Value conjunct) {
    return llvm::is_contained(maskConjuncts, conjunct);
  });
}

/// Returns true if \p mask is false on every lane where \p facts hold: the mask
/// requires a predicate to hold that the facts require not to.
static bool maskExcludedByFacts(Value mask, const Facts &facts) {
  SmallVector<Value> conjuncts;
  collectConjuncts(mask, conjuncts);
  return llvm::any_of(facts, [&](const std::pair<Value, bool> &fact) {
    return !fact.second && maskImplies(conjuncts, fact.first);
  });
}

/// Erases the operations `simplify` orphaned. It replaces the value feeding a
/// select arm and leaves the old one in place, so what it retires is a whole
/// subtree whose every internal value still has a consumer -- walking a use
/// chain cannot recognise that, and the second phase has to not see it.
///
/// Only operations that produce a value and hold no region are considered,
/// which is everything `simplify` clones and leaves behind, and keeps functions
/// and anything with hidden effects out of reach.
static void eraseOrphanedOps(ModuleOp mod, RewriterBase &rewriter) {
  SmallVector<Operation *> candidates;
  mod.walk([&](Operation *op) {
    if (op->getNumResults() != 0 && op->getNumRegions() == 0)
      candidates.push_back(op);
  });
  // `walk` yields a block in program order, so visiting in reverse considers
  // every consumer of a value before the value itself. One pass therefore
  // retires a whole chain.
  for (Operation *op : llvm::reverse(candidates))
    if (isOpTriviallyDead(op))
      rewriter.eraseOp(op);
}

/// Cleans up after `simplify` so `narrowMasksByUse` can see what it left
/// behind.
///
/// `simplify` clones the operations between a load and the select arm it feeds,
/// so the load it kept ends up with several identical consumers -- and the
/// narrowing, which follows a single-use chain, stops at the first of them.
/// Deduplicating those is not enough on its own: what they feed are selects
/// whose two arms `simplify` made equal, and each of those has to fold to its
/// arm before the duplicates above it become dead and CSE can merge what
/// remains.
///
/// Both are needed. Neither CSE nor folding alone exposes the narrowing on a
/// `tl.where` tree, because the two take turns: folding a select makes a
/// duplicate dead, erasing it makes the next select foldable.
static void cleanupAfterReuse(ModuleOp mod, RewriterBase &rewriter) {
  MLIRContext *ctx = mod.getContext();
  eraseOrphanedOps(mod, rewriter);
  DominanceInfo domInfo(mod);
  eliminateCommonSubExpressions(rewriter, domInfo, mod);
  // The greedy driver folds and retires dead operations on its own, which is
  // what `select %c, %v, %v -> %v` needs; the select patterns are here for the
  // arms `simplify` leaves in other shapes.
  RewritePatternSet patterns(ctx);
  arith::SelectOp::getCanonicalizationPatterns(patterns, ctx);
  // The result is deliberately dropped. The greedy driver also reports failure
  // when it merely runs out of iterations, which says nothing about the IR: the
  // rewrites already applied are valid either way, and the only cost of an
  // unfinished clean-up is that the narrowing below sees fewer opportunities.
  // Failing the pass over it would abort a compile on valid input.
  if (failed(applyPatternsGreedily(mod, std::move(patterns))))
    LDBG("clean-up did not converge; continuing");
}

/// Collects into \p toMove the pure operations that have to be moved above \p
/// before for \p v to be available there, in an order where each comes after
/// what it depends on. Returns false, having collected nothing usable, if some
/// definition cannot be moved.
///
/// A condition is routinely computed *after* the load it could narrow -- the
/// two are unrelated in the source, so Inductor emits them in source order. The
/// values that condition is computed from are already available; only its own
/// mask arithmetic sits too late.
///
/// Moving anything at all has a price nothing here can weigh: the moved value
/// is live across the load where before it did not exist yet, and the load
/// waits on it instead of issuing while it computes. Two bounds keep that
/// price to what narrowing was going to pay regardless. Only a mask, or a
/// constant, may move: narrowing needs a mask live at the load in any case and
/// a constant is rematerialized rather than carried, where the wider arithmetic
/// behind a mask -- an index tensor of the loaded shape, several times the
/// load's own destination -- would be held across the load for nothing, and
/// would delay it. And no more than `MaxHoist` operations, so a wide mask DAG
/// cannot pile up what the depth limit alone would admit.
static bool collectHoistable(Value v, Operation *before, DominanceInfo &domInfo,
                             SetVector<Operation *> &toMove, unsigned depth) {
  if (domInfo.properlyDominates(v, before))
    return true;
  if (depth > MaxDepth)
    return false;
  Operation *def = v.getDefiningOp();
  // A value that does not already dominate and has no definition to move (a
  // block argument of another block) cannot be made available. Neither can one
  // whose definition is not pure, sits in another block, or hides operations in
  // a region.
  if (!def || def->getBlock() != before->getBlock() || !isPure(def) ||
      def->getNumRegions() != 0)
    return false;
  if (!getElementTypeOrSelf(v.getType()).isInteger(1) &&
      !def->hasTrait<OpTrait::ConstantLike>())
    return false;
  for (Value operand : def->getOperands())
    if (!collectHoistable(operand, before, domInfo, toMove, depth + 1))
      return false;
  toMove.insert(def);
  return toMove.size() <= MaxHoist;
}

class Propagator {
public:
  Propagator(ModuleOp mod) : rewriter(mod.getContext()) {}

  /// Rewrites the select arms and load masks of \p mod, returning whether the
  /// IR was modified. Every rewrite is optional, so this cannot fail: one that
  /// does not apply is simply not made.
  bool run(ModuleOp mod);

private:
  /// Strengthens load masks by the conditions their consumers impose.
  bool narrowMasksByUse(ModuleOp mod);

  /// Returns a value equal to \p v on every lane where \p facts hold, or \p v
  /// itself if nothing could be simplified. Operations created along the way
  /// are recorded so that they can be dropped if the rewrite is not worth
  /// keeping.
  Value simplify(Value v, const Facts &facts, unsigned depth);
  Value simplifyLoad(tt::LoadOp load, const Facts &facts, unsigned depth);

  /// Returns a load already in the program that produces, on the lanes where
  /// the assumption holds, what \p load produces -- given that its mask is \p
  /// mask there. Null if there is none.
  tt::LoadOp findRedundantLoad(tt::LoadOp load, Value mask);

  Value boolConstant(Type type, bool value, Location loc);
  Value cloneWithOperands(Operation *op, ValueRange operands);

  /// Undoes everything `simplify` created since the last `reset`.
  void rollback();
  void reset();

  IRRewriter rewriter;
  /// Operations created by the current propagation, in creation order.
  SmallVector<Operation *> created;
  unsigned steps = 0;
  /// How many `tt.load`s the current propagation made redundant. The rewrite is
  /// only worth its cloned operations if this is non-zero.
  unsigned loadsFolded = 0;
};

Value Propagator::boolConstant(Type type, bool value, Location loc) {
  TypedAttr attr = boolAttr(type, value, rewriter);
  Operation *op = arith::ConstantOp::create(rewriter, loc, type, attr);
  created.push_back(op);
  return op->getResult(0);
}

Value Propagator::cloneWithOperands(Operation *op, ValueRange operands) {
  Operation *clone = rewriter.clone(*op);
  clone->setOperands(operands);
  created.push_back(clone);
  return clone->getResult(0);
}

void Propagator::rollback() {
  // Nothing outside `created` uses these results yet, and a created operation
  // is only ever used by a later one, so erasing in reverse order is safe.
  for (Operation *op : llvm::reverse(created))
    rewriter.eraseOp(op);
  reset();
}

void Propagator::reset() {
  created.clear();
  steps = 0;
  loadsFolded = 0;
}

Value Propagator::simplify(Value v, const Facts &facts, unsigned depth) {
  if (depth > MaxDepth || ++steps > MaxSteps)
    return v;

  // A predicate the arm assumes is that constant here. Substituting it is what
  // lets a *nested* conjunction collapse: `(%a & %b) & %w` reaches `%w` only
  // once `%a & %b` has folded to true, which needs `%b` itself to become a
  // constant. Facts are select conditions, so this only ever rewrites an `i1`.
  if (std::optional<bool> holds = lookupFact(facts, v))
    return boolConstant(v.getType(), *holds, v.getLoc());

  Operation *def = v.getDefiningOp();
  if (!def)
    return v;

  if (auto load = dyn_cast<tt::LoadOp>(def))
    return simplifyLoad(load, facts, depth);

  if (!isLaneWise(def) || def->getNumResults() != 1)
    return v;

  // A select on an assumed predicate is just the arm the assumption picks; a
  // select on anything else is where new assumptions come from.
  if (auto select = dyn_cast<arith::SelectOp>(def)) {
    Value cond = select.getCondition();
    if (std::optional<bool> holds = lookupFact(facts, cond))
      return simplify(*holds ? select.getTrueValue() : select.getFalseValue(),
                      facts, depth + 1);

    Facts whenTrue(facts), whenFalse(facts);
    whenTrue.emplace_back(cond, true);
    whenFalse.emplace_back(cond, false);
    Value trueVal = simplify(select.getTrueValue(), whenTrue, depth + 1);
    Value falseVal = simplify(select.getFalseValue(), whenFalse, depth + 1);
    if (trueVal == select.getTrueValue() && falseVal == select.getFalseValue())
      return v;
    return cloneWithOperands(select, {cond, trueVal, falseVal});
  }

  // `%c & %w` is `%w` wherever `%c` holds, and false wherever it does not. This
  // is the rule the whole pass exists for: it is what makes a load under a
  // narrow mask coincide with a load under a wider one.
  if (auto andOp = dyn_cast<arith::AndIOp>(def);
      andOp && getElementTypeOrSelf(andOp.getType()).isInteger(1)) {
    Value lhs = andOp.getLhs(), rhs = andOp.getRhs();
    std::optional<bool> lhsHolds = lookupFact(facts, lhs);
    std::optional<bool> rhsHolds = lookupFact(facts, rhs);
    if ((lhsHolds && !*lhsHolds) || (rhsHolds && !*rhsHolds))
      return boolConstant(andOp.getType(), false, andOp.getLoc());
    if (lhsHolds && *lhsHolds)
      return simplify(rhs, facts, depth + 1);
    if (rhsHolds && *rhsHolds)
      return simplify(lhs, facts, depth + 1);
  }

  // Otherwise recurse: an assumed predicate may sit deeper in the tree, behind
  // a nested conjunction or a value cast.
  SmallVector<Value> operands;
  bool anyChanged = false;
  for (Value operand : def->getOperands()) {
    Value simplified = simplify(operand, facts, depth + 1);
    anyChanged |= simplified != operand;
    operands.push_back(simplified);
  }
  if (!anyChanged)
    return v;

  // Folding the rebuilt operation is left to the canonicalizer, except for the
  // conjunction, whose collapse is what the load lookup keys on.
  if (auto andOp = dyn_cast<arith::AndIOp>(def)) {
    std::optional<bool> lhs = getConstantMask(operands[0]);
    std::optional<bool> rhs = getConstantMask(operands[1]);
    if ((lhs && !*lhs) || (rhs && !*rhs))
      return boolConstant(andOp.getType(), false, andOp.getLoc());
    if (lhs && *lhs)
      return operands[1];
    if (rhs && *rhs)
      return operands[0];
  }
  return cloneWithOperands(def, operands);
}

Value Propagator::simplifyLoad(tt::LoadOp load, const Facts &facts,
                               unsigned depth) {
  Value mask = load.getMask();
  // A volatile load must be issued exactly as written; an unmasked one has no
  // mask to simplify.
  if (!mask || load.getIsVolatile())
    return load.getResult();

  Value simplified = simplify(mask, facts, depth + 1);
  if (simplified == mask)
    return load.getResult();

  // A mask that is false on every lane the assumption covers means this load
  // never contributed memory here: `other` is the whole result.
  if (std::optional<bool> constant = getConstantMask(simplified);
      constant && !*constant) {
    if (Value other = load.getOther()) {
      ++loadsFolded;
      return other;
    }
    return load.getResult();
  }

  // Otherwise the mask got weaker. Issuing a read under it would touch
  // addresses this load never touched, so never synthesize one -- only reuse a
  // load the program already performs, which cannot introduce an out-of-bounds
  // access.
  if (tt::LoadOp redundant = findRedundantLoad(load, simplified)) {
    LDBG("reusing " << *redundant.getOperation() << " for "
                    << *load.getOperation());
    ++loadsFolded;
    return redundant.getResult();
  }
  return load.getResult();
}

tt::LoadOp Propagator::findRedundantLoad(tt::LoadOp load, Value mask) {
  std::optional<bool> constant = getConstantMask(mask);
  bool wantUnmasked = constant && *constant;

  for (Operation &op : *load->getBlock()) {
    auto candidate = dyn_cast<tt::LoadOp>(&op);
    if (!candidate || candidate == load || candidate.getIsVolatile())
      continue;
    // Grouping on the pointer *value* is what makes "same address" trivially
    // true; it holds in practice because CSE has unified the `tt.addptr`
    // chains.
    if (candidate.getPtr() != load.getPtr() ||
        candidate.getType() != load.getType() ||
        candidate.getCache() != load.getCache() ||
        candidate.getEvict() != load.getEvict())
      continue;
    // The candidate has to read exactly the weakened mask's lanes.
    if (wantUnmasked ? static_cast<bool>(candidate.getMask())
                     : candidate.getMask() != mask)
      continue;
    // It has to be available where the value is needed, and observe the same
    // memory contents `load` observed.
    if (!candidate->isBeforeInBlock(load) || writesBetween(candidate, load))
      continue;

    // An unmasked candidate reads every lane, so it has no masked-off lanes to
    // agree about.
    if (wantUnmasked)
      return candidate;

    // Otherwise the two have to produce the same thing on the lanes the
    // weakened mask excludes, where each yields its own `other`.
    if (!sameMaskedOffValue(load.getOther(), candidate.getOther()))
      continue;
    return candidate;
  }
  return nullptr;
}

/// Strengthens the mask of a load by the conditions under which its consumers
/// observe it, so it stops reading lanes nothing ever looks at. When the
/// strengthened mask is empty the load goes away entirely.
///
/// This is the mirror image of `simplify`. There a select's condition is
/// carried *down* into a mask to make it weaker, which pays off when the
/// weakened mask makes the load coincide with one the program already performs.
/// Here the consumers' conditions are carried *up* into a mask to make it
/// stronger, which pays off in the lanes the load stops reading.
///
/// Only this direction may be applied to the mask of an existing load.
/// Weakening one would read addresses the program never read, so `simplify`
/// never edits a load and only ever redirects a use to a different load;
/// strengthening one reads a subset of what the load already read, so it can
/// neither fault nor change an observed value.
bool Propagator::narrowMasksByUse(ModuleOp mod) {
  SmallVector<tt::LoadOp> loads;
  mod.walk([&](tt::LoadOp load) { loads.push_back(load); });

  // Built once. `collectHoistable` only ever accepts definitions from the
  // load's own block, so every move below is within a block, which leaves the
  // dominator tree intact -- an intra-block query answers from the block's own
  // operation order, which MLIR maintains itself.
  DominanceInfo domInfo(mod);
  bool changed = false;
  for (tt::LoadOp load : loads) {
    Value mask = load.getMask();
    // A volatile load must be issued exactly as written; an unmasked one has no
    // mask operand to strengthen.
    if (!mask || load.getIsVolatile())
      continue;

    Facts facts;
    if (!collectUseFacts(load.getResult(), facts))
      continue;

    // The mask is already false wherever anything looks: the load contributes
    // nothing but `other`, which is an explicit zero when absent.
    if (maskExcludedByFacts(mask, facts)) {
      Value other = load.getOther();
      if (!other) {
        TypedAttr zero = rewriter.getZeroAttr(load.getType());
        if (!zero)
          continue;
        rewriter.setInsertionPoint(load);
        other = arith::ConstantOp::create(rewriter, load.getLoc(), zero);
      }
      LDBG("dropping unobserved " << *load.getOperation());
      // `collectUseFacts` only walks single-use edges, so the first of them is
      // the load's sole use and replacing the result affects nothing else.
      rewriter.replaceAllUsesWith(load.getResult(), other);
      changed = true;
      continue;
    }

    // Everything past this point ANDs a condition into the mask, which a
    // rank-2-or-higher load cannot afford. `MaterializeBlockPointer` withholds
    // `ttig.block_io` from a mask whose per-dimension constancy is not a power
    // of two of at least 2 (`maskPermitsBlockTile`), and the conditions
    // narrowing has to offer are routinely data-dependent, whose constancy is
    // 1. Losing the 2D block tile costs a hardware message per row plus a
    // shuffle per element, orders more than the lanes narrowing saves.
    //
    // Deciding this properly would need the axis info of a mask not yet built,
    // from an analysis that only runs after layout assignment, so bound the
    // rewrite by the rank instead: `maskPermitsBlockTile` is only consulted on
    // the rank >= 2 path. A rank-1 load does still get a tile, from the
    // 1D-to-2D reshape in `reshape1DStridedLoad`, but that path decides on the
    // pointer alone and never inspects the mask, so an extra data-dependent
    // conjunct cannot cost it -- and rank < 2 is where the win was measured.
    // Dropping an unobserved load above is exempt: it adds no mask arithmetic,
    // it removes the load.
    if (auto tensorTy = dyn_cast<RankedTensorType>(load.getType());
        tensorTy && tensorTy.getRank() >= 2)
      continue;

    // Otherwise AND in the conditions the mask does not already imply. A
    // positive condition the mask already implies contributes nothing, and
    // re-anding it would only add an operation for the canonicalizer to remove.
    SmallVector<Value> conjuncts;
    collectConjuncts(mask, conjuncts);
    SmallVector<std::pair<Value, bool>> missing;
    for (auto [cond, holds] : facts) {
      if (holds && maskImplies(conjuncts, cond))
        continue;
      if (llvm::is_contained(missing, std::make_pair(cond, holds)))
        continue;
      // A mask and a condition on differently shaped lanes are not the same
      // lanes; only combine what has one shape.
      if (cond.getType() != mask.getType())
        continue;
      missing.emplace_back(cond, holds);
    }
    if (missing.empty())
      continue;

    // The conditions have to be available at the load, which usually means
    // moving the pure arithmetic that computes them above it.
    SetVector<Operation *> toMove;
    if (!llvm::all_of(missing, [&](std::pair<Value, bool> lit) {
          return collectHoistable(lit.first, load, domInfo, toMove, 0);
        }))
      continue;
    for (Operation *op : toMove)
      op->moveBefore(load);

    rewriter.setInsertionPoint(load);
    Value narrowed = mask;
    for (auto [cond, holds] : missing) {
      Value literal = cond;
      if (!holds) {
        Value allOnes = arith::ConstantOp::create(
            rewriter, cond.getLoc(), boolAttr(cond.getType(), true, rewriter));
        literal = arith::XOrIOp::create(rewriter, cond.getLoc(), cond, allOnes);
      }
      narrowed =
          arith::AndIOp::create(rewriter, load.getLoc(), narrowed, literal);
    }

    LDBG("narrowing mask of " << *load.getOperation());
    rewriter.modifyOpInPlace(load,
                             [&] { load.getMaskMutable().assign(narrowed); });
    changed = true;
  }
  return changed;
}

bool Propagator::run(ModuleOp mod) {
  bool changed = false;
  SmallVector<arith::SelectOp> selects;
  mod.walk([&](arith::SelectOp op) { selects.push_back(op); });

  // Outermost select first. In SSA order the outermost consumer comes last, and
  // its arms carry the largest set of assumptions, so visiting in reverse finds
  // the most profitable rewrite before an inner select clones a subtree under a
  // narrower assumption.
  for (arith::SelectOp select : llvm::reverse(selects)) {
    for (unsigned arm : {0u, 1u}) {
      reset();
      Value armValue =
          arm == 0 ? select.getTrueValue() : select.getFalseValue();
      Facts facts;
      facts.emplace_back(select.getCondition(), arm == 0);

      rewriter.setInsertionPoint(select);
      Value simplified = simplify(armValue, facts, 0);

      // Cloning operations to remove no load is a pure loss, so only keep the
      // rewrite when it made a load redundant.
      if (simplified == armValue || loadsFolded == 0) {
        rollback();
        continue;
      }
      LDBG("rewriting arm " << arm << " of " << *select.getOperation());
      reset();
      // Operand 0 is the condition, 1 the true value, 2 the false value.
      rewriter.modifyOpInPlace(
          select, [&] { select->setOperand(arm + 1, simplified); });
      changed = true;
    }
  }

  // Second, and only second. The two directions compete for the same loads, and
  // reuse wins: it retires a send outright, where dropping an unobserved load
  // retires one that the reuse would have retired anyway. Reuse also gives the
  // load it kept an extra consumer, so running this first would both forfeit
  // the reuse and see fewer loads as unobserved.
  if (changed)
    cleanupAfterReuse(mod, rewriter);
  return narrowMasksByUse(mod) || changed;
}

struct OptimizeLoadMasksPass
    : public tt::intel::impl::TritonIntelOptimizeLoadMasksBase<
          OptimizeLoadMasksPass> {
  void runOnOperation() override {
    Propagator propagator(getOperation());
    if (!propagator.run(getOperation()))
      markAllAnalysesPreserved();
  }
};

} // namespace
