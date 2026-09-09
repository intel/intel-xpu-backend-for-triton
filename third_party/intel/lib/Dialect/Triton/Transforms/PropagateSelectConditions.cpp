#include "intel/include/Dialect/Triton/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-intel-propagate-select-conditions"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = mlir::triton;

namespace mlir::triton::intel {
#define GEN_PASS_DEF_TRITONINTELPROPAGATESELECTCONDITIONS
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

class Propagator {
public:
  Propagator(ModuleOp mod) : rewriter(mod.getContext()) {}

  /// Rewrites the select arms of \p mod. Returns true if the IR was modified.
  bool run(ModuleOp mod);

private:
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
  TypedAttr attr;
  if (auto shaped = dyn_cast<ShapedType>(type))
    attr = DenseElementsAttr::get(shaped, value);
  else
    attr = rewriter.getBoolAttr(value);
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

bool Propagator::run(ModuleOp mod) {
  SmallVector<arith::SelectOp> selects;
  mod.walk([&](arith::SelectOp op) { selects.push_back(op); });

  // Outermost select first. In SSA order the outermost consumer comes last, and
  // its arms carry the largest set of assumptions, so visiting in reverse finds
  // the most profitable rewrite before an inner select clones a subtree under a
  // narrower assumption.
  bool changed = false;
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
  return changed;
}

struct PropagateSelectConditionsPass
    : public tt::intel::impl::TritonIntelPropagateSelectConditionsBase<
          PropagateSelectConditionsPass> {
  void runOnOperation() override {
    Propagator propagator(getOperation());
    if (!propagator.run(getOperation()))
      markAllAnalysesPreserved();
  }
};

} // namespace
