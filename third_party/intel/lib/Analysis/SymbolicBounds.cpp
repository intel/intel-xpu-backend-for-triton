//===- SymbolicBounds.cpp -------------------------------------------------===//
//
// The prover declared in SymbolicBounds.h: the symbol order, overflow-checked
// affine-form arithmetic, normalization, bounding over a loop's iteration
// space, obligation discharge and guard materialization.
//
//===----------------------------------------------------------------------===//

#include "intel/include/Analysis/SymbolicBounds.h"
#include "intel/include/Analysis/Range.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir::triton::intel {

//===----------------------------------------------------------------------===//
// Symbol
//===----------------------------------------------------------------------===//

bool Symbol::operator<(const Symbol &o) const {
  if (kind_ != o.kind_)
    return kind_ < o.kind_;
  if (order_ != o.order_)
    return order_ < o.order_;
  return placement_ < o.placement_;
}

/// The prover reasons in int64_t. An integer type wider than 64 bits cannot be
/// represented there, so it is unsupported: the prover answers Unknown for it
/// and reads no range of it. `index` counts as 64 bits.
static bool isSupportedWidth(Type t) {
  if (auto intTy = dyn_cast<IntegerType>(getElementTypeOrSelf(t)))
    return intTy.getWidth() <= 64;
  return true;
}

//===----------------------------------------------------------------------===//
// AffineForm
//===----------------------------------------------------------------------===//

AffineForm AffineForm::constant(int64_t c) {
  AffineForm af;
  af.c0_ = c;
  return af;
}

AffineForm
AffineForm::makeChecked(int64_t c0,
                        SmallVector<std::pair<Symbol, int64_t>, 4> terms,
                        bool overflowed) {
  AffineForm af;
  af.c0_ = c0;
  af.terms_ = std::move(terms);
  af.overflowed_ = overflowed;
  return af;
}

AffineForm AffineForm::symbol(Symbol s) {
  assert(s.order() != 0 && "symbol built outside symbolFor");
  AffineForm af;
  af.terms_.emplace_back(std::move(s), 1);
  return af;
}

/// Adds `b` scaled by `k` into `a`, keeping terms sorted with nonzero
/// coefficients. Every arithmetic step is checked: an overflow anywhere makes
/// the result unusable rather than silently wrapping.
static AffineForm combine(const AffineForm &a, const AffineForm &b, int64_t k) {
  AffineForm out;
  bool ovf = a.overflowed() || b.overflowed();

  int64_t scaledC0;
  ovf |= llvm::MulOverflow(b.constant(), k, scaledC0);
  int64_t c0;
  ovf |= llvm::AddOverflow(a.constant(), scaledC0, c0);

  SmallVector<std::pair<Symbol, int64_t>, 4> terms;
  auto ia = a.terms().begin(), ea = a.terms().end();
  auto ib = b.terms().begin(), eb = b.terms().end();
  while (ia != ea || ib != eb) {
    if (ib == eb || (ia != ea && ia->first < ib->first)) {
      terms.push_back(*ia++);
      continue;
    }
    if (ia == ea || ib->first < ia->first) {
      int64_t coeff;
      ovf |= llvm::MulOverflow(ib->second, k, coeff);
      if (coeff != 0)
        terms.emplace_back(ib->first, coeff);
      ++ib;
      continue;
    }
    // Same symbol in both forms: collect like terms.
    int64_t scaled;
    ovf |= llvm::MulOverflow(ib->second, k, scaled);
    int64_t sum;
    ovf |= llvm::AddOverflow(ia->second, scaled, sum);
    if (sum != 0)
      terms.emplace_back(ia->first, sum);
    ++ia;
    ++ib;
  }
  return AffineForm::makeChecked(c0, std::move(terms), ovf);
}

AffineForm AffineForm::add(const AffineForm &o) const {
  return combine(*this, o, 1);
}

AffineForm AffineForm::sub(const AffineForm &o) const {
  // -1 * INT64_MIN overflows, which `combine` reports rather than wrapping.
  return combine(*this, o, -1);
}

AffineForm AffineForm::scale(int64_t k) const {
  AffineForm zero;
  return combine(zero, *this, k);
}

bool AffineForm::operator==(const AffineForm &o) const {
  return c0_ == o.c0_ && terms_.size() == o.terms_.size() &&
         std::equal(terms_.begin(), terms_.end(), o.terms_.begin(),
                    [](const auto &x, const auto &y) {
                      return x.first == y.first && x.second == y.second;
                    });
}

//===----------------------------------------------------------------------===//
// Rendering
//===----------------------------------------------------------------------===//

/// A stable name for a symbol: `argN` for a function argument, the `loc` name
/// when the defining operation has one (suffixed `#<result>` when it has
/// several results), else the operation name.
static std::string symbolName(const Symbol &s) {
  Value v = s.value();
  if (!v)
    return "<synthetic>";
  if (auto blockArg = dyn_cast<BlockArgument>(v)) {
    if (isa_and_nonnull<tt::FuncOp>(blockArg.getOwner()->getParentOp()))
      return ("arg" + Twine(blockArg.getArgNumber())).str();
    return "blockarg";
  }
  Operation *def = v.getDefiningOp();
  std::string base;
  if (auto nameLoc = dyn_cast<NameLoc>(def->getLoc()))
    base = nameLoc.getName().str();
  else
    base = def->getName().getStringRef().str();
  if (def->getNumResults() > 1)
    base += "#" + std::to_string(cast<OpResult>(v).getResultNumber());
  return base;
}

static std::string renderSymbol(const Symbol &s) {
  std::string name = symbolName(s);
  if (s.kind() == SymbolKind::Opaque)
    return "opaque(" + name + ")";
  return name;
}

std::string toString(const AffineForm &af) {
  if (af.overflowed())
    return "<overflowed>";
  std::string out;
  llvm::raw_string_ostream os(out);
  bool first = true;
  for (auto &[sym, coeff] : af.terms()) {
    if (first) {
      if (coeff < 0)
        os << "-";
    } else {
      os << (coeff < 0 ? " - " : " + ");
    }
    // Unsigned: the magnitude of INT64_MIN is not an int64_t.
    uint64_t mag = coeff < 0 ? 0 - static_cast<uint64_t>(coeff)
                             : static_cast<uint64_t>(coeff);
    if (mag != 1)
      os << mag << "*";
    os << renderSymbol(sym);
    first = false;
  }
  int64_t c0 = af.constant();
  if (first)
    os << c0;
  else if (c0 > 0)
    os << " + " << c0;
  else if (c0 < 0)
    os << " - " << 0 - static_cast<uint64_t>(c0);
  return out;
}

/// True when the terms of `e`, its constant aside, have magnitude at most 2^63
/// for every symbol value: no term, or one +-1 term over at most 64 bits.
static bool symbolicPartWithin2Pow63(const AffineForm &e) {
  if (e.isConstant())
    return true;
  if (e.numTerms() != 1)
    return false;
  auto &[sym, k] = e.terms().front();
  if ((k != 1 && k != -1) || !sym.value())
    return false;
  Type t = getElementTypeOrSelf(sym.value());
  if (auto intTy = dyn_cast<IntegerType>(t))
    return intTy.getWidth() <= 64;
  return isa<IndexType>(t);
}

/// Normalizes a condition so its subject is as simple as the goal allows: the
/// constant term folds into the bound, and a single-symbol condition is
/// divided by |k| with the bound rounded inward, swapping AtLeast/AtMost when
/// k < 0. Written `-N >= 1` the checked negation of
/// INT64_MIN overflows and rejects a launch the condition admits; written
/// `N <= -1` it does not.
bool normalizeCondition(BoundCondition &cond) {
  if (cond.expr.overflowed())
    return false;

  // Fold the constant into the bound, keeping the rendering of a bare
  // `s >= 0` unchanged.
  int64_t c0 = cond.expr.constant();
  int64_t bound = cond.goal == BoundGoal::NonNegative ? 0 : cond.c;
  bool atMost = cond.goal == BoundGoal::AtMost;
  if (c0 != 0) {
    int64_t folded;
    if (llvm::SubOverflow(bound, c0, folded)) {
      // Folding would need a bound beyond i64. A symbolic part within 2^63
      // cannot pass it, so `(x - 1) <= INT64_MAX` is vacuous; a compound one
      // such as `a - b` can.
      bool vacuous =
          (atMost ? c0 < 0 : c0 > 0) && symbolicPartWithin2Pow63(cond.expr);
      if (!vacuous)
        return false; // the opposite direction, or a part that can pass it
      cond.expr = AffineForm::constant(0);
      cond.c = 0;
      return true;
    }
    bound = folded;
    cond.expr = cond.expr.sub(AffineForm::constant(c0));
    cond.goal = atMost ? BoundGoal::AtMost : BoundGoal::AtLeast;
    cond.c = bound;
    if (cond.expr.overflowed())
      return false;
  }

  // Single symbol with |k| != 1: divide, rounding the bound inward.
  if (cond.expr.numTerms() == 1 && cond.expr.constant() == 0) {
    auto [sym, k] = cond.expr.terms().front();
    if (k == 0)
      return false;
    if (k != 1) {
      if (k == INT64_MIN)
        return false; // |INT64_MIN| is not representable
      int64_t mag = k < 0 ? -k : k;
      bool wantAtMost = atMost;
      if (k < 0)
        wantAtMost = !wantAtMost; // dividing by a negative swaps the relation
      // Round inward so the divided condition is no weaker than the original.
      int64_t num =
          cond.goal == BoundGoal::AtMost || cond.goal == BoundGoal::AtLeast
              ? cond.c
              : bound;
      int64_t q;
      if (k < 0 && num == INT64_MIN) {
        // -num is 2^63: divide in unsigned. Only mag == 1 leaves the int64_t
        // range, where x <= 2^63 holds for any value and x >= 2^63 for none.
        uint64_t two63 = uint64_t(1) << 63, um = static_cast<uint64_t>(mag);
        uint64_t uq = two63 / um;
        if (!wantAtMost && uq * um != two63)
          ++uq;
        if (uq > static_cast<uint64_t>(INT64_MAX)) {
          if (!wantAtMost)
            return false;
          uq = INT64_MAX;
        }
        q = static_cast<int64_t>(uq);
      } else {
        int64_t div = k < 0 ? -num : num;
        q = wantAtMost ? llvm::divideFloorSigned(div, mag)
                       : llvm::divideCeilSigned(div, mag);
      }
      cond.expr = AffineForm::symbol(sym);
      cond.goal = wantAtMost ? BoundGoal::AtMost : BoundGoal::AtLeast;
      cond.c = q;
    }
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Guard materialization
//===----------------------------------------------------------------------===//

namespace {

/// Builds one guard expression in i64, either with plain arithmetic (when the
/// static fit check proves it cannot overflow) or with an overflow predicate
/// per step. `ok` accumulates the predicates of the checked path; it stays
/// null on the fast path.
class GuardBuilder {
public:
  GuardBuilder(OpBuilder &b, Location loc, bool checked)
      : b(b), loc(loc), checked(checked) {}

  Value constant(int64_t v) {
    return arith::ConstantIntOp::create(b, loc, b.getI64Type(), v);
  }

  /// Sign-extends a narrow symbol to i64. Sign extension is what makes the
  /// i64 arithmetic agree with the narrow value the hardware computes.
  Value extend(Value v) {
    Type i64 = b.getI64Type();
    if (v.getType() == i64)
      return v;
    if (isa<IndexType>(v.getType()))
      return arith::IndexCastOp::create(b, loc, i64, v);
    return arith::ExtSIOp::create(b, loc, i64, v);
  }

  Value mul(Value lhs, int64_t k) {
    if (k == 1)
      return lhs;
    Value rhs = constant(k);
    if (!checked)
      return arith::MulIOp::create(b, loc, lhs, rhs);
    // mulsi_extended gives the full 128-bit product as (low, high); the
    // multiply fits i64 exactly when high is the sign extension of low. This
    // handles INT64_MIN without taking a magnitude.
    auto ext = arith::MulSIExtendedOp::create(b, loc, lhs, rhs);
    Value low = ext.getLow(), high = ext.getHigh();
    Value signBits = arith::ShRSIOp::create(b, loc, low, constant(63));
    addOk(arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, high,
                                signBits));
    return low;
  }

  Value add(Value lhs, Value rhs) {
    Value sum = arith::AddIOp::create(b, loc, lhs, rhs);
    if (checked) {
      // Signed add overflows exactly when both operands differ in sign from
      // the result: ((a ^ r) & (b ^ r)) < 0.
      Value xa = arith::XOrIOp::create(b, loc, lhs, sum);
      Value xb = arith::XOrIOp::create(b, loc, rhs, sum);
      Value both = arith::AndIOp::create(b, loc, xa, xb);
      addOk(arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sge, both,
                                  constant(0)));
    }
    return sum;
  }

  /// Conjoins the overflow predicates with `cmp`. An overflow makes the guard
  /// false, which forfeits the fast path for that launch and never unmasks.
  Value finish(Value cmp) const { return ok ? andAll(cmp) : cmp; }

private:
  void addOk(Value pred) {
    ok = ok ? arith::AndIOp::create(b, loc, ok, pred) : pred;
  }
  Value andAll(Value cmp) const {
    return arith::AndIOp::create(b, loc, ok, cmp);
  }

  OpBuilder &b;
  Location loc;
  bool checked;
  Value ok;
};

/// The static fit check: `|c0| + sum(|ci| * 2^(w_i - 1)) < 2^63`, with
/// saturating 128-bit accumulation that stops at the first partial sum
/// reaching 2^63. Saturation matters: four i64 symbols with coefficient
/// INT64_MIN sum to exactly 2^128, which a fixed-width 128-bit accumulator
/// would wrap to zero and wrongly admit as "fits".
bool guardFitsPlainI64(const AffineForm &e) {
  const unsigned kW = 128;
  APInt limit = APInt::getOneBitSet(kW, 63); // 2^63
  APInt acc(kW, 0);
  auto absToAP = [&](int64_t v) {
    // INT64_MIN has no positive counterpart in int64_t; widen first.
    APInt a(kW, static_cast<uint64_t>(v), /*isSigned=*/true);
    return a.isNegative() ? APInt(kW, 0) - a : a;
  };
  acc = acc.uadd_sat(absToAP(e.constant()));
  if (acc.uge(limit))
    return false;
  for (auto &[sym, k] : e.terms()) {
    unsigned w = 64;
    if (sym.value())
      if (auto intTy = dyn_cast<IntegerType>(getElementTypeOrSelf(sym.value())))
        w = intTy.getWidth();
    APInt magnitude = APInt::getOneBitSet(kW, w - 1); // 2^(w-1)
    acc = acc.uadd_sat(absToAP(k).umul_sat(magnitude));
    if (acc.uge(limit))
      return false;
  }
  return acc.ult(limit);
}

} // namespace

Value materialize(ArrayRef<BoundCondition> conds, Operation *before,
                  OpBuilder &builder) {
  assert(before && "need an insertion anchor");
  // Place the guard before `before` whatever the caller's insertion point is,
  // and leave that insertion point as it was.
  OpBuilder::InsertionGuard insertionGuard(builder);
  builder.setInsertionPoint(before);
  Location loc = before->getLoc();
  DominanceInfo domInfo(before->getParentOp());
  Value result;

  for (const BoundCondition &cond : conds) {
    // The guard is built in i64, and `materialize` has no way to decline, so a
    // subject wider than 64 bits is a precondition violation. Checked before
    // `guardFitsPlainI64`, which sizes its arithmetic from the subject's width.
    assert(llvm::all_of(cond.expr.terms(),
                        [](const auto &term) {
                          Value v = term.first.value();
                          return !v || isSupportedWidth(v.getType());
                        }) &&
           "condition subject wider than 64 bits");
    bool checked = !guardFitsPlainI64(cond.expr);
    GuardBuilder gb(builder, loc, checked);

    Value sum = gb.constant(cond.expr.constant());
    for (auto &[sym, k] : cond.expr.terms()) {
      Value v = sym.value();
      assert(v && "condition subject has no SSA value");
      assert(!isa<ShapedType>(v.getType()) &&
             "condition subject must be a scalar");
      assert(domInfo.properlyDominates(v, before) &&
             "condition subject must dominate the guard");
      sum = gb.add(sum, gb.mul(gb.extend(v), k));
    }

    Value cmp;
    switch (cond.goal) {
    case BoundGoal::NonNegative:
      cmp = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sge, sum,
                                  gb.constant(0));
      break;
    case BoundGoal::AtLeast:
      cmp = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sge, sum,
                                  gb.constant(cond.c));
      break;
    case BoundGoal::AtMost:
      cmp = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sle, sum,
                                  gb.constant(cond.c));
      break;
    }
    Value guard = gb.finish(cmp);
    result =
        result ? arith::AndIOp::create(builder, loc, result, guard).getResult()
               : guard;
  }
  return result;
}

std::string toString(const BoundCondition &c) {
  std::string expr = toString(c.expr);
  switch (c.goal) {
  case BoundGoal::NonNegative:
    return expr + " >= 0";
  case BoundGoal::AtLeast:
    return expr + " >= " + std::to_string(c.c);
  case BoundGoal::AtMost:
    return expr + " <= " + std::to_string(c.c);
  }
  llvm_unreachable("unhandled goal");
}

std::string toString(const BoundProof &p) {
  switch (p.verdict) {
  case BoundProof::Satisfied:
    return "Satisfied";
  case BoundProof::Refuted:
    return "Refuted";
  case BoundProof::Unknown:
    return "Unknown";
  case BoundProof::ConditionallySatisfied:
    break;
  }
  std::string out = "Conditional{";
  for (auto [i, c] : llvm::enumerate(p.conditions)) {
    if (i)
      out += "; ";
    out += toString(c);
  }
  return out + "}";
}

//===----------------------------------------------------------------------===//
// SymbolicBoundsProver
//===----------------------------------------------------------------------===//

/// The operation whose values the prover numbers: `root` if it is a function,
/// else its enclosing function, else `root`. A module has no enclosing
/// function, so a module root numbers exactly what it always did.
static Operation *numberingScope(Operation *root) {
  if (isa<tt::FuncOp>(root))
    return root;
  if (auto func = root->getParentOfType<tt::FuncOp>())
    return func.getOperation();
  return root;
}

SymbolicBoundsProver::SymbolicBoundsProver(const DataFlowSolver &solver,
                                           Operation *root)
    : solver(solver), scope(numberingScope(root)) {
  // Number every value in pre-order, from 1, so the symbol sort is total over
  // distinct values and reproducible across processes: a block's arguments as
  // the walk enters it, then each operation's results in result order. 0 is
  // reserved as "unassigned", which AffineForm::symbol asserts against.
  unsigned next = 1;
  scope->walk<WalkOrder::PreOrder>([&](Operation *op) {
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument arg : block.getArguments())
          valueOrder.try_emplace(arg, next++);
    for (OpResult result : op->getResults())
      valueOrder.try_emplace(result, next++);
  });
  nextOrder = next;
}

Symbol SymbolicBoundsProver::symbolFor(SymbolKind kind, Value v,
                                       AxisPlacement placement) const {
  // A value outside the numbering - outside `scope`, or created after
  // construction - takes the next unused order the first time it is seen and
  // keeps it. Sharing one fallback key would make two such values tie in the
  // sort order, and `combine` would then cancel them as like terms.
  auto [it, inserted] = valueOrder.try_emplace(v, nextOrder);
  if (inserted)
    ++nextOrder;
  return Symbol(kind, v, std::move(placement), it->second);
}

AffineForm SymbolicBoundsProver::opaque(Value v,
                                        AxisPlacement placement) const {
  return AffineForm::symbol(
      symbolFor(SymbolKind::Opaque, v, std::move(placement)));
}

void SymbolicBoundsProver::recordWrap(
    Operation *op, const AffineForm &result,
    SmallVectorImpl<Obligation> &obligations) const {
  unsigned width = 0;
  if (auto intTy =
          dyn_cast<IntegerType>(getElementTypeOrSelf(op->getResult(0))))
    width = intTy.getWidth();
  obligations.push_back({Obligation::Wrap, op, result, width});
}

/// The constant an operation folds to, if any: a scalar `arith.constant` or a
/// splat dense attribute, as RemoveMasks' own `getIntConstantValue` accepts.
static std::optional<int64_t> getFoldedConstant(Value v) {
  APInt intVal;
  if (matchPattern(v, m_ConstantInt(&intVal)))
    return intVal.trySExtValue();
  DenseElementsAttr constAttr;
  if (matchPattern(v, m_Constant(&constAttr)) && constAttr.isSplat()) {
    auto attr = constAttr.getSplatValue<Attribute>();
    if (auto intAttr = dyn_cast_or_null<IntegerAttr>(attr))
      return intAttr.getValue().trySExtValue();
  }
  return std::nullopt;
}

static unsigned bitWidth(Type t) {
  Type elem = getElementTypeOrSelf(t);
  if (auto intTy = dyn_cast<IntegerType>(elem))
    return intTy.getWidth();
  // `index` is modelled as 64 bits, as the guard arithmetic does.
  return 64;
}

AffineForm
SymbolicBoundsProver::normalize(Value v, QueryContext ctx,
                                SmallVectorImpl<Obligation> &obligations) {
  exhausted = false;
  return normalizeImpl(v, ctx, obligations, identityPlacement(v), 0);
}

AxisPlacement SymbolicBoundsProver::identityPlacement(Value v) {
  AxisPlacement identity;
  if (auto shaped = dyn_cast<ShapedType>(v.getType()))
    for (int64_t i = 0, e = shaped.getRank(); i < e; ++i)
      identity.push_back(i);
  return identity;
}

AffineForm SymbolicBoundsProver::leafForBlockArg(BlockArgument arg,
                                                 AxisPlacement placement) {
  Operation *owner = arg.getOwner()->getParentOp();
  if (isa_and_nonnull<tt::FuncOp>(owner))
    return AffineForm::symbol(symbolFor(SymbolKind::KernelArg, arg, placement));
  return opaque(arg, placement);
}

AffineForm
SymbolicBoundsProver::normalizeImpl(Value v, QueryContext ctx,
                                    SmallVectorImpl<Obligation> &obligations,
                                    AxisPlacement placement, unsigned depth) {
  MemoKey key{v.getAsOpaquePointer(),
              ctx.loop ? ctx.loop.getOperation() : nullptr, placement};
  auto it = memo.find(key);
  if (it != memo.end()) {
    const MemoEntry &e = it->second;
    // A hit must reproduce everything the computation would have contributed,
    // not just the form: obligations the caller has to discharge, the budget
    // flag, and the depth the subtree would have reached from here.
    llvm::append_range(obligations, e.obligations);
    if (e.exhausted || depth + e.height > kMaxDepth)
      exhausted = true;
    deepest = std::max(deepest, depth + e.height);
    return e.af;
  }

  size_t mark = obligations.size();
  unsigned savedDeepest = deepest;
  bool savedExhausted = exhausted;
  // `deepest` and `exhausted` are measured for this subtree alone, then folded
  // back into the enclosing query, which keeps `exhausted` sticky.
  deepest = depth;
  exhausted = false;
  AffineForm af = normalizeUncached(v, ctx, obligations, placement, depth);

  MemoEntry e;
  e.af = af;
  e.obligations.assign(obligations.begin() + mark, obligations.end());
  e.exhausted = exhausted;
  e.height = deepest - depth;
  memo.try_emplace(key, std::move(e));

  exhausted = savedExhausted || exhausted;
  deepest = std::max(savedDeepest, deepest);
  return af;
}

AffineForm SymbolicBoundsProver::normalizeUncached(
    Value v, QueryContext ctx, SmallVectorImpl<Obligation> &obligations,
    AxisPlacement placement, unsigned depth) {
  deepest = std::max(deepest, depth);
  if (depth > kMaxDepth) {
    // Budgets degrade the whole query to Unknown.
    exhausted = true;
    return opaque(v, placement);
  }
  if (!isSupportedWidth(v.getType())) {
    // Outside the int64_t domain nothing about the value can be represented,
    // so it is opaque and the query is Unknown, as for an exhausted budget.
    exhausted = true;
    return opaque(v, placement);
  }
  if (std::optional<int64_t> cst = getFoldedConstant(v))
    return AffineForm::constant(*cst);

  Operation *def = v.getDefiningOp();
  if (!def)
    return leafForBlockArg(cast<BlockArgument>(v), placement);

  // Obligations of an expression the prover stops looking through are dropped.
  size_t mark = obligations.size();
  auto giveUp = [&]() -> AffineForm {
    obligations.truncate(mark);
    return opaque(v, placement);
  };

  return llvm::TypeSwitch<Operation *, AffineForm>(def)
      .Case<arith::AddIOp, arith::SubIOp>([&](auto op) {
        AffineForm l =
            normalizeImpl(op.getLhs(), ctx, obligations, placement, depth + 1);
        AffineForm r =
            normalizeImpl(op.getRhs(), ctx, obligations, placement, depth + 1);
        AffineForm res = isa<arith::AddIOp>(def) ? l.add(r) : l.sub(r);
        if (res.numTerms() > kMaxTerms)
          exhausted = true;
        if (res.overflowed() || exhausted)
          return giveUp();
        recordWrap(def, res, obligations);
        return res;
      })
      .Case<arith::MulIOp>([&](auto op) {
        std::optional<int64_t> k = getFoldedConstant(op.getRhs());
        Value other = op.getLhs();
        if (!k) {
          k = getFoldedConstant(op.getLhs());
          other = op.getRhs();
        }
        if (!k)
          return giveUp(); // product of two symbols: no derived facts
        AffineForm res =
            normalizeImpl(other, ctx, obligations, placement, depth + 1)
                .scale(*k);
        if (res.overflowed() || exhausted)
          return giveUp();
        recordWrap(def, res, obligations);
        return res;
      })
      .Case<arith::ExtSIOp>([&](auto op) {
        return normalizeImpl(op.getIn(), ctx, obligations, placement,
                             depth + 1);
      })
      .Case<arith::IndexCastOp>([&](auto op) {
        // Widening only: a narrowing cast truncates.
        if (bitWidth(op.getIn().getType()) > bitWidth(op.getType()))
          return giveUp();
        return normalizeImpl(op.getIn(), ctx, obligations, placement,
                             depth + 1);
      })
      .Case<arith::ExtUIOp>([&](auto op) {
        // Equal to extsi exactly when the operand is non-negative.
        AffineForm in =
            normalizeImpl(op.getIn(), ctx, obligations, placement, depth + 1);
        if (in.overflowed() || exhausted)
          return giveUp();
        obligations.push_back({Obligation::NonNegative, def, in, 0});
        return in;
      })
      .Case<tt::MakeRangeOp>([&](auto op) {
        return AffineForm::symbol(
            symbolFor(SymbolKind::Lane, op.getResult(), placement));
      })
      .Case<tt::SplatOp>([&](auto op) {
        // A scalar has the same value in every element, so the placement of
        // the result says nothing about the operand.
        return normalizeImpl(op.getSrc(), ctx, obligations, {}, depth + 1);
      })
      .Case<tt::ExpandDimsOp>([&](auto op) {
        // The operand's axis i lands on result axis i for i < axis, else
        // i + 1; expressed by deleting the inserted axis from the
        // result-indexed placement.
        int32_t axis = op.getAxis();
        if (axis >= static_cast<int32_t>(placement.size()))
          return giveUp();
        AxisPlacement inner(placement);
        inner.erase(inner.begin() + axis);
        return normalizeImpl(op.getSrc(), ctx, obligations, inner, depth + 1);
      })
      .Case<tt::BroadcastOp>([&](auto op) {
        // Operand and result have equal rank; keeping the entries of size-1
        // operand axes is what lets a later expand_dims delete the right one.
        return normalizeImpl(op.getSrc(), ctx, obligations, placement,
                             depth + 1);
      })
      .Case<ttg::ConvertLayoutOp>([&](auto op) {
        // A layout change moves no element.
        return normalizeImpl(op.getSrc(), ctx, obligations, placement,
                             depth + 1);
      })
      .Case<tt::GetProgramIdOp>([&](auto op) {
        return AffineForm::symbol(
            symbolFor(SymbolKind::ProgramId, op.getResult(), placement));
      })
      .Case<tt::GetNumProgramsOp>([&](auto op) {
        return AffineForm::symbol(
            symbolFor(SymbolKind::NumPrograms, op.getResult(), placement));
      })
      // tt.reshape, tt.trans, tt.join and tt.split permute or merge elements;
      // axis tracking through them is not supported.
      .Default([&](Operation *) { return giveUp(); });
}

//===----------------------------------------------------------------------===//
// Bounding and decision
//===----------------------------------------------------------------------===//

/// Constant bounds of one symbol. `Lane` is exact; anything else falls back
/// to the range analysis, and a symbol with no inferable range is unbounded,
/// which makes the query `Unknown`.
std::optional<std::pair<int64_t, int64_t>>
SymbolicBoundsProver::symbolConstantBounds(const Symbol &sym) const {
  if (sym.kind() == SymbolKind::Lane) {
    auto rangeOp = cast<tt::MakeRangeOp>(sym.value().getDefiningOp());
    // Signed attribute getters: the generated getStart()/getEnd() return
    // uint32_t although the attributes are signed, so a range [-4, 0) would
    // otherwise bound as positive.
    int64_t start = rangeOp.getStartAttr().getInt();
    int64_t end = rangeOp.getEndAttr().getInt();
    return std::make_pair(start, end - 1);
  }
  return rangeOf(sym.value());
}

/// Bounds `e` by replacing every symbol with its constant bounds, taking the
/// low or high end per the sign of the coefficient.
std::optional<std::pair<int64_t, int64_t>>
SymbolicBoundsProver::boundConstant(const AffineForm &e) const {
  if (e.overflowed())
    return std::nullopt;
  int64_t lo = e.constant(), hi = e.constant();
  for (auto &[sym, k] : e.terms()) {
    std::optional<std::pair<int64_t, int64_t>> b = symbolConstantBounds(sym);
    if (!b)
      return std::nullopt;
    int64_t t1, t2, l, h;
    if (llvm::MulOverflow(k > 0 ? b->first : b->second, k, t1) ||
        llvm::MulOverflow(k > 0 ? b->second : b->first, k, t2) ||
        llvm::AddOverflow(lo, t1, l) || llvm::AddOverflow(hi, t2, h))
      return std::nullopt;
    lo = l;
    hi = h;
  }
  return std::make_pair(lo, hi);
}

//===----------------------------------------------------------------------===//
// Bounding over a loop's iteration space
//===----------------------------------------------------------------------===//

SymbolicBoundsProver::Bounds
SymbolicBoundsProver::symbolBounds(const Symbol &sym, QueryContext ctx) {
  Bounds out;
  if (sym.kind() == SymbolKind::Lane) {
    auto rangeOp = cast<tt::MakeRangeOp>(sym.value().getDefiningOp());
    // Signed attribute getters: getStart()/getEnd() return uint32_t although
    // the attributes are signed, so [-4, 0) would bound as positive.
    int64_t start = rangeOp.getStartAttr().getInt();
    int64_t end = rangeOp.getEndAttr().getInt();
    out.lo = AffineForm::constant(start);
    out.hi = AffineForm::constant(end - 1);
    out.isVarying = true;
    return out;
  }
  // Loop-invariant symbols stay symbolic; a symbol defined
  // inside the loop is bounded from its constant range, or is unbounded.
  if (ctx.loop && sym.value() &&
      ctx.loop->isAncestor(sym.value().getParentBlock()->getParentOp())) {
    out.isVarying = true;
    if (std::optional<std::pair<int64_t, int64_t>> b =
            symbolConstantBounds(sym)) {
      out.lo = AffineForm::constant(b->first);
      out.hi = AffineForm::constant(b->second);
    } else {
      out.finite = false;
    }
  }
  return out;
}

SymbolicBoundsProver::Bounds SymbolicBoundsProver::bound(const AffineForm &e,
                                                         QueryContext ctx) {
  Bounds out{AffineForm::constant(e.constant()),
             AffineForm::constant(e.constant())};
  for (auto &[sym, k] : e.terms()) {
    Bounds sb = symbolBounds(sym, ctx); // loop-varying only
    out.finite &= sb.finite;
    if (!sb.isVarying) { // keep loop-invariant symbols symbolic
      out.lo = out.lo.add(AffineForm::symbol(sym).scale(k));
      out.hi = out.hi.add(AffineForm::symbol(sym).scale(k));
      continue;
    }
    out.lo = out.lo.add((k > 0 ? sb.lo : sb.hi).scale(k));
    out.hi = out.hi.add((k > 0 ? sb.hi : sb.lo).scale(k));
  }
  // AffineForm::add already collects like terms.
  out.exhausted = e.overflowed() || out.lo.overflowed() ||
                  out.hi.overflowed() || out.lo.numTerms() > kMaxTerms ||
                  out.hi.numTerms() > kMaxTerms;
  return out;
}

bool SymbolicBoundsProver::termSignOk(const Symbol &sym, int64_t k,
                                      QueryContext ctx) {
  std::optional<std::pair<int64_t, int64_t>> b = sym.kind() == SymbolKind::Lane
                                                     ? symbolConstantBounds(sym)
                                                     : rangeOf(sym.value());
  if (!b)
    return false;
  return k > 0 ? b->first >= 0 : b->second <= 0;
}

bool SymbolicBoundsProver::decideResidual(const AffineForm &lo, int64_t g,
                                          QueryContext ctx) {
  if (lo.overflowed())
    return false;
  if (lo.isConstant())
    return lo.constant() >= g;
  // A term whose sign is safe cannot make the residual smaller, so it is
  // dropped at zero and the constant term alone is compared against `g`.
  for (auto &[sym, k] : lo.terms())
    if (!termSignOk(sym, k, ctx))
      return false;
  return lo.constant() >= g;
}

/// The constant range of `v` from the range analysis.
std::optional<std::pair<int64_t, int64_t>>
SymbolicBoundsProver::rangeOf(Value v) const {
  if (!isSupportedWidth(v.getType()))
    return std::nullopt;
  std::optional<ConstantIntRanges> r = collectRange(solver, v);
  if (!r)
    return std::nullopt;
  return std::make_pair(r->smin().getSExtValue(), r->smax().getSExtValue());
}

//===----------------------------------------------------------------------===//
// Obligation discharge and guard materialization
//===----------------------------------------------------------------------===//

bool SymbolicBoundsProver::dischargeTier1(const Obligation &o, QueryContext ctx,
                                          bool &unknown) {
  Bounds b = bound(o.expr, ctx);
  if (b.exhausted) {
    // Overflowed endpoints are not bounds: the obligation can be neither
    // discharged nor guarded, so the query ends Unknown.
    unknown = true;
    return false;
  }
  if (!b.finite)
    return false;

  if (o.kind == Obligation::NonNegative)
    return decideResidual(b.lo, 0, ctx);

  // Tier 1 derives bounds from the OPERANDS, never from the range analysis's
  // range for the result: inferAdd intersects the unsigned and signed ranges,
  // so for i8 x in [100, 110] the result range of x + 100 is the narrow
  // [-56, -46] - a correct description of the wrapped value and useless as a
  // no-wrap proof.
  std::optional<std::pair<int64_t, int64_t>> lo = boundConstant(b.lo);
  std::optional<std::pair<int64_t, int64_t>> hi = boundConstant(b.hi);
  if (!lo || !hi)
    return false;
  int64_t width = o.width ? o.width : 64;
  if (width > 64)
    return false; // the limits of such a type are not int64_t values
  return hi->second <= APInt::getSignedMaxValue(width).getSExtValue() &&
         lo->first >= APInt::getSignedMinValue(width).getSExtValue();
}

/// The guards that close an open wrap obligation: the result must fit its
/// width at both ends.
void SymbolicBoundsProver::guardsForObligation(
    const Obligation &o, QueryContext ctx, SmallVectorImpl<BoundCondition> &out,
    bool &unknown) {
  Bounds b = bound(o.expr, ctx);
  if (!b.finite || b.exhausted) {
    // lo/hi are not bounds (a non-finite symbol is left at zero in them), so
    // no guard closes the obligation: the query ends Unknown.
    unknown = true;
    return;
  }

  auto push = [&](AffineForm e, BoundGoal goal, int64_t c) {
    BoundCondition cond{std::move(e), goal, c};
    if (!normalizeCondition(cond)) {
      // Not expressible: leave the obligation open, which finalize reads as
      // a verdict of Unknown rather than a silently dropped guard.
      unknown = true;
      return;
    }
    out.push_back(std::move(cond));
  };

  if (o.kind == Obligation::NonNegative) {
    push(b.lo, BoundGoal::NonNegative, 0);
    return;
  }
  int64_t width = o.width ? o.width : 64;
  if (width > 64) {
    unknown = true; // no int64_t guard can state this type's limits
    return;
  }
  push(b.hi, BoundGoal::AtMost, APInt::getSignedMaxValue(width).getSExtValue());
  push(b.lo, BoundGoal::AtLeast,
       APInt::getSignedMinValue(width).getSExtValue());
}

/// True when the symbols' constant ranges alone already imply `cond`, so it
/// would be a guard that is true on every launch.
bool SymbolicBoundsProver::impliedByRanges(const BoundCondition &cond) const {
  std::optional<std::pair<int64_t, int64_t>> b = boundConstant(cond.expr);
  if (!b)
    return false;
  switch (cond.goal) {
  case BoundGoal::NonNegative:
    return b->first >= 0;
  case BoundGoal::AtLeast:
    return b->first >= cond.c;
  case BoundGoal::AtMost:
    return b->second <= cond.c;
  }
  return false;
}

BoundProof SymbolicBoundsProver::finalize(BoundProof::Verdict onD,
                                          ArrayRef<Obligation> obligations,
                                          QueryContext ctx) {
  // Close every obligation at tier 1, else as a runtime guard.
  SmallVector<BoundCondition, 4> guards;
  SmallVector<Obligation, 8> seen;
  bool unknown = false;
  for (const Obligation &o : obligations) {
    if (llvm::is_contained(seen, o))
      continue;
    seen.push_back(o);
    if (dischargeTier1(o, ctx, unknown))
      continue;
    if (unknown)
      return {};
    guardsForObligation(o, ctx, guards, unknown);
    if (guards.size() > kMaxGuards)
      return {};
  }
  if (unknown)
    return {};

  BoundProof proof;
  // A guard is dropped when unconditional range evidence implies it.
  auto emit = [&](const BoundCondition &c) {
    if (c.expr.isConstant()) {
      // A condition over no symbol is decided now: dropped when true, and it
      // makes the verdict Unknown when false.
      bool holds = false;
      switch (c.goal) {
      case BoundGoal::NonNegative:
        holds = c.expr.constant() >= 0;
        break;
      case BoundGoal::AtLeast:
        holds = c.expr.constant() >= c.c;
        break;
      case BoundGoal::AtMost:
        holds = c.expr.constant() <= c.c;
        break;
      }
      return holds;
    }
    if (impliedByRanges(c))
      return true;
    if (!llvm::is_contained(proof.conditions, c))
      proof.conditions.push_back(c);
    return true;
  };

  bool feasible = true;
  for (const BoundCondition &c : guards)
    feasible &= emit(c);
  if (!feasible)
    return {};

  // Every emitted condition must have a scalar subject that dominates the
  // loop, or no guard can be placed - as when extui and extsi of one tensor
  // cancel in d.
  for (const BoundCondition &c : proof.conditions)
    for (auto &[sym, k] : c.expr.terms())
      if (!sym.value() || isa<ShapedType>(sym.value().getType()))
        return {};

  if (onD == BoundProof::Refuted) {
    // Refuted is only ever unconditional.
    if (!proof.conditions.empty())
      return {};
    proof.verdict = BoundProof::Refuted;
    return proof;
  }
  proof.verdict = proof.conditions.empty() ? BoundProof::Satisfied
                                           : BoundProof::ConditionallySatisfied;
  return proof;
}

BoundProof SymbolicBoundsProver::prove(arith::CmpIPredicate pred, Value lhs,
                                       Value rhs, QueryContext ctx) {
  SmallVector<Obligation, 4> obligations;
  exhausted = false;
  AffineForm l =
      normalizeImpl(lhs, ctx, obligations, identityPlacement(lhs), 0);
  AffineForm r =
      normalizeImpl(rhs, ctx, obligations, identityPlacement(rhs), 0);
  if (exhausted || l.overflowed() || r.overflowed())
    return {};

  // Reduce the predicate to `d >= g`. eq/ne are declined: the predicate
  // whitelist in RemoveMasks exists because eq produced unsound versioning
  // conditions (#7791). Unsigned predicates additionally need both sides
  // non-negative, which is added as obligations below.
  int64_t g;
  AffineForm d;
  switch (pred) {
  case arith::CmpIPredicate::slt:
  case arith::CmpIPredicate::ult:
    d = r.sub(l);
    g = 1;
    break;
  case arith::CmpIPredicate::sle:
  case arith::CmpIPredicate::ule:
    d = r.sub(l);
    g = 0;
    break;
  case arith::CmpIPredicate::sgt:
  case arith::CmpIPredicate::ugt:
    d = l.sub(r);
    g = 1;
    break;
  case arith::CmpIPredicate::sge:
  case arith::CmpIPredicate::uge:
    d = l.sub(r);
    g = 0;
    break;
  default:
    return {};
  }
  // An overflowed difference carries no information: for i64 operands
  // INT64_MAX < INT64_MIN would wrap to d = 1 and read as a proof.
  if (d.overflowed())
    return {};

  // The reduction above decides `d >= g` with signed arithmetic; an unsigned
  // predicate agrees with it only when both operands are non-negative, so
  // that case is undecided (not merely unproven) without this obligation:
  // `x ult 100` reduced to `100 - x >= 1` holds at the bit pattern of x = -1,
  // where the unsigned value 255 is not less than 100.
  switch (pred) {
  case arith::CmpIPredicate::ult:
  case arith::CmpIPredicate::ule:
  case arith::CmpIPredicate::ugt:
  case arith::CmpIPredicate::uge:
    obligations.push_back({Obligation::NonNegative, ctx.at, l, 0});
    obligations.push_back({Obligation::NonNegative, ctx.at, r, 0});
    break;
  default:
    break;
  }

  Bounds b = bound(d, ctx);
  if (!b.finite || b.exhausted)
    return {};
  if (decideResidual(b.lo, g, ctx))
    return finalize(BoundProof::Satisfied, obligations, ctx);
  // Only a constant hi refutes.
  if (b.hi.isConstant() && b.hi.constant() < g)
    return finalize(BoundProof::Refuted, obligations, ctx);
  return {};
}

} // namespace mlir::triton::intel
