//===- SymbolicBounds.cpp -------------------------------------------------===//
//
// The prover declared in SymbolicBounds.h: the symbol order, overflow-checked
// affine-form arithmetic, normalization, bounding over a loop's iteration
// space, assume facts, the candidate search and guard materialization.
//
//===----------------------------------------------------------------------===//

#include "intel/include/Analysis/SymbolicBounds.h"
#include "intel/include/Analysis/Range.h"
#include "intel/include/Utils/Utility.h"
#include "mlir/IR/Matchers.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <numeric>

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
  if (divisor_ != o.divisor_)
    return divisor_ < o.divisor_;
  return placement_ < o.placement_;
}

/// The loop's step when it is a positive constant.
static std::optional<int64_t> constantStep(scf::ForOp loop) {
  if (std::optional<APInt> step = loop.getConstantStep())
    return step->getSExtValue();
  return std::nullopt;
}

/// The query context one level out, for normalizing a loop's own bounds.
static QueryContext parentContext(scf::ForOp loop) {
  return {loop.getOperation(), loop->getParentOfType<scf::ForOp>()};
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
  switch (s.kind()) {
  case SymbolKind::Opaque:
    return "opaque(" + name + ")";
  case SymbolKind::Quotient:
    return "(" + name + " div " + std::to_string(s.divisor()) + ")";
  case SymbolKind::TripCount:
    return "tripcount(" + name + ")";
  default:
    return name;
  }
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
  if ((k != 1 && k != -1) || !sym.value() ||
      sym.kind() == SymbolKind::TripCount)
    return false;
  Type t = getElementTypeOrSelf(sym.value());
  if (auto intTy = dyn_cast<IntegerType>(t))
    return intTy.getWidth() <= 64;
  return isa<IndexType>(t);
}

/// Normalizes a condition so its subject is as simple as the goal allows: the
/// constant term folds into the bound, and a single-symbol ordered condition
/// is divided by |k| with the bound rounded inward, swapping AtLeast/AtMost
/// when k < 0. Written `-N >= 1` the checked negation of
/// INT64_MIN overflows and rejects a launch the condition admits; written
/// `N <= -1` it does not.
///
/// `DivisibleBy` is not divided that way: with g = gcd(k, c), `k*s + c0`
/// divisible by `c` is satisfiable exactly when g divides c0, and is
/// expressible as `DivisibleBy(s, c/g)` only when the residue is zero, i.e.
/// when (c/g) divides (c0/g). A nonzero residue is a congruence BoundGoal
/// cannot state, so the condition is left alone for the caller to decline.
bool normalizeCondition(BoundCondition &cond) {
  if (cond.expr.overflowed())
    return false;

  if (cond.goal == BoundGoal::DivisibleBy) {
    if (cond.c <= 0 || cond.expr.numTerms() != 1)
      return cond.expr.constant() == 0; // nothing to normalize
    auto &[sym, k] = cond.expr.terms().front();
    int64_t c0 = cond.expr.constant();
    uint64_t magK =
        k < 0 ? 0 - static_cast<uint64_t>(k) : static_cast<uint64_t>(k);
    int64_t g =
        static_cast<int64_t>(std::gcd(magK, static_cast<uint64_t>(cond.c)));
    if (g == 0 || c0 % g != 0)
      return false; // unsatisfiable
    int64_t divisor = cond.c / g;
    if (divisor != 0 && (c0 / g) % divisor != 0)
      return false; // nonzero residue: not expressible as DivisibleBy
    cond.expr = AffineForm::symbol(sym);
    cond.c = divisor;
    return true;
  }

  // Ordered goals: fold the constant into the bound, keeping the rendering of
  // a bare `s >= 0` unchanged.
  int64_t c0 = cond.expr.constant();
  int64_t bound;
  switch (cond.goal) {
  case BoundGoal::NonNegative:
    bound = 0;
    break;
  case BoundGoal::StrictlyPositive:
    bound = 1;
    break;
  default:
    bound = cond.c;
    break;
  }
  bool atMost = cond.goal == BoundGoal::AtMost;
  if (c0 != 0) {
    int64_t folded;
    if (llvm::SubOverflow(bound, c0, folded)) {
      // Folding would need a bound beyond i64. A symbolic part within 2^63
      // cannot pass it, so `(x - 1) <= INT64_MAX` is vacuous; a compound one
      // such as `ub - lb` can.
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
  Type i64 = builder.getI64Type();
  DominanceInfo domInfo(before->getParentOp());
  Value result;

  for (const BoundCondition &cond : conds) {
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
      Value term = gb.extend(v);
      // A quotient's runtime value is the division itself, which equals the
      // narrow quotient because the divisor is positive.
      if (sym.kind() == SymbolKind::Quotient)
        term = arith::DivSIOp::create(builder, loc, term,
                                      gb.constant(sym.divisor()));
      sum = gb.add(sum, gb.mul(term, k));
    }

    Value cmp;
    switch (cond.goal) {
    case BoundGoal::NonNegative:
      cmp = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sge, sum,
                                  gb.constant(0));
      break;
    case BoundGoal::StrictlyPositive:
      cmp = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sgt, sum,
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
    case BoundGoal::DivisibleBy: {
      // remsi against a positive divisor is a valid divisibility test for
      // negative values too, and cannot overflow.
      Value rem =
          arith::RemSIOp::create(builder, loc, sum, gb.constant(cond.c));
      cmp = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq, rem,
                                  gb.constant(0));
      break;
    }
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
  case BoundGoal::StrictlyPositive:
    return expr + " > 0";
  case BoundGoal::DivisibleBy:
    return expr + " divisible by " + std::to_string(c.c);
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
                                           DominanceInfo &domInfo,
                                           Operation *root)
    : solver(solver), domInfo(domInfo), root(root),
      scope(numberingScope(root)) {
  // Number every value in pre-order, from 1, so the symbol sort is total over
  // distinct values and reproducible across processes: a block's arguments as
  // the walk enters it, then each operation's results in result order. 0 is
  // reserved as "unassigned", which AffineForm::symbol asserts against.
  buildFactIndex();
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
                                       int64_t divisor,
                                       AxisPlacement placement) const {
  // A value outside the numbering - outside `scope`, or created after
  // construction - takes the next unused order the first time it is seen and
  // keeps it. Sharing one fallback key would make two such values tie in the
  // sort order, and `combine` would then cancel them as like terms.
  auto [it, inserted] = valueOrder.try_emplace(v, nextOrder);
  if (inserted)
    ++nextOrder;
  return Symbol(kind, v, divisor, std::move(placement), it->second);
}

AffineForm SymbolicBoundsProver::opaque(Value v,
                                        AxisPlacement placement) const {
  return AffineForm::symbol(
      symbolFor(SymbolKind::Opaque, v, 0, std::move(placement)));
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
    return intVal.getSExtValue();
  DenseElementsAttr constAttr;
  if (matchPattern(v, m_Constant(&constAttr)) && constAttr.isSplat()) {
    auto attr = constAttr.getSplatValue<Attribute>();
    if (auto intAttr = dyn_cast_or_null<IntegerAttr>(attr))
      return intAttr.getValue().getSExtValue();
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

AffineForm
SymbolicBoundsProver::leafForBlockArg(BlockArgument arg, QueryContext ctx,
                                      SmallVectorImpl<Obligation> &obligations,
                                      AxisPlacement placement) {
  Operation *owner = arg.getOwner()->getParentOp();
  if (isa_and_nonnull<tt::FuncOp>(owner))
    return AffineForm::symbol(
        symbolFor(SymbolKind::KernelArg, arg, 0, placement));

  if (ctx.loop && owner == ctx.loop.getOperation()) {
    // The induction variable itself.
    if (std::optional<Value> iv = ctx.loop.getSingleInductionVar();
        iv && *iv == arg)
      return AffineForm::symbol(
          symbolFor(SymbolKind::LoopIV, arg, 0, placement));

    // An iter_arg equal to the IV up to an offset: init `lb + c`, yield
    // `arg + step`. Any other iter_arg is opaque.
    std::optional<int64_t> step = constantStep(ctx.loop);
    unsigned numIVs = ctx.loop.getNumInductionVars();
    if (step && arg.getArgNumber() >= numIVs) {
      unsigned idx = arg.getArgNumber() - numIVs;
      Value init = ctx.loop.getInitArgs()[idx];
      Value yielded = cast<scf::YieldOp>(ctx.loop.getBody()->getTerminator())
                          .getOperand(idx);
      bool yieldIsStep = false;
      if (auto add = yielded.getDefiningOp<arith::AddIOp>()) {
        std::optional<int64_t> k = getFoldedConstant(add.getRhs());
        Value other = add.getLhs();
        if (!k) {
          k = getFoldedConstant(add.getLhs());
          other = add.getRhs();
        }
        yieldIsStep = k && *k == *step && other == arg;
      }
      if (yieldIsStep) {
        QueryContext outer = parentContext(ctx.loop);
        SmallVector<Obligation, 4> initObls;
        AffineForm initAF = normalizeImpl(init, outer, initObls, {}, 0);
        AffineForm lb =
            normalizeImpl(ctx.loop.getLowerBound(), outer, initObls, {}, 0);
        AffineForm off = initAF.sub(lb);
        if (!off.overflowed() && off.isConstant()) {
          AffineForm res =
              AffineForm::symbol(symbolFor(SymbolKind::LoopIV,
                                           *ctx.loop.getSingleInductionVar()))
                  .add(AffineForm::constant(off.constant()));
          // The yield's addi is never traversed and the scf.for contract
          // covers only the IV, so `IV + c` carries its own wrap obligation:
          // for i8 lb=120 ub=124 step=1 an iter_arg started at lb + 5
          // wraps to -128 while the IV exit value 124 is representable.
          unsigned width = bitWidth(arg.getType());
          obligations.push_back(
              {Obligation::Wrap, ctx.loop.getOperation(), res, width});
          llvm::append_range(obligations, initObls);
          return res;
        }
      }
    }
  }
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
  if (std::optional<int64_t> cst = getFoldedConstant(v))
    return AffineForm::constant(*cst);

  Operation *def = v.getDefiningOp();
  if (!def)
    return leafForBlockArg(cast<BlockArgument>(v), ctx, obligations, placement);

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
      .Case<arith::DivSIOp>([&](auto op) {
        std::optional<int64_t> c = getFoldedConstant(op.getRhs());
        if (!c || *c <= 0)
          return giveUp(); // non-positive or non-constant divisor
        Value dividend = op.getLhs();
        Symbol q = symbolFor(SymbolKind::Quotient, dividend, *c, placement);
        // The cdiv shape `(X + c - 1) / c` has sharper facts, stated about X.
        QuotientInfo info;
        SmallVector<Obligation, 4> divObls;
        Value factSubject = dividend;
        if (auto add = dividend.getDefiningOp<arith::AddIOp>()) {
          std::optional<int64_t> k = getFoldedConstant(add.getRhs());
          Value other = add.getLhs();
          if (!k) {
            k = getFoldedConstant(add.getLhs());
            other = add.getRhs();
          }
          if (k && *k == *c - 1) {
            info.isCdiv = true;
            factSubject = other;
          }
        }
        // Normalized into a scratch vector: the dividend's arithmetic is an
        // obligation only for a proof that actually uses these facts. Placed
        // like q, or the dividends of q[:, None] and q[None, :] would cancel.
        info.dividend =
            normalizeImpl(factSubject, ctx, divObls, placement, depth + 1);
        if (info.dividend.overflowed())
          return giveUp();
        info.dividendObligations.assign(divObls.begin(), divObls.end());
        if (info.isCdiv) {
          // The `X + c - 1` addition is not traversed - the facts are stated
          // about X - so its wrap obligation would otherwise be lost, yet
          // using those facts depends on it.
          AffineForm num = info.dividend.add(AffineForm::constant(*c - 1));
          if (num.overflowed())
            return giveUp();
          info.dividendObligations.push_back({Obligation::Wrap,
                                              dividend.getDefiningOp(), num,
                                              bitWidth(dividend.getType())});
        }
        quotientInfo[{q, varyingLoopKey(dividend)}] = std::move(info);
        return AffineForm::symbol(q);
      })
      .Case<arith::RemSIOp>([&](auto op) {
        std::optional<int64_t> c = getFoldedConstant(op.getRhs());
        if (!c || *c <= 0)
          return giveUp();
        // X % c == X - c * (X / c), sharing the quotient symbol of the
        // matching division so that X == c*q + r holds by construction. No
        // divsi need exist: the symbol's identity and facts use X and c only.
        Value dividend = op.getLhs();
        // Placed like q: unplaced, x[:, None] and x[None, :] would cancel.
        AffineForm x =
            normalizeImpl(dividend, ctx, obligations, placement, depth + 1);
        if (x.overflowed())
          return giveUp();
        Symbol q = symbolFor(SymbolKind::Quotient, dividend, *c, placement);
        if (!quotientInfo.count({q, varyingLoopKey(dividend)})) {
          QuotientInfo info;
          info.dividend = x;
          quotientInfo[{q, varyingLoopKey(dividend)}] = std::move(info);
        }
        AffineForm res = x.sub(AffineForm::symbol(q).scale(*c));
        if (res.overflowed())
          return giveUp();
        return res;
      })
      .Case<tt::MakeRangeOp>([&](auto op) {
        return AffineForm::symbol(
            symbolFor(SymbolKind::Lane, op.getResult(), 0, placement));
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
            symbolFor(SymbolKind::ProgramId, op.getResult(), 0, placement));
      })
      .Case<tt::GetNumProgramsOp>([&](auto op) {
        return AffineForm::symbol(
            symbolFor(SymbolKind::NumPrograms, op.getResult(), 0, placement));
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
  std::optional<ConstantIntRanges> r = collectRange(solver, sym.value());
  if (!r)
    return std::nullopt;
  int64_t lo = r->smin().getSExtValue(), hi = r->smax().getSExtValue();
  // A quotient's value is its dividend. Truncating division by a positive
  // constant is monotone, so dividing both ends is exact.
  if (sym.kind() == SymbolKind::Quotient)
    return std::make_pair(lo / sym.divisor(), hi / sym.divisor());
  return std::make_pair(lo, hi);
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
SymbolicBoundsProver::symbolBounds(const Symbol &sym, QueryContext ctx,
                                   const CandidateSet &cs) {
  Bounds out;
  switch (sym.kind()) {
  case SymbolKind::Lane: {
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
  case SymbolKind::LoopIV: {
    if (!ctx.loop || sym.value() != *ctx.loop.getSingleInductionVar()) {
      // An IV of some other loop: bounded from constants if at all.
      break;
    }
    std::optional<int64_t> step = constantStep(ctx.loop);
    if (!step || *step <= 0) {
      // Only a constant positive step is supported.
      out.isVarying = true;
      out.finite = false;
      return out;
    }
    QueryContext outer = parentContext(ctx.loop);
    SmallVector<Obligation, 4> boundObls;
    AffineForm lb =
        normalizeImpl(ctx.loop.getLowerBound(), outer, boundObls, {}, 0);
    AffineForm ub =
        normalizeImpl(ctx.loop.getUpperBound(), outer, boundObls, {}, 0);
    out.lo = lb;
    out.hi = cs.exactLoopEnd ? ub.sub(AffineForm::constant(*step))
                             : ub.sub(AffineForm::constant(1));
    out.isVarying = true;
    out.finite = !lb.overflowed() && !ub.overflowed();
    // The obligations of the bounds' own arithmetic travel with the result:
    // a signed i8 `ub = n - 1` is 127 for n = -128, not the mathematical -129.
    llvm::append_range(out.factObligations, boundObls);
    if (ctx.loop.getUnsignedCmp()) {
      // The scf.for contract reads the bounds as unsigned; treating them as
      // signed needs all three preconditions.
      unsigned width = bitWidth(ctx.loop.getLowerBound().getType());
      int64_t intMax = APInt::getSignedMaxValue(width).getSExtValue();
      out.preconditions.push_back(
          {lb, BoundGoal::NonNegative, 0, ConditionKind::Precondition});
      out.preconditions.push_back(
          {ub, BoundGoal::NonNegative, 0, ConditionKind::Precondition});
      out.preconditions.push_back({ub, BoundGoal::AtMost, intMax - *step + 1,
                                   ConditionKind::Precondition});
    }
    return out;
  }
  default:
    break;
  }
  // Loop-invariant symbols stay symbolic; a symbol defined
  // inside the loop is bounded from its constant range, or is unbounded.
  if (ctx.loop && sym.value() &&
      ctx.loop->isAncestor(sym.value().getParentBlock()->getParentOp())) {
    out.isVarying = true;
    std::optional<std::pair<int64_t, int64_t>> b = symbolConstantBounds(sym);
    // The range analysis assigns a value it never narrowed the type's own
    // full-width lattice point, not "no information" (collectRange only
    // returns nullopt for an uninitialized or empty lattice state). Treating
    // that extremum as a usable bound would let a residual candidate turn any
    // unconstrained Opaque value's own type width into a "proof" that only
    // ever holds by forcing the loop empty - sound, but not the no-range
    // Unknown this symbol is actually supposed to be. Tested on value()'s
    // range, a quotient's dividend, since dividing hides the full width.
    std::optional<std::pair<int64_t, int64_t>> valueRange =
        rangeOf(sym.value(), ctx);
    unsigned width = bitWidth(sym.value().getType());
    bool fullWidth =
        valueRange &&
        valueRange->first == APInt::getSignedMinValue(width).getSExtValue() &&
        valueRange->second == APInt::getSignedMaxValue(width).getSExtValue();
    if (b && !fullWidth) {
      out.lo = AffineForm::constant(b->first);
      out.hi = AffineForm::constant(b->second);
    } else {
      out.finite = false;
    }
  }
  return out;
}

SymbolicBoundsProver::Bounds
SymbolicBoundsProver::bound(const AffineForm &e, QueryContext ctx,
                            const CandidateSet &cs) {
  Bounds out{AffineForm::constant(e.constant()),
             AffineForm::constant(e.constant())};
  for (auto &[sym, k] : e.terms()) {
    Bounds sb = symbolBounds(sym, ctx, cs); // loop-varying only
    out.finite &= sb.finite;
    llvm::append_range(out.preconditions, sb.preconditions);
    llvm::append_range(out.factObligations, sb.factObligations);
    llvm::append_range(out.assumes, sb.assumes);
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

/// Replaces every quotient term whose coefficient is a multiple of its
/// divisor with the lower bound the quotient facts give, so the dividend can
/// cancel against other occurrences of itself - which is how the tutorial-03
/// K loop's `K - 64*q(K+63, 64)` collapses to a constant. Returns nullopt when
/// a quotient term cannot be substituted this way; the caller then decides from
/// constant ranges alone. Appends the facts' preconditions and the dividend's
/// wrap obligations to `cs`, since using a fact inherits them. `maximize`
/// substitutes the upper bound instead, which an upper wrap guard needs.
std::optional<AffineForm>
SymbolicBoundsProver::substituteQuotients(const AffineForm &e, QueryContext ctx,
                                          CandidateSet &cs, bool maximize) {
  AffineForm out = AffineForm::constant(e.constant());
  bool substituted = false;
  for (auto &[sym, k] : e.terms()) {
    if (sym.kind() != SymbolKind::Quotient) {
      out = out.add(AffineForm::symbol(sym).scale(k));
      continue;
    }
    int64_t c = sym.divisor();
    if (c <= 0 || k % c != 0) {
      // Not a multiple of the divisor: the quotient-threshold candidate
      // handles this shape; here the term stays symbolic.
      out = out.add(AffineForm::symbol(sym).scale(k));
      continue;
    }
    const QuotientInfo *info = findQuotientInfo(sym, ctx);
    if (!info)
      return std::nullopt;

    int64_t m = k / c; // k*q == m*(c*q)
    bool exact = llvm::is_contained(cs.exactCdiv, sym);
    // The end of c*q that minimizes m*(c*q) - the low end when m > 0 - or
    // with `maximize` the end that maximizes it.
    bool highEnd = (m < 0) != maximize;
    AffineForm bound;
    if (exact) {
      // The exact-cdiv candidate: the division is exact, so c*q == X.
      bound = info->dividend;
    } else if (info->isCdiv) {
      // (X + c - 1) / c gives X <= c*q <= X + c - 1.
      bound = highEnd ? info->dividend.add(AffineForm::constant(c - 1))
                      : info->dividend;
    } else {
      // X - (c - 1) <= c*q <= X.
      bound = highEnd ? info->dividend
                      : info->dividend.sub(AffineForm::constant(c - 1));
    }
    AffineForm term = bound.scale(m);
    if (term.overflowed())
      return std::nullopt;
    out = out.add(term);
    substituted = true;

    // The division facts hold only for a non-negative dividend, and using
    // them inherits the numerator's wrap obligations.
    BoundCondition nonNeg{info->dividend, BoundGoal::NonNegative, 0,
                          ConditionKind::Precondition};
    if (!llvm::is_contained(cs.extra, nonNeg))
      cs.extra.push_back(nonNeg);
    for (const Obligation &o : info->dividendObligations)
      if (!llvm::is_contained(cs.factObligations, o))
        cs.factObligations.push_back(o);
  }
  if (out.overflowed())
    return std::nullopt;
  return substituted ? std::optional<AffineForm>(out) : std::nullopt;
}

const SymbolicBoundsProver::QuotientInfo *
SymbolicBoundsProver::findQuotientInfo(const Symbol &sym,
                                       QueryContext ctx) const {
  // Keyed by (symbol, the loop in which the dividend varies), so the entry is
  // the same whether the quotient is reached from a loop bound - normalized in
  // the parent context - or from the residual inside the loop.
  auto it = quotientInfo.find({sym, varyingLoopKey(sym.value())});
  return it != quotientInfo.end() ? &it->second : nullptr;
}

Operation *SymbolicBoundsProver::varyingLoopKey(Value v) {
  if (!v)
    return nullptr;
  Operation *anchor = v.getDefiningOp();
  if (!anchor)
    anchor = cast<BlockArgument>(v).getOwner()->getParentOp();
  if (!anchor)
    return nullptr;
  if (auto self = dyn_cast<scf::ForOp>(anchor))
    return self.getOperation();
  if (auto loop = anchor->getParentOfType<scf::ForOp>())
    return loop.getOperation();
  return nullptr;
}

bool SymbolicBoundsProver::termSignOk(const Symbol &sym, int64_t k,
                                      QueryContext ctx, CandidateSet &cs) {
  // An applicable assume fact is checked first, and recorded as provenance.
  if (sym.value() && sym.kind() != SymbolKind::Quotient &&
      sym.kind() != SymbolKind::TripCount) {
    for (const Fact &f : factsFor(sym.value())) {
      if (!ctx.at ||
          !assumeApplies(cast<LLVM::AssumeOp>(f.assume), ctx.at, domInfo))
        continue;
      bool good = k > 0 ? (f.goal == BoundGoal::NonNegative ||
                           (f.goal == BoundGoal::AtLeast && f.c >= 0))
                        : (f.goal == BoundGoal::AtMost && f.c <= 0);
      if (good) {
        if (!llvm::is_contained(cs.assumes, f.assume))
          cs.assumes.push_back(f.assume);
        return true;
      }
    }
  }
  std::optional<std::pair<int64_t, int64_t>> b =
      sym.kind() == SymbolKind::Lane ? symbolConstantBounds(sym)
                                     : rangeOf(sym.value(), ctx, &cs.assumes);
  if (!b)
    return false;
  return k > 0 ? b->first >= 0 : b->second <= 0;
}

bool SymbolicBoundsProver::decideResidual(const AffineForm &lo, int64_t g,
                                          QueryContext ctx, CandidateSet &cs) {
  if (lo.overflowed())
    return false;
  if (lo.isConstant())
    return lo.constant() >= g;
  // Quotient facts first: substituting `c*q` by its bound on the dividend is
  // what lets the dividend cancel against its other occurrences.
  if (std::optional<AffineForm> sub =
          substituteQuotients(lo, ctx, cs, /*maximize=*/false))
    if (decideResidual(*sub, g, ctx, cs))
      return true;
  // Every term contributes at least `k * floor` to the residual, where
  // `floor` is 0 for a term only sign-checked (a term with the safe sign
  // cannot make things worse, so it is dropped at zero) or the trial
  // hypothesis the term-sign or quotient-threshold candidate is testing
  // (`cs.signFloor`/`signCeil`), checked first since it is the most
  // specific. The margin starts at the constant term and accumulates every
  // contribution, which is what lets a term-sign `StrictlyPositive` (floor
  // 1) and a quotient threshold (an arbitrary floor) close a gap the plain
  // sign check cannot.
  int64_t margin = lo.constant();
  for (auto &[sym, k] : lo.terms()) {
    auto bump = [&](int64_t bound) {
      int64_t contribution;
      if (llvm::MulOverflow(k, bound, contribution) ||
          llvm::AddOverflow(margin, contribution, margin))
        return false;
      return true;
    };
    if (k > 0) {
      auto it =
          llvm::find_if(cs.signFloor, [&](auto &e) { return e.first == sym; });
      if (it != cs.signFloor.end()) {
        if (!bump(it->second))
          return false;
        continue;
      }
    } else {
      auto it =
          llvm::find_if(cs.signCeil, [&](auto &e) { return e.first == sym; });
      if (it != cs.signCeil.end()) {
        if (!bump(it->second))
          return false;
        continue;
      }
    }
    if (!termSignOk(sym, k, ctx, cs))
      return false;
    // No trial hypothesis for this term: the plain sign check, floor 0,
    // contributes nothing extra to the margin.
  }
  return margin >= g;
}

std::optional<int64_t>
SymbolicBoundsProver::residualConstant(const AffineForm &d, QueryContext ctx,
                                       CandidateSet &cs) {
  Bounds b = bound(d, ctx, cs);
  if (!b.finite || b.exhausted)
    return std::nullopt;
  if (std::optional<std::pair<int64_t, int64_t>> cb = boundConstant(b.lo))
    return cb->first;
  return std::nullopt;
}

void SymbolicBoundsProver::mergePreconditions(CandidateSet &cs,
                                              const Bounds &b) const {
  for (const BoundCondition &c : b.preconditions)
    if (!llvm::is_contained(cs.extra, c))
      cs.extra.push_back(c);
  for (const Obligation &o : b.factObligations)
    if (!llvm::is_contained(cs.factObligations, o))
      cs.factObligations.push_back(o);
  for (Operation *a : b.assumes)
    if (!llvm::is_contained(cs.assumes, a))
      cs.assumes.push_back(a);
  cs.exhausted |= b.exhausted;
}

//===----------------------------------------------------------------------===//
// Assume facts
//===----------------------------------------------------------------------===//

void SymbolicBoundsProver::buildFactIndex() {
  // Indexed by the SUBJECT the fact is about, not by the comparison's
  // immediate operands: `assume((n % 64) == 0)` is a fact about `n`, and
  // IntegerRangeAnalysis::collectAssumptions would file it under the
  // remainder value, invisible to a query about `n`.
  root->walk([&](LLVM::AssumeOp assume) {
    auto cmp = assume.getCond().getDefiningOp<arith::CmpIOp>();
    if (!cmp)
      return;
    Value lhs = cmp.getLhs(), rhs = cmp.getRhs();
    arith::CmpIPredicate pred = cmp.getPredicate();

    // `remsi X, c == 0` -> DivisibleBy(X, c).
    if (pred == arith::CmpIPredicate::eq) {
      for (auto [a, b] : {std::pair{lhs, rhs}, std::pair{rhs, lhs}}) {
        std::optional<int64_t> zero = getFoldedConstant(b);
        auto rem = a.getDefiningOp<arith::RemSIOp>();
        if (zero && *zero == 0 && rem)
          if (std::optional<int64_t> c = getFoldedConstant(rem.getRhs()))
            if (*c > 0)
              factIndex[rem.getLhs()].push_back(
                  {BoundGoal::DivisibleBy, *c, assume.getOperation()});
      }
      return; // no other eq form yields a Goal
    }

    // `X pred const`, either operand order.
    Value subject = lhs;
    std::optional<int64_t> k = getFoldedConstant(rhs);
    if (!k) {
      subject = rhs;
      k = getFoldedConstant(lhs);
      if (!k)
        return;
      // Swapping the operands mirrors the predicate.
      pred = arith::invertPredicate(pred) == pred ? pred : pred;
      switch (cmp.getPredicate()) {
      case arith::CmpIPredicate::slt:
        pred = arith::CmpIPredicate::sgt;
        break;
      case arith::CmpIPredicate::sle:
        pred = arith::CmpIPredicate::sge;
        break;
      case arith::CmpIPredicate::sgt:
        pred = arith::CmpIPredicate::slt;
        break;
      case arith::CmpIPredicate::sge:
        pred = arith::CmpIPredicate::sle;
        break;
      case arith::CmpIPredicate::ult:
        pred = arith::CmpIPredicate::ugt;
        break;
      case arith::CmpIPredicate::ule:
        pred = arith::CmpIPredicate::uge;
        break;
      case arith::CmpIPredicate::ugt:
        pred = arith::CmpIPredicate::ult;
        break;
      case arith::CmpIPredicate::uge:
        pred = arith::CmpIPredicate::ule;
        break;
      default:
        return;
      }
    }
    unsigned w = bitWidth(subject.getType());
    int64_t intMax = APInt::getSignedMaxValue(w).getSExtValue();
    Operation *op = assume.getOperation();
    auto add = [&](BoundGoal g, int64_t c) {
      factIndex[subject].push_back({g, c, op});
    };
    // Strict bounds are translated to non-strict at the subject's width; a
    // strict bound at the extreme is unsatisfiable and yields no fact.
    switch (pred) {
    case arith::CmpIPredicate::sge:
      add(BoundGoal::AtLeast, *k);
      if (*k >= 0)
        add(BoundGoal::NonNegative, 0);
      break;
    case arith::CmpIPredicate::sgt:
      if (*k == intMax)
        break; // x > INT_MAX is unsatisfiable
      add(BoundGoal::AtLeast, *k + 1);
      if (*k + 1 >= 0)
        add(BoundGoal::NonNegative, 0);
      break;
    case arith::CmpIPredicate::sle:
      add(BoundGoal::AtMost, *k);
      break;
    case arith::CmpIPredicate::slt:
      if (*k == APInt::getSignedMinValue(w).getSExtValue())
        break; // x < INT_MIN is unsatisfiable
      add(BoundGoal::AtMost, *k - 1);
      break;
    case arith::CmpIPredicate::ult:
    case arith::CmpIPredicate::ule: {
      // An unsigned UPPER bound within the signed range puts x in [0, c), so
      // it gives both non-negativity and a signed upper bound.
      if (*k < 0 || *k > intMax)
        break; // the constant itself is outside [0, INT_MAX]: no signed fact
      int64_t bound = pred == arith::CmpIPredicate::ult ? *k - 1 : *k;
      add(BoundGoal::NonNegative, 0);
      if (bound >= 0)
        add(BoundGoal::AtMost, bound);
      break;
    }
    default:
      // uge/ugt give no signed fact: an unsigned lower bound admits negative
      // signed values (`assume(x uge 128)` on i8 means x < 0).
      break;
    }
  });
}

ArrayRef<SymbolicBoundsProver::Fact>
SymbolicBoundsProver::factsFor(Value v) const {
  auto it = factIndex.find(v);
  return it != factIndex.end() ? ArrayRef<Fact>(it->second) : ArrayRef<Fact>();
}

Operation *SymbolicBoundsProver::assumedBy(const BoundCondition &cond,
                                           QueryContext ctx) const {
  // Only a bare single symbol: `1*s + 0`.
  if (cond.expr.numTerms() != 1 || cond.expr.constant() != 0)
    return nullptr;
  auto &[sym, k] = cond.expr.terms().front();
  if (k != 1 || !sym.value())
    return nullptr;
  // The index is keyed by Value, and a Quotient symbol stores its DIVIDEND,
  // so a key match is not a subject match: a fact about X must never satisfy
  // a condition on q(X, c). No assume names a derived value.
  if (sym.kind() == SymbolKind::Quotient || sym.kind() == SymbolKind::TripCount)
    return nullptr;
  for (const Fact &f : factsFor(sym.value())) {
    if (!ctx.at ||
        !assumeApplies(cast<LLVM::AssumeOp>(f.assume), ctx.at, domInfo))
      continue;
    bool implies = false;
    switch (cond.goal) {
    case BoundGoal::DivisibleBy:
      implies =
          f.goal == BoundGoal::DivisibleBy && cond.c != 0 && f.c % cond.c == 0;
      break;
    case BoundGoal::AtLeast:
      implies = f.goal == BoundGoal::AtLeast && f.c >= cond.c;
      break;
    case BoundGoal::AtMost:
      implies = f.goal == BoundGoal::AtMost && f.c <= cond.c;
      break;
    case BoundGoal::NonNegative:
      implies = f.goal == BoundGoal::NonNegative ||
                (f.goal == BoundGoal::AtLeast && f.c >= 0);
      break;
    case BoundGoal::StrictlyPositive:
      implies = f.goal == BoundGoal::AtLeast && f.c >= 1;
      break;
    }
    if (implies)
      return f.assume;
  }
  return nullptr;
}

std::optional<std::pair<int64_t, int64_t>>
SymbolicBoundsProver::rangeOf(Value v, QueryContext ctx,
                              SmallVectorImpl<Operation *> *assumes) const {
  std::optional<ConstantIntRanges> r = collectRange(solver, v);
  if (!r)
    return std::nullopt;
  int64_t lo = r->smin().getSExtValue(), hi = r->smax().getSExtValue();
  if (!assumes)
    return std::make_pair(lo, hi);
  // A leaf range can come from assumes this index does not model (eq, uge,
  // ...) and from assumes on values UPSTREAM of the leaf, since ranges
  // propagate forward while the lattice keeps no provenance. So when the range
  // is narrower than the type, record every applicable assume in the function:
  // a coarse over-approximation, never an omission.
  unsigned w = bitWidth(v.getType());
  bool narrowed = lo > APInt::getSignedMinValue(w).getSExtValue() ||
                  hi < APInt::getSignedMaxValue(w).getSExtValue();
  if (narrowed && ctx.at) {
    if (auto func = ctx.at->getParentOfType<tt::FuncOp>())
      func->walk([&](LLVM::AssumeOp a) {
        if (!llvm::is_contained(*assumes, a.getOperation()))
          assumes->push_back(a.getOperation());
      });
  }
  return std::make_pair(lo, hi);
}

CandidateResult SymbolicBoundsProver::addCandidate(CandidateSet &cs,
                                                   BoundCondition cond,
                                                   QueryContext ctx) {
  // Every condition subject must be a scalar that dominates the loop, so the
  // guard can be placed before it. Checked here and again in finalize,
  // because an obligation guard can reach the verdict with no candidate.
  for (auto &[sym, k] : cond.expr.terms()) {
    if (!sym.value() || sym.kind() == SymbolKind::TripCount)
      return CandidateResult::Declined;
    if (isa<ShapedType>(sym.value().getType()))
      return CandidateResult::Declined;
  }
  if (Operation *assume = assumedBy(cond, ctx)) {
    // Established outright: no runtime condition, but record the provenance.
    if (!llvm::is_contained(cs.assumes, assume))
      cs.assumes.push_back(assume);
    return CandidateResult::Accepted;
  }
  if (!normalizeCondition(cond))
    return CandidateResult::Declined;
  if (llvm::is_contained(cs.facts, cond))
    return CandidateResult::Accepted;
  if (cs.facts.size() >= kMaxFactConditions)
    return CandidateResult::Exhausted;
  cond.kind = ConditionKind::Fact;
  cs.facts.push_back(std::move(cond));
  return CandidateResult::Accepted;
}

//===----------------------------------------------------------------------===//
// Obligation discharge and guard materialization
//===----------------------------------------------------------------------===//

/// True when `op` is `iv + c` with 0 <= c <= step for the IV of `ctx.loop`.
/// The scf.for contract makes `lb + n*step` representable, so such an addition
/// cannot wrap, which discharges its obligation at tier 1 for free.
bool SymbolicBoundsProver::isLoopIvPlusSmallConstant(arith::AddIOp add,
                                                     QueryContext ctx,
                                                     const CandidateSet &cs) {
  if (!ctx.loop)
    return false;
  std::optional<int64_t> step = constantStep(ctx.loop);
  std::optional<Value> iv = ctx.loop.getSingleInductionVar();
  if (!step || !iv)
    return false;
  // An unsigned loop only gets the signed rules under its three
  // preconditions, which symbolBounds has already emitted.
  Value lhs = add.getLhs(), rhs = add.getRhs();
  std::optional<int64_t> c = getFoldedConstant(rhs);
  Value other = lhs;
  if (!c) {
    c = getFoldedConstant(lhs);
    other = rhs;
  }
  return c && *c >= 0 && *c <= *step && other == *iv;
}

bool SymbolicBoundsProver::dischargeTier1(const Obligation &o, QueryContext ctx,
                                          CandidateSet &cs) {
  Bounds b = bound(o.expr, ctx, cs);
  mergePreconditions(cs, b);
  if (b.exhausted) {
    // Overflowed endpoints are not bounds: the obligation can be neither
    // discharged nor guarded, so the query ends Unknown.
    cs.exhausted = true;
    return false;
  }
  if (!b.finite)
    return false;

  if (o.kind == Obligation::NonNegative)
    return decideResidual(b.lo, 0, ctx, cs);

  if (auto add = dyn_cast_or_null<arith::AddIOp>(o.op))
    if (isLoopIvPlusSmallConstant(add, ctx, cs))
      return true;

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
  return hi->second <= APInt::getSignedMaxValue(width).getSExtValue() &&
         lo->first >= APInt::getSignedMinValue(width).getSExtValue();
}

/// The guards that close an open wrap obligation: the result must fit its
/// width at both ends.
void SymbolicBoundsProver::guardsForObligation(
    const Obligation &o, QueryContext ctx, CandidateSet &cs,
    SmallVectorImpl<BoundCondition> &out) {
  Bounds b = bound(o.expr, ctx, cs);
  if (!b.finite || b.exhausted) {
    // lo/hi are not bounds (a non-finite symbol is left at zero in them), so
    // no guard closes the obligation: the query ends Unknown.
    cs.exhausted = true;
    return;
  }
  // Substitute quotient terms first, so a dividend cancels against its other
  // occurrences: the tutorial-03 K loop's `64*q(K+63,64) - 64` collapses to
  // `K - 1` rather than becoming a guard on an opaque division.
  AffineForm lo = b.lo, hi = b.hi;
  if (std::optional<AffineForm> s =
          substituteQuotients(lo, ctx, cs, /*maximize=*/false))
    lo = *s;
  if (std::optional<AffineForm> s =
          substituteQuotients(hi, ctx, cs, /*maximize=*/true))
    hi = *s;

  auto push = [&](AffineForm e, BoundGoal goal, int64_t c) {
    BoundCondition cond{std::move(e), goal, c, ConditionKind::Guard};
    if (!normalizeCondition(cond)) {
      // Not expressible: leave the obligation open, which finalize reads as
      // a verdict of Unknown rather than a silently dropped guard.
      cs.exhausted = true;
      return;
    }
    out.push_back(std::move(cond));
  };

  if (o.kind == Obligation::NonNegative) {
    push(lo, BoundGoal::NonNegative, 0);
    return;
  }
  int64_t width = o.width ? o.width : 64;
  push(hi, BoundGoal::AtMost, APInt::getSignedMaxValue(width).getSExtValue());
  push(lo, BoundGoal::AtLeast, APInt::getSignedMinValue(width).getSExtValue());
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
  case BoundGoal::StrictlyPositive:
    return b->first >= 1;
  case BoundGoal::AtLeast:
    return b->first >= cond.c;
  case BoundGoal::AtMost:
    return b->second <= cond.c;
  case BoundGoal::DivisibleBy:
    // Divisibility does not follow from a range unless the range pins one
    // value, which boundConstant reports as lo == hi.
    return cond.c != 0 && b->first == b->second && b->first % cond.c == 0;
  }
  return false;
}

namespace {
/// The lower/upper bound a goal states, in a form two conditions on the same
/// subject can be compared by, or `nullopt` when the goal states no order
/// bound (not implied by, and does not imply, a sign condition on the other
/// side).
std::optional<int64_t> lowerBoundOf(const BoundCondition &c) {
  switch (c.goal) {
  case BoundGoal::NonNegative:
    return 0;
  case BoundGoal::StrictlyPositive:
    return 1;
  case BoundGoal::AtLeast:
    return c.c;
  default:
    return std::nullopt;
  }
}
std::optional<int64_t> upperBoundOf(const BoundCondition &c) {
  return c.goal == BoundGoal::AtMost ? std::optional<int64_t>(c.c)
                                     : std::nullopt;
}

/// True when `stronger`, already retained, makes `weaker` redundant: same
/// subject, and `stronger`'s bound is at least as tight. Complements
/// `impliedByRanges`, which can only use unconditional range evidence.
bool conditionImplies(const BoundCondition &stronger,
                      const BoundCondition &weaker) {
  if (!(stronger.expr == weaker.expr))
    return false;
  if (std::optional<int64_t> sl = lowerBoundOf(stronger))
    if (std::optional<int64_t> wl = lowerBoundOf(weaker))
      return *sl >= *wl;
  if (std::optional<int64_t> su = upperBoundOf(stronger))
    if (std::optional<int64_t> wu = upperBoundOf(weaker))
      return *su <= *wu;
  if (stronger.goal == BoundGoal::DivisibleBy &&
      weaker.goal == BoundGoal::DivisibleBy)
    return stronger.c != 0 && weaker.c != 0 && stronger.c % weaker.c == 0;
  return false;
}

/// The width metric: true when `a`'s admitted set is a (non-strict)
/// superset of `b`'s, decided structurally rather than by evaluating over the
/// argument domain. Sound only when it returns true: a false result means
/// "not provably wider", not "narrower" - the two could be genuinely
/// incomparable (different subjects), which the caller's tie-break handles.
bool admitsAtLeast(const BoundProof &a, const BoundProof &b) {
  // Every restriction `a` imposes must follow from one of `b`'s; conditions
  // only `b` has just narrow `b` further.
  return llvm::all_of(a.conditions, [&](const BoundCondition &ac) {
    return llvm::any_of(b.conditions, [&](const BoundCondition &bc) {
      return conditionImplies(bc, ac);
    });
  });
}

/// Picks the proof with the widest admitted set among `finishers`, which all
/// decide the same query: by exact superset comparison when one side's
/// conditions dominate every one of the other's on a matched subject;
/// otherwise (genuinely incomparable subject sets) fewer total conditions
/// wins, and the earliest-discovered kind is the final tie-break, so an
/// undecidable case never becomes nondeterministic.
size_t pickWidest(ArrayRef<BoundProof> finishers) {
  size_t best = 0;
  for (size_t i = 1; i < finishers.size(); ++i) {
    const BoundProof &cand = finishers[i];
    const BoundProof &cur = finishers[best];
    bool candWider = admitsAtLeast(cand, cur) && !admitsAtLeast(cur, cand);
    bool curWider = admitsAtLeast(cur, cand) && !admitsAtLeast(cand, cur);
    if (candWider) {
      best = i;
    } else if (!curWider && cand.conditions.size() < cur.conditions.size()) {
      // Neither structurally dominates the other: fewer conditions is the
      // documented proxy, and ties keep whichever came first (do nothing).
      best = i;
    }
  }
  return best;
}
} // namespace

BoundProof SymbolicBoundsProver::finalize(BoundProof::Verdict onD,
                                          CandidateSet cs,
                                          ArrayRef<Obligation> obligations,
                                          QueryContext ctx) {
  // Close every obligation - the query's own and those inherited from
  // the dividends of quotient facts - at tier 1, else as a runtime guard. A
  // worklist, because closing one obligation can add another.
  SmallVector<BoundCondition, 4> guards;
  SmallVector<Obligation, 8> work(obligations.begin(), obligations.end());
  SmallVector<Obligation, 8> seen;
  for (unsigned i = 0; i < work.size(); ++i) {
    Obligation o = work[i];
    if (llvm::is_contained(seen, o))
      continue;
    seen.push_back(o);
    if (dischargeTier1(o, ctx, cs))
      continue;
    if (cs.exhausted)
      return {};
    guardsForObligation(o, ctx, cs, guards);
    if (guards.size() > kMaxGuards)
      return {};
    // Using a quotient fact pulls in its dividend's obligations.
    for (const Obligation &extra : cs.factObligations)
      if (!llvm::is_contained(work, extra))
        work.push_back(extra);
  }
  // Index-based: closing one of these can append more (a quotient fact pulls
  // in its dividend's obligations), so the container may grow under us.
  for (unsigned i = 0; i < cs.factObligations.size(); ++i) {
    Obligation extra = cs.factObligations[i];
    if (llvm::is_contained(seen, extra))
      continue;
    seen.push_back(extra);
    if (dischargeTier1(extra, ctx, cs))
      continue;
    if (cs.exhausted)
      return {};
    guardsForObligation(extra, ctx, cs, guards);
    if (guards.size() > kMaxGuards)
      return {};
  }
  if (cs.exhausted)
    return {};

  BoundProof proof;
  // Facts, then preconditions, then guards, pruned by `emit` below: a
  // condition is dropped when unconditional range evidence implies it OR
  // when an earlier-emitted, still-retained condition does.
  auto emit = [&](const BoundCondition &c, ConditionKind kind) {
    BoundCondition out = c;
    out.kind = kind;
    if (out.expr.isConstant()) {
      // A condition over no symbol is decided now: dropped when true, and it
      // makes the verdict Unknown when false.
      bool holds = false;
      switch (out.goal) {
      case BoundGoal::NonNegative:
        holds = out.expr.constant() >= 0;
        break;
      case BoundGoal::StrictlyPositive:
        holds = out.expr.constant() > 0;
        break;
      case BoundGoal::AtLeast:
        holds = out.expr.constant() >= out.c;
        break;
      case BoundGoal::AtMost:
        holds = out.expr.constant() <= out.c;
        break;
      case BoundGoal::DivisibleBy:
        holds = out.c != 0 && out.expr.constant() % out.c == 0;
        break;
      }
      if (!holds)
        proof.verdict = BoundProof::Unknown;
      return holds;
    }
    // Closed-evidence pruning: a condition is omitted when unconditional
    // range evidence implies it, OR when it is implied by a condition this
    // same proof has already retained - a fact can make a later precondition
    // or guard redundant, since facts emit first. Nothing is dropped
    // using a condition that was itself dropped, so no condition can justify
    // itself: `proof.conditions` holds only what survived pruning so far.
    if (impliedByRanges(out) ||
        llvm::any_of(proof.conditions, [&](const BoundCondition &kept) {
          return conditionImplies(kept, out);
        }))
      return true;
    if (!llvm::is_contained(proof.conditions, out))
      proof.conditions.push_back(out);
    return true;
  };

  bool feasible = true;
  for (const BoundCondition &c : cs.facts)
    feasible &= emit(c, ConditionKind::Fact);
  for (const BoundCondition &c : cs.extra)
    feasible &= emit(c, ConditionKind::Precondition);
  for (const BoundCondition &c : guards)
    feasible &= emit(c, ConditionKind::Guard);
  if (!feasible)
    return {};

  // Every emitted condition must have a scalar subject that dominates the
  // loop, or no guard can be placed. Candidate formation checks this, but an
  // obligation guard can reach here with no candidate at all - as when extui
  // and extsi of one tensor cancel in d.
  for (const BoundCondition &c : proof.conditions)
    for (auto &[sym, k] : c.expr.terms())
      if (!sym.value() || isa<ShapedType>(sym.value().getType()) ||
          sym.kind() == SymbolKind::TripCount)
        return {};

  proof.factsUsed.assign(cs.assumes.begin(), cs.assumes.end());
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
  // that case is undecided (not merely unproven) without this obligation.
  // Without it, the term-sign or residual-guard candidate could pick the
  // direction that makes the SIGNED inequality hold while the actual
  // UNSIGNED comparison disagrees: `x ult 100` reduced to `100 - x >= 1`
  // would accept `x <= 0`, true at the bit pattern of x = -1, where the
  // unsigned value 255 is not less than 100.
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

  CandidateSet base;

  // The direct decision, on a trial copy so a failed attempt leaks no
  // evidence into the candidate search.
  {
    CandidateSet direct = base;
    Bounds b = bound(d, ctx, direct);
    mergePreconditions(direct, b);
    if (b.finite && !b.exhausted && decideResidual(b.lo, g, ctx, direct))
      return finalize(BoundProof::Satisfied, std::move(direct), obligations,
                      ctx);
  }
  {
    // Refutation bounds d too, and bounding an unsigned-loop IV yields the
    // loop-contract preconditions; dropping them would refute `iv < 120` in an
    // unsigned i8 loop 120 to 132 (bit pattern) step 4, whose IV 128 reads as
    // -128. Only a constant hi refutes.
    CandidateSet ref = base;
    Bounds b = bound(d, ctx, ref);
    mergePreconditions(ref, b);
    if (b.finite && !b.exhausted && b.hi.isConstant() && b.hi.constant() < g)
      return finalize(BoundProof::Refuted, std::move(ref), obligations, ctx);
  }

  // The candidate search. Five kinds, tried once each in this order:
  //   ExactLoopEnd       the loop span is a multiple of the step;
  //   ExactCdiv          a quotient's dividend is a multiple of its divisor;
  //   TermSign           a sign for each residual term nothing else decides;
  //   QuotientThreshold  a bound on one quotient, stated on its dividend;
  //   ResidualGuard      the whole bounded residual, as a last resort.
  // Candidates are re-discovered on the residual after every commit, and a
  // candidate is kept in `acc` when it strictly improves lo(d) even if it
  // does not finish the proof. A kind that *does* finish is finalized and
  // recorded in `finishers` rather than returned immediately, so every kind
  // gets a chance - a cheap single-symbol guard (TermSign) must not pre-empt
  // a wider one a later kind would have found (ResidualGuard) - and does
  // *not* feed its own trial back into `acc`: the next kind explores
  // independently from the same baseline, not contaminated by a hypothesis
  // that only mattered because an earlier kind happened to finish with it.
  CandidateSet acc = base;
  std::optional<int64_t> best = residualConstant(d, ctx, acc);
  SmallVector<BoundProof, 2> finishers;
  enum CandidateKind {
    ExactLoopEnd,
    ExactCdiv,
    TermSign,
    QuotientThreshold,
    ResidualGuard
  };
  for (CandidateKind kind :
       {ExactLoopEnd, ExactCdiv, TermSign, QuotientThreshold, ResidualGuard}) {
    // Discover on the bounded residual, never on d: in the tutorial-03 K loop
    // the quotient only appears once hi(k) = q - 1 has been substituted.
    SmallVector<BoundCondition, 2> candidates;
    CandidateSet probe = acc;
    Bounds pb = bound(d, ctx, probe);
    if (!pb.finite || pb.exhausted)
      break;

    bool wantExactLoopEnd = false;
    bool alwaysSucceeds = false;
    SmallVector<Symbol, 2> wantExactCdiv;
    SmallVector<std::pair<Symbol, int64_t>, 2> wantSignFloor, wantSignCeil;
    SmallVector<BoundCondition, 1> wantExtra;
    SmallVector<Obligation, 2> wantFactObligations;
    if (kind == ExactLoopEnd) {
      if (!ctx.loop || !constantStep(ctx.loop))
        continue;
      QueryContext outer = parentContext(ctx.loop);
      SmallVector<Obligation, 4> boundObls;
      AffineForm lb =
          normalizeImpl(ctx.loop.getLowerBound(), outer, boundObls, {}, 0);
      AffineForm ub =
          normalizeImpl(ctx.loop.getUpperBound(), outer, boundObls, {}, 0);
      AffineForm span = ub.sub(lb);
      if (span.overflowed())
        continue;
      wantExactLoopEnd = true;
      candidates.push_back({span, BoundGoal::DivisibleBy,
                            *constantStep(ctx.loop), ConditionKind::Fact});
    } else if (kind == ExactCdiv) {
      for (auto &[sym, k] : pb.lo.terms()) {
        if (sym.kind() != SymbolKind::Quotient)
          continue;
        const QuotientInfo *info = findQuotientInfo(sym, ctx);
        if (!info)
          continue;
        wantExactCdiv.push_back(sym);
        candidates.push_back({info->dividend, BoundGoal::DivisibleBy,
                              sym.divisor(), ConditionKind::Fact});
      }
      if (candidates.empty())
        continue;
    } else if (kind == TermSign) {
      // Scalar loop-invariant residual terms with a POSITIVE coefficient
      // whose sign `termSignOk` cannot establish from a fact or a range, as
      // `NonNegative` or `StrictlyPositive`; no decomposition of the symbol,
      // even when it is itself Opaque (an unanalyzed product, say). A
      // negative-coefficient term is left to ResidualGuard: assuming `s <= 0`
      // here would pick the sign that fits this residual's SIGNED reduction
      // while actively contradicting an UNSIGNED predicate's own `s >= 0`
      // obligation (prove()), which ResidualGuard's whole-residual guard does
      // not, since it never touches a term's sign in isolation.
      SmallVector<std::pair<Symbol, int64_t>, 2> undecided;
      for (auto &[sym, k] : pb.lo.terms()) {
        if (k <= 0)
          continue;
        if (sym.kind() == SymbolKind::Quotient ||
            sym.kind() == SymbolKind::TripCount)
          continue; // QuotientThreshold's shape, or no runtime value to guard
        if (!sym.value() || isa<ShapedType>(sym.value().getType()))
          continue; // a guard subject must be a scalar
        if (termSignOk(sym, k, ctx, probe))
          continue; // decideResidual's generic path already covers this term
        undecided.push_back({sym, k});
      }
      if (undecided.empty())
        continue;
      // Floor 0 (NonNegative) is the weakest hypothesis and is always
      // proposed; it costs nothing (contributes 0 to the margin). Upgrading a
      // term to floor 1 (StrictlyPositive) adds k to the margin, so upgrade
      // the fewest terms - largest k first - needed to close the gap. Sound
      // regardless of which terms are chosen: a term left at its weak floor
      // still gets a guard, which the sign check genuinely requires either
      // way.
      int64_t need;
      if (llvm::SubOverflow(g, pb.lo.constant(), need))
        continue; // the gap itself is not representable
      llvm::sort(undecided, [](const auto &a, const auto &b) {
        return a.second > b.second;
      });
      SmallVector<bool, 2> upgrade(undecided.size(), false);
      for (unsigned i = 0; i < undecided.size() && need > 0; ++i) {
        upgrade[i] = true;
        need -= undecided[i].second;
      }
      if (need > 0)
        continue; // even every term upgraded cannot close the gap
      for (auto [idx, pr] : llvm::enumerate(undecided)) {
        auto &[sym, k] = pr;
        wantSignFloor.push_back({sym, upgrade[idx] ? 1 : 0});
        candidates.push_back({AffineForm::symbol(sym),
                              upgrade[idx] ? BoundGoal::StrictlyPositive
                                           : BoundGoal::NonNegative,
                              0, ConditionKind::Fact});
      }
    } else if (kind == QuotientThreshold) {
      // A single Quotient q(X, c) whose coefficient is not a multiple of c -
      // the shape `substituteQuotients` leaves symbolic, declined to ExactCdiv.
      // Translate the threshold on q that the goal requires into a condition
      // on the dividend X.
      SmallVector<std::pair<Symbol, int64_t>, 1> quotientTerms;
      for (auto &[sym, k] : pb.lo.terms())
        if (sym.kind() == SymbolKind::Quotient && sym.divisor() > 0 &&
            k % sym.divisor() != 0)
          quotientTerms.push_back({sym, k});
      if (quotientTerms.size() != 1)
        continue; // this kind handles exactly one such quotient
      Symbol qsym = quotientTerms.front().first;
      int64_t k = quotientTerms.front().second;
      const QuotientInfo *info = findQuotientInfo(qsym, ctx);
      if (!info)
        continue;
      // Every other term must already be sign-decidable: this kind closes
      // only the gap the quotient's own threshold leaves, the same convention
      // decideResidual uses for terms it drops at a safe floor of zero.
      bool otherUndecided = false;
      for (auto &[sym2, k2] : pb.lo.terms())
        if (!(sym2 == qsym) && !termSignOk(sym2, k2, ctx, probe)) {
          otherUndecided = true;
          break;
        }
      if (otherUndecided)
        continue;

      int64_t c = qsym.divisor();
      // t = ceildiv(g - c0, k) for k > 0, floordiv for k < 0, in
      // APInt(128) so the subtraction and division cannot overflow even at
      // the i64 extremes.
      APInt gA(128, static_cast<uint64_t>(g), /*isSigned=*/true);
      APInt c0A(128, static_cast<uint64_t>(pb.lo.constant()),
                /*isSigned=*/true);
      APInt kA(128, static_cast<uint64_t>(k), /*isSigned=*/true);
      APInt need = gA - c0A;
      APInt tA = llvm::APIntOps::RoundingSDiv(
          need, kA, k > 0 ? APInt::Rounding::UP : APInt::Rounding::DOWN);
      if (tA.getSignificantBits() > 64)
        continue; // threshold does not fit i64
      int64_t t = tA.getSExtValue();

      int64_t boundVal;
      bool overflowed;
      BoundGoal goal = k > 0 ? BoundGoal::AtLeast : BoundGoal::AtMost;
      if (k > 0 && info->isCdiv) {
        int64_t tm1;
        overflowed = llvm::SubOverflow(t, int64_t{1}, tm1) ||
                     llvm::MulOverflow(c, tm1, boundVal) ||
                     llvm::AddOverflow(boundVal, int64_t{1}, boundVal);
      } else if (k > 0) {
        overflowed = llvm::MulOverflow(c, t, boundVal);
      } else if (info->isCdiv) {
        overflowed = llvm::MulOverflow(c, t, boundVal);
      } else {
        int64_t ct;
        overflowed = llvm::MulOverflow(c, t, ct) ||
                     llvm::AddOverflow(ct, c - 1, boundVal);
      }
      if (overflowed)
        continue;

      candidates.push_back(
          {info->dividend, goal, boundVal, ConditionKind::Fact});
      if (k > 0)
        wantSignFloor.push_back({qsym, t});
      else
        wantSignCeil.push_back({qsym, t});
      // Using the quotient's facts inherits what `substituteQuotients`
      // registers for a divisible coefficient: the division facts hold only
      // for a non-negative dividend, and the dividend's own arithmetic
      // carries its wrap obligations. This kind reaches neither through
      // `substituteQuotients`, since the coefficient is deliberately not a
      // multiple of the divisor here.
      wantExtra.push_back({info->dividend, BoundGoal::NonNegative, 0,
                           ConditionKind::Precondition});
      llvm::append_range(wantFactObligations, info->dividendObligations);
    } else { // ResidualGuard
      // Last resort: guard the whole bounded residual directly. Sound
      // whenever every symbol in it is scalar and loop-invariant - which
      // `bound` already guarantees for anything still symbolic here, since a
      // loop-varying symbol would have been substituted by its bound - and
      // reaches the multi-symbol residual no single-symbol candidate can
      // (`MultiSymbolResidualGuard`).
      bool guardable = !pb.lo.isConstant(); // the constant case decided already
      for (auto &[sym, k] : pb.lo.terms())
        if (!sym.value() || isa<ShapedType>(sym.value().getType()) ||
            sym.kind() == SymbolKind::TripCount ||
            sym.kind() == SymbolKind::Quotient) {
          // A Quotient's own `.value()` is the divisor operand's raw,
          // possibly-wrapped SSA value (the dividend `symbolFor` was given,
          // not `QuotientInfo::dividend`'s obligation-checked form) -
          // `materialize`'s Quotient case divides it directly, so guarding
          // it here would read a wrapped numerator exactly like
          // QuotientThreshold exists to prevent. QuotientThreshold is where a
          // Quotient term gets a sound guard, via the translated dividend
          // condition and its own registered wrap obligation; this kind
          // declines rather than risk the untranslated one.
          guardable = false;
          break;
        }
      if (!guardable)
        continue;
      alwaysSucceeds = true;
      candidates.push_back({pb.lo, BoundGoal::AtLeast, g, ConditionKind::Fact});
    }

    CandidateSet trial = acc;
    trial.exactLoopEnd |= wantExactLoopEnd;
    llvm::append_range(trial.exactCdiv, wantExactCdiv);
    llvm::append_range(trial.signFloor, wantSignFloor);
    llvm::append_range(trial.signCeil, wantSignCeil);
    for (const BoundCondition &c : wantExtra)
      if (!llvm::is_contained(trial.extra, c))
        trial.extra.push_back(c);
    for (const Obligation &o : wantFactObligations)
      if (!llvm::is_contained(trial.factObligations, o))
        trial.factObligations.push_back(o);
    Bounds b = bound(d, ctx, trial);
    mergePreconditions(trial, b);
    if (!b.finite || b.exhausted)
      continue;

    if (alwaysSucceeds || decideResidual(b.lo, g, ctx, trial)) {
      bool declined = false;
      for (const BoundCondition &cond : candidates) {
        CandidateResult res = addCandidate(trial, cond, ctx);
        // Exhausted/Declined both just mean this kind's own candidate list
        // doesn't work; that does not abort the whole query, since an
        // earlier kind may already have finished (or a later one still might)
        // - the 5-kind loop is bounded regardless of how many individual kinds
        // fail this way.
        if (res == CandidateResult::Exhausted ||
            res == CandidateResult::Declined) {
          trial = acc; // not expressible: discard and try the next kind
          declined = true;
          break;
        }
      }
      // Always finalize once the residual decides: a candidate established by
      // a dominating assume adds no runtime condition, so the condition set
      // can legitimately be empty - finalize is what turns that into
      // Satisfied rather than ConditionallySatisfied. Recorded, not returned:
      // Satisfied is also strictly better than any ConditionallySatisfied
      // finisher, so prefer it immediately rather than let pickWidest compare
      // an empty condition set against a non-empty one by its generic rules.
      if (!declined) {
        BoundProof proof = finalize(BoundProof::ConditionallySatisfied, trial,
                                    obligations, ctx);
        if (proof.verdict == BoundProof::Satisfied)
          return proof;
        if (proof.verdict != BoundProof::Unknown)
          finishers.push_back(std::move(proof));
        continue; // do not fold this trial into acc (see the loop's own doc)
      }
    }
    // Keep a candidate that strictly improves lo(d) without finishing. Its
    // conditions go with it: a later kind finishes from `acc`.
    std::optional<int64_t> lo = residualConstant(d, ctx, trial);
    if (lo && (!best || *lo > *best) &&
        llvm::all_of(candidates, [&](const BoundCondition &cond) {
          return addCandidate(trial, cond, ctx) == CandidateResult::Accepted;
        })) {
      acc = std::move(trial);
      best = lo;
    }
  }
  if (!finishers.empty())
    return finishers[pickWidest(finishers)];
  return {};
}

//===----------------------------------------------------------------------===//
// Mask queries
//===----------------------------------------------------------------------===//

BoundProof SymbolicBoundsProver::proveTrue(Value v, QueryContext ctx) {
  using V = BoundProof::Verdict;
  ++maskEvaluations;

  // A block argument is where the "look through" must stop. getFinalValue
  // would substitute an iter_arg's init value without inspecting the yield, so
  // a mask initialized `true` and yielding a computed value would read as
  // unconditionally true and unmask an access the mask was guarding.
  if (isa<BlockArgument>(v))
    return {};

  Operation *def = v.getDefiningOp();
  if (!def)
    return {};

  // Shape and width changes preserve "true in every element": a splat
  // replicates one bit, expand_dims/broadcast replicate existing ones, and
  // extending an i1 keeps zero zero and nonzero nonzero.
  if (auto splat = dyn_cast<tt::SplatOp>(def))
    return proveTrue(splat.getSrc(), ctx);
  if (auto expand = dyn_cast<tt::ExpandDimsOp>(def))
    return proveTrue(expand.getSrc(), ctx);
  if (auto bcast = dyn_cast<tt::BroadcastOp>(def))
    return proveTrue(bcast.getSrc(), ctx);
  if (auto ext = dyn_cast<arith::ExtSIOp>(def))
    return proveTrue(ext.getIn(), ctx);
  if (auto ext = dyn_cast<arith::ExtUIOp>(def))
    return proveTrue(ext.getIn(), ctx);

  // A constant mask needs no proof. A dense splat is handled too, since that
  // is what a folded `tt.splat` of a constant becomes.
  if (auto cst = dyn_cast<arith::ConstantOp>(def)) {
    auto boolOf = [](Attribute a) -> std::optional<bool> {
      if (auto b = dyn_cast<BoolAttr>(a))
        return b.getValue();
      if (auto i = dyn_cast<IntegerAttr>(a))
        return i.getValue().getBoolValue();
      if (auto d = dyn_cast<SplatElementsAttr>(a))
        return d.getSplatValue<APInt>().getBoolValue();
      return std::nullopt;
    };
    if (std::optional<bool> b = boolOf(cst.getValue())) {
      BoundProof p;
      p.verdict = *b ? V::Satisfied : V::Refuted;
      return p;
    }
    return {};
  }

  if (auto cmp = dyn_cast<arith::CmpIOp>(def))
    return prove(cmp.getPredicate(), cmp.getLhs(), cmp.getRhs(), ctx);

  // `a && b`. One `Refuted` side refutes the conjunction outright, and
  // does so unconditionally, so it needs nothing from the other side - which
  // is why this precedes the Unknown check.
  if (auto andOp = dyn_cast<arith::AndIOp>(def)) {
    BoundProof a = proveTrue(andOp.getLhs(), ctx);
    BoundProof b = proveTrue(andOp.getRhs(), ctx);
    if (a.verdict == V::Refuted || b.verdict == V::Refuted) {
      BoundProof p;
      p.verdict = V::Refuted;
      return p;
    }
    auto decided = [](V x) {
      return x == V::Satisfied || x == V::ConditionallySatisfied;
    };
    if (!decided(a.verdict) || !decided(b.verdict))
      return {};

    BoundProof p;
    p.verdict = a.verdict == V::Satisfied && b.verdict == V::Satisfied
                    ? V::Satisfied
                    : V::ConditionallySatisfied;
    p.conditions = a.conditions;
    for (const BoundCondition &c : b.conditions) {
      auto it = llvm::find(p.conditions, c);
      if (it == p.conditions.end())
        p.conditions.push_back(c);
      else if (c.kind < it->kind)
        it->kind = c.kind; // one condition, several kinds: the strongest wins
    }
    p.factsUsed = a.factsUsed;
    for (Operation *f : b.factsUsed)
      if (!llvm::is_contained(p.factsUsed, f))
        p.factsUsed.push_back(f);

    // The budgets bound each side separately, so their union can exceed them;
    // an over-budget conjunction is Unknown, not a larger guard.
    auto count = [&](ConditionKind k) {
      return llvm::count_if(
          p.conditions, [&](const BoundCondition &c) { return c.kind == k; });
    };
    if (count(ConditionKind::Fact) > kMaxFactConditions ||
        count(ConditionKind::Guard) > kMaxGuards)
      return {};
    llvm::stable_sort(p.conditions,
                      [](const BoundCondition &x, const BoundCondition &y) {
                        return x.kind < y.kind;
                      });
    return p;
  }

  return {};
}

} // namespace mlir::triton::intel
