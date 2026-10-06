//===- SymbolicBounds.cpp -------------------------------------------------===//
//
// See SymbolicBounds.h and the design document it names. This file implements
// the symbol order, affine-form arithmetic (overflow-checked throughout) and
// scalar normalization; tensor placement, loop induction variables, bounding
// and the candidate search arrive in later tasks.
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
/// the result unusable rather than silently wrapping (§4.4).
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
    int64_t mag = coeff < 0 ? -coeff : coeff;
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
    os << " - " << -c0;
  return out;
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

SymbolicBoundsProver::SymbolicBoundsProver(const DataFlowSolver &solver,
                                           DominanceInfo &domInfo,
                                           Operation *root)
    : solver(solver), domInfo(domInfo), root(root) {
  // Number every value in pre-order, from 1, so the symbol sort is total over
  // distinct values and reproducible across processes: a block's arguments as
  // the walk enters it, then each operation's results in result order. 0 is
  // reserved as "unassigned", which AffineForm::symbol asserts against.
  unsigned next = 1;
  root->walk<WalkOrder::PreOrder>([&](Operation *op) {
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument arg : block.getArguments())
          valueOrder.try_emplace(arg, next++);
    for (OpResult result : op->getResults())
      valueOrder.try_emplace(result, next++);
  });
}

Symbol SymbolicBoundsProver::symbolFor(SymbolKind kind, Value v,
                                       int64_t divisor,
                                       AxisPlacement placement) const {
  auto it = valueOrder.find(v);
  // A value created after construction has no index; give it one past the end
  // so the order stays total. Deterministic because the prover is rebuilt
  // after any mutation (§4.6).
  unsigned order = it != valueOrder.end() ? it->second : valueOrder.size() + 1;
  return Symbol(kind, v, divisor, std::move(placement), order);
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
  // `index` is modelled as 64 bits, as the guard arithmetic does (§4.4).
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
SymbolicBoundsProver::normalizeImpl(Value v, QueryContext ctx,
                                    SmallVectorImpl<Obligation> &obligations,
                                    AxisPlacement placement, unsigned depth) {
  if (depth > kMaxDepth) {
    // Budgets degrade the whole query to Unknown (§4.3).
    exhausted = true;
    return opaque(v, placement);
  }
  if (std::optional<int64_t> cst = getFoldedConstant(v))
    return AffineForm::constant(*cst);

  Operation *def = v.getDefiningOp();
  if (!def) {
    auto blockArg = cast<BlockArgument>(v);
    // Loop induction variables and iter_args arrive in Task 3.
    if (isa_and_nonnull<tt::FuncOp>(blockArg.getOwner()->getParentOp()))
      return AffineForm::symbol(
          symbolFor(SymbolKind::KernelArg, v, 0, placement));
    return opaque(v, placement);
  }

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
          return giveUp(); // product of two symbols: no derived facts (§4.1)
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
        // Widening only: a narrowing cast truncates (§4.1).
        if (bitWidth(op.getIn().getType()) > bitWidth(op.getType()))
          return giveUp();
        return normalizeImpl(op.getIn(), ctx, obligations, placement,
                             depth + 1);
      })
      .Case<arith::ExtUIOp>([&](auto op) {
        // Equal to extsi exactly when the operand is non-negative (§4.1).
        AffineForm in =
            normalizeImpl(op.getIn(), ctx, obligations, placement, depth + 1);
        if (in.overflowed() || exhausted)
          return giveUp();
        obligations.push_back({Obligation::NonNegative, def, in, 0});
        return in;
      })
      .Case<tt::MakeRangeOp>([&](auto op) {
        return AffineForm::symbol(
            symbolFor(SymbolKind::Lane, op.getResult(), 0, placement));
      })
      .Case<tt::SplatOp>([&](auto op) {
        // A scalar has the same value in every element, so the placement of
        // the result says nothing about the operand (§4.1).
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
        // A layout change moves no element (§4.1).
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
      // axis tracking through them is deferred (§4.1).
      .Default([&](Operation *) { return giveUp(); });
}

//===----------------------------------------------------------------------===//
// Bounding and decision
//===----------------------------------------------------------------------===//

/// Constant bounds of one symbol, for the loop-free queries of this task.
/// `Lane` is exact; anything else falls back to the range analysis, and a
/// symbol with no inferable range is unbounded, which makes the query
/// `Unknown`.
std::optional<std::pair<int64_t, int64_t>>
SymbolicBoundsProver::symbolConstantBounds(const Symbol &sym) const {
  if (sym.kind() == SymbolKind::Lane) {
    auto rangeOp = cast<tt::MakeRangeOp>(sym.value().getDefiningOp());
    // Signed attribute getters: the generated getStart()/getEnd() return
    // uint32_t although the attributes are signed, so a range [-4, 0) would
    // otherwise bound as positive (§4.1).
    int64_t start = rangeOp.getStartAttr().getInt();
    int64_t end = rangeOp.getEndAttr().getInt();
    return std::make_pair(start, end - 1);
  }
  if (std::optional<ConstantIntRanges> r = collectRange(solver, sym.value()))
    return std::make_pair(r->smin().getSExtValue(), r->smax().getSExtValue());
  return std::nullopt;
}

/// Bounds `e` by replacing every symbol with its constant bounds, taking the
/// low or high end per the sign of the coefficient. Loop-varying symbols
/// arrive in Task 3; here every symbol is bounded from constants.
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

  // Reduce the predicate to `d >= g` with d = rhs - lhs. eq/ne are declined:
  // the predicate whitelist in RemoveMasks exists because eq produced unsound
  // versioning conditions (#7791).
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
  // An overflowed difference carries no information: INT64_MAX < INT64_MIN
  // would otherwise wrap to d = 1 and read as a proof (§4.1).
  if (d.overflowed())
    return {};

  std::optional<std::pair<int64_t, int64_t>> b = boundConstant(d);
  if (!b)
    return {};

  BoundProof proof;
  if (b->first >= g)
    proof.verdict = BoundProof::Satisfied;
  else if (b->second < g)
    // Refutation needs a constant hi, which boundConstant always gives here.
    proof.verdict = BoundProof::Refuted;
  return proof;
}

} // namespace mlir::triton::intel
