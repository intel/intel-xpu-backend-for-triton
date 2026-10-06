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
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
namespace tt = mlir::triton;

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
  unsigned rank = 0;
  if (auto shaped = dyn_cast<ShapedType>(v.getType()))
    rank = shaped.getRank();
  AxisPlacement identity;
  for (unsigned i = 0; i < rank; ++i)
    identity.push_back(i);
  return normalizeImpl(v, ctx, obligations, identity, 0);
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
      .Default([&](Operation *) { return giveUp(); });
}

} // namespace mlir::triton::intel
