//===- SymbolicBounds.h -----------------------------------------*- C++ -*-===//
//
// A goal-directed symbolic bounds prover: decides `lhs pred rhs` for scalar or
// tensor operands in terms of kernel arguments and program ids, where the
// constant-interval `IntegerRangeAnalysis` cannot, and when it cannot decide
// outright it returns the runtime conditions under which the comparison holds,
// as scalars that dominate the loop, so a consumer can guard an unmasked copy.
//
// The prover reasons in the integers; the IR's unflagged `arith` integer ops
// wrap. Every traversed add/subtract/multiply therefore carries a *wrap
// obligation*, discharged statically, folded into the runtime guard, or - when
// neither is possible - degrading the query to `Unknown`. Nothing here assumes
// absence of overflow silently.
//
// Coefficients, bounds and constants are `int64_t`, so integers wider than 64
// bits are outside the domain: `prove` answers `Unknown`, `normalize` returns
// an opaque symbol, no range is read for such a value, and `materialize`
// requires every condition subject to be at most 64 bits wide. `index` counts
// as 64 bits.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_INTEL_ANALYSIS_SYMBOLICBOUNDS_H
#define TRITON_INTEL_ANALYSIS_SYMBOLICBOUNDS_H

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include <map>
#include <optional>
#include <string>

namespace mlir::triton::intel {

/// A symbol is an integer SSA value the prover does not look through.
enum class SymbolKind { KernelArg, ProgramId, NumPrograms, Lane, Opaque };

/// placement[i] = the result-tensor axis occupied by the symbol's own axis i,
/// one entry per axis of the symbol's value, size-1 axes included. Empty for
/// scalars. Two occurrences denote the same element - and so may cancel - only
/// if value and placement agree.
using AxisPlacement = SmallVector<int32_t, 4>;

/// Built only by `SymbolicBoundsProver::symbolFor`, so `order` is always
/// nonzero and two distinct values never tie in the sort order: a value the
/// prover numbered at construction keeps its IR pre-order index, and any other
/// value takes the next unused order when first seen and keeps it.
class Symbol {
public:
  SymbolKind kind() const { return kind_; }
  /// The value the symbol stands for: the `make_range` result for `Lane`.
  Value value() const { return value_; }
  const AxisPlacement &placement() const { return placement_; }
  /// Sort key only, unique per distinct value within one prover; see
  /// `SymbolicBoundsProver::symbolFor`.
  unsigned order() const { return order_; }

  bool operator==(const Symbol &o) const {
    return kind_ == o.kind_ && value_ == o.value_ && placement_ == o.placement_;
  }
  bool operator!=(const Symbol &o) const { return !(*this == o); }
  /// Total order over distinct values: (kind, order, placement).
  bool operator<(const Symbol &o) const;

private:
  friend class SymbolicBoundsProver;
  Symbol(SymbolKind kind, Value value, AxisPlacement placement, unsigned order)
      : kind_(kind), value_(value), placement_(std::move(placement)),
        order_(order) {}

  SymbolKind kind_;
  Value value_;
  AxisPlacement placement_;
  unsigned order_;
};

/// `c0 + sum(ci * si)` over the integers, with int64_t coefficients. Terms are
/// kept sorted by `Symbol::operator<` with nonzero coefficients, so equality is
/// structural and, for values the prover numbered at construction, rendering is
/// identical across processes. Every arithmetic
/// method is overflow-checked: an overflow sets a sticky flag, and a flagged
/// form is unusable - `normalize` turns it into an `Opaque` symbol, and a
/// flagged comparison difference ends the query `Unknown`.
class AffineForm {
public:
  AffineForm() = default;
  static AffineForm constant(int64_t c);
  static AffineForm symbol(Symbol s);

  AffineForm add(const AffineForm &o) const;
  AffineForm sub(const AffineForm &o) const;
  AffineForm scale(int64_t k) const;

  bool overflowed() const { return overflowed_; }
  bool isConstant() const { return terms_.empty(); }
  int64_t constant() const { return c0_; }
  ArrayRef<std::pair<Symbol, int64_t>> terms() const { return terms_; }
  unsigned numTerms() const { return terms_.size(); }

  /// Structural: same constant, same (symbol, coefficient) terms. Symbol
  /// identity is the SSA value plus kind and placement, never a `loc` name.
  bool operator==(const AffineForm &o) const;

  /// Internal: builds a form whose terms are already sorted and nonzero,
  /// carrying the sticky overflow flag of the arithmetic that produced them.
  static AffineForm
  makeChecked(int64_t c0, SmallVector<std::pair<Symbol, int64_t>, 4> terms,
              bool overflowed);

private:
  int64_t c0_ = 0;
  bool overflowed_ = false;
  SmallVector<std::pair<Symbol, int64_t>, 4> terms_;
};

enum class BoundGoal { NonNegative, AtLeast, AtMost };

/// A runtime condition a conditional proof depends on. The subject is an
/// affine form rather than an SSA value because the wrap bounds usually have
/// no SSA value; every symbol in it must map to one scalar (rank-0) value that
/// dominates the loop, so `materialize` can place the guard.
struct BoundCondition {
  AffineForm expr;
  BoundGoal goal;
  /// Bound for `AtLeast`/`AtMost`.
  int64_t c = 0;

  bool operator==(const BoundCondition &o) const {
    return expr == o.expr && goal == o.goal && c == o.c;
  }
};

/// Closed statically (tier 1) or by a runtime guard (tier 2). `Wrap`: a
/// traversed add/subtract/multiply result fits its width. `NonNegative`:
/// `expr >= 0`, from `extui` and from unsigned predicates.
struct Obligation {
  enum Kind { Wrap, NonNegative };

  Kind kind;
  /// The operation that created it.
  Operation *op;
  /// The result's affine form for `Wrap`, the form that must be >= 0 for
  /// `NonNegative`.
  AffineForm expr;
  /// `Wrap` only.
  unsigned width = 0;

  /// `finalize`'s worklist deduplicates on this.
  bool operator==(const Obligation &o) const {
    return kind == o.kind && op == o.op && expr == o.expr && width == o.width;
  }
};

/// The four-valued verdict. `Refuted` and `Unknown` are not
/// interchangeable: `Refuted` is a positive, unconditional disproof a consumer
/// may act on, `Unknown` means the prover gave up and the IR must be left
/// alone.
struct BoundProof {
  enum Verdict { Satisfied, Refuted, ConditionallySatisfied, Unknown };

  Verdict verdict = Unknown;
  /// The runtime guards that close the obligations left open.
  SmallVector<BoundCondition, 4> conditions;
};

/// `loop` is nullable: a consumer may ask about a comparison outside any loop,
/// where every symbol is loop-invariant.
struct QueryContext {
  /// The program point of the query.
  Operation *at = nullptr;
  scf::ForOp loop;
};

class SymbolicBoundsProver {
public:
  /// `solver` must already have `IntegerRangeAnalysis` loaded and run, as
  /// `SignednessProver` requires. Numbers every value of its enclosing
  /// function (of `root` itself, for a module) for the symbol order, so a
  /// prover rooted at a loop still orders the function arguments and
  /// everything else a query can reach.
  SymbolicBoundsProver(const DataFlowSolver &solver, Operation *root);

  /// The only way to create a `Symbol`: fills `order` from the pre-order
  /// numbering built at construction. A value that numbering does not cover -
  /// outside the numbered scope, or created after construction - takes the
  /// next unused order the first time it is seen, so distinct values never
  /// tie. Those orders follow first-query order rather than IR position: that
  /// can change where such a value's term renders, never whether a verdict is
  /// sound.
  Symbol symbolFor(SymbolKind kind, Value v,
                   AxisPlacement placement = {}) const;

  /// Decides `lhs pred rhs` at `ctx`: normalizes both sides, reduces the
  /// predicate to a bound on their difference, decides it, and closes the
  /// obligations of what it looked through, statically or as runtime guards.
  /// `Unknown` for operands wider than 64 bits.
  BoundProof prove(arith::CmpIPredicate pred, Value lhs, Value rhs,
                   QueryContext ctx);

  /// Normalizes `v` at `ctx` and appends the wrap obligations of every
  /// operation looked through. Never fails: an unsupported operation, or a
  /// value wider than 64 bits, becomes an `Opaque` symbol.
  AffineForm normalize(Value v, QueryContext ctx,
                       SmallVectorImpl<Obligation> &obligations);

  static constexpr unsigned kMaxDepth = 16;
  static constexpr unsigned kMaxTerms = 16;
  static constexpr unsigned kMaxGuards = 8;

private:
  /// The result of bounding an affine form over a loop's iteration space.
  struct Bounds {
    AffineForm lo, hi;
    bool isVarying = false;
    /// False when a bound is not finite over loop-invariant symbols.
    bool finite = true;
    /// `lo` or `hi` overflowed or exceeded kMaxTerms after substitution.
    bool exhausted = false;
  };

  /// Keeps loop-invariant symbols symbolic, substitutes bounds for
  /// loop-varying ones, collects like terms.
  Bounds bound(const AffineForm &e, QueryContext ctx);
  /// Bounds one symbol over the loop's iteration space.
  Bounds symbolBounds(const Symbol &sym, QueryContext ctx);
  /// Is the residual `lo` at least `g`?
  bool decideResidual(const AffineForm &lo, int64_t g, QueryContext ctx);
  /// A residual term's sign check: the symbol's own constant range.
  bool termSignOk(const Symbol &sym, int64_t k, QueryContext ctx);
  /// Tier 1: discharges an obligation from operand bounds. Sets `unknown`
  /// when the obligation can be neither discharged nor guarded.
  bool dischargeTier1(const Obligation &o, QueryContext ctx, bool &unknown);
  /// Tier 2: the runtime guards that close an open obligation. Sets
  /// `unknown` when none can.
  void guardsForObligation(const Obligation &o, QueryContext ctx,
                           SmallVectorImpl<BoundCondition> &out, bool &unknown);

  /// True when the symbols' constant ranges alone imply `cond`.
  bool impliedByRanges(const BoundCondition &cond) const;

  /// The single exit of every successful path.
  BoundProof finalize(BoundProof::Verdict onD, ArrayRef<Obligation> obligations,
                      QueryContext ctx);
  /// The constant range of `v`.
  std::optional<std::pair<int64_t, int64_t>> rangeOf(Value v) const;
  /// A block argument's affine form: a function argument, else opaque.
  AffineForm leafForBlockArg(BlockArgument arg, AxisPlacement placement);

  /// The memo wrapper every recursive normalization goes through.
  AffineForm normalizeImpl(Value v, QueryContext ctx,
                           SmallVectorImpl<Obligation> &obligations,
                           AxisPlacement placement, unsigned depth);
  /// The normalization itself, called only on a memo miss.
  AffineForm normalizeUncached(Value v, QueryContext ctx,
                               SmallVectorImpl<Obligation> &obligations,
                               AxisPlacement placement, unsigned depth);
  /// An `Opaque` symbol for a value the prover does not look through.
  AffineForm opaque(Value v, AxisPlacement placement) const;
  /// The identity placement [0..rank-1] of `v`'s type, empty for a scalar.
  static AxisPlacement identityPlacement(Value v);
  /// Constant bounds of one symbol: exact for `Lane`, else from the range
  /// analysis; nullopt when no range can be inferred.
  std::optional<std::pair<int64_t, int64_t>>
  symbolConstantBounds(const Symbol &sym) const;
  /// Bounds an affine form from constants alone; nullopt on an unbounded
  /// symbol or on overflow.
  std::optional<std::pair<int64_t, int64_t>>
  boundConstant(const AffineForm &e) const;
  /// Records the wrap obligation of one traversed arithmetic operation.
  void recordWrap(Operation *op, const AffineForm &result,
                  SmallVectorImpl<Obligation> &obligations) const;

  const DataFlowSolver &solver;
  /// The operation whose values are numbered: the root if it is a function,
  /// else its enclosing function, else the root (a module).
  Operation *scope;
  /// The symbol order of every value seen so far: the pre-order index of each
  /// value under `scope`, from 1, assigned at construction, then the first-seen
  /// values after them. An order never changes once assigned. Mutable because
  /// `symbolFor` is const and extends it.
  mutable DenseMap<Value, unsigned> valueOrder;
  /// The next order to hand out.
  mutable unsigned nextOrder = 1;
  /// Per-query: set when a budget is exhausted, which makes the query
  /// `Unknown`.
  bool exhausted = false;

  /// The normalization memo. Keyed by placement as well as value,
  /// because the same value on two axes does not normalize to the same form.
  struct MemoKey {
    const void *value;
    Operation *loop;
    AxisPlacement placement;
    bool operator<(const MemoKey &o) const {
      if (value != o.value)
        return value < o.value;
      if (loop != o.loop)
        return loop < o.loop;
      return std::lexicographical_compare(placement.begin(), placement.end(),
                                          o.placement.begin(),
                                          o.placement.end());
    }
  };
  /// `height` is the deepest recursion below the value. A hit re-appends the
  /// obligations and re-sets the flag, and a hit at depth `d` is exhausted
  /// when `d + height > kMaxDepth`: a subtree normalized near the root must
  /// not bypass the depth cap when it is reused below a deep chain.
  struct MemoEntry {
    AffineForm af;
    SmallVector<Obligation, 4> obligations;
    bool exhausted = false;
    unsigned height = 0;
  };
  std::map<MemoKey, MemoEntry> memo;
  /// The deepest `depth` reached since the enclosing memo miss began, which is
  /// what makes `height` computable without threading a return value through
  /// every recursive case.
  unsigned deepest = 0;
};

/// Normalizes a condition in place: folds the constant into the bound and
/// divides a single-symbol condition by |k| rounding inward. Returns false when
/// the condition is unsatisfiable or not expressible; the caller then leaves
/// the obligation open.
bool normalizeCondition(BoundCondition &cond);

/// Builds the i64 guard for `conds` immediately before `before`.
/// Asserts that every symbol is a scalar (rank-0) SSA value that properly
/// dominates `before`; the prover never produces a `Lane` or tensor-valued
/// subject. "The guard cannot wrap" is a checked guarantee, not
/// an assumption: a form that passes the static fit check uses plain i64
/// arithmetic, and one that does not is paired with overflow predicates that
/// make the guard false rather than wrong. Every symbol must be at most 64 bits
/// wide (asserted): there is no way to decline, so that is the caller's
/// precondition.
Value materialize(ArrayRef<BoundCondition> conds, Operation *before,
                  OpBuilder &builder);

/// Stable renderings for tests and the test pass.
std::string toString(const AffineForm &af);
std::string toString(const BoundCondition &c);
std::string toString(const BoundProof &p);

} // namespace mlir::triton::intel

#endif // TRITON_INTEL_ANALYSIS_SYMBOLICBOUNDS_H
