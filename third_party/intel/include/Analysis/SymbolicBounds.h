//===- SymbolicBounds.h -----------------------------------------*- C++ -*-===//
//
// A goal-directed symbolic bounds prover: decides `lhs pred rhs` for scalar or
// tensor operands inside a loop nest in terms of kernel arguments, program ids
// and `llvm.intr.assume` facts, where the constant-interval
// `IntegerRangeAnalysis` cannot, and when it cannot decide outright it returns
// the runtime conditions under which the comparison holds, as scalars that
// dominate the loop, so a consumer can guard an unmasked copy.
//
// The prover reasons in the integers; the IR's unflagged `arith` integer ops
// wrap. Every traversed add/subtract/multiply therefore carries a *wrap
// obligation*, discharged statically, folded into the runtime guard, or - when
// neither is possible - degrading the query to `Unknown`. Nothing here assumes
// absence of overflow silently.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_INTEL_ANALYSIS_SYMBOLICBOUNDS_H
#define TRITON_INTEL_ANALYSIS_SYMBOLICBOUNDS_H

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include <map>
#include <optional>
#include <string>

namespace mlir::triton::intel {

/// A symbol is an integer SSA value the prover does not look through.
/// `TripCount` is reserved for the trip-count API sketched below, which is
/// not implemented; nothing creates one yet.
enum class SymbolKind {
  KernelArg,
  ProgramId,
  NumPrograms,
  LoopIV,
  Lane,
  Quotient,
  TripCount,
  Opaque
};

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
  /// The IV for `LoopIV`, the `make_range` result for `Lane`, the dividend `X`
  /// for `Quotient`; for `TripCount` the loop's IV, used only as a key, since
  /// no SSA value holds the count.
  Value value() const { return value_; }
  /// `Quotient`: the divisor of `divsi X, divisor`. `TripCount`: the step.
  int64_t divisor() const { return divisor_; }
  const AxisPlacement &placement() const { return placement_; }
  /// Sort key only, unique per distinct value within one prover; see
  /// `SymbolicBoundsProver::symbolFor`.
  unsigned order() const { return order_; }

  bool operator==(const Symbol &o) const {
    return kind_ == o.kind_ && value_ == o.value_ && divisor_ == o.divisor_ &&
           placement_ == o.placement_;
  }
  bool operator!=(const Symbol &o) const { return !(*this == o); }
  /// Total order over distinct values: (kind, order, divisor, placement).
  bool operator<(const Symbol &o) const;

private:
  friend class SymbolicBoundsProver;
  Symbol(SymbolKind kind, Value value, int64_t divisor, AxisPlacement placement,
         unsigned order)
      : kind_(kind), value_(value), divisor_(divisor),
        placement_(std::move(placement)), order_(order) {}

  SymbolKind kind_;
  Value value_;
  int64_t divisor_;
  AxisPlacement placement_;
  unsigned order_;
};

/// `c0 + sum(ci * si)` over the integers, with int64_t coefficients. Terms are
/// kept sorted by `Symbol::operator<` with nonzero coefficients, so equality is
/// structural and, for values the prover numbered at construction, rendering is
/// identical across processes. Every arithmetic
/// method is overflow-checked: an overflow sets a sticky flag, and a flagged
/// form is unusable - `normalize` turns it into an `Opaque` symbol, candidate
/// formation rejects it, and a flagged comparison difference ends the query
/// `Unknown`.
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
  /// identity is the SSA value plus kind, divisor and placement, never a `loc`
  /// name.
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

enum class BoundGoal {
  NonNegative,
  StrictlyPositive,
  DivisibleBy,
  AtLeast,
  AtMost
};

/// Budgets are per kind, and the kind also fixes the order conditions
/// render and materialize in: facts, then preconditions, then guards.
enum class ConditionKind { Fact, Precondition, Guard };

/// A runtime condition a conditional proof depends on. The subject is an
/// affine form rather than an SSA value because `ub - lb` and the wrap bounds
/// usually have no SSA value; every symbol in it must map to one scalar
/// (rank-0) value that dominates the loop, so `materialize` can place the
/// guard.
struct BoundCondition {
  AffineForm expr;
  BoundGoal goal;
  /// Divisor for `DivisibleBy`, bound for `AtLeast`/`AtMost`.
  int64_t c = 0;
  /// Not part of identity: the same condition can arrive as a fact in one
  /// proof and as a guard in another, and the stronger kind wins.
  ConditionKind kind = ConditionKind::Fact;

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
  /// Facts, then preconditions, then guards.
  SmallVector<BoundCondition, 4> conditions;
  /// The `llvm.intr.assume` operations the proof consulted.
  SmallVector<Operation *, 4> factsUsed;
};

/// `loop` is nullable: a consumer may ask about a comparison outside any loop,
/// where every symbol is loop-invariant.
struct QueryContext {
  /// The program point the facts must be certain to execute at.
  Operation *at = nullptr;
  scf::ForOp loop;
};

// Not implemented: a symbolic trip-count API, in the shape its TTGIR
// consumers would use:
//   struct SymbolicTripCount { AffineForm count;
//                              SmallVector<BoundCondition> preconditions;
//                              SmallVector<Obligation> obligations; };
//   std::optional<SymbolicTripCount> symbolicTripCount(scf::ForOp loop);
//   std::optional<int64_t> minTripCount(scf::ForOp loop);
//   BoundProof tripCountAtLeast(scf::ForOp loop, int64_t n);

/// Applying a candidate has three outcomes and only one ends the query:
/// `Accepted`; `Declined`, when the condition is not expressible (an
/// unnormalizable divisibility, a non-scalar or non-dominating subject), in
/// which case the trial is discarded and the search continues; and
/// `Exhausted`, when a budget is exceeded, which is `Unknown`.
enum class CandidateResult { Accepted, Declined, Exhausted };

class SymbolicBoundsProver {
public:
  /// `solver` must already have `IntegerRangeAnalysis` loaded and run, as
  /// `SignednessProver` requires. Collects the assume facts under `root` once
  /// and numbers every value of its enclosing function (of `root` itself, for a
  /// module) for the symbol order, so a prover rooted at a loop still orders
  /// the function arguments and everything else a query can reach.
  SymbolicBoundsProver(const DataFlowSolver &solver, DominanceInfo &domInfo,
                       Operation *root);

  /// The only way to create a `Symbol`: fills `order` from the pre-order
  /// numbering built at construction. A value that numbering does not cover -
  /// outside the numbered scope, or created after construction - takes the
  /// next unused order the first time it is seen, so distinct values never
  /// tie. Those orders follow first-query order rather than IR position: that
  /// can change where such a value's term renders and, among equal
  /// coefficients, which sound candidate guard `prove` picks, never whether a
  /// verdict is sound.
  Symbol symbolFor(SymbolKind kind, Value v, int64_t divisor = 0,
                   AxisPlacement placement = {}) const;

  /// Decides `lhs pred rhs` at `ctx`. Normalizes both sides and delegates to
  /// the affine-form overload.
  BoundProof prove(arith::CmpIPredicate pred, Value lhs, Value rhs,
                   QueryContext ctx);

  /// Normalizes `v` at `ctx` and appends the wrap obligations of every
  /// operation looked through. Never fails: an unsupported operation becomes
  /// an `Opaque` symbol.
  AffineForm normalize(Value v, QueryContext ctx,
                       SmallVectorImpl<Obligation> &obligations);

  /// Decides whether the i1 (or i1 tensor) `v` is true in every element at
  /// `ctx`. Looks through `tt.splat`, `tt.expand_dims`, `tt.broadcast` and
  /// `arith.ext*` only, and deliberately NOT through `getFinalValue`, which
  /// substitutes an iter_arg's init value without inspecting the yield: a
  /// loop-carried mask initialized `true` and yielding `false` would read as
  /// always true. Any block argument is therefore `Unknown`.
  BoundProof proveTrue(Value v, QueryContext ctx);

  /// Diagnostic, for tests: how many mask nodes `proveTrue` has evaluated over
  /// this prover's lifetime.
  unsigned numMaskEvaluations() const { return maskEvaluations; }

  static constexpr unsigned kMaxDepth = 16;
  static constexpr unsigned kMaxTerms = 16;
  static constexpr unsigned kMaxFactConditions = 4;
  static constexpr unsigned kMaxGuards = 8;
  /// Bounds on `proveTrue`'s walk over the mask expression itself, as opposed
  /// to the comparisons it hands to `prove`. The depth is for stack safety, so
  /// it is far above `kMaxDepth`: reusing that would turn a mask with more than
  /// 16 nested conjuncts from decided to `Unknown`. The visits bound how many
  /// mask nodes one query evaluates; the work inside each comparison's `prove`
  /// is not counted against it.
  static constexpr unsigned kMaxMaskDepth = 64;
  static constexpr unsigned kMaxMaskVisits = 1024;

private:
  /// The result of bounding an affine form over a loop's iteration space.
  /// Side-effect free: evidence the bounding needed comes back here and the
  /// caller merges it.
  struct Bounds {
    AffineForm lo, hi;
    bool isVarying = false;
    /// False when a bound is not finite over loop-invariant symbols.
    bool finite = true;
    /// `lo` or `hi` overflowed or exceeded kMaxTerms after substitution.
    bool exhausted = false;
    SmallVector<BoundCondition, 4> preconditions;
    SmallVector<Obligation, 4> factObligations;
    SmallVector<Operation *, 4> assumes;
  };

  /// The candidate conditions a proof attempt has accumulated. Every trial
  /// runs on a copy and is committed only if it succeeds, so a failed attempt
  /// leaks nothing into a later one.
  struct CandidateSet {
    /// Exact loop end: the IV's high bound is `ub - step` rather than `ub - 1`.
    bool exactLoopEnd = false;
    /// Exact cdiv: quotients known to divide exactly.
    SmallVector<Symbol, 2> exactCdiv;
    /// Accepted candidate conditions.
    SmallVector<BoundCondition, 4> facts;
    /// Preconditions of the facts actually used.
    SmallVector<BoundCondition, 4> extra;
    /// Assume ops that established a candidate instead of a runtime condition.
    SmallVector<Operation *, 4> assumes;
    /// Obligations inherited from the dividends of quotient facts used.
    SmallVector<Obligation, 4> factObligations;
    /// A bound overflowed while closing an obligation: the verdict is Unknown.
    bool exhausted = false;
    /// Term-sign and quotient-threshold candidates: trial hypotheses this
    /// attempt is testing on a residual term, consulted by `decideResidual`
    /// ahead of the general fact/range check. `signFloor` is `sym >= bound`
    /// (used when the term's own coefficient in the residual is positive);
    /// `signCeil` is `sym <= bound` (negative coefficient). The term-sign
    /// candidate populates these with 0 (`NonNegative`/`AtMost(_,0)`) or 1/-1
    /// (`StrictlyPositive`/`AtMost(_,-1)`); the quotient-threshold candidate
    /// with the quotient threshold translated from the dividend condition it
    /// emits, so the bound the dividend condition justifies is also usable
    /// internally, without being emitted twice.
    SmallVector<std::pair<Symbol, int64_t>, 2> signFloor;
    SmallVector<std::pair<Symbol, int64_t>, 2> signCeil;
  };

  /// The facts of one quotient symbol.
  struct QuotientInfo {
    /// True for the `(X + c - 1) / c` shape, whose facts are sharper.
    bool isCdiv = false;
    /// The dividend the facts refer to: `X'` for a cdiv, else `X`.
    AffineForm dividend;
    /// The wrap obligations of the dividend's own arithmetic, inherited by
    /// any proof that uses these facts.
    SmallVector<Obligation, 4> dividendObligations;
  };

  /// Substitutes quotient terms by their bounds so the dividend can cancel:
  /// the bounds that minimize `e`, or with `maximize` those that maximize it.
  std::optional<AffineForm> substituteQuotients(const AffineForm &e,
                                                QueryContext ctx,
                                                CandidateSet &cs,
                                                bool maximize);
  const QuotientInfo *findQuotientInfo(const Symbol &sym,
                                       QueryContext ctx) const;
  /// The innermost enclosing `scf.for` of `v`, as a map key; null when `v` is
  /// invariant to every enclosing loop.
  static Operation *varyingLoopKey(Value v);

  /// Keeps loop-invariant symbols symbolic, substitutes bounds for
  /// loop-varying ones, collects like terms.
  Bounds bound(const AffineForm &e, QueryContext ctx, const CandidateSet &cs);
  /// Bounds one symbol over the loop's iteration space.
  Bounds symbolBounds(const Symbol &sym, QueryContext ctx,
                      const CandidateSet &cs);
  /// Is the residual `lo` at least `g`?
  bool decideResidual(const AffineForm &lo, int64_t g, QueryContext ctx,
                      CandidateSet &cs);
  /// The fact/range half of a residual term's sign check: an applicable
  /// assume, or the symbol's own constant range. Excludes `cs.signFloor` /
  /// `signCeil`, which `decideResidual` checks first; records provenance
  /// into `cs` on success.
  bool termSignOk(const Symbol &sym, int64_t k, QueryContext ctx,
                  CandidateSet &cs);
  /// The constant `decideResidual` compares against `g`, when there is one.
  std::optional<int64_t> residualConstant(const AffineForm &d, QueryContext ctx,
                                          CandidateSet &cs);
  /// Tier 1: discharges an obligation from operand bounds, or from
  /// the loop's no-overflow rule.
  bool dischargeTier1(const Obligation &o, QueryContext ctx, CandidateSet &cs);
  /// True for `iv + c` with 0 <= c <= step, which the scf.for contract
  /// discharges for free.
  bool isLoopIvPlusSmallConstant(arith::AddIOp add, QueryContext ctx,
                                 const CandidateSet &cs);
  /// Tier 2: the runtime guards that close an open obligation.
  void guardsForObligation(const Obligation &o, QueryContext ctx,
                           CandidateSet &cs,
                           SmallVectorImpl<BoundCondition> &out);

  /// True when the symbols' constant ranges alone imply `cond`.
  bool impliedByRanges(const BoundCondition &cond) const;

  /// The single exit of every successful path.
  BoundProof finalize(BoundProof::Verdict onD, CandidateSet cs,
                      ArrayRef<Obligation> obligations, QueryContext ctx);
  /// Adds a candidate condition, or reports why it cannot be added.
  CandidateResult addCandidate(CandidateSet &cs, BoundCondition cond,
                               QueryContext ctx);
  /// Merges the evidence a bounding produced into the candidate set.
  void mergePreconditions(CandidateSet &cs, const Bounds &b) const;
  /// One `llvm.intr.assume`-derived fact, normalized onto its subject.
  struct Fact {
    BoundGoal goal;
    int64_t c;
    Operation *assume;
  };

  /// Builds the normalized fact index, keyed by the subject each fact is
  /// about rather than by the comparison's immediate operands.
  void buildFactIndex();
  ArrayRef<Fact> factsFor(Value v) const;
  /// The assume establishing `cond` outright, or null.
  Operation *assumedBy(const BoundCondition &cond, QueryContext ctx) const;
  /// The constant range of `v`, recording into `assumes` every assume that
  /// could have narrowed it (over-approximate provenance, never an omission).
  std::optional<std::pair<int64_t, int64_t>>
  rangeOf(Value v, QueryContext ctx,
          SmallVectorImpl<Operation *> *assumes = nullptr) const;
  /// A block argument's affine form: the loop IV, an IV-offset iter_arg, a
  /// function argument, else opaque.
  AffineForm leafForBlockArg(BlockArgument arg, QueryContext ctx,
                             SmallVectorImpl<Obligation> &obligations,
                             AxisPlacement placement);

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
  DominanceInfo &domInfo;
  Operation *root;
  /// The operation whose values are numbered: `root` if it is a function,
  /// else its enclosing function, else `root` (a module).
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
  /// Quotient facts, keyed by (symbol, the loop in which its dividend
  /// varies), so one entry serves every context the quotient is reached from.
  std::map<std::pair<Symbol, Operation *>, QuotientInfo> quotientInfo;
  /// Assume facts, keyed by the subject value they constrain.
  DenseMap<Value, SmallVector<Fact, 2>> factIndex;

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
  /// Mask nodes `proveTrue` has evaluated; see `numMaskEvaluations`.
  unsigned maskEvaluations = 0;
};

/// Normalizes a condition in place: folds the constant into the bound,
/// divides a single-symbol ordered condition by |k| rounding inward, and
/// reduces `DivisibleBy` through gcd. Returns false when the condition is
/// unsatisfiable or states a congruence `BoundGoal` cannot express; the caller
/// then declines the candidate or leaves the obligation open.
bool normalizeCondition(BoundCondition &cond);

/// Builds the i64 guard for `conds` immediately before `before`.
/// Asserts that every symbol is a scalar (rank-0) SSA value that properly
/// dominates `before`; the prover never produces a `LoopIV`, `Lane` or
/// tensor-valued subject. "The guard cannot wrap" is a checked guarantee, not
/// an assumption: a form that passes the static fit check uses plain i64
/// arithmetic, and one that does not is paired with overflow predicates that
/// make the guard false rather than wrong.
Value materialize(ArrayRef<BoundCondition> conds, Operation *before,
                  OpBuilder &builder);

/// Stable renderings for tests and the test pass.
std::string toString(const AffineForm &af);
std::string toString(const BoundCondition &c);
std::string toString(const BoundProof &p);

} // namespace mlir::triton::intel

#endif // TRITON_INTEL_ANALYSIS_SYMBOLICBOUNDS_H
