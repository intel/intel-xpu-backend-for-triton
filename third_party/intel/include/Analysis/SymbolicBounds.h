//===- SymbolicBounds.h -----------------------------------------*- C++ -*-===//
//
// A goal-directed symbolic bounds prover: decides `lhs pred rhs` for scalar or
// tensor operands inside a loop nest in terms of kernel arguments, program ids
// and `llvm.intr.assume` facts, where the constant-interval
// `IntegerRangeAnalysis` cannot, and when it cannot decide outright it returns
// the runtime conditions under which the comparison holds, as scalars that
// dominate the loop, so a consumer can guard an unmasked copy.
//
// Design: `~/.claude/handoffs/assets/missing-analyses/
// 2026-10-02-symbolic-bounds-prover-design.md` (v11). Section numbers in the
// comments below refer to it.
//
// The prover reasons in the integers; the IR's unflagged `arith` integer ops
// wrap. Every traversed add/subtract/multiply therefore carries a *wrap
// obligation*, discharged statically, folded into the runtime guard, or - when
// neither is possible - degrading the query to `Unknown`. Nothing here assumes
// absence of overflow silently (§4.1).
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
#include <optional>
#include <string>

namespace mlir::triton::intel {

/// A symbol is an integer SSA value the prover does not look through.
/// `TripCount` is phase 3 (design §4.5): no phase-1 path creates one.
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
/// if value and placement agree (§4.1, element correspondence).
using AxisPlacement = SmallVector<int32_t, 4>;

/// Built only by `SymbolicBoundsProver::symbolFor`, so `order` is always
/// assigned and two distinct values can never tie in the sort order.
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
  /// Sort key only: the prover's pre-order index of `value`, from 1.
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
/// structural and rendering is identical across processes. Every arithmetic
/// method is overflow-checked: an overflow sets a sticky flag, and a flagged
/// form is unusable - `normalize` turns it into an `Opaque` symbol, candidate
/// formation rejects it, and a flagged comparison difference ends the query
/// `Unknown` (§4.1, §4.4).
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

/// Budgets are per kind (§4.3), and the kind also fixes the order conditions
/// render and materialize in: facts, then preconditions, then guards.
enum class ConditionKind { Fact, Precondition, Guard };

/// A runtime condition a conditional proof depends on. The subject is an
/// affine form rather than an SSA value because `ub - lb` and the wrap bounds
/// usually have no SSA value; every symbol in it must map to one scalar
/// (rank-0) value that dominates the loop, so `materialize` can place the
/// guard (§4.4).
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

/// Closed by the discharge tiers of §4.1. `Wrap`: a traversed
/// add/subtract/multiply result fits its width. `NonNegative`: `expr >= 0`,
/// from `extui` and from unsigned predicates.
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

/// The four-valued verdict of §4.3. `Refuted` and `Unknown` are not
/// interchangeable: `Refuted` is a positive, unconditional disproof a consumer
/// may act on, `Unknown` means the prover gave up and the IR must be left
/// alone.
struct BoundProof {
  enum Verdict { Satisfied, Refuted, ConditionallySatisfied, Unknown };

  Verdict verdict = Unknown;
  /// Facts, then preconditions, then guards (§4.4).
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

// Phase 3 (design §4.5), not implemented by this plan. Declared for the shape
// its three TTGIR consumers will use:
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
/// `Exhausted`, when a budget is exceeded, which is `Unknown` (§4.3).
enum class CandidateResult { Accepted, Declined, Exhausted };

class SymbolicBoundsProver {
public:
  /// `solver` must already have `IntegerRangeAnalysis` loaded and run, as
  /// `SignednessProver` requires. Collects the assume facts under `root` once
  /// and numbers every value under it for the symbol order.
  SymbolicBoundsProver(const DataFlowSolver &solver, DominanceInfo &domInfo,
                       Operation *root);

  /// The only way to create a `Symbol`: fills `order` from the pre-order
  /// numbering built at construction.
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

  static constexpr unsigned kMaxDepth = 16;
  static constexpr unsigned kMaxTerms = 16;
  static constexpr unsigned kMaxFactConditions = 4;
  static constexpr unsigned kMaxGuards = 8;

private:
  /// The result of bounding an affine form over a loop's iteration space
  /// (§4.2). Side-effect free: evidence the bounding needed comes back here
  /// and the caller merges it.
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
  /// leaks nothing into a later one (§4.3 step 4).
  struct CandidateSet {
    /// 4a: the IV's high bound is `ub - step` rather than `ub - 1`.
    bool exactLoopEnd = false;
    /// 4b: quotients known to divide exactly.
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
  };

  /// §4.2: keeps loop-invariant symbols symbolic, substitutes bounds for
  /// loop-varying ones, collects like terms.
  Bounds bound(const AffineForm &e, QueryContext ctx, const CandidateSet &cs);
  /// Bounds one symbol over the loop's iteration space.
  Bounds symbolBounds(const Symbol &sym, QueryContext ctx,
                      const CandidateSet &cs);
  /// §4.2 step 4: is the residual `lo` at least `g`?
  bool decideResidual(const AffineForm &lo, int64_t g, QueryContext ctx,
                      CandidateSet &cs);
  /// The constant `decideResidual` compares against `g`, when there is one.
  std::optional<int64_t> residualConstant(const AffineForm &d, QueryContext ctx,
                                          CandidateSet &cs);
  /// The single exit of every successful path (§4.3 steps 1-6).
  BoundProof finalize(BoundProof::Verdict onD, CandidateSet cs,
                      ArrayRef<Obligation> obligations, QueryContext ctx);
  /// Adds a candidate condition, or reports why it cannot be added.
  CandidateResult addCandidate(CandidateSet &cs, BoundCondition cond,
                               QueryContext ctx);
  /// Merges the evidence a bounding produced into the candidate set.
  void mergePreconditions(CandidateSet &cs, const Bounds &b) const;
  /// The assume establishing `cond` outright, or null (Task 6).
  Operation *assumedBy(const BoundCondition &cond, QueryContext ctx) const;
  /// A block argument's affine form: the loop IV, an IV-offset iter_arg, a
  /// function argument, else opaque.
  AffineForm leafForBlockArg(BlockArgument arg, QueryContext ctx,
                             SmallVectorImpl<Obligation> &obligations,
                             AxisPlacement placement);

  AffineForm normalizeImpl(Value v, QueryContext ctx,
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
  /// Pre-order index of every integer value under `root`, from 1, so the
  /// symbol order is total and reproducible across processes.
  DenseMap<Value, unsigned> valueOrder;
  /// Per-query: set when a budget is exhausted, which makes the query
  /// `Unknown` (§4.3).
  bool exhausted = false;
};

/// Stable renderings for tests and the test pass.
std::string toString(const AffineForm &af);
std::string toString(const BoundCondition &c);
std::string toString(const BoundProof &p);

} // namespace mlir::triton::intel

#endif // TRITON_INTEL_ANALYSIS_SYMBOLICBOUNDS_H
