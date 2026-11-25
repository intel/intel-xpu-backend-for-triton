#include "intel/include/Analysis/SignednessProver.h"
#include "intel/include/Analysis/Range.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-intel-signedness-prover"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = mlir::triton;

namespace {

using tt::intel::Goal;
using tt::intel::Proof;
using tt::intel::RequiredCondition;

/// Goal-directed prover for the sign of a value, backing
/// tt::intel::proveSign().
///
/// A forward abstract interpreter answers "what is this value's range"; this
/// answers "what would make this value non-negative", which is the question
/// speculation needs. It is not a second range analysis: every step consults
/// the one it is given, and only decomposes where that analysis has no answer.
class SignednessProver {
public:
  explicit SignednessProver(const DataFlowSolver &solver) : solver(solver) {}

  Proof prove(Value v, Goal goal, unsigned depth = 0);

private:
  /// Chains deeper than this are abandoned rather than recursed into. A
  /// backstop against pathological IR, not a tuning knob: the chains this fires
  /// on in practice are under ten operations deep.
  static constexpr unsigned kMaxDepth = 16;

  /// Step 1: what the range analysis already knows. Returns std::nullopt when
  /// it can neither confirm nor refute the goal.
  std::optional<Proof> askRangeAnalysis(Value v, Goal goal) const;

  /// Step 2: the decomposition rules. Returns std::nullopt when no sound rule
  /// applies to `v`'s defining operation, which is the signal to plant.
  std::optional<Proof> decompose(Value v, Goal goal, unsigned depth);

  /// Proves every sub-goal and conjoins the results.
  Proof proveAll(ArrayRef<RequiredCondition> subGoals, unsigned depth);

  const DataFlowSolver &solver;

  /// Memoizes `(value, goal)` so a value reached along two paths is proven
  /// once. The goal is stored as an unsigned because llvm::DenseMapInfo covers
  /// a pair of types that each have it, and an `enum class` does not.
  ///
  /// A cached entry can be pessimistic: a sub-proof abandoned at the depth cap
  /// memoizes the Refuted it propagated, and a later shallower path reuses it
  /// instead of retrying. That declines a candidate the recursion might have
  /// proven, which is the safe direction.
  DenseMap<std::pair<Value, unsigned>, Proof> memo;
};

std::optional<Proof> SignednessProver::askRangeAnalysis(Value v,
                                                        Goal goal) const {
  std::optional<ConstantIntRanges> range = tt::intel::collectRange(solver, v);
  if (!range)
    return std::nullopt;

  switch (goal) {
  case Goal::NonNegative:
    if (range->smin().isNonNegative())
      return Proof{Proof::Satisfied, {}};
    if (range->smax().isNegative())
      return Proof{Proof::Refuted, {}};
    break;
  case Goal::StrictlyPositive:
    if (range->smin().isStrictlyPositive())
      return Proof{Proof::Satisfied, {}};
    if (!range->smax().isStrictlyPositive())
      return Proof{Proof::Refuted, {}};
    break;
  }

  return std::nullopt;
}

Proof SignednessProver::proveAll(ArrayRef<RequiredCondition> subGoals,
                                 unsigned depth) {
  Proof result{Proof::Satisfied, {}};

  for (auto [value, goal] : subGoals) {
    Proof sub = prove(value, goal, depth + 1);

    // A sub-goal that cannot be met by any runtime check abandons the whole
    // proof: there is nothing left to ask of the kernel.
    if (sub.verdict == Proof::Refuted)
      return Proof{Proof::Refuted, {}};

    // Otherwise the conjunction is the union of what the parts still need.
    for (const RequiredCondition &condition : sub.requiredConditions)
      if (!llvm::is_contained(result.requiredConditions, condition))
        result.requiredConditions.push_back(condition);
  }

  if (!result.requiredConditions.empty())
    result.verdict = Proof::ConditionallySatisfied;

  return result;
}

std::optional<Proof> SignednessProver::decompose(Value v, Goal goal,
                                                 unsigned depth) {
  Operation *defOp = v.getDefiningOp();
  if (!defOp)
    return std::nullopt;

  return llvm::TypeSwitch<Operation *, std::optional<Proof>>(defOp)
      // Both operands carrying the goal implies the result does. The additive
      // and multiplicative cases assume no signed overflow, the same assumption
      // TritonIntelSimplifySignedArithmetic already makes.
      .Case<arith::AddIOp, arith::MulIOp>([&](auto op) {
        return proveAll({{op.getLhs(), goal}, {op.getRhs(), goal}}, depth);
      })
      // min(a, b) meets the goal exactly when both operands do - and fails it
      // as soon as either does, which is why a refuted operand refutes the min
      // rather than merely blocking the proof.
      .Case<arith::MinSIOp>([&](arith::MinSIOp op) {
        return proveAll({{op.getLhs(), goal}, {op.getRhs(), goal}}, depth);
      })
      // Either arm can be the one that runs, so both must meet the goal. The
      // condition is irrelevant.
      .Case<arith::SelectOp>([&](arith::SelectOp op) {
        return proveAll({{op.getTrueValue(), goal}, {op.getFalseValue(), goal}},
                        depth);
      })
      // max(a, b) meets the goal when *either* operand does, and a disjunction
      // of obligations is not a set of facts a runtime check can express. So
      // only an unconditionally satisfied operand discharges it; otherwise the
      // sub-proofs are discarded and the max itself is planted. Both operands
      // being refuted does refute the max.
      .Case<arith::MaxSIOp>([&](arith::MaxSIOp op) -> std::optional<Proof> {
        bool allRefuted = true;
        for (Value operand : {op.getLhs(), op.getRhs()}) {
          Proof sub = prove(operand, goal, depth + 1);
          if (sub.verdict == Proof::Satisfied)
            return sub;
          allRefuted &= (sub.verdict == Proof::Refuted);
        }
        if (allRefuted)
          return Proof{Proof::Refuted, {}};
        return std::nullopt;
      })
      // Floor, truncating and ceiling division all preserve a non-negative
      // dividend when the divisor is strictly positive. None of them preserves
      // strict positivity: 1 / 2 is 0.
      .Case<arith::DivSIOp, arith::FloorDivSIOp, arith::CeilDivSIOp>(
          [&](auto op) -> std::optional<Proof> {
            if (goal != Goal::NonNegative)
              return std::nullopt;
            return proveAll({{op.getLhs(), Goal::NonNegative},
                             {op.getRhs(), Goal::StrictlyPositive}},
                            depth);
          })
      // A signed remainder takes the sign of its dividend, so the divisor is
      // irrelevant to non-negativity. It says nothing about positivity, since
      // the remainder can be zero. This rule is sharper than MLIR's inferRemS,
      // which is why askRangeAnalysis() cannot replace it.
      .Case<arith::RemSIOp>([&](arith::RemSIOp op) -> std::optional<Proof> {
        if (goal != Goal::NonNegative)
          return std::nullopt;
        return prove(op.getLhs(), Goal::NonNegative, depth + 1);
      })
      // Sign extension preserves the value, so it preserves both goals.
      .Case<arith::ExtSIOp>(
          [&](arith::ExtSIOp op) { return prove(op.getIn(), goal, depth + 1); })
      // An arithmetic right shift preserves the sign bit, but shifts a positive
      // value down to zero.
      .Case<arith::ShRSIOp>([&](arith::ShRSIOp op) -> std::optional<Proof> {
        if (goal != Goal::NonNegative)
          return std::nullopt;
        return prove(op.getLhs(), Goal::NonNegative, depth + 1);
      })
      // Shape operations replicate their source element-wise, and a goal on a
      // tensor is a goal on every element.
      .Case<tt::SplatOp, tt::BroadcastOp, tt::ExpandDimsOp>(
          [&](auto op) { return prove(op.getSrc(), goal, depth + 1); })
      // Everything else is planted. Notably arith.subi, where the recursion has
      // to stop: `a >= 0` and `b >= 0` say nothing about `a - b`. Likewise
      // arith.trunci, which can turn a positive value negative by dropping the
      // high bits, and any load, call or opaque operation.
      .Default([](Operation *) { return std::nullopt; });
}

Proof SignednessProver::prove(Value v, Goal goal, unsigned depth) {
  // Abandoning has to mean Refuted rather than an obligation on `v`: planting
  // here would claim the goal holds iff `v` does, of a proof that stopped
  // halfway. Refuted propagates out through the conjunction, so no caller can
  // act on a partial set.
  //
  // Not memoized, since the outcome is an artifact of where `v` happened to sit
  // in this recursion rather than a fact about it.
  if (depth >= kMaxDepth) {
    LDBG("depth limit reached, abandoning proof at: " << v);
    return Proof{Proof::Refuted, {}};
  }

  auto key = std::make_pair(v, static_cast<unsigned>(goal));
  if (auto it = memo.find(key); it != memo.end())
    return it->second;

  Proof result = [&]() -> Proof {
    if (std::optional<Proof> known = askRangeAnalysis(v, goal))
      return *known;
    if (std::optional<Proof> decomposed = decompose(v, goal, depth))
      return *decomposed;
    return Proof{Proof::ConditionallySatisfied, {RequiredCondition{v, goal}}};
  }();

  memo.insert({key, result});
  return result;
}

} // namespace

namespace mlir::triton::intel {

Proof proveSign(Value v, Goal goal, const DataFlowSolver &solver) {
  SignednessProver prover(solver);
  return prover.prove(v, goal);
}

} // namespace mlir::triton::intel
