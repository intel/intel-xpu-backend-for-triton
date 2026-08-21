#ifndef TRITON_INTEL_ANALYSIS_SIGNEDNESSPROVER_H
#define TRITON_INTEL_ANALYSIS_SIGNEDNESSPROVER_H

#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
class DataFlowSolver;
} // namespace mlir

namespace mlir::triton::intel {

/// The sign facts proveSign() can establish about a value. A tensor satisfies a
/// goal only when every element does.
enum class Goal {
  NonNegative,      ///< `>= 0`
  StrictlyPositive, ///< `> 0`
};

/// A fact that a proof could not establish, and that a runtime check would have
/// to establish in its place.
using RequiredCondition = std::pair<Value, Goal>;

/// The outcome of proveSign().
struct Proof {
  enum Verdict {
    /// The goal holds on every launch. `requiredConditions` is empty.
    Satisfied,
    /// The prover could not establish the goal, and no runtime check would
    /// change that: either the value is provably outside the goal, or a value
    /// it depends on is. `requiredConditions` is empty.
    Refuted,
    /// The goal is guaranteed to hold once every member of `requiredConditions`
    /// is confirmed. An unconfirmed condition does not mean the goal is false,
    /// only that it remains unproven.
    ConditionallySatisfied,
  };

  Verdict verdict;
  SmallVector<RequiredCondition> requiredConditions;
};

/// Attempts to prove that `v` satisfies `goal`.
///
/// `solver` must be an initialized tt::intel::IntegerRangeAnalysis, and is the
/// source of truth: the goal is decided by asking it first, and the defining
/// operation is decomposed into sub-goals only where it has no answer. Where no
/// sound decomposition exists - `arith.subi`, a load, a block argument - the
/// recursion stops and the value becomes a required condition.
///
/// Required conditions are reported, not vetted. Whether a fact is acceptable
/// to demand of a kernel at runtime is the caller's policy, so the caller must
/// filter them (assertable type, provenance, count) before acting on a
/// ConditionallySatisfied verdict.
Proof proveSign(Value v, Goal goal, const DataFlowSolver &solver);

} // namespace mlir::triton::intel

#endif // TRITON_INTEL_ANALYSIS_SIGNEDNESSPROVER_H
