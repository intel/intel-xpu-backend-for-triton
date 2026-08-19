#ifndef TRITON_INTEL_ANALYSIS_NONNEGATIVEPROVER_H
#define TRITON_INTEL_ANALYSIS_NONNEGATIVEPROVER_H

#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
class DataFlowSolver;
} // namespace mlir

namespace mlir::triton::intel {

/// The sign facts proveSign() can establish about a value. A tensor satisfies a
/// goal only when every element does.
enum class Goal {
  NonNegative, ///< `v >= 0`
  Positive,    ///< `v > 0`
};

/// A fact that a proof could not establish, and that a runtime check would have
/// to establish in its place.
using Obligation = std::pair<Value, Goal>;

/// The outcome of proveSign().
struct Proof {
  enum Verdict {
    /// The goal holds on every launch. `obligations` is empty.
    Satisfied,
    /// The prover will not establish the goal, and no runtime check would
    /// change that: either the value is provably outside the goal, or a value
    /// it depends on is. `obligations` is empty.
    Refuted,
    /// The goal holds if and only if every member of `obligations` holds.
    Obligated,
  };

  Verdict verdict;
  SmallVector<Obligation> obligations;
};

/// Attempts to prove that `v` satisfies `goal`.
///
/// `solver` must be an initialized tt::intel::IntegerRangeAnalysis, and is the
/// source of truth: the goal is decided by asking it first, and the defining
/// operation is decomposed into sub-goals only where it has no answer. Where no
/// sound decomposition exists - `arith.subi`, a load, a block argument - the
/// recursion stops and the value becomes an obligation.
///
/// Obligations are reported, not vetted. Whether a fact is acceptable to demand
/// of a kernel at runtime is the caller's policy, so the caller must filter
/// them (assertable type, provenance, count) before acting on an Obligated
/// verdict.
Proof proveSign(Value v, Goal goal, const DataFlowSolver &solver);

} // namespace mlir::triton::intel

#endif // TRITON_INTEL_ANALYSIS_NONNEGATIVEPROVER_H
