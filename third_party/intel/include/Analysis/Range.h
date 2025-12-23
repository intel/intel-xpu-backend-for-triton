#ifndef TRITON_INTEL_ANALYSIS_RANGE_H
#define TRITON_INTEL_ANALYSIS_RANGE_H

#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

namespace mlir::triton::intel {

/// Determines the range of integer variables.
/// This pass is based on MLIR's dataflow framework and extends upstream's
/// IntegerRangeAnalysis to better support Triton-specific constructs.
class IntegerRangeAnalysis : public dataflow::IntegerRangeAnalysis {
public:
  using Base = dataflow::IntegerRangeAnalysis;
  using AssumptionsOps = SetVector<Operation *>;

  IntegerRangeAnalysis(DataFlowSolver &solver, ModuleOp &mod,
                       DominanceInfo &domInfo);
  virtual ~IntegerRangeAnalysis() = default;

  virtual LogicalResult initialize(Operation *top) final {
    integerValues.clear();
    return Base::initialize(top);
  }

  virtual void
  setToEntryState(dataflow::IntegerValueRangeLattice *lattice) final;

  virtual LogicalResult visitOperation(
      Operation *op,
      ArrayRef<const dataflow::IntegerValueRangeLattice *> operands,
      ArrayRef<dataflow::IntegerValueRangeLattice *> resultsLattices) final;

  /// Implements "abstract interpretation" of loops with known bounds in order
  /// to infer the ranges for loop carried values. Lattice states are propagated
  /// to all region successors N times, where N is the total trip count of the
  /// loop. After propagation both loop body values and users of loop results
  /// will have accurate ranges.
  ///
  /// Notes:
  ///   - it attempts to compute loop's total trip count in the context of a
  ///     loop nest. Note, due to how Dataflow analysis works we have to
  ///     actually visit the loop N times for each iter_arg (each argument
  ///     lattice) so we actually track visit count for (loop, arg) not just
  ///     (loop).
  ///   - before propagating, we check if we have propagated for (loop, arg)
  ///     >= N times. If so, we do not propagate (and thus the traversal
  ///     converges/ends).
  ///
  /// Note, for loops where the trip count cannot be inferred *and* loops
  /// with a total trip count larger than `kDefaultMaxTripCount`, fallback
  /// to upstream's conservative inference (i.e., we infer [min_int,
  /// max_int]) for the loop operands and all users and all users of the
  /// results of the loop.
  virtual void visitRegionSuccessors(
      ProgramPoint *point, RegionBranchOpInterface branch,
      RegionSuccessor successor,
      ArrayRef<dataflow::AbstractSparseLattice *> abstractLattices) final;

  /// Collects assumptions in the given operation.
  static DenseMap<Value, AssumptionsOps>
  collectAssumptions(Operation *top, bool filterConstants = true);

  /// Returns the trip count of the given loop if it can be inferred.
  std::optional<uint64_t> getTripCount(LoopLikeOpInterface loop) const;

private:
  void initializeModule(ModuleOp &mod);

  LogicalResult visitOperationHelper(
      Operation *op,
      ArrayRef<const dataflow::IntegerValueRangeLattice *> operands,
      ArrayRef<dataflow::IntegerValueRangeLattice *> resultsLattices);

  DenseSet<Value> integerValues;
  DenseMap<Value, AssumptionsOps> assumptions;
  llvm::SmallMapVector<Value, ConstantIntRanges, 2> opResultAssumption;
  DominanceInfo &domInfo;

  /// Trip counts of all loops with static loop bounds contained under the root
  /// operation being analyzed. Note, nested loops have trip counts computed as
  /// a product of enclosing loops; i.e. for
  ///   scf.for i = 1 to 10
  ///     scf.for j = 1 to 10
  /// the trip count of the outer loop (on i) is 10 but the trip count of the
  /// inner loop (on j) is 100.
  llvm::SmallDenseMap<LoopLikeOpInterface, int64_t> loopTripCounts;

  /// Visit counts tabulating how many times each lattice has been propagated
  /// through each loop. This is used in visitRegionSuccessors to end
  /// propagation when loopVisits[loop, lattice] reaches loopTripCounts[loop].
  llvm::SmallDenseMap<
      std::pair<LoopLikeOpInterface, dataflow::IntegerValueRangeLattice *>,
      int64_t>
      loopVisits;
};

/// Collects the inferred integer ranges for the given value. If the value's
/// range cannot be inferred, std::nullopt is returned for that value.
std::optional<ConstantIntRanges> collectRange(const DataFlowSolver &solver,
                                              Value value);

/// Collects the inferred integer ranges for the given values. If a value's
/// range cannot be inferred, std::nullopt is returned for that value.
std::optional<SmallVector<std::optional<ConstantIntRanges>>>
collectRanges(const DataFlowSolver &solver, ValueRange values);

/// Returns the total trip count of the given loop by multiplying the trip
/// counts of all enclosing loops.
uint64_t getTotalTripCount(LoopLikeOpInterface loop,
                           IntegerRangeAnalysis &analysis);

/// Determine whether the given comparison operation always evaluates to true.
bool evaluatesToTrue(arith::CmpIOp cmpOp, const DataFlowSolver &solver);

} // namespace mlir::triton::intel

#endif // TRITON_INTEL_ANALYSIS_RANGE_H
