#ifndef TRITON_INTEL_ANALYSIS_RANGE_H
#define TRITON_INTEL_ANALYSIS_RANGE_H

#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

namespace mlir::triton {
class FuncOp;
}

namespace mlir::triton::intel {

/// Determines the range of integer variables.
/// This pass is based on MLIR's dataflow framework and extends upstream's
/// IntegerRangeAnalysis to better support Triton-specific constructs.
class IntegerRangeAnalysis : public dataflow::IntegerRangeAnalysis {
public:
  using Base = dataflow::IntegerRangeAnalysis;

  IntegerRangeAnalysis(Operation *top, DataFlowSolver &solver,
                       DominanceInfo &domInfo);
  virtual ~IntegerRangeAnalysis() = default;

  virtual LogicalResult initialize(Operation *top) final {
    integerValues.clear();
    return Base::initialize(top);
  }

  void setToEntryState(dataflow::IntegerValueRangeLattice *lattice) final;

  LogicalResult visitOperation(
      Operation *op,
      ArrayRef<const dataflow::IntegerValueRangeLattice *> operands,
      ArrayRef<dataflow::IntegerValueRangeLattice *> resultsLattices) final;

  std::optional<int64_t> getTripCount(LoopLikeOpInterface loop);

  /// This method implements "abstract interpretation" of loops with statically
  /// known bounds in order to infer tight ranges for loop carried values.
  /// By "abstract interpretation" we mean lattice states are propagated to all
  /// region successors N times, where N is the total trip count of the loop.
  /// After propagation both loop body values and users of loop results will
  /// have accurate ranges. Major enhancements in this pass are:
  ///
  /// 1. it attempts to compute loop's total trip count (nested loop trip counts
  /// multiply) and initialize a visit count to 0. Note, due to how Dataflow
  /// analysis works we have to actually visit the loop N times for each
  /// iter_arg (each argument lattice) so we actually track visit count for
  /// (loop, arg) not just (loop).
  ///
  /// 2. Before propagating, we check if we have propagated for (loop, arg)
  /// >= N times. If so, we do not propagate (and thus the traversal
  /// converges/ends).
  ///
  /// Note, for loops where the trip count cannot be inferred *and* loops
  /// with a total trip count larger than `kDefaultMaxTripCount`, fallback
  /// to upstream's conservative inference (i.e., we infer [min_int,
  /// max_int]) for the loop operands and all users and all users of the
  /// results of the loop.
  void visitRegionSuccessors(
      ProgramPoint *point, RegionBranchOpInterface branch,
      RegionSuccessor successor,
      ArrayRef<dataflow::AbstractSparseLattice *> abstractLattices) final;

  // TODO: Can we place this into constructor and make it private ?
  void initializeModule(ModuleOp &mod);

private:
  //  void visitYieldHelper(Operation *yieldOp, Value value);
  LogicalResult visitOperationHelper(
      Operation *op,
      ArrayRef<const dataflow::IntegerValueRangeLattice *> operands,
      ArrayRef<dataflow::IntegerValueRangeLattice *> resultsLattices);

  DenseSet<Value> integerValues;
  DenseMap<Value, SetVector<Operation *>> assumptions;
  llvm::SmallMapVector<Value, ConstantIntRanges, 2> opResultAssumption;
  DominanceInfo &domInfo;
};

#if 0
std::optional<SmallVector<std::optional<ConstantIntRanges>>>
collectRanges(const DataFlowSolver &solver, ValueRange values);

#endif

} // namespace mlir::triton::intel

#endif // TRITON_INTEL_ANALYSIS_RANGE_H
