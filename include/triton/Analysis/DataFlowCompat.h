#ifndef TRITON_ANALYSIS_DATAFLOWCOMPAT_H
#define TRITON_ANALYSIS_DATAFLOWCOMPAT_H

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Support/LLVM.h"

#include <type_traits>

// Compatibility layer for the SparseForwardDataFlowAnalysis API change in MLIR
// that added a GetLatticeFn parameter to visitNonControlFlowArguments.
namespace mlir::triton {

template <typename StateT, typename = void>
struct HasGetLatticeFn : std::false_type {};
template <typename StateT>
struct HasGetLatticeFn<
    StateT, std::void_t<typename dataflow::SparseForwardDataFlowAnalysis<
                StateT>::GetLatticeFn>> : std::true_type {};

template <typename StateT, typename = void>
class SparseForwardDataFlowAnalysisCompat
    : public dataflow::SparseForwardDataFlowAnalysis<StateT> {
  using Base = dataflow::SparseForwardDataFlowAnalysis<StateT>;

public:
  using Base::Base;
  using GetLatticeFn = function_ref<const StateT *(Value)>;

  virtual void visitNonSuccessorInputs(Operation *op,
                                       const RegionSuccessor &successor,
                                       ValueRange nonSuccessorInputs,
                                       ArrayRef<StateT *> lattices,
                                       GetLatticeFn getLattice) {
    this->setAllToEntryStates(lattices);
  }

  void visitNonControlFlowArguments(Operation *op,
                                    const RegionSuccessor &successor,
                                    ValueRange nonSuccessorInputs,
                                    ArrayRef<StateT *> lattices) final {
    ProgramPoint *point = this->getProgramPointAfter(op);
    visitNonSuccessorInputs(
        op, successor, nonSuccessorInputs, lattices,
        [&](Value value) { return this->getLatticeElementFor(point, value); });
  }
};

template <typename StateT>
class SparseForwardDataFlowAnalysisCompat<
    StateT, std::enable_if_t<HasGetLatticeFn<StateT>::value>>
    : public dataflow::SparseForwardDataFlowAnalysis<StateT> {
  using Base = dataflow::SparseForwardDataFlowAnalysis<StateT>;

public:
  using Base::Base;
  using typename Base::GetLatticeFn;

  virtual void visitNonSuccessorInputs(Operation *op,
                                       const RegionSuccessor &successor,
                                       ValueRange nonSuccessorInputs,
                                       ArrayRef<StateT *> lattices,
                                       GetLatticeFn getLattice) {
    this->setAllToEntryStates(lattices);
  }

  void visitNonControlFlowArguments(Operation *op,
                                    const RegionSuccessor &successor,
                                    ValueRange nonSuccessorInputs,
                                    ArrayRef<StateT *> lattices,
                                    GetLatticeFn getLattice) final {
    visitNonSuccessorInputs(op, successor, nonSuccessorInputs, lattices,
                            getLattice);
  }
};

template <typename StateT, typename AnalysisT, typename GetLatticeT>
void callVisitNonControlFlowArguments(AnalysisT &analysis, Operation *op,
                                      const RegionSuccessor &successor,
                                      ValueRange nonSuccessorInputs,
                                      ArrayRef<StateT *> lattices,
                                      GetLatticeT &&getLattice) {
  if constexpr (HasGetLatticeFn<StateT>::value)
    analysis.visitNonControlFlowArguments(op, successor, nonSuccessorInputs,
                                          lattices, getLattice);
  else
    analysis.visitNonControlFlowArguments(op, successor, nonSuccessorInputs,
                                          lattices);
}

} // namespace mlir::triton

#endif // TRITON_ANALYSIS_DATAFLOWCOMPAT_H
