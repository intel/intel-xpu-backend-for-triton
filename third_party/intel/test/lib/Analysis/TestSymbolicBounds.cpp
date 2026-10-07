//===- TestSymbolicBounds.cpp ---------------------------------------------===//
//
// Emits the symbolic bounds prover's verdict for every `arith.cmpi` in the
// module as a remark, so a lit test can pin both the verdict and the runtime
// conditions a conditional proof depends on.
//
// The prover's individual rules are unit-tested directly in
// `unittest/Analysis/SymbolicBoundsTest.cpp`; this pass covers what a gtest
// string cannot carry conveniently - a real TTGIR module with layout
// attributes, above all.
//
// There is no `scf.for` remark: the prover has no trip-count API.
//
//===----------------------------------------------------------------------===//

#include "intel/include/Analysis/Range.h"
#include "intel/include/Analysis/SymbolicBounds.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Utility.h"

using namespace mlir;
using namespace mlir::triton::intel;

namespace {

struct TestSymbolicBoundsPass
    : public PassWrapper<TestSymbolicBoundsPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestSymbolicBoundsPass)

  StringRef getArgument() const final { return "test-intel-symbolic-bounds"; }
  StringRef getDescription() const final {
    return "print the symbolic bounds prover's verdict for every arith.cmpi";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    // The prover reads leaf ranges from the range analysis, so the solver is
    // built and run exactly as a consumer pass does.
    std::unique_ptr<DataFlowSolver> solver = createDataFlowSolver();
    solver->load<IntegerRangeAnalysis>(mod, getAnalysis<DominanceInfo>());

    if (failed(solver->initializeAndRun(mod)))
      return signalPassFailure();

    SymbolicBoundsProver prover(*solver, getAnalysis<DominanceInfo>(), mod);

    // Pre-order, so the remarks appear in program order: the prover's
    // condition insertion order - and so the rendered condition order - is
    // deterministic only for a deterministic visit order.
    mod.walk<WalkOrder::PreOrder>([&](arith::CmpIOp cmpOp) {
      QueryContext ctx{cmpOp.getOperation(),
                       cmpOp->getParentOfType<scf::ForOp>()};
      BoundProof proof = prover.prove(cmpOp.getPredicate(), cmpOp.getLhs(),
                                      cmpOp.getRhs(), ctx);
      emitRemark(cmpOp.getLoc(), "verdict: " + toString(proof));
    });
  }
};

} // namespace

namespace mlir::test::intel {
void registerTestSymbolicBoundsPass() {
  PassRegistration<TestSymbolicBoundsPass>();
}
} // namespace mlir::test::intel
