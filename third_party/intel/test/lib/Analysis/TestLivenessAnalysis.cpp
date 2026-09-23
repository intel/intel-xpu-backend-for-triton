#include "intel/include/Analysis/Liveness.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;

namespace {

struct TestLivenessPass
    : public PassWrapper<TestLivenessPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestLivenessPass)

  StringRef getArgument() const final { return "test-liveness"; }

  StringRef getDescription() const final {
    return "print the result of the liveness analysis pass";
  }

  void runOnOperation() override {
    Operation *op = getOperation();
    ModuleOp mod = cast<ModuleOp>(op);
    raw_ostream &os = llvm::errs();

    mod.walk<WalkOrder::PreOrder>([&](triton::FuncOp func) {
      auto opName = SymbolTable::getSymbolName(func).getValue().str();
      os << opName << "\n";

      func.walk<WalkOrder::PreOrder>([&](Operation *op) {
        if (op->getRegions().empty())
          return;

        os << op->getName() << "\n";
        auto liveness = triton::gpu::intel::LivenessAnalysis(op);
        liveness.printLiveIntervals(os);
      });
    });
  }
};

} // end anonymous namespace

namespace mlir {
namespace test {
void registerTestLivenessPass() { PassRegistration<TestLivenessPass>(); }
} // end namespace test
} // end namespace mlir
