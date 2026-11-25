#include "intel/include/Analysis/RegisterPressure.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;

namespace {

struct TestRegisterPressurePass
    : public PassWrapper<TestRegisterPressurePass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestRegisterPressurePass)

  StringRef getArgument() const final { return "test-register-pressure"; }

  StringRef getDescription() const final {
    return "print the result of the register pressure analysis pass";
  }

  void runOnOperation() override {
    Operation *op = getOperation();
    auto mod = cast<ModuleOp>(op);
    raw_ostream &os = llvm::outs();

    mod.walk<WalkOrder::PreOrder>([&](triton::FuncOp func) {
      auto opName = SymbolTable::getSymbolName(func).getValue().str();
      os << opName << "\n";

      // Build the analysis once per function; print() reports peak pressure
      // for every block, including those nested in loops.
      triton::gpu::intel::RegisterPressureAnalysis analysis(func);
      analysis.print(os);
    });
  }
};

} // end anonymous namespace

namespace mlir {
namespace test {
void registerTestRegisterPressurePass() {
  PassRegistration<TestRegisterPressurePass>();
}
} // end namespace test
} // end namespace mlir
