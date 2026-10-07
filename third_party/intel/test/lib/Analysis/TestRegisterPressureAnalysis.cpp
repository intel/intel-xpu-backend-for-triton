#include "intel/include/Analysis/RegisterPressure.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;

namespace {

struct TestRegisterPressurePass
    : public PassWrapper<TestRegisterPressurePass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestRegisterPressurePass)

  TestRegisterPressurePass() = default;
  // A pass carrying an Option member is not implicitly copyable, but
  // PassWrapper::clonePass needs a copy constructor.
  TestRegisterPressurePass(const TestRegisterPressurePass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const final { return "test-register-pressure"; }

  StringRef getDescription() const final {
    return "print the result of the register pressure analysis pass";
  }

  /// `print()` reports one figure per block, so it cannot exercise
  /// `pressureBefore` at all: that primitive takes an operation, and a lit file
  /// has no way to name one. This mode walks the function instead and reports
  /// it per operation. It is a separate mode rather than extra lines in the
  /// default output so that the existing per-block CHECK groups are left alone.
  Option<bool> perOp{
      *this, "per-op",
      llvm::cl::desc("report the pressure immediately above each operation "
                     "instead of the per-block peaks"),
      llvm::cl::init(false)};

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
      if (!perOp) {
        analysis.print(os);
        return;
      }

      // Pre-order so the lines come out in program order, which is what the
      // CHECK-NEXT sequences in the lit file rely on to tell two operations of
      // the same name apart.
      func->walk<WalkOrder::PreOrder>([&](Operation *nested) {
        if (nested == func.getOperation())
          return;
        os << "  Before " << nested->getName() << ": "
           << analysis.pressureBefore(nested) << " bytes\n";
      });
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
