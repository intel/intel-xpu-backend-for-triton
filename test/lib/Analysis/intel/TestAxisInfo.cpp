#include "intel/include/Analysis/AxisInfo.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::triton::intel;

namespace {

struct TestAxisInfoPass
    : public PassWrapper<TestAxisInfoPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestAxisInfoPass);

  StringRef getArgument() const final { return "test-print-axis-info"; }
  StringRef getDescription() const final {
    return "print the result of the axis analysis pass";
  }

  void runOnOperation() override {
    Operation *operation = getOperation();
    ModuleOp moduleOp = cast<ModuleOp>(operation);
    ModuleAxisInfoAnalysis moduleAxisInfoAnalysis(moduleOp);
    moduleOp.walk([&](triton::FuncOp funcOp) {
      auto &os = llvm::errs();
      auto opName = SymbolTable::getSymbolName(funcOp).getValue().str();
      os << "@" << opName << "\n";
      funcOp.walk([&](Operation *op) {
        if (op->getNumResults() < 1)
          return;
        for (Value result : op->getResults()) {
          result.print(os);
          os << " => ";
          auto *axisInfo = moduleAxisInfoAnalysis.getAxisInfo(result);
          if (axisInfo)
            axisInfo->print(os);
          os << "\n";
        }
      });
    });
  }
};

} // namespace

namespace mlir::test::intel {
void registerTestAxisInfoPass() { PassRegistration<TestAxisInfoPass>(); }
} // namespace mlir::test::intel
