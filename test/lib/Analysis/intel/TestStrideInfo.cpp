#include "intel/include/Analysis/AxisInfo.h"
#include "intel/include/Analysis/StrideInfo.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::triton::intel;

namespace {

struct TestStrideInfoPass
    : public PassWrapper<TestStrideInfoPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestStrideInfoPass);

  StringRef getArgument() const final { return "test-print-stride-info"; }
  StringRef getDescription() const final {
    return "print the result of the stride analysis pass";
  }

  void runOnOperation() override {
    Operation *operation = getOperation();
    ModuleOp moduleOp = cast<ModuleOp>(operation);
    ModuleAxisInfoAnalysis moduleAxisInfoAnalysis(moduleOp);
    ModuleStrideAnalysis moduleStrideAnalysis(moduleOp,
                                              &moduleAxisInfoAnalysis);
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
          auto *strideInfo = moduleStrideAnalysis.getStrideInfo(result);
          if (strideInfo)
            strideInfo->print(os);
          os << "\n";
        }
      });
    });
  }
};

} // namespace

namespace mlir::test::intel {
void registerTestStrideInfoPass() { PassRegistration<TestStrideInfoPass>(); }
} // namespace mlir::test::intel
