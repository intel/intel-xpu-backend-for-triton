#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

struct FreezeMaskedDivRemPass : PassInfoMixin<FreezeMaskedDivRemPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  static StringRef name() { return "FreezeMaskedDivRemPass"; }
};

}