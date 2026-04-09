#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

struct FreezeMaskedDivRemPass : PassInfoMixin<FreezeMaskedDivRemPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  static StringRef name() { return "FreezeMaskedDivRemPass"; }
};

struct ExpandSaddWithOverflowPass : PassInfoMixin<ExpandSaddWithOverflowPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static StringRef name() { return "ExpandSaddWithOverflowPass"; }
};

} // namespace llvm
