#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

struct GuardMaskedDivRemPass : PassInfoMixin<GuardMaskedDivRemPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  static StringRef name() { return "GuardMaskedDivRemPass"; }
};

struct ExpandSaddWithOverflowPass : PassInfoMixin<ExpandSaddWithOverflowPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static StringRef name() { return "ExpandSaddWithOverflowPass"; }
};

} // namespace llvm
