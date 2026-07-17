#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

struct GuardMaskedDivRemPass : PassInfoMixin<GuardMaskedDivRemPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  static StringRef name() { return "GuardMaskedDivRemPass"; }
};

struct ScalarizePtrVectorsPass : PassInfoMixin<ScalarizePtrVectorsPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  static StringRef name() { return "ScalarizePtrVectorsPass"; }
};

struct ExpandSaddWithOverflowPass : PassInfoMixin<ExpandSaddWithOverflowPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static StringRef name() { return "ExpandSaddWithOverflowPass"; }
};

struct ExpandSubByteBitReversePass
    : PassInfoMixin<ExpandSubByteBitReversePass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static StringRef name() { return "ExpandSubByteBitReversePass"; }
};

struct ExpandSubByteBitwiseAndPass
    : PassInfoMixin<ExpandSubByteBitwiseAndPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static StringRef name() { return "ExpandSubByteBitwiseAndPass"; }
};

struct ExpandSubByteVectorBitcastPass
    : PassInfoMixin<ExpandSubByteVectorBitcastPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static StringRef name() { return "ExpandSubByteVectorBitcastPass"; }
};

} // namespace llvm
