#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

struct GuardMaskedDivRemPass : OptionalPassInfoMixin<GuardMaskedDivRemPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  static StringRef name() { return "GuardMaskedDivRemPass"; }
};

struct ScalarizePtrVectorsPass
    : OptionalPassInfoMixin<ScalarizePtrVectorsPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  static StringRef name() { return "ScalarizePtrVectorsPass"; }
};

struct ExpandSaddWithOverflowPass
    : OptionalPassInfoMixin<ExpandSaddWithOverflowPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static StringRef name() { return "ExpandSaddWithOverflowPass"; }
};

struct ExpandSubByteBitReversePass
    : OptionalPassInfoMixin<ExpandSubByteBitReversePass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static StringRef name() { return "ExpandSubByteBitReversePass"; }
};

struct ExpandSubByteBitwiseAndPass
    : OptionalPassInfoMixin<ExpandSubByteBitwiseAndPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static StringRef name() { return "ExpandSubByteBitwiseAndPass"; }
};

struct ExpandSubByteVectorBitcastPass
    : OptionalPassInfoMixin<ExpandSubByteVectorBitcastPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static StringRef name() { return "ExpandSubByteVectorBitcastPass"; }
};

struct LegalizeI1VectorMemoryPass
    : OptionalPassInfoMixin<LegalizeI1VectorMemoryPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static StringRef name() { return "LegalizeI1VectorMemoryPass"; }
};

} // namespace llvm
