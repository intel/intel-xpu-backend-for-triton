#include "third_party/intel/include/Target/LLVMIR/CodeSink.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/MemorySSAUpdater.h"
#include "llvm/Analysis/ScopedNoAliasAA.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/TypeBasedAliasAnalysis.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Value.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/AssumeBundleBuilder.h"
#include "llvm/Transforms/Utils/Local.h"
#include <map>
#include <optional>

using namespace llvm;

namespace {

class CodeSinkPass : public PassInfoMixin<CodeSinkPass> {
public:
  CodeSinkPass(bool trace) : trace(trace) {}

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {

    if (!rescheduleInstruction(F))
      return PreservedAnalyses::all();

    return PreservedAnalyses::none();
  }

private:
  bool trace;

  bool rescheduleInstruction(Function &F) const {
    bool Changed = false;

    // Only reschedule instructions in the basic block.
    for (BasicBlock *BB : post_order(&F)) {
      SmallVector<Instruction *> Candidates;
      for (Instruction &I : *BB) {
        if (I.hasOneUse()) {
          User *user = *(I.user_begin());
          if (auto CI = llvm::dyn_cast<llvm::CallInst>(user)) {
            auto Callee = CI->getCalledFunction()->getName();

            if ((Callee.contains("intel_sub_group_") &&
                 Callee.contains("_matrix_mad"))) {
              Value *C = CI->getOperand(2);
              if (&I == dyn_cast<Instruction>(C)) {
                if (auto *BinOp = dyn_cast<BinaryOperator>(&I)) {
                  if (BinOp->getOpcode() == Instruction::FMul) {
                    Candidates.push_back(&I);
                  }
                }
              }
            }

            if (Callee.contains("llvm.genx.GenISA.sub.group.dpas")) {
              Value *C = CI->getOperand(0);
              if (&I == dyn_cast<Instruction>(C)) {
                if (auto *BinOp = dyn_cast<BinaryOperator>(&I)) {
                  if (BinOp->getOpcode() == Instruction::FMul) {
                    Candidates.push_back(&I);
                  }
                }
              }
            }
          }
        }
      }

      for (Instruction *I : Candidates) {
        Changed = true;
        if (Instruction *User = dyn_cast<Instruction>(*I->user_begin()))
          I->moveBefore(User);
      }
    }

    return Changed;
  }
};

} // namespace

/// FIXME: This is a temporary workaround (should be done by IGC). We should
/// remove it once the IGC instruction reschedule works.
void mlir::triton::intel::CodeSink(llvm::Module &mod, bool trace) {
  FunctionAnalysisManager FAM;
  FAM.registerPass([&] { return AssumptionAnalysis(); });
  FAM.registerPass([&] { return DominatorTreeAnalysis(); });
  FAM.registerPass([&] { return LoopAnalysis(); });
  FAM.registerPass([&] { return PassInstrumentationAnalysis(); });
  FAM.registerPass([&] { return TargetLibraryAnalysis(); });
  FAM.registerPass([&] { return TargetIRAnalysis(); });
  FAM.registerPass([&] { return BasicAA(); });
  FAM.registerPass([&] { return ScopedNoAliasAA(); });
  FAM.registerPass([&] { return TypeBasedAA(); });
  FAM.registerPass([&] { return MemorySSAAnalysis(); });
  FAM.registerPass([&] {
    AAManager AA;
    AA.registerFunctionAnalysis<BasicAA>();
    AA.registerFunctionAnalysis<ScopedNoAliasAA>();
    AA.registerFunctionAnalysis<TypeBasedAA>();
    return AA;
  });

  FunctionPassManager FPM;
  FPM.addPass(CodeSinkPass(trace));

  for (llvm::Function &function : mod.functions()) {
    if (function.getCallingConv() == CallingConv::SPIR_KERNEL)
      FPM.run(function, FAM);
  }
}
