#include "LLVMPasses.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Dominators.h"

using namespace llvm;

static bool processPhiNode(PHINode *phiNode, BasicBlock& BB) {
      llvm::errs() << "YOLO: " << *phiNode << "\n";

  const auto phiHasNullValue = any_of(phiNode->incoming_values(), [](Use& U) {
    if (Constant *C = dyn_cast<Constant>(&U)) {
      return C->isNullValue();
    }
    return false; 
  });

  if (phiHasNullValue) {
        auto FindUse = llvm::find_if(phiNode->users(), [](auto *U) {
          auto *Use = cast<Instruction>(U);
          llvm::errs() << "User: " << *Use << "\n";
          return (Use->getOpcode() == Instruction::SDiv);
        });
        if (FindUse == phiNode->user_end()) {
          llvm::errs() << "no div :(\n";
          return false; 
        }
        auto *Use = cast<Instruction>(*FindUse);
        assert()
        llvm::errs() << "Got our user! " << *Use << "\n";

        // insert freeze between phi and sdiv 
        // 
  }
  return false; 
}

static bool runOnFunction(Function& F, const TargetTransformInfo &TTI,
                           const DominatorTree &DT) {
    bool Changed = false;

    SmallVector<PHINode *> PhiNodes;
    for (BasicBlock &BB : F) {
    for (Instruction &inst : BB) {
      if (PHINode *phiNode = dyn_cast<PHINode>(&inst)) {
        Changed |= processPhiNode(phiNode, BB);
        continue;
      }
      break;
    }
  }

    return Changed;
}

 PreservedAnalyses FreezeMaskedDivRemPass::run(Function &F, FunctionAnalysisManager &FAM) {
    TargetTransformInfo &TTI = FAM.getResult<TargetIRAnalysis>(F);
    DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(F);
    const auto b = runOnFunction(F, TTI, DT);

    return b ? PreservedAnalyses::none() : PreservedAnalyses::all();
}