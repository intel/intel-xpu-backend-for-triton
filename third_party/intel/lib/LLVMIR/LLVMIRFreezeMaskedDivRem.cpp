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

  bool Changed = false;
  if (phiHasNullValue) {
    for (Instruction &I : BB) {
      if (I.getOpcode() == Instruction::SDiv || I.getOpcode() == Instruction::SRem) {
        const size_t OpIdx = 1; // I.getOpcode() == Instruction::SRem ? 0 : 1;
        if (I.getOperand(OpIdx) == phiNode) {
          auto *freezePhi = new FreezeInst(phiNode, phiNode->getName() + ".frozen", I.getIterator());
          I.setOperand(OpIdx, freezePhi);
          Changed = true;
        }
      }
    }
#if 0
        auto FindUse = llvm::find_if(phiNode->users(), [](auto *U) {
          auto *Use = cast<Instruction>(U);
          llvm::errs() << "User: " << *Use << "\n";
          return (Use->getOpcode() == Instruction::SDiv || Use->getOpcode() == Instruction::SRem);
        });
        if (FindUse == phiNode->user_end()) {
          llvm::errs() << "no div :(\n";
          return false; 
        }
        auto *Use = cast<Instruction>(*FindUse);
        assert(Use->isIntDivRem());
        const size_t OpIdx = Use->getOpcode() == Instruction::SRem ? 0 : 1;
        if (Use->getOperand(OpIdx) == phiNode) {
          llvm::errs() << "Got our user! " << *Use << "\n";
          llvm::errs() << "Operand 1: " << *Use->getOperand(1) << "\n";
          auto *freezePhi = new FreezeInst(phiNode, phiNode->getName() + ".frozen", Use->getIterator());
          Use->setOperand(OpIdx, freezePhi);
          Changed = true;
        }
#endif
  }
  return Changed; 
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