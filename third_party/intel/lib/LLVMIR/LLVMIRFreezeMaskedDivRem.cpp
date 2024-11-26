#include "LLVMPasses.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

static bool processBasicBlock(BasicBlock &BB, PHINode *PhiNode) {
  if (!any_of(PhiNode->incoming_values(), [](Use &U) {
        Constant *C = dyn_cast<Constant>(&U);
        return C && C->isNullValue();
      })) {
    return false;
  }

  bool Changed = false;
  for (Instruction &I : BB) {
    if (I.getOpcode() == Instruction::SDiv ||
        I.getOpcode() == Instruction::SRem) {
      const size_t OpIdx = 1;
      if (I.getOperand(OpIdx) == PhiNode) {
        auto *freezePhi = new FreezeInst(
            PhiNode, PhiNode->getName() + ".frozen", I.getIterator());
        I.setOperand(OpIdx, freezePhi);
        Changed = true;
      }
    }
  }
  return Changed;
}

static bool runOnFunction(Function &F) {
  bool Changed = false;

  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (PHINode *PhiNode = dyn_cast<PHINode>(&I)) {
        Changed |= processBasicBlock(BB, PhiNode);
        continue;
      }
      break;
    }
  }

  return Changed;
}

PreservedAnalyses FreezeMaskedDivRemPass::run(Function &F,
                                              FunctionAnalysisManager &FAM) {
  const auto b = runOnFunction(F);

  return b ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
