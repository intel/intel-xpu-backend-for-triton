#include "LLVMPasses.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

static bool processPhiNode(PHINode *PhiNode) {
  if (none_of(PhiNode->incoming_values(), [](Use &U) {
        Constant *C = dyn_cast<Constant>(&U);
        return isa<UndefValue>(U) || C && C->isNullValue();
      })) {
    return false;
  }

  bool Changed = false;
  BasicBlock *BB = const_cast<BasicBlock *>(PhiNode->getParent());
  for (Instruction &I : *BB) {
    if (I.getOpcode() == Instruction::SDiv ||
        I.getOpcode() == Instruction::SRem ||
        I.getOpcode() == Instruction::UDiv ||
        I.getOpcode() == Instruction::URem) {
      const size_t OpIdx = 1;
      if (I.getOperand(OpIdx) == PhiNode) {
        // Triton masked loads lower to conditional blocks that produce
        // phi nodes with a zero default on the false path:
        //
        //   br i1 %mask, label %load_bb, label %merge
        // load_bb:
        //   %val = load ...
        //   br label %merge
        // merge:
        //   %phi = phi [%val, %load_bb], [0, %entry]
        //   %res = sdiv %x, %phi          ; UB when %mask is false
        //
        // LLVM exploits the sdiv-by-zero UB to insert llvm.assume(%mask)
        // which propagates and corrupts unrelated operations (e.g. makes
        // predicated stores unconditional).
        //
        // Replace the divisor with select(divisor == 0, 1, divisor).
        // This is legal: the zero case only arises on the false path
        // where both dividend and divisor are zero, so 0/0 (UB) becomes
        // 0/1 = 0, a well-defined value whose result is never observed.
        IRBuilder<> Builder(&I);
        Type *Ty = PhiNode->getType();
        Value *Zero = ConstantInt::get(Ty, 0);
        Value *One = ConstantInt::get(Ty, 1);
        Value *IsZero = Builder.CreateICmpEQ(PhiNode, Zero,
                                             PhiNode->getName() + ".is_zero");
        Value *SafeDiv = Builder.CreateSelect(IsZero, One, PhiNode,
                                              PhiNode->getName() + ".safe");
        I.setOperand(OpIdx, SafeDiv);
        Changed = true;
      }
    }
  }
  return Changed;
}

static bool runOnFunction(Function &F) {
  bool Changed = false;

  for (BasicBlock &BB : F) {
    for (PHINode &PhiNode : BB.phis()) {
      Changed |= processPhiNode(&PhiNode);
    }
  }

  return Changed;
}

PreservedAnalyses GuardMaskedDivRemPass::run(Function &F,
                                             FunctionAnalysisManager &FAM) {
  const auto b = runOnFunction(F);

  return b ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
