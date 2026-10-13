#include "LLVMPasses.h"
#include "llvm/Analysis/SimplifyQuery.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

// Triton evaluates arithmetic on masked-off lanes; a masked load's default
// value (usually 0) can reach the divisor of an integer div/rem. The language
// allows it because the result on those lanes is never used, but in LLVM IR
// (and SPIR-V) division by zero is immediate undefined behavior.
//
// LLVM >= 20 SimplifyCFG treats a path whose phi value feeds a div/rem as
// unreachable and replaces the branch with llvm.assume(mask); GVN then folds
// predicated-store masks to true, so masked-off lanes store out of bounds.
//
// The previous version only guarded a phi used directly as the divisor in the
// phi's own block. That missed vectorized masked loads (phi <N x iK> ->
// extractelement -> div in a later block), which IGC's scalarizer later turns
// into the exploitable shape. This version guards by value: every integer
// div/rem whose divisor is not provably non-zero gets select(freeze(d) == 0, 1,
// freeze(d)). freeze makes an undef/poison divisor a fixed value so the select
// really excludes 0.
//
// Side effect: a genuine x / 0 now yields x / 1 (accepted for the phi case in
// intel-xpu-backend-for-triton#6903).
//
// Signed INT_MIN / -1 is also UB but is not produced by masked-load defaults;
// not handled.

static bool guardDivisor(BinaryOperator &I, const SimplifyQuery &SQ) {
  if (!I.isIntDivRem())
    return false;

  Value *Divisor = I.getOperand(1);

  if (isKnownNonZero(Divisor, SQ.getWithInstruction(&I)))
    return false;

  IRBuilder<> Builder(&I);
  Type *Ty = Divisor->getType();
  Value *Frozen = Builder.CreateFreeze(Divisor, Divisor->getName() + ".fr");
  Value *IsZero = Builder.CreateICmpEQ(Frozen, Constant::getNullValue(Ty),
                                       Divisor->getName() + ".is_zero");
  Value *Safe = Builder.CreateSelect(IsZero, ConstantInt::get(Ty, 1), Frozen,
                                     Divisor->getName() + ".safe");
  I.setOperand(1, Safe);

  return true;
}

static bool runOnFunction(Function &F) {
  SimplifyQuery SQ(F.getDataLayout());

  SmallVector<BinaryOperator *, 16> DivRems;
  for (Instruction &I : instructions(F)) {
    if (auto *BO = dyn_cast<BinaryOperator>(&I)) {
      if (BO->isIntDivRem()) {
        DivRems.push_back(BO);
      }
    }
  }

  bool Changed = false;
  for (BinaryOperator *BO : DivRems) {
    Changed |= guardDivisor(*BO, SQ);
  }

  return Changed;
}

PreservedAnalyses GuardMaskedDivRemPass::run(Function &F,
                                             FunctionAnalysisManager &FAM) {
  const auto b = runOnFunction(F);

  return b ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
