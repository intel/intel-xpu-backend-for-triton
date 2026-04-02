#include "LLVMPasses.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Local.h"

#define DEBUG_TYPE "scalarize-ptr-vectors"

using namespace llvm;

static bool runOnFunction(Function &F) {
  SmallVector<InsertElementInst *> roots;

  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      auto *ie = dyn_cast<InsertElementInst>(&I);
      if (!ie)
        continue;
      auto *vecTy = dyn_cast<FixedVectorType>(ie->getType());
      if (!vecTy || !vecTy->getElementType()->isPointerTy())
        continue;
      if (none_of(ie->users(),
                  [](User *u) { return isa<InsertElementInst>(u); }))
        roots.push_back(ie);
    }
  }

  bool changed = false;
  for (auto *root : roots) {
    auto *vecTy = cast<FixedVectorType>(root->getType());
    unsigned numElems = vecTy->getNumElements();

    SmallVector<Value *> elements(numElems, nullptr);
    for (Value *cur = root; auto *ie = dyn_cast<InsertElementInst>(cur);
         cur = ie->getOperand(0)) {
      if (auto *idx = dyn_cast<ConstantInt>(ie->getOperand(2))) {
        unsigned i = idx->getZExtValue();
        if (i < numElems && !elements[i])
          elements[i] = ie->getOperand(1);
      }
    }

    if (!all_of(elements, [](Value *v) { return v != nullptr; }))
      continue;

    bool hasUnhandledUsers = false;
    SmallVector<ExtractElementInst *> toErase;
    for (auto *user : root->users()) {
      auto *ee = dyn_cast<ExtractElementInst>(user);
      if (!ee) {
        hasUnhandledUsers = true;
        continue;
      }

      if (auto *idx = dyn_cast<ConstantInt>(ee->getIndexOperand())) {
        unsigned i = idx->getZExtValue();
        if (i < numElems) {
          ee->replaceAllUsesWith(elements[i]);
          toErase.push_back(ee);
        }
      } else {
        IRBuilder<> builder(ee);
        Value *dynIdx = ee->getIndexOperand();
        Type *idxTy = dynIdx->getType();
        Value *result = elements[numElems - 1];
        for (int i = static_cast<int>(numElems) - 2; i >= 0; --i) {
          Value *cmp = builder.CreateICmpEQ(dynIdx, ConstantInt::get(idxTy, i));
          result = builder.CreateSelect(cmp, elements[i], result);
        }
        ee->replaceAllUsesWith(result);
        toErase.push_back(ee);
      }
    }

    if (hasUnhandledUsers)
      errs() << "warning: " DEBUG_TYPE ": pointer vector has non-extractelement"
                " users that may trigger SPV_INTEL_masked_gather_scatter\n";

    for (auto *ee : toErase)
      ee->eraseFromParent();
    if (root->use_empty())
      RecursivelyDeleteTriviallyDeadInstructions(root);

    changed |= !toErase.empty();
  }
  return changed;
}

PreservedAnalyses ScalarizePtrVectorsPass::run(Function &F,
                                               FunctionAnalysisManager &FAM) {
  return runOnFunction(F) ? PreservedAnalyses::none()
                          : PreservedAnalyses::all();
}
