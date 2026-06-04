#include "LLVMPasses.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "scalarize-ptr-vectors"

// Rewrite `<N x ptr>` insert/extractelement chains into scalar pointers,
// replacing dynamic vector indexing with a select cascade.
//
// FIXME: Remove this pass once the SPIR-V translator handles dynamic indexing
// of pointer vectors.
static bool scalarizePtrVectorRoot(InsertElementInst *root) {
  auto *vecTy = cast<FixedVectorType>(root->getType());
  unsigned numElems = vecTy->getNumElements();

  SmallVector<Value *> elements(numElems);
  for (unsigned i = 0; i < numElems; ++i) {
    Value *elem = findScalarElement(root, i);
    if (!elem || isa<UndefValue>(elem) || isa<PoisonValue>(elem))
      return false;
    elements[i] = elem;
  }

  bool changed = false;
  bool hasUnhandledUser = false;
  SmallVector<ExtractElementInst *> toErase;
  for (User *user : root->users()) {
    auto *ee = dyn_cast<ExtractElementInst>(user);
    if (!ee) {
      hasUnhandledUser = true;
      continue;
    }

    Value *idx = ee->getIndexOperand();
    if (auto *cst = dyn_cast<ConstantInt>(idx)) {
      // Out-of-bounds index is poison; replace it either way so the
      // extractelement still gets erased below.
      unsigned i = cst->getZExtValue();
      ee->replaceAllUsesWith(i < numElems ? elements[i]
                                          : PoisonValue::get(ee->getType()));
    } else {
      IRBuilder<> builder(ee);
      Value *result = elements[numElems - 1];
      for (int i = static_cast<int>(numElems) - 2; i >= 0; --i) {
        Value *cmp =
            builder.CreateICmpEQ(idx, ConstantInt::get(idx->getType(), i));
        result = builder.CreateSelect(cmp, elements[i], result);
      }
      ee->replaceAllUsesWith(result);
    }
    toErase.push_back(ee);
    changed = true;
  }

  if (hasUnhandledUser)
    errs() << "warning: " DEBUG_TYPE
              ": pointer vector has a non-extractelement user\n";

  for (auto *ee : toErase)
    ee->eraseFromParent();
  return changed;
}

static bool runOnFunction(Function &F) {
  SmallVector<InsertElementInst *> roots;
  for (Instruction &I : instructions(F)) {
    auto *ie = dyn_cast<InsertElementInst>(&I);
    if (!ie)
      continue;
    auto *vecTy = dyn_cast<FixedVectorType>(ie->getType());
    if (!vecTy || !vecTy->getElementType()->isPointerTy())
      continue;
    if (none_of(ie->users(), [](User *u) { return isa<InsertElementInst>(u); }))
      roots.push_back(ie);
  }

  bool changed = false;
  for (auto *root : roots)
    changed |= scalarizePtrVectorRoot(root);
  return changed;
}

PreservedAnalyses ScalarizePtrVectorsPass::run(Function &F,
                                               FunctionAnalysisManager &FAM) {
  return runOnFunction(F) ? PreservedAnalyses::none()
                          : PreservedAnalyses::all();
}
