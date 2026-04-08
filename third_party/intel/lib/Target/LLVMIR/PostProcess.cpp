#include "third_party/intel/include/Target/LLVMIR/PostProcess.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"

using namespace llvm;

namespace mlir::triton::intel {

// W/A for LTS driver - 1146
//
// Replace every llvm.sadd.with.overflow.iN call with equivalent plain
// arithmetic before SPIR-V translation.
//
// The SPIRV-LLVM-Translator is supposed to lower this intrinsic by linking in
// a software emulation body (LLVMSaddWithOverflow.h).  It does so via
// Linker::LinkOnlyNeeded, which assigns available_externally linkage to the
// body.  LLVMToSPIRVPass then emits it as an Import OpFunction (no body),
// which IGC cannot call and hangs on.
//
// Expanding here means the translator never sees the intrinsic and never
// attempts the broken linking path.
//
// Overflow condition (standard sign-bit trick):
//   overflow = (~(lhs ^ rhs) & (lhs ^ sum)) < 0
//   i.e. both operands shared a sign but the sum has a different sign.
static void expandSaddWithOverflow(Module &module) {
  SmallVector<CallInst *> calls;

  for (auto &func : module)
    for (auto &block : func)
      for (auto &inst : block)
        if (auto *call = dyn_cast<CallInst>(&inst))
          if (auto *callee = call->getCalledFunction())
            if (callee->getIntrinsicID() == Intrinsic::sadd_with_overflow)
              calls.push_back(call);

  for (CallInst *call : calls) {
    IRBuilder<> builder(call);
    Value *lhs = call->getArgOperand(0);
    Value *rhs = call->getArgOperand(1);

    Value *sum      = builder.CreateAdd(lhs, rhs);
    Value *overflow = builder.CreateICmpSLT(
        builder.CreateAnd(builder.CreateNot(builder.CreateXor(lhs, rhs)),
                          builder.CreateXor(lhs, sum)),
        ConstantInt::get(lhs->getType(), 0));

    Value *result = builder.CreateInsertValue(
        builder.CreateInsertValue(UndefValue::get(call->getType()), sum, {0}),
        overflow, {1});

    call->replaceAllUsesWith(result);
    call->eraseFromParent();
  }
}

void postProcessLLVMIR(llvm::Module &mod) {
  // __devicelib_assert_fail must be a declaration so that
  // IGC can replace it with a runtime assert function.
  // If a 'fallback' implementation is defined in SYCL libarary, the
  // assertion does not work correctly.
  for (auto &func : mod) {
    if (func.getName() == "__devicelib_assert_fail") {
      assert(func.isDeclaration() &&
             "__devicelib_assert_fail must be a declaration!");
    }
  }

  // Pre-expand llvm.sadd.with.overflow.* so the SPIR-V translator never sees
  // the intrinsic and never emits the broken emulation-function Import.
  expandSaddWithOverflow(mod);
}

} // namespace mlir::triton::intel
