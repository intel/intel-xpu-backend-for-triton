#include "third_party/intel/include/Target/LLVMIR/PostProcess.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"

using namespace llvm;

namespace mlir::triton::intel {

// W/A for IGC bug on LTS2 driver (IGC < 2.19.0, Agama <= 1197)
//
// Replace every llvm.sadd.with.overflow.iN call with equivalent plain
// arithmetic before SPIR-V translation.
//
// The SPIRV-LLVM-Translator emulates this intrinsic via a helper function
// returning {iN, i1}. IGC's PromoteBools pass promotes i1->i8 inside that
// struct, but on LTS2 it uses a whole-struct bitcast ({i32,i1} -> {i32,i8})
// instead of element-wise promotion. The resulting extractvalue returning i8
// hits PromoteInt8Type, whose switch has no ExtractValue case — newVal is
// never set, blocking all downstream i8 consumers in the readiness check and
// causing an infinite loop in the promotion worklist.
//
// Fixed in IGC commit c1d34755f (v2.19.0) which replaced the bitcast with
// castAggregate() for proper element-by-element promotion. Pre-expanding
// here avoids the {iN, i1} struct-returning call entirely.
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

    Value *sum = builder.CreateAdd(lhs, rhs);
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
    if (func.getName().str() == "__devicelib_assert_fail") {
      assert(func.isDeclaration() &&
             "__devicelib_assert_fail must be a declaration!");
    }
  }

  // Pre-expand llvm.sadd.with.overflow.* so the SPIR-V translator never
  // links in the {iN, i1} emulation function that triggers the IGC bug.
  expandSaddWithOverflow(mod);
}

} // namespace mlir::triton::intel
