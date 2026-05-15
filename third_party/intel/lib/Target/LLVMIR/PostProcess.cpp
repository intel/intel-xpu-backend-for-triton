#include "third_party/intel/include/Target/LLVMIR/PostProcess.h"
#include "third_party/intel/lib/Target/LLVMIR/LLVMPasses.h"

#include "llvm/IR/DerivedTypes.h"
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
void expandSaddWithOverflow(Module &module) {
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
    // Use explicit temporaries to guarantee deterministic IR emission
    // order (nested builder calls have unspecified C++ evaluation order).
    Value *xorAB = builder.CreateXor(lhs, rhs);
    Value *notAB = builder.CreateNot(xorAB);
    Value *xorAS = builder.CreateXor(lhs, sum);
    Value *andVal = builder.CreateAnd(notAB, xorAS);
    Value *overflow =
        builder.CreateICmpSLT(andVal, ConstantInt::get(lhs->getType(), 0));

    Value *result = builder.CreateInsertValue(
        builder.CreateInsertValue(UndefValue::get(call->getType()), sum, {0}),
        overflow, {1});

    call->replaceAllUsesWith(result);
    call->eraseFromParent();
  }
}

// Returns the sub-byte (N<8) integer element type of `t` (scalar IntegerType
// or FixedVectorType of integers), or nullptr otherwise.
static IntegerType *getSubByteIntElement(Type *t) {
  if (auto *intTy = dyn_cast<IntegerType>(t))
    return intTy->getBitWidth() < 8 ? intTy : nullptr;
  if (auto *vecTy = dyn_cast<FixedVectorType>(t))
    if (auto *intTy = dyn_cast<IntegerType>(vecTy->getElementType()))
      return intTy->getBitWidth() < 8 ? intTy : nullptr;
  return nullptr;
}

// Returns an i32 type matching the shape of `narrowTy`: scalar i32 for a
// scalar narrow type, <K x i32> for <K x iN>.
static Type *getWideTypeForNarrow(Type *narrowTy, IRBuilder<> &builder) {
  Type *i32ElemTy = builder.getInt32Ty();
  if (auto *vecTy = dyn_cast<FixedVectorType>(narrowTy))
    return FixedVectorType::get(i32ElemTy, vecTy->getNumElements());
  return i32ElemTy;
}

// SPV_INTEL_int4 does not permit OpBitReverse on sub-byte OpTypeInt. Widen
// via i32 bitreverse (supported by SPV_KHR_bit_instructions):
//   zext iN -> i32; bitreverse.i32; lshr (32-N); trunc i32 -> iN
// Handles scalar iN and <K x iN> vectors uniformly.
void expandSubByteBitReverse(Module &module) {
  SmallVector<CallInst *> calls;

  for (auto &func : module)
    for (auto &block : func)
      for (auto &inst : block)
        if (auto *call = dyn_cast<CallInst>(&inst))
          if (auto *callee = call->getCalledFunction())
            if (callee->getIntrinsicID() == Intrinsic::bitreverse)
              if (getSubByteIntElement(call->getType()))
                calls.push_back(call);

  for (CallInst *call : calls) {
    IRBuilder<> builder(call);
    Type *narrowTy = call->getType();
    unsigned narrowBits = getSubByteIntElement(narrowTy)->getBitWidth();
    Type *wideTy = getWideTypeForNarrow(narrowTy, builder);

    Value *zext = builder.CreateZExt(call->getArgOperand(0), wideTy);
    Value *rev32 = builder.CreateIntrinsic(Intrinsic::bitreverse, {wideTy},
                                           {zext}, /*FMFSource=*/nullptr);
    Value *shr =
        builder.CreateLShr(rev32, ConstantInt::get(wideTy, 32 - narrowBits));
    Value *result = builder.CreateTrunc(shr, narrowTy);

    call->replaceAllUsesWith(result);
    call->eraseFromParent();
  }
}

// SPV_INTEL_int4 does not permit OpBitwiseAnd on sub-byte OpTypeInt. Widen
// through i32:
//   zext iN -> i32 (both operands); and i32; trunc i32 -> iN
// Handles scalar iN and <K x iN> vectors uniformly.
void expandSubByteBitwiseAnd(Module &module) {
  SmallVector<BinaryOperator *> ands;

  for (auto &func : module)
    for (auto &block : func)
      for (auto &inst : block)
        if (inst.getOpcode() == Instruction::And)
          if (getSubByteIntElement(inst.getType()))
            ands.push_back(cast<BinaryOperator>(&inst));

  for (BinaryOperator *op : ands) {
    IRBuilder<> builder(op);
    Type *narrowTy = op->getType();
    Type *wideTy = getWideTypeForNarrow(narrowTy, builder);

    Value *a32 = builder.CreateZExt(op->getOperand(0), wideTy);
    Value *b32 = builder.CreateZExt(op->getOperand(1), wideTy);
    Value *r32 = builder.CreateAnd(a32, b32);
    Value *result = builder.CreateTrunc(r32, narrowTy);

    op->replaceAllUsesWith(result);
    op->eraseFromParent();
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

  // Widen sub-byte bit ops forbidden by SPV_INTEL_int4.
  expandSubByteBitReverse(mod);
  expandSubByteBitwiseAnd(mod);
}

} // namespace mlir::triton::intel

PreservedAnalyses ExpandSaddWithOverflowPass::run(Module &M,
                                                  ModuleAnalysisManager &) {
  mlir::triton::intel::expandSaddWithOverflow(M);
  return PreservedAnalyses::none();
}

PreservedAnalyses ExpandSubByteBitReversePass::run(Module &M,
                                                   ModuleAnalysisManager &) {
  mlir::triton::intel::expandSubByteBitReverse(M);
  return PreservedAnalyses::none();
}

PreservedAnalyses ExpandSubByteBitwiseAndPass::run(Module &M,
                                                   ModuleAnalysisManager &) {
  mlir::triton::intel::expandSubByteBitwiseAnd(M);
  return PreservedAnalyses::none();
}
