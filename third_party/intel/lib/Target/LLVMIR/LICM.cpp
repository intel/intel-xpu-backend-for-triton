#include "third_party/intel/include/Target/LLVMIR/LICM.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/LoopIterator.h"
#include "llvm/Analysis/MemorySSAUpdater.h"
#include "llvm/Analysis/MustExecute.h"
#include "llvm/Analysis/ScopedNoAliasAA.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/TypeBasedAliasAnalysis.h"
#include "llvm/Passes/PassBuilder.h"

using namespace llvm;

namespace {

class LICMPass : public PassInfoMixin<LICMPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
    auto &LI = AM.getResult<LoopAnalysis>(F);
    auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
    auto &AA = AM.getResult<AAManager>(F);
    auto &SE = AM.getResult<ScalarEvolutionAnalysis>(F);
    auto &MSSA = AM.getResult<MemorySSAAnalysis>(F).getMSSA();

    auto inSubLoop = [](BasicBlock *BB, Loop *L, LoopInfo &LI) {
      return LI.getLoopFor(BB) != L;
    };

    for (auto *L : LI) {
      MemorySSAUpdater MSSAU(&MSSA);
      ICFLoopSafetyInfo SafetyInfo;
      SafetyInfo.computeLoopSafetyInfo(L);

      LoopBlocksRPO workList(L);
      workList.perform(&LI);

      BasicBlock *Preheader = L->getLoopPreheader();
      assert(Preheader && "Loop does not have a preheader");

      for (BasicBlock *BB : workList) {
        for (Instruction &I : llvm::make_early_inc_range(*BB)) {
          if (inSubLoop(BB, L, LI))
            continue;

          if (L->hasLoopInvariantOperands(&I) && canHoist(I, AA, DT, MSSA, L)) {
            hoist(I, DT, L, L->getLoopPreheader(), SafetyInfo, MSSAU, SE);
            continue;
          }
        }
      }
    }

    return PreservedAnalyses::all();
  }

private:
  bool isHoistable(Instruction &I) const {
    return (isa<CallInst>(I) || isa<CastInst>(I) || isa<UnaryOperator>(I) ||
            isa<BinaryOperator>(I) || isa<SelectInst>(I) ||
            isa<GetElementPtrInst>(I) || isa<CmpInst>(I) ||
            isa<InsertElementInst>(I) || isa<ExtractElementInst>(I) ||
            isa<ShuffleVectorInst>(I) || isa<ExtractValueInst>(I) ||
            isa<InsertValueInst>(I) || isa<FreezeInst>(I));
  }

  bool canHoist(Instruction &I, AAResults &AA, DominatorTree &DT,
                MemorySSA &MSSA, Loop *L) {
    if (!isHoistable(I))
      return false;

    // Return true if MSSA knows there are no MemoryDefs in the loop.
    auto isReadOnly = [](MemorySSA &MSSA, const Loop *L) {
      for (auto *BB : L->getBlocks())
        if (MSSA.getBlockDefs(BB))
          return false;
      return true;
    };

    auto pointerInvalidatedByLoop = [&](MemorySSA &MSSA, MemoryUse *MU, Loop *L,
                                        Instruction *I) {
      BatchAAResults BAA(MSSA.getAA());
      auto getClobberingMemoryAccess = [](MemorySSA &MSSA, BatchAAResults &BAA,
                                          MemoryUseOrDef *MA) {
        return MSSA.getSkipSelfWalker()->getClobberingMemoryAccess(MA, BAA);
      };
      MemoryAccess *Source = getClobberingMemoryAccess(MSSA, BAA, MU);
      return !MSSA.isLiveOnEntryDef(Source) && L->contains(Source->getBlock());
    };

    if (CallInst *CI = dyn_cast<CallInst>(&I)) {
      // Only allow hoisting builtin calls.
      if (!CI->getCalledFunction()->getName().starts_with(
              "__builtin_IB_subgroup"))
        return false;

      if (CI->mayThrow() || CI->isConvergent())
        return false;

      MemoryEffects Behavior = AA.getMemoryEffects(CI);
      if (Behavior.doesNotAccessMemory())
        return true;

      if (Behavior.onlyReadsMemory()) {
        // A readonly argmemonly function only reads from memory pointed to by
        // it's arguments with arbitrary offsets. If we can prove there are no
        // writes to this memory in the loop, we can hoist it.
        if (Behavior.onlyAccessesArgPointees()) {
          for (Value *Op : CI->args())
            if (Op->getType()->isPointerTy() &&
                pointerInvalidatedByLoop(
                    MSSA, cast<MemoryUse>(MSSA.getMemoryAccess(CI)), L, &I))
              return false;
          return true;
        }

        // If this call only reads from memory and there are no writes to
        // memory in the loop, we can hoist or sink the call as appropriate.
        if (isReadOnly(MSSA, L))
          return true;
      }
      return false;
    }

    assert(!I.mayReadOrWriteMemory() && "unhandled aliasing");

    return true;
  }

  static void hoist(Instruction &I, const DominatorTree &DT, const Loop *L,
                    BasicBlock *Dest, ICFLoopSafetyInfo &SafetyInfo,
                    MemorySSAUpdater &MSSAU, ScalarEvolution &SE) {
    auto moveInstructionBefore = [](Instruction &I, BasicBlock::iterator Dest,
                                    ICFLoopSafetyInfo &SafetyInfo,
                                    MemorySSAUpdater &MSSAU,
                                    ScalarEvolution &SE) {
      SafetyInfo.removeInstruction(&I);
      SafetyInfo.insertInstructionTo(&I, Dest->getParent());
      I.moveBefore(*Dest->getParent(), Dest);
      if (MemoryUseOrDef *OldMemAcc = cast_or_null<MemoryUseOrDef>(
              MSSAU.getMemorySSA()->getMemoryAccess(&I)))
        MSSAU.moveToPlace(OldMemAcc, Dest->getParent(),
                          MemorySSA::BeforeTerminator);
      SE.forgetBlockAndLoopDispositions(&I);
    };

    if (isa<CallInst>(I) && !SafetyInfo.isGuaranteedToExecute(I, &DT, L))
      I.dropUBImplyingAttrsAndMetadata();

    if (isa<PHINode>(I))
      // Move the new node to the end of the phi list in the destination block.
      moveInstructionBefore(I, Dest->getFirstNonPHIIt(), SafetyInfo, MSSAU, SE);
    else
      // Move the new node to the destination block, before its terminator.
      moveInstructionBefore(I, Dest->getTerminator()->getIterator(), SafetyInfo,
                            MSSAU, SE);

    I.updateLocationAfterHoist();
  }
};
} // namespace

/// Attempt to hoist loop invariant calls (for address payloads)
/// FIXME: This is a temporary workaround (should be done by IGC). We should
/// remove it once that feature is implemented.
void mlir::triton::intel::LICM(llvm::Module &mod) {
  std::set<llvm::Function *> kernels;
  uint32_t numKernels = 0;
  for (llvm::Function &function : mod.functions())
    if (function.getCallingConv() == CallingConv::SPIR_KERNEL) {
      kernels.insert(&function);
      ++numKernels;
    }
  assert(numKernels == 1 && "Expecting a single SPIR kernel");
  llvm::Function *kernel = *kernels.begin();

  FunctionAnalysisManager FAM;
  FAM.registerPass([&] { return AssumptionAnalysis(); });
  FAM.registerPass([&] { return DominatorTreeAnalysis(); });
  FAM.registerPass([&] { return LoopAnalysis(); });
  FAM.registerPass([&] { return PassInstrumentationAnalysis(); });
  FAM.registerPass([&] { return ScalarEvolutionAnalysis(); });
  FAM.registerPass([&] { return TargetLibraryAnalysis(); });
  FAM.registerPass([&] { return TargetIRAnalysis(); });
  FAM.registerPass([&] { return BasicAA(); });
  FAM.registerPass([&] { return ScopedNoAliasAA(); });
  FAM.registerPass([&] { return TypeBasedAA(); });
  FAM.registerPass([&] { return MemorySSAAnalysis(); });
  FAM.registerPass([&] {
    AAManager AA;
    AA.registerFunctionAnalysis<BasicAA>();
    AA.registerFunctionAnalysis<ScopedNoAliasAA>();
    AA.registerFunctionAnalysis<TypeBasedAA>();
    return AA;
  });

  FunctionPassManager FPM;
  FPM.addPass(LICMPass());
  FPM.run(*kernel, FAM);
}
