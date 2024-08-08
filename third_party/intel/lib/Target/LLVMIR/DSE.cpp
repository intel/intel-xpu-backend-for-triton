#include "third_party/intel/include/Target/LLVMIR/DSE.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/MemorySSAUpdater.h"
#include "llvm/Analysis/ScopedNoAliasAA.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/TypeBasedAliasAnalysis.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Value.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/AssumeBundleBuilder.h"
#include "llvm/Transforms/Utils/Local.h"
#include <map>
#include <optional>

using namespace llvm;

static constexpr unsigned MemorySSADefsPerBlockLimit = 1000;

namespace {

using OverlapIntervalsTy = std::map<int64_t, int64_t>;
using InstOverlapIntervalsTy = DenseMap<Instruction *, OverlapIntervalsTy>;

struct DSEState {
  DSEState(const DSEState &) = delete;
  DSEState &operator=(const DSEState &) = delete;
  DSEState(Function &F, AliasAnalysis &AA, MemorySSA &MSSA, DominatorTree &DT,
           const TargetLibraryInfo &TLI, const LoopInfo &LI)
      : F(F), AA(AA), MSSA(MSSA), DT(DT), TLI(TLI), LI(LI) {
    // Collect blocks with throwing instructions not modeled in MemorySSA and
    // alloc-like objects.
    unsigned PO = 0;
    for (BasicBlock *BB : post_order(&F)) {
      PostOrderNumbers[BB] = PO++;
      for (Instruction &I : *BB) {
        MemoryAccess *MA = MSSA.getMemoryAccess(&I);
        auto *MD = dyn_cast_or_null<MemoryDef>(MA);
        if (MD && MemDefs.size() < MemorySSADefsPerBlockLimit &&
            (getLocForWrite(&I) || isMemTerminatorInst(&I)))
          MemDefs.push_back(MD);
      }
    }
  }

  std::optional<MemoryLocation> getLocForWrite(Instruction *I) const {
    if (!I->mayWriteToMemory())
      return std::nullopt;

    if (auto *CB = dyn_cast<CallBase>(I))
      return MemoryLocation::getForDest(CB, TLI);

    return MemoryLocation::getOrNone(I);
  }

  /// Returns true if \p I is a memory terminator instruction like
  /// llvm.lifetime.end or free.
  bool isMemTerminatorInst(Instruction *I) const {
    auto *CB = dyn_cast<CallBase>(I);
    return CB && (CB->getIntrinsicID() == Intrinsic::lifetime_end ||
                  getFreedOperand(CB, &TLI) != nullptr);
  }

  /// Delete dead memory defs and recursively add their operands to ToRemove if
  /// they became dead.
  void
  deleteDeadInstruction(Instruction *SI,
                        SmallPtrSetImpl<MemoryAccess *> *Deleted = nullptr) {
    MemorySSAUpdater Updater(&MSSA);
    SmallVector<Instruction *, 32> NowDeadInsts;
    NowDeadInsts.push_back(SI);

    while (!NowDeadInsts.empty()) {
      Instruction *DeadInst = NowDeadInsts.pop_back_val();

      // Try to preserve debug information attached to the dead instruction.
      salvageDebugInfo(*DeadInst);
      salvageKnowledge(DeadInst);

      // Remove the Instruction from MSSA.
      MemoryAccess *MA = MSSA.getMemoryAccess(DeadInst);
      bool IsMemDef = MA && isa<MemoryDef>(MA);
      if (MA) {
        if (IsMemDef) {
          auto *MD = cast<MemoryDef>(MA);
          SkipStores.insert(MD);
          if (Deleted)
            Deleted->insert(MD);
          if (auto *SI = dyn_cast<StoreInst>(MD->getMemoryInst())) {
            if (SI->getValueOperand()->getType()->isPointerTy()) {
              const Value *UO = getUnderlyingObject(SI->getValueOperand());
              InvisibleToCallerAfterRet.erase(UO);
            }
          }
        }

        Updater.removeMemoryAccess(MA);
      }

      auto I = IOLs.find(DeadInst->getParent());
      if (I != IOLs.end())
        I->second.erase(DeadInst);
      // Remove its operands
      for (Use &O : DeadInst->operands())
        if (Instruction *OpI = dyn_cast<Instruction>(O)) {
          O.set(PoisonValue::get(O->getType()));
          if (isInstructionTriviallyDead(OpI, &TLI))
            NowDeadInsts.push_back(OpI);
        }

      //  Remove memory defs directly if they don't produce results, but only
      //  queue other dead instructions for later removal. They may have been
      //  used as memory locations that have been cached by BatchAA. Removing
      //  them here may lead to newly created instructions to be allocated at
      //  the same address, yielding stale cache entries.
      if (IsMemDef && DeadInst->getType()->isVoidTy())
        DeadInst->eraseFromParent();
      else
        ToRemove.push_back(DeadInst);
    }
  }

  Function &F;
  AliasAnalysis &AA;
  MemorySSA &MSSA;
  DominatorTree &DT;
  const TargetLibraryInfo &TLI;
  const LoopInfo &LI;

  // All MemoryDefs that potentially could kill other MemDefs.
  SmallVector<MemoryDef *, 64> MemDefs;
  // Any that should be skipped as they are already deleted
  SmallPtrSet<MemoryAccess *, 4> SkipStores;
  // Keep track of all of the objects that are invisible to the caller after
  // the function returns.
  DenseMap<const Value *, bool> InvisibleToCallerAfterRet;
  // Post-order numbers for each basic block. Used to figure out if memory
  // accesses are executed before another access.
  DenseMap<BasicBlock *, unsigned> PostOrderNumbers;
  /// Keep track of instructions (partly) overlapping with killing MemoryDefs
  /// per basic block.
  MapVector<BasicBlock *, InstOverlapIntervalsTy> IOLs;
  /// Dead instructions to be removed at the end of DSE.
  SmallVector<Instruction *> ToRemove;
};

class SetAddressPayloadInfo {
  friend raw_ostream &operator<<(raw_ostream &, const SetAddressPayloadInfo &);

public:
  static constexpr StringLiteral prefix =
      "__builtin_IB_subgroup_setBlock2DAddressPayloadBlock";

  /// The field to set in the address payload.
  enum class Field { X, Y, Unknown };

  SetAddressPayloadInfo() = default;

  SetAddressPayloadInfo(CallInst *CI, MemoryAccess *memAccess)
      : ptr(nullptr), val(nullptr), field(Field::Unknown),
        memAccess(memAccess) {
    const StringRef funcName = CI->getCalledFunction()->getName();
    assert(funcName.starts_with(prefix) &&
           "Expecting address payload set function");
    assert(CI->arg_size() == 2 && "Expecting 2 arguments");

    ptr = CI->getArgOperand(0);
    val = CI->getArgOperand(1);
    field = funcName.ends_with("X")
                ? Field::X
                : (funcName.ends_with("Y") ? Field::Y : Field::Unknown);
    assert(field != Field::Unknown && "Expecting X or Y");
  }

  bool samePtrAndValueAndField(const SetAddressPayloadInfo &other) const {
    return ptr == other.ptr && val == other.val && field == other.field;
  }
  bool samePtrAndField(const SetAddressPayloadInfo &other) const {
    return ptr == other.ptr && field == other.field;
  }
  bool operator==(const SetAddressPayloadInfo &other) const {
    return ptr == other.ptr && val == other.val && field == other.field &&
           memAccess == other.memAccess;
  }

private:
  Value *ptr;              // the pointer to the payload
  Value *val;              // the value to set
  Field field;             // the payload field to set (X or Y)
  MemoryAccess *memAccess; // memory access associated with this set payload.
};

raw_ostream &operator<<(raw_ostream &OS, const SetAddressPayloadInfo &info) {
  OS << "SetAddressPayloadInfo: ptr=" << info.ptr->getNameOrAsOperand()
     << ", val=" << info.val->getNameOrAsOperand() << ", field=";
  switch (info.field) {
  case SetAddressPayloadInfo::Field::X:
    OS << "X";
    break;
  case SetAddressPayloadInfo::Field::Y:
    OS << "Y";
    break;
  case SetAddressPayloadInfo::Field::Unknown:
    OS << "Unknown";
    break;
  }
  OS << ", memAccess=" << *info.memAccess;
  return OS;
}

class DSEPass : public PassInfoMixin<DSEPass> {
public:
  DSEPass(bool trace) : trace(trace) {}

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
    auto &LI = AM.getResult<LoopAnalysis>(F);
    auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
    auto &AA = AM.getResult<AAManager>(F);
    auto &MSSA = AM.getResult<MemorySSAAnalysis>(F).getMSSA();
    auto &TLI = AM.getResult<TargetLibraryAnalysis>(F);

    if (!eliminateDeadStores(F, AA, MSSA, DT, TLI, LI))
      return PreservedAnalyses::all();

    PreservedAnalyses PA;
    PA.preserveSet<CFGAnalyses>();
    PA.preserve<MemorySSAAnalysis>();
    PA.preserve<LoopAnalysis>();
    return PA;
  }

private:
  bool trace;

  static bool isCandidate(Instruction *I) {
    if (auto *CI = dyn_cast<CallInst>(I))
      return CI->getCalledFunction()->getName().starts_with(
          SetAddressPayloadInfo::prefix);
    return false;
  }

  bool getDomMemoryDef(const SetAddressPayloadInfo &info,
                       MemoryAccess *StartAccess, MemorySSA &MSSA,
                       SmallVector<SetAddressPayloadInfo> &SetPayloads) const {
    [[maybe_unused]] auto print =
        [](const SmallVector<SetAddressPayloadInfo> &SetPayloads) {
          llvm::errs() << "SetPayloads:\n";
          for (const SetAddressPayloadInfo &e : SetPayloads)
            llvm::errs() << "   e: " << e << "\n";
        };

    for (MemoryAccess *Current = StartAccess; Current;
         Current = cast<MemoryDef>(Current)->getDefiningAccess()) {
      if (trace) {
        llvm::errs() << "   visiting " << *Current;
        if (!MSSA.isLiveOnEntryDef(Current) && isa<MemoryUseOrDef>(Current))
          llvm::errs() << " ("
                       << *cast<MemoryUseOrDef>(Current)->getMemoryInst()
                       << ")";
        llvm::errs() << "\n";
      }

      if (MSSA.isLiveOnEntryDef(Current) || isa<MemoryPhi>(Current))
        return false;
      if (isa<MemoryUse>(Current))
        continue; // reads aren't interesting

      Instruction *memInstr = cast<MemoryDef>(Current)->getMemoryInst();
      if (!isCandidate(memInstr)) {
        // possible store clobber, bail out.
        SetPayloads.clear();
        continue;
      }

      SetAddressPayloadInfo prev(cast<CallInst>(memInstr), Current);
      if (llvm::none_of(SetPayloads,
                        [&](const auto &entry) { return entry == prev; })) {
        continue;
      }

      if (info.samePtrAndValueAndField(prev)) {
        if (trace)
          llvm::errs() << "   ... found dead entry: " << info << "\n";

        for (auto it = SetPayloads.begin(); it != SetPayloads.end(); ++it) {
          if (*it == info) {
            SetPayloads.erase(it);
            break;
          }
        }

        if (trace)
          print(SetPayloads);

        return true;
      }
      if (info.samePtrAndField(prev)) {
        if (trace)
          llvm::errs() << "   ... remove previous entry: " << prev << "\n";

        for (auto it = SetPayloads.begin(); it != SetPayloads.end(); ++it) {
          if (*it == prev) {
            SetPayloads.erase(it);
            break;
          }
        }

        if (trace)
          print(SetPayloads);
      }
    }

    return false;
  }

  bool eliminateDeadStores(Function &F, AliasAnalysis &AA, MemorySSA &MSSA,
                           DominatorTree &DT, const TargetLibraryInfo &TLI,
                           const LoopInfo &LI) const {
    bool Changed = false;

    DSEState State(F, AA, MSSA, DT, TLI, LI);

    SmallVector<SetAddressPayloadInfo> SetPayloads;

    for (unsigned I = 0; I < State.MemDefs.size(); ++I) {
      MemoryDef *KillingDef = State.MemDefs[I];
      Instruction *KillingI = KillingDef->getMemoryInst();
      if (!isCandidate(KillingI))
        continue;

      std::optional<MemoryLocation> MaybeKillingLoc =
          State.getLocForWrite(KillingI);
      if (!MaybeKillingLoc)
        continue;

      if (trace)
        llvm::errs() << "\nKillingDef: " << *KillingDef << " (" << *KillingI
                     << ")\n";

      SetAddressPayloadInfo info(cast<CallInst>(KillingI), KillingDef);
      SetPayloads.push_back(info);

      if (trace) {
        llvm::errs() << "SetPayloads:\n";
        for (auto &e : SetPayloads)
          llvm::errs() << "   e: " << e << "\n";
      }

      // Worklist of MemoryAccesses that may be killed by KillingDef.
      SmallSetVector<MemoryAccess *, 8> ToCheck;
      ToCheck.insert(KillingDef->getDefiningAccess());

      // Check if MemoryAccesses in the worklist are killed by KillingDef.
      for (unsigned J = 0; J < ToCheck.size(); ++J) {
        MemoryAccess *Current = ToCheck[J];
        bool isDead = getDomMemoryDef(info, Current, MSSA, SetPayloads);
        if (isDead) {
          Instruction *DeadI = KillingI;
          State.deleteDeadInstruction(DeadI);
        }
      }
    }

    while (!State.ToRemove.empty()) {
      Instruction *DeadInst = State.ToRemove.pop_back_val();
      DeadInst->eraseFromParent();
      Changed = true;
    }

    return Changed;
  }
};

} // namespace

/// FIXME: This is a temporary workaround (should be done by IGC). We should
/// remove it once that feature is implemented.
void mlir::triton::intel::DSE(llvm::Module &mod, bool trace) {
  FunctionAnalysisManager FAM;
  FAM.registerPass([&] { return AssumptionAnalysis(); });
  FAM.registerPass([&] { return DominatorTreeAnalysis(); });
  FAM.registerPass([&] { return LoopAnalysis(); });
  FAM.registerPass([&] { return PassInstrumentationAnalysis(); });
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
  FPM.addPass(DSEPass(trace));

  for (llvm::Function &function : mod.functions()) {
    if (function.getCallingConv() == CallingConv::SPIR_KERNEL)
      FPM.run(function, FAM);
  }
}
