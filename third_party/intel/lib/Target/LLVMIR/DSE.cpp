#include "third_party/intel/include/Target/LLVMIR/DSE.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/MemorySSAUpdater.h"
#include "llvm/Analysis/MustExecute.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ScopedNoAliasAA.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/TypeBasedAliasAnalysis.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Value.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Casting.h"
#include "llvm/Transforms/Utils/AssumeBundleBuilder.h"
#include "llvm/Transforms/Utils/BuildLibCalls.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;
using namespace PatternMatch;

static constexpr unsigned MemorySSAScanLimit = 150;
static constexpr unsigned MemorySSAUpwardsStepLimit = 90;
static constexpr unsigned MemorySSAPartialStoreLimit = 5;
static constexpr unsigned MemorySSADefsPerBlockLimit = 5000;
static constexpr unsigned MemorySSASameBBStepCost = 1;
static constexpr unsigned MemorySSAOtherBBStepCost = 5;
static constexpr unsigned MemorySSAPathCheckLimit = 50;

namespace {

std::optional<TypeSize> getPointerSize(const Value *V, const DataLayout &DL,
                                       const TargetLibraryInfo &TLI,
                                       const Function *F) {
  uint64_t Size;
  ObjectSizeOpts Opts;
  Opts.NullIsUnknownSize = NullPointerIsDefined(F);

  if (getObjectSize(V, Size, DL, &TLI, Opts))
    return TypeSize::getFixed(Size);
  return std::nullopt;
}

// Returns true if \p I is an intrinsic that does not read or write memory.
bool isNoopIntrinsic(Instruction *I) {
  if (const IntrinsicInst *II = dyn_cast<IntrinsicInst>(I)) {
    switch (II->getIntrinsicID()) {
    case Intrinsic::lifetime_start:
    case Intrinsic::lifetime_end:
    case Intrinsic::invariant_end:
    case Intrinsic::launder_invariant_group:
    case Intrinsic::assume:
      return true;
    case Intrinsic::dbg_declare:
    case Intrinsic::dbg_label:
    case Intrinsic::dbg_value:
      llvm_unreachable("Intrinsic should not be modeled in MemorySSA");
    default:
      return false;
    }
  }
  return false;
}

// Check if we can ignore \p D for DSE.
bool canSkipDef(MemoryDef *D, bool DefVisibleToCaller) {
  Instruction *DI = D->getMemoryInst();
  // Calls that only access inaccessible memory cannot read or write any memory
  // locations we consider for elimination.
  if (auto *CB = dyn_cast<CallBase>(DI))
    if (CB->onlyAccessesInaccessibleMemory())
      return true;

  // We can eliminate stores to locations not visible to the caller across
  // throwing instructions.
  if (DI->mayThrow() && !DefVisibleToCaller)
    return true;

  // We can remove the dead stores, irrespective of the fence and its ordering
  // (release/acquire/seq_cst). Fences only constraints the ordering of
  // already visible stores, it does not make a store visible to other
  // threads. So, skipping over a fence does not change a store from being
  // dead.
  if (isa<FenceInst>(DI))
    return true;

  // Skip intrinsics that do not really read or modify memory.
  if (isNoopIntrinsic(DI))
    return true;

  return false;
}

enum OverwriteResult {
  OW_Begin,
  OW_Complete,
  OW_End,
  OW_PartialEarlierWithFullLater,
  OW_MaybePartial,
  OW_None,
  OW_Unknown
};

/// Check if two instruction are masked stores that completely
/// overwrite one another. More specifically, \p KillingI has to
/// overwrite \p DeadI.
OverwriteResult isMaskedStoreOverwrite(const Instruction *KillingI,
                                       const Instruction *DeadI,
                                       BatchAAResults &AA) {
  const auto *KillingII = dyn_cast<IntrinsicInst>(KillingI);
  const auto *DeadII = dyn_cast<IntrinsicInst>(DeadI);
  if (KillingII == nullptr || DeadII == nullptr)
    return OW_Unknown;
  if (KillingII->getIntrinsicID() != DeadII->getIntrinsicID())
    return OW_Unknown;
  if (KillingII->getIntrinsicID() == Intrinsic::masked_store) {
    // Type size.
    VectorType *KillingTy =
        cast<VectorType>(KillingII->getArgOperand(0)->getType());
    VectorType *DeadTy = cast<VectorType>(DeadII->getArgOperand(0)->getType());
    if (KillingTy->getScalarSizeInBits() != DeadTy->getScalarSizeInBits())
      return OW_Unknown;
    // Element count.
    if (KillingTy->getElementCount() != DeadTy->getElementCount())
      return OW_Unknown;
    // Pointers.
    Value *KillingPtr = KillingII->getArgOperand(1)->stripPointerCasts();
    Value *DeadPtr = DeadII->getArgOperand(1)->stripPointerCasts();
    if (KillingPtr != DeadPtr && !AA.isMustAlias(KillingPtr, DeadPtr))
      return OW_Unknown;
    // Masks.
    // TODO: check that KillingII's mask is a superset of the DeadII's mask.
    if (KillingII->getArgOperand(3) != DeadII->getArgOperand(3))
      return OW_Unknown;
    return OW_Complete;
  }
  return OW_Unknown;
}

struct DSEState {
  DSEState(const DSEState &) = delete;
  DSEState &operator=(const DSEState &) = delete;
  DSEState(Function &F, AliasAnalysis &AA, MemorySSA &MSSA, DominatorTree &DT,
           PostDominatorTree &PDT, const TargetLibraryInfo &TLI,
           const LoopInfo &LI)
      : F(F), AA(AA), EI(DT, &LI), BatchAA(AA, &EI), MSSA(MSSA), DT(DT),
        PDT(PDT), TLI(TLI), DL(F.getParent()->getDataLayout()), LI(LI) {
    // Collect blocks with throwing instructions not modeled in MemorySSA and
    // alloc-like objects.
    unsigned PO = 0;
    for (BasicBlock *BB : post_order(&F)) {
      PostOrderNumbers[BB] = PO++;
      for (Instruction &I : *BB) {
        MemoryAccess *MA = MSSA.getMemoryAccess(&I);
        if (I.mayThrow() && !MA)
          ThrowingBlocks.insert(I.getParent());

        auto *MD = dyn_cast_or_null<MemoryDef>(MA);
        if (MD && MemDefs.size() < MemorySSADefsPerBlockLimit &&
            (getLocForWrite(&I) || isMemTerminatorInst(&I)))
          MemDefs.push_back(MD);
      }
    }

    // Treat byval or inalloca arguments the same as Allocas, stores to them are
    // dead at the end of the function.
    for (Argument &AI : F.args())
      if (AI.hasPassPointeeByValueCopyAttr())
        InvisibleToCallerAfterRet.insert({&AI, true});

    // Collect whether there is any irreducible control flow in the function.
    ContainsIrreducibleLoops = mayContainIrreducibleControl(F, &LI);

    AnyUnreachableExit = any_of(PDT.roots(), [](const BasicBlock *E) {
      return isa<UnreachableInst>(E->getTerminator());
    });
  }

  LocationSize strengthenLocationSize(const Instruction *I,
                                      LocationSize Size) const {
    if (auto *CB = dyn_cast<CallBase>(I)) {
      LibFunc F;
      if (TLI.getLibFunc(*CB, F) && TLI.has(F) &&
          (F == LibFunc_memset_chk || F == LibFunc_memcpy_chk)) {
        // Use the precise location size specified by the 3rd argument
        // for determining KillingI overwrites DeadLoc if it is a memset_chk
        // instruction. memset_chk will write either the amount specified as 3rd
        // argument or the function will immediately abort and exit the program.
        // NOTE: AA may determine NoAlias if it can prove that the access size
        // is larger than the allocation size due to that being UB. To avoid
        // returning potentially invalid NoAlias results by AA, limit the use of
        // the precise location size to isOverwrite.
        if (const auto *Len = dyn_cast<ConstantInt>(CB->getArgOperand(2)))
          return LocationSize::precise(Len->getZExtValue());
      }
    }
    return Size;
  }

  /// Return 'OW_Complete' if a store to the 'KillingLoc' location (by \p
  /// KillingI instruction) completely overwrites a store to the 'DeadLoc'
  /// location (by \p DeadI instruction).
  /// Return OW_MaybePartial if \p KillingI does not completely overwrite
  /// \p DeadI, but they both write to the same underlying object. In that
  /// case, use isPartialOverwrite to check if \p KillingI partially overwrites
  /// \p DeadI. Returns 'OR_None' if \p KillingI is known to not overwrite the
  /// \p DeadI. Returns 'OW_Unknown' if nothing can be determined.
  OverwriteResult isOverwrite(const Instruction *KillingI,
                              const Instruction *DeadI,
                              const MemoryLocation &KillingLoc,
                              const MemoryLocation &DeadLoc,
                              int64_t &KillingOff, int64_t &DeadOff) {
    // AliasAnalysis does not always account for loops. Limit overwrite checks
    // to dependencies for which we can guarantee they are independent of any
    // loops they are in.
    if (!isGuaranteedLoopIndependent(DeadI, KillingI, DeadLoc))
      return OW_Unknown;

    LocationSize KillingLocSize =
        strengthenLocationSize(KillingI, KillingLoc.Size);
    const Value *DeadPtr = DeadLoc.Ptr->stripPointerCasts();
    const Value *KillingPtr = KillingLoc.Ptr->stripPointerCasts();
    const Value *DeadUndObj = getUnderlyingObject(DeadPtr);
    const Value *KillingUndObj = getUnderlyingObject(KillingPtr);

    // Check whether the killing store overwrites the whole object, in which
    // case the size/offset of the dead store does not matter.
    if (DeadUndObj == KillingUndObj && KillingLocSize.isPrecise() &&
        isIdentifiedObject(KillingUndObj)) {
      std::optional<TypeSize> KillingUndObjSize =
          getPointerSize(KillingUndObj, DL, TLI, &F);
      if (KillingUndObjSize && *KillingUndObjSize == KillingLocSize.getValue())
        return OW_Complete;
    }

    // FIXME: Vet that this works for size upper-bounds. Seems unlikely that
    // we'll get imprecise values here, though (except for unknown sizes).
    if (!KillingLocSize.isPrecise() || !DeadLoc.Size.isPrecise()) {
      // In case no constant size is known, try to an IR values for the number
      // of bytes written and check if they match.
      const auto *KillingMemI = dyn_cast<MemIntrinsic>(KillingI);
      const auto *DeadMemI = dyn_cast<MemIntrinsic>(DeadI);
      if (KillingMemI && DeadMemI) {
        const Value *KillingV = KillingMemI->getLength();
        const Value *DeadV = DeadMemI->getLength();
        if (KillingV == DeadV && BatchAA.isMustAlias(DeadLoc, KillingLoc))
          return OW_Complete;
      }

      // Masked stores have imprecise locations, but we can reason about them
      // to some extent.
      return isMaskedStoreOverwrite(KillingI, DeadI, BatchAA);
    }

    const TypeSize KillingSize = KillingLocSize.getValue();
    const TypeSize DeadSize = DeadLoc.Size.getValue();
    // Bail on doing Size comparison which depends on AA for now
    // TODO: Remove AnyScalable once Alias Analysis deal with scalable vectors
    const bool AnyScalable =
        DeadSize.isScalable() || KillingLocSize.isScalable();

    if (AnyScalable)
      return OW_Unknown;
    // Query the alias information
    AliasResult AAR = BatchAA.alias(KillingLoc, DeadLoc);

    // If the start pointers are the same, we just have to compare sizes to see
    // if the killing store was larger than the dead store.
    if (AAR == AliasResult::MustAlias) {
      // Make sure that the KillingSize size is >= the DeadSize size.
      if (KillingSize >= DeadSize)
        return OW_Complete;
    }

    // If we hit a partial alias we may have a full overwrite
    if (AAR == AliasResult::PartialAlias && AAR.hasOffset()) {
      int32_t Off = AAR.getOffset();
      if (Off >= 0 && (uint64_t)Off + DeadSize <= KillingSize)
        return OW_Complete;
    }

    // If we can't resolve the same pointers to the same object, then we can't
    // analyze them at all.
    if (DeadUndObj != KillingUndObj) {
      // Non aliasing stores to different objects don't overlap. Note that
      // if the killing store is known to overwrite whole object (out of
      // bounds access overwrites whole object as well) then it is assumed to
      // completely overwrite any store to the same object even if they don't
      // actually alias (see next check).
      if (AAR == AliasResult::NoAlias)
        return OW_None;
      return OW_Unknown;
    }

    // Okay, we have stores to two completely different pointers.  Try to
    // decompose the pointer into a "base + constant_offset" form.  If the base
    // pointers are equal, then we can reason about the two stores.
    DeadOff = 0;
    KillingOff = 0;
    const Value *DeadBasePtr =
        GetPointerBaseWithConstantOffset(DeadPtr, DeadOff, DL);
    const Value *KillingBasePtr =
        GetPointerBaseWithConstantOffset(KillingPtr, KillingOff, DL);

    // If the base pointers still differ, we have two completely different
    // stores.
    if (DeadBasePtr != KillingBasePtr)
      return OW_Unknown;

    // The killing access completely overlaps the dead store if and only if
    // both start and end of the dead one is "inside" the killing one:
    //    |<->|--dead--|<->|
    //    |-----killing------|
    // Accesses may overlap if and only if start of one of them is "inside"
    // another one:
    //    |<->|--dead--|<-------->|
    //    |-------killing--------|
    //           OR
    //    |-------dead-------|
    //    |<->|---killing---|<----->|
    //
    // We have to be careful here as *Off is signed while *.Size is unsigned.

    // Check if the dead access starts "not before" the killing one.
    if (DeadOff >= KillingOff) {
      // If the dead access ends "not after" the killing access then the
      // dead one is completely overwritten by the killing one.
      if (uint64_t(DeadOff - KillingOff) + DeadSize <= KillingSize)
        return OW_Complete;
      // If start of the dead access is "before" end of the killing access
      // then accesses overlap.
      else if ((uint64_t)(DeadOff - KillingOff) < KillingSize)
        return OW_MaybePartial;
    }
    // If start of the killing access is "before" end of the dead access then
    // accesses overlap.
    else if ((uint64_t)(KillingOff - DeadOff) < DeadSize) {
      return OW_MaybePartial;
    }

    // Can reach here only if accesses are known not to overlap.
    return OW_None;
  }

  bool isInvisibleToCallerAfterRet(const Value *V) {
    if (isa<AllocaInst>(V))
      return true;
    auto I = InvisibleToCallerAfterRet.insert({V, false});
    if (I.second) {
      if (!isInvisibleToCallerOnUnwind(V)) {
        I.first->second = false;
      } else if (isNoAliasCall(V)) {
        I.first->second = !PointerMayBeCaptured(V, true, false);
      }
    }
    return I.first->second;
  }

  bool isInvisibleToCallerOnUnwind(const Value *V) {
    bool RequiresNoCaptureBeforeUnwind;
    if (!isNotVisibleOnUnwind(V, RequiresNoCaptureBeforeUnwind))
      return false;
    if (!RequiresNoCaptureBeforeUnwind)
      return true;

    auto I = CapturedBeforeReturn.insert({V, true});
    if (I.second)
      // NOTE: This could be made more precise by PointerMayBeCapturedBefore
      // with the killing MemoryDef. But we refrain from doing so for now to
      // limit compile-time and this does not cause any changes to the number
      // of stores removed on a large test set in practice.
      I.first->second = PointerMayBeCaptured(V, false, true);
    return !I.first->second;
  }

  std::optional<MemoryLocation> getLocForWrite(Instruction *I) const {
    if (!I->mayWriteToMemory())
      return std::nullopt;

    if (auto *CB = dyn_cast<CallBase>(I))
      return MemoryLocation::getForDest(CB, TLI);

    return MemoryLocation::getOrNone(I);
  }

  /// Assuming this instruction has a dead analyzable write, can we delete
  /// this instruction?
  bool isRemovable(Instruction *I) {
    assert(getLocForWrite(I) && "Must have analyzable write");

    // Don't remove volatile/atomic stores.
    if (StoreInst *SI = dyn_cast<StoreInst>(I))
      return SI->isUnordered();

    if (auto *CB = dyn_cast<CallBase>(I)) {
      // Don't remove volatile memory intrinsics.
      if (auto *MI = dyn_cast<MemIntrinsic>(CB))
        return !MI->isVolatile();

      // Never remove dead lifetime intrinsics, e.g. because they are followed
      // by a free.
      if (CB->isLifetimeStartOrEnd())
        return false;

      return CB->use_empty() && CB->willReturn() && CB->doesNotThrow() &&
             !CB->isTerminator();
    }

    return false;
  }

  /// Returns true if \p UseInst completely overwrites \p DefLoc
  /// (stored by \p DefInst).
  bool isCompleteOverwrite(const MemoryLocation &DefLoc, Instruction *DefInst,
                           Instruction *UseInst) {
    // UseInst has a MemoryDef associated in MemorySSA. It's possible for a
    // MemoryDef to not write to memory, e.g. a volatile load is modeled as a
    // MemoryDef.
    if (!UseInst->mayWriteToMemory())
      return false;

    if (auto *CB = dyn_cast<CallBase>(UseInst))
      if (CB->onlyAccessesInaccessibleMemory())
        return false;

    int64_t InstWriteOffset, DepWriteOffset;
    if (auto CC = getLocForWrite(UseInst))
      return isOverwrite(UseInst, DefInst, *CC, DefLoc, InstWriteOffset,
                         DepWriteOffset) == OW_Complete;
    return false;
  }

  /// If \p I is a memory  terminator like llvm.lifetime.end or free, return a
  /// pair with the MemoryLocation terminated by \p I and a boolean flag
  /// indicating whether \p I is a free-like call.
  std::optional<std::pair<MemoryLocation, bool>>
  getLocForTerminator(Instruction *I) const {
    uint64_t Len;
    Value *Ptr;
    if (match(I, m_Intrinsic<Intrinsic::lifetime_end>(m_ConstantInt(Len),
                                                      m_Value(Ptr))))
      return {std::make_pair(MemoryLocation(Ptr, Len), false)};

    if (auto *CB = dyn_cast<CallBase>(I)) {
      if (Value *FreedOp = getFreedOperand(CB, &TLI))
        return {std::make_pair(MemoryLocation::getAfter(FreedOp), true)};
    }

    return std::nullopt;
  }

  /// Returns true if \p I is a memory terminator instruction like
  /// llvm.lifetime.end or free.
  bool isMemTerminatorInst(Instruction *I) const {
    auto *CB = dyn_cast<CallBase>(I);
    return CB && (CB->getIntrinsicID() == Intrinsic::lifetime_end ||
                  getFreedOperand(CB, &TLI) != nullptr);
  }

  /// Returns true if \p MaybeTerm is a memory terminator for \p Loc from
  /// instruction \p AccessI.
  bool isMemTerminator(const MemoryLocation &Loc, Instruction *AccessI,
                       Instruction *MaybeTerm) {
    std::optional<std::pair<MemoryLocation, bool>> MaybeTermLoc =
        getLocForTerminator(MaybeTerm);

    if (!MaybeTermLoc)
      return false;

    // If the terminator is a free-like call, all accesses to the underlying
    // object can be considered terminated.
    if (getUnderlyingObject(Loc.Ptr) !=
        getUnderlyingObject(MaybeTermLoc->first.Ptr))
      return false;

    auto TermLoc = MaybeTermLoc->first;
    if (MaybeTermLoc->second) {
      const Value *LocUO = getUnderlyingObject(Loc.Ptr);
      return BatchAA.isMustAlias(TermLoc.Ptr, LocUO);
    }
    int64_t InstWriteOffset = 0;
    int64_t DepWriteOffset = 0;
    return isOverwrite(MaybeTerm, AccessI, TermLoc, Loc, InstWriteOffset,
                       DepWriteOffset) == OW_Complete;
  }

  // Returns true if \p Use may read from \p DefLoc.
  bool isReadClobber(const MemoryLocation &DefLoc, Instruction *UseInst) {
    if (isNoopIntrinsic(UseInst))
      return false;

    // Monotonic or weaker atomic stores can be re-ordered and do not need to be
    // treated as read clobber.
    if (auto SI = dyn_cast<StoreInst>(UseInst))
      return isStrongerThan(SI->getOrdering(), AtomicOrdering::Monotonic);

    if (!UseInst->mayReadFromMemory())
      return false;

    if (auto *CB = dyn_cast<CallBase>(UseInst))
      if (CB->onlyAccessesInaccessibleMemory())
        return false;

    return isRefSet(BatchAA.getModRefInfo(UseInst, DefLoc));
  }

  /// Returns true if a dependency between \p Current and \p KillingDef is
  /// guaranteed to be loop invariant for the loops that they are in. Either
  /// because they are known to be in the same block, in the same loop level or
  /// by guaranteeing that \p CurrentLoc only references a single MemoryLocation
  /// during execution of the containing function.
  bool isGuaranteedLoopIndependent(const Instruction *Current,
                                   const Instruction *KillingDef,
                                   const MemoryLocation &CurrentLoc) {
    // If the dependency is within the same block or loop level (being careful
    // of irreducible loops), we know that AA will return a valid result for the
    // memory dependency. (Both at the function level, outside of any loop,
    // would also be valid but we currently disable that to limit compile time).
    if (Current->getParent() == KillingDef->getParent())
      return true;
    const Loop *CurrentLI = LI.getLoopFor(Current->getParent());
    if (!ContainsIrreducibleLoops && CurrentLI &&
        CurrentLI == LI.getLoopFor(KillingDef->getParent()))
      return true;
    // Otherwise check the memory location is invariant to any loops.
    return isGuaranteedLoopInvariant(CurrentLoc.Ptr);
  }

  /// Returns true if \p Ptr is guaranteed to be loop invariant for any possible
  /// loop. In particular, this guarantees that it only references a single
  /// MemoryLocation during execution of the containing function.
  bool isGuaranteedLoopInvariant(const Value *Ptr) {
    Ptr = Ptr->stripPointerCasts();
    if (auto *GEP = dyn_cast<GEPOperator>(Ptr))
      if (GEP->hasAllConstantIndices())
        Ptr = GEP->getPointerOperand()->stripPointerCasts();

    if (auto *I = dyn_cast<Instruction>(Ptr)) {
      return I->getParent()->isEntryBlock() ||
             (!ContainsIrreducibleLoops && !LI.getLoopFor(I->getParent()));
    }
    return true;
  }

  // Check for any extra throws between \p KillingI and \p DeadI that block
  // DSE.  This only checks extra maythrows (those that aren't MemoryDef's).
  // MemoryDef that may throw are handled during the walk from one def to the
  // next.
  bool mayThrowBetween(Instruction *KillingI, Instruction *DeadI,
                       const Value *KillingUndObj) {
    // First see if we can ignore it by using the fact that KillingI is an
    // alloca/alloca like object that is not visible to the caller during
    // execution of the function.
    if (KillingUndObj && isInvisibleToCallerOnUnwind(KillingUndObj))
      return false;

    if (KillingI->getParent() == DeadI->getParent())
      return ThrowingBlocks.count(KillingI->getParent());
    return !ThrowingBlocks.empty();
  }

  // Check if \p DeadI acts as a DSE barrier for \p KillingI. The following
  // instructions act as barriers:
  //  * A memory instruction that may throw and \p KillingI accesses a non-stack
  //  object.
  //  * Atomic stores stronger that monotonic.
  bool isDSEBarrier(const Value *KillingUndObj, Instruction *DeadI) {
    // If DeadI may throw it acts as a barrier, unless we are to an
    // alloca/alloca like object that does not escape.
    if (DeadI->mayThrow() && !isInvisibleToCallerOnUnwind(KillingUndObj))
      return true;

    // If DeadI is an atomic load/store stronger than monotonic, do not try to
    // eliminate/reorder it.
    if (DeadI->isAtomic()) {
      if (auto *LI = dyn_cast<LoadInst>(DeadI))
        return isStrongerThanMonotonic(LI->getOrdering());
      if (auto *SI = dyn_cast<StoreInst>(DeadI))
        return isStrongerThanMonotonic(SI->getOrdering());
      if (auto *ARMW = dyn_cast<AtomicRMWInst>(DeadI))
        return isStrongerThanMonotonic(ARMW->getOrdering());
      if (auto *CmpXchg = dyn_cast<AtomicCmpXchgInst>(DeadI))
        return isStrongerThanMonotonic(CmpXchg->getSuccessOrdering()) ||
               isStrongerThanMonotonic(CmpXchg->getFailureOrdering());
      llvm_unreachable("other instructions should be skipped in MemorySSA");
    }
    return false;
  }

  // Find a MemoryDef writing to \p KillingLoc and dominating \p StartAccess,
  // with no read access between them or on any other path to a function exit
  // block if \p KillingLoc is not accessible after the function returns. If
  // there is no such MemoryDef, return std::nullopt. The returned value may not
  // (completely) overwrite \p KillingLoc. Currently we bail out when we
  // encounter an aliasing MemoryUse (read).
  std::optional<MemoryAccess *>
  getDomMemoryDef(MemoryDef *KillingDef, MemoryAccess *StartAccess,
                  const MemoryLocation &KillingLoc, const Value *KillingUndObj,
                  unsigned &ScanLimit, unsigned &WalkerStepLimit,
                  bool IsMemTerm, unsigned &PartialLimit) {
    if (ScanLimit == 0 || WalkerStepLimit == 0) {
      llvm::errs() << "\n    ...  hit scan limit\n";
      return std::nullopt;
    }

    MemoryAccess *Current = StartAccess;
    Instruction *KillingI = KillingDef->getMemoryInst();
    llvm::errs() << "  trying to get dominating access\n";

    // Only optimize defining access of KillingDef when directly starting at its
    // defining access. The defining access also must only access KillingLoc. At
    // the moment we only support instructions with a single write location, so
    // it should be sufficient to disable optimizations for instructions that
    // also read from memory.
    bool CanOptimize = KillingDef->getDefiningAccess() == StartAccess &&
                       !KillingI->mayReadFromMemory();

    // Find the next clobbering Mod access for DefLoc, starting at StartAccess.
    std::optional<MemoryLocation> CurrentLoc;
    for (;; Current = cast<MemoryDef>(Current)->getDefiningAccess()) {
      if (1) {
        llvm::errs() << "   visiting " << *Current;
        if (!MSSA.isLiveOnEntryDef(Current) && isa<MemoryUseOrDef>(Current))
          llvm::errs() << " ("
                       << *cast<MemoryUseOrDef>(Current)->getMemoryInst()
                       << ")";
        llvm::errs() << "\n";
      };

      // Reached TOP.
      if (MSSA.isLiveOnEntryDef(Current)) {
        llvm::errs() << "   ...  found LiveOnEntryDef\n";
        if (CanOptimize && Current != KillingDef->getDefiningAccess())
          // The first clobbering def is... none.
          KillingDef->setOptimized(Current);
        return std::nullopt;
      }

      // Cost of a step. Accesses in the same block are more likely to be valid
      // candidates for elimination, hence consider them cheaper.
      unsigned StepCost = KillingDef->getBlock() == Current->getBlock()
                              ? MemorySSASameBBStepCost
                              : MemorySSAOtherBBStepCost;
      if (WalkerStepLimit <= StepCost) {
        llvm::errs() << "   ...  hit walker step limit\n";
        return std::nullopt;
      }
      WalkerStepLimit -= StepCost;

      // Return for MemoryPhis. They cannot be eliminated directly and the
      // caller is responsible for traversing them.
      if (isa<MemoryPhi>(Current)) {
        llvm::errs() << "   ...  found MemoryPhi\n";
        return Current;
      }

      // Below, check if CurrentDef is a valid candidate to be eliminated by
      // KillingDef. If it is not, check the next candidate.
      MemoryDef *CurrentDef = cast<MemoryDef>(Current);
      Instruction *CurrentI = CurrentDef->getMemoryInst();

      if (canSkipDef(CurrentDef, !isInvisibleToCallerOnUnwind(KillingUndObj))) {
        CanOptimize = false;
        continue;
      }

      // Before we try to remove anything, check for any extra throwing
      // instructions that block us from DSEing
      if (mayThrowBetween(KillingI, CurrentI, KillingUndObj)) {
        llvm::errs() << "  ... skip, may throw!\n";
        return std::nullopt;
      }

      // Check for anything that looks like it will be a barrier to further
      // removal
      if (isDSEBarrier(KillingUndObj, CurrentI)) {
        llvm::errs() << "  ... skip, barrier\n";
        return std::nullopt;
      }

      // If Current is known to be on path that reads DefLoc or is a read
      // clobber, bail out, as the path is not profitable. We skip this check
      // for intrinsic calls, because the code knows how to handle memcpy
      // intrinsics.
      if (!isa<IntrinsicInst>(CurrentI) && isReadClobber(KillingLoc, CurrentI))
        return std::nullopt;

      // Quick check if there are direct uses that are read-clobbers.
      if (any_of(Current->uses(), [this, &KillingLoc, StartAccess](Use &U) {
            if (auto *UseOrDef = dyn_cast<MemoryUseOrDef>(U.getUser()))
              return !MSSA.dominates(StartAccess, UseOrDef) &&
                     isReadClobber(KillingLoc, UseOrDef->getMemoryInst());
            return false;
          })) {
        llvm::errs() << "   ...  found a read clobber\n";
        return std::nullopt;
      }

      // If Current does not have an analyzable write location or is not
      // removable, skip it.
      CurrentLoc = getLocForWrite(CurrentI);
      if (!CurrentLoc || !isRemovable(CurrentI)) {
        CanOptimize = false;
        continue;
      }

      // AliasAnalysis does not account for loops. Limit elimination to
      // candidates for which we can guarantee they always store to the same
      // memory location and not located in different loops.
      if (!isGuaranteedLoopIndependent(CurrentI, KillingI, *CurrentLoc)) {
        llvm::errs() << "  ... not guaranteed loop independent\n";
        CanOptimize = false;
        continue;
      }

      if (IsMemTerm) {
        // If the killing def is a memory terminator (e.g. lifetime.end), check
        // the next candidate if the current Current does not write the same
        // underlying object as the terminator.
        if (!isMemTerminator(*CurrentLoc, CurrentI, KillingI)) {
          CanOptimize = false;
          continue;
        }
      } else {
        int64_t KillingOffset = 0;
        int64_t DeadOffset = 0;
        auto OR = isOverwrite(KillingI, CurrentI, KillingLoc, *CurrentLoc,
                              KillingOffset, DeadOffset);
        if (CanOptimize) {
          // CurrentDef is the earliest write clobber of KillingDef. Use it as
          // optimized access. Do not optimize if CurrentDef is already the
          // defining access of KillingDef.
          if (CurrentDef != KillingDef->getDefiningAccess() &&
              (OR == OW_Complete || OR == OW_MaybePartial))
            KillingDef->setOptimized(CurrentDef);

          // Once a may-aliasing def is encountered do not set an optimized
          // access.
          if (OR != OW_None)
            CanOptimize = false;
        }

        // If Current does not write to the same object as KillingDef, check
        // the next candidate.
        if (OR == OW_Unknown || OR == OW_None)
          continue;
        else if (OR == OW_MaybePartial) {
          // If KillingDef only partially overwrites Current, check the next
          // candidate if the partial step limit is exceeded. This aggressively
          // limits the number of candidates for partial store elimination,
          // which are less likely to be removable in the end.
          if (PartialLimit <= 1) {
            WalkerStepLimit -= 1;
            llvm::errs() << "   ... reached partial limit ... continue "
                            "with next access\n";
            continue;
          }
          PartialLimit -= 1;
        }
      }
      break;
    };

    // Accesses to objects accessible after the function returns can only be
    // eliminated if the access is dead along all paths to the exit. Collect
    // the blocks with killing (=completely overwriting MemoryDefs) and check if
    // they cover all paths from MaybeDeadAccess to any function exit.
    SmallPtrSet<Instruction *, 16> KillingDefs;
    KillingDefs.insert(KillingDef->getMemoryInst());
    MemoryAccess *MaybeDeadAccess = Current;
    MemoryLocation MaybeDeadLoc = *CurrentLoc;
    Instruction *MaybeDeadI = cast<MemoryDef>(MaybeDeadAccess)->getMemoryInst();
    llvm::errs() << "  Checking for reads of " << *MaybeDeadAccess << " ("
                 << *MaybeDeadI << ")\n";

    SmallSetVector<MemoryAccess *, 32> WorkList;
    auto PushMemUses = [&WorkList](MemoryAccess *Acc) {
      for (Use &U : Acc->uses())
        WorkList.insert(cast<MemoryAccess>(U.getUser()));
    };
    PushMemUses(MaybeDeadAccess);

    // Check if DeadDef may be read.
    for (unsigned I = 0; I < WorkList.size(); I++) {
      MemoryAccess *UseAccess = WorkList[I];

      llvm::errs() << "   " << *UseAccess;
      // Bail out if the number of accesses to check exceeds the scan limit.
      if (ScanLimit < (WorkList.size() - I)) {
        llvm::errs() << "\n    ...  hit scan limit\n";
        return std::nullopt;
      }
      --ScanLimit;

      if (isa<MemoryPhi>(UseAccess)) {
        if (any_of(KillingDefs, [this, UseAccess](Instruction *KI) {
              return DT.properlyDominates(KI->getParent(),
                                          UseAccess->getBlock());
            })) {
          llvm::errs() << " ... skipping, dominated by killing block\n";
          continue;
        }
        llvm::errs() << "\n    ... adding PHI uses\n";
        PushMemUses(UseAccess);
        continue;
      }

      Instruction *UseInst = cast<MemoryUseOrDef>(UseAccess)->getMemoryInst();
      llvm::errs() << " (" << *UseInst << ")\n";

      if (any_of(KillingDefs, [this, UseInst](Instruction *KI) {
            return DT.dominates(KI, UseInst);
          })) {
        llvm::errs() << " ... skipping, dominated by killing def\n";
        continue;
      }

      // A memory terminator kills all preceeding MemoryDefs and all succeeding
      // MemoryAccesses. We do not have to check it's users.
      if (isMemTerminator(MaybeDeadLoc, MaybeDeadI, UseInst)) {
        llvm::errs()
            << " ... skipping, memterminator invalidates following accesses\n";
        continue;
      }

      if (isNoopIntrinsic(cast<MemoryUseOrDef>(UseAccess)->getMemoryInst())) {
        llvm::errs() << "    ... adding uses of intrinsic\n";
        PushMemUses(UseAccess);
        continue;
      }

      if (UseInst->mayThrow() && !isInvisibleToCallerOnUnwind(KillingUndObj)) {
        llvm::errs() << "  ... found throwing instruction\n";
        return std::nullopt;
      }

      // Uses which may read the original MemoryDef mean we cannot eliminate the
      // original MD. Stop walk.
      if (isReadClobber(MaybeDeadLoc, UseInst)) {
        llvm::errs() << "    ... found read clobber\n";
        return std::nullopt;
      }

      // If this worklist walks back to the original memory access (and the
      // pointer is not guarenteed loop invariant) then we cannot assume that a
      // store kills itself.
      if (MaybeDeadAccess == UseAccess &&
          !isGuaranteedLoopInvariant(MaybeDeadLoc.Ptr)) {
        llvm::errs() << "    ... found not loop invariant self access\n";
        return std::nullopt;
      }
      // Otherwise, for the KillingDef and MaybeDeadAccess we only have to check
      // if it reads the memory location.
      // TODO: It would probably be better to check for self-reads before
      // calling the function.
      if (KillingDef == UseAccess || MaybeDeadAccess == UseAccess) {
        llvm::errs() << "    ... skipping killing def/dom access\n";
        continue;
      }

      // Check all uses for MemoryDefs, except for defs completely overwriting
      // the original location. Otherwise we have to check uses of *all*
      // MemoryDefs we discover, including non-aliasing ones. Otherwise we might
      // miss cases like the following
      //   1 = Def(LoE) ; <----- DeadDef stores [0,1]
      //   2 = Def(1)   ; (2, 1) = NoAlias,   stores [2,3]
      //   Use(2)       ; MayAlias 2 *and* 1, loads [0, 3].
      //                  (The Use points to the *first* Def it may alias)
      //   3 = Def(1)   ; <---- Current  (3, 2) = NoAlias, (3,1) = MayAlias,
      //                  stores [0,1]
      if (MemoryDef *UseDef = dyn_cast<MemoryDef>(UseAccess)) {
        if (isCompleteOverwrite(MaybeDeadLoc, MaybeDeadI, UseInst)) {
          BasicBlock *MaybeKillingBlock = UseInst->getParent();
          if (PostOrderNumbers.find(MaybeKillingBlock)->second <
              PostOrderNumbers.find(MaybeDeadAccess->getBlock())->second) {
            if (!isInvisibleToCallerAfterRet(KillingUndObj)) {
              llvm::errs() << "    ... found killing def " << *UseInst << "\n";
              KillingDefs.insert(UseInst);
            }
          } else {
            llvm::errs() << "    ... found preceeding def " << *UseInst << "\n";
            return std::nullopt;
          }
        } else
          PushMemUses(UseDef);
      }
    }

    // For accesses to locations visible after the function returns, make sure
    // that the location is dead (=overwritten) along all paths from
    // MaybeDeadAccess to the exit.
    if (!isInvisibleToCallerAfterRet(KillingUndObj)) {
      SmallPtrSet<BasicBlock *, 16> KillingBlocks;
      for (Instruction *KD : KillingDefs)
        KillingBlocks.insert(KD->getParent());
      assert(!KillingBlocks.empty() &&
             "Expected at least a single killing block");

      // Find the common post-dominator of all killing blocks.
      BasicBlock *CommonPred = *KillingBlocks.begin();
      for (BasicBlock *BB : llvm::drop_begin(KillingBlocks)) {
        if (!CommonPred)
          break;
        CommonPred = PDT.findNearestCommonDominator(CommonPred, BB);
      }

      // If the common post-dominator does not post-dominate MaybeDeadAccess,
      // there is a path from MaybeDeadAccess to an exit not going through a
      // killing block.
      if (!PDT.dominates(CommonPred, MaybeDeadAccess->getBlock())) {
        if (!AnyUnreachableExit)
          return std::nullopt;

        // Fall back to CFG scan starting at all non-unreachable roots if not
        // all paths to the exit go through CommonPred.
        CommonPred = nullptr;
      }

      // If CommonPred itself is in the set of killing blocks, we're done.
      if (KillingBlocks.count(CommonPred))
        return {MaybeDeadAccess};

      SetVector<BasicBlock *> WorkList;
      // If CommonPred is null, there are multiple exits from the function.
      // They all have to be added to the worklist.
      if (CommonPred)
        WorkList.insert(CommonPred);
      else
        for (BasicBlock *R : PDT.roots()) {
          if (!isa<UnreachableInst>(R->getTerminator()))
            WorkList.insert(R);
        }

      // Check if all paths starting from an exit node go through one of the
      // killing blocks before reaching MaybeDeadAccess.
      for (unsigned I = 0; I < WorkList.size(); I++) {
        BasicBlock *Current = WorkList[I];
        if (KillingBlocks.count(Current))
          continue;
        if (Current == MaybeDeadAccess->getBlock())
          return std::nullopt;

        // MaybeDeadAccess is reachable from the entry, so we don't have to
        // explore unreachable blocks further.
        if (!DT.isReachableFromEntry(Current))
          continue;

        for (BasicBlock *Pred : predecessors(Current))
          WorkList.insert(Pred);

        if (WorkList.size() >= MemorySSAPathCheckLimit)
          return std::nullopt;
      }
    }

    // No aliasing MemoryUses of MaybeDeadAccess found, MaybeDeadAccess is
    // potentially dead.
    return {MaybeDeadAccess};
  }

  Function &F;
  AliasAnalysis &AA;
  BatchAAResults BatchAA;
  EarliestEscapeInfo EI;
  MemorySSA &MSSA;
  DominatorTree &DT;
  PostDominatorTree &PDT;
  const TargetLibraryInfo &TLI;
  const DataLayout &DL;
  const LoopInfo &LI;

  // Whether the function contains any irreducible control flow, useful for
  // being accurately able to detect loops.
  bool ContainsIrreducibleLoops;
  // All MemoryDefs that potentially could kill other MemDefs.
  SmallVector<MemoryDef *, 64> MemDefs;
  // Any that should be skipped as they are already deleted
  SmallPtrSet<MemoryAccess *, 4> SkipStores;
  // Keep track whether a given object is captured before return or not.
  DenseMap<const Value *, bool> CapturedBeforeReturn;

  // Keep track of all of the objects that are invisible to the caller after
  // the function returns.
  DenseMap<const Value *, bool> InvisibleToCallerAfterRet;
  // Keep track of blocks with throwing instructions not modeled in MemorySSA.
  SmallPtrSet<BasicBlock *, 16> ThrowingBlocks;
  // Post-order numbers for each basic block. Used to figure out if memory
  // accesses are executed before another access.
  DenseMap<BasicBlock *, unsigned> PostOrderNumbers;

  // Check if there are root nodes that are terminated by UnreachableInst.
  // Those roots pessimize post-dominance queries. If there are such roots,
  // fall back to CFG scan starting from all non-unreachable roots.
  bool AnyUnreachableExit;
};

class DSEPass : public PassInfoMixin<DSEPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
    auto &LI = AM.getResult<LoopAnalysis>(F);
    auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
    auto &AA = AM.getResult<AAManager>(F);
    auto &PDT = AM.getResult<PostDominatorTreeAnalysis>(F);
    auto &MSSA = AM.getResult<MemorySSAAnalysis>(F).getMSSA();
    auto &TLI = AM.getResult<TargetLibraryAnalysis>(F);

    DSEState State(F, AA, MSSA, DT, PDT, TLI, LI);

    for (unsigned I = 0; I < State.MemDefs.size(); I++) {
      MemoryDef *KillingDef = State.MemDefs[I];
      if (State.SkipStores.count(KillingDef))
        continue;

      Instruction *KillingI = KillingDef->getMemoryInst();
      if (!isa<CallInst>(KillingI))
        continue;

      auto *CI = cast<CallInst>(KillingI);
      if (!CI->getCalledFunction()->getName().starts_with(
              "__builtin_IB_subgroup"))
        continue;

      llvm::errs() << "KillingDef: " << *KillingDef << "\n";
      llvm::errs() << "KillingI: " << *KillingI << "\n";

      std::optional<MemoryLocation> MaybeKillingLoc;
      if (State.isMemTerminatorInst(KillingI)) {
        if (auto KillingLoc = State.getLocForTerminator(KillingI))
          MaybeKillingLoc = KillingLoc->first;
      } else {
        MaybeKillingLoc = State.getLocForWrite(KillingI);
      }

      if (!MaybeKillingLoc) {
        llvm::errs() << "Failed to find analyzable write location for "
                     << *KillingI << "\n";
        continue;
      }

      MemoryLocation KillingLoc = *MaybeKillingLoc;
      assert(KillingLoc.Ptr && "KillingLoc should not be null");
      const Value *KillingUndObj = getUnderlyingObject(KillingLoc.Ptr);
      llvm::errs() << "Trying to eliminate MemoryDefs killed by " << *KillingDef
                   << " (" << *KillingI << ")\n";

      unsigned ScanLimit = MemorySSAScanLimit;
      unsigned WalkerStepLimit = MemorySSAUpwardsStepLimit;
      unsigned PartialLimit = MemorySSAPartialStoreLimit;

      // Worklist of MemoryAccesses that may be killed by KillingDef.
      SmallSetVector<MemoryAccess *, 8> ToCheck;
      // Track MemoryAccesses that have been deleted in the loop below, so we
      // can skip them. Don't use SkipStores for this, which may contain reused
      // MemoryAccess addresses.
      SmallPtrSet<MemoryAccess *, 8> Deleted;
      [[maybe_unused]] unsigned OrigNumSkipStores = State.SkipStores.size();
      ToCheck.insert(KillingDef->getDefiningAccess());

      bool Shortend = false;
      bool IsMemTerm = State.isMemTerminatorInst(KillingI);
      // Check if MemoryAccesses in the worklist are killed by KillingDef.
      for (unsigned I = 0; I < ToCheck.size(); I++) {
        MemoryAccess *Current = ToCheck[I];
        if (Deleted.contains(Current))
          continue;

        std::optional<MemoryAccess *> MaybeDeadAccess = State.getDomMemoryDef(
            KillingDef, Current, KillingLoc, KillingUndObj, ScanLimit,
            WalkerStepLimit, IsMemTerm, PartialLimit);

        if (!MaybeDeadAccess) {
          llvm::errs() << "  finished walk\n";
          continue;
        }

        MemoryAccess *DeadAccess = *MaybeDeadAccess;
        llvm::errs() << " Checking if we can kill " << *DeadAccess;
      }
    }

    auto inSubLoop = [](BasicBlock *BB, Loop *L, LoopInfo &LI) {
      return LI.getLoopFor(BB) != L;
    };

    return PreservedAnalyses::all();
  }
};
} // namespace

void mlir::triton::intel::DSE(llvm::Module &mod) {
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
  FAM.registerPass([&] { return PostDominatorTreeAnalysis(); });
  FAM.registerPass([&] { return LoopAnalysis(); });
  FAM.registerPass([&] { return PassInstrumentationAnalysis(); });
  FAM.registerPass([&] { return TargetLibraryAnalysis(); });
  FAM.registerPass([&] { return TargetIRAnalysis(); });
  FAM.registerPass([&] { return BasicAA(); });
  FAM.registerPass([&] { return ScopedNoAliasAA(); });
  FAM.registerPass([&] { return TypeBasedAA(); });
  FAM.registerPass([&] { return MemorySSAAnalysis(); });
  FAM.registerPass([&] { return TargetLibraryAnalysis(); });
  FAM.registerPass([&] {
    AAManager AA;
    AA.registerFunctionAnalysis<BasicAA>();
    AA.registerFunctionAnalysis<ScopedNoAliasAA>();
    AA.registerFunctionAnalysis<TypeBasedAA>();
    return AA;
  });

  FunctionPassManager FPM;
  FPM.addPass(DSEPass());
  FPM.run(*kernel, FAM);
}
