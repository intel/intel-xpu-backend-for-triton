#include "intel/include/Analysis/AliasAnalysis.h"
#include "intel/include/Analysis/AxisInfoExt.h"
#include "intel/include/Analysis/ReuseAnalysis.h"
#include "intel/include/Analysis/StrideInfo.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

namespace tt = mlir::triton;
namespace tti = mlir::triton::intel;

namespace mlir::triton::gpu::intel {

#define GEN_PASS_DEF_TRITONINTELGPUANNOTATECACHECONTROL
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"

namespace {

using AliasKind = tti::AliasAnalysis::PointerRootKind;
using RootsResult = tti::AliasAnalysis::PointerRootsResult;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Propagates a use of `v` through structured control flow, inserting any
/// downstream values that may transitively carry data from `v` into `worklist`.
/// Returns true if the caller should skip the default fallback (i.e. the use
/// was already handled here). Handles scf.for (init -> iter_arg, result),
/// scf.if (yield -> result), and scf.while (init -> before-region arg,
/// condition -> after-region arg + while result, after-region yield ->
/// before-region arg).
static bool propagateThroughSCF(OpOperand &use, SetVector<Value> &worklist) {
  Operation *user = use.getOwner();

  if (auto forOp = dyn_cast<scf::ForOp>(user)) {
    unsigned operandIdx = use.getOperandNumber();
    unsigned numCtrl = forOp.getNumControlOperands();
    if (operandIdx >= numCtrl) {
      unsigned iterIdx = operandIdx - numCtrl;
      worklist.insert(forOp.getRegionIterArg(iterIdx));
      worklist.insert(forOp.getResult(iterIdx));
    }
    return true;
  }

  if (auto whileOp = dyn_cast<scf::WhileOp>(user)) {
    // scf.while operands are the init values for the before-region args.
    unsigned idx = use.getOperandNumber();
    if (idx < whileOp.getBeforeArguments().size())
      worklist.insert(whileOp.getBeforeArguments()[idx]);
    return true;
  }

  if (auto condOp = dyn_cast<scf::ConditionOp>(user)) {
    // scf.condition(cond, args...) — operand 0 is the i1 break condition;
    // operands 1..N map to after-region args and while results at 0..N-1.
    unsigned idx = use.getOperandNumber();
    if (idx == 0)
      return true;
    unsigned resultIdx = idx - 1;
    auto whileOp = cast<scf::WhileOp>(condOp->getParentOp());
    if (resultIdx < whileOp.getAfterArguments().size())
      worklist.insert(whileOp.getAfterArguments()[resultIdx]);
    if (resultIdx < whileOp.getResults().size())
      worklist.insert(whileOp.getResult(resultIdx));
    return true;
  }

  if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
    Operation *parent = yieldOp->getParentOp();
    unsigned idx = use.getOperandNumber();
    if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
      worklist.insert(forOp.getResult(idx));
      worklist.insert(forOp.getRegionIterArg(idx));
    } else if (auto ifOp = dyn_cast<scf::IfOp>(parent)) {
      worklist.insert(ifOp.getResult(idx));
    } else if (auto whileOp = dyn_cast<scf::WhileOp>(parent)) {
      // Yield is the terminator of the after region; it feeds the next
      // iteration's before-region arguments.
      if (idx < whileOp.getBeforeArguments().size())
        worklist.insert(whileOp.getBeforeArguments()[idx]);
    }
    return true;
  }

  return false;
}

/// True if `arg` is an entry-block block-argument of some FunctionOpInterface.
static bool isEntryBlockFuncArg(Value v) {
  auto arg = dyn_cast<BlockArgument>(v);
  if (!arg)
    return false;
  Block *owner = arg.getOwner();
  if (!owner)
    return false;
  auto funcOp = dyn_cast_or_null<FunctionOpInterface>(owner->getParentOp());
  if (!funcOp)
    return false;
  Region &body = funcOp.getFunctionBody();
  return !body.empty() && owner == &body.front();
}

/// Returns true iff `ptr`'s origin is opaque (Unknown) or was never seeded
/// by the alias analysis (NotTracked). Both cases are conservatively treated
/// as "origin unresolved" by callers — we cannot prove the pointer is safe
/// for L1 bypass in the presence of an unknown root.
static bool ptrRootsUnknown(Value ptr, const tti::AliasAnalysis &alias) {
  return alias.getPointerRoots(ptr).kind != AliasKind::Known;
}

/// Returns true iff every root of `ptr` is an entry-block function argument.
/// Used by the atomic-policy gate. `Unknown` / `NotTracked` return false —
/// the caller (`ptrRootsUnknown` gate) handles those separately.
static bool isPointerArgRooted(Value ptr, const tti::AliasAnalysis &alias) {
  RootsResult result = alias.getPointerRoots(ptr);
  if (result.kind != AliasKind::Known || result.roots.empty())
    return false;
  return llvm::all_of(result.roots, isEntryBlockFuncArg);
}

/// Returns true iff `op` is a write-effect memory op with respect to the
/// cache-control policy: any `tt.store`, `tt.atomic_*`, their descriptor
/// counterparts, or any op implementing `MemoryEffectOpInterface` with a
/// `MemoryEffects::Write` effect.
static bool hasWriteEffect(Operation *op) {
  if (isa<tt::StoreOp, tt::AtomicRMWOp, tt::AtomicCASOp, tt::DescriptorStoreOp,
          tt::DescriptorScatterOp, tt::DescriptorReduceOp>(op))
    return true;
  if (auto effectOp = dyn_cast<MemoryEffectOpInterface>(op)) {
    SmallVector<MemoryEffects::EffectInstance> effects;
    effectOp.getEffects(effects);
    for (const auto &effect : effects)
      if (isa<MemoryEffects::Write>(effect.getEffect()))
        return true;
  }
  return false;
}

/// Returns true iff `load` has at least one aliasing peer with a write effect.
/// Read-only peers (e.g. a second `tt.load` of the same arg) are not a reason
/// to suppress `.cg`: see AliasAnalysisTest.cpp TwoLoadsSameArgDifferentOffsets
/// — read-only loads from the same arg alias each other and `.cg` is still
/// correct for both.
static bool aliasesWritingPeer(tt::LoadOp load,
                               const tti::AliasAnalysis &alias) {
  for (Operation *peer : alias.getAliasingMemOps(load))
    if (hasWriteEffect(peer))
      return true;
  return false;
}

/// Returns true iff the function contains any atomic memory op.
static bool funcContainsAtomic(tt::FuncOp func) {
  auto result = func.walk([&](Operation *op) {
    if (isa<tt::AtomicRMWOp, tt::AtomicCASOp>(op))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return result.wasInterrupted();
}

/// Computes the set of entry-block pointer arguments on which `.cg` annotation
/// must be suppressed to preserve L1 locality:
///   - read-write args (loaded AND stored within the same function), OR
///   - if any store's pointer origin is unresolved, all pointer args, OR
///   - if the function contains atomics, all pointer args.
static SetVector<BlockArgument>
computeSkippedArgs(tt::FuncOp func, const tti::AliasAnalysis &alias,
                   bool hasAtomic) {
  SetVector<BlockArgument> skipped;
  DenseSet<BlockArgument> loadedArgs;
  DenseSet<BlockArgument> storedArgs;
  bool hasUnresolvedStore = false;

  auto collectRootArgs = [&](Value ptr, DenseSet<BlockArgument> &sink) -> bool {
    RootsResult result = alias.getPointerRoots(ptr);
    if (result.kind != AliasKind::Known)
      return false;
    for (Value r : result.roots)
      if (auto arg = dyn_cast<BlockArgument>(r);
          arg && isEntryBlockFuncArg(arg))
        sink.insert(arg);
    return true;
  };

  func.walk([&](Operation *op) {
    if (auto loadOp = dyn_cast<tt::LoadOp>(op))
      collectRootArgs(loadOp.getPtr(), loadedArgs);
    else if (auto storeOp = dyn_cast<tt::StoreOp>(op)) {
      if (!collectRootArgs(storeOp.getPtr(), storedArgs))
        hasUnresolvedStore = true;
    }
  });

  // Read-write args are always excluded.
  for (BlockArgument arg : loadedArgs)
    if (storedArgs.contains(arg))
      skipped.insert(arg);

  // Atomic kernels and unresolved-store functions: exclude every pointer-typed
  // entry-block arg. The rationale matches the original pass:
  //   - atomic kernels often mix matmul operand loads with streaming loads
  //     and we cannot distinguish them;
  //   - unresolved stores mean we cannot prove which args are written.
  if (hasAtomic || hasUnresolvedStore) {
    for (BlockArgument arg : func.getBody().front().getArguments()) {
      Type argTy = arg.getType();
      if (isa<tt::PointerType>(argTy)) {
        skipped.insert(arg);
      } else if (auto tensorTy = dyn_cast<RankedTensorType>(argTy)) {
        if (isa<tt::PointerType>(tensorTy.getElementType()))
          skipped.insert(arg);
      }
    }
  }

  return skipped;
}

/// Returns true iff the store's pointer origin is unknown OR any of its
/// roots intersects the `skipped` arg set.
static bool storePtrIsSkipped(Value storePtr, const tti::AliasAnalysis &alias,
                              const SetVector<BlockArgument> &skipped) {
  RootsResult result = alias.getPointerRoots(storePtr);
  if (result.kind != AliasKind::Known)
    return true;
  for (Value r : result.roots)
    if (auto arg = dyn_cast<BlockArgument>(r); arg && skipped.contains(arg))
      return true;
  return false;
}

/// Returns true iff any transitive forward user of `root` is a `tt.store`
/// whose destination pointer root-intersects `skipped` (or is unknown). This
/// catches the pattern where the loaded value is reduced into a read-write
/// accumulator buffer even though the load's own pointer is read-only.
/// Walks through scf.for/scf.if/scf.while via `propagateThroughSCF`.
///
/// This filter is retained because the alias analysis (P3) cannot replace it:
/// the filter tracks data flow, not pointer aliasing. See
/// `annotate-cc-rewrite.md` §"What stays and why" — tests m, p, r have the
/// pattern `load %X; ...; store %DW, derived_from_load` where `%X` and `%DW`
/// are distinct args with disjoint root sets.
static bool valueFlowsToSkippedStore(Value root,
                                     const tti::AliasAnalysis &alias,
                                     const SetVector<BlockArgument> &skipped) {
  if (skipped.empty())
    return false;

  SetVector<Value> worklist;
  worklist.insert(root);

  for (unsigned i = 0; i < worklist.size(); ++i) {
    Value v = worklist[i];
    for (OpOperand &use : v.getUses()) {
      Operation *user = use.getOwner();

      if (auto storeOp = dyn_cast<tt::StoreOp>(user)) {
        // Only the value operand reaching a store matters; if the use is the
        // pointer or mask operand, it's irrelevant to where the loaded *data*
        // ends up.
        if (use.get() == storeOp.getValue() &&
            storePtrIsSkipped(storeOp.getPtr(), alias, skipped))
          return true;
        continue;
      }

      if (propagateThroughSCF(use, worklist))
        continue;

      // Generic forward propagation: any value produced by this op may carry
      // data from `v`. Safe overapproximation.
      for (Value res : user->getResults())
        worklist.insert(res);
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Per-function context
//===----------------------------------------------------------------------===//

struct FuncContext {
  ReuseAnalysis reuse;
  std::unique_ptr<tti::AliasAnalysis> alias;
  bool hasAtomic;
  SetVector<BlockArgument> skipped;
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct AnnotateCacheControlPass
    : public impl::TritonIntelGPUAnnotateCacheControlBase<
          AnnotateCacheControlPass> {
  using TritonIntelGPUAnnotateCacheControlBase::
      TritonIntelGPUAnnotateCacheControlBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    // ModuleStrideAnalysis requires a ModuleAxisInfoAnalysis& (StrideInfo.h).
    tti::ModuleAxisInfoAnalysis axisInfo(moduleOp);
    tti::ModuleStrideAnalysis strideAnalysis(moduleOp, axisInfo);
    bool changed = false;

    moduleOp.walk([&](tt::FuncOp func) {
      if (func.getBody().empty())
        return;

      auto alias = std::make_unique<tti::AliasAnalysis>(func);
      bool hasAtomic = funcContainsAtomic(func);
      SetVector<BlockArgument> skipped =
          computeSkippedArgs(func, *alias, hasAtomic);
      FuncContext ctx{
          ReuseAnalysis(moduleOp, strideAnalysis),
          std::move(alias),
          hasAtomic,
          std::move(skipped),
      };

      func.walk([&](tt::LoadOp load) {
        if (tryAnnotate(load, ctx))
          changed = true;
      });
    });

    // Note: stores are intentionally NOT annotated. Unlike loads, store `.cg`
    // (L1-uncached) on cross-kernel producer/consumer buffers (e.g. layer-norm
    // backward partial-sum scratch) introduces data races — this is a true
    // coherence concern. Load `.cg` is always coherence-safe; the load filters
    // above are cost-model heuristics to preserve L1 locality.

    if (!changed)
      markAllAnalysesPreserved();
  }

private:
  static bool tryAnnotate(tt::LoadOp load, FuncContext &ctx) {
    // Gate 1: user override — never overwrite an explicitly-set cache modifier.
    if (load.getCache() != tt::CacheModifier::NONE)
      return false;

    // Gate 1b: user-specified eviction policy (e.g. evict_first/evict_last
    // from inductor) is honored at lowering time via LSC cache modes — do
    // not override it here.
    if (load.getEvict() != tt::EvictionPolicy::NORMAL)
      return false;

    // Gate 2: scalar loads don't get encoding-based annotation.
    auto loadTy = dyn_cast<RankedTensorType>(load.getType());
    if (!loadTy)
      return false;

    // Gate 3: reuse analysis (P1 OR P2). Only consult when an encoding is
    // present. Without an encoding, SpatialReuseAnalysis is conservative
    // (reuse = true), which would incorrectly suppress `.cg` on bare
    // tensor<Nx!tt.ptr<T>> loads (test f).
    if (loadTy.getEncoding() && ctx.reuse.anyReuse(load))
      return false;

    // Gate 3b: lane-broadcast suppression.
    //
    // SpatialReuseAnalysis::hasCrossSubgroupReuse only counts cross-WARP
    // broadcast. When a SliceEncodingAttr's parent layout placed lane
    // basis vectors along the sliced-out axis, those bases become all-zero
    // on the surviving out-dim — every lane in the warp issues the same
    // address. Such loads are not streaming; bypassing L1 with `.cg` for
    // them produces redundant DRAM/L3 traffic and, on dual-tile PVC,
    // forces cross-tile coherence on every reload.
    //
    // Suppress only when the layout structurally proves lane-broadcast
    // (factor >= 2). std::nullopt (any fallback case) means "no proof"
    // → fall through to subsequent gates.
    if (loadTy.getEncoding()) {
      std::optional<unsigned> laneFactor =
          ctx.reuse.getSpatial().knownLaneBroadcastFactor(load);
      if (laneFactor && *laneFactor >= 2)
        return false;
    }

    // Gate 4: atomic policy — in atomic kernels, any load whose pointer roots
    // to an entry-block pointer arg is suppressed (cost-model policy — see
    // the pass description in Passes.td).
    if (ctx.hasAtomic && isPointerArgRooted(load.getPtr(), *ctx.alias))
      return false;

    // Gate 5: unresolved pointer origin — conservative skip.
    if (ptrRootsUnknown(load.getPtr(), *ctx.alias))
      return false;

    // Gate 6: alias analysis — suppress only on aliasing with a write peer.
    // Two read-only loads of the same arg are fine; both can keep `.cg`.
    if (aliasesWritingPeer(load, *ctx.alias))
      return false;

    // Gate 7: forward data-flow into a store on an excluded arg. Retained
    // verbatim — alias analysis cannot replace it (see §"What stays and why"
    // in annotate-cc-rewrite.md).
    if (valueFlowsToSkippedStore(load.getResult(), *ctx.alias, ctx.skipped))
      return false;

    load.setCacheAttr(
        tt::CacheModifierAttr::get(load.getContext(), tt::CacheModifier::CG));
    return true;
  }
};

} // namespace
} // namespace mlir::triton::gpu::intel
