#include "intel/include/Analysis/AliasAnalysis.h"
#include "intel/include/Analysis/AxisInfoExt.h"
#include "intel/include/Analysis/ReuseAnalysis.h"
#include "intel/include/Analysis/StrideInfo.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include <type_traits>

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace tti = mlir::triton::intel;

namespace mlir::triton::gpu::intel {

#define GEN_PASS_DEF_TRITONINTELGPUANNOTATECACHECONTROL
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"

namespace {

using AliasKind = tti::AliasAnalysis::PointerRootKind;
using RootsResult = tti::AliasAnalysis::PointerRootsResult;

//===----------------------------------------------------------------------===//
// Tunable budget knobs for EVICT_LAST promotion. Exposed as `cl::opt` so
// measurement can refine the defaults without recompiling. The per-load knob
// caps a single load's tile bytes; the per-loop knob caps the running total
// across all promoted loads in the enclosing loop, bounding total L1 pressure.
//===----------------------------------------------------------------------===//

static llvm::cl::opt<int64_t> kEvictLastPerLoadBudgetBytes(
    "tritonintelgpu-evict-last-per-load-bytes",
    llvm::cl::desc("Per-load tile-byte budget for promoting tt.load to "
                   "EVICT_LAST in AnnotateCacheControl"),
    llvm::cl::init(32 * 1024));

static llvm::cl::opt<int64_t> kEvictLastPerLoopBudgetBytes(
    "tritonintelgpu-evict-last-per-loop-bytes",
    llvm::cl::desc("Per-loop running-total tile-byte budget for EVICT_LAST "
                   "promotion in AnnotateCacheControl"),
    llvm::cl::init(48 * 1024));

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
///
/// Cache-fill prefetches (`ttig.prefetch`, `ttig.descriptor_prefetch`) are
/// excluded: they declare `MemWrite<L2Cache>` to keep the optimizer from
/// CSE/DCE-ing them, but they do not mutate observable memory. Treating them
/// as writing peers would block evict_last promotion of any load on the same
/// pointer (the canonical Pipeline-pass shape: pre-prefetch + in-loop
/// prefetch + load on the same iter-arg).
static bool hasWriteEffect(Operation *op) {
  if (isa<tt::StoreOp, tt::AtomicRMWOp, tt::AtomicCASOp, tt::DescriptorStoreOp,
          tt::DescriptorScatterOp, tt::DescriptorReduceOp>(op))
    return true;
  if (isa<ttg::intel::PrefetchOp, ttg::intel::DescriptorPrefetchOp>(op))
    return false;
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
/// correct for both. Constrained to the load op types that
/// `AliasAnalysis::getAliasingMemOps` accepts.
template <typename OpTy, typename = std::enable_if_t<llvm::is_one_of<
                             OpTy, tt::LoadOp, tt::DescriptorLoadOp>::value>>
static bool aliasesWritingPeer(OpTy load, const tti::AliasAnalysis &alias) {
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
// EVICT_LAST policy helpers: dot-feed predicate and per-loop budget tracker.
//===----------------------------------------------------------------------===//

/// True iff every transitive user of `v` reaches a `tt::DotOp` or
/// `tt::DotScaledOp` operand through layout-only intermediates.
///
/// Layout-only intermediate ops accepted: `ttg::ConvertLayoutOp`,
/// `tt::BroadcastOp`, `tt::ExpandDimsOp`, `tt::TransOp`, `tt::ReshapeOp`.
/// Multi-consumer intermediates where any consumer is non-layout-only reject
/// the chain.
///
/// Loop-carried block arguments / `scf::YieldOp` carries are followed only for
/// `scf::ForOp` (the canonical pipelined-load shape). `scf::IfOp` /
/// `scf::WhileOp` carries are intentionally not supported in this predicate.
///
/// Reject (any of these aborts the predicate as a leaf):
///   - any `arith::*` op (including `arith::SelectOp`),
///   - `tt::ReduceOp`, `tt::StoreOp`, `func::ReturnOp`,
///   - any other terminator,
///   - any op not in the layout-only set and not a dot consumer.
///
/// Bounded DFS depth = 8 to keep the predicate cheap.
static bool feedsDotOperand(Value v) {
  constexpr unsigned kMaxDepth = 8;

  SetVector<Value> visited;
  // Worklist of (value, depth) pairs.
  SmallVector<std::pair<Value, unsigned>> worklist;
  worklist.push_back({v, 0});
  visited.insert(v);

  while (!worklist.empty()) {
    auto [cur, depth] = worklist.pop_back_val();
    if (depth > kMaxDepth)
      return false;

    // The chain must reach at least one user; an unused load value is not a
    // dot operand.
    if (cur.use_empty())
      return false;

    for (OpOperand &use : cur.getUses()) {
      Operation *user = use.getOwner();

      // Leaf accept: dot or dot_scaled operand.
      if (isa<tt::DotOp, tt::DotScaledOp>(user))
        continue;

      // Layout-only intermediates: must have a single user (otherwise some
      // sibling consumer might pull the value into a streaming pattern).
      if (isa<ttg::ConvertLayoutOp, tt::BroadcastOp, tt::ExpandDimsOp,
              tt::TransOp, tt::ReshapeOp>(user)) {
        if (!user->hasOneUse())
          return false;
        for (Value res : user->getResults())
          if (visited.insert(res))
            worklist.push_back({res, depth + 1});
        continue;
      }

      // scf.for loop-carried argument: follow the carry through the matching
      // region iter-arg AND the matching for-result.
      if (auto forOp = dyn_cast<scf::ForOp>(user)) {
        unsigned operandIdx = use.getOperandNumber();
        unsigned numCtrl = forOp.getNumControlOperands();
        if (operandIdx < numCtrl)
          return false; // Control operand (lb/ub/step) — not a data carry.
        unsigned iterIdx = operandIdx - numCtrl;
        Value iterArg = forOp.getRegionIterArg(iterIdx);
        Value forRes = forOp.getResult(iterIdx);
        if (visited.insert(iterArg))
          worklist.push_back({iterArg, depth + 1});
        if (visited.insert(forRes))
          worklist.push_back({forRes, depth + 1});
        continue;
      }

      // scf.yield inside an scf.for: carry the value to the for-result + the
      // next iteration's iter-arg.
      if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
        auto forOp = dyn_cast<scf::ForOp>(yieldOp->getParentOp());
        if (!forOp)
          return false; // Only scf.for carries are supported here.
        unsigned idx = use.getOperandNumber();
        Value forRes = forOp.getResult(idx);
        Value iterArg = forOp.getRegionIterArg(idx);
        if (visited.insert(forRes))
          worklist.push_back({forRes, depth + 1});
        if (visited.insert(iterArg))
          worklist.push_back({iterArg, depth + 1});
        continue;
      }

      // Anything else: reject the whole chain.
      return false;
    }
  }

  return true;
}

/// Returns the total tile size in bytes for a load-like op's result value, or
/// 0 if the result is not a ranked tensor or its element type's bit-width is
/// not a positive multiple of 8. Works for any op with a single tensor result
/// (`tt.load`, `tt.descriptor_load`, ...).
static int64_t computeTileBytes(Value result) {
  auto ty = dyn_cast<RankedTensorType>(result.getType());
  if (!ty)
    return 0;
  unsigned bits = ty.getElementTypeBitWidth();
  if (bits == 0 || bits % 8 != 0)
    return 0;
  int64_t numElements = ty.getNumElements();
  if (numElements <= 0)
    return 0;
  return numElements * static_cast<int64_t>(bits / 8);
}

/// Per-function tracker for the EVICT_LAST per-loop byte budget. Implements
/// the canonical rule from the plan §2.1: a load is admitted iff its tile
/// bytes are within `kEvictLastPerLoadBudgetBytes` AND, if enclosed in an
/// `scf::ForOp`, the running total in that loop's bucket plus the candidate's
/// tile bytes does not exceed `kEvictLastPerLoopBudgetBytes`. Outside-loop
/// loads are subject to the per-load cap only and never touch any bucket.
struct EvictLastBudgetTracker {
  llvm::DenseMap<scf::ForOp, int64_t> bytesPromotedPerLoop;

  template <typename OpTy, typename = std::enable_if_t<llvm::is_one_of<
                               OpTy, tt::LoadOp, tt::DescriptorLoadOp>::value>>
  bool fits(OpTy load) const {
    int64_t tileBytes = computeTileBytes(load.getResult());
    if (tileBytes <= 0)
      return false;
    if (tileBytes > kEvictLastPerLoadBudgetBytes)
      return false;
    scf::ForOp loop = load->template getParentOfType<scf::ForOp>();
    if (!loop)
      return true; // Outside any loop: per-load cap only.
    int64_t already = bytesPromotedPerLoop.lookup(loop);
    return already + tileBytes <= kEvictLastPerLoopBudgetBytes;
  }

  template <typename OpTy, typename = std::enable_if_t<llvm::is_one_of<
                               OpTy, tt::LoadOp, tt::DescriptorLoadOp>::value>>
  void account(OpTy load) {
    scf::ForOp loop = load->template getParentOfType<scf::ForOp>();
    if (!loop)
      return; // Outside-loop loads do not consume any per-loop bucket.
    bytesPromotedPerLoop[loop] += computeTileBytes(load.getResult());
  }
};

//===----------------------------------------------------------------------===//
// Per-function context
//===----------------------------------------------------------------------===//

struct FuncContext {
  ReuseAnalysis reuse;
  std::unique_ptr<tti::AliasAnalysis> alias;
  bool hasAtomic;
  SetVector<BlockArgument> skipped;
  EvictLastBudgetTracker evictLastBudget;
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
          EvictLastBudgetTracker(),
      };

      func.walk([&](Operation *op) {
        llvm::TypeSwitch<Operation *>(op)
            .Case<tt::LoadOp, tt::DescriptorLoadOp>([&](auto load) {
              if (tryAnnotate(load, ctx))
                changed = true;
            });
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
  /// Four-requirement gate for promoting a load to EVICT_LAST. All four
  /// must hold:
  ///   1. Spatial known cross-subgroup reuse (drops temporal-only).
  ///   2. Warp-broadcast factor >= 2: at least 2 warps share the same
  ///      address. Without this, a DPAS dot operand with `warpsPerCTA`
  ///      that does not tile its non-K axis (factor == 1) would be
  ///      promoted purely on the strength of K being warp-invariant —
  ///      a degenerate "reuse" with no real cross-warp sharing.
  ///   3. The load reaches a `tt.dot` / `tt.dot_scaled` operand through
  ///      layout-only ops only.
  ///   4. The per-load + per-loop byte budget admits this load.
  /// `knownReuse(load)` is checked separately by the caller in `tryAnnotate`
  /// — this helper layers on the additional policy filters.
  template <typename OpTy, typename = std::enable_if_t<llvm::is_one_of<
                               OpTy, tt::LoadOp, tt::DescriptorLoadOp>::value>>
  static bool shouldUseEvictLast(OpTy load, FuncContext &ctx) {
    // Requirement 1: spatial known reuse only — drop temporal-only.
    if (!ctx.reuse.getSpatial().knownCrossSubgroupReuse(load))
      return false;
    // Requirement 2: non-trivial warp-broadcast factor. A factor of 1 means
    // warps strictly partition the tensor, so no inter-warp reuse exists
    // even though the layout has a warp-invariant axis.
    std::optional<unsigned> factor =
        ctx.reuse.getSpatial().knownWarpBroadcastFactor(load);
    if (!factor || *factor < 2)
      return false;
    // Requirement 3: load value reaches a tt.dot or tt.dot_scaled operand,
    // possibly through layout-only ops.
    if (!feedsDotOperand(load.getResult()))
      return false;
    // Requirement 4: per-load + per-loop byte budget.
    if (!ctx.evictLastBudget.fits(load))
      return false;
    return true;
  }

  /// Returns true when the load should be skipped because its pointer roots
  /// fail the atomic/unresolved-origin disqualifiers.
  ///
  /// `tt.load` exposes a pointer operand: in atomic kernels we drop loads
  /// rooted in entry-block pointer args, and we drop loads whose roots can't
  /// be resolved.
  ///
  /// `tt.descriptor_load` has no pointer operand to inspect, so the check
  /// degrades to "skip on atomic". The equivalent of root-based filtering is
  /// already provided by `ctx.skipped` (see `computeSkippedArgs`), which
  /// excludes every pointer-typed entry-block arg in atomic / unresolved-store
  /// cases.
  template <typename OpTy, typename = std::enable_if_t<llvm::is_one_of<
                               OpTy, tt::LoadOp, tt::DescriptorLoadOp>::value>>
  static bool failsPointerArgGates(OpTy load, FuncContext &ctx) {
    if constexpr (std::is_same_v<OpTy, tt::LoadOp>) {
      if (ctx.hasAtomic && isPointerArgRooted(load.getPtr(), *ctx.alias))
        return true;
      return ptrRootsUnknown(load.getPtr(), *ctx.alias);
    }
    return ctx.hasAtomic;
  }

  /// Default-branch policy for the no-reuse fall-through. `tt.load` is tagged
  /// with `.cg` to bypass L1; `tt.descriptor_load` is left alone because the
  /// 2D-block-I/O lowering for descriptor loads (see `LowerTo2DBlockLoad`)
  /// drops the cache modifier when rewriting to `ttig.2d_block_load`, so the
  /// `.cg` annotation would never reach the hardware.
  template <typename OpTy, typename = std::enable_if_t<llvm::is_one_of<
                               OpTy, tt::LoadOp, tt::DescriptorLoadOp>::value>>
  static bool applyNoReuseDefault(OpTy load) {
    if constexpr (std::is_same_v<OpTy, tt::LoadOp>) {
      load.setCacheAttr(
          tt::CacheModifierAttr::get(load.getContext(), tt::CacheModifier::CG));
      return true;
    }
    return false;
  }

  /// Annotate a load (`tt.load` or `tt.descriptor_load`) with the appropriate
  /// cache control. The op-kind-specific behavior is isolated in
  /// `failsPointerArgGates` (atomic / unresolved-pointer-roots check) and
  /// `applyNoReuseDefault` (no-reuse fall-through); everything else is
  /// identical between the two op kinds.
  ///
  /// The structural disqualifiers (atomic / unresolved roots, aliasing-write
  /// peer, forward flow into a skipped store) BLOCK BOTH `.cg` AND
  /// `EVICT_LAST` — a load that flows through atomic context, has unknown
  /// root pointers, aliases a writing peer, or feeds a skipped store should
  /// remain at the default cache policy regardless of reuse signal. We
  /// evaluate them BEFORE the reuse decision so the same suppression covers
  /// both annotations.
  ///
  /// The reuse-driven decision splits as follows:
  ///   - encoded type AND any reuse signal -> suppress `.cg`;
  ///   - additionally, when reuse is *known* AND `shouldUseEvictLast` accepts
  ///     the candidate, promote to `EVICT_LAST`;
  ///   - any other reuse-branch hit stays at the default policy.
  /// Mutual exclusion: only one of `setEvictAttr` or `setCacheAttr` is ever
  /// called per load — guaranteed by the early-return structure.
  template <typename OpTy, typename = std::enable_if_t<llvm::is_one_of<
                               OpTy, tt::LoadOp, tt::DescriptorLoadOp>::value>>
  static bool tryAnnotate(OpTy load, FuncContext &ctx) {
    // Frontend cache override — never overwrite an explicitly-set modifier.
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

    // Pointer-rooted disqualifiers (op-kind-specific).
    if (failsPointerArgGates(load, ctx))
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

    // Alias analysis — suppress only on aliasing with a write peer.
    // Two read-only loads of the same arg are fine; both can keep `.cg`.
    if (aliasesWritingPeer(load, *ctx.alias))
      return false;

    // Forward data-flow into a store on an excluded arg.
    if (valueFlowsToSkippedStore(load.getResult(), *ctx.alias, ctx.skipped))
      return false;

    // Reuse-driven decision.
    if (loadTy.getEncoding() && ctx.reuse.anyReuse(load)) {
      if (ctx.reuse.knownReuse(load) && shouldUseEvictLast(load, ctx)) {
        load.setEvictAttr(tt::EvictionPolicyAttr::get(
            load.getContext(), tt::EvictionPolicy::EVICT_LAST));
        ctx.evictLastBudget.account(load);
        return true;
      }
      return false; // Reuse suspected but not promoted.
    }

    // No reuse evidence -> op-kind-specific default.
    return applyNoReuseDefault(load);
  }
};

} // namespace
} // namespace mlir::triton::gpu::intel
