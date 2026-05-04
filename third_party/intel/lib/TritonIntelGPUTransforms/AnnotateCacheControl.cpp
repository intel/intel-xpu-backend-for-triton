#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir::triton::gpu::intel {

#define GEN_PASS_DEF_TRITONINTELGPUANNOTATECACHECONTROL
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"

namespace {

// Propagates a use of `v` through structured control flow, inserting any
// downstream values that may transitively carry data from `v` into `worklist`.
// Returns true if the caller should skip the default fallback (i.e. the use
// was already handled here). Handles scf.for (init -> iter_arg, result),
// scf.if (yield -> result), and scf.while (init -> before-region arg,
// condition -> after-region arg + while result, after-region yield ->
// before-region arg).
static bool propagateThroughSCF(OpOperand &use,
                                llvm::SetVector<Value> &worklist) {
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

// Returns true when `.cg` annotation should be suppressed on `loadOp` because
// the HW access pattern implied by the load's result encoding already yields
// cross-subgroup L1 reuse. Replaces an older forward-dataflow heuristic
// (see PR #6723 review).
static bool skipForL1Reuse(tt::LoadOp loadOp) {
  auto tensorTy = dyn_cast<RankedTensorType>(loadOp.getType());
  if (!tensorTy)
    return false;
  Attribute enc = tensorTy.getEncoding();
  if (!enc)
    return false;

  // Unwrap SliceEncodingAttr chain (may be nested).
  while (auto sliceEnc = dyn_cast<ttg::SliceEncodingAttr>(enc))
    enc = sliceEnc.getParent();

  // DotOperandEncodingAttr whose parent is a DPAS/MMA-family encoding
  // (DpasEncodingAttr and Subgroup2DBlockEncodingAttr both implement
  // MmaEncodingTrait).
  if (auto dotEnc = dyn_cast<ttg::DotOperandEncodingAttr>(enc))
    if (isa<ttg::MmaEncodingTrait>(dotEnc.getParent()))
      return true;

  // Scale operand of tt.dot_scaled: scales are consumed by the same
  // DPAS-backed matmul as the A/B operands and benefit from the same
  // cross-subgroup L1 reuse, so `.cg` would defeat that. The
  // LinearEncodingAttr built by BlockScaledDPAStoLinearLayout has no
  // signature we can match by attribute, so use a use-site check instead.
  // A single-hop walk suffices because AccelerateMatmul materialises the
  // scale encoding immediately before the tt.dot_scaled consumer — no
  // scf.for / convert_layout sits between them at this pass position.
  for (OpOperand &use : loadOp.getResult().getUses()) {
    if (auto ds = dyn_cast<tt::DotScaledOp>(use.getOwner()))
      if (use.get() == ds.getAScale() || use.get() == ds.getBScale())
        return true;
  }

  return false;
}

// Tracks kernel-argument pointers where `.cg` annotation is suppressed to
// preserve L1 locality for loads that may benefit from L1 reuse.
struct FuncSkipInfo {
  llvm::DenseSet<BlockArgument> skipArgs;
  bool hasAtomic = false;
};

// Walks the pointer SSA chain backward and collects all entry-block
// BlockArguments of `func` that `ptr` may resolve to. Returns true iff every
// path in the SSA chain bottomed out at a known handled op. Returns false if
// any path hit an unknown producer — the caller should skip annotation.
static bool collectRoots(Value ptr, tt::FuncOp func,
                         llvm::SmallVectorImpl<BlockArgument> &roots,
                         llvm::DenseSet<Value> &visited) {
  if (!ptr)
    return false;
  if (!visited.insert(ptr).second)
    return true;

  Block &entryBlock = func.getBody().front();

  if (auto blockArg = dyn_cast<BlockArgument>(ptr)) {
    if (blockArg.getOwner() == &entryBlock) {
      roots.push_back(blockArg);
      return true;
    }
    // Region iter_arg of an scf.for: follow matching init + yield operands.
    if (auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp())) {
      unsigned argIdx = blockArg.getArgNumber();
      // argIdx 0 is the induction variable; iter_args start at 1.
      if (argIdx == 0)
        return false;
      unsigned iterIdx = argIdx - 1;
      if (iterIdx >= forOp.getInitArgs().size())
        return false;
      bool resolved =
          collectRoots(forOp.getInitArgs()[iterIdx], func, roots, visited);
      auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      if (iterIdx < yieldOp.getNumOperands())
        resolved &=
            collectRoots(yieldOp.getOperand(iterIdx), func, roots, visited);
      else
        resolved = false;
      return resolved;
    }
    return false;
  }

  Operation *defOp = ptr.getDefiningOp();
  if (!defOp)
    return false;

  // Pointer arithmetic / shape/layout reshuffling: follow operand 0.
  if (isa<tt::AddPtrOp, tt::SplatOp, tt::BroadcastOp, tt::BitcastOp,
          tt::ExpandDimsOp, tt::ReshapeOp, tt::TransOp, ttg::ConvertLayoutOp>(
          defOp)) {
    return collectRoots(defOp->getOperand(0), func, roots, visited);
  }

  // scf.for result: forward to matching init and yield operands.
  if (auto forOp = dyn_cast<scf::ForOp>(defOp)) {
    auto result = cast<OpResult>(ptr);
    unsigned iterIdx = result.getResultNumber();
    bool resolved = true;
    if (iterIdx < forOp.getInitArgs().size())
      resolved &=
          collectRoots(forOp.getInitArgs()[iterIdx], func, roots, visited);
    else
      resolved = false;
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    if (iterIdx < yieldOp.getNumOperands())
      resolved &=
          collectRoots(yieldOp.getOperand(iterIdx), func, roots, visited);
    else
      resolved = false;
    return resolved;
  }

  // scf.if result: follow both branches.
  if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
    auto result = cast<OpResult>(ptr);
    unsigned idx = result.getResultNumber();
    bool resolved = true;
    for (Region *region : {&ifOp.getThenRegion(), &ifOp.getElseRegion()}) {
      if (region->empty())
        continue;
      auto yieldOp = cast<scf::YieldOp>(region->front().getTerminator());
      if (idx < yieldOp.getNumOperands())
        resolved &= collectRoots(yieldOp.getOperand(idx), func, roots, visited);
      else
        resolved = false;
    }
    return resolved;
  }

  // Unknown producer: signal incomplete resolution to the caller.
  return false;
}

// Returns true if `.cg` annotation should be skipped for a load at `ptr`:
// (a) the pointer SSA chain could not be fully resolved to entry-block function
// arguments, OR (b) any resolved root is in `skipArgs`.
static bool shouldSkipLoad(Value ptr, tt::FuncOp func,
                           const llvm::DenseSet<BlockArgument> &skipArgs) {
  llvm::SmallVector<BlockArgument, 4> roots;
  llvm::DenseSet<Value> visited;
  bool resolved = collectRoots(ptr, func, roots, visited);
  if (!resolved)
    return true;
  for (BlockArgument arg : roots) {
    if (skipArgs.contains(arg))
      return true;
  }
  return false;
}

// Returns true if any transitive forward user of `root` is a `tt.store` whose
// destination pointer resolves to an arg in `skipArgs`. This catches the
// pattern where the loaded value is reduced into a read-write accumulator
// buffer even though the load's own pointer is read-only.
// Walks through scf.for/scf.if/scf.while.
static bool
valueFlowsToSkippedStore(Value root, tt::FuncOp func,
                         const llvm::DenseSet<BlockArgument> &skipArgs) {
  if (skipArgs.empty())
    return false;

  llvm::SetVector<Value> worklist;
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
            shouldSkipLoad(storeOp.getPtr(), func, skipArgs))
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

// Scans `func` once to determine which entry-block pointer arguments should
// have `.cg` annotation suppressed. `.cg` (L1UC_L3C) on loads is always
// coherence-safe because it reads from the shared L3 rather than the
// incoherent per-core L1. These filters preserve L1 locality for loads that
// may benefit from L1 reuse (e.g. matmul operand loads in complex kernels):
//   - read-write args (loaded AND stored within the same function), OR
//   - if the function contains any atomic op, every pointer-typed entry-block
//     arg is excluded (atomic kernels often mix matmul operand loads with
//     streaming loads; excluding all args avoids regressing matmul operand
//     L1 reuse), OR
//   - if any store's pointer could not be resolved to known roots (so we
//     cannot tell which args are actually written), treat every pointer-typed
//     entry-block arg as potentially read-write.
static FuncSkipInfo computeSkipArgs(tt::FuncOp func) {
  FuncSkipInfo info;
  llvm::DenseSet<BlockArgument> loadedArgs;
  llvm::DenseSet<BlockArgument> storedArgs;
  bool hasUnresolvedStore = false;

  auto collectFor = [&](Value ptr,
                        llvm::DenseSet<BlockArgument> &sink) -> bool {
    llvm::SmallVector<BlockArgument, 4> roots;
    llvm::DenseSet<Value> visited;
    bool resolved = collectRoots(ptr, func, roots, visited);
    for (BlockArgument arg : roots)
      sink.insert(arg);
    return resolved;
  };

  func.walk([&](Operation *op) {
    if (auto loadOp = dyn_cast<tt::LoadOp>(op))
      collectFor(loadOp.getPtr(), loadedArgs);
    else if (auto storeOp = dyn_cast<tt::StoreOp>(op)) {
      if (!collectFor(storeOp.getPtr(), storedArgs))
        hasUnresolvedStore = true;
    } else if (isa<tt::AtomicRMWOp, tt::AtomicCASOp>(op))
      info.hasAtomic = true;
  });

  // Read-write args are always excluded.
  for (BlockArgument arg : loadedArgs) {
    if (storedArgs.contains(arg))
      info.skipArgs.insert(arg);
  }
  // If the function has any atomic, exclude every pointer-typed entry-block
  // arg — atomic kernels often mix matmul operand loads with streaming loads
  // and we cannot distinguish them; excluding all args preserves L1 reuse.
  // Likewise, if any store has an unresolved pointer, we can't prove which
  // args are written; treat every pointer arg as potentially RW.
  if (info.hasAtomic || hasUnresolvedStore) {
    for (BlockArgument arg : func.getBody().front().getArguments()) {
      Type argTy = arg.getType();
      if (isa<tt::PointerType>(argTy))
        info.skipArgs.insert(arg);
      else if (auto tensorTy = dyn_cast<RankedTensorType>(argTy)) {
        if (isa<tt::PointerType>(tensorTy.getElementType()))
          info.skipArgs.insert(arg);
      }
    }
  }

  return info;
}

struct AnnotateCacheControlPass
    : public impl::TritonIntelGPUAnnotateCacheControlBase<
          AnnotateCacheControlPass> {
  using TritonIntelGPUAnnotateCacheControlBase::
      TritonIntelGPUAnnotateCacheControlBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    bool changed = false;

    moduleOp.walk([&](tt::FuncOp func) {
      if (func.getBody().empty())
        return;

      FuncSkipInfo info = computeSkipArgs(func);

      func.walk([&](tt::LoadOp loadOp) {
        if (loadOp.getCache() != tt::CacheModifier::NONE)
          return;
        if (!isa<RankedTensorType>(loadOp.getType()))
          return;
        if (skipForL1Reuse(loadOp))
          return;
        if (shouldSkipLoad(loadOp.getPtr(), func, info.skipArgs))
          return;
        if (valueFlowsToSkippedStore(loadOp.getResult(), func, info.skipArgs))
          return;
        loadOp.setCacheAttr(tt::CacheModifierAttr::get(loadOp.getContext(),
                                                       tt::CacheModifier::CG));
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
};

} // namespace
} // namespace mlir::triton::gpu::intel
