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

// Returns true if any transitive (forward) user of `root` is a tt::DotOp or
// tt::DotScaledOp. Walks through scf.for: init -> iter_arg -> yield -> result.
static bool flowsToDot(Value root) {
  llvm::SetVector<Value> worklist;
  worklist.insert(root);

  for (unsigned i = 0; i < worklist.size(); ++i) {
    Value v = worklist[i];
    for (OpOperand &use : v.getUses()) {
      Operation *user = use.getOwner();
      if (isa<tt::DotOp, tt::DotScaledOp>(user))
        return true;

      if (auto forOp = dyn_cast<scf::ForOp>(user)) {
        // Map init operand -> region iter_arg and corresponding loop result.
        unsigned operandIdx = use.getOperandNumber();
        unsigned numCtrl = forOp.getNumControlOperands();
        if (operandIdx >= numCtrl) {
          unsigned iterIdx = operandIdx - numCtrl;
          worklist.insert(forOp.getRegionIterArg(iterIdx));
          worklist.insert(forOp.getResult(iterIdx));
        }
        continue;
      }

      if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
        // yield inside scf.for: forward to the matching loop result and
        // iter_arg. yield inside scf.if: forward to the matching if result.
        Operation *parent = yieldOp->getParentOp();
        unsigned idx = use.getOperandNumber();
        if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
          worklist.insert(forOp.getResult(idx));
          worklist.insert(forOp.getRegionIterArg(idx));
        } else if (auto ifOp = dyn_cast<scf::IfOp>(parent)) {
          worklist.insert(ifOp.getResult(idx));
        }
        continue;
      }

      // Generic forward propagation: any value produced by this op may carry
      // data from `v`. Safe overapproximation.
      for (Value res : user->getResults())
        worklist.insert(res);
    }
  }
  return false;
}

// Tracks kernel-argument pointers that must not be annotated `.cg` to preserve
// cross-workgroup coherency.
struct FuncUnsafeInfo {
  llvm::DenseSet<BlockArgument> unsafeArgs;
  bool hasAtomic = false;
};

// Walks the pointer SSA chain backward and collects all entry-block
// BlockArguments of `func` that `ptr` may resolve to. Returns true iff every
// path in the SSA chain bottomed out at a known handled op. Returns false if
// any path hit an unknown producer — the caller must treat that as unsafe.
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

// A load is unsafe to annotate `.cg` if (a) its pointer SSA chain could not be
// fully resolved to entry-block function arguments, OR (b) any resolved root is
// in `unsafeArgs`. "Unknown producer => unsafe" is the conservative default.
static bool isUnsafePtr(Value ptr, tt::FuncOp func,
                        const llvm::DenseSet<BlockArgument> &unsafeArgs) {
  llvm::SmallVector<BlockArgument, 4> roots;
  llvm::DenseSet<Value> visited;
  bool resolved = collectRoots(ptr, func, roots, visited);
  if (!resolved)
    return true;
  for (BlockArgument arg : roots) {
    if (unsafeArgs.contains(arg))
      return true;
  }
  return false;
}

// Returns true if any transitive forward user of `root` is a `tt.store` whose
// destination pointer resolves to an arg in `unsafeArgs`. This catches the
// cross-kernel accumulation pattern where the loaded value is reduced into an
// unsafe accumulator buffer even though the load's own pointer is safe.
static bool
valueFlowsToUnsafeStore(Value root, tt::FuncOp func,
                        const llvm::DenseSet<BlockArgument> &unsafeArgs) {
  if (unsafeArgs.empty())
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
            isUnsafePtr(storeOp.getPtr(), func, unsafeArgs))
          return true;
        continue;
      }

      if (auto forOp = dyn_cast<scf::ForOp>(user)) {
        unsigned operandIdx = use.getOperandNumber();
        unsigned numCtrl = forOp.getNumControlOperands();
        if (operandIdx >= numCtrl) {
          unsigned iterIdx = operandIdx - numCtrl;
          worklist.insert(forOp.getRegionIterArg(iterIdx));
          worklist.insert(forOp.getResult(iterIdx));
        }
        continue;
      }

      if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
        Operation *parent = yieldOp->getParentOp();
        unsigned idx = use.getOperandNumber();
        if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
          worklist.insert(forOp.getResult(idx));
          worklist.insert(forOp.getRegionIterArg(idx));
        } else if (auto ifOp = dyn_cast<scf::IfOp>(parent)) {
          worklist.insert(ifOp.getResult(idx));
        }
        continue;
      }

      // Generic forward propagation: any value produced by this op may carry
      // data from `v`. Safe overapproximation.
      for (Value res : user->getResults())
        worklist.insert(res);
    }
  }
  return false;
}

// Scans `func` once to determine which entry-block pointer arguments are
// unsafe to annotate `.cg`:
//   - read-write args (loaded AND stored within the same function), OR
//   - if the function contains any atomic op, every loaded arg (atomics imply
//     cross-workgroup synchronized communication through the pointer args), OR
//   - if any store's pointer could not be resolved to known roots (so we
//     cannot tell which args are actually written), treat every loaded arg
//     as read-write.
static FuncUnsafeInfo computeUnsafeArgs(tt::FuncOp func) {
  FuncUnsafeInfo info;
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
    if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
      collectFor(loadOp.getPtr(), loadedArgs);
      return;
    }
    if (auto storeOp = dyn_cast<tt::StoreOp>(op)) {
      if (!collectFor(storeOp.getPtr(), storedArgs))
        hasUnresolvedStore = true;
      return;
    }
    if (isa<tt::AtomicRMWOp, tt::AtomicCASOp>(op)) {
      info.hasAtomic = true;
      return;
    }
  });

  // Read-write args are always unsafe.
  for (BlockArgument arg : loadedArgs) {
    if (storedArgs.contains(arg))
      info.unsafeArgs.insert(arg);
  }
  // If the function has any atomic, poison every pointer-typed entry-block
  // arg — the kernel is using cross-workgroup synchronization and the
  // synchronized buffer may be any pointer arg, not only ones we saw loaded.
  // Likewise, if any store has an unresolved pointer, we can't prove which
  // args are written; treat every pointer arg as potentially RW.
  if (info.hasAtomic || hasUnresolvedStore) {
    for (BlockArgument arg : func.getBody().front().getArguments()) {
      Type argTy = arg.getType();
      if (isa<tt::PointerType>(argTy))
        info.unsafeArgs.insert(arg);
      else if (auto tensorTy = dyn_cast<RankedTensorType>(argTy)) {
        if (isa<tt::PointerType>(tensorTy.getElementType()))
          info.unsafeArgs.insert(arg);
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

      FuncUnsafeInfo info = computeUnsafeArgs(func);

      func.walk([&](tt::LoadOp loadOp) {
        if (loadOp.getCache() != tt::CacheModifier::NONE)
          return;
        if (!isa<RankedTensorType>(loadOp.getType()))
          return;
        if (flowsToDot(loadOp.getResult()))
          return;
        if (isUnsafePtr(loadOp.getPtr(), func, info.unsafeArgs))
          return;
        if (valueFlowsToUnsafeStore(loadOp.getResult(), func, info.unsafeArgs))
          return;
        loadOp.setCacheAttr(tt::CacheModifierAttr::get(loadOp.getContext(),
                                                       tt::CacheModifier::CG));
        changed = true;
      });
    });

    // Note: stores are intentionally NOT annotated. Annotating stores with .cg
    // (L1-uncached) on cross-kernel producer/consumer buffers (e.g. layer-norm
    // backward partial-sum scratch) introduces data races. Loads with .cg are
    // hints only and safe when the pointer does not alias a synchronized
    // cross-workgroup buffer (see computeUnsafeArgs).

    if (!changed)
      markAllAnalysesPreserved();
  }
};

} // namespace
} // namespace mlir::triton::gpu::intel
