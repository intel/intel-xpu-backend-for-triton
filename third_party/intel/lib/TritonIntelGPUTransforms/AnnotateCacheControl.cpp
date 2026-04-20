#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/SetVector.h"

namespace tt = mlir::triton;

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
        // iter_arg (so subsequent iterations see it).
        if (auto forOp = dyn_cast<scf::ForOp>(yieldOp->getParentOp())) {
          unsigned idx = use.getOperandNumber();
          worklist.insert(forOp.getResult(idx));
          worklist.insert(forOp.getRegionIterArg(idx));
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

// Returns true if `root` is (transitively) produced by a tt::DotOp or
// tt::DotScaledOp. Walks backward through defining ops and scf.for.
static bool producedByDot(Value root) {
  llvm::SetVector<Value> worklist;
  worklist.insert(root);

  for (unsigned i = 0; i < worklist.size(); ++i) {
    Value v = worklist[i];

    if (auto blockArg = dyn_cast<BlockArgument>(v)) {
      // scf.for iter_arg: producer is either the init operand (first
      // iteration) or the yield operand (subsequent iterations).
      if (auto forOp =
              dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp())) {
        unsigned argIdx = blockArg.getArgNumber();
        if (argIdx >= forOp.getNumInductionVars()) {
          unsigned iterIdx = argIdx - forOp.getNumInductionVars();
          worklist.insert(forOp.getInitArgs()[iterIdx]);
          auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
          worklist.insert(yieldOp.getOperand(iterIdx));
        }
      }
      continue;
    }

    Operation *def = v.getDefiningOp();
    if (!def)
      continue;
    if (isa<tt::DotOp, tt::DotScaledOp>(def))
      return true;

    if (auto forOp = dyn_cast<scf::ForOp>(def)) {
      // Backward through loop result -> yield operand.
      unsigned resultIdx = cast<OpResult>(v).getResultNumber();
      auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      worklist.insert(yieldOp.getOperand(resultIdx));
      worklist.insert(forOp.getInitArgs()[resultIdx]);
      continue;
    }

    // Generic backward propagation through all operands.
    for (Value operand : def->getOperands())
      worklist.insert(operand);
  }
  return false;
}

struct AnnotateCacheControlPass
    : public impl::TritonIntelGPUAnnotateCacheControlBase<
          AnnotateCacheControlPass> {
  using TritonIntelGPUAnnotateCacheControlBase::
      TritonIntelGPUAnnotateCacheControlBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    bool changed = false;

    moduleOp.walk([&](tt::LoadOp loadOp) {
      if (loadOp.getCache() != tt::CacheModifier::NONE)
        return;
      if (!isa<RankedTensorType>(loadOp.getType()))
        return;
      if (flowsToDot(loadOp.getResult()))
        return;
      loadOp.setCacheAttr(tt::CacheModifierAttr::get(loadOp.getContext(),
                                                     tt::CacheModifier::CG));
      changed = true;
    });

    moduleOp.walk([&](tt::StoreOp storeOp) {
      if (storeOp.getCache() != tt::CacheModifier::NONE)
        return;
      if (!isa<RankedTensorType>(storeOp.getValue().getType()))
        return;
      if (producedByDot(storeOp.getValue()))
        return;
      storeOp.setCacheAttr(tt::CacheModifierAttr::get(storeOp.getContext(),
                                                      tt::CacheModifier::CG));
      changed = true;
    });

    if (!changed)
      markAllAnalysesPreserved();
  }
};

} // namespace
} // namespace mlir::triton::gpu::intel
