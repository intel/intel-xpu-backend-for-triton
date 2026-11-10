#include "intel/include/Analysis/ReuseAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::triton::gpu::intel {

/// Returns true iff `op` reads memory, per `MemoryEffectOpInterface`.
static bool readsMemory(Operation *op) {
  auto effectOp = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effectOp)
    return false;
  SmallVector<MemoryEffects::EffectInstance> effects;
  effectOp.getEffects(effects);
  return llvm::any_of(effects, [](const MemoryEffects::EffectInstance &e) {
    return isa<MemoryEffects::Read>(e.getEffect());
  });
}

bool addressDependsOnLoadedTensor(Value ptr) {
  SetVector<Value> worklist;
  worklist.insert(ptr);
  for (unsigned i = 0; i < worklist.size(); ++i) {
    Value v = worklist[i];

    // A scalar is uniform across the tile whatever produced it (a scalar
    // load, or a tensor read reduced to a scalar), so it cannot make the
    // address vary per lane. Only tensor values carry a gather.
    if (!isa<RankedTensorType>(v.getType()))
      continue;

    if (auto arg = dyn_cast<BlockArgument>(v)) {
      // Function arguments and induction variables are structured; only
      // loop-carried values need to be followed to their init and yield.
      if (auto forOp = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp())) {
        if (OpOperand *init = forOp.getTiedLoopInit(arg)) {
          worklist.insert(init->get());
          worklist.insert(forOp.getTiedLoopYieldedValue(arg)->get());
        }
      }
      continue;
    }

    Operation *def = v.getDefiningOp();
    unsigned resultIdx = cast<OpResult>(v).getResultNumber();
    if (auto forOp = dyn_cast<scf::ForOp>(def)) {
      worklist.insert(forOp.getInitArgs()[resultIdx]);
      worklist.insert(forOp.getYieldedValues()[resultIdx]);
      continue;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(def)) {
      // An scf.if with results always has an else region (verifier-enforced).
      worklist.insert(ifOp.thenYield().getOperand(resultIdx));
      worklist.insert(ifOp.elseYield().getOperand(resultIdx));
      continue;
    }

    if (readsMemory(def))
      return true;

    for (Value operand : def->getOperands())
      worklist.insert(operand);
  }
  return false;
}

} // namespace mlir::triton::gpu::intel
