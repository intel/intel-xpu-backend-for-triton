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
      worklist.insert(ifOp.thenYield().getOperand(resultIdx));
      if (!ifOp.getElseRegion().empty())
        worklist.insert(ifOp.elseYield().getOperand(resultIdx));
      continue;
    }

    if (readsMemory(def)) {
      if (isa<RankedTensorType>(v.getType()))
        return true;
      continue; // Scalar read: uniform across the tile.
    }

    for (Value operand : def->getOperands())
      worklist.insert(operand);
  }
  return false;
}

} // namespace mlir::triton::gpu::intel
