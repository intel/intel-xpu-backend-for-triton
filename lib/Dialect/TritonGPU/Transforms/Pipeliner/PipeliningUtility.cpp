#include "PipeliningUtility.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

// Combine the current mask with the given predicate.
static Value getPredMask(RewriterBase &rewriter, Type typeLike,
                         Value currentMask, Value pred) {
  Type maskType = tt::getI1SameShape(typeLike);
  Location loc = pred.getLoc();
  Value mask = pred;
  if (maskType.isa<RankedTensorType>()) {
    mask = rewriter.create<tt::SplatOp>(loc, maskType, pred);
  }
  if (currentMask) {
    mask = rewriter.create<arith::AndIOp>(loc, mask, currentMask);
  }
  return mask;
}

// Function to mask operations during scheduling.
Operation *mlir::triton::predicateOp(RewriterBase &rewriter, Operation *op,
                                     Value pred) {
  OpBuilder::InsertionGuard guard(rewriter);
  if (mlir::isMemoryEffectFree(op))
    return op;
  if (isa<ttg::AsyncCommitGroupOp>(op))
    return op;
  if (isa<ttg::AsyncWaitOp>(op))
    return op;
  if (auto insertOp = dyn_cast<ttg::InsertSliceAsyncOp>(op)) {
    rewriter.setInsertionPoint(insertOp);
    Value mask = getPredMask(rewriter, insertOp.getSrc().getType(),
                             insertOp.getMask(), pred);
    insertOp.getMaskMutable().assign(mask);
    return op;
  }
  if (auto insertOp = dyn_cast<ttng::InsertSliceTMAOp>(op)) {
    rewriter.setInsertionPoint(insertOp);
    Value mask = getPredMask(
        rewriter,
        insertOp.getSrc().getType().cast<tt::PointerType>().getPointeeType(),
        insertOp.getMask(), pred);
    insertOp.getMaskMutable().assign(mask);
    return op;
  }
  if (auto arriveOp = dyn_cast<ttng::MBarrierArriveOp>(op)) {
    rewriter.setInsertionPoint(arriveOp);
    Value mask = getPredMask(rewriter, rewriter.getIntegerType(1),
                             arriveOp.getPred(), pred);
    arriveOp.getPredMutable().assign(mask);
    return op;
  }
  if (isa<ttng::MBarrierWaitOp>(op)) {
    return op;
  }
  if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
    rewriter.setInsertionPoint(loadOp);
    Value mask = getPredMask(rewriter, loadOp.getPtr().getType(),
                             loadOp.getMask(), pred);
    loadOp.getMaskMutable().assign(mask);
    return op;
  }

  assert("don't know how to predicate this op" && false);
  return op;
}

/// Helper to recursively add dependencies to the same stage.
void mlir::triton::addDep(Operation *op, DenseSet<Operation *> &deps,
                          bool includeArg, DenseSet<Operation *> *filter) {
  if (filter && filter->count(op))
    return;
  if (!deps.insert(op).second)
    return;
  for (Value operand : op->getOperands()) {
    Value v = operand;
    llvm::SmallDenseSet<Value> seen;
    while (auto arg = v.dyn_cast<BlockArgument>()) {
      if (!includeArg)
        break;
      if (!seen.insert(v).second)
        break;
      if (arg.getArgNumber() > 0 && arg.getOwner() == op->getBlock()) {
        auto yieldOp = op->getBlock()->getTerminator();
        v = yieldOp->getOperand(arg.getArgNumber() - 1);
        continue;
      }
      break;
    }
    Operation *defOp = v.getDefiningOp();
    if (defOp && defOp->getBlock() == op->getBlock()) {
      addDep(defOp, deps, includeArg, filter);
    }
  }
}

// Add operations to the schedule with the given stage based on the filter
// function.
void mlir::triton::addOps(
    scf::ForOp forOp, int stage,
    std::vector<std::pair<Operation *, unsigned>> &schedule,
    std::function<bool(Operation *)> filter) {
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (!filter(&op))
      continue;
    schedule.emplace_back(&op, stage);
  }
}
