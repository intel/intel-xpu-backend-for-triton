#include "triton/Analysis/Membar.h"
#include "triton/Analysis/Alias.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include <deque>

namespace mlir {

void MembarAnalysis::run(FuncBlockInfoMapT &funcBlockInfoMap) {
  FunctionOpInterface funcOp =
      dyn_cast<FunctionOpInterface>(allocation->getOperation());
  OpBuilder builder(funcOp.getContext());
  resolve(funcOp, &funcBlockInfoMap, &builder);
}

void MembarAnalysis::resolve(FunctionOpInterface funcOp,
                             FuncBlockInfoMapT *funcBlockInfoMap,
                             OpBuilder *builder) {
  // Initialize the blockList
  DenseMap<Block *, BlockInfo> inputBlockInfoMap;
  DenseMap<Block *, BlockInfo> outputBlockInfoMap;
  std::deque<Block *> blockList;
  funcOp.walk<WalkOrder::PreOrder>([&](Block *block) {
    for (auto &op : block->getOperations()) {
      // Check if the operation belongs to scf dialect, if so, we need to
      // throw an error
      if (op.getDialect()->getNamespace() == "scf") {
        llvm::report_fatal_error(
            "scf dialect is not supported in membar. Please lower it "
            "to cf dialect first.");
        return;
      }
    }
    if (block->isEntryBlock())
      blockList.emplace_back(block);
  });

  // A fixed point algorithm
  while (!blockList.empty()) {
    auto *block = blockList.front();
    blockList.pop_front();
    // Make a copy of the inputblockInfo but not update
    auto inputBlockInfo = inputBlockInfoMap[block];
    SmallVector<Block *> successors;
    for (auto &op : block->getOperations()) {
      if (op.hasTrait<OpTrait::IsTerminator>()) {
        visitTerminator(&op, successors);
      } else {
        update(&op, &inputBlockInfo, funcBlockInfoMap, builder);
      }
    }
    // Get the reference because we want to update if it changed
    if (outputBlockInfoMap.count(block) &&
        inputBlockInfo == outputBlockInfoMap[block]) {
      // If we have seen the block before and the inputBlockInfo is the same as
      // the outputBlockInfo, we skip the successors
      continue;
    }
    // Update the current block
    outputBlockInfoMap[block].join(inputBlockInfo);
    // Update the successors
    for (auto *successor : successors) {
      inputBlockInfoMap[successor].join(outputBlockInfoMap[block]);
      blockList.emplace_back(successor);
    }
  }

  // Update the final dangling buffers that haven't been synced
  auto &funcBlockInfo = (*funcBlockInfoMap)[funcOp];
  funcOp.walk<WalkOrder::PreOrder>([&](Block *block) {
    block->walk([&](triton::ReturnOp returnOp) {
      funcBlockInfo.join(outputBlockInfoMap[block]);
    });
  });
}

void MembarAnalysis::visitTerminator(Operation *op,
                                     SmallVector<Block *> &successors) {
  if (auto branchInterface = dyn_cast<BranchOpInterface>(op)) {
    Block *parentBlock = branchInterface->getBlock();
    successors.append(std::begin(parentBlock->getSuccessors()),
                      std::end(parentBlock->getSuccessors()));
    return;
  }
  // Otherwise, it could be a return op
  if (isa<triton::ReduceReturnOp, triton::ScanReturnOp, triton::ReturnOp>(op)) {
    return;
  }
  llvm_unreachable("Unknown terminator encountered in membar analysis");
}

void MembarAnalysis::insertBarrier(Operation *op, OpBuilder *builder) {
  OpBuilder::InsertionGuard g(*builder);
  auto barrierOp = builder->create<gpu::BarrierOp>(op->getLoc());
}

void MembarAnalysis::update(Operation *op, BlockInfo *blockInfo,
                            FuncBlockInfoMapT *funcBlockInfoMap,
                            OpBuilder *builder) {
  if (isa<triton::gpu::LocalDeallocOp, triton::gpu::MemDescSubviewOp,
          triton::TransOp>(op)) {
    return;
  }
  if (auto alloc = dyn_cast<triton::gpu::LocalAllocOp>(op)) {
    if (!alloc.getInit())
      return;
  }

  if (isa<gpu::BarrierOp>(op)) {
    // If the current op is a barrier, we sync previous reads and writes
    blockInfo->sync();
    return;
  }

  if (isa<triton::gpu::AsyncWaitOp, triton::gpu::AsyncBulkWaitOp>(op) &&
      !isa<gpu::BarrierOp>(op->getNextNode())) {
    // If the current op is an async wait and the next op is not a barrier we
    // insert a barrier op and sync
    builder->setInsertionPointAfter(op);
    insertBarrier(op, builder);
    blockInfo->sync();
    return;
  }

  BlockInfo curBlockInfo;
  if (isa<triton::CallOp>(op)) {
    // Inter-function dependencies
    auto callOpInterface = dyn_cast<CallOpInterface>(op);
    if (auto callee =
            dyn_cast<FunctionOpInterface>(callOpInterface.resolveCallable())) {
      curBlockInfo = funcBlockInfoMap->lookup(callee);
    }
  } else {
    // Intra-function dependencies
    for (Value value : op->getOperands()) {
      for (auto bufferId : allocation->getBufferIds(value)) {
        if (bufferId != Allocation::InvalidBufferId) {
          if (isa<triton::gpu::AsyncCopyGlobalToLocalOp>(op)) {
            // Global -> shared memory
            curBlockInfo.syncWriteIntervals.insert(
                allocation->getAllocatedInterval(bufferId));
          } else {
            // ConvertLayoutOp: shared memory -> registers
            curBlockInfo.syncReadIntervals.insert(
                allocation->getAllocatedInterval(bufferId));
          }
        }
      }
    }
    for (Value value : op->getResults()) {
      // ConvertLayoutOp: registers -> shared memory
      auto bufferId = allocation->getBufferId(value);
      if (bufferId != Allocation::InvalidBufferId) {
        curBlockInfo.syncWriteIntervals.insert(
            allocation->getAllocatedInterval(bufferId));
      }
    }
    // Scratch buffer is considered as both shared memory write & read
    auto bufferId = allocation->getBufferId(op);
    if (bufferId != Allocation::InvalidBufferId) {
      curBlockInfo.syncWriteIntervals.insert(
          allocation->getAllocatedInterval(bufferId));
      curBlockInfo.syncReadIntervals.insert(
          allocation->getAllocatedInterval(bufferId));
    }
  }

  if (blockInfo->isIntersected(curBlockInfo)) {
    builder->setInsertionPoint(op);
    insertBarrier(op, builder);
    blockInfo->sync();
  }
  // Update the region info, even if barrier is inserted, we have to maintain
  // the current op's read/write buffers.
  blockInfo->join(curBlockInfo);
}
} // namespace mlir
