#include "intel/include/Analysis/Liveness.h"
#include "mlir/IR/Operation.h"

using namespace mlir;

namespace mlir::triton::gpu::intel {

raw_ostream &operator<<(raw_ostream &OS, const LiveInterval &LI) {
  OpPrintingFlags flags;
  flags.skipRegions();

  auto printOp = [&](Operation *op) {
    switch (op->getNumResults()) {
    case 0:
      OS << op->getName();
      break;
    default:
      llvm::interleaveComma(op->getResults(), OS,
                            [&](Value res) { res.printAsOperand(OS, flags); });
    }
  };

  OS << "[";
  printOp(LI.start);
  OS << ", ";
  printOp(LI.end);
  OS << "] for value: ";
  LI.liveValue.printAsOperand(OS, flags);
  return OS;
}

void LivenessAnalysis::printLiveIntervals(raw_ostream &OS) const {
  for (const auto &[block, LIs] : blockToLiveIntervals) {
    OS << "LiveIntervals for block: ";
    block->printAsOperand(OS);
    OS << "\n";
    for (const LiveInterval &LI : LIs)
      OS.indent(2) << LI << "\n";
  }
}

LivenessAnalysis::LivenessAnalysis(Operation *op) : mlir::Liveness(op) {
  assert(getOperation() && !getOperation()->getRegions().empty() &&
         "root operation should not be null and should contain a region");

  for (Region &rgn : getOperation()->getRegions())
    for (Block &block : rgn)
      computeLiveIntervals(block);
}

unsigned LivenessAnalysis::numOverlappingLiveIntervals(
    Operation *op, const LivenessBlockInfo &livenessInfo) const {
  assert(livenessInfo.getBlock() == op->getBlock() &&
         "liveness info must be for the block containing the operation");
  return livenessInfo.currentlyLiveValues(op).size();
}

unsigned LivenessAnalysis::maxNumOverlappingLiveIntervals(Block *block) const {
  assert(block && (block->getParentOp() == getOperation()) &&
         "block must be contained by the root operation");

  const LivenessBlockInfo *livenessInfo = getLiveness(block);
  if (!livenessInfo)
    return 0;

  unsigned max = 0;
  for (Operation &op : block->getOperations()) {
    unsigned num = numOverlappingLiveIntervals(&op, *livenessInfo);
    max = (max > num) ? max : num;
  }
  return max;
}

Liveness::ValueSetT LivenessAnalysis::getLiveValues(Operation *op) const {
  assert(op && (op->getParentOp() == getOperation()) &&
         "operation must be contained by the root operation");

  ValueSetT liveSet;
  if (Block *block = op->getBlock()) {
    if (const LivenessBlockInfo *livenessInfo = getLiveness(block))
      liveSet = livenessInfo->currentlyLiveValues(op);
  }
  return liveSet;
}

void LivenessAnalysis::computeLiveIntervals(Block &block) {
  assert(block.getParentOp() == getOperation() &&
         "block must be contained by the root operation");

  const LivenessBlockInfo *livenessInfo = getLiveness(&block);
  if (!livenessInfo)
    return;

  // First compute the live intervals for all liveIn values into the block.
  for (Value liveVal : getLiveIn(&block)) {
    Operation *startOp = livenessInfo->getStartOperation(liveVal);
    Operation *endOp = livenessInfo->getEndOperation(liveVal, startOp);
    LiveInterval liveInterval(startOp, endOp, liveVal);
    blockToLiveIntervals[&block].push_back(liveInterval);
  }

  // Then compute the live intervals for all value defined by operations in the
  // block.
  for (Operation &op : block.getOperations()) {
    for (OpResult res : op.getResults()) {
      Operation *startOp = &op;
      Operation *endOp = livenessInfo->getEndOperation(res, startOp);
      LiveInterval liveInterval(startOp, endOp, res);
      blockToLiveIntervals[&block].push_back(liveInterval);
    }
  }
}

} // namespace mlir::triton::gpu::intel
