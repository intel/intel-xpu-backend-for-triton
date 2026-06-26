//===- RegisterPressure.cpp - Register Pressure Analysis -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "intel/include/Analysis/RegisterPressure.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir::triton::gpu::intel {

RegisterPressureAnalysis::RegisterPressureAnalysis(Operation *op,
                                                   RegisterPressureOptions opts)
    : liveness(op), options(opts) {}

unsigned RegisterPressureAnalysis::getPerThreadSizeInBytes(Type type) {
  // Handle RankedTensorType with distributed encoding
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    Type elType = tensorType.getElementType();
    if (!elType.isIntOrFloat())
      return 0;
    unsigned elemsPerThread = gpu::getTotalElemsPerThread(tensorType);
    // Round up to whole bytes so sub-byte types (fp8/fp4/i1) are not counted
    // as zero pressure, which would systematically under-count FP8 kernels.
    unsigned bytesPerElem = (elType.getIntOrFloatBitWidth() + 7) / 8;
    return elemsPerThread * bytesPerElem;
  }

  // Handle scalar int/float types
  if (type.isIntOrFloat())
    return (type.getIntOrFloatBitWidth() + 7) / 8;

  // All other types contribute zero pressure
  return 0;
}

unsigned RegisterPressureAnalysis::getGRFBytesPerThread(StringRef grfMode) {
  // Explicit GRF modes map to exact per-thread budgets.
  // For "default" and "auto", conservatively assume 128-register mode (4096
  // bytes) to avoid exceeding hardware limits when the compiler ultimately
  // chooses a smaller configuration.
  return llvm::StringSwitch<unsigned>(grfMode)
      .Case("128", 4096)
      .Case("256", 8192)
      .Case("512", 16384)
      .Default(4096);
}

bool RegisterPressureAnalysis::isRematerializable(Value value) const {
  Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return false; // Block arguments are not rematerializable

  // Check for constant-like operations that are cheap to regenerate.
  if (isa<arith::ConstantOp>(defOp))
    return true;

  // make_range is always cheap to regenerate (no inputs).
  if (isa<triton::MakeRangeOp>(defOp))
    return true;

  // A splat is only free to rematerialize if its scalar source is itself
  // rematerializable. A splat of a loop-variant scalar is NOT free: it would
  // require the scalar to be live (or recomputed) at the point of use.
  if (auto splatOp = dyn_cast<triton::SplatOp>(defOp))
    return isRematerializable(splatOp.getSrc());

  // Check for constant splat patterns using MLIR pattern matchers.
  Attribute constVal;
  if (matchPattern(defOp, m_Constant(&constVal)))
    return true;

  return false;
}

unsigned RegisterPressureAnalysis::pressureAt(Operation *op) const {
  unsigned pressure = 0;

  // Query the base mlir::Liveness block info directly (rather than the Intel
  // LivenessAnalysis wrapper, which asserts the op is a direct child of the
  // analysis root). This works for ops nested in any region, e.g. an scf.for
  // body. currentlyLiveValues() takes an expansive view: a value defined by or
  // consumed by `op` is counted, so a value is counted at its defining op and
  // loop-carried iter args (block live-in) are included.
  const LivenessBlockInfo *blockInfo = liveness.getLiveness(op->getBlock());
  if (!blockInfo)
    return 0;
  Liveness::ValueSetT liveValues = blockInfo->currentlyLiveValues(op);

  for (Value liveVal : liveValues) {
    // Skip rematerializable values if the option is enabled
    if (options.excludeRematerializable && isRematerializable(liveVal))
      continue;

    // Accumulate the per-thread size in bytes
    pressure += getPerThreadSizeInBytes(liveVal.getType());
  }

  return pressure;
}

unsigned RegisterPressureAnalysis::peakPressure(Block *block) const {
  unsigned peak = 0;

  // Iterate over all operations in the block and track the maximum pressure
  for (Operation &op : block->getOperations()) {
    unsigned pressure = pressureAt(&op);
    peak = std::max(peak, pressure);
  }

  return peak;
}

unsigned
RegisterPressureAnalysis::peakPressure(LoopLikeOpInterface loop) const {
  unsigned peak = 0;

  // A loop may expose multiple body regions, each with multiple blocks; check
  // all blocks across all loop regions.
  for (Region *region : loop.getLoopRegions()) {
    if (!region)
      continue;
    for (Block &block : *region) {
      unsigned blockPeak = peakPressure(&block);
      peak = std::max(peak, blockPeak);
    }
  }

  return peak;
}

void RegisterPressureAnalysis::print(raw_ostream &os) const {
  Operation *rootOp = liveness.getOperation();
  if (!rootOp)
    return;

  os << "Register Pressure Analysis (per-thread bytes):\n";

  // Walk all regions and blocks to report peak pressure. Qualify each block by
  // its parent op name so blocks in different regions (all named ^bb0) are
  // distinguishable in the output.
  rootOp->walk([&](Block *block) {
    unsigned peak = peakPressure(block);
    os << "  Block ";
    block->printAsOperand(os);
    os << " in " << block->getParentOp()->getName() << ": peak = " << peak
       << " bytes\n";
  });
}

} // namespace mlir::triton::gpu::intel
