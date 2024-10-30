//===- Utils.h - TritonIntelGPU Utils -----------------------------------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_DIALECT_TRITON_INTEL_GPU_IR_UTILS_H
#define TRITON_DIALECT_TRITON_INTEL_GPU_IR_UTILS_H

#include "intel/include/Analysis/AxisInfo.h"
#include "mlir/IR/Operation.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include <triton/Tools/Sys/GetEnv.hpp>

namespace mlir::triton::gpu::intel {

/// Calculate the optimal number of elements per thread for a given operation
/// along an axis with greatest continuity.
inline unsigned getNumElementsPerThread(
    Operation *op, SmallVector<unsigned> order,
    mlir::triton::intel::ModuleAxisInfoAnalysis &axisInfoAnalysis) {
  Value val = getMemAccessPtr(op);
  Type valTy = val.getType();
  auto ty =
      isTensorPointerType(valTy)
          ? cast<RankedTensorType>(cast<PointerType>(valTy).getPointeeType())
          : cast<RankedTensorType>(valTy);
  auto shapePerCTA = getShapePerCTA(ty);
  mlir::triton::AxisInfo &valInfo = *axisInfoAnalysis.getAxisInfo(val);

  unsigned elemNumBits = getElementBitWidth(ty);
  unsigned elemNumBytes = std::max(elemNumBits / 8, 1u);
  unsigned maxMultipleBytes = valInfo.getDivisibility(order[0]);
  unsigned maxMultiple = std::max(maxMultipleBytes / elemNumBytes, 1u);
  unsigned maxContig =
      std::min(valInfo.getContiguity(order[0]), shapePerCTA[order[0]]);
  unsigned alignment = std::min(maxMultiple, maxContig);
  return std::min(alignment, 128 / elemNumBits);
}

/// Check whether transposed reduction should be performed.
///
/// See: https://github.com/intel/intel-xpu-backend-for-triton/issues/1637
inline bool applyTransposedReduction() {
  return tools::getBoolEnv("TRITON_INTEL_REDUCE_TRANSPOSE");
}
} // namespace mlir::triton::gpu::intel

#endif // TRITON_DIALECT_TRITON_INTEL_GPU_IR_UTILS_H
