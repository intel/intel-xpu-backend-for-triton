//===- Utils.h - TritonIntelGPU Utils -----------------------------------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_DIALECT_TRITON_INTEL_GPU_IR_UTILS_H
#define TRITON_DIALECT_TRITON_INTEL_GPU_IR_UTILS_H

#include <cstdlib>

namespace mlir::triton::gpu::intel {
/// Check whether transposed reduction should be performed.
///
/// See: https://github.com/intel/intel-xpu-backend-for-triton/issues/1637
inline bool applyTransposedReduction() {
  return std::getenv("TRITON_INTEL_REDUCE_TRANSPOSE");
}
} // namespace mlir::triton::gpu::intel

#endif // TRITON_DIALECT_TRITON_INTEL_GPU_IR_UTILS_H
