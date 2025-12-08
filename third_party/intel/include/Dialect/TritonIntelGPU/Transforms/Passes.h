//===- Passes.h - Intel Pass Construction and Registration ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_DIALECT_TRITON_INTEL_GPU_TRANSFORMS_PASSES_H
#define TRITON_DIALECT_TRITON_INTEL_GPU_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir::triton::gpu::intel {

/// Split barrier scope
enum SplitBarrierScope { None = 0, Workgroup = 1, Subgroup = 2 };

#define GEN_PASS_DECL
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

#endif // TRITON_DIALECT_TRITON_INTEL_GPU_TRANSFORMS_PASSES_H
