//===- Passes.h - Intel Pass Construction and Registration ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_DIALECT_TRITON_INTEL_TRANSFORMS_PASSES_H
#define TRITON_DIALECT_TRITON_INTEL_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir::triton::intel {

#define GEN_PASS_DECL
#include "intel/include/Dialect/Triton/Transforms/Passes.h.inc"

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "intel/include/Dialect/Triton/Transforms/Passes.h.inc"

} // namespace mlir::triton::intel

#endif // TRITON_DIALECT_TRITON_INTEL_TRANSFORMS_PASSES_H
