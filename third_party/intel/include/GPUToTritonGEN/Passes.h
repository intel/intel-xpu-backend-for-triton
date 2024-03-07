//===- Passes.h - GPU to TritonGEN Conversion Passes ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GPU_TO_TRITONGEN_CONVERSION_PASSES_H
#define GPU_TO_TRITONGEN_CONVERSION_PASSES_H

#include "intel/include/GPUToTritonGEN/GPUToTritonGENPass.h"

namespace mlir {
namespace triton {

/// Generate the code for registering the conversion pass.
#define GEN_PASS_REGISTRATION
#include "intel/include/GPUToTritonGEN/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif // GPU_TO_TRITONGEN_CONVERSION_PASSES_H
