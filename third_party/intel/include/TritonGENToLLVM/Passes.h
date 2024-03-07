//===- Passes.h - TritonGEN to LLVM Conversion Passes ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITONGEN_TO_LLVM_CONVERSION_PASSES_H
#define TRITONGEN_TO_LLVM_CONVERSION_PASSES_H

#include "intel/include/TritonGENToLLVM/TritonGENToLLVMPass.h"

namespace mlir {
namespace triton {

/// Generate the code for registering the conversion pass.
#define GEN_PASS_REGISTRATION
#include "intel/include/TritonGENToLLVM/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif // TRITONGEN_TO_LLVM_CONVERSION_PASSES_H
