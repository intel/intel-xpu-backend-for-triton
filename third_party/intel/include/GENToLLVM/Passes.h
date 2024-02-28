//===- Passes.h - GEN Conversion Pass Construction and Registration -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GEN_CONVERSION_PASSES_H
#define GEN_CONVERSION_PASSES_H

#include "intel/include/GENToLLVM/GENToLLVMPass.h"

namespace mlir {
namespace triton {

/// Generate the code for registering the conversion pass.
#define GEN_PASS_REGISTRATION
#include "intel/include/GENToLLVM/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif // GEN_CONVERSION_PASSES_H
