//===- TritonGENToSPIRVPass.h - TritonGEN to SPIRV dialect conv. -*- C++ --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_CONVERSION_TRITONGENTOSPIRVPASS_H
#define TRITON_CONVERSION_TRITONGENTOSPIRVPASS_H

#include <memory>

namespace mlir {

class DialectRegistry;
class RewritePatternSet;
class Pass;

namespace triton {

#define GEN_PASS_DECL
#include "intel/include/TritonGENToSPIRV/Passes.h.inc"

void populateTritonGENToSPIRVConversionPatterns(RewritePatternSet &patterns);

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TRITONGENTOSPIRVPASS_H
