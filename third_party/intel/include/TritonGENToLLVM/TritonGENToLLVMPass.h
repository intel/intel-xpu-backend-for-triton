//===- TritonGENToLLVMPass.h - TritonGEN to LLVM dialect conv. --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_CONVERSION_TRITONGENTOLLVMPASS_H
#define TRITON_CONVERSION_TRITONGENTOLLVMPASS_H

#include <memory>

namespace mlir {

class DialectRegistry;
class LLVMTypeConverter;
class RewritePatternSet;
class Pass;

namespace triton {

#define GEN_PASS_DECL
#include "intel/include/TritonGENToLLVM/Passes.h.inc"

void populateTritonGENToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                               RewritePatternSet &patterns);

void registerConvertTritonGENToLLVMInterface(DialectRegistry &registry);

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TRITONGENTOLLVMPASS_H
