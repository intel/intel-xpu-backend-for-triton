//===- GENToLLVMPass.h - GEN to LLVM dialect conversion ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_CONVERSION_GENTOLLVMPASS_H
#define TRITON_CONVERSION_GENTOLLVMPASS_H

#include <memory>

namespace mlir {

class DialectRegistry;
class LLVMTypeConverter;
class RewritePatternSet;
class Pass;

namespace triton {

#define GEN_PASS_DECL
#include "intel/include/GENToLLVM/Passes.h.inc"

void populateGENToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                         RewritePatternSet &patterns);

void registerConvertGENToLLVMInterface(DialectRegistry &registry);

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_GENTOLLVMPASS_H
