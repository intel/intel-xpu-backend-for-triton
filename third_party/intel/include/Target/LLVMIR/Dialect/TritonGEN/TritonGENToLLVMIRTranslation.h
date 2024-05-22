//===-TritonGENToLLVMIRTranslation.h-TritonGEN Dialect to LLVM IR - C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides registration calls for TritonGEN dialect to LLVM IR
// translation.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_TARGET_LLVMIR_DIALECT_TRITONGEN_TRITONGENTOLLVMIRTRANSLATION_H
#define TRITON_TARGET_LLVMIR_DIALECT_TRITONGEN_TRITONGENTOLLVMIRTRANSLATION_H

namespace mlir {

class DialectRegistry;
class MLIRContext;

/// Register the TritonGEN dialect and the translation from it to the LLVM IR in
/// the given registry;
void registerTritonGENDialectTranslation(DialectRegistry &registry);

/// Register the TritonGEN dialect and the translation from it in the registry
/// associated with the given context.
void registerTritonGENDialectTranslation(MLIRContext &context);

} // namespace mlir

#endif // TRITON_TARGET_LLVMIR_DIALECT_TRITONGEN_TRITONGENTOLLVMIRTRANSLATION_H
