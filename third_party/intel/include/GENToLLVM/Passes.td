//===-- Passes.td - GEN to LLVM Conversion Passes ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GEN_TO_LLVM_CONVERSION_PASSES
#define GEN_TO_LLVM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertGENToLLVM : Pass<"convert-gen-to-llvm"> {
  let summary = "Convert the GEN dialect to the LLVM dialect";
  let description = [{
    This pass converts GEN dialect operations to LLVM dialect operations.
  }];
  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
}

#endif // GEN_TO_LLVM_CONVERSION_PASSES
