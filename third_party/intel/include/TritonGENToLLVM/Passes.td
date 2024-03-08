//===-- Passes.td - TritonGEN to LLVM Conversion Passes ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITONGEN_TO_LLVM_CONVERSION_PASSES
#define TRITONGEN_TO_LLVM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTritonGENToLLVM : Pass<"convert-tritongen-to-llvm"> {
  let summary = "Convert the Triton GEN dialect to the LLVM dialect";
  let description = [{
    This pass converts the TritonGEN dialect operations to LLVM dialect operations.
  }];
  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
}

#endif // TRITONGEN_TO_LLVM_CONVERSION_PASSES
