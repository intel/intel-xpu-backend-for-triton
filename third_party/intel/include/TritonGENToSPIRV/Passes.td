//===-- Passes.td - TritonGEN to SPIRV Conversion Passes ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITONGEN_TO_SPIRV_CONVERSION_PASSES
#define TRITONGEN_TO_SPIRV_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTritonGENToSPIRV : Pass<"convert-tritongen-to-spirv"> {
  let summary = "Convert the Triton GEN dialect to the SPIRV dialect";
  let description = [{
    This pass converts the TritonGEN dialect operations to SPIRV dialect operations.
  }];
  let dependentDialects = ["spirv::SPIRVDialect"];
}

#endif // TRITONGEN_TO_SPIRV_CONVERSION_PASSES
