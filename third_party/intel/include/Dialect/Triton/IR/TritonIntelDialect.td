//===- TritonIntelDialect.td - Triton Intel dialect ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_INTEL_DIALECT
#define TRITON_INTEL_DIALECT

include "mlir/IR/OpBase.td"

def TritonIntel_Dialect : Dialect {
  let name = "tti";
  let cppNamespace = "::mlir::triton::intel";
  let summary = "Triton Intel IR dialect";
  let description = [{
    Triton Intel dialect defining Intel-specific Triton operations.
  }];

  let dependentDialects = [
    "arith::ArithDialect",
    "ub::UBDialect"
  ];
}

#endif // TRITON_INTEL_DIALECT
