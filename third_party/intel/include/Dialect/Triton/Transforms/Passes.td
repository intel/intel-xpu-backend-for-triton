//===-- Passes.td - Intel TritonDialect passes definition --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_DIALECT_TRITON_INTEL_TRANSFORMS_PASSES
#define TRITON_DIALECT_TRITON_INTEL_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def TritonIntelRemoveMasks
    : Pass<"triton-intel-remove-masks", "mlir::ModuleOp"> {
  let summary = "Remove masks from tt.load and tt.store operations";

  let description = [{
    This pass attempts to remove the mask for tt.load and tt.store operations.
    If the masked operation is in a loop, the pass attempts to find a loop
    invariant condition equivalent to the mask condition, and then use it to
    version the loop.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect",
    "mlir::triton::TritonDialect"
  ];
}

#endif // TRITON_DIALECT_TRITON_INTEL_TRANSFORMS_PASSES
