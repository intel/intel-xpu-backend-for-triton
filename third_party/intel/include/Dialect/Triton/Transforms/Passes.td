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

def TritonIntelTensorDescToBlockPointer
    : Pass<"triton-intel-tdesc-to-block-pointer", "mlir::ModuleOp"> {
  let summary = "Convert tensor descriptors into block pointers";

  let description = [{
    This pass attempts to convert tensor descriptors into block pointers.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect",
    "mlir::triton::TritonDialect"
  ];
}

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

def TritonIntelFuseReshape
    : Pass<"triton-intel-fuse-reshape", "mlir::ModuleOp"> {
  let summary = "Fuse a tt.reshape operation with a tt.load operation";

  let description = [{
    This pass attempts to fuse a tt.reshape operation with a tt.load operation using a block pointer.
    For example, given:
        %q_27 = arith.constant 1 : i64
        %ptr = tt.make_tensor_ptr %q_view, [%q, %q_23, %q_24], [%q_25, %q_26, %q_27], [%offset_5, %offset_1_13, %q_28]
                                  {order = array<i32: 2, 1, 0>} : <tensor<1x512x64xf16>>
        %load = tt.load %ptr {boundaryCheck = array<i32: 1, 2>} : !tt.ptr<tensor<1x512x64xf16>>
        %reshape = tt.reshape %load : tensor<1x512x64xf16> -> tensor<512x64xf16>
    The transformation would drops the reshape operation and adjust the make_tensor_ptr operation as follows:
        %q_27 = arith.constant 1 : i64
        %ptr = tt.make_tensor_ptr %q_view, [%q_23, %q_24], [%q_26, %q_27], [%offset_1_13, %offset_5*%q_25+%q_28]
                                  {order = array<i32: 1, 0>} : <tensor<512x64xf16>>
        %load = tt.load %ptr {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<512x64xf16>>
  }];

  let dependentDialects = [
    "mlir::triton::TritonDialect"
  ];
}

#endif // TRITON_DIALECT_TRITON_INTEL_TRANSFORMS_PASSES
