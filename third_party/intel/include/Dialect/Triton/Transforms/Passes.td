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
  let summary = "Fuse a tt.reshape operation with a tt.load operation (block ptrs only)";

  let description = [{
    This pass attempts to fuse a tt.reshape operation with a tt.load operation.
    For example, given:
        %ptr = tt.make_tensor_ptr %base_ptr, [%s0, %s1, %s2], [%a, %b, %c], [%x, %y, %z]
                                  {order = array<i32: 2, 1, 0>} : <tensor<1x512x64xf16>>
        %load = tt.load %ptr {boundaryCheck = array<i32: 2>} : !tt.ptr<tensor<1x512x64xf16>>
        %A = tt.reshape %load : tensor<1x512x64xf16> -> tensor<512x64xf16>
        %dot %A, ... : tensor<512x64xf16> x tensor<64x32xf16> -> tensor<512x32xf16>

    The transformation drops the reshape operation, and generates:
        %div = %a / %b
        %ptr = tt.make_tensor_ptr %base_ptr, [%s0 * %div + %s1, %s2], [%b, %c], [%x * %div + %y, %z]
                                  {order = array<i32: 1, 0>} : <tensor<512x64xf16>>
        %A = tt.load %ptr {boundaryCheck = array<i32: 1>} : !tt.ptr<tensor<512x64xf16>>
        %dot %A, ... : tensor<512x64xf16> x tensor<64x32xf16> -> tensor<512x32xf16>
  }];

  let dependentDialects = [
    "mlir::triton::TritonDialect"
  ];
}

def TritonIntelRemoveBoundaryChecks
    : Pass<"triton-intel-remove-boundary-checks", "mlir::ModuleOp"> {
  let summary = "Remove unnecessary boundary checks from load operations (block pointers only)";

  let description = [{
    This pass attempts to remove boundary checks that aren't necessary in a tt.load operation.
    For example, given:
    %lb = arith.bitcast %c0_i32 : i32 to i32
    %ub = arith.bitcast %c1024_i32 : i32 to i32
    %st = arith.bitcast %c64_i32 : i32 to i32
    scf.for %i = %lb to %ub step %st : i32 {
      %s0 = arith.constant 512 : i64
      %s1 = arith.constant 64 : i64
      %s2 = arith.constant 1024 : i64
      %a = arith.constant 65536 : i64
      %b = arith.constant 1 : i64
      %b = arith.constant 64 : i64
      %y = arith.constant 0 : i32
      %ptr = tt.make_tensor_ptr %base, [%s0, %s1, %s2], [%a, %b, %c], [%x, %y, %iv]
                                {order = array<i32: 2, 1, 0>} : <tensor<1x512x64xf16>>
      %load = tt.load %ptr {boundaryCheck = array<i32: 2>} : !tt.ptr<tensor<1x512x64xf16>>
      ...
      // here %ptr is never updated.
    }

    The transformation would drop the boundary check on the load operation because:
      - `%ptr` is never advanced in the loop
      - `%iv` has values [0, 64, 128, ..., 960], max(%iv) = 960
      - `%s2` is qual to 1014
      - the boundary check expression `%iv` < `%s2` is always true
  }];

  let dependentDialects = [
    "mlir::triton::TritonDialect"
  ];
}

#endif // TRITON_DIALECT_TRITON_INTEL_TRANSFORMS_PASSES
