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

def TritonIntelBlockPointerToTensorDesc
    : Pass<"triton-intel-block-pointer-to-tdesc", "mlir::ModuleOp"> {
  let summary = "Convert block pointers into tensor descriptors";

  let description = [{
    This pass attempts to convert block pointers into tensor descriptors.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect",
    "mlir::triton::TritonDialect"
  ];
}

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
        dot %A, ... : tensor<512x64xf16> x tensor<64x32xf16> -> tensor<512x32xf16>

    The transformation drops the reshape operation, and generates:
        %div = %a / %b
        %ptr = tt.make_tensor_ptr %base_ptr, [%s0 * %div + %s1, %s2], [%b, %c], [%x * %div + %y, %z]
                                  {order = array<i32: 1, 0>} : <tensor<512x64xf16>>
        %A = tt.load %ptr {boundaryCheck = array<i32: 1>} : !tt.ptr<tensor<512x64xf16>>
        dot %A, ... : tensor<512x64xf16> x tensor<64x32xf16> -> tensor<512x32xf16>
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
    scf.for %iv = %lb to %ub step %st : i32 {
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
      - `%s2` is equal to 1024
      - the boundary check expression `max(%iv) + load_res.shape_in_dim -1` < `%s2` is true.
  }];

  let dependentDialects = [
    "mlir::triton::TritonDialect"
  ];
}

def TritonIntelStrideVersioning
    : Pass<"triton-intel-stride-versioning", "mlir::ModuleOp"> {
  let summary = "Version loops containing block pointers loads if none of them is 1";

  let description = [{
    This pass versions loops that contain tt.load on a block pointer, if none of the block pointer strides is 1.
    For example, given:

    %cst = arith.constant ...
    %ptr = tt.make_tensor_ptr %base, [%s0, %s1], [%cst, %b], [%x, %y] : <tensor<512x64xf16>>
    scf.for ... {
      %load = tt.load %ptr : !tt.ptr<tensor<512x64xf16>>
      ...
    }

    The transformation creates:

    %ptr = tt.make_tensor_ptr %base, [%s0, %s1], [%a, %b], [%x, %y] : <tensor<512x64xf16>>
    %ptr' = tt.make_tensor_ptr %base, [%s0, %s1], [%a, 1], [%x, %y] : <tensor<512x64xf16>>
    if (%b == 1)
      scf.for ... {
        %load = tt.load %ptr' : !tt.ptr<tensor<512x64xf16>>
        ...
      }
    else
      scf.for ... {
        %load = tt.load %ptr : !tt.ptr<tensor<512x64xf16>>
        ...
      }
  }];

  let dependentDialects = [
    "mlir::triton::TritonDialect"
  ];
}

def TritonIntelSimplifySignedArithmetic
    : Pass<"triton-intel-simplify-signed-arithmetic", "mlir::ModuleOp"> {
  let summary = "Simplify signed arithmetic ops by converting to unsigned when operands are non-negative";

  let description = [{
    Converts signed arithmetic operations (remsi, divsi) to their unsigned
    equivalents when operands are provably non-negative. This enables simpler
    code generation (e.g., power-of-2 remainder becomes bitwise AND, division
    becomes logical shift right).

    Conversions:
    - arith.remsi -> arith.remui (when dividend >= 0, divisor > 0)
    - arith.divsi -> arith.divui (when dividend >= 0, divisor > 0)

    Sources of non-negative values:
    - tt.get_program_id (always [0, 2^31-1])
    - tt.get_num_programs (always [1, 2^31])
    - tt.make_range(start, end) with start >= 0
    - Non-negative integer constants
    - arith.addi/muli of non-negative operands
    - arith.divsi/divui/remui with non-negative dividend and positive divisor
  }];

  let dependentDialects = [
    "mlir::triton::TritonDialect"
  ];
}

#endif // TRITON_DIALECT_TRITON_INTEL_TRANSFORMS_PASSES
