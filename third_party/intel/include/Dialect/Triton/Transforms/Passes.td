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

def TritonRewriteTensorDescriptorToPointer : Pass</*cli-arg*/"triton-intel-rewrite-tensor-descriptor-to-pointer", /*Op*/"mlir::ModuleOp"> {
  let summary = "Rewrite load/stores of tensor descriptors into pointer load/stores";
  let description = [{
    This pass rewrites all load/store semantics initiated by a `tt.make_tensor_descriptor` into pointer semantics. After
    this pass, `tt.make_tensor_descriptor`  will disappear, and it generates logics to compute the pointer/mask/other
    for each load/store.
  }];

  let dependentDialects = ["mlir::triton::TritonDialect"];
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
  let summary = "Fuse a tt.reshape operation with a tt.load operation (tensor descriptors only)";

  let description = [{
    This pass attempts to fuse a tt.reshape operation with a tt.descriptor_load operation.
    For example, given:
        %desc = tt.make_tensor_descriptor %base_ptr, [%s0, %s1, %s2], [%a, %b, %c]
                                          : !tt.tensordesc<tensor<1x512x64xf16>>
        %load = tt.descriptor_load %desc[%x, %y, %z]
                                   : !tt.tensordesc<tensor<1x512x64xf16>> -> tensor<1x512x64xf16>
        %A = tt.reshape %load : tensor<1x512x64xf16> -> tensor<512x64xf16>
        dot %A, ... : tensor<512x64xf16> x tensor<64x32xf16> -> tensor<512x32xf16>

    The transformation drops the reshape operation, and generates:
        %div = %a / %b
        %desc = tt.make_tensor_descriptor %base_ptr, [%s0 * %div + %s1, %s2], [%b, %c]
                                          : !tt.tensordesc<tensor<512x64xf16>>
        %A = tt.descriptor_load %desc[%x * %div + %y, %z]
                                : !tt.tensordesc<tensor<512x64xf16>> -> tensor<512x64xf16>
        dot %A, ... : tensor<512x64xf16> x tensor<64x32xf16> -> tensor<512x32xf16>
  }];

  let dependentDialects = [
    "mlir::triton::TritonDialect"
  ];
}

def TritonIntelStrideVersioning
    : Pass<"triton-intel-stride-versioning", "mlir::ModuleOp"> {
  let summary = "Version loops containing tensor descriptor loads if none of the strides is 1";

  let description = [{
    This pass versions loops that contain tt.descriptor_load on a tensor descriptor, if none of the descriptor strides is 1.
    For example, given:

    %cst = arith.constant ...
    %desc = tt.make_tensor_descriptor %base, [%s0, %s1], [%cst, %b]
            : !tt.tensordesc<tensor<512x64xf16>>
    scf.for ... {
      %load = tt.descriptor_load %desc[%x, %y]
              : !tt.tensordesc<tensor<512x64xf16>> -> tensor<512x64xf16>
      ...
    }

    The transformation creates:

    %desc = tt.make_tensor_descriptor %base, [%s0, %s1], [%a, %b]
            : !tt.tensordesc<tensor<512x64xf16>>
    %desc' = tt.make_tensor_descriptor %base, [%s0, %s1], [%a, 1]
             : !tt.tensordesc<tensor<512x64xf16>>
    if (%b == 1)
      scf.for ... {
        %load = tt.descriptor_load %desc'[%x, %y]
                : !tt.tensordesc<tensor<512x64xf16>> -> tensor<512x64xf16>
        ...
      }
    else
      scf.for ... {
        %load = tt.descriptor_load %desc[%x, %y]
                : !tt.tensordesc<tensor<512x64xf16>> -> tensor<512x64xf16>
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
