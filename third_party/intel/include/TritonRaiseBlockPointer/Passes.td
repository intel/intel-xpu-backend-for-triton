//===-- Passes.td - Triton to Block Pointer Passes ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_RAISE_BLOCK_POINTER_PASSES
#define TRITON_RAISE_BLOCK_POINTER_PASSES

include "mlir/Pass/PassBase.td"

def TritonRaiseBlockPointer
    : Pass<"triton-raise-block-pointer", "mlir::ModuleOp"> {
  let summary = "Convert Triton non-block pointer to block pointer";
  let description = [{
    Pass to raise different patterns operating on pointers to block pointers.
    The basic idea of this pass is to convert code operating on tensors of
    pointers (`tensor<...x!tt.ptr<ty>>`) to pointers to tensors
    (`!tt.ptr<tensor<...xty>>`) using `tt.make_block_ptr` operations.
  }];

  let dependentDialects = [
      "mlir::arith::ArithDialect",
      "mlir::triton::TritonDialect",
  ];
}

#endif // TRITON_RAISE_BLOCK_POINTER_PASSES
