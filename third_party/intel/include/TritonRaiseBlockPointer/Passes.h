//===- Passes.h - Triton to Block Pointer Pass ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_RAISE_BLOCK_POINTER_PASSES_H
#define TRITON_RAISE_BLOCK_POINTER_PASSES_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::intel {
#define GEN_PASS_DECL
#include "intel/include/TritonRaiseBlockPointer/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "intel/include/TritonRaiseBlockPointer/Passes.h.inc"
} // namespace mlir::triton::intel

#endif // TRITON_RAISE_BLOCK_POINTER_PASSES_H
