//===- TritonIntelDialect.h - Triton Intel dialect -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_DIALECT_TRITONINTEL_IR_DIALECT_H
#define TRITON_DIALECT_TRITONINTEL_IR_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#include "intel/include/Dialect/Triton/IR/TritonIntelDialect.h.inc"

#define GET_OP_CLASSES
#include "intel/include/Dialect/Triton/IR/TritonIntelOps.h.inc"

#endif // TRITON_DIALECT_TRITONINTEL_IR_DIALECT_H
