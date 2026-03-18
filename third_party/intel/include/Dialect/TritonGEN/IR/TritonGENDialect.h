//===- TritonGENDialect.h - MLIR TritonGEN dialect --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the TritonGEN dialect in MLIR, containing Intel GEN
// operations.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_DIALECT_TRITONGENDIALECT_H
#define TRITON_DIALECT_TRITONGENDIALECT_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#include "intel/include/Dialect/TritonGEN/IR/TritonGENDialect.h.inc"
#include "intel/include/Dialect/TritonGEN/IR/TritonGENOpsEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "intel/include/Dialect/TritonGEN/IR/TritonGENOpsAttrDefs.h.inc"

#define GET_OP_CLASSES
#include "intel/include/Dialect/TritonGEN/IR/TritonGENOps.h.inc"

#include "intel/include/Dialect/TritonGEN/IR/TritonGENMemorySpace.h"

namespace mlir::triton::TritonGEN {

struct L2Cache : public SideEffects::Resource::Base<L2Cache> {
  StringRef getName() const final { return "<L2Cache>"; }
};

} // namespace mlir::triton::TritonGEN
#endif // TRITON_DIALECT_TRITONGENDIALECT_H
