//===- Dialect.h - MLIR GEN dialect -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the GEN dialect in MLIR, containing Intel GEN operations.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_DIALECT_GEN_IR_DIALECT_H_
#define TRITON_DIALECT_GEN_IR_DIALECT_H_

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "triton/Dialect/GEN/IR/Dialect.h.inc"
#include "triton/Dialect/GEN/IR/GENOpsEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "triton/Dialect/GEN/IR/GENOpsAttrDefs.h.inc"

#define GET_OP_CLASSES
#include "triton/Dialect/GEN/IR/GENOps.h.inc"

namespace mlir {
namespace GEN {

/// GEN memory space identifiers following SPIRV storage class convention
/// https://github.com/KhronosGroup/SPIRV-LLVM-Translator/blob/main/docs/SPIRVRepresentationInLLVM.rst#address-spaces
///
enum GENMemorySpace {
  kFunction = 0,        // OpenCL workitem address space
  kCrossWorkgroup = 1,  // OpenCL Global memory
  kUniformConstant = 2, // OpenCL Constant memory
  kWorkgroup = 3,       // OpenCL Local memory
  kGeneric = 4          // OpenCL Generic memory
};

} // namespace GEN
} // namespace mlir

#endif // TRITON_DIALECT_GEN_IR_DIALECT_H_
