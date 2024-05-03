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

namespace mlir::triton::TritonGEN {

/// TritonGEN memory space identifiers following SPIRV storage class convention
/// https://github.com/KhronosGroup/SPIRV-LLVM-Translator/blob/main/docs/SPIRVRepresentationInLLVM.rst#address-spaces
///
enum TritonGENMemorySpace {
  kFunction = 0,        // OpenCL workitem address space
  kCrossWorkgroup = 1,  // OpenCL Global memory
  kUniformConstant = 2, // OpenCL Constant memory
  kWorkgroup = 3,       // OpenCL Local memory
  kGeneric = 4          // OpenCL Generic memory
};

} // namespace mlir::triton::TritonGEN

#endif // TRITON_DIALECT_TRITONGENDIALECT_H
