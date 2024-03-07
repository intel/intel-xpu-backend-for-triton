//===- TritonGENDialect.cpp - TritonGEN Dialect registration --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "triton/Dialect/TritonGEN/IR/TritonGENDialect.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace mlir::triton::TritonGEN;

void TritonGENDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "triton/Dialect/TritonGEN/IR/TritonGENOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "triton/Dialect/TritonGEN/IR/TritonGENOpsAttrDefs.cpp.inc"
      >();

  // Support unknown operations because not all GEN operations are registered.
  allowUnknownOperations();
}

#include "triton/Dialect/TritonGEN/IR/TritonGENDialect.cpp.inc"
#include "triton/Dialect/TritonGEN/IR/TritonGENOpsEnums.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "triton/Dialect/TritonGEN/IR/TritonGENOpsAttrDefs.cpp.inc"
#define GET_OP_CLASSES
#include "triton/Dialect/TritonGEN/IR/TritonGENOps.cpp.inc"
