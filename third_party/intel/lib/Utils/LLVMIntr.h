//===- LLVMIntr.h - Utilities to emit LLVM intrinsic calls ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_INTEL_UTILS_LLVMINTR_H
#define TRITON_INTEL_UTILS_LLVMINTR_H

#include "intel/lib/TritonGENToLLVM/Attributes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"

#include <string>

namespace mlir::triton::gpu::intel {

struct LLVMFuncAttributeOptions {
  bool isConvergent = false;
  bool isNoUnwind = false;
  bool isWillReturn = false;
  LLVM::MemoryEffectsAttr memEffectsAttr{};
};

constexpr LLVMFuncAttributeOptions convergentAttrs = {true, false, false, {}};
constexpr LLVMFuncAttributeOptions noUnwindAttrs = {false, true, false, {}};
constexpr LLVMFuncAttributeOptions noUnwindWillReturnAttrs = {
    false, true, true, {}};
constexpr LLVMFuncAttributeOptions convergentNoUnwindWillReturnAttrs = {
    true, true, true, {}};

LLVM::CallOp createDeviceFunctionCall(
    ConversionPatternRewriter &rewriter, StringRef funcName, Type retType,
    mlir::ArrayRef<mlir::Type> argTypes, ArrayRef<Value> args,
    ArrayRef<std::pair<unsigned, StringRef>> paramAttrs,
    const LLVMFuncAttributeOptions &funcAttributeOptions,
    const AttributeList &passthroughAttrs = {},
    LLVM::cconv::CConv cc = LLVM::cconv::CConv::SPIR_FUNC);

} // namespace mlir::triton::gpu::intel

#endif // TRITON_INTEL_UTILS_LLVMINTR_H
