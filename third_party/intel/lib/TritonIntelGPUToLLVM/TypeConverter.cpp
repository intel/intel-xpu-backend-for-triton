//===- TypeConverter.cpp - TritonIntelGPUToLLVM Type Converter --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "intel/include/TritonIntelGPUToLLVM/TypeConverter.h"
#include "triton/Tools/Sys/GetEnv.hpp"

TritonIntelGPUToLLVMTypeConverter::TritonIntelGPUToLLVMTypeConverter(
    MLIRContext *ctx, LowerToLLVMOptions &option, bool isAdvancedPathEnabled,
    const DataLayoutAnalysis *analysis)
    : TritonGPUToLLVMTypeConverter(ctx, option, analysis) {
  // Augment/overwrite type conversions required for the Intel conversion
  // passes.
  if (isAdvancedPathEnabled) {
    // tt::pointer to v2i32.
    addConversion([&](PointerType type) -> std::optional<Type> {
      if (isa<RankedTensorType>(type.getPointeeType())) {
        auto i32Type = mlir::IntegerType::get(type.getContext(), 32);
        return mlir::VectorType::get(2, i32Type);
      }
      return LLVM::LLVMPointerType::get(type.getContext(),
                                        type.getAddressSpace());
    });

    // tensor type is flattened and divided by 16 (subgroupSize).
    addConversion([&](mlir::RankedTensorType type) -> mlir::Type {
      unsigned num = type.getNumElements();
      Type elmTy = type.getElementType();
      if (!type.getEncoding() ||
          isa<mlir::triton::gpu::DotOperandEncodingAttr>(type.getEncoding()))
        num /= 16;
      if (num == 1)
        return elmTy;
      return mlir::VectorType::get(num, elmTy);
    });
  }
}
