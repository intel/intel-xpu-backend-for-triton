//===- TypeConverter.h - TritonIntelGPUToLLVM Type Converter ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_CONVERSION_TRITONINTELGPUTOLLVM_TYPECONVERTER_H
#define TRITON_CONVERSION_TRITONINTELGPUTOLLVM_TYPECONVERTER_H

#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"

/// Extends the type converter used upstream with custom type conversion
/// patterns required by the TritonIntelGPU -> LLVM conversion.
class TritonIntelGPUToLLVMTypeConverter : public TritonGPUToLLVMTypeConverter {
public:
  using TypeConverter::convertType;

  TritonIntelGPUToLLVMTypeConverter(
      MLIRContext *ctx, LowerToLLVMOptions &option,
      const DataLayoutAnalysis *analysis = nullptr);
};

#endif // TRITON_CONVERSION_TRITONINTELGPUTOLLVM_TYPECONVERTER_H
