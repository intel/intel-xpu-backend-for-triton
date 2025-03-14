//===- TypeConverter.cpp - TritonIntelGPUToLLVM Type Converter --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "intel/include/TritonIntelGPUToLLVM/TypeConverter.h"
#include "triton/Tools/Sys/GetEnv.hpp"

static Type convertTritonPointerType(triton::PointerType type) {
  auto ctx = type.getContext();
  auto pointeeType = type.getPointeeType();
  if (isa<RankedTensorType>(pointeeType)) {
    auto rankedTensorType = cast<RankedTensorType>(pointeeType);
    // struct { offset0, offset1, shape0, shape1, stride0,
    // stride1, base_ptr};
    auto eleType = rankedTensorType.getElementType();
    auto shape = rankedTensorType.getShape();
    SmallVector<Type, 4> types;
    // offsets
    for (size_t i = 0; i < shape.size(); ++i)
      types.push_back(IntegerType::get(ctx, 32));
    // shapes, strides
    for (size_t i = 0; i < 2 * shape.size(); ++i)
      types.push_back(IntegerType::get(ctx, 64));

    types.push_back(LLVM::LLVMPointerType::get(ctx, type.getAddressSpace()));

    return LLVM::LLVMStructType::getLiteral(ctx, types);
  }
  return LLVM::LLVMPointerType::get(ctx, type.getAddressSpace());
}

TritonIntelGPUToLLVMTypeConverter::TritonIntelGPUToLLVMTypeConverter(
    MLIRContext *ctx, LowerToLLVMOptions &option,
    const TargetInfoBase &targetInfo, bool isAdvancedPathEnabled,
    const DataLayoutAnalysis *analysis)
    : TritonGPUToLLVMTypeConverter(ctx, option, targetInfo, analysis) {
  addConversion([&](triton::PointerType type) -> std::optional<Type> {
    return convertTritonPointerType(type);
  });
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
