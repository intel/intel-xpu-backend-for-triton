//===- TypeConverter.cpp - TritonIntelGPUToLLVM Type Converter --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "intel/include/TritonIntelGPUToLLVM/TypeConverter.h"

static Type convertTritonPointerType(triton::PointerType type) {
  MLIRContext *ctx = type.getContext();
  Type pointeeType = type.getPointeeType();

  if (isa<RankedTensorType>(pointeeType)) {
    auto rankedTensorType = cast<RankedTensorType>(pointeeType);
    // struct { offset0, offset1, shape0, shape1, stride0,
    // stride1, base_ptr};
    Type eleType = rankedTensorType.getElementType();
    ArrayRef<int64_t> shape = rankedTensorType.getShape();
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

static Type convertTritonDescType(triton::TensorDescType type) {
  MLIRContext *ctx = type.getContext();
  RankedTensorType blockType = type.getBlockType();
  // struct { shape0, shape1, ..., stride0, stride1, ..., base_ptr }
  ArrayRef<int64_t> shape = blockType.getShape();
  SmallVector<Type, 4> types;
  // shapes, strides
  for (size_t i = 0; i < 2 * shape.size(); ++i)
    types.push_back(IntegerType::get(ctx, 64));
  // base_ptr (global address space = 1 on Intel GPUs)
  types.push_back(LLVM::LLVMPointerType::get(ctx, 1));
  return LLVM::LLVMStructType::getLiteral(ctx, types);
}

TritonIntelGPUToLLVMTypeConverter::TritonIntelGPUToLLVMTypeConverter(
    MLIRContext *ctx, LowerToLLVMOptions &option,
    const TargetInfoBase &targetInfo, const DataLayoutAnalysis *analysis)
    : TritonGPUToLLVMTypeConverter(ctx, option, targetInfo, analysis) {
  addConversion([&](triton::PointerType type) -> std::optional<Type> {
    return convertTritonPointerType(type);
  });
  addConversion([&](triton::TensorDescType type) -> std::optional<Type> {
    return convertTritonDescType(type);
  });
}
