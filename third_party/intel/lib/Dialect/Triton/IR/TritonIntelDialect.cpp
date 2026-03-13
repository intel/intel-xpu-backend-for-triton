//===- TritonIntelDialect.cpp - Triton Intel dialect registration ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "intel/include/Dialect/Triton/IR/TritonIntelDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include "triton/Dialect/Triton/IR/Types.h"

using namespace mlir;
using namespace mlir::triton::intel;

void TritonIntelDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "intel/include/Dialect/Triton/IR/TritonIntelOps.cpp.inc"
      >();
}

void MakeTensorPtrOp::build(OpBuilder &builder, OperationState &state,
                            Value base, ValueRange shape, ValueRange strides,
                            ValueRange offsets, ArrayRef<int32_t> tensorShape,
                            ArrayRef<int32_t> order) {
  auto pointerType = cast<triton::PointerType>(base.getType());

  auto tensorType = RankedTensorType::get(
      SmallVector<int64_t>(tensorShape.begin(), tensorShape.end()),
      pointerType.getPointeeType());
  auto result =
      triton::PointerType::get(tensorType, pointerType.getAddressSpace());

  build(builder, state, result, base, shape, strides, offsets,
        builder.getDenseI32ArrayAttr(order));
}

OpFoldResult AdvanceOp::fold(FoldAdaptor adaptor) {
  SmallVector<OpFoldResult> rawOffsets = getOffsets();
  auto offsets = triton::getConstantIntValues(rawOffsets);
  if (!offsets.has_value())
    return {};

  for (int64_t offset : offsets.value())
    if (offset != 0)
      return {};

  return getPtr();
}

#include "intel/include/Dialect/Triton/IR/TritonIntelDialect.cpp.inc"

#define GET_OP_CLASSES
#include "intel/include/Dialect/Triton/IR/TritonIntelOps.cpp.inc"
