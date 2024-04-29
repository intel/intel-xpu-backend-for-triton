//===- Utility.h - Triton Intel GPU utilities -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_DIALECT_TRITONINTELGPU_TRANSFORMS_UTILITY_H
#define TRITON_DIALECT_TRITONINTELGPU_TRANSFORMS_UTILITY_H

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonIntelGPU/Transforms/Passes.h"

namespace mlir {
namespace triton {
namespace gpu {

class DotOperandEncodingAttr;
namespace intel {

// data type for D_C_A_B.
enum class DPASEngineType : uint8_t {
  // floating-point XMX engine instr
  FP32_FP32_FP16_FP16 = 0, // default
  FP32_FP32_BF16_BF16,
  FP32_FP32_TF32_TF32,
  FP16_FP16_FP16_FP16,
  BF16_BF16_BF16_BF16,
  // integer XMX engine instr
  U32_U32_U8_U8,
  S32_S32_S8_S8,
  //
  NOT_APPLICABLE,
};

bool supportDPAS(DotOp op, DeviceArch arch);
DPASEngineType getDPASType(DotOp op);

// Infers the encoding of the source of op given the result encoding.
std::optional<Attribute> inferSrcEncoding(Operation *op, Attribute encoding);

// Retuns true is the operation is an expensive load or store operation.
bool isExpensiveLoadOrStore(Operation *op);

// Returns true if the tensor type has a dot dpas encoding.
bool hasDotDpasEncoding(RankedTensorType tensorType);

// Returns the dot encoding of the tensor type or std::nullopt.
std::optional<DotOperandEncodingAttr>
getDotEncoding(RankedTensorType tensorType);

// Get backward slice of tensor values starting from the root node along with
// encoding propagation.
LogicalResult getConvertBackwardSlice(
    Value root, SetVector<Value> &slice, Attribute rootEncoding,
    DenseMap<Value, Attribute> &layout,
    std::function<bool(Operation *)> stopPropagation = nullptr);

} // namespace intel
} // namespace gpu
} // namespace triton
} // namespace mlir

#endif // TRITON_DIALECT_TRITONINTELGPU_TRANSFORMS_UTILITY_H
