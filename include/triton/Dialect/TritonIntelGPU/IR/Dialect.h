#ifndef TRITON_DIALECT_TRITON_INTEL_GPU_IR_DIALECT_H
#define TRITON_DIALECT_TRITON_INTEL_GPU_IR_DIALECT_H

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"

// TritonGPU depends on Triton
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonIntelGPU/IR/Attributes.h"
#include "triton/Dialect/TritonIntelGPU/IR/Dialect.h.inc"

#define GET_OP_CLASSES
#include "triton/Dialect/TritonIntelGPU/IR/Ops.h.inc"

namespace mlir {
namespace triton {
namespace gpu {
namespace intel {} // namespace intel
} // namespace gpu
} // namespace triton
} // namespace mlir

#endif // TRITON_DIALECT_TRITON_INTEL_GPU_IR_DIALECT_H
