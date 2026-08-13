//===- Dialect.h - TritonIntelGPU Dialect -------------------------------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_DIALECT_TRITON_INTEL_GPU_IR_DIALECT_H
#define TRITON_DIALECT_TRITON_INTEL_GPU_IR_DIALECT_H

#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "intel/include/Dialect/TritonIntelGPU/IR/Attributes.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h.inc"

#define GET_OP_CLASSES
#include "intel/include/Dialect/TritonIntelGPU/IR/Ops.h.inc"

namespace mlir::triton::gpu::intel {
struct L2Cache : public SideEffects::Resource::Base<L2Cache> {
  StringRef getName() const final { return "<intel::L2Cache>"; }
};

/// Return true when the base address of a 2D block IO operation must be
/// compensated to satisfy the HW base-address alignment requirement.
///
/// MaterializeBlockPointer guarantees a 4-byte aligned base address, so the
/// compensation can be skipped when the target's requirement is at most 4 bytes
/// (as advertised by the `ttig.2d_block_io_base_alignment` module attribute).
/// Otherwise (e.g. the 64-byte requirement on BMG) the base address must be
/// compensated. Absence of the attribute is conservatively treated as requiring
/// compensation.
inline bool needs2DBlockIOAlignmentCompensation(Operation *op) {
  if (!isa<ModuleOp>(op))
    op = op->getParentOfType<ModuleOp>();
  auto alignment = op->getAttrOfType<IntegerAttr>(
      TritonIntelGPUDialect::get2DBlockIOBaseAlignmentAttrName());
  return !alignment || alignment.getInt() > 4;
}

/// Derive the compact scale encoding from the src encoding, or {} on failure.
/// Used by DecomposeScaledBlocked to constrain the scale operand's layout.
Attribute deriveScaleEncoding(Attribute srcEnc, ArrayRef<int64_t> srcShape,
                              int64_t axis, int64_t scaleFactor);
} // namespace mlir::triton::gpu::intel

#endif // TRITON_DIALECT_TRITON_INTEL_GPU_IR_DIALECT_H
