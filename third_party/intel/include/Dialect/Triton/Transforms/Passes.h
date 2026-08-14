//===- Passes.h - Intel Pass Construction and Registration ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_DIALECT_TRITON_INTEL_TRANSFORMS_PASSES_H
#define TRITON_DIALECT_TRITON_INTEL_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir::triton {
class AxisInfo;
} // namespace mlir::triton

namespace mlir::triton::intel {

#define GEN_PASS_DECL
#include "intel/include/Dialect/Triton/Transforms/Passes.h.inc"

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "intel/include/Dialect/Triton/Transforms/Passes.h.inc"

/// Returns true if `op` is an `arith.divsi` or `arith.remsi` that the AxisInfo
/// deduction would have drawn a stronger conclusion from, were its dividend
/// known to be non-negative.
///
/// The deduction is only sound for a non-negative dividend, so it is disabled
/// for the signed operations (see DivOpAxisInfoVisitor::getConstancy and
/// RemOpAxisInfoVisitor::getContiguity in lib/Analysis/AxisInfo.cpp). This
/// query mirrors those two sites, and exists so
/// TritonIntelSpeculateSignedDivRem can find the operations worth converting to
/// their unsigned form under a runtime check. It is exposed only so the mirror
/// can be unit tested against hand-built lattice values.
///
/// `lhs` and `rhs` are the AxisInfo of the operands.
bool signedDivRemDeductionApplies(Operation *op, const AxisInfo &lhs,
                                  const AxisInfo &rhs);

} // namespace mlir::triton::intel

#endif // TRITON_DIALECT_TRITON_INTEL_TRANSFORMS_PASSES_H
