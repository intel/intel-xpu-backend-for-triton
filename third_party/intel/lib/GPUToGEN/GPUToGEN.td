//==-- GPUToGEN.td - GPU Ops to GEN Patterns -----------------*- tablegen -*==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines Patterns to lower GPU ops to GEN ops.
//
//===----------------------------------------------------------------------===//

#ifndef GPUTOGEN_CONVERSION_TD
#define GPUTOGEN_CONVERSION_TD

include "mlir/IR/PatternBase.td"
include "mlir/Dialect/GPU/IR/GPUOps.td"
include "triton/Dialect/GEN/IR/GENOps.td"

def : Pat<(GPU_BarrierOp), (GEN_BarrierOp)>;

#endif // GPUTOGEN_CONVERSION_TD
