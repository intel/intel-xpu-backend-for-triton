//==-- GPUToTritonGEN.td - GPU Ops to TritonGEN Patterns -----*- tablegen -*==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines Patterns to lower GPU ops to TritonGEN ops.
//
//===----------------------------------------------------------------------===//

#ifndef GPUTOTRITONGEN_CONVERSION_TD
#define GPUTOTRITONGEN_CONVERSION_TD

include "mlir/IR/PatternBase.td"
include "mlir/Dialect/GPU/IR/GPUOps.td"
include "triton/Dialect/TritonGEN/IR/TritonGENOps.td"

def : Pat<(GPU_BarrierOp), (TritonGEN_BarrierOp)>;

#endif // GPUTOTRITONGEN_CONVERSION_TD
