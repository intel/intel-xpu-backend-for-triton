//===- SPIRVTargetInfo.h - Target dependent information for SPIRV arch ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_SPIRVTARGETINFO_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_SPIRVTARGETINFO_H

#include "TargetInfo.h"

namespace mlir::triton::intel {
class SPIRVTargetInfo : public TargetInfo {
protected:
  bool isSupportedWarpReduceOp(Operation *op,
                               unsigned warpSize) const final;
  Value genWarpReduce(RewriterBase &rewriter, Location loc, Value acc,
                      Operation *reduceOp, unsigned activeLanes,
                      unsigned warpSize) const final;
};
} // namespace mlir::triton::intel

#endif // TRITON_CONVERSION_TRITONGPU_TO_LLVM_SPIRVTARGETINFO_H
