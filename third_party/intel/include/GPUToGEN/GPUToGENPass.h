//===- GPUToGENPass.h - Convert GPU dialect to GEN dialect ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_CONVERSION_GPUTOGENPASS_H
#define TRITON_CONVERSION_GPUTOGENPASS_H

#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include <memory>

namespace mlir {

class LLVMTypeConverter;
class ConversionTarget;
class RewritePatternSet;

template <typename OpT> class OperationPass;

namespace gpu {
class GPUModuleOp;
} // namespace gpu

namespace triton {

#define GEN_PASS_DECL
#include "intel/include/GPUToGEN/Passes.h.inc"

void populateGPUToGENConversionPatterns(LLVMTypeConverter &converter,
                                        RewritePatternSet &patterns);

void configureGPUToGENConversionLegality(ConversionTarget &target);

std::unique_ptr<OperationPass<mlir::gpu::GPUModuleOp>>
createLowerGPUToGENPass(unsigned indexBitwidth);

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_GPUTOGENPASS_H
