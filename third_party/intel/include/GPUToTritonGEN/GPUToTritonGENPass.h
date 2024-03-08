//===- GPUToTritonGENPass.h - Convert GPU to TritonGEN ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_CONVERSION_GPUTOTRITONGENPASS_H
#define TRITON_CONVERSION_GPUTOTRITONGENPASS_H

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
#include "intel/include/GPUToTritonGEN/Passes.h.inc"

void populateGPUToTritonGENConversionPatterns(LLVMTypeConverter &converter,
                                              RewritePatternSet &patterns);

void configureGPUToTritonGENConversionLegality(ConversionTarget &target);

std::unique_ptr<OperationPass<mlir::gpu::GPUModuleOp>>
createLowerGPUToTritonGENPass(unsigned indexBitwidth);

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_GPUTOTRITONGENPASS_H
