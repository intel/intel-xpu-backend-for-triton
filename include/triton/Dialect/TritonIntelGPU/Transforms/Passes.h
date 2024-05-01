//===- Passes.h - Intel Pass Construction and Registration ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_DIALECT_TRITON_INTEL_GPU_TRANSFORMS_PASSES_H
#define TRITON_DIALECT_TRITON_INTEL_GPU_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include "triton/Dialect/TritonGEN/IR/TritonGENDialect.h"

namespace mlir {
namespace triton {
namespace gpu {
namespace intel {

enum class DeviceArch {
  UNKNOWN = 0,
  ATS,
  PVC,
};

std::unique_ptr<Pass> createTritonIntelGPUAccelerateMatmulPass(
    intel::DeviceArch arch = intel::DeviceArch::UNKNOWN);

std::unique_ptr<Pass> createTritonIntelGPUDistributeToWarpsPass();

std::unique_ptr<Pass> createTritonIntelGPUPipelinePass(
    int numStages = 3, intel::DeviceArch arch = intel::DeviceArch::UNKNOWN);

std::unique_ptr<Pass> createTritonIntelGPURemoveLayoutConversionsPass();

std::unique_ptr<Pass> createTritonIntelGPURewriteTensorPointerPass(
    intel::DeviceArch arch = intel::DeviceArch::UNKNOWN);

std::unique_ptr<Pass> createPrefetchBlockPass();

std::unique_ptr<Pass> createMatchTargetSizePass();

} // namespace intel
} // namespace gpu
} // namespace triton

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "triton/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"

} // namespace mlir

#endif // TRITON_DIALECT_TRITON_INTEL_GPU_TRANSFORMS_PASSES_H
