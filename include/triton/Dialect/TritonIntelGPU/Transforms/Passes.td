//===-- Passes.td - TritonIntelGPU pass definition file ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_INTEL_GPU_PASSES
#define TRITON_INTEL_GPU_PASSES

include "mlir/Pass/PassBase.td"

def TritonIntelGPUAccelerateMatmul
    : Pass<"tritonintelgpu-accelerate-matmul", "mlir::ModuleOp"> {
  let summary = "intel accelerate matmul";

  let description = [{
    Optimize the input/output layout of the `tl.dot` operation to make them
    compatible with the Intel DPAS instruction requirements.
  }];

  let constructor = "mlir::createTritonIntelGPUAccelerateMatmulPass()";

  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::gpu::intel::TritonIntelGPUDialect",
    "mlir::triton::TritonDialect"
  ];

  let options = [
    Option<"deviceArch", "device-architecture",
            "mlir::triton::gpu::intel::DeviceArch", /*default*/" mlir::triton::gpu::intel::DeviceArch::PVC",
            "device architecture",
            "llvm::cl::values("
            "clEnumValN(mlir::triton::gpu::intel::DeviceArch::UNKNOWN, \"UNKNOWN\", \"Unknown arch\"), "
            "clEnumValN(mlir::triton::gpu::intel::DeviceArch::ATS, \"ATS\", \"ATS arch\"), "
            "clEnumValN(mlir::triton::gpu::intel::DeviceArch::PVC, \"PVC\", \"PVC arch\"))">
  ];
}

#endif // TRITON_INTEL_GPU_PASSES
