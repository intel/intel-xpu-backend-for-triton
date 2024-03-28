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

def TritonIntelGPUPrefetchBlock : Pass<"tritonintelgpu-prefetch-block", "mlir::ModuleOp"> {
  let summary = "prefetch a tensor block around loop";
  let description = [{
    This pass match pattern of certain scf.loop with tt.load and then
    add prefetch both in the loop preheader(3 stages in advance) and loop body.
    This pass only support block pointer.
    This pass only support target that has a dedicated prefetch instruction.
  }];
  let constructor = "mlir::triton::gpu::intel::createPrefetchBlockPass()";
  let dependentDialects = ["mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::gpu::intel::TritonIntelGPUDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::gpu::GPUDialect"];
  let options = [
    Option<"numWarps", "num-warps",
           "unsigned", /*default*/"4",
           "number of warps per block">];
}

def TritonIntelGPUMatchTargetSize : Pass<"tritonintelgpu-match-target-size", "mlir::ModuleOp"> {
  let summary = "match the target size of specific op (dot, load, store)";

  let description = [{
    this pass should be run after tritongpu-distribute-to-warps
  }];

  let constructor = "mlir::triton::gpu::intel::createMatchTargetSizePass()";

  let dependentDialects = ["mlir::triton::TritonDialect",
                           "mlir::triton::gpu::intel::TritonIntelGPUDialect",
                           "mlir::triton::gpu::TritonGPUDialect"];

  let options = [
    ListOption<"dotSize", "dot-size",
           "int64_t", "target LLVM IR dot operation size(m, n, k)">
    // fixme : add operate-size load-size, store-size
    // load-operate-store-size 256DW which is 16x16xf32, 32x16xf16
  ];
}

def TritonIntelGPUPrepareGenxLsc : Pass<"tritonintelgpu-prepare-genxlsc", "mlir::ModuleOp"> {
  let summary = "tweak load operation to i16/i32";

  let description = [{
  }];

  let constructor = "mlir::triton::gpu::intel::createPrepareGenxLscPass()";

  let dependentDialects = ["mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::gpu::intel::TritonIntelGPUDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::arith::ArithDialect"];
}

#endif // TRITON_INTEL_GPU_PASSES
