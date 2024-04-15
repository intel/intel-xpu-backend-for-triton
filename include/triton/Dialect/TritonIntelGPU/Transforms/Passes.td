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
  let summary = "Intel accelerate matmul";

  let description = [{
    Optimize the input/output layout of the `tl.dot` operation to make them
    compatible with the Intel DPAS instruction requirements.
  }];

  let constructor = "mlir::triton::gpu::intel::createTritonIntelGPUAccelerateMatmulPass()";

  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::intel::TritonIntelGPUDialect",
    "mlir::arith::ArithDialect"
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

def TritonIntelGPUDistributeToWarps
    : Pass<"tritonintelgpu-distribute-to-warps", "mlir::ModuleOp"> {
  let summary = "distribute the thread block workload to the warps";

  let description = [{
    Changes the tensor type and layout

    For example, given:

    ```mlir
    #blockedC = #triton_gpu.blocked<{sizePerThread = [64, 64], threadsPerWarp = [1, 1], warpsPerCTA = [2, 2], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
    #blockedA = #triton_gpu.dot_op<{opIdx = 0, parent = #blockedC}>
    #blockedB = #triton_gpu.dot_op<{opIdx = 1, parent = #blockedC}>

    tt.func @gemm(%arg0, %arg1, %arg2) {
      %0 = tt.get_program_id x: i32
      ...
      %18 = tt.make_tensor_ptr %arg0, [affine function of %0] : <tensor<128x32xf16, #blockedA>, 1>
      %22 = tt.make_tensor_ptr %arg1, [affine function of %0] : <tensor<32x128xf16, #blockedB>, 1>
      scf.for loop {
        %28 = tt.load %18 : <tensor<128x32xf16, #blockedA>, 1>
        %29 = tt.load %22 : <tensor<32x128xf16, #blockedB>, 1>
        %30 = tt.dot %28, %29, %acc : <tensor<128x32xf16, #blockedA>, <tensor<32x128xf16, #blockedB> -> tensor<128x128xf32, #blockedC>
        ...
      }
      ...
    }
    ```

    after this pass, the workload is distributed so that each op works on warp/subgroup level
    with smaller size:

    ```mlir
    #warpA = #triton_gpu.warp<{sizePerThread = [64, 32], threadsPerWarp = [1, 1], order = [1, 0]}>
    #warpB = #triton_gpu.warp<{sizePerThread = [32, 64], threadsPerWarp = [1, 1], order = [1, 0]}>
    #warpC = #triton_gpu.warp<{sizePerThread = [64, 64], threadsPerWarp = [1, 1], order = [1, 0]}>

    tt.func @gemm(%arg0, %arg1, %arg2) {
      %0 = tt.get_program_id x: i32
      %1 = gpu.subgroup_id : i32
      ...
      %18 = tt.make_tensor_ptr %arg0, [affine function of (%0, %1) ] : <tensor<64x32xf16, #warpA>, 1>
      %22 = tt.make_tensor_ptr %arg1, [affine function of (%0, %1) ] : <tensor<32x64xf16, #warpB>, 1>
      scf.for loop {
        %28 = tt.load %18 : <tensor<64x32xf16, #warpA>, 1>
        %29 = tt.load %22 : <tensor<32x64xf16, #warpB>, 1>
        %30 = tt.dot %28, %29, %acc : <tensor<64x32xf16, #warpA>, <tensor<32x64xf16, #warpB> -> tensor<64x64xf32, #warpC>
        ...
      }
      ...
    }
    ```
  }];

  let constructor = "mlir::triton::gpu::intel::createTritonIntelGPUDistributeToWarpsPass()";

  let dependentDialects = ["mlir::triton::TritonDialect",
                           "mlir::triton::gpu::intel::TritonIntelGPUDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::gpu::GPUDialect"];
}

def TritonIntelGPURemoveLayoutConversions : Pass<"tritonintelgpu-remove-layout-conversions", "mlir::ModuleOp"> {
  let summary = "remove superfluous layout conversions";

  let description = [{
    This is a customized remove layout conversion for Intel GPU arch.
    There are different preference in optimization when removing the layout conversions for Intel GPU which may diff to NV and AMD.

    For example, given:

    ```mlir
    #blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [1, 0]}>
    #blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
    #mma = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4], A = [8, 16], B = [16, 16], C = [8, 16]}>
    ...
    %28 = tt.load %arg11 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x32xf16, #blocked>>
    %29 = tt.load %arg12 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<32x256xf16, #blocked1>>
    %30 = triton_gpu.convert_layout %28 : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>>
    %31 = triton_gpu.convert_layout %29 : tensor<32x256xf16, #blocked1> -> tensor<32x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma}>>
    %32 = tt.dot %30, %31, %arg10, inputPrecision = tf32 : tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>> * tensor<32x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma}>> -> tensor<64x256xf32, #mma>
    ```

    After this pass, the convert layout ops is removed which is different to the common TTGIR remove layout conversions.
    Like this:

    ```mlir
    #mma = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4], A = [8, 16], B = [16, 16], C = [8, 16]}>
    ...
    %28 = tt.load %arg11 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>>>
    %29 = tt.load %arg12 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<32x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma}>>
    %32 = tt.dot %28, %29, %arg10, inputPrecision = tf32 : tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>> * tensor<32x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma}>> -> tensor<64x256xf32, #mma>
    ```

    We'd like to trade off to load the duplicated values from global memory for removing convert layout ops.
    Intel GPU prefer to load the duplicated value because the Intel memory sub-system can combine the redundant memory accessing in the IO buffer or cache system.
    To load duplicated value is lower cost than to duplicate the dot operands in share local memory which requires extra synchronization and SLM read/write.
  }];

  let constructor = "mlir::triton::gpu::intel::createTritonIntelGPURemoveLayoutConversionsPass()";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];

}

def TritonIntelGPUPrefetchBlock : Pass<"tritonintelgpu-prefetch-block", "mlir::ModuleOp"> {
  let summary = "Prefetch a tensor block around loop";

  let description = [{
    This pass injects prefetch operations for loads that 'feed' a `tt.dot` operation in a loop.
    Prefetch operations are inserted in the loop preheader (the number of iterations to prefetch
    in advance is controlable by a pass option) and in the loop body.
    Notes:
      - only loads that use a block pointer are considered
      - only targets that have a dedicated prefetch instruction are supported
  }];

  let constructor = "mlir::triton::gpu::intel::createPrefetchBlockPass()";
  let dependentDialects = ["mlir::triton::TritonDialect",
                           "mlir::triton::gpu::intel::TritonIntelGPUDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::gpu::GPUDialect"];
  let options = [
    Option<"numAdvancePrefetches", "num-advance-prefetched",
           "int32_t", /*default*/"3",
           "Number of loop iteration to prefetch in advance of the loop">,
  ];
}

def TritonIntelGPUMatchTargetSize : Pass<"tritonintelgpu-match-target-size", "mlir::ModuleOp"> {
  let summary = "Split tensor operations to match target architecture";

  let description = [{
    This pass splits certain Triton tensor operations (e.g dot, load, store) so that each
    operation matches the native operation shape supported by the target architecture.

    Notes:
      - only block pointers are supported
      - this pass should be run after 'tritonintelgpu-distribute-to-warps'

    For example, given:
      ```mlir
      %A = tt.load %arg1 : !tt.ptr<tensor<32x32xf16>> -> tensor<32x32xf16>
      %B = tt.load %arg2 : !tt.ptr<tensor<32x64xf16>> -> tensor<32x64xf16>
      %C = tt.load %arg3 : !tt.ptr<tensor<32x64xf16>> -> tensor<32x64xf32>
      %D = tt.dot %A, %B, %C : tensor<32x32xf16> * tensor<32x64xf16> -> tensor<32x64xf32>
      ```

    Assuming that the native 'dot' shape supported by the target architecture is <8x16x16>
    and that the max supported load size is 512DW (<32x32xf16>), after this pass:
      - the load of <32x64xf16> is split to 2 loads of <32x32xf16>
      - the dot operation is split so that each resulting operation matches the native target
        shape supported by the architecture

      ```mlir
      %A = tt.load %arg1 : !tt.ptr<tensor<32x32xf16>> -> tensor<32x32xf16>
      %B1 = tt.load %arg21 : !tt.ptr<tensor<32x32xf16>> -> tensor<32x32xf16>
      %B2 = tt.load %arg22 : !tt.ptr<tensor<32x32xf16>> -> tensor<32x32xf16>
      %C1 = tt.load %arg31 : !tt.ptr<tensor<32x32xf16>> -> tensor<32x32xf16>
      %C2 = tt.load %arg32 : !tt.ptr<tensor<32x32xf16>> -> tensor<32x32xf16>
      ... <extract operations>
      %dot_0 = tt.dot %tile_A, %tile_B, %tile_C : tensor<8x16xf16> * tensor<16x16xf16>
             -> tensor<8x16xf16>
      ... <extract operations>
      %dot_1 = tt.dot %tile_A', %tile_B', %dot_0 : tensor<8x16xf16> * tensor<16x16xf16>
             -> tensor<8x16xf16>
      ...
      ```
  }];

  let constructor = "mlir::triton::gpu::intel::createMatchTargetSizePass()";

  let dependentDialects = ["mlir::triton::TritonDialect",
                           "mlir::triton::gpu::intel::TritonIntelGPUDialect"];
}

#endif // TRITON_INTEL_GPU_PASSES
