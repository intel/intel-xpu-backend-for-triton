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

  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::intel::TritonIntelGPUDialect",
    "mlir::arith::ArithDialect"
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

  let dependentDialects = ["mlir::triton::TritonDialect",
                           "mlir::triton::gpu::intel::TritonIntelGPUDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::gpu::GPUDialect"];
}

def TritonIntelGPUPipeline : Pass<"tritonintelgpu-pipeline", "mlir::ModuleOp"> {
  let summary = "Pipeline loops";

  let description = [{
    Apply software pipelinining to loops containing `tt.dot` operations.
    The pass supports prefetching `tt.dot` operands. The `num-stages` argument controls
    the prefetching and distance (i.e. the number of iterations to prefetch in advance).
  }];

  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::gpu::intel::TritonIntelGPUDialect"];

  let options = [
    Option<"numStages", "num-stages",
           "int32_t", /*default*/"3",
           "number of pipeline stages">,
    Option<"supportRegularPtr", "support-regular-ptr",
           "bool", /*default*/"false",
           "Enable support for prefetching non-block pointers">,
  ];
}

def TritonIntelGPURemoveLayoutConversions : Pass<"tritonintelgpu-remove-layout-conversions", "mlir::ModuleOp"> {
  let summary = "remove superfluous layout conversions";

  let description = [{
    This is a customized remove layout conversion for Intel GPU arch.
    Different GPUs characteristics make it profitable for Intel HW to load the
    operands of a `tt.dot` operation into registers.
    Therefore given the following example:

    ```mlir
    #blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [1, 0]}>
    #blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
    #dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4], A = [8, 16], B = [16, 16], C = [8, 16]}>
    ...
    %28 = tt.load %arg11 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x32xf16, #blocked>>
    %29 = tt.load %arg12 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<32x256xf16, #blocked1>>
    %30 = triton_gpu.convert_layout %28 : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas}>>
    %31 = triton_gpu.convert_layout %29 : tensor<32x256xf16, #blocked1> -> tensor<32x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #dpas}>>
    %32 = tt.dot %30, %31, %arg10, inputPrecision = tf32 : tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas}>> * tensor<32x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #dpas}>> -> tensor<64x256xf32, #dpas>
    ```

    After this pass, the convert layout ops is removed which deviates from the
    common TTGIR remove layout conversion.
    Like this:

    ```mlir
    #dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4], A = [8, 16], B = [16, 16], C = [8, 16]}>
    ...
    %28 = tt.load %arg11 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas}>>>
    %29 = tt.load %arg12 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<32x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #dpas}>>
    %32 = tt.dot %28, %29, %arg10, inputPrecision = tf32 : tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas}>> * tensor<32x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #dpas}>> -> tensor<64x256xf32, #dpas>
    ```

    On NVidia GPUs it is profitable to load the operands of a tt.dot operation
    into shared local memory (therefore the layout conversion operations are
    necessary). On Intel GPUs loading into SLM is not profitable because it
    would require synchronization operations that are expensive. Therefore it
    is better to load the operands directly into registers and incur the cost
    of duplicating the load, because the HW can combine redundant memory
    accesses in the IO buffer or cache them.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];

}

def TritonIntelGPURewriteTensorPointer : Pass<"tritonintelgpu-rewrite-tensor-pointer", "mlir::ModuleOp"> {
  let summary = "Rewrite load/store operations using tensor pointers that cannot be lowered to 2D Block Load/Store intrinsics";
  let description = [{
    This pass determines whether a load/store operation can be lowered to 2D
    Block Load/Store intrinsic. If it cannot, it replaces the load/store
    operation with a legacy pointer and removes the Triton operations that
    create and advance the block pointer (that is `tt.make_tensor_tr` and
    `tt.advance`).
  }];

  let dependentDialects = ["mlir::triton::TritonDialect"];
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

  let dependentDialects = ["mlir::triton::TritonDialect",
                           "mlir::triton::TritonGEN::TritonGENDialect",
                           "mlir::triton::gpu::intel::TritonIntelGPUDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::gpu::GPUDialect"];
  let options = [
    Option<"numAdvancePrefetches", "num-advance-prefetches",
           "int32_t", /*default*/"3",
           "Number of loop iteration to prefetch in advance of the loop">,
    Option<"injectSplitBarriers", "inject-split-barriers",
           "bool", /*default*/"true",
           "Whether to inject split barriers in (and around) the loop">,
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

  let dependentDialects = ["mlir::triton::TritonDialect",
                           "mlir::triton::gpu::intel::TritonIntelGPUDialect"];
}

def TritonIntelGPUReduceDataDuplication: Pass<"tritonintelgpu-reduce-data-duplication", "mlir::ModuleOp"> {
  let summary = "Reduce data duplication in register by decomposing convert[distributed -> dotOperand] "
                "into convert[distributed -> shared -> dotOperand]";

  let description = [{
    Decomposing conversions this way makes it possible to use CSE and reuse #shared tensors.
    This Intel pass supports the Intel DPAS layout in additional to the upstream Triton pass.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::gpu::intel::TritonIntelGPUDialect",
                           "mlir::triton::TritonDialect"];
}

def TritonIntelGPUScheduleLoad : Pass<"tritonintelgpu-schedule-load", "mlir::ModuleOp"> {
  let summary = "naive ra-aware instr scheduler";

  let description = [{
    This pass works for FlashAttention.
    This pass moves loads to be adjacent to their user(tt.dot) to help IGC do better RegisterAllocation.
  }];

  let dependentDialects = ["mlir::triton::TritonDialect",
                           "mlir::triton::gpu::intel::TritonIntelGPUDialect",
                           "mlir::triton::gpu::TritonGPUDialect"];
}

def TritonIntelGPUMaterializeBlockPointer : Pass<"tritonintelgpu-materialize-block-pointer", "mlir::ModuleOp"> {
  let summary = "annotate load operations with information required to exploit 2D block HW instructions";

  let description = [{
    This pass annotates load operations using a block pointer with information required to exploit 2D
    block HW instructions during lowering (e.g. whether the memory access pattern is row or column major).
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::gpu::intel::TritonIntelGPUDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::arith::ArithDialect"];
}

def TritonIntelGPUOptimizeReductionLocality
    : Pass<"tritonintelgpu-optimize-reduction-locality", "mlir::ModuleOp"> {
  let summary = "Minimize number of reductions within sub-groups";

  let description = [{
    This pass performs layout conversions so `tt.reduce` operations resulting in
    sub-group reductions are converted to `tt.reshape`, `tt.reduce`, and
    `triton_gpu.convert_layout` operations, e.g.:
    ```mlir
#mma = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [1, 1]}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "xpu", "triton_gpu.threads-per-warp" = 16 : i32, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block} {
  tt.func @test.work(%arg0: tensor<32x32xf32, #mma>) -> tensor<32xf32, #triton_gpu.slice<{dim = 0, parent = #mma}>> {
    %0 = "tt.reduce"(%arg0) <{axis = 0 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      tt.reduce.return %1 : f32
    }) : (tensor<32x32xf32, #mma>) -> tensor<32xf32, #triton_gpu.slice<{dim = 0, parent = #mma}>>
    tt.return %0 : tensor<32xf32, #triton_gpu.slice<{dim = 0, parent = #mma}>>
  }
}
    ```
    Is converted to:
    ```mlir
#blocked = #triton_gpu.blocked<{sizePerThread = [8, 1, 1, 1, 1], threadsPerWarp = [1, 16, 1, 1, 1], warpsPerCTA = [2, 1, 1, 2, 1], order = [4, 0, 1, 2, 3]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 16, 1], threadsPerWarp = [16, 1, 1], warpsPerCTA = [2, 1, 2], order = [2, 0, 1]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [2, 2], order = [1, 0]}>
#mma = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [1, 1], A = [8, 8], B = [8, 16], C = [8, 16]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "xpu", "triton_gpu.threads-per-warp" = 16 : i32, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block} {
  tt.func @test_two_warps_twice(%arg0: tensor<32x32xf32, #mma>) -> tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>> {
    %0 = tt.reshape %arg0 {allow_reorder = true} : tensor<32x32xf32, #mma> -> tensor<32x16x1x2x1xf32, #blocked>
    %1 = "tt.reduce"(%0) <{axis = 4 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %7 = arith.addf %arg1, %arg2 : f32
      tt.reduce.return %7 : f32
    }) : (tensor<32x16x1x2x1xf32, #blocked>) -> tensor<32x16x1x2xf32, #triton_gpu.slice<{dim = 4, parent = #blocked}>>
    %2 = "tt.reduce"(%1) <{axis = 2 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %7 = arith.addf %arg1, %arg2 : f32
      tt.reduce.return %7 : f32
    }) : (tensor<32x16x1x2xf32, #triton_gpu.slice<{dim = 4, parent = #blocked}>>) -> tensor<32x16x2xf32, #triton_gpu.slice<{dim = 2, parent = #triton_gpu.slice<{dim = 4, parent = #blocked}>}>>
    %3 = triton_gpu.convert_layout %2 : tensor<32x16x2xf32, #triton_gpu.slice<{dim = 2, parent = #triton_gpu.slice<{dim = 4, parent = #blocked}>}>> -> tensor<32x16x2xf32, #blocked1>
    %4 = tt.reshape %3 {allow_reorder = true} : tensor<32x16x2xf32, #blocked1> -> tensor<32x32xf32, #blocked2>
    %5 = "tt.reduce"(%4) <{axis = 1 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %7 = arith.addf %arg1, %arg2 : f32
      tt.reduce.return %7 : f32
    }) : (tensor<32x32xf32, #blocked2>) -> tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %6 = triton_gpu.convert_layout %5 : tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> -> tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    tt.return %6 : tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
  }
}
    ```
    The `tt.reshape` operation is a NOP so that the following `tt.reduce`
    operation performs a reduction within the work-item. Then,
    `triton_gpu.convert_layout` performs the actual data movement to prevent
    within the sub-group reductions so that the following `tt.reduce` performs
    a reduction within the work-item again. Finally, we convert back to the
    expected layout.

    Note this pass only supports `triton_intel_gpu.dpas` input layouts at the
    moment, but it should be easily extended.
  }];

  let dependentDialects = ["mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect"];
}

#endif // TRITON_INTEL_GPU_PASSES
