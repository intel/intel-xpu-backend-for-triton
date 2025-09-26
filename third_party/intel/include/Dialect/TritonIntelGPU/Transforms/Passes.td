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

def TritonIntelGPUOptimizeDotOperands
    : Pass<"tritonintelgpu-optimize-dot-operands", "mlir::ModuleOp"> {
  let summary = "Intel optimize dot operands";

  let description = [{
    Re-arranged layouts of tensors used as matrix multiplication operands to
    promote the use of hardware-accelerated operations.
  }];

  let dependentDialects = ["mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect"];
}

def TritonIntelGPUCoalesce
    : Pass<"tritonintelgpu-coalesce", "mlir::ModuleOp"> {
  let summary = "Intel Coalesce";

  let description = [{
    The pass analyses loads/stores with type `tensor<tt.ptr<>>` or
    `tt.ptr<tensor<>>` and replaces the layouts of these operations with
    coalesced layouts, i.e. cache friendly access patterns.
    Layout conversions are inserted before and after the load/store op
    to maintain consistency with the rest of the program.
  }];

  let dependentDialects = ["mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect"];
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
                           "mlir::spirv::SPIRVDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::gpu::intel::TritonIntelGPUDialect"];

  let options = [
    Option<"numStages", "num-stages",
           "int32_t", /*default*/"3",
           "number of pipeline stages">,
    Option<"splitBarrierScope", "split-barriers-scope",
           "enum SplitBarrierScope", "SplitBarrierScope::None",
           "insert split barriers in a loop",
           "llvm::cl::values("
           "clEnumValN(SplitBarrierScope::None, \"none\", \"No scope\"), "
           "clEnumValN(SplitBarrierScope::Workgroup, \"workgroup\", \"Workgroup scope\"), "
           "clEnumValN(SplitBarrierScope::Subgroup, \"subgroup\", \"Subgroup scope\"))">,
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
    #blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [1, 0]}>
    #blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
    #dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4], A = [8, 16], B = [16, 16], C = [8, 16]}>
    ...
    %28 = tt.load %arg11 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x32xf16, #blocked>>
    %29 = tt.load %arg12 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<32x256xf16, #blocked1>>
    %30 = ttg.convert_layout %28 : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas}>>
    %31 = ttg.convert_layout %29 : tensor<32x256xf16, #blocked1> -> tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas}>>
    %32 = tt.dot %30, %31, %arg10, inputPrecision = tf32 : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas}>> * tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas}>> -> tensor<64x256xf32, #dpas>
    ```

    After this pass, the convert layout ops is removed which deviates from the
    common TTGIR remove layout conversion.
    Like this:

    ```mlir
    #dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4], A = [8, 16], B = [16, 16], C = [8, 16]}>
    ...
    %28 = tt.load %arg11 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas}>>>
    %29 = tt.load %arg12 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas}>>
    %32 = tt.dot %28, %29, %arg10, inputPrecision = tf32 : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas}>> * tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas}>> -> tensor<64x256xf32, #dpas>
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
    `ttg.convert_layout` operations, e.g.:
    ```mlir
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [2, 2]}>
tt.func @test(%arg0: tensor<16x32xf32, #mma>) -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #mma}>> {
  %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
  ^bb0(%arg1: f32, %arg2: f32):
    %1 = arith.addf %arg1, %arg2 : f32
    tt.reduce.return %1 : f32
  }) : (tensor<16x32xf32, #mma>) -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #mma}>>
  tt.return %0 : tensor<16xf32, #ttg.slice<{dim = 1, parent = #mma}>>
}
    ```
    Is converted to:
    ```mlir
#blocked = #ttg.blocked<{sizePerThread = [1, 8, 2, 2, 1, 1, 1], threadsPerWarp = [16, 1, 1, 1, 1, 1, 1], warpsPerCTA = [1, 1, 1, 1, 1, 1, 1], order = [0, 1, 2, 3, 4, 5, 6]}>
#blocked1 = #ttg.blocked<{sizePerThread = [16, 1, 1, 1, 1], threadsPerWarp = [1, 8, 2, 1, 1], warpsPerCTA = [1, 1, 1, 1, 1], order = [0, 1, 2, 3, 4]}>
#blocked2 = #ttg.blocked<{sizePerThread = [16, 1, 1, 1], threadsPerWarp = [1, 16, 1, 1], warpsPerCTA = [1, 1, 1, 1], order = [0, 1, 2, 3]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [0, 1]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [2, 2], A = [16, 16], B = [16, 32], C = [16, 32]}>
tt.func @test(%arg0: tensor<16x32xf32, #mma>) -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #mma}>> {
  %0 = tt.reshape %arg0 allow_reorder efficient_layout : tensor<16x32xf32, #mma> -> tensor<16x8x2x2x1x1x1xf32, #blocked>
  %1 = "tt.reduce"(%0) <{axis = 2 : i32}> ({
  ^bb0(%arg1: f32, %arg2: f32):
    %9 = arith.addf %arg1, %arg2 : f32
    tt.reduce.return %9 : f32
  }) : (tensor<16x8x2x2x1x1x1xf32, #blocked>) -> tensor<16x8x2x1x1x1xf32, #ttg.slice<{dim = 2, parent = #blocked}>>
  %2 = "tt.reduce"(%1) <{axis = 4 : i32}> ({
  ^bb0(%arg1: f32, %arg2: f32):
    %9 = arith.addf %arg1, %arg2 : f32
    tt.reduce.return %9 : f32
  }) : (tensor<16x8x2x1x1x1xf32, #ttg.slice<{dim = 2, parent = #blocked}>>) -> tensor<16x8x2x1x1xf32, #ttg.slice<{dim = 4, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>
  %3 = ttg.convert_layout %2 : tensor<16x8x2x1x1xf32, #ttg.slice<{dim = 4, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>> -> tensor<16x8x2x1x1xf32, #blocked1>
  %4 = tt.reshape %3 allow_reorder efficient_layout : tensor<16x8x2x1x1xf32, #blocked1> -> tensor<16x16x1x1xf32, #blocked2>
  %5 = "tt.reduce"(%4) <{axis = 0 : i32}> ({
  ^bb0(%arg1: f32, %arg2: f32):
    %9 = arith.addf %arg1, %arg2 : f32
    tt.reduce.return %9 : f32
  }) : (tensor<16x16x1x1xf32, #blocked2>) -> tensor<16x1x1xf32, #ttg.slice<{dim = 0, parent = #blocked2}>>
  %6 = "tt.reduce"(%5) <{axis = 1 : i32}> ({
  ^bb0(%arg1: f32, %arg2: f32):
    %9 = arith.addf %arg1, %arg2 : f32
    tt.reduce.return %9 : f32
  }) : (tensor<16x1x1xf32, #ttg.slice<{dim = 0, parent = #blocked2}>>) -> tensor<16x1xf32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 0, parent = #blocked2}>}>>
  %7 = tt.reshape %6 allow_reorder efficient_layout : tensor<16x1xf32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 0, parent = #blocked2}>}>> -> tensor<16xf32, #ttg.slice<{dim = 0, parent = #blocked3}>>
  %8 = ttg.convert_layout %7 : tensor<16xf32, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #mma}>>
  tt.return %8 : tensor<16xf32, #ttg.slice<{dim = 1, parent = #mma}>>
}
    ```
    The `tt.reshape` operation is a NOP so that the following `tt.reduce`
    operation performs a reduction within the work-item. Then,
    `ttg.convert_layout` performs the actual data movement to prevent
    within the sub-group reductions so that the following `tt.reduce` performs
    a reduction within the work-item again. Finally, we convert back to the
    original type. Note the order of the operations to go back to the original
    type is important: reshape to original shape and set an anchor for the
    layout conversion removal pass, and convert to original layout.

    Note this pass only supports `ttig.dpas` input layouts at the
    moment, but it should be easily extended.

    See pass implementation for more detailed implementation documentation.
  }];

  let dependentDialects = ["mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect"];
}

def TritonIntelGPURewriteStackPtr
    : Pass<"tritonintelgpu-rewrite-stack-ptr", "mlir::ModuleOp"> {
  let summary = "rewrite the getStackPointer for Intel by addressofOp replacement";

  let description = [{
    This pass searches for the global_smem symbol and replaces the addressOfOp with a newly inserted
    SLM parameter or a PoisonOp to rewrite the getStackPointer for Intel.
  }];

  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::gpu::intel::TritonIntelGPUDialect", "mlir::scf::SCFDialect",
    "mlir::arith::ArithDialect"
  ];
}

def TritonIntelGPUReduceVariableLiveness
    : Pass<"tritonintelgpu-reduce-variable-liveness", "mlir::ModuleOp"> {
  let summary = "Attempt to reduce the variable liveness";

  let description = [{
    This pass attempts to reduce the variable liveness
    by reducing the distance between loads and usage.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::arith::ArithDialect"];
}
#endif // TRITON_INTEL_GPU_PASSES
