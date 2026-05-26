// RUN: triton-opt %s -split-input-file -tritongpu-assign-latencies=num-stages=3 -tritongpu-schedule-loops | FileCheck %s --check-prefix=ANNOTATE
// RUN: triton-opt %s -split-input-file -tritongpu-assign-latencies=num-stages=3 -tritongpu-schedule-loops -tritonintelgpu-pipeline=num-stages=3 | FileCheck %s --check-prefix=PIPELINE

// COM: Verifies that the upstream `tritongpu-assign-latencies` and
// COM: `tritongpu-schedule-loops` passes annotate Intel TTGIR (ANNOTATE prefix)
// COM: and that Intel's pipeline pass produces a structurally equivalent result
// COM: when fed the annotated IR (PIPELINE prefix). The kernel uses
// COM: `tt.descriptor_load` because the upstream `AssignLoadLatencies` skips
// COM: regular `tt.load` ops whose effective access width is < 32 bits, which
// COM: is the case for f16 loads through a blocked layout with sizePerThread=1.

#blocked3d = #ttg.blocked<{sizePerThread = [1, 1, 8], threadsPerWarp = [1, 4, 4], warpsPerCTA = [1, 2, 4], order = [2, 1, 0]}>
#blocked2d = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 4], warpsPerCTA = [2, 4], order = [1, 0]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [2, 4], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>

module attributes {ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_2d_block_io, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
  tt.func public @rank3_descriptor_pipeline(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32, %arg6: i32 {tt.divisibility = 16 : i32}) {
    // ANNOTATE-LABEL: tt.func public @rank3_descriptor_pipeline
    // PIPELINE-LABEL: tt.func public @rank3_descriptor_pipeline
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %0 = arith.extsi %arg6 : i32 to i64

    %descA = tt.make_tensor_descriptor %arg0, [%arg3, %arg3, %arg6], [%0, %0, %c1_i64] : <f16>, <1x128x64xf16>
    %descB = tt.make_tensor_descriptor %arg1, [%arg3, %arg6, %arg6], [%0, %0, %c1_i64] : <f16>, <1x64x256xf16>

    // COM: Upstream passes attach scheduling attributes to the loop and ops.
    // ANNOTATE: scf.for
    // ANNOTATE: tt.descriptor_load {{.*}}loop.cluster = {{[0-9]+}} : i32, loop.stage = {{[0-9]+}} : i32
    // ANNOTATE: tt.descriptor_load {{.*}}loop.cluster = {{[0-9]+}} : i32, loop.stage = {{[0-9]+}} : i32
    // ANNOTATE: tt.dot
    // ANNOTATE: tt.scheduled_max_stage = {{[0-9]+}} : i32

    // COM: Intel pipeline pass produces 4 hoisted prefetches (2 per stage, 2 stages)
    // COM: and maintains prefetch-load-dot ordering inside the loop.
    // PIPELINE-COUNT-4: ttig.descriptor_prefetch
    // PIPELINE: scf.for
    // PIPELINE: ttig.descriptor_prefetch
    // PIPELINE: ttig.descriptor_prefetch
    // PIPELINE: tt.descriptor_load
    // PIPELINE: tt.descriptor_load
    // PIPELINE: tt.dot

    %result:2 = scf.for %k = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%acc = %cst, %koff = %c0_i32) -> (tensor<128x256xf32, #mma>, i32) : i32 {
      %a3d = tt.descriptor_load %descA[%c0_i32, %c0_i32, %koff] {ttig.block_io = "row_major"} : !tt.tensordesc<1x128x64xf16> -> tensor<1x128x64xf16, #blocked3d>
      %a2d = tt.reshape %a3d : tensor<1x128x64xf16, #blocked3d> -> tensor<128x64xf16, #blocked2d>
      %a = ttg.convert_layout %a2d : tensor<128x64xf16, #blocked2d> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %b3d = tt.descriptor_load %descB[%c0_i32, %koff, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<1x64x256xf16> -> tensor<1x64x256xf16, #blocked3d>
      %b2d = tt.reshape %b3d : tensor<1x64x256xf16, #blocked3d> -> tensor<64x256xf16, #blocked2d>
      %b = ttg.convert_layout %b2d : tensor<64x256xf16, #blocked2d> -> tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %d = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x256xf32, #mma>
      %next_koff = arith.addi %koff, %c1_i32 : i32
      scf.yield %d, %next_koff : tensor<128x256xf32, #mma>, i32
    }
    tt.return
  }
}
