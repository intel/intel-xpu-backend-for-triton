// RUN: triton-opt %s -split-input-file -tritonintelgpu-pipeline=num-stages=3 | FileCheck %s --check-prefix=DEFAULT
// RUN: env TRITON_INTEL_ANNOTATE_LATENCIES=1 triton-opt %s -split-input-file -tritongpu-assign-latencies=num-stages=3 -tritongpu-schedule-loops -tritonintelgpu-pipeline=num-stages=3 | FileCheck %s --check-prefix=ANNOTATED

// COM: Verifies that Intel's matmul pipeliner produces structurally identical
// COM: results whether or not upstream annotation passes run first. The test
// COM: uses a rank-3 GEMM with descriptor_load ops (matching the existing
// COM: upstream-latency-annotation.mlir test) and asserts:
// COM:   - Same prefetch op count (4 hoisted, 2 in-loop)
// COM:   - Same scf.for iter-arg count
// COM:   - Same in-loop op ordering (prefetch → load → dot)
// COM:   - DEFAULT: Prefetch ops do NOT carry loop.stage/loop.cluster attributes
// COM:   - ANNOTATED: Prefetch ops DO carry loop.stage=0 and loop.cluster (copied from source load)
// COM:   - ANNOTATED: Load ops DO carry loop.stage/loop.cluster (set by upstream)

#blocked3d = #ttg.blocked<{sizePerThread = [1, 1, 8], threadsPerWarp = [1, 4, 4], warpsPerCTA = [1, 2, 4], order = [2, 1, 0]}>
#blocked2d = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 4], warpsPerCTA = [2, 4], order = [1, 0]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [2, 4], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>

module attributes {ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_2d_block_io, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
  tt.func public @gemm_pipeline_structural_equivalence(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32, %arg6: i32 {tt.divisibility = 16 : i32}) {
    // DEFAULT-LABEL: tt.func public @gemm_pipeline_structural_equivalence
    // ANNOTATED-LABEL: tt.func public @gemm_pipeline_structural_equivalence
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %0 = arith.extsi %arg6 : i32 to i64

    %descA = tt.make_tensor_descriptor %arg0, [%arg3, %arg3, %arg6], [%0, %0, %c1_i64] : <f16>, <1x128x64xf16>
    %descB = tt.make_tensor_descriptor %arg1, [%arg3, %arg6, %arg6], [%0, %0, %c1_i64] : <f16>, <1x64x256xf16>

    // COM: Structural invariants verified by both DEFAULT and ANNOTATED prefixes:
    // COM:   - 4 hoisted prefetches before loop (2 per stage × 2 stages)
    // COM:   - 2 prefetches inside loop body
    // COM:   - 2 loads inside loop body
    // COM:   - 1 dot inside loop body
    // COM:   - Loop has 3 iter-args (acc, next_koff, current_koff)

    // COM: DEFAULT path: verify op counts and ordering. Without annotations,
    // COM: prefetches do NOT carry loop.stage/loop.cluster.
    // DEFAULT-COUNT-4: ttig.descriptor_prefetch
    // DEFAULT-NOT: ttig.descriptor_prefetch{{.*}}loop.stage
    // DEFAULT-NOT: ttig.descriptor_prefetch{{.*}}loop.cluster
    // DEFAULT: scf.for {{.*}} iter_args({{.*}}) -> (tensor<128x256xf32, #{{.*}}>, i32, i32)
    // DEFAULT-COUNT-2: ttig.descriptor_prefetch
    // DEFAULT-NOT: ttig.descriptor_prefetch{{.*}}loop.stage
    // DEFAULT-NOT: ttig.descriptor_prefetch{{.*}}loop.cluster
    // DEFAULT: tt.descriptor_load
    // DEFAULT: tt.descriptor_load
    // DEFAULT: tt.reshape
    // DEFAULT: ttg.convert_layout
    // DEFAULT: tt.reshape
    // DEFAULT: ttg.convert_layout
    // DEFAULT: tt.dot
    // COM: Catch extra prefetches beyond the expected 4+2 (legacy hardcoded).
    // DEFAULT-NOT: ttig.descriptor_prefetch

    // COM: ANNOTATED path: verify op counts and ordering. Hoisted and in-loop
    // COM: prefetches carry loop.cluster and loop.stage = 0.
    // ANNOTATED-COUNT-4: ttig.descriptor_prefetch {{.*}}loop.cluster = {{[0-9]+}} : i32, loop.stage = 0 : i32
    // ANNOTATED: scf.for {{.*}} iter_args({{.*}}) -> (tensor<128x256xf32, #{{.*}}>, i32, i32)
    // ANNOTATED-COUNT-2: ttig.descriptor_prefetch {{.*}}loop.cluster = {{[0-9]+}} : i32, loop.stage = 0 : i32
    // ANNOTATED: tt.descriptor_load {{.*}}loop.stage = {{[0-9]+}} : i32{{.*}}ttig.block_io
    // ANNOTATED: tt.descriptor_load {{.*}}loop.stage = {{[0-9]+}} : i32{{.*}}ttig.block_io
    // ANNOTATED: tt.reshape {{.*}}loop.stage
    // ANNOTATED: ttg.convert_layout {{.*}}loop.stage
    // ANNOTATED: tt.reshape {{.*}}loop.stage
    // ANNOTATED: ttg.convert_layout {{.*}}loop.stage
    // ANNOTATED: tt.dot {{.*}}loop.stage
    // COM: Catch extra prefetches beyond the expected 4+2 (annotation-driven).
    // ANNOTATED-NOT: ttig.descriptor_prefetch

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
