// RUN: triton-opt %s -split-input-file --tritonintelgpu-materialize-block-pointer --tritonintelgpu-lower-to-2d-block-load | FileCheck %s

// Runs the two passes in pipeline order (MaterializeBlockPointer tags, then
// LowerTo2DBlockLoad lowers; the real pipeline runs many other passes between
// them) on UNTAGGED IR: no load carries a hand-written ttig.block_io
// attribute. This proves that MaterializeBlockPointer tags a descriptor load
// whose height is the compile-time constant 0, and that LowerTo2DBlockLoad
// then keeps it as tt.descriptor_load (the gather path) instead of emitting a
// 2D block load (intel/intel-xpu-backend-for-triton#8071). The hand-tagged
// bounds cases live in descriptor-load-bigstride.mlir.

// Case 1 (control): height 4096, width 4096 f16, pitch 64 f16 = 128 B, all in range; the pipeline reaches the 2D block load.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @descriptor_load_pipeline_control_f16
  tt.func @descriptor_load_pipeline_control_f16(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<64x32xf16, #dot0> {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %sh0 = arith.constant 4096 : i32
    %sh1 = arith.constant 4096 : i32
    %st0 = arith.constant 64 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%sh0, %sh1], [%st0, %c1_i64] : <f16>, <64x32xf16>
    // CHECK: ttig.2d_block_load
    %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #dot0>
    tt.return %0 : tensor<64x32xf16, #dot0>
  }
}

// -----

// Case 2: Case 1 with the height folded to 0; the load must be tagged by MaterializeBlockPointer yet kept by LowerTo2DBlockLoad.
// The ttig.block_io tag on the surviving load is the oracle: it shows the bail comes from LowerTo2DBlockLoad, not a missing tag.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @descriptor_load_pipeline_height_zero_f16
  tt.func @descriptor_load_pipeline_height_zero_f16(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<64x32xf16, #dot0> {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %sh0 = arith.constant 0 : i32
    %sh1 = arith.constant 4096 : i32
    %st0 = arith.constant 64 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%sh0, %sh1], [%st0, %c1_i64] : <f16>, <64x32xf16>
    // CHECK-NOT: ttig.2d_block_load
    // CHECK-NOT: ttig.extract_desc
    // CHECK: tt.descriptor_load {{.*}}ttig.block_io = "row_major"
    %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #dot0>
    tt.return %0 : tensor<64x32xf16, #dot0>
  }
}
