// RUN: triton-opt %s --tritonintelgpu-lower-to-2d-block-load -verify-diagnostics

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  tt.func @descriptor_load_misaligned_pitch_f16(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32) -> tensor<64x32xf16, #dot0> {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %stride = arith.constant 36 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%stride, %c1_i64] : <f16>, <64x32xf16>
    // expected-error @+1 {{descriptor pitch of 72 bytes is not a multiple of 16 bytes}}
    %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #dot0>
    tt.return %0 : tensor<64x32xf16, #dot0>
  }
}
