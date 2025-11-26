// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -tritongpu-assign-latencies=num-stages=3 -canonicalize | FileCheck %s

// Test that ub.poison producing a ptr<tensor> gets correct rank in AxisInfo
// analysis (rank=2 for tensor<128x64>, not rank=1).
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
// CHECK-LABEL: @test_poison_rank
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func public @test_poison_rank(%arg0: !tt.ptr<f16>, %lb: i32, %ub: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1_i64 = arith.constant 1 : i64
    %c128_i64 = arith.constant 128 : i64
    %c64_i64 = arith.constant 64 : i64

    %0 = ub.poison : !tt.ptr<tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>>

    %1 = tt.make_tensor_ptr %arg0, [%c128_i64, %c64_i64], [%c64_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>>

    %result = scf.for %i = %lb to %ub step %c1_i32
      iter_args(%ptr = %0) -> !tt.ptr<tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>> : i32 {

      %advanced = tt.advance %ptr, [%c0_i32, %c0_i32] : <tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>>

      scf.yield %advanced : !tt.ptr<tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>>
    }

    tt.return
  }
}
