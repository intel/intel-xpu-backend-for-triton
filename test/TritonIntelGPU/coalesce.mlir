// RUN: triton-opt %s -split-input-file -tritongpu-coalesce | FileCheck %s

// COM: Test coalescing on tensor descriptor load with DPAS/dot_op layout.
// COM: This is Intel-specific since it uses #ttig.dpas encoding.

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK: #[[$COALESCED_LAYOUT0:.*]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
  // CHECK: #[[$COALESCED_LAYOUT1:.*]] = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
  // CHECK-LABEL: @test_tdesc_load_dot_op
  tt.func public @test_tdesc_load_dot_op(
      %arg0: !tt.tensordesc<tensor<8x64xf8E5M2>>,
      %arg1: !tt.tensordesc<tensor<64x16xf8E5M2>>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<8x16xf32, #dpas>
    %c0_i32 = arith.constant 0 : i32
    // CHECK: %[[LOAD0:.*]] = tt.descriptor_load %arg0[{{.*}}] : !tt.tensordesc<tensor<8x64xf8E5M2>> -> tensor<8x64xf8E5M2, #[[$COALESCED_LAYOUT0]]>
    // CHECK-NEXT: ttg.convert_layout %[[LOAD0]] : tensor<8x64xf8E5M2, #[[$COALESCED_LAYOUT0]]> -> tensor<8x64xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %0 = tt.descriptor_load %arg0[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<8x64xf8E5M2>> -> tensor<8x64xf8E5M2, #dot0>
    // CHECK: %[[LOAD1:.*]] = tt.descriptor_load %arg1[{{.*}}] : !tt.tensordesc<tensor<64x16xf8E5M2>> -> tensor<64x16xf8E5M2, #[[$COALESCED_LAYOUT1]]>
    // CHECK-NEXT: ttg.convert_layout %[[LOAD1]] : tensor<64x16xf8E5M2, #[[$COALESCED_LAYOUT1]]> -> tensor<64x16xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %1 = tt.descriptor_load %arg1[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<64x16xf8E5M2>> -> tensor<64x16xf8E5M2, #dot1>
    %2 = tt.fp_to_fp %0 : tensor<8x64xf8E5M2, #dot0> -> tensor<8x64xf16, #dot0>
    %3 = tt.fp_to_fp %1 : tensor<64x16xf8E5M2, #dot1> -> tensor<64x16xf16, #dot1>
    %4 = tt.dot %2, %3, %cst, inputPrecision = tf32 : tensor<8x64xf16, #dot0> * tensor<64x16xf16, #dot1> -> tensor<8x16xf32, #dpas>
    tt.return
  }
}
