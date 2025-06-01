// RUN: triton-opt %s -split-input-file --tritonintelgpu-reduce-data-duplication | FileCheck %s

#dpas1 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [32, 1], repCluster = [1, 2], A = [8, 16], B = [16, 32], C = [8, 32]}>
#dpas2 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 32], repCluster = [1, 2], A = [8, 16], B = [16, 32], C = [8, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_dpas, ttig.support_sg_2d_block} {
  // CHECK-LABEL: no_duplication
  tt.func public @no_duplication() {
    %cst1 = arith.constant dense<0.000000e+00> : tensor<64x32xf16, #dpas1>
    %cst2 = arith.constant dense<0.000000e+00> : tensor<64x32xf16, #dpas2>
    // CHECK-NOT: ttg.local_alloc
    // CHECK-NOT: ttg.local_load
    %108 = ttg.convert_layout %cst1 : tensor<64x32xf16, #dpas1> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas1, kWidth = 1}>>
    %109 = ttg.convert_layout %cst2 : tensor<64x32xf16, #dpas2> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas2, kWidth = 2}>>
    tt.return
  }
}

// -----

#dpas1 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [32, 1], repCluster = [1, 2], A = [8, 16], B = [16, 32], C = [8, 32]}>
#dpas2 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 32], repCluster = [1, 2], A = [8, 16], B = [16, 32], C = [8, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_dpas, ttig.support_sg_2d_block} {
  // CHECK-LABEL: reduce_duplication
  tt.func public @reduce_duplication() {
    %cst1 = arith.constant dense<0.000000e+00> : tensor<64x32xf16, #dpas1>
    %cst2 = arith.constant dense<0.000000e+00> : tensor<64x32xf16, #dpas2>
    // CHECK: ttg.local_alloc
    // CHECK: ttg.local_load
    %108 = ttg.convert_layout %cst1 : tensor<64x32xf16, #dpas1> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas1, kWidth = 2}>>
    // CHECK: ttg.convert_layout
    %109 = ttg.convert_layout %cst2 : tensor<64x32xf16, #dpas2> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas2, kWidth = 1}>>
    tt.return
  }
}
