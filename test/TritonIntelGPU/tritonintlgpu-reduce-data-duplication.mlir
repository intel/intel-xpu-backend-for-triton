// RUN: triton-opt %s -split-input-file --tritonintelgpu-reduce-data-duplication | FileCheck %s

#dpas1 = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [32, 1], repCluster = [1, 2], A = [8, 16], B = [16, 32], C = [8, 32]}>
#dpas2 = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 32], repCluster = [1, 2], A = [8, 16], B = [16, 32], C = [8, 32]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 32 : i32, "triton_gpu.threads-per-warp" = 16 : i32, triton_intel_gpu.min_sg_size = 16 : i32, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block} {
  // CHECK-LABEL: no_duplication
  tt.func public @no_duplication() {
    %cst1 = arith.constant dense<0.000000e+00> : tensor<64x32xf16, #dpas1>
    %cst2 = arith.constant dense<0.000000e+00> : tensor<64x32xf16, #dpas2>
    // CHECK-NOT: triton_gpu.local_alloc
    // CHECK-NOT: triton_gpu.local_load
    %108 = triton_gpu.convert_layout %cst1 : tensor<64x32xf16, #dpas1> -> tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas1, kWidth = 2}>>
    %109 = triton_gpu.convert_layout %cst2 : tensor<64x32xf16, #dpas2> -> tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #dpas2, kWidth = 2}>>
    tt.return
  }
}

// -----

#dpas1 = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [32, 1], repCluster = [1, 2], A = [8, 16], B = [16, 32], C = [8, 32]}>
#dpas2 = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 32], repCluster = [1, 2], A = [8, 16], B = [16, 32], C = [8, 32]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 32 : i32, "triton_gpu.threads-per-warp" = 16 : i32, triton_intel_gpu.min_sg_size = 16 : i32, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block} {
  // CHECK-LABEL: reduce_duplication
  tt.func public @reduce_duplication() {
    %cst1 = arith.constant dense<0.000000e+00> : tensor<64x32xf16, #dpas1>
    %cst2 = arith.constant dense<0.000000e+00> : tensor<64x32xf16, #dpas2>
    // CHECK: triton_gpu.local_alloc
    // CHECK: triton_gpu.local_load
    // CHECK-NOT: triton_gpu.convert_layout
    %108 = triton_gpu.convert_layout %cst1 : tensor<64x32xf16, #dpas1> -> tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #dpas1, kWidth = 2}>>
    %109 = triton_gpu.convert_layout %cst2 : tensor<64x32xf16, #dpas2> -> tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas2, kWidth = 2}>>
    tt.return
  }
}
