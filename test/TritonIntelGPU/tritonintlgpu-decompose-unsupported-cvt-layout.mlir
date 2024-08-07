// RUN: triton-opt %s -split-input-file --intel-decompose-unsupported-conversions | FileCheck %s

// CHECK: #[[$BLOCKED:.+]] = #triton_gpu.blocked
// CHECK: #[[$DPAS:.+]] = #triton_intel_gpu.dpas
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [32, 1], repCluster = [1, 2], A = [8, 16], B = [16, 32], C = [8, 32]}>
#dot_a = #triton_gpu.dot_op<{opIdx = 0, parent = #dpas, kWidth = 2}>
#dot_b = #triton_gpu.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 32 : i32, "triton_gpu.threads-per-warp" = 16 : i32, triton_intel_gpu.min_sg_size = 16 : i32, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block} {
  // CHECK-LABEL: decompose_dpas_2_dot
  tt.func public @decompose_dpas_2_dot() {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf16, #dpas>
    // CHECK: %[[VAL_1:.*]] = triton_gpu.convert_layout {{.*}} : tensor<64x32xf16, #[[$DPAS]]> -> tensor<64x32xf16, #[[$BLOCKED]]>
    // CHECK: %[[VAL_2:.*]] = triton_gpu.local_alloc %[[VAL_1]] : (tensor<64x32xf16, #[[$BLOCKED]]>) -> !tt.memdesc<64x32xf16, {{.*}}>
    // CHECK: %[[VAL_3:.*]] = triton_gpu.local_load %[[VAL_2]] : !tt.memdesc<64x32xf16, {{.*}}> -> tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[$DPAS]], kWidth = 2}>>
    %0 = triton_gpu.convert_layout %cst : tensor<64x32xf16, #dpas> -> tensor<64x32xf16, #dot_b>
    // COM: There is a shortcut for converting the layout from DPAS -> DotOp A operands.
    // COM: The convert_layout op is not replaced by local_alloc/local_load.
    // CHECK: triton_gpu.convert_layout
    // CHECK-NOT: triton_gpu.local_alloc
    // CHECK-NOT: triton_gpu.local_load
    %1 = triton_gpu.convert_layout %cst : tensor<64x32xf16, #dpas> -> tensor<64x32xf16, #dot_a>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [4, 1], threadsPerWarp = [16, 1], warpsPerCTA = [1, 32], order = [0, 1]}>
#dot_a = #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>
#dot_b = #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 32 : i32, "triton_gpu.threads-per-warp" = 16 : i32, triton_intel_gpu.min_sg_size = 16 : i32, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block} {
  // CHECK-LABEL: decompose_block_2_dot
  tt.func public @decompose_block_2_dot() {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf16, #blocked>
    // CHECK-NOT: triton_gpu.convert_layout
    // CHECK: triton_gpu.local_alloc
    // CHECK: triton_gpu.local_load
    %0 = triton_gpu.convert_layout %cst : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #dot_a>
    // CHECK: triton_gpu.local_alloc
    // CHECK: triton_gpu.local_load
    %1 = triton_gpu.convert_layout %cst : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #dot_b>
    tt.return
  }
}
