// RUN: triton-opt %s -split-input-file --intel-decompose-unsupported-conversions | FileCheck %s

// CHECK: #[[$BLOCKED:.+]] = #ttg.blocked
// CHECK: #[[$DPAS:.+]] = #triton_intel_gpu.dpas
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [32, 1], repCluster = [1, 2], A = [8, 16], B = [16, 32], C = [8, 32]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, triton_intel_gpu.min_sg_size = 16 : i32, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block} {
  // CHECK-LABEL: decompose_dpas_2_dot
  tt.func public @decompose_dpas_2_dot() {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf16, #dpas>
    // CHECK: %[[VAL_1:.*]] = ttg.convert_layout {{.*}} : tensor<64x32xf16, #[[$DPAS]]> -> tensor<64x32xf16, #[[$BLOCKED]]>
    // CHECK: %[[VAL_2:.*]] = ttg.local_alloc %[[VAL_1]] : (tensor<64x32xf16, #[[$BLOCKED]]>) -> !ttg.memdesc<64x32xf16, {{.*}}>
    // CHECK: %[[VAL_3:.*]] = ttg.local_load %[[VAL_2]] : !ttg.memdesc<64x32xf16, {{.*}}> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$DPAS]], kWidth = 2}>>
    %0 = ttg.convert_layout %cst : tensor<64x32xf16, #dpas> -> tensor<64x32xf16, #dot_b>
    // COM: There is a shortcut for converting the layout from DPAS -> DotOp A operands.
    // COM: The convert_layout op is not replaced by local_alloc/local_load.
    // CHECK: ttg.convert_layout
    // CHECK-NOT: ttg.local_alloc
    // CHECK-NOT: ttg.local_load
    %1 = ttg.convert_layout %cst : tensor<64x32xf16, #dpas> -> tensor<64x32xf16, #dot_a>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [16, 1], warpsPerCTA = [1, 32], order = [0, 1]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, triton_intel_gpu.min_sg_size = 16 : i32, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block} {
  // CHECK-LABEL: decompose_block_2_dot
  tt.func public @decompose_block_2_dot() {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf16, #blocked>
    // CHECK-NOT: ttg.convert_layout
    // CHECK: ttg.local_alloc
    // CHECK: ttg.local_load
    %0 = ttg.convert_layout %cst : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #dot_a>
    // CHECK: ttg.local_alloc
    // CHECK: ttg.local_load
    %1 = ttg.convert_layout %cst : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #dot_b>
    tt.return
  }
}
