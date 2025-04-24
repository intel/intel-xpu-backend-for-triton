// RUN: triton-opt %s -tritonintelgpu-reduce-variable-liveness -cse | FileCheck %s

// CHECK: #[[$DPAS:.+]] = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {triton_intel_gpu.support_sg_2d_block, "ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    // CHECK-LABEL:   tt.func public @matmul_kernel
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #dpas>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    %0 = tt.make_tensor_ptr %arg0, [%c0_i64, %c0_i64], [%c0_i64, %c0_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x64xf16, #dot0>>
    // CHECK:      triton_intel_gpu.prefetch {{.*}} : !tt.ptr<tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>>
    // CHECK-NOT:  tt.load {{.*}} : !tt.ptr<tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>>
    %1 = tt.make_tensor_ptr %arg1, [%c0_i64, %c0_i64], [%c0_i64, %c0_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x256xf16, #dot1>>
    %2 = tt.load %0 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<128x64xf16, #dot0>>
    %3 = arith.muli %c64_i32, %c0_i32 : i32
    triton_intel_gpu.prefetch %1 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>} : !tt.ptr<tensor<64x256xf16, #dot1>>
    %4:2 = scf.for %arg2 = %c0_i32 to %c64_i32 step %c64_i32 iter_args(%arg3 = %cst, %arg4 = %1) -> (tensor<128x256xf32, #dpas>, !tt.ptr<tensor<64x256xf16, #dot1>>)  : i32 {
      // CHECK:  scf.for
      // CHECK:  tt.load {{.*}} : !tt.ptr<tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>>
      %7 = tt.advance %arg4, [%c64_i32, %c0_i32] : <tensor<64x256xf16, #dot1>>
      triton_intel_gpu.prefetch %7 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>} : !tt.ptr<tensor<64x256xf16, #dot1>>
      %8 = tt.load %arg4 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x256xf16, #dot1>>
      %9 = tt.dot %2, %8, %arg3, inputPrecision = tf32 : tensor<128x64xf16, #dot0> * tensor<64x256xf16, #dot1> -> tensor<128x256xf32, #dpas>
      scf.yield %9, %7 : tensor<128x256xf32, #dpas>, !tt.ptr<tensor<64x256xf16, #dot1>>
    }
    tt.return
  }
}
