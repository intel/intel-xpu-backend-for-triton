// RUN: triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm | FileCheck %s --implicit-check-not=llvm.inline_asm

// CHECK-DAG: llvm.func spir_funccc @llvm.genx.GenISA.LSC2DBlockWrite.v8i16(i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1, i32, vector<8xi16>)
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #triton_gpu.dot_op<{opIdx = 0, parent = #dpas, kWidth=2}>
#dot1 = #triton_gpu.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
  tt.func public @matmul_no_scf_with_advance_kernel(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #dpas>
    %c32_i32 = arith.constant 32 : i32
    %c-64_i32 = arith.constant -64 : i32
    %c-32_i32 = arith.constant -32 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %3 = tt.make_tensor_ptr %arg0, [%arg3, %arg5], [%arg6, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x32xf16, #dot0>>
    %6 = tt.make_tensor_ptr %arg1, [%arg3, %arg4], [%arg7, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x64xf16, #dot1>>
    %7 = tt.advance %3, [%c64_i32, %c-32_i32] : <tensor<64x32xf16, #dot0>>
    %8 = tt.advance %7, [%c-64_i32, %c32_i32] : <tensor<64x32xf16, #dot0>>
    %9 = tt.load %8 {boundaryCheck = array<i32: 1>, padding = 1 : i32} : !tt.ptr<tensor<64x32xf16, #dot0>>
    %10 = tt.load %6 {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<32x64xf16, #dot1>>
    %11 = tt.dot %9, %10, %cst, inputPrecision = tf32 : tensor<64x32xf16, #dot0> * tensor<32x64xf16, #dot1> -> tensor<64x64xf32, #dpas>
    %12 = arith.truncf %11#0 : tensor<64x64xf32, #dpas> to tensor<64x64xf16, #dpas>
    %13 = tt.make_tensor_ptr %arg2, [%arg3, %arg5], [%arg6, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x64xf16, #dpas>>
    // CHECK-COUNT-4: llvm.call @llvm.genx.GenISA.LSC2DBlockWrite.v8i16{{.*}}
    tt.store %13, %12 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x64xf16, #dpas>>
    tt.return
  }
}

// -----

// CHECK-DAG: llvm.func spir_funccc @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1, i32, vector<8xi32>)
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #triton_gpu.dot_op<{opIdx = 0, parent = #dpas, kWidth=2}>
#dot1 = #triton_gpu.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
  tt.func public @matmul_no_scf_with_advance_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #dpas>
    %c32_i32 = arith.constant 32 : i32
    %c-64_i32 = arith.constant -64 : i32
    %c-32_i32 = arith.constant -32 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %3 = tt.make_tensor_ptr %arg0, [%arg3, %arg5], [%arg6, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x32xf32, #dot0>>
    %6 = tt.make_tensor_ptr %arg1, [%arg3, %arg4], [%arg7, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x64xf32, #dot1>>
    %7 = tt.advance %3, [%c64_i32, %c-32_i32] : <tensor<64x32xf32, #dot0>>
    %8 = tt.advance %7, [%c-64_i32, %c32_i32] : <tensor<64x32xf32, #dot0>>
    %9 = tt.load %8 {boundaryCheck = array<i32: 1>, padding = 1 : i32} : !tt.ptr<tensor<64x32xf32, #dot0>>
    %10 = tt.load %6 {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<32x64xf32, #dot1>>
    %11 = tt.dot %9, %10, %cst, inputPrecision = tf32 : tensor<64x32xf32, #dot0> * tensor<32x64xf32, #dot1> -> tensor<64x64xf32, #dpas>
    %13 = tt.make_tensor_ptr %arg2, [%arg3, %arg5], [%arg6, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x64xf32, #dpas>>
    // CHECK-COUNT-4: llvm.call @llvm.genx.GenISA.LSC2DBlockWrite.v8i32{{.*}}
    tt.store %13, %11 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x64xf32, #dpas>>
    tt.return
  }
}
