// RUN: triton-opt %s --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm --canonicalize | FileCheck %s --implicit-check-not=llvm.inline_asm

// CHECK-DAG: llvm.func spir_funccc @_Z38intel_sub_group_f16_f16_matrix_mad_k16Dv8_sDv8_iDv8_f(vector<8xi16>, vector<8xi32>, vector<8xf32>) -> vector<8xf32> attributes {passthrough = ["convergent"]}
// CHECK-DAG: llvm.func spir_funccc @llvm.genx.GenISA.LSC2DBlockRead.v8i32(i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1, i32) -> vector<8xi32>
// CHECK-DAG: llvm.func spir_funccc @llvm.genx.GenISA.LSC2DBlockRead.v8i16(i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1, i32) -> vector<8xi16>
// CHECK-DAG: llvm.func spir_funccc @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1, i32)
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #triton_gpu.dot_op<{opIdx = 0, parent = #dpas, kWidth = 2}>
#dot1 = #triton_gpu.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {"triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
  tt.func public @matmul_with_prefetch(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: tensor<32x32x!tt.ptr<f32>, #blocked>, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64) {
    // CHECK-LABEL: @matmul_with_prefetch
    // CHECK-COUNT-2: llvm.call @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid{{.*}} -> ()
    // CHECK-COUNT-1: llvm.call @llvm.genx.GenISA.LSC2DBlockRead.v8i16{{.*}} -> vector<8xi16>
    // CHECK-COUNT-1: llvm.call @llvm.genx.GenISA.LSC2DBlockRead.v8i32{{.*}} -> vector<8xi32>
    // CHECK-COUNT-1: llvm.call @_Z38intel_sub_group_f16_f16_matrix_mad_k16Dv8_sDv8_iDv8_f({{.*}}) {passthrough = ["convergent"]} : (vector<8xi16>, vector<8xi32>, vector<8xf32>) -> vector<8xf32>
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %3 = tt.make_tensor_ptr %arg0, [%arg3, %arg5], [%arg6, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x16xf16, #dot0>>
    %6 = tt.make_tensor_ptr %arg1, [%arg5, %arg4], [%arg7, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x32xf16, #dot1>>
    triton_intel_gpu.prefetch %3 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<32x16xf16, #dot0>>
    triton_intel_gpu.prefetch %6 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<16x32xf16, #dot1>>
    %9 = tt.load %3 {boundaryCheck = array<i32: 1>, padding = 1 : i32} : !tt.ptr<tensor<32x16xf16, #dot0>>
    %10 = tt.load %6 {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<16x32xf16, #dot1>>
    %11 = tt.dot %9, %10, %cst, inputPrecision = tf32 : tensor<32x16xf16, #dot0> * tensor<16x32xf16, #dot1> -> tensor<32x32xf32, #dpas>
    %12 = triton_gpu.convert_layout %11 {allocation.offset = 0 : i32} : tensor<32x32xf32, #dpas> -> tensor<32x32xf32, #blocked>
    tt.return
  }
}
