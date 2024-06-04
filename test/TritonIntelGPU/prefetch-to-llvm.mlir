// RUN: triton-opt %s --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s --implicit-check-not=llvm.inline_asm

// CHECK-DAG: llvm.func spir_funccc @_Z38intel_sub_group_f16_f16_matrix_mad_k16Dv8_sDv8_iDv8_f(vector<8xi16>, vector<8xi32>, vector<8xf32>) -> vector<8xf32> attributes {passthrough = ["convergent"]}
// CHECK-DAG: llvm.func spir_funccc @llvm.genx.GenISA.LSC2DBlockRead.v8i32(i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1, i32) -> vector<8xi32>
// CHECK-DAG: llvm.func spir_funccc @llvm.genx.GenISA.LSC2DBlockRead.v8i16(i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1, i32) -> vector<8xi16>
// CHECK-DAG: llvm.func spir_funccc @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1, i32)
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #triton_gpu.dot_op<{opIdx = 0, parent = #dpas, kWidth = 2}>
#dot1 = #triton_gpu.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {"triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
  tt.func public @matmul_with_prefetch(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64) {
    // CHECK-LABEL: @matmul_with_prefetch
    // CHECK-COUNT-2: llvm.call @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid{{.*}} -> ()
    // CHECK-COUNT-1: llvm.call @llvm.genx.GenISA.LSC2DBlockRead.v8i16{{.*}} -> vector<8xi16>
    // CHECK-COUNT-1: llvm.call @llvm.genx.GenISA.LSC2DBlockRead.v8i32{{.*}} -> vector<8xi32>
    // CHECK-COUNT-1: llvm.call spir_funccc @_Z38intel_sub_group_f16_f16_matrix_mad_k16Dv8_sDv8_iDv8_f({{.*}}) {{.*}} : (vector<8xi16>, vector<8xi32>, vector<8xf32>) -> vector<8xf32>
    %C = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %ptrA = tt.make_tensor_ptr %arg0, [%arg2, %arg4], [%arg5, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x16xf16, #dot0>>
    %ptrB = tt.make_tensor_ptr %arg1, [%arg4, %arg3], [%arg6, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x32xf16, #dot1>>
    triton_intel_gpu.prefetch %ptrA {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<32x16xf16, #dot0>>
    triton_intel_gpu.prefetch %ptrB {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<16x32xf16, #dot1>>
    %A = tt.load %ptrA {boundaryCheck = array<i32: 1>, padding = 1 : i32} : !tt.ptr<tensor<32x16xf16, #dot0>>
    %B = tt.load %ptrB {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<16x32xf16, #dot1>>
    %D = tt.dot %A, %B, %C, inputPrecision = tf32 : tensor<32x16xf16, #dot0> * tensor<16x32xf16, #dot1> -> tensor<32x32xf32, #dpas>
    %0 = triton_gpu.convert_layout %D {allocation.offset = 0 : i32} : tensor<32x32xf32, #dpas> -> tensor<32x32xf32, #blocked>
    tt.return
  }
}
