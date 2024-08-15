// RUN: triton-opt %s --tritonintelgpu-materialize-block-pointer | FileCheck %s

#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot_a = #triton_gpu.dot_op<{opIdx = 0, parent = #dpas, kWidth = 2}>
#dot_b = #triton_gpu.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, triton_gpu.target = "xpu", triton_intel_gpu.support_sg_2d_block} {
  // CHECK-LABEL: tt.func public @materialize_block_pointer(
  tt.func public @materialize_block_pointer(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %pitch: i64 {tt.divisibility = 16 : i32}, %pitch_odd: i64 {tt.divisibility = 15 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64

    // CHECK: tt.load {{.*}} {boundaryCheck = array<i32: 1>, padding = 1 : i32, triton_intel_gpu.block_io = "row_major"}
    // CHECK: tt.load {{.*}} {boundaryCheck = array<i32: 0>, padding = 1 : i32, triton_intel_gpu.block_io = "row_major"}
    %3 = tt.make_tensor_ptr %arg0, [%c0_i64, %c0_i64], [%pitch, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x32xf16, #dot_a>>
    %4 = tt.make_tensor_ptr %arg0, [%c0_i64, %c0_i64], [%pitch, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x64xf16, #dot_b>>
    %5 = tt.load %3 {boundaryCheck = array<i32: 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, padding = 1 : i32} : !tt.ptr<tensor<64x32xf16, #dot_a>>
    %6 = tt.load %4 {boundaryCheck = array<i32: 0>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, padding = 1 : i32} : !tt.ptr<tensor<32x64xf16, #dot_b>>

    // CHECK: tt.load {{.*}} {boundaryCheck = array<i32: 1>, padding = 1 : i32, triton_intel_gpu.block_io = "column_major"}
    // CHECK: tt.load {{.*}} {boundaryCheck = array<i32: 0>, padding = 1 : i32, triton_intel_gpu.block_io = "column_major"}
    %7 = tt.make_tensor_ptr %arg0, [%c0_i64, %c0_i64], [%c1_i64, %pitch], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<64x32xf16, #dot_a>>
    %8 = tt.make_tensor_ptr %arg0, [%c0_i64, %c0_i64], [%c1_i64, %pitch], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<32x64xf16, #dot_b>>
    %9 = tt.load %7 {boundaryCheck = array<i32: 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, padding = 1 : i32} : !tt.ptr<tensor<64x32xf16, #dot_a>>
    %10 = tt.load %8 {boundaryCheck = array<i32: 0>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, padding = 1 : i32} : !tt.ptr<tensor<32x64xf16, #dot_b>>

    // COM: Non-constant stride on fast changing dim.
    // CHECK: tt.load {{.*}} {boundaryCheck = array<i32: 1>, padding = 1 : i32}
    // CHECK: tt.load {{.*}} {boundaryCheck = array<i32: 0>, padding = 1 : i32}
    %11 = tt.make_tensor_ptr %arg0, [%c0_i64, %c0_i64], [%pitch, %pitch], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<64x32xf16, #dot_a>>
    %12 = tt.make_tensor_ptr %arg0, [%c0_i64, %c0_i64], [%pitch, %pitch], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<32x64xf16, #dot_b>>
    %13 = tt.load %11 {boundaryCheck = array<i32: 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, padding = 1 : i32} : !tt.ptr<tensor<64x32xf16, #dot_a>>
    %14 = tt.load %12 {boundaryCheck = array<i32: 0>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, padding = 1 : i32} : !tt.ptr<tensor<32x64xf16, #dot_b>>

    // COM: Non-64 divisible pitch.
    // CHECK: tt.load {{.*}} {boundaryCheck = array<i32: 1>, padding = 1 : i32}
    // CHECK: tt.load {{.*}} {boundaryCheck = array<i32: 0>, padding = 1 : i32}
    %15 = tt.make_tensor_ptr %arg0, [%c0_i64, %c0_i64], [%c1_i64, %pitch_odd], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<64x32xf16, #dot_a>>
    %16 = tt.make_tensor_ptr %arg0, [%c0_i64, %c0_i64], [%c1_i64, %pitch_odd], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<32x64xf16, #dot_b>>
    %17 = tt.load %15 {boundaryCheck = array<i32: 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, padding = 1 : i32} : !tt.ptr<tensor<64x32xf16, #dot_a>>
    %18 = tt.load %16 {boundaryCheck = array<i32: 0>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, padding = 1 : i32} : !tt.ptr<tensor<32x64xf16, #dot_b>>
    tt.return
  }
}
