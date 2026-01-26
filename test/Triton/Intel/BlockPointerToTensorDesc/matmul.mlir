// RUN: triton-opt %s -triton-intel-block-pointer-to-tdesc | FileCheck %s

// CHECK-NOT: tt.make_tensor_ptr
// CHECK-NOT: tt.advance
tt.func public @matmul(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2: !tt.ptr<f32>) {
  %c4096_i32 = arith.constant 4096 : i32
  %c64_i32 = arith.constant 64 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32>
  %c32_i32 = arith.constant 32 : i32
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c4096_i64 = arith.constant 4096 : i64
  // CHECK: [[c4096_i32:%.*]] = arith.trunci %c4096_i64 : i64 to i32
  %c256_i32 = arith.constant 256 : i32
  %c4_i32 = arith.constant 4 : i32
  %c16_i32 = arith.constant 16 : i32
  %0 = tt.get_program_id x : i32
  %1 = arith.divsi %0, %c64_i32 : i32
  %2 = arith.muli %1, %c4_i32 : i32
  %3 = arith.subi %c16_i32, %2 : i32
  %4 = arith.minsi %3, %c4_i32 : i32
  %5 = arith.remsi %0, %c64_i32 : i32
  %6 = arith.remsi %5, %4 : i32
  %7 = arith.addi %2, %6 : i32
  %8 = arith.remsi %0, %c64_i32 : i32
  %9 = arith.divsi %8, %4 : i32
  // CHECK: [[off00:%.*]] = arith.muli {{.*}}, %c256_i32 : i32
  %10 = arith.muli %7, %c256_i32 : i32
  // CHECK: [[tdesc0:%.*]] = tt.make_tensor_descriptor %arg0, [[[c4096_i32]], [[c4096_i32]]], [%c4096_i64, %c1_i64] : <bf16>, <tensor<256x32xbf16>>
  %11 = tt.make_tensor_ptr %arg0, [%c4096_i64, %c4096_i64], [%c4096_i64, %c1_i64], [%10, %c0_i32] {order = array<i32: 1, 0>} : <tensor<256x32xbf16>>
  // CHECK: [[off11:%.*]] = arith.muli {{.*}}, %c256_i32 : i32
  %12 = arith.muli %9, %c256_i32 : i32
  // CHECK: [[tdesc1:%.*]] = tt.make_tensor_descriptor %arg1, [[[c4096_i32]], [[c4096_i32]]], [%c4096_i64, %c1_i64] : <bf16>, <tensor<32x256xbf16>>
  %13 = tt.make_tensor_ptr %arg1, [%c4096_i64, %c4096_i64], [%c4096_i64, %c1_i64], [%c0_i32, %12] {order = array<i32: 1, 0>} : <tensor<32x256xbf16>>
  // CHECK: [[for:%.*]]:3 = scf.for %arg3 = %c0_i32 to %c4096_i32 step %c32_i32 iter_args(%arg4 = %cst, [[off01:%.*]] = %c0_i32, [[off10:%.*]] = %c0_i32) -> (tensor<256x256xf32>, i32, i32)  : i32 {
  %14:3 = scf.for %arg3 = %c0_i32 to %c4096_i32 step %c32_i32 iter_args(%arg4 = %11, %arg5 = %13, %arg6 = %cst) -> (!tt.ptr<tensor<256x32xbf16>>, !tt.ptr<tensor<32x256xbf16>>, tensor<256x256xf32>)  : i32 {
    // CHECK: [[load0:%.*]] = tt.descriptor_load [[tdesc0]][[[off00]], [[off01]]] : !tt.tensordesc<tensor<256x32xbf16>> -> tensor<256x32xbf16>
    %18 = tt.load %arg4 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<256x32xbf16>>
    // CHECK: [[load1:%.*]] = tt.descriptor_load [[tdesc1]][[[off10]], [[off11]]] : !tt.tensordesc<tensor<32x256xbf16>> -> tensor<32x256xbf16>
    %19 = tt.load %arg5 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<32x256xbf16>>
    // CHECK: tt.dot [[load0]], [[load1]], %cst, inputPrecision = tf32 : tensor<256x32xbf16> * tensor<32x256xbf16> -> tensor<256x256xf32>
    %20 = tt.dot %18, %19, %cst, inputPrecision = tf32 : tensor<256x32xbf16> * tensor<32x256xbf16> -> tensor<256x256xf32>
    %21 = arith.addf %arg6, %20 : tensor<256x256xf32>
    // CHECK: [[off01_:%.*]] = arith.addi [[off01]], %c32_i32 : i32
    %22 = tt.advance %arg4, [%c0_i32, %c32_i32] : <tensor<256x32xbf16>>
    // CHECK: [[off10_:%.*]] = arith.addi [[off10]], %c32_i32 : i32
    %23 = tt.advance %arg5, [%c32_i32, %c0_i32] : <tensor<32x256xbf16>>
    // CHECK: scf.yield {{.*}}, [[off01_]], [[off10_]] : tensor<256x256xf32>, i32, i32
    scf.yield %22, %23, %21 : !tt.ptr<tensor<256x32xbf16>>, !tt.ptr<tensor<32x256xbf16>>, tensor<256x256xf32>
  }
  %15 = arith.muli %7, %c256_i32 : i32
  %16 = arith.muli %9, %c256_i32 : i32
  // CHECK: [[off0:%.*]] = arith.muli {{.*}}, %c256_i32 : i32
  // CHECK: [[off1:%.*]] = arith.muli {{.*}}, %c256_i32 : i32
  // CHECK: [[tdesc2:%.*]] = tt.make_tensor_descriptor %arg2, [[[c4096_i32]], [[c4096_i32]]], [%c4096_i64, %c1_i64] : <f32>, <tensor<256x256xf32>>
  %17 = tt.make_tensor_ptr %arg2, [%c4096_i64, %c4096_i64], [%c4096_i64, %c1_i64], [%15, %16] {order = array<i32: 1, 0>} : <tensor<256x256xf32>>
  // CHECK: tt.descriptor_store [[tdesc2]][[[off0]], [[off1]]], [[for]]#0 : !tt.tensordesc<tensor<256x256xf32>>, tensor<256x256xf32>
  tt.store %17, %14#2 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<256x256xf32>>
  tt.return
}
