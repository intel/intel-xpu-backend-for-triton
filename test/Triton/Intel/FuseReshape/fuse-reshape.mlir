// RUN: triton-opt %s -split-input-file -triton-intel-fuse-reshape | FileCheck %s

// COM: tt.load -> tt.reshape -> tt.dot chain, not in a loop.
tt.func public @fuseLoadWithReshape1(%arg0: !tt.ptr<tensor<256x32xbf16>>, %arg1: !tt.ptr<bf16>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %c1_i64 = arith.constant 1 : i64
  %c2_i64 = arith.constant 2 : i64
  %c4_i64 = arith.constant 4 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32>
  %0 = tt.make_tensor_ptr %arg1, [%c2_i64, %c1_i64, %c1024_i64], [%c1024_i64, %c4_i64, %c1_i64], [%c2_i32, %c1_i32, %c0_i32] {order = array<i32: 2, 1, 0>} : <tensor<1x32x256xbf16>>
  %1 = tt.load %arg0 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<256x32xbf16>>
  %3 = tt.load %0 {boundaryCheck = array<i32: 1, 2>} : !tt.ptr<tensor<1x32x256xbf16>>
  %4 = tt.reshape %3 : tensor<1x32x256xbf16> -> tensor<32x256xbf16>
  %5 = tt.dot %1, %4, %cst, inputPrecision = tf32 : tensor<256x32xbf16> * tensor<32x256xbf16> -> tensor<256x256xf32>
  tt.return
}
// CHECK-LABEL: fuseLoadWithReshape1
// CHECK-NOT: tt.reshape
// CHECK: [[DIV:%.*]] = arith.divui %c1024_i64, %c4_i64 : i64
// CHECK: [[MUL1:%.*]] = arith.muli %c2_i64, [[DIV]] : i64
// CHECK: [[ADD1:%.*]] = arith.addi [[MUL1]], %c1_i64 : i64
// CHECK: [[TRUNC:%.*]] = arith.trunci [[DIV]] : i64 to i32
// CHECK: [[MUL2:%.*]] = arith.muli %c2_i32, [[TRUNC]] : i32
// CHECK: [[ADD2:%.*]] = arith.addi [[MUL2]], %c1_i32 : i32

// CHECK: [[PTR:%.*]] = tt.make_tensor_ptr %arg1, [[[ADD1]], %c1024_i64], [%c4_i64, %c1_i64], [[[ADD2]], %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x256xbf16>>
// CHECK: [[LOAD_B:%.*]] = tt.load [[PTR]] {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<32x256xbf16>>
// CHECK: tt.dot {{.*}}, [[LOAD_B]], {{.*}}, inputPrecision = tf32 : tensor<256x32xbf16> * tensor<32x256xbf16> -> tensor<256x256xf32>

// -----

// COM: tt.load -> tt.reshape -> tt.dot chain, in a loop.
// COM: where the 'make_tensor_ptr' result is not loop carried.
tt.func public @fuseLoadWithReshape2(%arg0: !tt.ptr<tensor<32x256xbf16>>, %arg1: !tt.ptr<bf16>) {
  %c0_i32 = arith.constant 0 : i32
  %c32_i32 = arith.constant 32 : i32
  %c1024_i32 = arith.constant 1024 : i32
  %c32_i64 = arith.constant 32 : i64
  %c1_i64 = arith.constant 1 : i64
  %c512_i64 = arith.constant 512 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32>
  %0 = tt.make_tensor_ptr %arg1, [%c512_i64, %c1024_i64, %c32_i64], [%c1024_i64, %c1_i64, %c512_i64], [%c32_i32, %c32_i32, %c0_i32] {order = array<i32: 2, 0, 1>} : <tensor<1x256x32xbf16>>
  %res:2 = scf.for %arg3 = %c0_i32 to %c1024_i32 step %c32_i32 iter_args(%arg4 = %cst, %arg5 = %c0_i32) -> (tensor<256x256xf32>, i32) : i32 {
    %1 = tt.load %arg0 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<32x256xbf16>>
    %3 = tt.load %0 {boundaryCheck = array<i32: 2, 1>} : !tt.ptr<tensor<1x256x32xbf16>>
    %2 = tt.reshape %3 : tensor<1x256x32xbf16> -> tensor<256x32xbf16>
    %4 = tt.dot %2, %1, %arg4, inputPrecision = tf32 : tensor<256x32xbf16> * tensor<32x256xbf16> -> tensor<256x256xf32>
    %5 = arith.addi %arg5, %c32_i32 : i32
    scf.yield %4, %5 : tensor<256x256xf32>, i32
  }
  tt.return
}
// CHECK-LABEL: fuseLoadWithReshape2
// CHECK-NOT: tt.reshape
// CHECK: [[DIV:%.*]] = arith.divui %c1024_i64, %c512_i64 : i64
// CHECK: [[MUL1:%.*]] = arith.muli %c512_i64, [[DIV]] : i64
// CHECK: [[ADD1:%.*]] = arith.addi [[MUL1]], %c32_i64 : i64
// CHECK: [[TRUNC:%.*]] = arith.trunci [[DIV]] : i64 to i32
// CHECK: [[MUL2:%.*]] = arith.muli %c32_i32, [[TRUNC]] : i32
// CHECK: [[ADD2:%.*]] = arith.addi [[MUL2]], %c0_i32 : i32
// CHECK: [[PTR:%.*]] = tt.make_tensor_ptr %arg1, [%c1024_i64, [[ADD1]]], [%c1_i64, %c512_i64], [%c32_i32, [[ADD2]]] {order = array<i32: 0, 1>} : <tensor<256x32xbf16>>
// CHECK: scf.for
// CHECK:   [[LOAD_A:%.*]] = tt.load [[PTR]] {boundaryCheck = array<i32: 1, 0>} : !tt.ptr<tensor<256x32xbf16>>
// CHECK:   tt.dot [[LOAD_A]], {{.*}}, {{.*}}, inputPrecision = tf32 : tensor<256x32xbf16> * tensor<32x256xbf16> -> tensor<256x256xf32>

// -----

// COM: tt.load -> tt.reshape -> tt.dot chain, in a loop
// COM: Where the 'make_tensor_ptr' result is loop carried.
tt.func public @fuseLoadWithReshape3(%a_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %b_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %c_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %M: i32 {tt.divisibility = 16 : i32}, %N: i32 {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32}, %stride_am: i32 {tt.divisibility = 16 : i32}, %stride_bk: i32 {tt.divisibility = 16 : i32}, %stride_cm: i32 {tt.divisibility = 16 : i32}) {
  %c127_i32 = arith.constant 127 : i32
  %c255_i32 = arith.constant 255 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<256x128xf32>
  %c32_i32 = arith.constant 32 : i32
  %c128_i32 = arith.constant 128 : i32
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c256_i32 = arith.constant 256 : i32
  %c4_i32 = arith.constant 4 : i32
  %0 = tt.get_program_id x : i32
  %1 = arith.addi %M, %c255_i32 : i32
  %2 = arith.divsi %1, %c256_i32 : i32
  %3 = arith.addi %N, %c127_i32 : i32
  %4 = arith.divsi %3, %c128_i32 : i32
  %5 = arith.muli %4, %c4_i32 : i32
  %6 = arith.divsi %0, %5 : i32
  %7 = arith.muli %6, %c4_i32 : i32
  %8 = arith.subi %2, %7 : i32
  %9 = arith.minsi %8, %c4_i32 : i32
  %10 = arith.remsi %0, %5 : i32
  %11 = arith.remsi %10, %9 : i32
  %12 = arith.addi %7, %11 : i32
  %13 = arith.divsi %10, %9 : i32
  %14 = arith.muli %12, %c256_i32 : i32
  %15 = arith.extsi %M : i32 to i64
  %16 = arith.extsi %K : i32 to i64
  %17 = arith.extsi %stride_am : i32 to i64
  %18 = tt.make_tensor_ptr %a_ptr, [%c1_i64, %15, %16], [%c1_i64, %17, %c1_i64], [%c0_i32, %14, %c0_i32] {order = array<i32: 2, 1, 0>} : <tensor<1x256x32xf32>>
  %19 = arith.muli %13, %c128_i32 : i32
  %20 = arith.extsi %N : i32 to i64
  %21 = arith.extsi %stride_bk : i32 to i64
  %22 = tt.make_tensor_ptr %b_ptr, [%16, %20], [%21, %c1_i64], [%c0_i32, %19] {order = array<i32: 1, 0>} : <tensor<32x128xf32>>
  %accumulator:3 = scf.for %k = %c0_i32 to %K step %c32_i32 iter_args(%a_block_ptr = %18, %b_block_ptr = %22, %accumulator_0 = %cst) -> (!tt.ptr<tensor<1x256x32xf32>>, !tt.ptr<tensor<32x128xf32>>, tensor<256x128xf32>)  : i32 {
    %25 = tt.load %a_block_ptr {boundaryCheck = array<i32: 1, 2>} : !tt.ptr<tensor<1x256x32xf32>>
    %26 = tt.reshape %25 : tensor<1x256x32xf32> -> tensor<256x32xf32>
    %27 = tt.load %b_block_ptr {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<32x128xf32>>
    %28 = tt.dot %26, %27, %cst, inputPrecision = tf32 : tensor<256x32xf32> * tensor<32x128xf32> -> tensor<256x128xf32>
    %29 = arith.addf %accumulator_0, %28 : tensor<256x128xf32>
    %30 = tt.advance %a_block_ptr, [%c0_i32, %c0_i32, %c32_i32] : <tensor<1x256x32xf32>>
    %31 = tt.advance %b_block_ptr, [%c32_i32, %c0_i32] : <tensor<32x128xf32>>
    scf.yield %30, %31, %29 : !tt.ptr<tensor<1x256x32xf32>>, !tt.ptr<tensor<32x128xf32>>, tensor<256x128xf32>
  }
  %23 = arith.extsi %stride_cm : i32 to i64
  %24 = tt.make_tensor_ptr %c_ptr, [%15, %20], [%23, %c1_i64], [%14, %19] {order = array<i32: 1, 0>} : <tensor<256x128xf32>>
  tt.store %24, %accumulator#2 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<256x128xf32>>
  tt.return
}
// CHECK-LABEL: fuseLoadWithReshape3
// CHECK-NOT: tt.reshape
// CHECK: [[DIV:%.*]] = arith.divui %c1_i64, %17 : i64
// CHECK: [[MUL1:%.*]] = arith.muli %c1_i64, [[DIV]] : i64
// CHECK: [[ADD1:%.*]] = arith.addi [[MUL1]], %15 : i64
// CHECK: [[TRUNC:%.*]] = arith.trunci [[DIV]] : i64 to i32
// CHECK: [[MUL2:%.*]] = arith.muli %c0_i32, [[TRUNC]] : i32
// CHECK: [[ADD2:%.*]] = arith.addi [[MUL2]], %14 : i32
// CHECK: [[PTR:%.*]] = tt.make_tensor_ptr %arg0, [[[ADD1]], %16], [%17, %c1_i64], [[[ADD2]], %c0_i32] {order = array<i32: 1, 0>} : <tensor<256x32xf32>>
// CHECK: scf.for {{.*}} = %c0_i32 to {{.*}} step %c32_i32 iter_args([[ARG:%.*]] = [[PTR]]
// CHECK:   [[LOAD_A:%.*]] = tt.load [[ARG]] {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<256x32xf32>>
// CHECK:   tt.dot [[LOAD_A]], {{.*}}, {{.*}}, inputPrecision = tf32 : tensor<256x32xf32> * tensor<32x128xf32> -> tensor<256x128xf32>
// CHECK:   tt.advance [[ARG]], [%c0_i32, %c32_i32] : <tensor<256x32xf32>>

// -----

// COM: tt.load -> tt.reshape -> tt.dot chain, in 2 loops.
// COM: Where the block ptr used by the loads in the 2 loops is created by the same make_tensor_ptr operation.
tt.func public @fuseLoadWithTrans4(%arg0: i32, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32    
  %c32_i32 = arith.constant 32 : i32
  %c1_i64 = arith.constant 1 : i64  
  %c64_i64 = arith.constant 64 : i64
  %c256_i64 = arith.constant 256 : i64  
  %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
  %7 = tt.make_tensor_ptr %arg1, [%c1_i64, %c64_i64], [%c64_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x32xf16>>
  %9 = tt.make_tensor_ptr %arg2, [%c1_i64, %c256_i64, %c64_i64], [%c256_i64, %c64_i64, %c1_i64], [%c0_i32, %c1_i32, %c2_i32] {order = array<i32: 2, 1, 0>} : <tensor<1x32x64xf16>>
  %10 = tt.advance %7, [%arg0, %c0_i32] : <tensor<64x32xf16>>
  %11 = tt.load %10 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x32xf16>>
  %res1:1 = scf.for %arg3 = %c0_i32 to %arg0 step %c32_i32 iter_args(%arg4 = %arg0) -> (i32) : i32 {
    %adv = tt.advance %9, [%arg4, %c0_i32] : <tensor<1x32x64xf16>>
    %load = tt.load %adv {boundaryCheck = array<i32: 1, 2>} : !tt.ptr<tensor<1x32x64xf16>>
    %reshape = tt.reshape %load : tensor<1x32x64xf16> -> tensor<32x64xf16>
    %dot = tt.dot %11, %reshape, %cst, inputPrecision = tf32 : tensor<64x32xf16> * tensor<32x64xf16> -> tensor<64x64xf32>
    %add = arith.addi %arg4, %c32_i32 : i32
    scf.yield %add : i32
  }
  %res2:1 = scf.for %arg3 = %c0_i32 to %arg0 step %c32_i32 iter_args(%arg4 = %arg0) -> (i32) : i32 {
    %adv = tt.advance %9, [%arg4, %c0_i32] : <tensor<1x32x64xf16>>
    %load = tt.load %adv {boundaryCheck = array<i32: 2, 1>} : !tt.ptr<tensor<1x32x64xf16>>
    %reshape = tt.reshape %load : tensor<1x32x64xf16> -> tensor<32x64xf16>
    %dot = tt.dot %11, %reshape, %cst, inputPrecision = tf32 : tensor<64x32xf16> * tensor<32x64xf16> -> tensor<64x64xf32>
    %add = arith.addi %arg4, %c32_i32 : i32
    scf.yield %add : i32
  }
  tt.return
  
}
// CHECK-LABEL: fuseLoadWithTrans4
// CHECK-NOT: tt.reshape
// CHECK: [[DIV1:%.*]] = arith.divui %c256_i64, %c64_i64 : i64
// CHECK: [[MUL11:%.*]] = arith.muli %c1_i64, [[DIV1]] : i64
// CHECK: [[ADD11:%.*]] = arith.addi [[MUL11]], %c256_i64 : i64
// CHECK: [[TRUNC1:%.*]] = arith.trunci [[DIV1]] : i64 to i32
// CHECK: [[MUL21:%.*]] = arith.muli %c0_i32, [[TRUNC1]] : i32
// CHECK: [[ADD21:%.*]] = arith.addi [[MUL21]], %c1_i32 : i32
// CHECK: [[PTR1:%.*]] = tt.make_tensor_ptr %arg2, [[[ADD11]], %c64_i64], [%c64_i64, %c1_i64], [[[ADD21]], %c2_i32] {order = array<i32: 1, 0>} : <tensor<32x64xf16>>
// CHECK: [[DIV2:%.*]] = arith.divui %c256_i64, %c64_i64 : i64
// CHECK: [[MUL12:%.*]] = arith.muli %c1_i64, [[DIV2]] : i64
// CHECK: [[ADD12:%.*]] = arith.addi [[MUL12]], %c256_i64 : i64
// CHECK: [[TRUNC2:%.*]] = arith.trunci [[DIV2]] : i64 to i32
// CHECK: [[MUL22:%.*]] = arith.muli %c0_i32, [[TRUNC2]] : i32
// CHECK: [[ADD22:%.*]] = arith.addi [[MUL22]], %c1_i32 : i32
// CHECK: [[PTR2:%.*]] = tt.make_tensor_ptr %arg2, [[[ADD12]], %c64_i64], [%c64_i64, %c1_i64], [[[ADD22]], %c2_i32] {order = array<i32: 1, 0>} : <tensor<32x64xf16>>
// CHECK: scf.for
// CHECK:   [[ADV:%.*]] = tt.advance [[PTR2]], {{.*}} : <tensor<32x64xf16>>
// CHECK:   [[LOAD_B1:%.*]] = tt.load [[ADV]] {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<32x64xf16>>
// CHECK:   tt.dot {{.*}}, [[LOAD_B1]], {{.*}}, inputPrecision = tf32 : tensor<64x32xf16> * tensor<32x64xf16> -> tensor<64x64xf32>
// CHECK:   scf.yield
// CHECK: scf.for
// CHECK:   [[ADV:%.*]] = tt.advance [[PTR1]], {{.*}} : <tensor<32x64xf16>>
// CHECK:   [[LOAD_B1:%.*]] = tt.load [[ADV]] {boundaryCheck = array<i32: 1, 0>} : !tt.ptr<tensor<32x64xf16>>
// CHECK:   tt.dot {{.*}}, [[LOAD_B1]], {{.*}}, inputPrecision = tf32 : tensor<64x32xf16> * tensor<32x64xf16> -> tensor<64x64xf32>
// CHECK:   scf.yield
