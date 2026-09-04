// RUN: triton-opt %s -split-input-file -triton-intel-fuse-reshape | FileCheck %s

// COM: Unit outermost dimension: merged extent is (1-1)*(1024/4) + 64 == 64.
tt.func public @fuseLoadWithReshape1(%arg0: tensor<256x32xbf16>, %arg1: !tt.ptr<bf16>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %c1_i64 = arith.constant 1 : i64
  %c4_i64 = arith.constant 4 : i64
  %c64_i32 = arith.constant 64 : i32
  %c1024_i32 = arith.constant 1024 : i32
  %c1024_i64 = arith.constant 1024 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c1_i32, %c64_i32, %c1024_i32], [%c1024_i64, %c4_i64, %c1_i64] : <bf16>, <1x32x256xbf16>
  %3 = tt.descriptor_load %0[%c2_i32, %c1_i32, %c0_i32]  : !tt.tensordesc<1x32x256xbf16> -> tensor<1x32x256xbf16>
  %4 = tt.reshape %3 : tensor<1x32x256xbf16> -> tensor<32x256xbf16>
  %5 = tt.dot %arg0, %4, %cst, inputPrecision = tf32 : tensor<256x32xbf16> * tensor<32x256xbf16> -> tensor<256x256xf32>
  tt.return
}
// CHECK-LABEL: fuseLoadWithReshape1
// CHECK-NOT: tt.reshape
// CHECK: [[DIV:%.*]] = arith.divui %c1024_i64, %c4_i64 : i64
// CHECK: [[TRUNC:%.*]] = arith.trunci [[DIV]] : i64 to i32
// CHECK-DAG: [[ONE:%.*]] = arith.constant 1 : i32
// CHECK: [[SUB:%.*]] = arith.subi %c1_i32, [[ONE]] : i32
// CHECK: [[MUL1:%.*]] = arith.muli [[SUB]], [[TRUNC]] : i32
// CHECK: [[ADD1:%.*]] = arith.addi [[MUL1]], %c64_i32 : i32
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor %arg1, [[[ADD1]], %c1024_i32], [%c4_i64, %c1_i64] : <bf16>, <32x256xbf16>
// CHECK: [[MUL2:%.*]] = arith.muli %c2_i32, [[TRUNC]] : i32
// CHECK: [[ADD2:%.*]] = arith.addi [[MUL2]], %c1_i32 : i32
// CHECK: [[LOAD_B:%.*]] = tt.descriptor_load [[DESC]][[[ADD2]], %c0_i32] : !tt.tensordesc<32x256xbf16> -> tensor<32x256xbf16>
// CHECK: tt.dot {{.*}}, [[LOAD_B]], {{.*}}, inputPrecision = tf32 : tensor<256x32xbf16> * tensor<32x256xbf16> -> tensor<256x256xf32>

// -----

// COM: Same, in a loop: (512-1)*(1024/1) + 1024 == 512*1024.
tt.func public @fuseLoadWithReshape2(%arg0: tensor<32x256xbf16>, %arg1: !tt.ptr<bf16>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c32_i32 = arith.constant 32 : i32
  %c512_i32 = arith.constant 512 : i32
  %c512_i64 = arith.constant 512 : i64
  %c1024_i32 = arith.constant 1024 : i32
  %c1024_i64 = arith.constant 1024 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c512_i32, %c1024_i32, %c32_i32], [%c1024_i64, %c1_i64, %c512_i64]: <bf16>, <1x256x32xbf16>
  %res:2 = scf.for %arg3 = %c0_i32 to %c1024_i32 step %c32_i32 iter_args(%arg4 = %cst, %arg5 = %c0_i32) -> (tensor<256x256xf32>, i32) : i32 {
    %1 = tt.descriptor_load %0[%c32_i32, %c32_i32, %c0_i32] : !tt.tensordesc<1x256x32xbf16> -> tensor<1x256x32xbf16>
    %2 = tt.reshape %1 : tensor<1x256x32xbf16> -> tensor<256x32xbf16>
    %4 = tt.dot %2, %arg0, %arg4, inputPrecision = tf32 : tensor<256x32xbf16> * tensor<32x256xbf16> -> tensor<256x256xf32>
    %5 = arith.addi %arg5, %c32_i32 : i32
    scf.yield %4, %5 : tensor<256x256xf32>, i32
  }
  tt.return
}
// CHECK-LABEL: fuseLoadWithReshape2
// CHECK-NOT: tt.reshape
// CHECK: [[DIV:%.*]] = arith.divui %c1024_i64, %c1_i64 : i64
// CHECK: [[TRUNC:%.*]] = arith.trunci [[DIV]] : i64 to i32
// CHECK-DAG: [[ONE:%.*]] = arith.constant 1 : i32
// CHECK: [[SUB:%.*]] = arith.subi %c512_i32, [[ONE]] : i32
// CHECK: [[MUL1:%.*]] = arith.muli [[SUB]], [[TRUNC]] : i32
// CHECK: [[ADD1:%.*]] = arith.addi [[MUL1]], %c1024_i32 : i32
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor %arg1, [[[ADD1]], %c32_i32], [%c1_i64, %c512_i64] : <bf16>, <256x32xbf16>
// CHECK: scf.for
// CHECK:   [[MUL2:%.*]] = arith.muli %c32_i32, [[TRUNC]] : i32
// CHECK:   [[ADD2:%.*]] = arith.addi [[MUL2]], %c32_i32 : i32
// CHECK:   [[LOAD_A:%.*]] = tt.descriptor_load [[DESC]][[[ADD2]], %c0_i32] : !tt.tensordesc<256x32xbf16> -> tensor<256x32xbf16>
// CHECK:   tt.dot [[LOAD_A]], {{.*}}, {{.*}}, inputPrecision = tf32 : tensor<256x32xbf16> * tensor<32x256xbf16> -> tensor<256x256xf32>

// -----

// Do not fuse when strides[0] is not provably divisible by strides[1]
// (e.g., padded strides as in github.com/intel/intel-xpu-backend-for-triton/issues/7030).
tt.func public @noFusePaddedStrides(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>, %G: i32, %K: i32, %M: i32, %stride0: i64, %stride1: i64) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%G, %K, %M], [%stride0, %stride1, %c1_i64] : <f32>, <1x16x16xf32>
  %1 = tt.descriptor_load %0[%c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<1x16x16xf32> -> tensor<1x16x16xf32>
  %2 = tt.reshape %1 : tensor<1x16x16xf32> -> tensor<16x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: noFusePaddedStrides
// CHECK: tt.descriptor_load
// CHECK: tt.reshape

// -----

// Do not fuse when the collapsed dimension's real extent is not provably a
// multiple of its block extent, even when strides[0] is provably divisible
// by strides[1]. Otherwise the per-dimension bounds check lost by fusion
// would let an over-sized block load spill into the next "row" of the
// outermost dimension (github.com/intel/intel-xpu-backend-for-triton/issues/7464).
tt.func public @noFuseNonDivisibleBlockExtent(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c8_i64 = arith.constant 8 : i64
  %c13_i32 = arith.constant 13 : i32
  %c5_i32 = arith.constant 5 : i32
  %c104_i64 = arith.constant 104 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<64x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c5_i32, %c13_i32, %c13_i32], [%c104_i64, %c8_i64, %c1_i64] : <f32>, <1x64x16xf32>
  %1 = tt.descriptor_load %0[%c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<1x64x16xf32> -> tensor<1x64x16xf32>
  %2 = tt.reshape %1 : tensor<1x64x16xf32> -> tensor<64x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<64x16xf32> * tensor<16x16xf32> -> tensor<64x16xf32>
  tt.return
}
// CHECK-LABEL: noFuseNonDivisibleBlockExtent
// CHECK: tt.descriptor_load
// CHECK: tt.reshape

// -----

// COM: Unit middle dimension: merged extent is (32-1)*128 + 128 == 4096.
tt.func public @fuseLoadWithReshapeMiddleDim(%arg0: tensor<128x256xbf16>, %arg1: !tt.ptr<bf16>) {
  %c0_i32 = arith.constant 0 : i32
  %c2_i32 = arith.constant 2 : i32
  %c3_i32 = arith.constant 3 : i32
  %c32_i32 = arith.constant 32 : i32
  %c128_i32 = arith.constant 128 : i32
  %c1024_i32 = arith.constant 1024 : i32
  %c1_i64 = arith.constant 1 : i64
  %c128_i64 = arith.constant 128 : i64
  %c4096_i64 = arith.constant 4096 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<64x256xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c1024_i32, %c32_i32, %c128_i32], [%c4096_i64, %c128_i64, %c1_i64] : <bf16>, <64x1x128xbf16>
  %1 = tt.descriptor_load %0[%c2_i32, %c3_i32, %c0_i32] : !tt.tensordesc<64x1x128xbf16> -> tensor<64x1x128xbf16>
  %2 = tt.reshape %1 : tensor<64x1x128xbf16> -> tensor<64x128xbf16>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<64x128xbf16> * tensor<128x256xbf16> -> tensor<64x256xf32>
  tt.return
}
// CHECK-LABEL: fuseLoadWithReshapeMiddleDim
// CHECK-NOT: tt.reshape
// CHECK: [[DIV:%.*]] = arith.divui %c128_i64, %c1_i64 : i64
// CHECK: [[TRUNC:%.*]] = arith.trunci [[DIV]] : i64 to i32
// CHECK-DAG: [[ONE:%.*]] = arith.constant 1 : i32
// CHECK: [[SUB:%.*]] = arith.subi %c32_i32, [[ONE]] : i32
// CHECK: [[MUL1:%.*]] = arith.muli [[SUB]], [[TRUNC]] : i32
// CHECK: [[ADD1:%.*]] = arith.addi [[MUL1]], %c128_i32 : i32
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor %arg1, [%c1024_i32, [[ADD1]]], [%c4096_i64, %c1_i64] : <bf16>, <64x128xbf16>
// CHECK: [[MUL2:%.*]] = arith.muli %c3_i32, [[TRUNC]] : i32
// CHECK: [[ADD2:%.*]] = arith.addi [[MUL2]], %c0_i32 : i32
// CHECK: [[LOAD:%.*]] = tt.descriptor_load [[DESC]][%c2_i32, [[ADD2]]] : !tt.tensordesc<64x128xbf16> -> tensor<64x128xbf16>
// CHECK: tt.dot [[LOAD]], {{.*}}, {{.*}}, inputPrecision = tf32 : tensor<64x128xbf16> * tensor<128x256xbf16> -> tensor<64x256xf32>

// -----

// COM: Do not fuse a unit middle dimension when the innermost extent is not
// COM: provably a multiple of its block extent.
tt.func public @noFuseMiddleDimNonDivisibleBlockExtent(%arg0: tensor<128x256xbf16>, %arg1: !tt.ptr<bf16>) {
  %c0_i32 = arith.constant 0 : i32
  %c3_i32 = arith.constant 3 : i32
  %c32_i32 = arith.constant 32 : i32
  %c130_i32 = arith.constant 130 : i32
  %c1024_i32 = arith.constant 1024 : i32
  %c1_i64 = arith.constant 1 : i64
  %c130_i64 = arith.constant 130 : i64
  %c4160_i64 = arith.constant 4160 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<64x256xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c1024_i32, %c32_i32, %c130_i32], [%c4160_i64, %c130_i64, %c1_i64] : <bf16>, <64x1x128xbf16>
  %1 = tt.descriptor_load %0[%c0_i32, %c3_i32, %c0_i32] : !tt.tensordesc<64x1x128xbf16> -> tensor<64x1x128xbf16>
  %2 = tt.reshape %1 : tensor<64x1x128xbf16> -> tensor<64x128xbf16>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<64x128xbf16> * tensor<128x256xbf16> -> tensor<64x256xf32>
  tt.return
}
// CHECK-LABEL: noFuseMiddleDimNonDivisibleBlockExtent
// CHECK: tt.descriptor_load
// CHECK: tt.reshape

// -----

// COM: Do not fuse (nor crash) a rank-reducing load: the block shape does not
// COM: match the loaded shape.
tt.func public @noFuseRankReducingDescriptor(%arg0: tensor<128x256xbf16>, %arg1: !tt.ptr<bf16>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c32_i32 = arith.constant 32 : i32
  %c128_i32 = arith.constant 128 : i32
  %c1024_i32 = arith.constant 1024 : i32
  %c1_i64 = arith.constant 1 : i64
  %c128_i64 = arith.constant 128 : i64
  %c4096_i64 = arith.constant 4096 : i64
  %c4194304_i64 = arith.constant 4194304 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<64x256xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c1_i32, %c1024_i32, %c32_i32, %c128_i32], [%c4194304_i64, %c4096_i64, %c128_i64, %c1_i64] : <bf16>, <1x64x1x128xbf16>
  %1 = tt.descriptor_load %0[%c0_i32, %c0_i32, %c1_i32, %c0_i32] : !tt.tensordesc<1x64x1x128xbf16> -> tensor<64x1x128xbf16>
  %2 = tt.reshape %1 : tensor<64x1x128xbf16> -> tensor<64x128xbf16>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<64x128xbf16> * tensor<128x256xbf16> -> tensor<64x256xf32>
  tt.return
}
// CHECK-LABEL: noFuseRankReducingDescriptor
// CHECK: tt.descriptor_load
// CHECK: tt.reshape

// -----

// COM: Do not fuse when the block shape puts the unit extent in a different
// COM: dimension than the loaded shape: the reshape drops 1, the descriptor 0.
tt.func public @noFuseBlockShapeMismatch(%arg0: tensor<128x256xbf16>, %arg1: !tt.ptr<bf16>) {
  %c0_i32 = arith.constant 0 : i32
  %c3_i32 = arith.constant 3 : i32
  %c64_i32 = arith.constant 64 : i32
  %c128_i32 = arith.constant 128 : i32
  %c1024_i32 = arith.constant 1024 : i32
  %c1_i64 = arith.constant 1 : i64
  %c128_i64 = arith.constant 128 : i64
  %c8192_i64 = arith.constant 8192 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<64x256xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c1024_i32, %c64_i32, %c128_i32], [%c8192_i64, %c128_i64, %c1_i64] : <bf16>, <1x64x128xbf16>
  %1 = tt.descriptor_load %0[%c0_i32, %c3_i32, %c0_i32] : !tt.tensordesc<1x64x128xbf16> -> tensor<64x1x128xbf16>
  %2 = tt.reshape %1 : tensor<64x1x128xbf16> -> tensor<64x128xbf16>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<64x128xbf16> * tensor<128x256xbf16> -> tensor<64x256xf32>
  tt.return
}
// CHECK-LABEL: noFuseBlockShapeMismatch
// CHECK: tt.descriptor_load
// CHECK: tt.reshape
