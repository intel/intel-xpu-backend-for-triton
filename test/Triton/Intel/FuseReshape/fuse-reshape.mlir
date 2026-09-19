// RUN: triton-opt %s -split-input-file -triton-intel-fuse-reshape | FileCheck %s

// COM: Unit outermost dimension. The merged extent is per *load*: the collapsed
// COM: index 2 clamps to max(1,1)-1 == 0, so the extent is 0*(1024/4) + 64 == 64
// COM: while the merged index stays unclamped at 2*256 + 1 == 513. 513 >= 64, so
// COM: the fused load pads exactly as the rank-3 one did (off_0 == 2 >= s_0).
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
// COM: Everything folds: the whole extent/offset computation is built with
// COM: `createOrFold` and every input here is constant.
// CHECK: arith.constant dense<0.000000e+00>
// CHECK: [[EXTENT:%.*]] = arith.constant 64 : i32
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor %arg1, [[[EXTENT]], %c1024_i32], [%c4_i64, %c1_i64] : <bf16>, <32x256xbf16>
// CHECK: [[INDEX:%.*]] = arith.constant 513 : i32
// CHECK: [[LOAD_B:%.*]] = tt.descriptor_load [[DESC]][[[INDEX]], %c0_i32] : !tt.tensordesc<32x256xbf16> -> tensor<32x256xbf16>
// CHECK: tt.dot {{.*}}, [[LOAD_B]], {{.*}}, inputPrecision = tf32 : tensor<256x32xbf16> * tensor<32x256xbf16> -> tensor<256x256xf32>

// -----

// COM: Same, in a loop. Collapsed index 32 is in range, so the extent is
// COM: 32*(1024/1) + 1024 == 33792 - not the whole collapsed dimension's
// COM: (512-1)*1024 + 1024. The merged index is 32*1024 + 32 == 32800.
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
// CHECK: arith.constant dense<0.000000e+00>
// CHECK: [[EXTENT:%.*]] = arith.constant 33792 : i32
// COM: The collapsed offset is loop-invariant, so the descriptor stays hoisted.
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor %arg1, [[[EXTENT]], %c32_i32], [%c1_i64, %c512_i64] : <bf16>, <256x32xbf16>
// CHECK: scf.for
// CHECK:   [[INDEX:%.*]] = arith.constant 32800 : i32
// CHECK:   [[LOAD_A:%.*]] = tt.descriptor_load [[DESC]][[[INDEX]], %c0_i32] : !tt.tensordesc<256x32xbf16> -> tensor<256x32xbf16>
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

// COM: Unit middle dimension. Collapsed index 3 is in range, so the extent is
// COM: 3*(128/1) + 128 == 512, not the dimension-wide (32-1)*128 + 128 == 4096;
// COM: the merged index is 3*128 + 0 == 384. The surface stays legal: width
// COM: 512*2 == 1024 bytes <= pitch 4096*2, and height 1024 <= 2^24.
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
// CHECK: arith.constant dense<0.000000e+00>
// CHECK: [[EXTENT:%.*]] = arith.constant 512 : i32
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor %arg1, [%c1024_i32, [[EXTENT]]], [%c4096_i64, %c1_i64] : <bf16>, <64x128xbf16>
// CHECK: [[INDEX:%.*]] = arith.constant 384 : i32
// CHECK: [[LOAD:%.*]] = tt.descriptor_load [[DESC]][%c2_i32, [[INDEX]]] : !tt.tensordesc<64x128xbf16> -> tensor<64x128xbf16>
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

// -----

// COM: Case 2 (issue #8001): a negative outer stride must not fuse. The ratio
// COM: is an unsigned division and `isDivisible` compares its operands as
// COM: unsigned, so without the sign reject the fused block reads as padding.
tt.func public @noFuseNegativeOuterStride(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c4_i64 = arith.constant 4 : i64
  %cm1024_i64 = arith.constant -1024 : i64
  %c8_i32 = arith.constant 8 : i32
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c8_i32, %c16_i32, %c16_i32], [%cm1024_i64, %c4_i64, %c1_i64] : <f32>, <1x16x16xf32>
  %1 = tt.descriptor_load %0[%c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<1x16x16xf32> -> tensor<1x16x16xf32>
  %2 = tt.reshape %1 : tensor<1x16x16xf32> -> tensor<16x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: noFuseNegativeOuterStride
// CHECK: tt.descriptor_load
// CHECK: tt.reshape

// -----

// COM: A merged-dimension stride of -2^32 is caught by the *sign* reject, which
// COM: runs first: this input says nothing about the `unsigned divisor`
// COM: narrowing guard (see the two tests below for that).
tt.func public @noFuseNegativeMergedStride(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %cm4294967296_i64 = arith.constant -4294967296 : i64
  %c8_i32 = arith.constant 8 : i32
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c8_i32, %c16_i32, %c16_i32], [%c1024_i64, %cm4294967296_i64, %c1_i64] : <f32>, <1x16x16xf32>
  %1 = tt.descriptor_load %0[%c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<1x16x16xf32> -> tensor<1x16x16xf32>
  %2 = tt.reshape %1 : tensor<1x16x16xf32> -> tensor<16x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: noFuseNegativeMergedStride
// CHECK: tt.descriptor_load
// CHECK: tt.reshape

// -----

// COM: `isDivisible` takes an `unsigned` divisor. A denominator of +2^32 narrows
// COM: to 0 and divides by zero, so `isProvablyDivisible` must decline instead
// COM: of narrowing. The numerator is genuinely not a multiple of it.
tt.func public @noFuseMergedStrideNarrowsToZero(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c5000000000_i64 = arith.constant 5000000000 : i64
  %c4294967296_i64 = arith.constant 4294967296 : i64
  %c8_i32 = arith.constant 8 : i32
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c8_i32, %c16_i32, %c16_i32], [%c5000000000_i64, %c4294967296_i64, %c1_i64] : <f32>, <1x16x16xf32>
  %1 = tt.descriptor_load %0[%c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<1x16x16xf32> -> tensor<1x16x16xf32>
  %2 = tt.reshape %1 : tensor<1x16x16xf32> -> tensor<16x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: noFuseMergedStrideNarrowsToZero
// CHECK: tt.descriptor_load
// CHECK: tt.reshape

// -----

// COM: The important narrowing case: a denominator of 2^32 + 1 narrows to 1 and
// COM: `isDivisible`'s `divisor == 1` early return reports a divisibility that
// COM: does not hold, so without the guard this input fuses with a truncated,
// COM: non-exact ratio and wrong addresses.
tt.func public @noFuseMergedStrideNarrowsToOne(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c5000000000_i64 = arith.constant 5000000000 : i64
  %c4294967297_i64 = arith.constant 4294967297 : i64
  %c8_i32 = arith.constant 8 : i32
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c8_i32, %c16_i32, %c16_i32], [%c5000000000_i64, %c4294967297_i64, %c1_i64] : <f32>, <1x16x16xf32>
  %1 = tt.descriptor_load %0[%c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<1x16x16xf32> -> tensor<1x16x16xf32>
  %2 = tt.reshape %1 : tensor<1x16x16xf32> -> tensor<16x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: noFuseMergedStrideNarrowsToOne
// CHECK: tt.descriptor_load
// CHECK: tt.reshape

// -----

// COM: Canary for the range test's boundary: a denominator of exactly
// COM: UINT32_MAX still fits an `unsigned`, so an exactly divisible numerator
// COM: must keep fusing.
tt.func public @fuseMergedStrideAtUnsignedMax(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c8589934590_i64 = arith.constant 8589934590 : i64
  %c4294967295_i64 = arith.constant 4294967295 : i64
  %c8_i32 = arith.constant 8 : i32
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c8_i32, %c16_i32, %c16_i32], [%c8589934590_i64, %c4294967295_i64, %c1_i64] : <f32>, <1x16x16xf32>
  %1 = tt.descriptor_load %0[%c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<1x16x16xf32> -> tensor<1x16x16xf32>
  %2 = tt.reshape %1 : tensor<1x16x16xf32> -> tensor<16x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: fuseMergedStrideAtUnsignedMax
// CHECK-NOT: tt.reshape
// CHECK: tt.make_tensor_descriptor
// CHECK: tt.descriptor_load

// -----

// COM: Regression test for placing the narrowing guard *inside*
// COM: `isProvablyDivisible`'s constant-denominator branch rather than ahead of
// COM: the whole function: a numerator that is literally muli(denominator, k) is
// COM: exact for a stride of any magnitude and must still fuse.
tt.func public @fuseStructurallyDivisibleWideStride(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c2_i64 = arith.constant 2 : i64
  %c4294967296_i64 = arith.constant 4294967296 : i64
  %c8_i32 = arith.constant 8 : i32
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %stride0 = arith.muli %c4294967296_i64, %c2_i64 : i64
  %0 = tt.make_tensor_descriptor %arg1, [%c8_i32, %c16_i32, %c16_i32], [%stride0, %c4294967296_i64, %c1_i64] : <f32>, <1x16x16xf32>
  %1 = tt.descriptor_load %0[%c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<1x16x16xf32> -> tensor<1x16x16xf32>
  %2 = tt.reshape %1 : tensor<1x16x16xf32> -> tensor<16x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: fuseStructurallyDivisibleWideStride
// CHECK-NOT: tt.reshape
// CHECK: tt.make_tensor_descriptor
// CHECK: tt.descriptor_load

// -----

// COM: Case 3 (issue #8001): an empty collapsed dimension makes the rank-3 load
// COM: pure padding. The old extent (0-1)*d + s_md declared -128, which a
// COM: surface field reads as 4294967168.
tt.func public @noFuseEmptyOuterDim(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c4_i64 = arith.constant 4 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c0_i32, %c16_i32, %c16_i32], [%c1024_i64, %c4_i64, %c1_i64] : <f32>, <1x16x16xf32>
  %1 = tt.descriptor_load %0[%c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<1x16x16xf32> -> tensor<1x16x16xf32>
  %2 = tt.reshape %1 : tensor<1x16x16xf32> -> tensor<16x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: noFuseEmptyOuterDim
// CHECK: tt.descriptor_load
// CHECK: tt.reshape

// -----

// COM: Same on the middle branch, where it is worse: the rank-3 load is entirely
// COM: padding (off_1 < s_1 == 0 is false) while the fused one read real data.
tt.func public @noFuseEmptyMiddleDim(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c16_i64 = arith.constant 16 : i64
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c16_i32, %c0_i32, %c16_i32], [%c16_i64, %c16_i64, %c1_i64] : <f32>, <16x1x16xf32>
  %1 = tt.descriptor_load %0[%c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<16x1x16xf32> -> tensor<16x1x16xf32>
  %2 = tt.reshape %1 : tensor<16x1x16xf32> -> tensor<16x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: noFuseEmptyMiddleDim
// CHECK: tt.descriptor_load
// CHECK: tt.reshape

// -----

// COM: A zero extent on the *merged* dimension passes the existing divisibility
// COM: guard (0 % n == 0) but makes the merged extent zero, which the surface
// COM: field emits as `0 - 1`.
tt.func public @noFuseZeroMergedExtent(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c4_i64 = arith.constant 4 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c8_i32 = arith.constant 8 : i32
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c8_i32, %c0_i32, %c16_i32], [%c1024_i64, %c4_i64, %c1_i64] : <f32>, <1x16x16xf32>
  %1 = tt.descriptor_load %0[%c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<1x16x16xf32> -> tensor<1x16x16xf32>
  %2 = tt.reshape %1 : tensor<1x16x16xf32> -> tensor<16x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: noFuseZeroMergedExtent
// CHECK: tt.descriptor_load
// CHECK: tt.reshape

// -----

// COM: INT32_MIN is the regression test for clamping the shape *before*
// COM: subtracting one: maxsi(s-1, 0) would compute INT32_MIN-1, wrap to
// COM: INT32_MAX and preserve the garbage. It must reject, not wrap.
tt.func public @noFuseIntMinCollapsedShape(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c4_i64 = arith.constant 4 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %cmin_i32 = arith.constant -2147483648 : i32
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%cmin_i32, %c16_i32, %c16_i32], [%c1024_i64, %c4_i64, %c1_i64] : <f32>, <1x16x16xf32>
  %1 = tt.descriptor_load %0[%c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<1x16x16xf32> -> tensor<1x16x16xf32>
  %2 = tt.reshape %1 : tensor<1x16x16xf32> -> tensor<16x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: noFuseIntMinCollapsedShape
// CHECK: tt.descriptor_load
// CHECK: tt.reshape

// -----

// COM: Case 4 (issue #8001) verbatim: a middle collapse whose pitch (strides[0])
// COM: is narrower than a whole collapsed row declared a 512-byte-wide surface
// COM: over a 64-byte pitch. With the per-load extent the reported input is
// COM: legal and *still fuses*: width == pitch == 64 bytes.
tt.func public @fuseMiddleDimNarrowPitch(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c16_i64 = arith.constant 16 : i64
  %c4_i32 = arith.constant 4 : i32
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c16_i32, %c4_i32, %c16_i32], [%c16_i64, %c16_i64, %c1_i64] : <f32>, <16x1x16xf32>
  %1 = tt.descriptor_load %0[%c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<16x1x16xf32> -> tensor<16x1x16xf32>
  %2 = tt.reshape %1 : tensor<16x1x16xf32> -> tensor<16x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: fuseMiddleDimNarrowPitch
// CHECK-NOT: tt.reshape
// CHECK: tt.make_tensor_descriptor
// CHECK: tt.descriptor_load

// -----

// COM: The same descriptor with a non-zero collapsed offset: the clamped extent
// COM: is 32 elements = 128 bytes over a 64-byte pitch, so it must decline. This
// COM: is the check that stops a different offset from re-creating case 4.
tt.func public @noFuseMiddleDimWidthPastPitch(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c1_i64 = arith.constant 1 : i64
  %c16_i64 = arith.constant 16 : i64
  %c4_i32 = arith.constant 4 : i32
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c16_i32, %c4_i32, %c16_i32], [%c16_i64, %c16_i64, %c1_i64] : <f32>, <16x1x16xf32>
  %1 = tt.descriptor_load %0[%c0_i32, %c1_i32, %c0_i32] : !tt.tensordesc<16x1x16xf32> -> tensor<16x1x16xf32>
  %2 = tt.reshape %1 : tensor<16x1x16xf32> -> tensor<16x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: noFuseMiddleDimWidthPastPitch
// CHECK: tt.descriptor_load
// CHECK: tt.reshape

// -----

// COM: Load-bearing: tightening the width can *newly* violate `width >= 64`
// COM: bytes, and no pass between here and the `triton_gen.2Dblockload` verifier
// COM: has a lower bound on the width. 8 elements = 32 bytes; the pitch (64
// COM: bytes) is legal, so only the width rule can fire.
tt.func public @noFuseMiddleDimWidthBelow64(%arg0: tensor<8x16xf32>, %arg1: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c8_i64 = arith.constant 8 : i64
  %c16_i64 = arith.constant 16 : i64
  %c4_i32 = arith.constant 4 : i32
  %c8_i32 = arith.constant 8 : i32
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c16_i32, %c4_i32, %c8_i32], [%c16_i64, %c8_i64, %c1_i64] : <f32>, <16x1x8xf32>
  %1 = tt.descriptor_load %0[%c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<16x1x8xf32> -> tensor<16x1x8xf32>
  %2 = tt.reshape %1 : tensor<16x1x8xf32> -> tensor<16x8xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x8xf32> * tensor<8x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: noFuseMiddleDimWidthBelow64
// CHECK: tt.descriptor_load
// CHECK: tt.reshape

// -----

// COM: Width alignment: 33 bf16 elements = 66 bytes, not a multiple of
// COM: max(4, elemBytes) == 4. Not load-bearing - `MaterializeBlockPointer`
// COM: sees this rule too and merely withholds `block_io` - but kept for
// COM: uniformity with the other width rules.
tt.func public @noFuseMiddleDimWidthMisaligned(%arg0: tensor<16x16xbf16>, %arg1: !tt.ptr<bf16>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c1_i64 = arith.constant 1 : i64
  %c17_i64 = arith.constant 17 : i64
  %c40_i64 = arith.constant 40 : i64
  %c4_i32 = arith.constant 4 : i32
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c16_i32, %c4_i32, %c16_i32], [%c40_i64, %c17_i64, %c1_i64] : <bf16>, <16x1x16xbf16>
  %1 = tt.descriptor_load %0[%c0_i32, %c1_i32, %c0_i32] : !tt.tensordesc<16x1x16xbf16> -> tensor<16x1x16xbf16>
  %2 = tt.reshape %1 : tensor<16x1x16xbf16> -> tensor<16x16xbf16>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xbf16> * tensor<16x16xbf16> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: noFuseMiddleDimWidthMisaligned
// CHECK: tt.descriptor_load
// CHECK: tt.reshape

// -----

// COM: The retained pitch, load-bearing: `LowerTo2DBlockLoad`'s static pitch
// COM: check folds an `extract_desc` result, which never folds, so it is dead
// COM: code for descriptor loads. 8 elements = 32 bytes < 64. The collapsed
// COM: offset is dynamic, so the width cannot fold: this pins the per-field
// COM: behaviour - a constant-bad pitch must reject even when the extent is not
// COM: foldable.
tt.func public @noFuseMiddleDimPitchBelow64(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>, %off: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c8_i64 = arith.constant 8 : i64
  %c16_i64 = arith.constant 16 : i64
  %c4_i32 = arith.constant 4 : i32
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c16_i32, %c4_i32, %c16_i32], [%c8_i64, %c16_i64, %c1_i64] : <f32>, <16x1x16xf32>
  %1 = tt.descriptor_load %0[%c0_i32, %off, %c0_i32] : !tt.tensordesc<16x1x16xf32> -> tensor<16x1x16xf32>
  %2 = tt.reshape %1 : tensor<16x1x16xf32> -> tensor<16x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: noFuseMiddleDimPitchBelow64
// CHECK: tt.descriptor_load
// CHECK: tt.reshape

// -----

// COM: Pitch alignment: 21 f32 elements = 84 bytes, not a multiple of 16. Like
// COM: the width alignment rule this duplicates a `MaterializeBlockPointer`
// COM: check; it is here for uniformity. The collapsed offset is dynamic so only
// COM: the pitch can fire.
tt.func public @noFuseMiddleDimPitchMisaligned(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>, %off: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c16_i64 = arith.constant 16 : i64
  %c21_i64 = arith.constant 21 : i64
  %c4_i32 = arith.constant 4 : i32
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c16_i32, %c4_i32, %c16_i32], [%c21_i64, %c16_i64, %c1_i64] : <f32>, <16x1x16xf32>
  %1 = tt.descriptor_load %0[%c0_i32, %off, %c0_i32] : !tt.tensordesc<16x1x16xf32> -> tensor<16x1x16xf32>
  %2 = tt.reshape %1 : tensor<16x1x16xf32> -> tensor<16x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: noFuseMiddleDimPitchMisaligned
// CHECK: tt.descriptor_load
// CHECK: tt.reshape

// -----

// COM: Load-bearing 24-bit cap on the width. strides[0] is dynamic so the pitch
// COM: cannot fold - otherwise no legal pitch could accommodate this width and
// COM: the two checks could not be told apart. 4194320 f32 elements = 16777280
// COM: bytes > 2^24.
tt.func public @noFuseMiddleDimWidthAbove24Bit(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>, %stride0: i64) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c4194320_i64 = arith.constant 4194320 : i64
  %c4_i32 = arith.constant 4 : i32
  %c16_i32 = arith.constant 16 : i32
  %c4194320_i32 = arith.constant 4194320 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c16_i32, %c4_i32, %c4194320_i32], [%stride0, %c4194320_i64, %c1_i64] : <f32>, <16x1x16xf32>
  %1 = tt.descriptor_load %0[%c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<16x1x16xf32> -> tensor<16x1x16xf32>
  %2 = tt.reshape %1 : tensor<16x1x16xf32> -> tensor<16x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: noFuseMiddleDimWidthAbove24Bit
// CHECK: tt.descriptor_load
// CHECK: tt.reshape

// -----

// COM: Load-bearing 24-bit cap on the promoted pitch: 4194308 f32 elements =
// COM: 16777232 bytes > 2^24, and a multiple of 16 so only the cap can fire. The
// COM: collapsed offset is dynamic, so the width cannot fold.
tt.func public @noFuseMiddleDimPitchAbove24Bit(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>, %off: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c16_i64 = arith.constant 16 : i64
  %c4194308_i64 = arith.constant 4194308 : i64
  %c4_i32 = arith.constant 4 : i32
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c16_i32, %c4_i32, %c16_i32], [%c4194308_i64, %c16_i64, %c1_i64] : <f32>, <16x1x16xf32>
  %1 = tt.descriptor_load %0[%c0_i32, %off, %c0_i32] : !tt.tensordesc<16x1x16xf32> -> tensor<16x1x16xf32>
  %2 = tt.reshape %1 : tensor<16x1x16xf32> -> tensor<16x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: noFuseMiddleDimPitchAbove24Bit
// CHECK: tt.descriptor_load
// CHECK: tt.reshape

// -----

// COM: The promoted pitch's sign. These two cannot *isolate* the sign reject:
// COM: `pitchBytes = strides[0] * elemBytes` is non-positive in both, so the
// COM: signed `pitchBytes < 64` rule fires as well and deleting the sign reject
// COM: would leave them green. They cover the reject outcome only; the sign
// COM: reject's justification is keeping a non-positive value away from
// COM: `isProvablyDivisible`.
tt.func public @noFuseMiddleDimZeroPitch(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %c16_i64 = arith.constant 16 : i64
  %c4_i32 = arith.constant 4 : i32
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c16_i32, %c4_i32, %c16_i32], [%c0_i64, %c16_i64, %c1_i64] : <f32>, <16x1x16xf32>
  %1 = tt.descriptor_load %0[%c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<16x1x16xf32> -> tensor<16x1x16xf32>
  %2 = tt.reshape %1 : tensor<16x1x16xf32> -> tensor<16x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: noFuseMiddleDimZeroPitch
// CHECK: tt.descriptor_load
// CHECK: tt.reshape

// -----

// COM: See @noFuseMiddleDimZeroPitch: same non-isolating caveat.
tt.func public @noFuseMiddleDimNegativePitch(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %cm64_i64 = arith.constant -64 : i64
  %c16_i64 = arith.constant 16 : i64
  %c4_i32 = arith.constant 4 : i32
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c16_i32, %c4_i32, %c16_i32], [%cm64_i64, %c16_i64, %c1_i64] : <f32>, <16x1x16xf32>
  %1 = tt.descriptor_load %0[%c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<16x1x16xf32> -> tensor<16x1x16xf32>
  %2 = tt.reshape %1 : tensor<16x1x16xf32> -> tensor<16x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: noFuseMiddleDimNegativePitch
// CHECK: tt.descriptor_load
// CHECK: tt.reshape

// -----

// COM: Boundary pair for the surface height on the outer branch, where the
// COM: merged extent *is* the height. 16 + 16777200 == 2^24 exactly: fuses.
tt.func public @fuseOuterHeightAt24Bit(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c1_i64 = arith.constant 1 : i64
  %c16_i64 = arith.constant 16 : i64
  %c8_i32 = arith.constant 8 : i32
  %c16_i32 = arith.constant 16 : i32
  %c16777200_i32 = arith.constant 16777200 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c8_i32, %c16777200_i32, %c16_i32], [%c16_i64, %c1_i64, %c1_i64] : <f32>, <1x16x16xf32>
  %1 = tt.descriptor_load %0[%c1_i32, %c0_i32, %c0_i32] : !tt.tensordesc<1x16x16xf32> -> tensor<1x16x16xf32>
  %2 = tt.reshape %1 : tensor<1x16x16xf32> -> tensor<16x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: fuseOuterHeightAt24Bit
// CHECK-NOT: tt.reshape
// CHECK: tt.make_tensor_descriptor
// CHECK: tt.descriptor_load

// -----

// COM: The other side of @fuseOuterHeightAt24Bit: 16 + 16777216 == 2^24 + 16.
tt.func public @noFuseOuterHeightAbove24Bit(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c1_i64 = arith.constant 1 : i64
  %c16_i64 = arith.constant 16 : i64
  %c8_i32 = arith.constant 8 : i32
  %c16_i32 = arith.constant 16 : i32
  %c16777216_i32 = arith.constant 16777216 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c8_i32, %c16777216_i32, %c16_i32], [%c16_i64, %c1_i64, %c1_i64] : <f32>, <1x16x16xf32>
  %1 = tt.descriptor_load %0[%c1_i32, %c0_i32, %c0_i32] : !tt.tensordesc<1x16x16xf32> -> tensor<1x16x16xf32>
  %2 = tt.reshape %1 : tensor<1x16x16xf32> -> tensor<16x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: noFuseOuterHeightAbove24Bit
// CHECK: tt.descriptor_load
// CHECK: tt.reshape

// -----

// COM: The middle branch *promotes* shapes[0] to the surface height, a field
// COM: `LowerTo2DBlockLoad::wouldOverflow` never inspects. 2^24 fuses...
tt.func public @fuseMiddleHeightAt24Bit(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c16_i64 = arith.constant 16 : i64
  %c4_i32 = arith.constant 4 : i32
  %c16_i32 = arith.constant 16 : i32
  %c16777216_i32 = arith.constant 16777216 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c16777216_i32, %c4_i32, %c16_i32], [%c16_i64, %c16_i64, %c1_i64] : <f32>, <16x1x16xf32>
  %1 = tt.descriptor_load %0[%c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<16x1x16xf32> -> tensor<16x1x16xf32>
  %2 = tt.reshape %1 : tensor<16x1x16xf32> -> tensor<16x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: fuseMiddleHeightAt24Bit
// CHECK-NOT: tt.reshape
// CHECK: tt.make_tensor_descriptor
// CHECK: tt.descriptor_load

// -----

// COM: ...2^24 + 1 does not.
tt.func public @noFuseMiddleHeightAbove24Bit(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c16_i64 = arith.constant 16 : i64
  %c4_i32 = arith.constant 4 : i32
  %c16_i32 = arith.constant 16 : i32
  %c16777217_i32 = arith.constant 16777217 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c16777217_i32, %c4_i32, %c16_i32], [%c16_i64, %c16_i64, %c1_i64] : <f32>, <16x1x16xf32>
  %1 = tt.descriptor_load %0[%c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<16x1x16xf32> -> tensor<16x1x16xf32>
  %2 = tt.reshape %1 : tensor<16x1x16xf32> -> tensor<16x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: noFuseMiddleHeightAbove24Bit
// CHECK: tt.descriptor_load
// CHECK: tt.reshape

// -----

// COM: The promoted height's sign, the third case-3 face: shapes[0] == 0 makes
// COM: the *retained* surface dimension empty, and the height reaches the
// COM: hardware as `0 - 1`.
tt.func public @noFuseMiddleZeroHeight(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c16_i64 = arith.constant 16 : i64
  %c4_i32 = arith.constant 4 : i32
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c0_i32, %c4_i32, %c16_i32], [%c16_i64, %c16_i64, %c1_i64] : <f32>, <16x1x16xf32>
  %1 = tt.descriptor_load %0[%c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<16x1x16xf32> -> tensor<16x1x16xf32>
  %2 = tt.reshape %1 : tensor<16x1x16xf32> -> tensor<16x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: noFuseMiddleZeroHeight
// CHECK: tt.descriptor_load
// CHECK: tt.reshape

// -----

// COM: See @noFuseMiddleZeroHeight.
tt.func public @noFuseMiddleNegativeHeight(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c16_i64 = arith.constant 16 : i64
  %cm8_i32 = arith.constant -8 : i32
  %c4_i32 = arith.constant 4 : i32
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%cm8_i32, %c4_i32, %c16_i32], [%c16_i64, %c16_i64, %c1_i64] : <f32>, <16x1x16xf32>
  %1 = tt.descriptor_load %0[%c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<16x1x16xf32> -> tensor<16x1x16xf32>
  %2 = tt.reshape %1 : tensor<16x1x16xf32> -> tensor<16x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: noFuseMiddleNegativeHeight
// CHECK: tt.descriptor_load
// CHECK: tt.reshape

// -----

// COM: The truncated ratio is i32 but it is computed in i64: 2^32 wraps to 0,
// COM: which silently rewrites every merged offset to `off_md`. The collapsed
// COM: offset is dynamic so nothing else can fold and reject first.
tt.func public @noFuseRatioAboveInt32(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>, %off: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c4294967296_i64 = arith.constant 4294967296 : i64
  %c8_i32 = arith.constant 8 : i32
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c8_i32, %c16_i32, %c16_i32], [%c4294967296_i64, %c1_i64, %c1_i64] : <f32>, <1x16x16xf32>
  %1 = tt.descriptor_load %0[%off, %c0_i32, %c0_i32] : !tt.tensordesc<1x16x16xf32> -> tensor<1x16x16xf32>
  %2 = tt.reshape %1 : tensor<1x16x16xf32> -> tensor<16x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: noFuseRatioAboveInt32
// CHECK: tt.descriptor_load
// CHECK: tt.reshape

// -----

// COM: The merged *offset* is a second, independent i32 obligation: the extent
// COM: here is legal (3*65536 + 16, clamped by shapes[0] == 4) but
// COM: 65536 * 65536 == 2^32 wraps to 0 and addresses the wrong row.
tt.func public @noFuseMergedOffsetMulWraps(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c65536_i64 = arith.constant 65536 : i64
  %c4_i32 = arith.constant 4 : i32
  %c16_i32 = arith.constant 16 : i32
  %c65536_i32 = arith.constant 65536 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c4_i32, %c16_i32, %c16_i32], [%c65536_i64, %c1_i64, %c1_i64] : <f32>, <1x16x16xf32>
  %1 = tt.descriptor_load %0[%c65536_i32, %c0_i32, %c0_i32] : !tt.tensordesc<1x16x16xf32> -> tensor<1x16x16xf32>
  %2 = tt.reshape %1 : tensor<1x16x16xf32> -> tensor<16x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: noFuseMergedOffsetMulWraps
// CHECK: tt.descriptor_load
// CHECK: tt.reshape

// -----

// COM: The addition alone can wrap even when the product fits:
// COM: 32767 * 65536 == 2147418112 <= INT32_MAX, + 100000 does not.
tt.func public @noFuseMergedOffsetAddWraps(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c65536_i64 = arith.constant 65536 : i64
  %c4_i32 = arith.constant 4 : i32
  %c16_i32 = arith.constant 16 : i32
  %c32767_i32 = arith.constant 32767 : i32
  %c100000_i32 = arith.constant 100000 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c4_i32, %c16_i32, %c16_i32], [%c65536_i64, %c1_i64, %c1_i64] : <f32>, <1x16x16xf32>
  %1 = tt.descriptor_load %0[%c32767_i32, %c100000_i32, %c0_i32] : !tt.tensordesc<1x16x16xf32> -> tensor<1x16x16xf32>
  %2 = tt.reshape %1 : tensor<1x16x16xf32> -> tensor<16x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: noFuseMergedOffsetAddWraps
// CHECK: tt.descriptor_load
// CHECK: tt.reshape

// -----

// COM: Canary for the i32 range checks being inclusive: a ratio of exactly
// COM: INT32_MAX with a zero collapsed offset is representable and must fuse.
tt.func public @fuseRatioAtInt32Max(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c2147483647_i64 = arith.constant 2147483647 : i64
  %c8_i32 = arith.constant 8 : i32
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c8_i32, %c16_i32, %c16_i32], [%c2147483647_i64, %c1_i64, %c1_i64] : <f32>, <1x16x16xf32>
  %1 = tt.descriptor_load %0[%c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<1x16x16xf32> -> tensor<1x16x16xf32>
  %2 = tt.reshape %1 : tensor<1x16x16xf32> -> tensor<16x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: fuseRatioAtInt32Max
// CHECK-NOT: tt.reshape
// CHECK: tt.make_tensor_descriptor
// CHECK: tt.descriptor_load

// -----

// COM: The byte conversions themselves are computed in int64 and can overflow
// COM: it: 2^62 f32 elements is 2^64 bytes. There is no `extentBytes`
// COM: counterpart - the extent is already known to fit i32 by then.
tt.func public @noFuseMiddlePitchBytesOverflow(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c16_i64 = arith.constant 16 : i64
  %chuge_i64 = arith.constant 4611686018427387904 : i64
  %c4_i32 = arith.constant 4 : i32
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c16_i32, %c4_i32, %c16_i32], [%chuge_i64, %c16_i64, %c1_i64] : <f32>, <16x1x16xf32>
  %1 = tt.descriptor_load %0[%c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<16x1x16xf32> -> tensor<16x1x16xf32>
  %2 = tt.reshape %1 : tensor<16x1x16xf32> -> tensor<16x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: noFuseMiddlePitchBytesOverflow
// CHECK: tt.descriptor_load
// CHECK: tt.reshape

// -----

// COM: A negative collapsed offset is legal input (the rank-3 load pads
// COM: entirely) and must keep fusing: the clamp pins the extent to shapes[md]
// COM: while the merged offset stays unclamped at -3 * 256, so the fused load
// COM: pads the same way.
tt.func public @fuseNegativeCollapsedOffset(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %cm3_i32 = arith.constant -3 : i32
  %c1_i64 = arith.constant 1 : i64
  %c256_i64 = arith.constant 256 : i64
  %c8_i32 = arith.constant 8 : i32
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c8_i32, %c16_i32, %c16_i32], [%c256_i64, %c1_i64, %c1_i64] : <f32>, <1x16x16xf32>
  %1 = tt.descriptor_load %0[%cm3_i32, %c0_i32, %c0_i32] : !tt.tensordesc<1x16x16xf32> -> tensor<1x16x16xf32>
  %2 = tt.reshape %1 : tensor<1x16x16xf32> -> tensor<16x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: fuseNegativeCollapsedOffset
// CHECK-NOT: tt.reshape
// CHECK: tt.make_tensor_descriptor
// CHECK: tt.descriptor_load

// -----

// COM: A dynamic collapsed shape still fuses, and the zero-forcing factor is
// COM: `muli(merged, extui(cmpi sgt))`, never an `arith.select`:
// COM: `ttgi::isDivisible` understands `muli` (either operand) but not `select`,
// COM: so a select would silently cost the fused load its `block_io` attribute.
tt.func public @fuseDynamicCollapsedShape(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>, %shape0: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c256_i64 = arith.constant 256 : i64
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%shape0, %c16_i32, %c16_i32], [%c256_i64, %c1_i64, %c1_i64] : <f32>, <1x16x16xf32>
  %1 = tt.descriptor_load %0[%c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<1x16x16xf32> -> tensor<1x16x16xf32>
  %2 = tt.reshape %1 : tensor<1x16x16xf32> -> tensor<16x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: fuseDynamicCollapsedShape
// CHECK: arith.cmpi sgt
// CHECK: arith.extui
// CHECK: arith.muli
// COM: The zero-forcing factor is the last thing computed before the new
// COM: descriptor, so this range is where a `select` formulation would land.
// CHECK-NOT: arith.select
// CHECK: tt.make_tensor_descriptor
// CHECK: tt.descriptor_load
// CHECK-NOT: tt.reshape

// -----

// COM: The extent now depends on the collapsed offset, so the descriptor can no
// COM: longer always stay where it was. A loop-carried offset forces it *into*
// COM: the loop.
tt.func public @fuseSinksDescriptorIntoLoop(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>) -> tensor<16x16xf32> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c8_i32 = arith.constant 8 : i32
  %c1_i64 = arith.constant 1 : i64
  %c256_i64 = arith.constant 256 : i64
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c8_i32, %c16_i32, %c16_i32], [%c256_i64, %c1_i64, %c1_i64] : <f32>, <1x16x16xf32>
  %1 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %cst) -> (tensor<16x16xf32>) : i32 {
    %2 = tt.descriptor_load %0[%iv, %c0_i32, %c0_i32] : !tt.tensordesc<1x16x16xf32> -> tensor<1x16x16xf32>
    %3 = tt.reshape %2 : tensor<1x16x16xf32> -> tensor<16x16xf32>
    %4 = tt.dot %3, %arg0, %acc, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
    scf.yield %4 : tensor<16x16xf32>
  }
  tt.return %1 : tensor<16x16xf32>
}
// CHECK-LABEL: fuseSinksDescriptorIntoLoop
// CHECK: scf.for
// CHECK: tt.make_tensor_descriptor
// CHECK: tt.descriptor_load
// CHECK-NOT: tt.reshape

// -----

// COM: The other `DominanceInfo` arm: an offset that already dominates the
// COM: descriptor leaves it hoisted, even though the load is in a loop.
tt.func public @fuseKeepsDescriptorOutsideLoop(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>) -> tensor<16x16xf32> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c8_i32 = arith.constant 8 : i32
  %c1_i64 = arith.constant 1 : i64
  %c256_i64 = arith.constant 256 : i64
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %pid = tt.get_program_id x : i32
  %0 = tt.make_tensor_descriptor %arg1, [%c8_i32, %c16_i32, %c16_i32], [%c256_i64, %c1_i64, %c1_i64] : <f32>, <1x16x16xf32>
  %1 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %cst) -> (tensor<16x16xf32>) : i32 {
    %2 = tt.descriptor_load %0[%pid, %c0_i32, %c0_i32] : !tt.tensordesc<1x16x16xf32> -> tensor<1x16x16xf32>
    %3 = tt.reshape %2 : tensor<1x16x16xf32> -> tensor<16x16xf32>
    %4 = tt.dot %3, %arg0, %acc, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
    scf.yield %4 : tensor<16x16xf32>
  }
  tt.return %1 : tensor<16x16xf32>
}
// CHECK-LABEL: fuseKeepsDescriptorOutsideLoop
// CHECK: tt.make_tensor_descriptor
// CHECK: scf.for
// CHECK: tt.descriptor_load
// CHECK-NOT: tt.reshape

// -----

// COM: A zero-extent block reaches `ttgi::isDivisible` as the divisor, which
// COM: divides by it unguarded: this input raised SIGFPE in the compiler before
// COM: the extent was required to be positive.
tt.func public @noFuseZeroBlockExtent(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c4_i64 = arith.constant 4 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c8_i32 = arith.constant 8 : i32
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<0x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c8_i32, %c16_i32, %c16_i32], [%c1024_i64, %c4_i64, %c1_i64] : <f32>, <1x0x16xf32>
  %1 = tt.descriptor_load %0[%c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<1x0x16xf32> -> tensor<1x0x16xf32>
  %2 = tt.reshape %1 : tensor<1x0x16xf32> -> tensor<0x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<0x16xf32> * tensor<16x16xf32> -> tensor<0x16xf32>
  tt.return
}
// CHECK-LABEL: noFuseZeroBlockExtent
// CHECK: tt.descriptor_load
// CHECK: tt.reshape

// -----

// COM: An exact stride ratio whose denominator exceeds `unsigned`. The ratio is
// COM: 8589934594/4294967297 == 2, computed exactly on the folded constants:
// COM: narrowing the denominator for `ttgi::isDivisible` would have declined it.
// COM: Neither collapsed stride is a surface field for a middle collapse, so the
// COM: surface stays legal (pitch 1024*4, width 16*4 == 64).
tt.func public @fuseWideExactStrideRatio(%arg0: tensor<16x256xf32>, %arg1: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c16_i32 = arith.constant 16 : i32
  %c64_i32 = arith.constant 64 : i32
  %c1024_i64 = arith.constant 1024 : i64
  %den = arith.constant 4294967297 : i64
  %num = arith.constant 8589934594 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<64x256xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c64_i32, %c16_i32, %c16_i32], [%c1024_i64, %num, %den] : <f32>, <64x1x16xf32>
  %1 = tt.descriptor_load %0[%c0_i32, %c1_i32, %c0_i32] : !tt.tensordesc<64x1x16xf32> -> tensor<64x1x16xf32>
  %2 = tt.reshape %1 : tensor<64x1x16xf32> -> tensor<64x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<64x16xf32> * tensor<16x256xf32> -> tensor<64x256xf32>
  tt.return
}
// CHECK-LABEL: fuseWideExactStrideRatio
// CHECK-NOT: tt.reshape
// COM: extent == clamp(1,0,15)*2 + 16 == 18, merged index == 1*2 + 0 == 2.
// CHECK: arith.constant dense<0.000000e+00>
// CHECK: [[EXTENT:%.*]] = arith.constant 18 : i32
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor %arg1, [%c64_i32, [[EXTENT]]], [%c1024_i64, %c4294967297_i64] : <f32>, <64x16xf32>
// CHECK: [[INDEX:%.*]] = arith.constant 2 : i32
// CHECK: [[LOAD:%.*]] = tt.descriptor_load [[DESC]][%c0_i32, [[INDEX]]] : !tt.tensordesc<64x16xf32> -> tensor<64x16xf32>
// CHECK: tt.dot [[LOAD]], {{.*}}, {{.*}}, inputPrecision = tf32 : tensor<64x16xf32> * tensor<16x256xf32> -> tensor<64x256xf32>

// -----

// COM: Same shape, but 8589934595 % 4294967297 != 0. The exact remainder check
// COM: must still decline; it widens what fuses, it does not skip the test.
tt.func public @noFuseWideInexactStrideRatio(%arg0: tensor<16x256xf32>, %arg1: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c16_i32 = arith.constant 16 : i32
  %c64_i32 = arith.constant 64 : i32
  %c1024_i64 = arith.constant 1024 : i64
  %den = arith.constant 4294967297 : i64
  %num = arith.constant 8589934595 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<64x256xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c64_i32, %c16_i32, %c16_i32], [%c1024_i64, %num, %den] : <f32>, <64x1x16xf32>
  %1 = tt.descriptor_load %0[%c0_i32, %c1_i32, %c0_i32] : !tt.tensordesc<64x1x16xf32> -> tensor<64x1x16xf32>
  %2 = tt.reshape %1 : tensor<64x1x16xf32> -> tensor<64x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<64x16xf32> * tensor<16x256xf32> -> tensor<64x256xf32>
  tt.return
}
// CHECK-LABEL: noFuseWideInexactStrideRatio
// CHECK: tt.descriptor_load
// CHECK: tt.reshape
