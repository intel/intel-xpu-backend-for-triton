// RUN: triton-opt %s -split-input-file -triton-intel-fuse-reshape | FileCheck %s

// COM: Unit outermost dimension. The merged extent is per *load*: the collapsed
// COM: index 2 clamps to max(1,1)-1 == 0, so the extent is 0*(1024/4) + 64 == 64.
// COM: That index is out of range (off_0 == 2 >= s_0 == 1), so the guard forces
// COM: the merged index to the extent and the fused load pads exactly as the
// COM: rank-3 one did.
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
// COM: Not 513: the guard folds the out-of-range index to the extent. Matched by
// COM: constant name, since folding it materializes more than one `64`.
// CHECK: [[LOAD_B:%.*]] = tt.descriptor_load [[DESC]][%c64_i32{{[_0-9]*}}, %c0_i32] : !tt.tensordesc<32x256xbf16> -> tensor<32x256xbf16>
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
// COM: *shape* is dynamic, so the merged extent cannot be bracketed: this pins
// COM: the per-field behaviour - a constant-bad pitch must reject even when the
// COM: extent is not foldable. It cannot isolate the rule, because the
// COM: empty-dimension width floor is 64 bytes and so exceeds any pitch this
// COM: rule rejects; no input can separate the two.
tt.func public @noFuseMiddleDimPitchBelow64(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>, %shape1: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c8_i64 = arith.constant 8 : i64
  %c16_i64 = arith.constant 16 : i64
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c16_i32, %shape1, %c16_i32], [%c8_i64, %c16_i64, %c1_i64] : <f32>, <16x1x16xf32>
  %1 = tt.descriptor_load %0[%c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<16x1x16xf32> -> tensor<16x1x16xf32>
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
// COM: check; it is here for uniformity. The collapsed *shape* is dynamic so the
// COM: merged extent cannot be bracketed, and 84 bytes is above both 64 and the
// COM: 64-byte width floor, so only the pitch alignment can fire.
tt.func public @noFuseMiddleDimPitchMisaligned(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>, %shape1: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c16_i64 = arith.constant 16 : i64
  %c21_i64 = arith.constant 21 : i64
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c16_i32, %shape1, %c16_i32], [%c21_i64, %c16_i64, %c1_i64] : <f32>, <16x1x16xf32>
  %1 = tt.descriptor_load %0[%c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<16x1x16xf32> -> tensor<16x1x16xf32>
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
// COM: collapsed offset is dynamic, so the width is bracketed by its clamp
// COM: endpoints rather than folded - [64, 256] bytes here, legal at both ends
// COM: and well under this pitch, so no width rule interferes.
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
// COM: and the guard forces the merged offset to that extent, so the fused load
// COM: pads the same way - through the upper bound instead of the lower one.
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

// COM: A dynamic collapsed shape still fuses, and an empty one does not get an
// COM: extent of 0 - every surface field is emitted as `extent - 1`, so a zero
// COM: would declare a 16MB surface - but the smallest legal one, here one
// COM: 16-element block of rows. The form is
// COM: `merged * nonEmpty + floor * (1 - nonEmpty)` with
// COM: `nonEmpty = extui(cmpi sgt)`, never an `arith.select`: `ttgi::isDivisible`
// COM: understands `muli` (either operand) and `addi` (both) but not `select`, so
// COM: a select would silently change what `DescriptorLoadOpConversion` decides
// COM: about the emitted shape. That argument is about the stride-one dimension
// COM: and so does not bite on this case, where the collapse of dimension 0 puts
// COM: the merged extent in the height; `fuseDynamicCollapsedShapeMiddleDim` is
// COM: the case it bites on. The form is kept uniform across both branches.
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
// CHECK-NOT: tt.reshape
// CHECK: [[MERGED:%.*]] = arith.addi {{%.*}}, %c16_i32
// CHECK: [[SGT:%.*]] = arith.cmpi sgt, %arg2, %c0_i32
// CHECK: [[NONEMPTY:%.*]] = arith.extui [[SGT]]
// CHECK: [[EMPTY:%.*]] = arith.subi %c1_i32, [[NONEMPTY]]
// CHECK: [[KEEP:%.*]] = arith.muli [[MERGED]], [[NONEMPTY]]
// COM: The floor: one 16-row block of the surface height.
// CHECK: [[FLOOR:%.*]] = arith.muli [[EMPTY]], %c16_i32
// CHECK: [[EXTENT:%.*]] = arith.addi [[KEEP]], [[FLOOR]]
// COM: The extent is the last thing computed before the new descriptor, so this
// COM: range is where a `select` formulation would land.
// CHECK-NOT: arith.select
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor %arg1, [[[EXTENT]], %c16_i32]
// COM: The index guard lands *after* the descriptor, and is multiply-based for
// COM: the same reason, so this range must be select-free too. The load index of
// COM: this case is a constant 0, so the guard's in-range term folds away and
// COM: what is left is the padding term: an empty dimension reads at exactly the
// COM: extent, which is out of range on both lowering paths.
// CHECK-NOT: arith.select
// CHECK: [[OUT:%.*]] = arith.subi %c1_i32, {{%.*}}
// CHECK: [[IDX:%.*]] = arith.muli [[OUT]], [[EXTENT]]
// CHECK: tt.descriptor_load [[DESC]][[[IDX]], %c0_i32]
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

// COM: Do not fuse, and do not crash, when the merged dimension of the block
// COM: has zero extent: the boundary check divides the merged shape by it
// COM: (issues/8073).
tt.func public @noFuseZeroExtentMergedDim(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c8_i32 = arith.constant 8 : i32
  %c16_i32 = arith.constant 16 : i32
  %c1_i64 = arith.constant 1 : i64
  %c4_i64 = arith.constant 4 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<0x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c8_i32, %c16_i32, %c16_i32], [%c1024_i64, %c4_i64, %c1_i64] : <f32>, <1x0x16xf32>
  %1 = tt.descriptor_load %0[%c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<1x0x16xf32> -> tensor<1x0x16xf32>
  %2 = tt.reshape %1 : tensor<1x0x16xf32> -> tensor<0x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<0x16xf32> * tensor<16x16xf32> -> tensor<0x16xf32>
  tt.return
}
// CHECK-LABEL: noFuseZeroExtentMergedDim
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

// -----

// COM: A dynamic collapsed index cannot be proven in range, so the merged index
// COM: is guarded: `(merged * inRange) + ((1 - inRange) * extent)`, which forces
// COM: it to the extent - and so makes the load pad, as the rank-3 form does -
// COM: whenever the index is out of range (issue #8070). Multiplies, not an
// COM: `arith.select`, for the same `ttgi::isDivisible` reason as the extent.
tt.func public @fuseGuardsDynamicOuterIndex(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>, %off: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c256_i64 = arith.constant 256 : i64
  %c8_i32 = arith.constant 8 : i32
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c8_i32, %c16_i32, %c16_i32], [%c256_i64, %c1_i64, %c1_i64] : <f32>, <1x16x16xf32>
  %1 = tt.descriptor_load %0[%off, %c0_i32, %c0_i32] : !tt.tensordesc<1x16x16xf32> -> tensor<1x16x16xf32>
  %2 = tt.reshape %1 : tensor<1x16x16xf32> -> tensor<16x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: fuseGuardsDynamicOuterIndex
// CHECK-NOT: tt.reshape
// CHECK: [[EXTENT:%.*]] = arith.addi
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor %arg1, [[[EXTENT]], %c16_i32]
// CHECK: [[MERGED:%.*]] = arith.muli %arg2, %c256_i32
// CHECK: [[GE:%.*]] = arith.cmpi sge, %arg2, %c0_i32
// CHECK: [[LT:%.*]] = arith.cmpi slt, %arg2, %c8_i32
// CHECK: [[AND:%.*]] = arith.andi [[GE]], [[LT]]
// CHECK: [[IN:%.*]] = arith.extui [[AND]]
// CHECK: [[OUT:%.*]] = arith.subi %c1_i32, [[IN]]
// CHECK: [[KEEP:%.*]] = arith.muli [[MERGED]], [[IN]]
// CHECK: [[PAD:%.*]] = arith.muli [[OUT]], [[EXTENT]]
// CHECK: [[IDX:%.*]] = arith.addi [[KEEP]], [[PAD]]
// CHECK-NOT: arith.select
// CHECK: tt.descriptor_load [[DESC]][[[IDX]], %c0_i32]

// -----

// COM: A merged dimension that is empty only at runtime (`%n == 0`). The merged
// COM: extent `clamp(%off) * 256 + 0` is then 0 for any `%off <= 0`, so the
// COM: floor and the guard both key on `shapes[1] > 0` too: without it in the
// COM: floor, a zero extent declares a 16MB surface; without it in the guard, an
// COM: in-range `%off` of 0 merges to row 0 of the floor and reads real data.
// COM: `shapes[0]` is a constant, so its own `sgt` folds away and the merged one
// COM: is all that is left in either place.
tt.func public @fuseDynamicMergedShape(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>, %n: i32, %off: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c256_i64 = arith.constant 256 : i64
  %c8_i32 = arith.constant 8 : i32
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %rows = arith.muli %n, %c16_i32 : i32
  %0 = tt.make_tensor_descriptor %arg1, [%c8_i32, %rows, %c16_i32], [%c256_i64, %c1_i64, %c1_i64] : <f32>, <1x16x16xf32>
  %1 = tt.descriptor_load %0[%off, %c0_i32, %c0_i32] : !tt.tensordesc<1x16x16xf32> -> tensor<1x16x16xf32>
  %2 = tt.reshape %1 : tensor<1x16x16xf32> -> tensor<16x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: fuseDynamicMergedShape
// CHECK-NOT: tt.reshape
// CHECK: [[ROWS:%.*]] = arith.muli %arg2, %c16_i32
// CHECK: [[MERGED:%.*]] = arith.addi {{%.*}}, [[ROWS]]
// CHECK: [[SGT:%.*]] = arith.cmpi sgt, [[ROWS]], %c0_i32
// CHECK: [[NONEMPTY:%.*]] = arith.extui [[SGT]]
// CHECK: [[EMPTY:%.*]] = arith.subi %c1_i32, [[NONEMPTY]]
// CHECK: [[KEEP:%.*]] = arith.muli [[MERGED]], [[NONEMPTY]]
// CHECK: [[FLOOR:%.*]] = arith.muli [[EMPTY]], %c16_i32
// CHECK: [[EXTENT:%.*]] = arith.addi [[KEEP]], [[FLOOR]]
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor %arg1, [[[EXTENT]], %c16_i32]
// CHECK: [[MIDX:%.*]] = arith.muli %arg3, %c256_i32
// CHECK: [[GE:%.*]] = arith.cmpi sge, %arg3, %c0_i32
// CHECK: [[LT:%.*]] = arith.cmpi slt, %arg3, %c8_i32
// CHECK: [[AND:%.*]] = arith.andi [[GE]], [[LT]]
// CHECK: [[INAND:%.*]] = arith.andi [[AND]], [[SGT]]
// CHECK: [[IN:%.*]] = arith.extui [[INAND]]
// CHECK: [[OUT:%.*]] = arith.subi %c1_i32, [[IN]]
// CHECK: [[KEEPIDX:%.*]] = arith.muli [[MIDX]], [[IN]]
// CHECK: [[PAD:%.*]] = arith.muli [[OUT]], [[EXTENT]]
// CHECK: [[IDX:%.*]] = arith.addi [[KEEPIDX]], [[PAD]]
// CHECK-NOT: arith.select
// CHECK: tt.descriptor_load [[DESC]][[[IDX]], %c0_i32]

// -----

// COM: The same guard on a middle collapse, where the guarded index becomes the
// COM: index `satisfies2DBlockReadAlignment` queries. This pins the shape that
// COM: keeps `block_io` reachable - `isDivisible` handles `muli`/`addi` but not
// COM: `arith.select`; that it is actually kept is checked end-to-end by
// COM: test_host_tensor_descriptor.py, since this RUN line stops at this pass.
tt.func public @fuseGuardsDynamicMiddleIndex(%arg0: tensor<16x256xf32>, %arg1: !tt.ptr<f32>, %off: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c16_i32 = arith.constant 16 : i32
  %c64_i32 = arith.constant 64 : i32
  %c1_i64 = arith.constant 1 : i64
  %c16_i64 = arith.constant 16 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<64x256xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c64_i32, %c16_i32, %c16_i32], [%c1024_i64, %c16_i64, %c1_i64] : <f32>, <64x1x16xf32>
  %1 = tt.descriptor_load %0[%c0_i32, %off, %c0_i32] : !tt.tensordesc<64x1x16xf32> -> tensor<64x1x16xf32>
  %2 = tt.reshape %1 : tensor<64x1x16xf32> -> tensor<64x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<64x16xf32> * tensor<16x256xf32> -> tensor<64x256xf32>
  tt.return
}
// CHECK-LABEL: fuseGuardsDynamicMiddleIndex
// CHECK-NOT: tt.reshape
// CHECK: [[EXTENT:%.*]] = arith.addi
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor %arg1, [%c64_i32, [[EXTENT]]]
// CHECK: [[MERGED:%.*]] = arith.muli %arg2, %c16_i32
// CHECK: [[GE:%.*]] = arith.cmpi sge, %arg2, %c0_i32
// CHECK: [[LT:%.*]] = arith.cmpi slt, %arg2, %c16_i32
// CHECK: [[AND:%.*]] = arith.andi [[GE]], [[LT]]
// CHECK: [[IN:%.*]] = arith.extui [[AND]]
// CHECK: [[OUT:%.*]] = arith.subi %c1_i32, [[IN]]
// CHECK: [[KEEP:%.*]] = arith.muli [[MERGED]], [[IN]]
// CHECK: [[PAD:%.*]] = arith.muli [[OUT]], [[EXTENT]]
// CHECK: [[IDX:%.*]] = arith.addi [[KEEP]], [[PAD]]
// CHECK-NOT: arith.select
// CHECK: tt.descriptor_load [[DESC]][%c0_i32, [[IDX]]]

// -----

// COM: Overlapping strides: ratio 16 < shapes[md] 32, so index 4 == shapes[cd]
// COM: merges to 4*16 + 0 == 64, inside the extent 3*16 + 32 == 80, and read
// COM: real data before the guard. It now folds to the extent instead, which
// COM: pads as the rank-3 form does.
tt.func public @fuseGuardsOutOfRangeIndexOverlappingStrides(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c4_i32 = arith.constant 4 : i32
  %c16_i32 = arith.constant 16 : i32
  %c32_i32 = arith.constant 32 : i32
  %c1_i64 = arith.constant 1 : i64
  %c16_i64 = arith.constant 16 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<32x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c4_i32, %c32_i32, %c16_i32], [%c16_i64, %c1_i64, %c1_i64] : <f32>, <1x32x16xf32>
  %1 = tt.descriptor_load %0[%c4_i32, %c0_i32, %c0_i32] : !tt.tensordesc<1x32x16xf32> -> tensor<1x32x16xf32>
  %2 = tt.reshape %1 : tensor<1x32x16xf32> -> tensor<32x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<32x16xf32> * tensor<16x16xf32> -> tensor<32x16xf32>
  tt.return
}
// CHECK-LABEL: fuseGuardsOutOfRangeIndexOverlappingStrides
// CHECK-NOT: tt.reshape
// CHECK: [[EXTENT:%.*]] = arith.constant 80 : i32
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor %arg1, [[[EXTENT]], %c16_i32], [%c1_i64, %c1_i64] : <f32>, <32x16xf32>
// COM: Not 64: everything folds, and the guarded index is the extent. Matched by
// COM: constant name, since folding it materializes more than one `80`.
// CHECK: tt.descriptor_load [[DESC]][%c80_i32{{[_0-9]*}}, %c0_i32]

// -----

// COM: Both indices out of range, in opposite directions: -1 * 32 + 32 == 0 read
// COM: row 0 before the guard. Exercises the `sge` half of it - the other
// COM: out-of-range cases only reach the `slt` half.
tt.func public @fuseGuardsCancellingOutOfRangeIndices(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %cm1_i32 = arith.constant -1 : i32
  %c4_i32 = arith.constant 4 : i32
  %c16_i32 = arith.constant 16 : i32
  %c32_i32 = arith.constant 32 : i32
  %c1_i64 = arith.constant 1 : i64
  %c32_i64 = arith.constant 32 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<32x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c4_i32, %c32_i32, %c16_i32], [%c32_i64, %c1_i64, %c1_i64] : <f32>, <1x32x16xf32>
  %1 = tt.descriptor_load %0[%cm1_i32, %c32_i32, %c0_i32] : !tt.tensordesc<1x32x16xf32> -> tensor<1x32x16xf32>
  %2 = tt.reshape %1 : tensor<1x32x16xf32> -> tensor<32x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<32x16xf32> * tensor<16x16xf32> -> tensor<32x16xf32>
  tt.return
}
// CHECK-LABEL: fuseGuardsCancellingOutOfRangeIndices
// CHECK-NOT: tt.reshape
// COM: Extent and index are both 32 here, so both are matched by constant name.
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor %arg1, [%c32_i32{{[_0-9]*}}, %c16_i32], [%c1_i64, %c1_i64] : <f32>, <32x16xf32>
// COM: Not 0, which is what -1*32 + 32 merged to before the guard.
// CHECK: tt.descriptor_load [[DESC]][%c32_i32{{[_0-9]*}}, %c0_i32]

// -----

// COM: A negative collapsed index whose merged dimension is *in* range: the base
// COM: merges to -1*32 + 24 == -8, and the per-element tile coordinate (added
// COM: before the signed bounds check) lifts rows 8..15 back inside the extent.
// COM: The guard forces the whole tile to pad.
tt.func public @fuseGuardsNegativeIndexLiftedByTile(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %cm1_i32 = arith.constant -1 : i32
  %c4_i32 = arith.constant 4 : i32
  %c16_i32 = arith.constant 16 : i32
  %c24_i32 = arith.constant 24 : i32
  %c32_i32 = arith.constant 32 : i32
  %c1_i64 = arith.constant 1 : i64
  %c32_i64 = arith.constant 32 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c4_i32, %c32_i32, %c16_i32], [%c32_i64, %c1_i64, %c1_i64] : <f32>, <1x16x16xf32>
  %1 = tt.descriptor_load %0[%cm1_i32, %c24_i32, %c0_i32] : !tt.tensordesc<1x16x16xf32> -> tensor<1x16x16xf32>
  %2 = tt.reshape %1 : tensor<1x16x16xf32> -> tensor<16x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: fuseGuardsNegativeIndexLiftedByTile
// CHECK-NOT: tt.reshape
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor %arg1, [%c32_i32{{[_0-9]*}}, %c16_i32], [%c1_i64, %c1_i64] : <f32>, <16x16xf32>
// COM: Not -8, the pre-guard merged base.
// CHECK: tt.descriptor_load [[DESC]][%c32_i32{{[_0-9]*}}, %c0_i32]

// -----

// COM: A provably in-range index pays nothing: the guard folds away entirely and
// COM: the load keeps the plain merged index 3*256 + 0 == 768.
tt.func public @fuseInRangeIndexFoldsGuard(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c3_i32 = arith.constant 3 : i32
  %c8_i32 = arith.constant 8 : i32
  %c16_i32 = arith.constant 16 : i32
  %c1_i64 = arith.constant 1 : i64
  %c256_i64 = arith.constant 256 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c8_i32, %c16_i32, %c16_i32], [%c256_i64, %c1_i64, %c1_i64] : <f32>, <1x16x16xf32>
  %1 = tt.descriptor_load %0[%c3_i32, %c0_i32, %c0_i32] : !tt.tensordesc<1x16x16xf32> -> tensor<1x16x16xf32>
  %2 = tt.reshape %1 : tensor<1x16x16xf32> -> tensor<16x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: fuseInRangeIndexFoldsGuard
// CHECK-NOT: tt.reshape
// CHECK: [[EXTENT:%.*]] = arith.constant 784 : i32
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor %arg1, [[[EXTENT]], %c16_i32], [%c1_i64, %c1_i64] : <f32>, <16x16xf32>
// CHECK-NOT: arith.cmpi
// CHECK: tt.descriptor_load [[DESC]][%c768_i32{{[_0-9]*}}, %c0_i32]

// -----

// COM: A folded negative index on the *merged* dimension is not expressible in
// COM: the merged form - it pads only some rows - so decline instead (#8070).
// COM: Only literal/folded negatives are caught: `getFoldedConstantValue` does
// COM: not see through a computed one.
tt.func public @noFuseNegativeMergedIndex(%arg0: tensor<64x64xf32>, %arg1: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %cm1_i32 = arith.constant -1 : i32
  %c2_i32 = arith.constant 2 : i32
  %c64_i32 = arith.constant 64 : i32
  %c1_i64 = arith.constant 1 : i64
  %c64_i64 = arith.constant 64 : i64
  %c4096_i64 = arith.constant 4096 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c2_i32, %c64_i32, %c64_i32], [%c4096_i64, %c64_i64, %c1_i64] : <f32>, <1x64x64xf32>
  %1 = tt.descriptor_load %0[%c1_i32, %cm1_i32, %c0_i32] : !tt.tensordesc<1x64x64xf32> -> tensor<1x64x64xf32>
  %2 = tt.reshape %1 : tensor<1x64x64xf32> -> tensor<64x64xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<64x64xf32> * tensor<64x64xf32> -> tensor<64x64xf32>
  tt.return
}
// CHECK-LABEL: noFuseNegativeMergedIndex
// CHECK: tt.descriptor_load
// CHECK: tt.reshape

// -----

// COM: The reviewer's reproducer for a *dynamic* collapsed offset. Same
// COM: descriptor as `fuseMiddleDimNarrowPitch`, which fuses at offset 0, but the
// COM: offset is now a function argument, so no single extent folds. The clamp
// COM: range is [0, 3] and the merged extent is monotone in it, so its endpoints
// COM: bracket every extent this load can declare: [16, 64] elements = [64, 256]
// COM: bytes. The upper end exceeds the 64-byte pitch, so this must decline -
// COM: before this bracket existed every width rule was dead here and the load
// COM: fused straight into case 4 of issue #8001.
tt.func public @noFuseMiddleDimDynamicOffsetWidthPastPitch(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>, %off: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c16_i64 = arith.constant 16 : i64
  %c4_i32 = arith.constant 4 : i32
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c16_i32, %c4_i32, %c16_i32], [%c16_i64, %c16_i64, %c1_i64] : <f32>, <16x1x16xf32>
  %1 = tt.descriptor_load %0[%c0_i32, %off, %c0_i32] : !tt.tensordesc<16x1x16xf32> -> tensor<16x1x16xf32>
  %2 = tt.reshape %1 : tensor<16x1x16xf32> -> tensor<16x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: noFuseMiddleDimDynamicOffsetWidthPastPitch
// CHECK: tt.descriptor_load
// CHECK: tt.reshape

// -----

// COM: The bracket is a bound, not a veto: the same load with a pitch wide enough
// COM: for its whole clamp range (strides[0] == 64 elements == 256 bytes) still
// COM: fuses. Guards against the endpoint rules over-declining every dynamic
// COM: collapsed offset, which is the motivating case for this pass.
tt.func public @fuseMiddleDimDynamicOffsetWithinPitch(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>, %off: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c16_i64 = arith.constant 16 : i64
  %c64_i64 = arith.constant 64 : i64
  %c4_i32 = arith.constant 4 : i32
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c16_i32, %c4_i32, %c16_i32], [%c64_i64, %c16_i64, %c1_i64] : <f32>, <16x1x16xf32>
  %1 = tt.descriptor_load %0[%c0_i32, %off, %c0_i32] : !tt.tensordesc<16x1x16xf32> -> tensor<16x1x16xf32>
  %2 = tt.reshape %1 : tensor<16x1x16xf32> -> tensor<16x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: fuseMiddleDimDynamicOffsetWithinPitch
// CHECK: tt.make_tensor_descriptor
// CHECK: tt.descriptor_load
// CHECK-NOT: tt.reshape

// -----

// COM: The *lower* endpoint of the bracket, which is the collapsed index 0 and so
// COM: is `shapes[2]` alone: 8 f32 elements = 32 bytes < 64. The pitch (64 bytes)
// COM: and the 64-byte empty-dimension floor are both legal, so only the minimum
// COM: width rule can fire. This is why the bracket needs both endpoints - the
// COM: maximum alone is legal here.
tt.func public @noFuseMiddleDimDynamicOffsetWidthBelow64(%arg0: tensor<8x16xf32>, %arg1: !tt.ptr<f32>, %off: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c8_i64 = arith.constant 8 : i64
  %c16_i64 = arith.constant 16 : i64
  %c4_i32 = arith.constant 4 : i32
  %c8_i32 = arith.constant 8 : i32
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c16_i32, %c4_i32, %c8_i32], [%c16_i64, %c8_i64, %c1_i64] : <f32>, <16x1x8xf32>
  %1 = tt.descriptor_load %0[%c0_i32, %off, %c0_i32] : !tt.tensordesc<16x1x8xf32> -> tensor<16x1x8xf32>
  %2 = tt.reshape %1 : tensor<16x1x8xf32> -> tensor<16x8xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x8xf32> * tensor<8x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: noFuseMiddleDimDynamicOffsetWidthBelow64
// CHECK: tt.descriptor_load
// CHECK: tt.reshape

// -----

// COM: The same bracket on the height, where an outermost collapse puts the
// COM: merged extent. The clamp range is [0, 7] and the ratio is 4194304, so the
// COM: upper endpoint is 7*4194304 + 16 == 29360144 rows > 2^24. No pitch or
// COM: width rule applies to a collapse of dimension 0, so only the height cap
// COM: can fire.
tt.func public @noFuseOuterDynamicOffsetHeightAbove24Bit(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>, %off: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c4194304_i64 = arith.constant 4194304 : i64
  %c8_i32 = arith.constant 8 : i32
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c8_i32, %c16_i32, %c16_i32], [%c4194304_i64, %c1_i64, %c1_i64] : <f32>, <1x16x16xf32>
  %1 = tt.descriptor_load %0[%off, %c0_i32, %c0_i32] : !tt.tensordesc<1x16x16xf32> -> tensor<1x16x16xf32>
  %2 = tt.reshape %1 : tensor<1x16x16xf32> -> tensor<16x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: noFuseOuterDynamicOffsetHeightAbove24Bit
// CHECK: tt.descriptor_load
// CHECK: tt.reshape

// -----

// COM: The empty-collapsed-dimension width floor is checked like any other field,
// COM: and unlike the merged extent it is a compile-time constant, so it is
// COM: checked even when nothing else folds. The collapsed shape is dynamic here,
// COM: so there is no extent bracket at all; the floor is one 32-element block ==
// COM: 128 bytes, over a 64-byte pitch. Reachable only where the non-empty width
// COM: is illegal by the same rule (`shapes[2]` is a positive multiple of the
// COM: block extent, so it is never narrower than the floor) - declining is still
// COM: better than declaring a surface wider than its own pitch.
tt.func public @noFuseMiddleDimFloorPastPitch(%arg0: tensor<32x16xf32>, %arg1: !tt.ptr<f32>, %shape1: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c16_i64 = arith.constant 16 : i64
  %c32_i64 = arith.constant 32 : i64
  %c16_i32 = arith.constant 16 : i32
  %c32_i32 = arith.constant 32 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c16_i32, %shape1, %c32_i32], [%c16_i64, %c32_i64, %c1_i64] : <f32>, <16x1x32xf32>
  %1 = tt.descriptor_load %0[%c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<16x1x32xf32> -> tensor<16x1x32xf32>
  %2 = tt.reshape %1 : tensor<16x1x32xf32> -> tensor<16x32xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x32xf32> * tensor<32x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: noFuseMiddleDimFloorPastPitch
// CHECK: tt.descriptor_load
// CHECK: tt.reshape

// -----

// COM: The middle-dimension counterpart of `fuseDynamicCollapsedShape`, where the
// COM: `isDivisible` argument for the multiply-based form is load-bearing: the
// COM: merged extent *is* the stride-one dimension here, so the emitted shape is
// COM: what `DescriptorLoadOpConversion` asks about when it classifies the mask.
// COM: Both terms stay multiples of the 16-element block extent - `merged`
// COM: because `shapes[2]` is, and the floor by construction - so the
// COM: classification is the one the unfused load got.
tt.func public @fuseDynamicCollapsedShapeMiddleDim(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>, %shape1: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c16_i64 = arith.constant 16 : i64
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c16_i32, %shape1, %c16_i32], [%c16_i64, %c16_i64, %c1_i64] : <f32>, <16x1x16xf32>
  %1 = tt.descriptor_load %0[%c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<16x1x16xf32> -> tensor<16x1x16xf32>
  %2 = tt.reshape %1 : tensor<16x1x16xf32> -> tensor<16x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: fuseDynamicCollapsedShapeMiddleDim
// CHECK-NOT: tt.reshape
// CHECK: [[MERGED:%.*]] = arith.addi {{%.*}}, %c16_i32
// CHECK: [[SGT:%.*]] = arith.cmpi sgt, %arg2, %c0_i32
// CHECK: [[NONEMPTY:%.*]] = arith.extui [[SGT]]
// CHECK: [[EMPTY:%.*]] = arith.subi %c1_i32, [[NONEMPTY]]
// CHECK: [[KEEP:%.*]] = arith.muli [[MERGED]], [[NONEMPTY]]
// COM: The floor here is a width, so it is also the 64-byte minimum: 16 f32
// COM: elements, which is a multiple of the 16-element block extent.
// CHECK: [[FLOOR:%.*]] = arith.muli [[EMPTY]], %c16_i32
// CHECK: [[EXTENT:%.*]] = arith.addi [[KEEP]], [[FLOOR]]
// CHECK-NOT: arith.select
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor %arg1, [%c16_i32, [[EXTENT]]]
// CHECK-NOT: arith.select
// CHECK: [[OUT:%.*]] = arith.subi %c1_i32, {{%.*}}
// CHECK: [[IDX:%.*]] = arith.muli [[OUT]], [[EXTENT]]
// CHECK: tt.descriptor_load [[DESC]][%c0_i32, [[IDX]]]
// CHECK-NOT: tt.reshape
