// RUN: triton-opt %s -split-input-file -triton-intel-fuse-reshape | FileCheck %s

// COM: Unit outermost dimension. The collapsed index 3 goes into the base -
// COM: 3 * 65536 elements - and the other two dimensions keep their own shapes
// COM: and strides, so nothing about the new descriptor is derived. Everything
// COM: folds: the whole computation is built with `createOrFold`.
tt.func public @fuseOuterDim(%arg0: tensor<256x32xbf16>, %arg1: !tt.ptr<bf16>) {
  %c0_i32 = arith.constant 0 : i32
  %c3_i32 = arith.constant 3 : i32
  %c8_i32 = arith.constant 8 : i32
  %c1_i64 = arith.constant 1 : i64
  %c64_i32 = arith.constant 64 : i32
  %c1024_i32 = arith.constant 1024 : i32
  %c1024_i64 = arith.constant 1024 : i64
  %c65536_i64 = arith.constant 65536 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c8_i32, %c64_i32, %c1024_i32], [%c65536_i64, %c1024_i64, %c1_i64] : <bf16>, <1x32x256xbf16>
  %1 = tt.descriptor_load %0[%c3_i32, %c8_i32, %c0_i32] : !tt.tensordesc<1x32x256xbf16> -> tensor<1x32x256xbf16>
  %2 = tt.reshape %1 : tensor<1x32x256xbf16> -> tensor<32x256xbf16>
  %3 = tt.dot %arg0, %2, %cst, inputPrecision = tf32 : tensor<256x32xbf16> * tensor<32x256xbf16> -> tensor<256x256xf32>
  tt.return
}
// CHECK-LABEL: fuseOuterDim
// CHECK-NOT: tt.reshape
// CHECK: [[BASE:%.*]] = tt.addptr %arg1, %c196608_i64 : !tt.ptr<bf16>, i64
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor [[BASE]], [%c64_i32, %c1024_i32], [%c1024_i64, %c1_i64] : <bf16>, <32x256xbf16>
// CHECK: [[LOAD:%.*]] = tt.descriptor_load [[DESC]][%c8_i32, %c0_i32] : !tt.tensordesc<32x256xbf16> -> tensor<32x256xbf16>
// CHECK: tt.dot %arg0, [[LOAD]], {{.*}} : tensor<256x32xbf16> * tensor<32x256xbf16> -> tensor<256x256xf32>

// -----

// COM: Same, in a loop. The collapsed index is loop-invariant, so the new base
// COM: and descriptor stay hoisted where the original descriptor was.
tt.func public @fuseOuterDimInLoop(%arg0: tensor<32x256xbf16>, %arg1: !tt.ptr<bf16>) {
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
// CHECK-LABEL: fuseOuterDimInLoop
// CHECK-NOT: tt.reshape
// CHECK: [[BASE:%.*]] = tt.addptr %arg1, %c32768_i64 : !tt.ptr<bf16>, i64
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor [[BASE]], [%c1024_i32, %c32_i32], [%c1_i64, %c512_i64] : <bf16>, <256x32xbf16>
// CHECK: scf.for
// CHECK:   [[LOAD:%.*]] = tt.descriptor_load [[DESC]][%c32_i32, %c0_i32] : !tt.tensordesc<256x32xbf16> -> tensor<256x32xbf16>
// CHECK:   tt.dot [[LOAD]], {{.*}} : tensor<256x32xbf16> * tensor<32x256xbf16> -> tensor<256x256xf32>

// -----

// COM: Unit middle dimension: a one-head tile of a (tokens, heads, head_dim)
// COM: tensor. Head 3 starts 3 * 128 elements into the tensor, and the tile is
// COM: the (tokens, head_dim) slice there.
tt.func public @fuseMiddleDim(%arg0: tensor<128x256xbf16>, %arg1: !tt.ptr<bf16>) {
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
// CHECK-LABEL: fuseMiddleDim
// CHECK-NOT: tt.reshape
// CHECK: [[BASE:%.*]] = tt.addptr %arg1, %c384_i64 : !tt.ptr<bf16>, i64
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor [[BASE]], [%c1024_i32, %c128_i32], [%c4096_i64, %c1_i64] : <bf16>, <64x128xbf16>
// CHECK: [[LOAD:%.*]] = tt.descriptor_load [[DESC]][%c2_i32, %c0_i32] : !tt.tensordesc<64x128xbf16> -> tensor<64x128xbf16>
// CHECK: tt.dot [[LOAD]], {{.*}} : tensor<64x128xbf16> * tensor<128x256xbf16> -> tensor<64x256xf32>

// -----

// COM: Issue #8072: a runtime strides[1], as a host TensorDescriptor delivers
// COM: it (strides[2] is the constant 1 that RewriteTensorDescriptorToPointer
// COM: installs). The stride only scales the base offset, in i64, and nothing
// COM: divides by it. The product carries the descriptor's 16-byte stride
// COM: contract (8 f16 elements) as a hint, which AxisInfo cannot derive for an
// COM: unhinted stride and `MaterializeBlockPointer` needs to keep the base
// COM: 4-byte aligned.
tt.func public @fuseRuntimeMiddleStride(%arg0: tensor<128x64xf16>, %arg1: !tt.ptr<f16>, %T: i32, %H: i32, %sT: i64, %sH: i64, %m: i32, %h: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c128_i32 = arith.constant 128 : i32
  %c1_i64 = arith.constant 1 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%T, %H, %c128_i32], [%sT, %sH, %c1_i64] : <f16>, <64x1x128xf16>
  %1 = tt.descriptor_load %0[%m, %h, %c0_i32] : !tt.tensordesc<64x1x128xf16> -> tensor<64x1x128xf16>
  %2 = tt.reshape %1 : tensor<64x1x128xf16> -> tensor<64x128xf16>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<64x128xf16> * tensor<128x64xf16> -> tensor<64x64xf32>
  tt.return
}
// CHECK-LABEL: fuseRuntimeMiddleStride
// CHECK-SAME: %[[T:[a-z0-9]+]]: i32, %[[H:[a-z0-9]+]]: i32, %[[ST:[a-z0-9]+]]: i64, %[[SH:[a-z0-9]+]]: i64, %[[M:[a-z0-9]+]]: i32, %[[IDX:[a-z0-9]+]]: i32
// CHECK-NOT: arith.div
// CHECK: [[SHAPE:%.*]] = arith.maxsi %[[H]], %c1_i32 : i32
// CHECK: [[LAST:%.*]] = arith.subi [[SHAPE]], %c1_i32 : i32
// CHECK: [[MIN:%.*]] = arith.minsi %[[IDX]], [[LAST]] : i32
// CHECK: [[CLAMPED:%.*]] = arith.maxsi [[MIN]], %c0_i32 : i32
// CHECK: [[WIDE:%.*]] = arith.extsi [[CLAMPED]] : i32 to i64
// CHECK: [[OFF:%.*]] = arith.muli [[WIDE]], %[[SH]] {tt.divisibility = 8 : i32} : i64
// CHECK: [[BASE:%.*]] = tt.addptr %arg1, [[OFF]] : !tt.ptr<f16>, i64
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor [[BASE]], [%[[T]], %c128_i32], [%[[ST]], %c1_i64] : <f16>, <64x128xf16>
// CHECK: [[GE:%.*]] = arith.cmpi sge, %[[IDX]], %c0_i32 : i32
// CHECK: [[LT:%.*]] = arith.cmpi slt, %[[IDX]], %[[H]] : i32
// CHECK: [[IN:%.*]] = arith.andi [[GE]], [[LT]] : i1
// CHECK: [[INR:%.*]] = arith.extui [[IN]] : i1 to i32
// CHECK: [[OUT:%.*]] = arith.subi %c1_i32, [[INR]] : i32
// CHECK: [[PAD:%.*]] = arith.muli [[OUT]], %c128_i32 : i32
// CHECK: [[LOAD:%.*]] = tt.descriptor_load [[DESC]][%[[M]], [[PAD]]] : !tt.tensordesc<64x128xf16> -> tensor<64x128xf16>
// CHECK-NOT: arith.div
// CHECK: tt.dot [[LOAD]], {{.*}} : tensor<64x128xf16> * tensor<128x64xf16> -> tensor<64x64xf32>

// -----

// COM: The same with a folded negative stride: heads stored in reverse order.
// COM: The base moves back by 3 * 128 elements.
tt.func public @fuseNegativeMiddleStride(%arg0: tensor<128x256xbf16>, %arg1: !tt.ptr<bf16>) {
  %c0_i32 = arith.constant 0 : i32
  %c2_i32 = arith.constant 2 : i32
  %c3_i32 = arith.constant 3 : i32
  %c32_i32 = arith.constant 32 : i32
  %c128_i32 = arith.constant 128 : i32
  %c1024_i32 = arith.constant 1024 : i32
  %c1_i64 = arith.constant 1 : i64
  %cm128_i64 = arith.constant -128 : i64
  %c4096_i64 = arith.constant 4096 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<64x256xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c1024_i32, %c32_i32, %c128_i32], [%c4096_i64, %cm128_i64, %c1_i64] : <bf16>, <64x1x128xbf16>
  %1 = tt.descriptor_load %0[%c2_i32, %c3_i32, %c0_i32] : !tt.tensordesc<64x1x128xbf16> -> tensor<64x1x128xbf16>
  %2 = tt.reshape %1 : tensor<64x1x128xbf16> -> tensor<64x128xbf16>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<64x128xbf16> * tensor<128x256xbf16> -> tensor<64x256xf32>
  tt.return
}
// CHECK-LABEL: fuseNegativeMiddleStride
// CHECK-NOT: tt.reshape
// CHECK: [[BASE:%.*]] = tt.addptr %arg1, %c-384_i64 : !tt.ptr<bf16>, i64
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor [[BASE]], [%c1024_i32, %c128_i32], [%c4096_i64, %c1_i64] : <bf16>, <64x128xbf16>
// CHECK: [[LOAD:%.*]] = tt.descriptor_load [[DESC]][%c2_i32, %c0_i32] : !tt.tensordesc<64x128xbf16> -> tensor<64x128xbf16>
// CHECK: tt.dot [[LOAD]], {{.*}} : tensor<64x128xbf16> * tensor<128x256xbf16> -> tensor<64x256xf32>

// -----

// COM: Issue #8001, case 2: a negative outer stride. Slice 3 starts 3 * 1024
// COM: elements *before* the base.
tt.func public @fuseNegativeOuterStride(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c3_i32 = arith.constant 3 : i32
  %c1_i64 = arith.constant 1 : i64
  %c16_i64 = arith.constant 16 : i64
  %cm1024_i64 = arith.constant -1024 : i64
  %c8_i32 = arith.constant 8 : i32
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c8_i32, %c16_i32, %c16_i32], [%cm1024_i64, %c16_i64, %c1_i64] : <f32>, <1x16x16xf32>
  %1 = tt.descriptor_load %0[%c3_i32, %c0_i32, %c0_i32] : !tt.tensordesc<1x16x16xf32> -> tensor<1x16x16xf32>
  %2 = tt.reshape %1 : tensor<1x16x16xf32> -> tensor<16x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: fuseNegativeOuterStride
// CHECK-NOT: tt.reshape
// CHECK: [[BASE:%.*]] = tt.addptr %arg1, %c-3072_i64 : !tt.ptr<f32>, i64
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor [[BASE]], [%c16_i32, %c16_i32], [%c16_i64, %c1_i64] : <f32>, <16x16xf32>
// CHECK: [[LOAD:%.*]] = tt.descriptor_load [[DESC]][%c0_i32, %c0_i32] : !tt.tensordesc<16x16xf32> -> tensor<16x16xf32>
// CHECK: tt.dot [[LOAD]], {{.*}} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>

// -----

// COM: A (heads, tokens, head_dim) tensor read as (tokens, heads, head_dim)
// COM: has strides [128, 8192 * 128, 1]. Head 8 starts 8 * 8192 * 128
// COM: elements in, and the surface is the head itself, 128 elements wide.
tt.func public @fuseWideMiddleStride(%arg0: tensor<128x64xf16>, %arg1: !tt.ptr<f16>) {
  %c0_i32 = arith.constant 0 : i32
  %c8_i32 = arith.constant 8 : i32
  %c16_i32 = arith.constant 16 : i32
  %c128_i32 = arith.constant 128 : i32
  %c8192_i32 = arith.constant 8192 : i32
  %c1_i64 = arith.constant 1 : i64
  %c128_i64 = arith.constant 128 : i64
  %c1048576_i64 = arith.constant 1048576 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c8192_i32, %c16_i32, %c128_i32], [%c128_i64, %c1048576_i64, %c1_i64] : <f16>, <64x1x128xf16>
  %1 = tt.descriptor_load %0[%c0_i32, %c8_i32, %c0_i32] : !tt.tensordesc<64x1x128xf16> -> tensor<64x1x128xf16>
  %2 = tt.reshape %1 : tensor<64x1x128xf16> -> tensor<64x128xf16>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<64x128xf16> * tensor<128x64xf16> -> tensor<64x64xf32>
  tt.return
}
// CHECK-LABEL: fuseWideMiddleStride
// CHECK-NOT: tt.reshape
// CHECK: [[BASE:%.*]] = tt.addptr %arg1, %c8388608_i64 : !tt.ptr<f16>, i64
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor [[BASE]], [%c8192_i32, %c128_i32], [%c128_i64, %c1_i64] : <f16>, <64x128xf16>
// CHECK: [[LOAD:%.*]] = tt.descriptor_load [[DESC]][%c0_i32, %c0_i32] : !tt.tensordesc<64x128xf16> -> tensor<64x128xf16>
// CHECK: tt.dot [[LOAD]], {{.*}} : tensor<64x128xf16> * tensor<128x64xf16> -> tensor<64x64xf32>

// -----

// COM: A collapsed stride beyond i32 is used as is: the offset is formed in the
// COM: stride's i64, so nothing narrows.
tt.func public @fuseHugeCollapsedStride(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c1_i64 = arith.constant 1 : i64
  %c5000000000_i64 = arith.constant 5000000000 : i64
  %c4294967297_i64 = arith.constant 4294967297 : i64
  %c8_i32 = arith.constant 8 : i32
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c8_i32, %c16_i32, %c16_i32], [%c5000000000_i64, %c4294967297_i64, %c1_i64] : <f32>, <1x16x16xf32>
  %1 = tt.descriptor_load %0[%c1_i32, %c0_i32, %c0_i32] : !tt.tensordesc<1x16x16xf32> -> tensor<1x16x16xf32>
  %2 = tt.reshape %1 : tensor<1x16x16xf32> -> tensor<16x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: fuseHugeCollapsedStride
// CHECK-NOT: tt.reshape
// CHECK: [[BASE:%.*]] = tt.addptr %arg1, %c5000000000_i64{{[_0-9]*}} : !tt.ptr<f32>, i64
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor [[BASE]], [%c16_i32, %c16_i32], [%c4294967297_i64, %c1_i64] : <f32>, <16x16xf32>
// CHECK: [[LOAD:%.*]] = tt.descriptor_load [[DESC]][%c0_i32, %c0_i32] : !tt.tensordesc<16x16xf32> -> tensor<16x16xf32>
// CHECK: tt.dot [[LOAD]], {{.*}} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>

// -----

// COM: With collapsed index 1 the base offset folds to the stride itself, here
// COM: the caller's own `arith.muli`. The stride-contract hint goes on a copy
// COM: of that op, so the caller's op is left as it was.
tt.func public @fuseUnitIndexLeavesStrideOp(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>, %a: i64, %b: i64) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c8_i32 = arith.constant 8 : i32
  %c16_i32 = arith.constant 16 : i32
  %c1_i64 = arith.constant 1 : i64
  %c16_i64 = arith.constant 16 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %s0 = arith.muli %a, %b : i64
  %0 = tt.make_tensor_descriptor %arg1, [%c8_i32, %c16_i32, %c16_i32], [%s0, %c16_i64, %c1_i64] : <f32>, <1x16x16xf32>
  %1 = tt.descriptor_load %0[%c1_i32, %c0_i32, %c0_i32] : !tt.tensordesc<1x16x16xf32> -> tensor<1x16x16xf32>
  %2 = tt.reshape %1 : tensor<1x16x16xf32> -> tensor<16x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: fuseUnitIndexLeavesStrideOp
// CHECK-SAME: %[[A:[a-z0-9]+]]: i64, %[[B:[a-z0-9]+]]: i64)
// CHECK: arith.muli %[[A]], %[[B]] : i64
// CHECK: [[OFF:%.*]] = arith.muli %[[A]], %[[B]] {tt.divisibility = 4 : i32} : i64
// CHECK: [[BASE:%.*]] = tt.addptr %arg1, [[OFF]] : !tt.ptr<f32>, i64
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor [[BASE]], [%c16_i32, %c16_i32], [%c16_i64, %c1_i64] : <f32>, <16x16xf32>
// CHECK: [[LOAD:%.*]] = tt.descriptor_load [[DESC]][%c0_i32, %c0_i32] : !tt.tensordesc<16x16xf32> -> tensor<16x16xf32>
// CHECK: tt.dot [[LOAD]], {{.*}} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>

// -----

// COM: Padded strides (issues/7030): strides[0] need not be a multiple of
// COM: strides[1], since nothing divides one by the other. Every input is a
// COM: runtime value, so this also shows the whole runtime form: the clamped
// COM: base offset, the one-block floor for an empty merged dimension (%K
// COM: rows - 32, not the other dimension's 16), and the guard that keys on
// COM: both dimensions.
tt.func public @fusePaddedStrides(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>, %G: i32, %K: i32, %M: i32, %stride0: i64, %stride1: i64, %g: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<32x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%G, %K, %M], [%stride0, %stride1, %c1_i64] : <f32>, <1x32x16xf32>
  %1 = tt.descriptor_load %0[%g, %c0_i32, %c0_i32] : !tt.tensordesc<1x32x16xf32> -> tensor<1x32x16xf32>
  %2 = tt.reshape %1 : tensor<1x32x16xf32> -> tensor<32x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<32x16xf32> * tensor<16x16xf32> -> tensor<32x16xf32>
  tt.return
}
// CHECK-LABEL: fusePaddedStrides
// CHECK-SAME: %[[G:[a-z0-9]+]]: i32, %[[K:[a-z0-9]+]]: i32, %[[M:[a-z0-9]+]]: i32, %[[S0:[a-z0-9]+]]: i64, %[[S1:[a-z0-9]+]]: i64, %[[IDX:[a-z0-9]+]]: i32
// CHECK-NOT: arith.select
// CHECK: [[SHAPE:%.*]] = arith.maxsi %[[G]], %c1_i32 : i32
// CHECK: [[LAST:%.*]] = arith.subi [[SHAPE]], %c1_i32 : i32
// CHECK: [[MIN:%.*]] = arith.minsi %[[IDX]], [[LAST]] : i32
// CHECK: [[CLAMPED:%.*]] = arith.maxsi [[MIN]], %c0_i32 : i32
// CHECK: [[WIDE:%.*]] = arith.extsi [[CLAMPED]] : i32 to i64
// COM: The stride contract on the product: 16 bytes are 4 f32 elements.
// CHECK: [[OFF:%.*]] = arith.muli [[WIDE]], %[[S0]] {tt.divisibility = 4 : i32} : i64
// CHECK: [[BASE:%.*]] = tt.addptr %arg1, [[OFF]] : !tt.ptr<f32>, i64
// CHECK: [[NONEMPTY:%.*]] = arith.cmpi sgt, %[[K]], %c0_i32 : i32
// CHECK: [[NE:%.*]] = arith.extui [[NONEMPTY]] : i1 to i32
// CHECK: [[EMPTY:%.*]] = arith.subi %c1_i32, [[NE]] : i32
// CHECK: [[KEEP:%.*]] = arith.muli %[[K]], [[NE]] : i32
// CHECK: [[FLOOR:%.*]] = arith.muli [[EMPTY]], %c32_i32 : i32
// CHECK: [[ROWS:%.*]] = arith.addi [[KEEP]], [[FLOOR]] : i32
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor [[BASE]], [[[ROWS]], %[[M]]], [%[[S1]], %c1_i64] : <f32>, <32x16xf32>
// CHECK: [[GE:%.*]] = arith.cmpi sge, %[[IDX]], %c0_i32 : i32
// CHECK: [[LT:%.*]] = arith.cmpi slt, %[[IDX]], %[[G]] : i32
// CHECK: [[IN:%.*]] = arith.andi [[GE]], [[LT]] : i1
// CHECK: [[READS:%.*]] = arith.andi [[IN]], [[NONEMPTY]] : i1
// CHECK: [[INR:%.*]] = arith.extui [[READS]] : i1 to i32
// CHECK: [[OUT:%.*]] = arith.subi %c1_i32, [[INR]] : i32
// CHECK: [[PAD:%.*]] = arith.muli [[OUT]], [[ROWS]] : i32
// CHECK: [[LOAD:%.*]] = tt.descriptor_load [[DESC]][[[PAD]], %c0_i32] : !tt.tensordesc<32x16xf32> -> tensor<32x16xf32>
// CHECK-NOT: arith.select
// CHECK: tt.dot [[LOAD]], {{.*}} : tensor<32x16xf32> * tensor<16x16xf32> -> tensor<32x16xf32>

// -----

// COM: A block taller than its dimension (issues/7464): rows 13..63 of the
// COM: slice pad because the rebased descriptor keeps the dimension's own
// COM: extent, so a ragged block cannot spill into the next slice.
tt.func public @fuseNonDivisibleBlockExtent(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c2_i32 = arith.constant 2 : i32
  %c1_i64 = arith.constant 1 : i64
  %c8_i64 = arith.constant 8 : i64
  %c13_i32 = arith.constant 13 : i32
  %c5_i32 = arith.constant 5 : i32
  %c104_i64 = arith.constant 104 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<64x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c5_i32, %c13_i32, %c13_i32], [%c104_i64, %c8_i64, %c1_i64] : <f32>, <1x64x16xf32>
  %1 = tt.descriptor_load %0[%c2_i32, %c0_i32, %c0_i32] : !tt.tensordesc<1x64x16xf32> -> tensor<1x64x16xf32>
  %2 = tt.reshape %1 : tensor<1x64x16xf32> -> tensor<64x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<64x16xf32> * tensor<16x16xf32> -> tensor<64x16xf32>
  tt.return
}
// CHECK-LABEL: fuseNonDivisibleBlockExtent
// CHECK-NOT: tt.reshape
// CHECK: [[BASE:%.*]] = tt.addptr %arg1, %c208_i64 : !tt.ptr<f32>, i64
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor [[BASE]], [%c13_i32, %c13_i32], [%c8_i64, %c1_i64] : <f32>, <64x16xf32>
// CHECK: [[LOAD:%.*]] = tt.descriptor_load [[DESC]][%c0_i32, %c0_i32] : !tt.tensordesc<64x16xf32> -> tensor<64x16xf32>
// CHECK: tt.dot [[LOAD]], {{.*}} : tensor<64x16xf32> * tensor<16x16xf32> -> tensor<64x16xf32>

// -----

// COM: The middle-dimension counterpart: a 128-wide block over a 104-element
// COM: head pads columns 104..127 instead of reading the next head
// COM: (issues/7679).
tt.func public @fuseMiddleDimNonDivisibleBlockExtent(%arg0: tensor<128x256xbf16>, %arg1: !tt.ptr<bf16>) {
  %c0_i32 = arith.constant 0 : i32
  %c3_i32 = arith.constant 3 : i32
  %c32_i32 = arith.constant 32 : i32
  %c104_i32 = arith.constant 104 : i32
  %c1024_i32 = arith.constant 1024 : i32
  %c1_i64 = arith.constant 1 : i64
  %c104_i64 = arith.constant 104 : i64
  %c3328_i64 = arith.constant 3328 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<64x256xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c1024_i32, %c32_i32, %c104_i32], [%c3328_i64, %c104_i64, %c1_i64] : <bf16>, <64x1x128xbf16>
  %1 = tt.descriptor_load %0[%c0_i32, %c3_i32, %c0_i32] : !tt.tensordesc<64x1x128xbf16> -> tensor<64x1x128xbf16>
  %2 = tt.reshape %1 : tensor<64x1x128xbf16> -> tensor<64x128xbf16>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<64x128xbf16> * tensor<128x256xbf16> -> tensor<64x256xf32>
  tt.return
}
// CHECK-LABEL: fuseMiddleDimNonDivisibleBlockExtent
// CHECK-NOT: tt.reshape
// CHECK: [[BASE:%.*]] = tt.addptr %arg1, %c312_i64 : !tt.ptr<bf16>, i64
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor [[BASE]], [%c1024_i32, %c104_i32], [%c3328_i64, %c1_i64] : <bf16>, <64x128xbf16>
// CHECK: [[LOAD:%.*]] = tt.descriptor_load [[DESC]][%c0_i32, %c0_i32] : !tt.tensordesc<64x128xbf16> -> tensor<64x128xbf16>
// CHECK: tt.dot [[LOAD]], {{.*}} : tensor<64x128xbf16> * tensor<128x256xbf16> -> tensor<64x256xf32>

// -----

// COM: Collapsing the middle dimension makes shapes[0] and strides[0] the
// COM: surface height and pitch. They are passed through as given - here a
// COM: negative pitch - and judged by `LowerTo2DBlockLoad` like any rank-2
// COM: descriptor's, which keeps a folded illegal field off 2D block I/O.
tt.func public @fuseMiddleDimKeepsLeadingFields(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c2_i32 = arith.constant 2 : i32
  %c1_i64 = arith.constant 1 : i64
  %cm64_i64 = arith.constant -64 : i64
  %c16_i64 = arith.constant 16 : i64
  %c4_i32 = arith.constant 4 : i32
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c16_i32, %c4_i32, %c16_i32], [%cm64_i64, %c16_i64, %c1_i64] : <f32>, <16x1x16xf32>
  %1 = tt.descriptor_load %0[%c0_i32, %c2_i32, %c0_i32] : !tt.tensordesc<16x1x16xf32> -> tensor<16x1x16xf32>
  %2 = tt.reshape %1 : tensor<16x1x16xf32> -> tensor<16x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: fuseMiddleDimKeepsLeadingFields
// CHECK-NOT: tt.reshape
// CHECK: [[BASE:%.*]] = tt.addptr %arg1, %c32_i64 : !tt.ptr<f32>, i64
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor [[BASE]], [%c16_i32, %c16_i32], [%c-64_i64, %c1_i64] : <f32>, <16x16xf32>
// CHECK: [[LOAD:%.*]] = tt.descriptor_load [[DESC]][%c0_i32, %c0_i32] : !tt.tensordesc<16x16xf32> -> tensor<16x16xf32>
// CHECK: tt.dot [[LOAD]], {{.*}} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>

// -----

// COM: The promoted pitch must not be narrower than the surface width, a 2D
// COM: block rule that nothing after this pass checks. Here strides[0] is 16
// COM: f32 elements (64 bytes) and the width 32 (128 bytes), so the load keeps
// COM: its rank-3 form. The second load collapses the outer dimension of the
// COM: same strides: its pitch is strides[1], as for the unfused load, so it
// COM: still fuses.
tt.func public @noFuseMiddleDimPitchBelowWidth(%arg0: tensor<32x16xf32>, %arg1: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c2_i32 = arith.constant 2 : i32
  %c4_i32 = arith.constant 4 : i32
  %c16_i32 = arith.constant 16 : i32
  %c32_i32 = arith.constant 32 : i32
  %c1_i64 = arith.constant 1 : i64
  %c16_i64 = arith.constant 16 : i64
  %c32_i64 = arith.constant 32 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c16_i32, %c4_i32, %c32_i32], [%c16_i64, %c32_i64, %c1_i64] : <f32>, <16x1x32xf32>
  %1 = tt.descriptor_load %0[%c0_i32, %c2_i32, %c0_i32] : !tt.tensordesc<16x1x32xf32> -> tensor<16x1x32xf32>
  %2 = tt.reshape %1 : tensor<16x1x32xf32> -> tensor<16x32xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x32xf32> * tensor<32x16xf32> -> tensor<16x16xf32>
  %4 = tt.make_tensor_descriptor %arg1, [%c4_i32, %c16_i32, %c32_i32], [%c16_i64, %c32_i64, %c1_i64] : <f32>, <1x16x32xf32>
  %5 = tt.descriptor_load %4[%c2_i32, %c0_i32, %c0_i32] : !tt.tensordesc<1x16x32xf32> -> tensor<1x16x32xf32>
  %6 = tt.reshape %5 : tensor<1x16x32xf32> -> tensor<16x32xf32>
  %7 = tt.dot %6, %arg0, %cst, inputPrecision = tf32 : tensor<16x32xf32> * tensor<32x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: noFuseMiddleDimPitchBelowWidth
// CHECK-NOT: tt.addptr
// CHECK: [[LOAD:%.*]] = tt.descriptor_load {{.*}} : !tt.tensordesc<16x1x32xf32> -> tensor<16x1x32xf32>
// CHECK: [[RESHAPE:%.*]] = tt.reshape [[LOAD]] : tensor<16x1x32xf32> -> tensor<16x32xf32>
// CHECK: tt.dot [[RESHAPE]], {{.*}} : tensor<16x32xf32> * tensor<32x16xf32> -> tensor<16x16xf32>
// CHECK: [[BASE:%.*]] = tt.addptr %arg1, %c32_i64{{[_0-9]*}} : !tt.ptr<f32>, i64
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor [[BASE]], [%c16_i32, %c32_i32], [%c32_i64, %c1_i64] : <f32>, <16x32xf32>
// CHECK: [[OUTER:%.*]] = tt.descriptor_load [[DESC]][%c0_i32, %c0_i32] : !tt.tensordesc<16x32xf32> -> tensor<16x32xf32>
// CHECK: tt.dot [[OUTER]], {{.*}} : tensor<16x32xf32> * tensor<32x16xf32> -> tensor<16x16xf32>

// -----

// COM: The same with a runtime width: the one-block floor it gets for the
// COM: empty case, 32 elements (128 bytes), is wider than the 64-byte pitch.
tt.func public @noFuseMiddleDimPitchBelowFloor(%arg0: tensor<32x16xf32>, %arg1: !tt.ptr<f32>, %D: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c2_i32 = arith.constant 2 : i32
  %c4_i32 = arith.constant 4 : i32
  %c16_i32 = arith.constant 16 : i32
  %c1_i64 = arith.constant 1 : i64
  %c16_i64 = arith.constant 16 : i64
  %c32_i64 = arith.constant 32 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c16_i32, %c4_i32, %D], [%c16_i64, %c32_i64, %c1_i64] : <f32>, <16x1x32xf32>
  %1 = tt.descriptor_load %0[%c0_i32, %c2_i32, %c0_i32] : !tt.tensordesc<16x1x32xf32> -> tensor<16x1x32xf32>
  %2 = tt.reshape %1 : tensor<16x1x32xf32> -> tensor<16x32xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x32xf32> * tensor<32x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: noFuseMiddleDimPitchBelowFloor
// CHECK-NOT: tt.addptr
// CHECK: [[LOAD:%.*]] = tt.descriptor_load {{.*}} : !tt.tensordesc<16x1x32xf32> -> tensor<16x1x32xf32>
// CHECK: [[RESHAPE:%.*]] = tt.reshape [[LOAD]] : tensor<16x1x32xf32> -> tensor<16x32xf32>
// CHECK: tt.dot [[RESHAPE]], {{.*}} : tensor<16x32xf32> * tensor<32x16xf32> -> tensor<16x16xf32>

// -----

// COM: An empty collapsed dimension makes the rank-3 load pure padding. The
// COM: clamp keeps the base where it is (the clamp range is [0, 0]) and the
// COM: guard folds to the padding index: the rows' own extent, 16.
tt.func public @fuseEmptyCollapsedDim(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c16_i64 = arith.constant 16 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c0_i32, %c16_i32, %c16_i32], [%c1024_i64, %c16_i64, %c1_i64] : <f32>, <1x16x16xf32>
  %1 = tt.descriptor_load %0[%c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<1x16x16xf32> -> tensor<1x16x16xf32>
  %2 = tt.reshape %1 : tensor<1x16x16xf32> -> tensor<16x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: fuseEmptyCollapsedDim
// CHECK-NOT: tt.addptr
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor %arg1, [%c16_i32, %c16_i32], [%c16_i64, %c1_i64] : <f32>, <16x16xf32>
// CHECK: [[LOAD:%.*]] = tt.descriptor_load [[DESC]][%c16_i32{{[_0-9]*}}, %c0_i32] : !tt.tensordesc<16x16xf32> -> tensor<16x16xf32>
// CHECK: tt.dot [[LOAD]], {{.*}} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>

// -----

// COM: INT32_MIN is the regression test for clamping the shape *before*
// COM: subtracting one: maxsi(s-1, 0) would compute INT32_MIN-1, wrap to
// COM: INT32_MAX and let index 5 move the base 5 slices out. The clamp pins
// COM: it to the base and the guard pads.
tt.func public @fuseIntMinCollapsedShape(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c5_i32 = arith.constant 5 : i32
  %c1_i64 = arith.constant 1 : i64
  %c16_i64 = arith.constant 16 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %cmin_i32 = arith.constant -2147483648 : i32
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%cmin_i32, %c16_i32, %c16_i32], [%c1024_i64, %c16_i64, %c1_i64] : <f32>, <1x16x16xf32>
  %1 = tt.descriptor_load %0[%c5_i32, %c0_i32, %c0_i32] : !tt.tensordesc<1x16x16xf32> -> tensor<1x16x16xf32>
  %2 = tt.reshape %1 : tensor<1x16x16xf32> -> tensor<16x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: fuseIntMinCollapsedShape
// CHECK-NOT: tt.addptr
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor %arg1, [%c16_i32, %c16_i32], [%c16_i64, %c1_i64] : <f32>, <16x16xf32>
// CHECK: [[LOAD:%.*]] = tt.descriptor_load [[DESC]][%c16_i32{{[_0-9]*}}, %c0_i32] : !tt.tensordesc<16x16xf32> -> tensor<16x16xf32>
// CHECK: tt.dot [[LOAD]], {{.*}} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>

// -----

// COM: A folded empty merged dimension keeps its 0 extent: `LowerTo2DBlockLoad`
// COM: sees it and keeps the load off 2D block I/O, so no floor is needed.
tt.func public @fuseZeroMergedExtent(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c16_i64 = arith.constant 16 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c8_i32 = arith.constant 8 : i32
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c8_i32, %c0_i32, %c16_i32], [%c1024_i64, %c16_i64, %c1_i64] : <f32>, <1x16x16xf32>
  %1 = tt.descriptor_load %0[%c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<1x16x16xf32> -> tensor<1x16x16xf32>
  %2 = tt.reshape %1 : tensor<1x16x16xf32> -> tensor<16x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: fuseZeroMergedExtent
// CHECK-NOT: arith.cmpi
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor %arg1, [%c0_i32, %c16_i32], [%c16_i64, %c1_i64] : <f32>, <16x16xf32>
// CHECK-NOT: arith.cmpi
// CHECK: [[LOAD:%.*]] = tt.descriptor_load [[DESC]][%c0_i32, %c0_i32] : !tt.tensordesc<16x16xf32> -> tensor<16x16xf32>
// CHECK: tt.dot [[LOAD]], {{.*}} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>

// -----

// COM: A dynamic collapsed shape: the clamp's upper end and the guard's `slt`
// COM: both use it. The load index is a constant 0, so the guard's in-range
// COM: term folds away and only the padding term is left.
tt.func public @fuseDynamicCollapsedShape(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>, %shape0: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c16_i64 = arith.constant 16 : i64
  %c256_i64 = arith.constant 256 : i64
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%shape0, %c16_i32, %c16_i32], [%c256_i64, %c16_i64, %c1_i64] : <f32>, <1x16x16xf32>
  %1 = tt.descriptor_load %0[%c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<1x16x16xf32> -> tensor<1x16x16xf32>
  %2 = tt.reshape %1 : tensor<1x16x16xf32> -> tensor<16x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: fuseDynamicCollapsedShape
// CHECK-SAME: %[[S:[a-z0-9]+]]: i32)
// CHECK: [[SHAPE:%.*]] = arith.maxsi %[[S]], %c1_i32 : i32
// CHECK: [[LAST:%.*]] = arith.subi [[SHAPE]], %c1_i32 : i32
// CHECK: [[MIN:%.*]] = arith.minsi [[LAST]], %c0_i32 : i32
// CHECK: [[CLAMPED:%.*]] = arith.maxsi [[MIN]], %c0_i32 : i32
// CHECK: [[WIDE:%.*]] = arith.extsi [[CLAMPED]] : i32 to i64
// CHECK: [[OFF:%.*]] = arith.muli [[WIDE]], %c256_i64 : i64
// CHECK: [[BASE:%.*]] = tt.addptr %arg1, [[OFF]] : !tt.ptr<f32>, i64
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor [[BASE]], [%c16_i32, %c16_i32], [%c16_i64, %c1_i64] : <f32>, <16x16xf32>
// COM: `0 < shape` folds to `shape > 0`.
// CHECK: [[IN:%.*]] = arith.cmpi sgt, %[[S]], %c0_i32 : i32
// CHECK: [[INR:%.*]] = arith.extui [[IN]] : i1 to i32
// CHECK: [[OUT:%.*]] = arith.subi %c1_i32, [[INR]] : i32
// CHECK: [[PAD:%.*]] = arith.muli [[OUT]], %c16_i32 : i32
// CHECK: [[LOAD:%.*]] = tt.descriptor_load [[DESC]][[[PAD]], %c0_i32] : !tt.tensordesc<16x16xf32> -> tensor<16x16xf32>
// CHECK: tt.dot [[LOAD]], {{.*}} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>

// -----

// COM: A merged dimension that is empty only at runtime (`%n == 0`). A zero
// COM: extent reaches the 2D block message as `0 - 1`, a 16MB surface that
// COM: reads real data, so the extent becomes one 32-row block instead and the
// COM: guard keys on `shapes[1] > 0` too: without it an in-range `%off` would
// COM: read row 0 of that block. Multiplies, never an `arith.select`:
// COM: `ttgi::isDivisible` understands `muli` (either operand) and `addi`
// COM: (both) but not `select`, so a select would change what the
// COM: descriptor-load fallback decides about the emitted shape and index.
tt.func public @fuseDynamicMergedShape(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>, %n: i32, %off: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c16_i64 = arith.constant 16 : i64
  %c256_i64 = arith.constant 256 : i64
  %c8_i32 = arith.constant 8 : i32
  %c16_i32 = arith.constant 16 : i32
  %c32_i32 = arith.constant 32 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<32x16xf32>
  %rows = arith.muli %n, %c32_i32 : i32
  %0 = tt.make_tensor_descriptor %arg1, [%c8_i32, %rows, %c16_i32], [%c256_i64, %c16_i64, %c1_i64] : <f32>, <1x32x16xf32>
  %1 = tt.descriptor_load %0[%off, %c0_i32, %c0_i32] : !tt.tensordesc<1x32x16xf32> -> tensor<1x32x16xf32>
  %2 = tt.reshape %1 : tensor<1x32x16xf32> -> tensor<32x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<32x16xf32> * tensor<16x16xf32> -> tensor<32x16xf32>
  tt.return
}
// CHECK-LABEL: fuseDynamicMergedShape
// CHECK-SAME: %[[N:[a-z0-9]+]]: i32, %[[IDX:[a-z0-9]+]]: i32)
// CHECK-NOT: arith.select
// CHECK: [[R:%.*]] = arith.muli %[[N]], %c32_i32 : i32
// CHECK: [[BASE:%.*]] = tt.addptr %arg1, {{.*}} : !tt.ptr<f32>, i64
// CHECK: [[NONEMPTY:%.*]] = arith.cmpi sgt, [[R]], %c0_i32 : i32
// CHECK: [[NE:%.*]] = arith.extui [[NONEMPTY]] : i1 to i32
// CHECK: [[EMPTY:%.*]] = arith.subi %c1_i32, [[NE]] : i32
// CHECK: [[KEEP:%.*]] = arith.muli [[R]], [[NE]] : i32
// CHECK: [[FLOOR:%.*]] = arith.muli [[EMPTY]], %c32_i32 : i32
// CHECK: [[ROWS:%.*]] = arith.addi [[KEEP]], [[FLOOR]] : i32
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor [[BASE]], [[[ROWS]], %c16_i32], [%c16_i64, %c1_i64] : <f32>, <32x16xf32>
// CHECK: [[GE:%.*]] = arith.cmpi sge, %[[IDX]], %c0_i32 : i32
// CHECK: [[LT:%.*]] = arith.cmpi slt, %[[IDX]], %c8_i32 : i32
// CHECK: [[IN:%.*]] = arith.andi [[GE]], [[LT]] : i1
// CHECK: [[READS:%.*]] = arith.andi [[IN]], [[NONEMPTY]] : i1
// CHECK: [[INR:%.*]] = arith.extui [[READS]] : i1 to i32
// CHECK: [[OUT:%.*]] = arith.subi %c1_i32, [[INR]] : i32
// CHECK: [[PAD:%.*]] = arith.muli [[OUT]], [[ROWS]] : i32
// CHECK: [[LOAD:%.*]] = tt.descriptor_load [[DESC]][[[PAD]], %c0_i32] : !tt.tensordesc<32x16xf32> -> tensor<32x16xf32>
// CHECK-NOT: arith.select
// CHECK: tt.dot [[LOAD]], {{.*}} : tensor<32x16xf32> * tensor<16x16xf32> -> tensor<32x16xf32>

// -----

// COM: The middle-dimension counterpart, where the floor is a width: one
// COM: 16-element block of columns, not the other dimension's 32 rows. The
// COM: collapsed index 1 is in range, so the
// COM: guard reduces to the merged dimension's own term.
tt.func public @fuseDynamicMergedShapeMiddleDim(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>, %shape2: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c1_i64 = arith.constant 1 : i64
  %c64_i64 = arith.constant 64 : i64
  %c256_i64 = arith.constant 256 : i64
  %c4_i32 = arith.constant 4 : i32
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<32x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c16_i32, %c4_i32, %shape2], [%c256_i64, %c64_i64, %c1_i64] : <f32>, <32x1x16xf32>
  %1 = tt.descriptor_load %0[%c0_i32, %c1_i32, %c0_i32] : !tt.tensordesc<32x1x16xf32> -> tensor<32x1x16xf32>
  %2 = tt.reshape %1 : tensor<32x1x16xf32> -> tensor<32x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<32x16xf32> * tensor<16x16xf32> -> tensor<32x16xf32>
  tt.return
}
// CHECK-LABEL: fuseDynamicMergedShapeMiddleDim
// CHECK-SAME: %[[S:[a-z0-9]+]]: i32)
// CHECK-NOT: arith.select
// CHECK: [[BASE:%.*]] = tt.addptr %arg1, %c64_i64{{[_0-9]*}} : !tt.ptr<f32>, i64
// CHECK: [[NONEMPTY:%.*]] = arith.cmpi sgt, %[[S]], %c0_i32 : i32
// CHECK: [[NE:%.*]] = arith.extui [[NONEMPTY]] : i1 to i32
// CHECK: [[EMPTY:%.*]] = arith.subi %c1_i32, [[NE]] : i32
// CHECK: [[KEEP:%.*]] = arith.muli %[[S]], [[NE]] : i32
// CHECK: [[FLOOR:%.*]] = arith.muli [[EMPTY]], %c16_i32 : i32
// CHECK: [[COLS:%.*]] = arith.addi [[KEEP]], [[FLOOR]] : i32
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor [[BASE]], [%c16_i32, [[COLS]]], [%c256_i64, %c1_i64] : <f32>, <32x16xf32>
// CHECK: [[INR:%.*]] = arith.extui [[NONEMPTY]] : i1 to i32
// CHECK: [[OUT:%.*]] = arith.subi %c1_i32, [[INR]] : i32
// CHECK: [[PAD:%.*]] = arith.muli [[OUT]], [[COLS]] : i32
// CHECK: [[LOAD:%.*]] = tt.descriptor_load [[DESC]][%c0_i32, [[PAD]]] : !tt.tensordesc<32x16xf32> -> tensor<32x16xf32>
// CHECK-NOT: arith.select
// CHECK: tt.dot [[LOAD]], {{.*}} : tensor<32x16xf32> * tensor<16x16xf32> -> tensor<32x16xf32>

// -----

// COM: A dynamic collapsed index cannot be proven in range, so the index of
// COM: the dimension after it is guarded: `idx * inRange + extent * (1 -
// COM: inRange)`, which forces it to that dimension's extent - and so makes
// COM: the load pad, as the rank-3 form does - whenever the collapsed index is
// COM: out of range (issue #8070). The base offset uses the clamped index, so
// COM: an out-of-range one never moves the base outside the tensor.
tt.func public @fuseGuardsDynamicOuterIndex(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>, %off: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c16_i64 = arith.constant 16 : i64
  %c256_i64 = arith.constant 256 : i64
  %c8_i32 = arith.constant 8 : i32
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c8_i32, %c16_i32, %c16_i32], [%c256_i64, %c16_i64, %c1_i64] : <f32>, <1x16x16xf32>
  %1 = tt.descriptor_load %0[%off, %c0_i32, %c0_i32] : !tt.tensordesc<1x16x16xf32> -> tensor<1x16x16xf32>
  %2 = tt.reshape %1 : tensor<1x16x16xf32> -> tensor<16x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: fuseGuardsDynamicOuterIndex
// CHECK-SAME: %[[IDX:[a-z0-9]+]]: i32)
// COM: The stride folds, so AxisInfo sees the product exactly: no hint.
// CHECK-NOT: arith.select
// CHECK: [[MIN:%.*]] = arith.minsi %[[IDX]], %c7_i32 : i32
// CHECK: [[CLAMPED:%.*]] = arith.maxsi [[MIN]], %c0_i32 : i32
// CHECK: [[WIDE:%.*]] = arith.extsi [[CLAMPED]] : i32 to i64
// CHECK: [[OFF:%.*]] = arith.muli [[WIDE]], %c256_i64 : i64
// CHECK: [[BASE:%.*]] = tt.addptr %arg1, [[OFF]] : !tt.ptr<f32>, i64
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor [[BASE]], [%c16_i32, %c16_i32], [%c16_i64, %c1_i64] : <f32>, <16x16xf32>
// CHECK: [[GE:%.*]] = arith.cmpi sge, %[[IDX]], %c0_i32 : i32
// CHECK: [[LT:%.*]] = arith.cmpi slt, %[[IDX]], %c8_i32 : i32
// CHECK: [[IN:%.*]] = arith.andi [[GE]], [[LT]] : i1
// CHECK: [[INR:%.*]] = arith.extui [[IN]] : i1 to i32
// CHECK: [[OUT:%.*]] = arith.subi %c1_i32, [[INR]] : i32
// CHECK: [[PAD:%.*]] = arith.muli [[OUT]], %c16_i32 : i32
// CHECK: [[LOAD:%.*]] = tt.descriptor_load [[DESC]][[[PAD]], %c0_i32] : !tt.tensordesc<16x16xf32> -> tensor<16x16xf32>
// CHECK-NOT: arith.select
// CHECK: tt.dot [[LOAD]], {{.*}} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>

// -----

// COM: The same guard on a middle collapse, where the guarded index is the
// COM: column, the index `satisfies2DBlockReadAlignment` queries.
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
// CHECK-SAME: %[[IDX:[a-z0-9]+]]: i32)
// CHECK-NOT: arith.select
// CHECK: [[MIN:%.*]] = arith.minsi %[[IDX]], %c15_i32 : i32
// CHECK: [[CLAMPED:%.*]] = arith.maxsi [[MIN]], %c0_i32 : i32
// CHECK: [[WIDE:%.*]] = arith.extsi [[CLAMPED]] : i32 to i64
// CHECK: [[OFF:%.*]] = arith.muli [[WIDE]], %c16_i64 : i64
// CHECK: [[BASE:%.*]] = tt.addptr %arg1, [[OFF]] : !tt.ptr<f32>, i64
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor [[BASE]], [%c64_i32, %c16_i32], [%c1024_i64, %c1_i64] : <f32>, <64x16xf32>
// CHECK: [[GE:%.*]] = arith.cmpi sge, %[[IDX]], %c0_i32 : i32
// CHECK: [[LT:%.*]] = arith.cmpi slt, %[[IDX]], %c16_i32 : i32
// CHECK: [[IN:%.*]] = arith.andi [[GE]], [[LT]] : i1
// CHECK: [[INR:%.*]] = arith.extui [[IN]] : i1 to i32
// CHECK: [[OUT:%.*]] = arith.subi %c1_i32, [[INR]] : i32
// CHECK: [[PAD:%.*]] = arith.muli [[OUT]], %c16_i32 : i32
// CHECK: [[LOAD:%.*]] = tt.descriptor_load [[DESC]][%c0_i32, [[PAD]]] : !tt.tensordesc<64x16xf32> -> tensor<64x16xf32>
// CHECK-NOT: arith.select
// CHECK: tt.dot [[LOAD]], {{.*}} : tensor<64x16xf32> * tensor<16x256xf32> -> tensor<64x256xf32>

// -----

// COM: A negative collapsed index is legal input: the rank-3 load pads
// COM: entirely. The clamp keeps the base at slice 0 and the guard (its `sge`
// COM: half) forces the row index to 16, so the fused load pads too.
tt.func public @fuseGuardsNegativeCollapsedIndex(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %cm3_i32 = arith.constant -3 : i32
  %c1_i64 = arith.constant 1 : i64
  %c16_i64 = arith.constant 16 : i64
  %c256_i64 = arith.constant 256 : i64
  %c8_i32 = arith.constant 8 : i32
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c8_i32, %c16_i32, %c16_i32], [%c256_i64, %c16_i64, %c1_i64] : <f32>, <1x16x16xf32>
  %1 = tt.descriptor_load %0[%cm3_i32, %c0_i32, %c0_i32] : !tt.tensordesc<1x16x16xf32> -> tensor<1x16x16xf32>
  %2 = tt.reshape %1 : tensor<1x16x16xf32> -> tensor<16x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: fuseGuardsNegativeCollapsedIndex
// CHECK-NOT: tt.addptr
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor %arg1, [%c16_i32, %c16_i32], [%c16_i64, %c1_i64] : <f32>, <16x16xf32>
// CHECK: [[LOAD:%.*]] = tt.descriptor_load [[DESC]][%c16_i32{{[_0-9]*}}, %c0_i32] : !tt.tensordesc<16x16xf32> -> tensor<16x16xf32>
// CHECK: tt.dot [[LOAD]], {{.*}} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>

// -----

// COM: The other half: a collapsed index at its extent (4 of 4). The clamp
// COM: keeps the base at the last slice, 3 * 512, and the `slt` half of the
// COM: guard forces the row index to 32 so the load pads instead of reading
// COM: that slice.
tt.func public @fuseGuardsCollapsedIndexAtExtent(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c4_i32 = arith.constant 4 : i32
  %c16_i32 = arith.constant 16 : i32
  %c32_i32 = arith.constant 32 : i32
  %c1_i64 = arith.constant 1 : i64
  %c16_i64 = arith.constant 16 : i64
  %c512_i64 = arith.constant 512 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<32x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c4_i32, %c32_i32, %c16_i32], [%c512_i64, %c16_i64, %c1_i64] : <f32>, <1x32x16xf32>
  %1 = tt.descriptor_load %0[%c4_i32, %c0_i32, %c0_i32] : !tt.tensordesc<1x32x16xf32> -> tensor<1x32x16xf32>
  %2 = tt.reshape %1 : tensor<1x32x16xf32> -> tensor<32x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<32x16xf32> * tensor<16x16xf32> -> tensor<32x16xf32>
  tt.return
}
// CHECK-LABEL: fuseGuardsCollapsedIndexAtExtent
// CHECK: [[BASE:%.*]] = tt.addptr %arg1, %c1536_i64 : !tt.ptr<f32>, i64
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor [[BASE]], [%c32_i32, %c16_i32], [%c16_i64, %c1_i64] : <f32>, <32x16xf32>
// CHECK: [[LOAD:%.*]] = tt.descriptor_load [[DESC]][%c32_i32{{[_0-9]*}}, %c0_i32] : !tt.tensordesc<32x16xf32> -> tensor<32x16xf32>
// CHECK: tt.dot [[LOAD]], {{.*}} : tensor<32x16xf32> * tensor<16x16xf32> -> tensor<32x16xf32>

// -----

// COM: A provably in-range index pays nothing: the guard folds away and the
// COM: load keeps its own row index.
tt.func public @fuseInRangeIndexFoldsGuard(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c3_i32 = arith.constant 3 : i32
  %c8_i32 = arith.constant 8 : i32
  %c16_i32 = arith.constant 16 : i32
  %c1_i64 = arith.constant 1 : i64
  %c16_i64 = arith.constant 16 : i64
  %c256_i64 = arith.constant 256 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c8_i32, %c16_i32, %c16_i32], [%c256_i64, %c16_i64, %c1_i64] : <f32>, <1x16x16xf32>
  %1 = tt.descriptor_load %0[%c3_i32, %c0_i32, %c0_i32] : !tt.tensordesc<1x16x16xf32> -> tensor<1x16x16xf32>
  %2 = tt.reshape %1 : tensor<1x16x16xf32> -> tensor<16x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: fuseInRangeIndexFoldsGuard
// CHECK: [[BASE:%.*]] = tt.addptr %arg1, %c768_i64 : !tt.ptr<f32>, i64
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor [[BASE]], [%c16_i32, %c16_i32], [%c16_i64, %c1_i64] : <f32>, <16x16xf32>
// CHECK-NOT: arith.cmpi
// CHECK: [[LOAD:%.*]] = tt.descriptor_load [[DESC]][%c0_i32, %c0_i32] : !tt.tensordesc<16x16xf32> -> tensor<16x16xf32>
// CHECK: tt.dot [[LOAD]], {{.*}} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>

// -----

// COM: A negative index on the merged dimension pads only the block's leading
// COM: rows (#8070). The fused load keeps the index, and the rank-2 bounds
// COM: check pads those rows.
tt.func public @fuseNegativeMergedIndex(%arg0: tensor<64x64xf32>, %arg1: !tt.ptr<f32>) {
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
// CHECK-LABEL: fuseNegativeMergedIndex
// CHECK-NOT: tt.reshape
// CHECK: [[BASE:%.*]] = tt.addptr %arg1, %c4096_i64{{[_0-9]*}} : !tt.ptr<f32>, i64
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor [[BASE]], [%c64_i32, %c64_i32], [%c64_i64, %c1_i64] : <f32>, <64x64xf32>
// CHECK: [[LOAD:%.*]] = tt.descriptor_load [[DESC]][%c-1_i32, %c0_i32] : !tt.tensordesc<64x64xf32> -> tensor<64x64xf32>
// CHECK: tt.dot [[LOAD]], {{.*}} : tensor<64x64xf32> * tensor<64x64xf32> -> tensor<64x64xf32>

// -----

// COM: The base depends on the collapsed index, so a loop-carried index forces
// COM: the new descriptor *into* the loop.
tt.func public @fuseSinksDescriptorIntoLoop(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>) -> tensor<16x16xf32> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c8_i32 = arith.constant 8 : i32
  %c1_i64 = arith.constant 1 : i64
  %c16_i64 = arith.constant 16 : i64
  %c256_i64 = arith.constant 256 : i64
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c8_i32, %c16_i32, %c16_i32], [%c256_i64, %c16_i64, %c1_i64] : <f32>, <1x16x16xf32>
  %1 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %cst) -> (tensor<16x16xf32>) : i32 {
    %2 = tt.descriptor_load %0[%iv, %c0_i32, %c0_i32] : !tt.tensordesc<1x16x16xf32> -> tensor<1x16x16xf32>
    %3 = tt.reshape %2 : tensor<1x16x16xf32> -> tensor<16x16xf32>
    %4 = tt.dot %3, %arg0, %acc, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
    scf.yield %4 : tensor<16x16xf32>
  }
  tt.return %1 : tensor<16x16xf32>
}
// CHECK-LABEL: fuseSinksDescriptorIntoLoop
// CHECK-NOT: tt.make_tensor_descriptor
// CHECK: scf.for [[IV:%[a-z0-9]+]] =
// CHECK:   [[MIN:%.*]] = arith.minsi [[IV]], %c7_i32 : i32
// CHECK:   [[BASE:%.*]] = tt.addptr %arg1, {{.*}} : !tt.ptr<f32>, i64
// CHECK:   [[DESC:%.*]] = tt.make_tensor_descriptor [[BASE]], [%c16_i32, %c16_i32], [%c16_i64, %c1_i64] : <f32>, <16x16xf32>
// CHECK:   [[LOAD:%.*]] = tt.descriptor_load [[DESC]]{{.*}} : !tt.tensordesc<16x16xf32> -> tensor<16x16xf32>
// CHECK:   tt.dot [[LOAD]]

// -----

// COM: The other `DominanceInfo` arm: an index that already dominates the
// COM: descriptor leaves it hoisted, even though the load is in a loop. Only
// COM: the guard is emitted at the load.
tt.func public @fuseKeepsDescriptorOutsideLoop(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>) -> tensor<16x16xf32> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c8_i32 = arith.constant 8 : i32
  %c1_i64 = arith.constant 1 : i64
  %c16_i64 = arith.constant 16 : i64
  %c256_i64 = arith.constant 256 : i64
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %pid = tt.get_program_id x : i32
  %0 = tt.make_tensor_descriptor %arg1, [%c8_i32, %c16_i32, %c16_i32], [%c256_i64, %c16_i64, %c1_i64] : <f32>, <1x16x16xf32>
  %1 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %cst) -> (tensor<16x16xf32>) : i32 {
    %2 = tt.descriptor_load %0[%pid, %c0_i32, %c0_i32] : !tt.tensordesc<1x16x16xf32> -> tensor<1x16x16xf32>
    %3 = tt.reshape %2 : tensor<1x16x16xf32> -> tensor<16x16xf32>
    %4 = tt.dot %3, %arg0, %acc, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
    scf.yield %4 : tensor<16x16xf32>
  }
  tt.return %1 : tensor<16x16xf32>
}
// CHECK-LABEL: fuseKeepsDescriptorOutsideLoop
// CHECK: [[PID:%.*]] = tt.get_program_id x : i32
// CHECK: [[MIN:%.*]] = arith.minsi [[PID]], %c7_i32 : i32
// CHECK: [[BASE:%.*]] = tt.addptr %arg1, {{.*}} : !tt.ptr<f32>, i64
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor [[BASE]], [%c16_i32, %c16_i32], [%c16_i64, %c1_i64] : <f32>, <16x16xf32>
// CHECK: scf.for
// CHECK-NOT: tt.make_tensor_descriptor
// CHECK:   arith.cmpi sge, [[PID]], %c0_i32 : i32
// CHECK:   [[LOAD:%.*]] = tt.descriptor_load [[DESC]]{{.*}} : !tt.tensordesc<16x16xf32> -> tensor<16x16xf32>
// CHECK:   tt.dot [[LOAD]]

// -----

// COM: A block with a zero-extent merged dimension fuses without crashing:
// COM: nothing divides by the block extent (issues/8073).
tt.func public @fuseZeroExtentMergedDim(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c8_i32 = arith.constant 8 : i32
  %c16_i32 = arith.constant 16 : i32
  %c1_i64 = arith.constant 1 : i64
  %c16_i64 = arith.constant 16 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<0x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c8_i32, %c16_i32, %c16_i32], [%c1024_i64, %c16_i64, %c1_i64] : <f32>, <1x0x16xf32>
  %1 = tt.descriptor_load %0[%c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<1x0x16xf32> -> tensor<1x0x16xf32>
  %2 = tt.reshape %1 : tensor<1x0x16xf32> -> tensor<0x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<0x16xf32> * tensor<16x16xf32> -> tensor<0x16xf32>
  tt.return
}
// CHECK-LABEL: fuseZeroExtentMergedDim
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor %arg1, [%c16_i32, %c16_i32], [%c16_i64, %c1_i64] : <f32>, <0x16xf32>
// CHECK: [[LOAD:%.*]] = tt.descriptor_load [[DESC]][%c0_i32, %c0_i32] : !tt.tensordesc<0x16xf32> -> tensor<0x16xf32>
// CHECK: tt.dot [[LOAD]], {{.*}} : tensor<0x16xf32> * tensor<16x16xf32> -> tensor<0x16xf32>

// -----

// COM: The fused descriptor keeps the original's padding option, so a load the
// COM: guard forces out of range pads with the descriptor's own value (NaN here,
// COM: `padding = 2`), as the rank-3 one did.
tt.func public @fuseKeepsPaddingOption(%arg0: tensor<16x16xf32>, %arg1: !tt.ptr<f32>, %off: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c8_i32 = arith.constant 8 : i32
  %c16_i32 = arith.constant 16 : i32
  %c1_i64 = arith.constant 1 : i64
  %c16_i64 = arith.constant 16 : i64
  %c256_i64 = arith.constant 256 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c8_i32, %c16_i32, %c16_i32], [%c256_i64, %c16_i64, %c1_i64] {padding = 2 : i32} : <f32>, <1x16x16xf32>
  %1 = tt.descriptor_load %0[%off, %c0_i32, %c0_i32] : !tt.tensordesc<1x16x16xf32> -> tensor<1x16x16xf32>
  %2 = tt.reshape %1 : tensor<1x16x16xf32> -> tensor<16x16xf32>
  %3 = tt.dot %2, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: fuseKeepsPaddingOption
// CHECK: [[BASE:%.*]] = tt.addptr %arg1, {{.*}} : !tt.ptr<f32>, i64
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor [[BASE]], [%c16_i32, %c16_i32], [%c16_i64, %c1_i64] {padding = 2 : i32} : <f32>, <16x16xf32>
// CHECK: [[LOAD:%.*]] = tt.descriptor_load [[DESC]]{{.*}} : !tt.tensordesc<16x16xf32> -> tensor<16x16xf32>
// CHECK: tt.dot [[LOAD]]

// -----

// COM: A sub-byte element type has no whole-element stride to offset the base
// COM: by (packed FP6, for one, counts its outer strides in bytes), so it
// COM: keeps the rank-3 load.
tt.func public @noFuseSubByteElementType(%arg0: tensor<16x16xf16>, %arg1: !tt.ptr<i4>) {
  %c0_i32 = arith.constant 0 : i32
  %c2_i32 = arith.constant 2 : i32
  %c8_i32 = arith.constant 8 : i32
  %c16_i32 = arith.constant 16 : i32
  %c1_i64 = arith.constant 1 : i64
  %c16_i64 = arith.constant 16 : i64
  %c256_i64 = arith.constant 256 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %0 = tt.make_tensor_descriptor %arg1, [%c8_i32, %c16_i32, %c16_i32], [%c256_i64, %c16_i64, %c1_i64] : <i4>, <1x16x16xi4>
  %1 = tt.descriptor_load %0[%c2_i32, %c0_i32, %c0_i32] : !tt.tensordesc<1x16x16xi4> -> tensor<1x16x16xi4>
  %2 = tt.reshape %1 : tensor<1x16x16xi4> -> tensor<16x16xi4>
  %3 = arith.sitofp %2 : tensor<16x16xi4> to tensor<16x16xf16>
  %4 = tt.dot %3, %arg0, %cst, inputPrecision = tf32 : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
  tt.return
}
// CHECK-LABEL: noFuseSubByteElementType
// CHECK-NOT: tt.addptr
// CHECK: [[LOAD:%.*]] = tt.descriptor_load {{.*}} : !tt.tensordesc<1x16x16xi4> -> tensor<1x16x16xi4>
// CHECK: [[RESHAPE:%.*]] = tt.reshape [[LOAD]] : tensor<1x16x16xi4> -> tensor<16x16xi4>
// CHECK: arith.sitofp [[RESHAPE]]

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
// CHECK: [[LOAD:%.*]] = tt.descriptor_load {{.*}} -> tensor<64x1x128xbf16>
// CHECK: [[RESHAPE:%.*]] = tt.reshape [[LOAD]] : tensor<64x1x128xbf16> -> tensor<64x128xbf16>
// CHECK: tt.dot [[RESHAPE]]

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
// CHECK: [[LOAD:%.*]] = tt.descriptor_load {{.*}} -> tensor<64x1x128xbf16>
// CHECK: [[RESHAPE:%.*]] = tt.reshape [[LOAD]] : tensor<64x1x128xbf16> -> tensor<64x128xbf16>
// CHECK: tt.dot [[RESHAPE]]

// -----

// COM: The #8511 Q chain after LICM. Hoisting retypes the chain in place, so it
// COM: stays outside the loop, and the shared rank-3 constant keeps its user.
tt.func public @hoistReshapeOverElementwiseChain(%arg0: tensor<64x64xbf16>, %arg1: !tt.ptr<bf16>, %arg2: tensor<1x128x64xf32>, %arg3: tensor<128x64xf32>) -> tensor<1x128x64xf32> {
  %cst = arith.constant dense<1.250000e-01> : tensor<1x128x64xf32>
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %c128_i32 = arith.constant 128 : i32
  %c1_i64 = arith.constant 1 : i64
  %c64_i64 = arith.constant 64 : i64
  %c8192_i64 = arith.constant 8192 : i64
  %0 = tt.make_tensor_descriptor %arg1, [%c128_i32, %c128_i32, %c64_i32], [%c8192_i64, %c64_i64, %c1_i64] : <bf16>, <1x128x64xbf16>
  %1 = tt.descriptor_load %0[%c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<1x128x64xbf16> -> tensor<1x128x64xbf16>
  %2 = arith.extf %1 : tensor<1x128x64xbf16> to tensor<1x128x64xf32>
  %3 = arith.mulf %2, %cst : tensor<1x128x64xf32>
  %4 = arith.truncf %3 : tensor<1x128x64xf32> to tensor<1x128x64xbf16>
  %5 = tt.reshape %4 : tensor<1x128x64xbf16> -> tensor<128x64xbf16>
  %6 = scf.for %arg4 = %c0_i32 to %c128_i32 step %c64_i32 iter_args(%arg5 = %arg3) -> (tensor<128x64xf32>) : i32 {
    %8 = tt.dot %5, %arg0, %arg5 : tensor<128x64xbf16> * tensor<64x64xbf16> -> tensor<128x64xf32>
    scf.yield %8 : tensor<128x64xf32>
  }
  %7 = arith.mulf %arg2, %cst : tensor<1x128x64xf32>
  tt.return %7 : tensor<1x128x64xf32>
}
// CHECK-LABEL: hoistReshapeOverElementwiseChain
// CHECK-DAG: [[CST3D:%.*]] = arith.constant dense<1.250000e-01> : tensor<1x128x64xf32>
// CHECK-DAG: [[CST2D:%.*]] = arith.constant dense<1.250000e-01> : tensor<128x64xf32>
// CHECK-DAG: [[DESC:%.*]] = tt.make_tensor_descriptor {{.*}} : <bf16>, <128x64xbf16>
// CHECK-DAG: [[LOAD:%.*]] = tt.descriptor_load [[DESC]]{{.*}} -> tensor<128x64xbf16>
// CHECK-DAG: [[EXT:%.*]] = arith.extf [[LOAD]] : tensor<128x64xbf16> to tensor<128x64xf32>
// CHECK: [[MUL:%.*]] = arith.mulf [[EXT]], [[CST2D]] : tensor<128x64xf32>
// CHECK: [[TRUNC:%.*]] = arith.truncf [[MUL]] : tensor<128x64xf32> to tensor<128x64xbf16>
// CHECK-NOT: tt.reshape
// CHECK: scf.for
// CHECK: tt.dot [[TRUNC]], %arg0
// CHECK: arith.mulf %arg2, [[CST3D]] : tensor<1x128x64xf32>

// -----

tt.func public @hoistReshapeOverSplatOperand(%arg0: tensor<64x64xbf16>, %arg1: !tt.ptr<bf16>, %arg2: bf16, %arg3: tensor<128x64xf32>, %arg4: tensor<1x128x64xbf16>) -> tensor<1x128x64xbf16> {
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %c128_i32 = arith.constant 128 : i32
  %c1_i64 = arith.constant 1 : i64
  %c64_i64 = arith.constant 64 : i64
  %c8192_i64 = arith.constant 8192 : i64
  %0 = tt.make_tensor_descriptor %arg1, [%c128_i32, %c128_i32, %c64_i32], [%c8192_i64, %c64_i64, %c1_i64] : <bf16>, <1x128x64xbf16>
  %1 = tt.descriptor_load %0[%c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<1x128x64xbf16> -> tensor<1x128x64xbf16>
  %2 = tt.splat %arg2 : bf16 -> tensor<1x128x64xbf16>
  %3 = arith.mulf %1, %2 : tensor<1x128x64xbf16>
  %4 = tt.reshape %3 : tensor<1x128x64xbf16> -> tensor<128x64xbf16>
  %5 = tt.dot %4, %arg0, %arg3 : tensor<128x64xbf16> * tensor<64x64xbf16> -> tensor<128x64xf32>
  %6 = arith.mulf %arg4, %2 : tensor<1x128x64xbf16>
  tt.return %6 : tensor<1x128x64xbf16>
}
// CHECK-LABEL: hoistReshapeOverSplatOperand
// CHECK: [[LOAD:%.*]] = tt.descriptor_load {{.*}} -> tensor<128x64xbf16>
// CHECK-DAG: [[SPLAT:%.*]] = tt.splat %arg2 : bf16 -> tensor<128x64xbf16>
// CHECK-DAG: [[SPLAT3D:%.*]] = tt.splat %arg2 : bf16 -> tensor<1x128x64xbf16>
// CHECK: arith.mulf [[LOAD]], [[SPLAT]] : tensor<128x64xbf16>
// CHECK: arith.mulf %arg4, [[SPLAT3D]] : tensor<1x128x64xbf16>

// -----

// COM: The #8511 K chain: a permute that keeps the unit dimension first.
tt.func public @hoistReshapeOverTrans(%arg0: tensor<128x64xbf16>, %arg1: !tt.ptr<bf16>, %arg2: tensor<128x32xf32>) {
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %c128_i32 = arith.constant 128 : i32
  %c1_i64 = arith.constant 1 : i64
  %c64_i64 = arith.constant 64 : i64
  %c8192_i64 = arith.constant 8192 : i64
  %0 = tt.make_tensor_descriptor %arg1, [%c128_i32, %c128_i32, %c64_i32], [%c8192_i64, %c64_i64, %c1_i64] : <bf16>, <1x32x64xbf16>
  %1 = tt.descriptor_load %0[%c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<1x32x64xbf16> -> tensor<1x32x64xbf16>
  %2 = tt.trans %1 {order = array<i32: 0, 2, 1>} : tensor<1x32x64xbf16> -> tensor<1x64x32xbf16>
  %3 = tt.reshape %2 : tensor<1x64x32xbf16> -> tensor<64x32xbf16>
  %4 = tt.dot %arg0, %3, %arg2 : tensor<128x64xbf16> * tensor<64x32xbf16> -> tensor<128x32xf32>
  tt.return
}
// CHECK-LABEL: hoistReshapeOverTrans
// CHECK: tt.descriptor_load {{.*}} -> tensor<32x64xbf16>
// CHECK: tt.trans {{.*}} {order = array<i32: 1, 0>} : tensor<32x64xbf16> -> tensor<64x32xbf16>

// -----

// COM: The op after a crossed tt.trans must take the rebuilt trans.
tt.func public @hoistReshapeOverExtfAfterTrans(%arg0: !tt.tensordesc<1x32x64xbf16>, %arg1: tensor<128x64xf32>, %arg2: tensor<128x32xf32>) -> tensor<128x32xf32> {
  %c0_i32 = arith.constant 0 : i32
  %0 = tt.descriptor_load %arg0[%c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<1x32x64xbf16> -> tensor<1x32x64xbf16>
  %1 = tt.trans %0 {order = array<i32: 0, 2, 1>} : tensor<1x32x64xbf16> -> tensor<1x64x32xbf16>
  %2 = arith.extf %1 : tensor<1x64x32xbf16> to tensor<1x64x32xf32>
  %3 = tt.reshape %2 : tensor<1x64x32xf32> -> tensor<64x32xf32>
  %4 = tt.dot %arg1, %3, %arg2 : tensor<128x64xf32> * tensor<64x32xf32> -> tensor<128x32xf32>
  tt.return %4 : tensor<128x32xf32>
}
// CHECK-LABEL: hoistReshapeOverExtfAfterTrans
// CHECK: arith.extf {{.*}} : tensor<64x32xbf16> to tensor<64x32xf32>

// -----

// COM: A tt.trans after a retyped elementwise op, and two tt.trans in one chain.
tt.func public @hoistReshapeOverTransAfterNegf(%arg0: !tt.tensordesc<1x32x64xbf16>) -> tensor<32x64xbf16> {
  %c0_i32 = arith.constant 0 : i32
  %0 = tt.descriptor_load %arg0[%c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<1x32x64xbf16> -> tensor<1x32x64xbf16>
  %1 = tt.trans %0 {order = array<i32: 0, 2, 1>} : tensor<1x32x64xbf16> -> tensor<1x64x32xbf16>
  %2 = arith.negf %1 : tensor<1x64x32xbf16>
  %3 = tt.trans %2 {order = array<i32: 0, 2, 1>} : tensor<1x64x32xbf16> -> tensor<1x32x64xbf16>
  %4 = tt.reshape %3 : tensor<1x32x64xbf16> -> tensor<32x64xbf16>
  tt.return %4 : tensor<32x64xbf16>
}
// CHECK-LABEL: hoistReshapeOverTransAfterNegf
// CHECK: [[RESHAPE:%.*]] = tt.reshape {{.*}} : tensor<1x32x64xbf16> -> tensor<32x64xbf16>
// CHECK: [[T1:%.*]] = tt.trans [[RESHAPE]] {order = array<i32: 1, 0>} : tensor<32x64xbf16> -> tensor<64x32xbf16>
// CHECK: [[NEG:%.*]] = arith.negf [[T1]] : tensor<64x32xbf16>
// CHECK: [[T2:%.*]] = tt.trans [[NEG]] {order = array<i32: 1, 0>} : tensor<64x32xbf16> -> tensor<32x64xbf16>
// CHECK: tt.return [[T2]]

// -----

// COM: Hoisted without a dot user; Combine rank-reduces such loads later.
tt.func public @hoistWithoutDot(%arg0: !tt.tensordesc<1x64x64xbf16>, %arg1: !tt.tensordesc<64x64xbf16>, %arg2: bf16) {
  %c0_i32 = arith.constant 0 : i32
  %0 = tt.descriptor_load %arg0[%c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<1x64x64xbf16> -> tensor<1x64x64xbf16>
  %1 = tt.splat %arg2 : bf16 -> tensor<1x64x64xbf16>
  %2 = arith.mulf %0, %1 : tensor<1x64x64xbf16>
  %3 = tt.reshape %2 : tensor<1x64x64xbf16> -> tensor<64x64xbf16>
  tt.descriptor_store %arg1[%c0_i32, %c0_i32], %3 : !tt.tensordesc<64x64xbf16>, tensor<64x64xbf16>
  tt.return
}
// CHECK-LABEL: hoistWithoutDot
// CHECK: [[LOAD:%.*]] = tt.descriptor_load {{.*}} -> tensor<1x64x64xbf16>
// CHECK-NEXT: tt.reshape [[LOAD]] : tensor<1x64x64xbf16> -> tensor<64x64xbf16>
// CHECK-NOT: tt.splat %arg2 : bf16 -> tensor<1x64x64xbf16>

// -----

tt.func public @hoistReshapeOverScalarOperand(%arg0: !tt.tensordesc<1x64x64xbf16>, %arg1: bf16, %arg2: i1) -> tensor<64x64xbf16> {
  %c0_i32 = arith.constant 0 : i32
  %0 = tt.descriptor_load %arg0[%c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<1x64x64xbf16> -> tensor<1x64x64xbf16>
  %1 = tt.splat %arg1 : bf16 -> tensor<1x64x64xbf16>
  %2 = arith.select %arg2, %0, %1 : tensor<1x64x64xbf16>
  %3 = tt.reshape %2 : tensor<1x64x64xbf16> -> tensor<64x64xbf16>
  tt.return %3 : tensor<64x64xbf16>
}
// CHECK-LABEL: hoistReshapeOverScalarOperand
// CHECK: arith.select %arg2, {{.*}} : tensor<64x64xbf16>

// -----

tt.func public @noHoistNonConstantOperand(%arg0: tensor<64x64xbf16>, %arg1: !tt.ptr<bf16>, %arg2: tensor<64x64xf32>, %arg3: tensor<1x64x64xbf16>) {
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %c1_i64 = arith.constant 1 : i64
  %c64_i64 = arith.constant 64 : i64
  %c4096_i64 = arith.constant 4096 : i64
  %0 = tt.make_tensor_descriptor %arg1, [%c64_i32, %c64_i32, %c64_i32], [%c4096_i64, %c64_i64, %c1_i64] : <bf16>, <1x64x64xbf16>
  %1 = tt.descriptor_load %0[%c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<1x64x64xbf16> -> tensor<1x64x64xbf16>
  %2 = arith.mulf %1, %arg3 : tensor<1x64x64xbf16>
  %3 = tt.reshape %2 : tensor<1x64x64xbf16> -> tensor<64x64xbf16>
  %4 = tt.dot %3, %arg0, %arg2 : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
  tt.return
}
// CHECK-LABEL: noHoistNonConstantOperand
// CHECK: tt.descriptor_load {{.*}} -> tensor<1x64x64xbf16>
// CHECK: [[X:%.*]] = arith.mulf
// CHECK: tt.reshape [[X]] : tensor<1x64x64xbf16> -> tensor<64x64xbf16>

// -----

tt.func public @noHoistMultiUse(%arg0: tensor<64x64xbf16>, %arg1: !tt.ptr<bf16>, %arg2: tensor<64x64xf32>) -> tensor<1x64x64xbf16> {
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %c1_i64 = arith.constant 1 : i64
  %c64_i64 = arith.constant 64 : i64
  %c4096_i64 = arith.constant 4096 : i64
  %0 = tt.make_tensor_descriptor %arg1, [%c64_i32, %c64_i32, %c64_i32], [%c4096_i64, %c64_i64, %c1_i64] : <bf16>, <1x64x64xbf16>
  %1 = tt.descriptor_load %0[%c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<1x64x64xbf16> -> tensor<1x64x64xbf16>
  %2 = arith.negf %1 : tensor<1x64x64xbf16>
  %3 = tt.reshape %2 : tensor<1x64x64xbf16> -> tensor<64x64xbf16>
  %4 = tt.dot %3, %arg0, %arg2 : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
  tt.return %2 : tensor<1x64x64xbf16>
}
// CHECK-LABEL: noHoistMultiUse
// CHECK: tt.descriptor_load {{.*}} -> tensor<1x64x64xbf16>
// CHECK: [[X:%.*]] = arith.negf
// CHECK: tt.reshape [[X]] : tensor<1x64x64xbf16> -> tensor<64x64xbf16>

// -----

// COM: The trans keeps a leading unit dim; only order[0] != 0 declines it.
tt.func public @noHoistTransMovingUnitDim(%arg0: tensor<64x16xbf16>, %arg1: !tt.ptr<bf16>, %arg2: tensor<1x16xf32>) {
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %c1_i64 = arith.constant 1 : i64
  %c64_i64 = arith.constant 64 : i64
  %c4096_i64 = arith.constant 4096 : i64
  %0 = tt.make_tensor_descriptor %arg1, [%c64_i32, %c64_i32, %c64_i32], [%c4096_i64, %c64_i64, %c1_i64] : <bf16>, <1x1x64xbf16>
  %1 = tt.descriptor_load %0[%c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<1x1x64xbf16> -> tensor<1x1x64xbf16>
  %2 = tt.trans %1 {order = array<i32: 1, 0, 2>} : tensor<1x1x64xbf16> -> tensor<1x1x64xbf16>
  %3 = tt.reshape %2 : tensor<1x1x64xbf16> -> tensor<1x64xbf16>
  %4 = tt.dot %3, %arg0, %arg2 : tensor<1x64xbf16> * tensor<64x16xbf16> -> tensor<1x16xf32>
  tt.return
}
// CHECK-LABEL: noHoistTransMovingUnitDim
// CHECK: tt.descriptor_load {{.*}} -> tensor<1x1x64xbf16>
// CHECK: [[X:%.*]] = tt.trans
// CHECK: tt.reshape [[X]] : tensor<1x1x64xbf16> -> tensor<1x64xbf16>

// -----

// COM: A plain tt.load is not a descriptor load.
tt.func public @noHoistWithoutDescriptorLoad(%arg0: tensor<64x64xbf16>, %arg1: !tt.ptr<bf16>, %arg2: tensor<64x64xf32>) {
  %0 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<1x64x64x!tt.ptr<bf16>>
  %1 = tt.load %0 : tensor<1x64x64x!tt.ptr<bf16>>
  %2 = arith.negf %1 : tensor<1x64x64xbf16>
  %3 = tt.reshape %2 : tensor<1x64x64xbf16> -> tensor<64x64xbf16>
  %4 = tt.dot %3, %arg0, %arg2 : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
  tt.return
}
// CHECK-LABEL: noHoistWithoutDescriptorLoad
// CHECK: tt.load
// CHECK: [[X:%.*]] = arith.negf
// CHECK: tt.reshape [[X]] : tensor<1x64x64xbf16> -> tensor<64x64xbf16>

// -----

// COM: Pure and shape-preserving but not elementwise: a scan along dim 1.
tt.func public @noHoistNonElementwise(%arg0: tensor<64x64xbf16>, %arg1: !tt.ptr<bf16>, %arg2: tensor<64x64xf32>) {
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %c1_i64 = arith.constant 1 : i64
  %c64_i64 = arith.constant 64 : i64
  %c4096_i64 = arith.constant 4096 : i64
  %0 = tt.make_tensor_descriptor %arg1, [%c64_i32, %c64_i32, %c64_i32], [%c4096_i64, %c64_i64, %c1_i64] : <bf16>, <1x64x64xbf16>
  %1 = tt.descriptor_load %0[%c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<1x64x64xbf16> -> tensor<1x64x64xbf16>
  %2 = "tt.scan"(%1) <{axis = 1 : i32, reverse = false}> ({
  ^bb0(%arg3: bf16, %arg4: bf16):
    %5 = arith.addf %arg3, %arg4 : bf16
    tt.scan.return %5 : bf16
  }) : (tensor<1x64x64xbf16>) -> tensor<1x64x64xbf16>
  %3 = tt.reshape %2 : tensor<1x64x64xbf16> -> tensor<64x64xbf16>
  %4 = tt.dot %3, %arg0, %arg2 : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
  tt.return
}
// CHECK-LABEL: noHoistNonElementwise
// CHECK: tt.descriptor_load {{.*}} -> tensor<1x64x64xbf16>
// CHECK: [[X:%.*]] = "tt.scan"
// CHECK: tt.reshape [[X]] : tensor<1x64x64xbf16> -> tensor<64x64xbf16>

// -----

// COM: A 3-entry hint left on a rank-2 value would corrupt AxisInfo.
tt.func public @noHoistRankIndexedHint(%arg0: tensor<64x64xbf16>, %arg1: !tt.ptr<bf16>, %arg2: tensor<64x64xf32>) {
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %c1_i64 = arith.constant 1 : i64
  %c64_i64 = arith.constant 64 : i64
  %c4096_i64 = arith.constant 4096 : i64
  %0 = tt.make_tensor_descriptor %arg1, [%c64_i32, %c64_i32, %c64_i32], [%c4096_i64, %c64_i64, %c1_i64] : <bf16>, <1x64x64xbf16>
  %1 = tt.descriptor_load %0[%c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<1x64x64xbf16> -> tensor<1x64x64xbf16>
  %2 = tt.annotate %1 {tt.contiguity = dense<[1, 1, 16]> : tensor<3xi32>} : tensor<1x64x64xbf16>
  %3 = tt.reshape %2 : tensor<1x64x64xbf16> -> tensor<64x64xbf16>
  %4 = tt.dot %3, %arg0, %arg2 : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
  tt.return
}
// CHECK-LABEL: noHoistRankIndexedHint
// CHECK: tt.descriptor_load {{.*}} -> tensor<1x64x64xbf16>
// CHECK: [[X:%.*]] = tt.annotate
// CHECK: tt.reshape [[X]] : tensor<1x64x64xbf16> -> tensor<64x64xbf16>

// -----

// COM: The fusion copies the load's attributes onto its rank-2 replacement.
tt.func public @noHoistHintedLoad(%arg0: tensor<64x64xbf16>, %arg1: !tt.ptr<bf16>, %arg2: tensor<64x64xf32>, %arg3: bf16) {
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %c1_i64 = arith.constant 1 : i64
  %c64_i64 = arith.constant 64 : i64
  %c4096_i64 = arith.constant 4096 : i64
  %0 = tt.make_tensor_descriptor %arg1, [%c64_i32, %c64_i32, %c64_i32], [%c4096_i64, %c64_i64, %c1_i64] : <bf16>, <1x64x64xbf16>
  %1 = tt.descriptor_load %0[%c0_i32, %c0_i32, %c0_i32] {tt.divisibility = dense<[1, 1, 16]> : tensor<3xi32>} : !tt.tensordesc<1x64x64xbf16> -> tensor<1x64x64xbf16>
  %2 = tt.splat %arg3 : bf16 -> tensor<1x64x64xbf16>
  %3 = arith.mulf %1, %2 : tensor<1x64x64xbf16>
  %4 = tt.reshape %3 : tensor<1x64x64xbf16> -> tensor<64x64xbf16>
  %5 = tt.dot %4, %arg0, %arg2 : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
  tt.return
}
// CHECK-LABEL: noHoistHintedLoad
// CHECK: tt.descriptor_load {{.*}} -> tensor<1x64x64xbf16>
// CHECK: [[X:%.*]] = arith.mulf
// CHECK: tt.reshape [[X]] : tensor<1x64x64xbf16> -> tensor<64x64xbf16>

// -----

// COM: Inline asm receives elements grouped by thread, not in logical order.
tt.func public @noHoistInlineAsm(%arg0: tensor<64x64xbf16>, %arg1: !tt.ptr<bf16>, %arg2: tensor<64x64xf32>) {
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %c1_i64 = arith.constant 1 : i64
  %c64_i64 = arith.constant 64 : i64
  %c4096_i64 = arith.constant 4096 : i64
  %0 = tt.make_tensor_descriptor %arg1, [%c64_i32, %c64_i32, %c64_i32], [%c4096_i64, %c64_i64, %c1_i64] : <bf16>, <1x64x64xbf16>
  %1 = tt.descriptor_load %0[%c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<1x64x64xbf16> -> tensor<1x64x64xbf16>
  %2 = tt.elementwise_inline_asm "mov.b16 $0, $1;" {constraints = "=h,h", packed_element = 1 : i32, pure = true} %1 : tensor<1x64x64xbf16> -> tensor<1x64x64xbf16>
  %3 = tt.reshape %2 : tensor<1x64x64xbf16> -> tensor<64x64xbf16>
  %4 = tt.dot %3, %arg0, %arg2 : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
  tt.return
}
// CHECK-LABEL: noHoistInlineAsm
// CHECK: tt.descriptor_load {{.*}} -> tensor<1x64x64xbf16>
// CHECK: [[X:%.*]] = tt.elementwise_inline_asm
// CHECK: tt.reshape [[X]] : tensor<1x64x64xbf16> -> tensor<64x64xbf16>

// -----

tt.func public @noHoistMultiResult(%arg0: !tt.tensordesc<1x64x64xi32>, %arg1: i32) -> tensor<64x64xi32> {
  %c0_i32 = arith.constant 0 : i32
  %0 = tt.descriptor_load %arg0[%c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<1x64x64xi32> -> tensor<1x64x64xi32>
  %1 = tt.splat %arg1 : i32 -> tensor<1x64x64xi32>
  %low, %high = arith.mulsi_extended %0, %1 : tensor<1x64x64xi32>
  %2 = tt.reshape %high : tensor<1x64x64xi32> -> tensor<64x64xi32>
  tt.return %2 : tensor<64x64xi32>
}
// CHECK-LABEL: noHoistMultiResult
// CHECK: tt.descriptor_load {{.*}} -> tensor<1x64x64xi32>
// CHECK: {{%[a-z0-9_]+}}, [[X:%[a-z0-9_]+]] = arith.mulsi_extended
// CHECK: tt.reshape [[X]] : tensor<1x64x64xi32> -> tensor<64x64xi32>
