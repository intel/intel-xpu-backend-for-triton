// RUN: triton-opt %s -split-input-file -triton-intel-fuse-reshape | FileCheck %s

// COM: A reshape that only prepends unit dims folds into the store, which then
// COM: stores the reshape's source (a rank-reduced store); the dead reshape goes.
tt.func public @foldReshapeIntoRankReducedStore(%arg0: !tt.tensordesc<1x128x64xbf16>, %arg1: tensor<128x64xbf16>, %arg2: i32, %arg3: i32) {
  %c0_i32 = arith.constant 0 : i32
  %0 = tt.reshape %arg1 : tensor<128x64xbf16> -> tensor<1x128x64xbf16>
  tt.descriptor_store %arg0[%arg2, %arg3, %c0_i32], %0 : !tt.tensordesc<1x128x64xbf16>, tensor<1x128x64xbf16>
  tt.return
}
// CHECK-LABEL: foldReshapeIntoRankReducedStore
// CHECK-NOT: tt.reshape
// CHECK: tt.descriptor_store %{{.*}}[%arg2, %arg3, %{{.*}}], %arg1 : !tt.tensordesc<1x128x64xbf16>, tensor<128x64xbf16>

// -----

// COM: Any number of leading unit dims is dropped, here two.
tt.func public @foldTwoLeadingUnitDims(%arg0: !tt.tensordesc<1x1x128x64xbf16>, %arg1: tensor<128x64xbf16>, %arg2: i32) {
  %0 = tt.reshape %arg1 : tensor<128x64xbf16> -> tensor<1x1x128x64xbf16>
  tt.descriptor_store %arg0[%arg2, %arg2, %arg2, %arg2], %0 : !tt.tensordesc<1x1x128x64xbf16>, tensor<1x1x128x64xbf16>
  tt.return
}
// CHECK-LABEL: foldTwoLeadingUnitDims
// CHECK: tt.descriptor_store {{.*}}, %arg1 : !tt.tensordesc<1x1x128x64xbf16>, tensor<128x64xbf16>

// -----

// COM: A reshape with another user still folds into the store, and is kept.
tt.func public @foldKeepsReshapeWithOtherUses(%arg0: !tt.tensordesc<1x128x64xbf16>, %arg1: tensor<128x64xbf16>, %arg2: i32) -> tensor<1x128x64xbf16> {
  %0 = tt.reshape %arg1 : tensor<128x64xbf16> -> tensor<1x128x64xbf16>
  tt.descriptor_store %arg0[%arg2, %arg2, %arg2], %0 : !tt.tensordesc<1x128x64xbf16>, tensor<1x128x64xbf16>
  tt.return %0 : tensor<1x128x64xbf16>
}
// CHECK-LABEL: foldKeepsReshapeWithOtherUses
// CHECK: tt.reshape
// CHECK: tt.descriptor_store {{.*}}, %arg1 :

// -----

// COM: An allow_reorder reshape may permute elements, so it is not folded.
tt.func public @noFoldAllowReorder(%arg0: !tt.tensordesc<1x128x64xbf16>, %arg1: tensor<128x64xbf16>, %arg2: i32) {
  %0 = tt.reshape %arg1 allow_reorder : tensor<128x64xbf16> -> tensor<1x128x64xbf16>
  tt.descriptor_store %arg0[%arg2, %arg2, %arg2], %0 : !tt.tensordesc<1x128x64xbf16>, tensor<1x128x64xbf16>
  tt.return
}
// CHECK-LABEL: noFoldAllowReorder
// CHECK: tt.descriptor_store {{.*}}, %0 :

// -----

// COM: The element counts match, but the block's inner dims (64x128) are not x's.
tt.func public @noFoldInnerShapeMismatch(%arg0: !tt.tensordesc<1x64x128xbf16>, %arg1: tensor<128x64xbf16>, %arg2: i32) {
  %0 = tt.reshape %arg1 : tensor<128x64xbf16> -> tensor<1x64x128xbf16>
  tt.descriptor_store %arg0[%arg2, %arg2, %arg2], %0 : !tt.tensordesc<1x64x128xbf16>, tensor<1x64x128xbf16>
  tt.return
}
// CHECK-LABEL: noFoldInnerShapeMismatch
// CHECK: tt.descriptor_store {{.*}}, %0 :

// -----

// COM: The store verifier only compares element counts, so it accepts a 2x64x64
// COM: value in a 1x128x64 block; a store whose value isn't block-shaped stays.
tt.func public @noFoldResultNotBlockShape(%arg0: !tt.tensordesc<1x128x64xbf16>, %arg1: tensor<128x64xbf16>, %arg2: i32) {
  %0 = tt.reshape %arg1 : tensor<128x64xbf16> -> tensor<2x64x64xbf16>
  tt.descriptor_store %arg0[%arg2, %arg2, %arg2], %0 : !tt.tensordesc<1x128x64xbf16>, tensor<2x64x64xbf16>
  tt.return
}
// CHECK-LABEL: noFoldResultNotBlockShape
// CHECK: tt.descriptor_store {{.*}}, %0 :

// -----

// COM: With zero elements the counts match although the dropped dim is 2, not 1.
tt.func public @noFoldZeroElementNonUnitLeading(%arg0: !tt.tensordesc<2x0x16xbf16>, %arg1: tensor<0x16xbf16>, %arg2: i32) {
  %0 = tt.reshape %arg1 : tensor<0x16xbf16> -> tensor<2x0x16xbf16>
  tt.descriptor_store %arg0[%arg2, %arg2, %arg2], %0 : !tt.tensordesc<2x0x16xbf16>, tensor<2x0x16xbf16>
  tt.return
}
// CHECK-LABEL: noFoldZeroElementNonUnitLeading
// CHECK: tt.descriptor_store {{.*}}, %0 :

// -----

// COM: A rank-1 value cannot take the 2D block store, so the 1x64 store stays.
tt.func public @noFoldRank1Source(%arg0: !tt.tensordesc<1x64xbf16>, %arg1: tensor<64xbf16>, %arg2: i32) {
  %0 = tt.reshape %arg1 : tensor<64xbf16> -> tensor<1x64xbf16>
  tt.descriptor_store %arg0[%arg2, %arg2], %0 : !tt.tensordesc<1x64xbf16>, tensor<1x64xbf16>
  tt.return
}
// CHECK-LABEL: noFoldRank1Source
// CHECK: tt.descriptor_store {{.*}}, %0 :
