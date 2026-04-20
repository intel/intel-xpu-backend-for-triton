// RUN: triton-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// -----

tt.func @ttig.prefetch(%arg0: tensor<2x32x!tt.ptr<f32>>, %arg1: tensor<2x32xi1>) {
  // CHECK-LABEL: @ttig.prefetch
  // CHECK:         ttig.prefetch %arg0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<2x32x!tt.ptr<f32>>
  ttig.prefetch %arg0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<2x32x!tt.ptr<f32>>
  // CHECK:         ttig.prefetch %arg0, %arg1 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<2x32x!tt.ptr<f32>>
  ttig.prefetch %arg0, %arg1 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<2x32x!tt.ptr<f32>>
  tt.return
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_2d_block_io} {
  tt.func @ttig.sub_group_transpose(%local_buffer : !tt.ptr<f16, 3>, %src : tensor<16x16xf16>) -> tensor<16x16xf16> {
    // CHECK-LABEL: @ttig.sub_group_transpose
    // CHECK:         ttig.sub_group_transpose %arg0, %arg1 : tensor<16x16xf16>
    %res = ttig.sub_group_transpose %local_buffer, %src : tensor<16x16xf16>
    tt.return %res : tensor<16x16xf16>
  }
}

// -----

tt.func @ttig.descriptor_prefetch(%desc: !tt.tensordesc<tensor<256x32xf16>>, %x: i32, %y: i32) {
  // CHECK-LABEL: @ttig.descriptor_prefetch
  // CHECK:         ttig.descriptor_prefetch %arg0[%arg1, %arg2] : !tt.tensordesc<tensor<256x32xf16>>
  ttig.descriptor_prefetch %desc[%x, %y] cacheModifier = none evictionPolicy = evict_normal : !tt.tensordesc<tensor<256x32xf16>>
  // CHECK:         ttig.descriptor_prefetch %arg0[%arg1, %arg2] : !tt.tensordesc<tensor<256x32xf16>>
  ttig.descriptor_prefetch %desc[%x, %y] : !tt.tensordesc<tensor<256x32xf16>>
  tt.return
}
