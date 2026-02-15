// RUN: triton-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// -----

tt.func @ttig.prefetch(%arg0: !tt.ptr<tensor<2x32xf32>>, %arg1: tensor<2x32xi1>) {
  // CHECK-LABEL: @ttig.prefetch
  // CHECK:         ttig.prefetch %arg0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<2x32xf32>>
  ttig.prefetch %arg0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<2x32xf32>>
  // CHECK:         ttig.prefetch %arg0, %arg1 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<2x32xf32>>
  ttig.prefetch %arg0, %arg1 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<2x32xf32>>
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
