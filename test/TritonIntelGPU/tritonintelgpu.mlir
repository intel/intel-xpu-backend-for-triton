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

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  tt.func @ttig.2d_block_load(%base_ptr: !tt.ptr<f16>, %width: i32, %height: i32, %pitch: i32, %x: i32, %y: i32) -> tensor<64x32xf16, #dot0> {
    // CHECK-LABEL: @ttig.2d_block_load
    // CHECK: ttig.2d_block_load %arg0, %arg1, %arg2, %arg3[%arg4, %arg5] {row_major}
    %0 = ttig.2d_block_load %base_ptr, %width, %height, %pitch[%x, %y] {row_major} : !tt.ptr<f16> -> tensor<64x32xf16, #dot0>
    tt.return %0 : tensor<64x32xf16, #dot0>
  }

  tt.func @ttig.2d_block_load_pad_nan(%base_ptr: !tt.ptr<f16>, %width: i32, %height: i32, %pitch: i32, %x: i32, %y: i32) -> tensor<64x32xf16, #dot0> {
    // CHECK-LABEL: @ttig.2d_block_load_pad_nan
    // CHECK: ttig.2d_block_load %arg0, %arg1, %arg2, %arg3[%arg4, %arg5] {row_major, pad_nan}
    %0 = ttig.2d_block_load %base_ptr, %width, %height, %pitch[%x, %y] {row_major, pad_nan} : !tt.ptr<f16> -> tensor<64x32xf16, #dot0>
    tt.return %0 : tensor<64x32xf16, #dot0>
  }
}

// -----

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  tt.func @ttig.2d_block_load_from_ptr(%ptr: tensor<64x32x!tt.ptr<f16>, #dot0>, %pitch: i32) -> tensor<64x32xf16, #dot0> {
    // CHECK-LABEL: @ttig.2d_block_load_from_ptr
    // CHECK: ttig.2d_block_load_from_ptr %arg0, %arg1 {row_major} {base_height = 8 : i32, base_width = 64 : i32}
    %0 = ttig.2d_block_load_from_ptr %ptr, %pitch {row_major} {base_width = 64 : i32, base_height = 8 : i32} : (tensor<64x32x!tt.ptr<f16>, #dot0>, i32) -> (tensor<64x32xf16, #dot0>)
    tt.return %0 : tensor<64x32xf16, #dot0>
  }

  tt.func @ttig.2d_block_load_from_ptr_with_mask(%ptr: tensor<64x32x!tt.ptr<f16>, #dot0>, %mask: tensor<64x32xi1, #dot0>, %other: tensor<64x32xf16, #dot0>, %pitch: i32) -> tensor<64x32xf16, #dot0> {
    // CHECK-LABEL: @ttig.2d_block_load_from_ptr_with_mask
    // CHECK: ttig.2d_block_load_from_ptr %arg0, %arg3, %arg1, %arg2 {row_major} {base_height = 8 : i32, base_width = 64 : i32}
    %0 = ttig.2d_block_load_from_ptr %ptr, %pitch, %mask, %other {row_major} {base_width = 64 : i32, base_height = 8 : i32} : (tensor<64x32x!tt.ptr<f16>, #dot0>, i32, tensor<64x32xi1, #dot0>, tensor<64x32xf16, #dot0>) -> (tensor<64x32xf16, #dot0>)
    tt.return %0 : tensor<64x32xf16, #dot0>
  }
}

// -----

tt.func @ttig.descriptor_prefetch(%desc: !tt.tensordesc<256x32xf16>, %x: i32, %y: i32) {
  // CHECK-LABEL: @ttig.descriptor_prefetch
  // CHECK:         ttig.descriptor_prefetch %arg0[%arg1, %arg2] : !tt.tensordesc<256x32xf16>
  ttig.descriptor_prefetch %desc[%x, %y] cacheModifier = none evictionPolicy = evict_normal : !tt.tensordesc<256x32xf16>
  // CHECK:         ttig.descriptor_prefetch %arg0[%arg1, %arg2] : !tt.tensordesc<256x32xf16>
  ttig.descriptor_prefetch %desc[%x, %y] : !tt.tensordesc<256x32xf16>
  tt.return
}

// -----

tt.func @ttig.extract_desc(%arg0: !tt.tensordesc<64x32xf16>) {
  // CHECK-LABEL: @ttig.extract_desc
  // CHECK: ttig.extract_desc %arg0[0] : <64x32xf16> -> i64
  %0 = ttig.extract_desc %arg0[0] : <64x32xf16> -> i64
  // CHECK: ttig.extract_desc %arg0[4] : <64x32xf16> -> !tt.ptr<f16>
  %1 = ttig.extract_desc %arg0[4] : <64x32xf16> -> !tt.ptr<f16>
  tt.return
}
