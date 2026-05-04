// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s

// COM: Test that ttig.2d_block_load with dot_op A encoding generates
// COM: triton_gen.2Dblockload instructions.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: @block_load_dot_a
  tt.func public @block_load_dot_a(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
    // For DPAS A with f16: tile_height=8, tile_width=16, v_blocks=2.
    // CHECK-COUNT-2: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 2, transpose = false, vnni_transform = false, cache_control = Default}
    %0 = ttig.2d_block_load %arg0, %arg1, %arg2, %arg3[%arg4, %arg5] {row_major} : !tt.ptr<f16> -> tensor<64x32xf16, #dot0>
    tt.return
  }
}

// -----

// COM: Test that ttig.2d_block_load with pad_nan generates block loads
// COM: followed by NaN select for out-of-bounds elements.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: @block_load_pad_nan
  tt.func public @block_load_pad_nan(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
    // CHECK: triton_gen.2Dblockload
    // CHECK: llvm.select
    %0 = ttig.2d_block_load %arg0, %arg1, %arg2, %arg3[%arg4, %arg5] {row_major, pad_nan} : !tt.ptr<f16> -> tensor<64x32xf16, #dot0>
    tt.return
  }
}

// -----

// COM: Test column-major block load with pad_nan. Exercises the dim-swap in
// COM: the NaN mask path (surfaceColDim/surfaceRowDim differ from row_major).
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: @block_load_column_major_pad_nan
  tt.func public @block_load_column_major_pad_nan(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
    // CHECK: triton_gen.2Dblockload {{.*}} transpose = true
    // CHECK: llvm.select
    %0 = ttig.2d_block_load %arg0, %arg1, %arg2, %arg3[%arg4, %arg5] {column_major, pad_nan} : !tt.ptr<f16> -> tensor<32x32xf16, #dot1>
    tt.return
  }
}

// -----

// COM: Test column-major (transposed) block load with dot_op B encoding.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: @block_load_column_major
  tt.func public @block_load_column_major(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
    // CHECK: triton_gen.2Dblockload {{.*}} transpose = true
    %0 = ttig.2d_block_load %arg0, %arg1, %arg2, %arg3[%arg4, %arg5] {column_major} : !tt.ptr<f16> -> tensor<32x32xf16, #dot1>
    tt.return
  }
}
