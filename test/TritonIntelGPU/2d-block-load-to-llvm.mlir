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

// -----

// COM: Test ttig.2d_block_load_from_ptr (pointer-tensor based) generates
// COM: triton_gen.2Dblockload instructions.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: @block_load_from_ptr
  tt.func public @block_load_from_ptr(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>> -> tensor<1x32xi32, #dot0>
    %2 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x32x!tt.ptr<f16>, #dot0>
    %3 = tt.addptr %2, %1 : tensor<1x32x!tt.ptr<f16>, #dot0>, tensor<1x32xi32, #dot0>
    %4 = tt.broadcast %3 : tensor<1x32x!tt.ptr<f16>, #dot0> -> tensor<64x32x!tt.ptr<f16>, #dot0>
    // CHECK-COUNT-2: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 2, transpose = false, vnni_transform = false, cache_control = Default}
    %5 = ttig.2d_block_load_from_ptr %4 {row_major} {base_height = 1 : i32, base_pitch = 64 : i32, base_width = 64 : i32} : (tensor<64x32x!tt.ptr<f16>, #dot0>) -> tensor<64x32xf16, #dot0>
    tt.return
  }
}

// -----

// COM: Test ttig.2d_block_load_from_ptr with mask generates block loads
// COM: with predicated out-of-bounds handling (select with other).
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: @block_load_from_ptr_masked
  tt.func public @block_load_from_ptr_masked(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf16, #dot0>
    %true = arith.constant dense<true> : tensor<64x32xi1, #dot0>
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>> -> tensor<1x32xi32, #dot0>
    %2 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x32x!tt.ptr<f16>, #dot0>
    %3 = tt.addptr %2, %1 : tensor<1x32x!tt.ptr<f16>, #dot0>, tensor<1x32xi32, #dot0>
    %4 = tt.broadcast %3 : tensor<1x32x!tt.ptr<f16>, #dot0> -> tensor<64x32x!tt.ptr<f16>, #dot0>
    // CHECK: triton_gen.2Dblockload
    // CHECK: llvm.select
    %5 = ttig.2d_block_load_from_ptr %4, %true, %cst {row_major} {base_height = 8 : i32, base_pitch = 64 : i32, base_width = 64 : i32} : (tensor<64x32x!tt.ptr<f16>, #dot0>, tensor<64x32xi1, #dot0>, tensor<64x32xf16, #dot0>) -> tensor<64x32xf16, #dot0>
    tt.return
  }
}

// -----

// COM: Test ttig.2d_block_load_from_ptr with stride=0 (broadcast). The
// COM: base_height=1 triggers row replication via shufflevector.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: @block_load_from_ptr_broadcast
  tt.func public @block_load_from_ptr_broadcast(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>> -> tensor<1x32xi32, #dot0>
    %2 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x32x!tt.ptr<f16>, #dot0>
    %3 = tt.addptr %2, %1 : tensor<1x32x!tt.ptr<f16>, #dot0>, tensor<1x32xi32, #dot0>
    %4 = tt.broadcast %3 : tensor<1x32x!tt.ptr<f16>, #dot0> -> tensor<64x32x!tt.ptr<f16>, #dot0>
    // CHECK: triton_gen.2Dblockload
    // CHECK: llvm.shufflevector
    %5 = ttig.2d_block_load_from_ptr %4 {row_major} {base_height = 1 : i32, base_pitch = 64 : i32, base_width = 64 : i32} : (tensor<64x32x!tt.ptr<f16>, #dot0>) -> tensor<64x32xf16, #dot0>
    tt.return
  }
}

// -----

// COM: Test ttig.2d_block_load_from_ptr with 1D->2D reshape stride attribute.
// COM: The blocked encoding + ttig.block_io_stride triggers the manual tile
// COM: construction path in the LLVM lowering.
#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [1, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: @block_load_from_ptr_1d_reshape
  tt.func public @block_load_from_ptr_1d_reshape(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x16xi32, #blocked>
    %2 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x16x!tt.ptr<f16>, #blocked>
    %3 = tt.addptr %2, %1 : tensor<1x16x!tt.ptr<f16>, #blocked>, tensor<1x16xi32, #blocked>
    %4 = tt.broadcast %3 : tensor<1x16x!tt.ptr<f16>, #blocked> -> tensor<64x16x!tt.ptr<f16>, #blocked>
    // CHECK: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, transpose = false, vnni_transform = false, cache_control = Default}
    %5 = ttig.2d_block_load_from_ptr %4 {row_major} {base_height = 64 : i32, base_pitch = 128 : i32, base_width = 32 : i32, ttig.block_io_stride = 64 : i64} : (tensor<64x16x!tt.ptr<f16>, #blocked>) -> tensor<64x16xf16, #blocked>
    tt.return
  }
}
