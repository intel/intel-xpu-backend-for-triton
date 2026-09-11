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
    %pitch = arith.constant 64 : i32
    // CHECK-COUNT-2: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 2, transpose = false, vnni_transform = false, cache_control = Default}
    %5 = ttig.2d_block_load_from_ptr %4, %pitch {row_major} {base_height = 1 : i32, base_width = 64 : i32} : (tensor<64x32x!tt.ptr<f16>, #dot0>, i32) -> tensor<64x32xf16, #dot0>
    tt.return
  }
}

// -----

// COM: Test ttig.2d_block_load_from_ptr with a sub-64-byte base_width. The
// COM: per-warp base_width (16 cols * 2 bytes = 32) is below the HW minimum,
// COM: so the lowering must floor the adjusted base_width to 64 bytes via
// COM: umax before emitting triton_gen.2Dblockload (see verifier constraint
// COM: "base width should be >= 64").
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: @block_load_from_ptr_small_base_width
  tt.func public @block_load_from_ptr_small_base_width(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #dot0}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #dot0}>> -> tensor<1x16xi32, #dot0>
    %2 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x16x!tt.ptr<f16>, #dot0>
    %3 = tt.addptr %2, %1 : tensor<1x16x!tt.ptr<f16>, #dot0>, tensor<1x16xi32, #dot0>
    %4 = tt.broadcast %3 : tensor<1x16x!tt.ptr<f16>, #dot0> -> tensor<64x16x!tt.ptr<f16>, #dot0>
    %pitch = arith.constant 128 : i32
    // CHECK: %[[C64:.*]] = llvm.mlir.constant(64 : i32) : i32
    // CHECK: llvm.intr.umax(%{{.*}}, %[[C64]]) : (i32, i32) -> i32
    // CHECK: triton_gen.2Dblockload
    %5 = ttig.2d_block_load_from_ptr %4, %pitch {row_major} {base_height = 1 : i32, base_width = 32 : i32} : (tensor<64x16x!tt.ptr<f16>, #dot0>, i32) -> tensor<64x16xf16, #dot0>
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
    %pitch = arith.constant 64 : i32
    // CHECK: llvm.select {{.*}} : i1, i32
    // CHECK: triton_gen.2Dblockload
    // CHECK: llvm.select
    %5 = ttig.2d_block_load_from_ptr %4, %pitch, %true, %cst {row_major} {base_height = 8 : i32, base_width = 64 : i32} : (tensor<64x32x!tt.ptr<f16>, #dot0>, i32, tensor<64x32xi1, #dot0>, tensor<64x32xf16, #dot0>) -> tensor<64x32xf16, #dot0>
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
    %pitch = arith.constant 64 : i32
    // CHECK: triton_gen.2Dblockload
    // CHECK: llvm.shufflevector
    %5 = ttig.2d_block_load_from_ptr %4, %pitch {row_major} {base_height = 1 : i32, base_width = 64 : i32} : (tensor<64x32x!tt.ptr<f16>, #dot0>, i32) -> tensor<64x32xf16, #dot0>
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
    %pitch = arith.constant 128 : i32
    // CHECK: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, transpose = false, vnni_transform = false, cache_control = Default}
    %5 = ttig.2d_block_load_from_ptr %4, %pitch {row_major} {base_height = 64 : i32, base_width = 32 : i32, ttig.block_io_stride = 64 : i64} : (tensor<64x16x!tt.ptr<f16>, #blocked>, i32) -> tensor<64x16xf16, #blocked>
    tt.return
  }
}

// -----

// COM: Test that ttig.extract_desc lowers to llvm.extractvalue from the
// COM: descriptor struct at the given index.
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: @extract_desc_fields
  tt.func public @extract_desc_fields(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64) {
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <64x32xf16>
    // CHECK: llvm.extractvalue {{.*}}[0] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    %0 = ttig.extract_desc %desc[0] : <64x32xf16> -> i64
    // CHECK: llvm.extractvalue {{.*}}[2] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    %1 = ttig.extract_desc %desc[2] : <64x32xf16> -> i64
    // CHECK: llvm.extractvalue {{.*}}[4] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    %2 = ttig.extract_desc %desc[4] : <64x32xf16> -> !tt.ptr<f16>
    tt.return
  }
}

// -----

// COM: Test a rank-3 (batched) block load. The 2D surface params describe one
// COM: tile plane, so the step between batch slices must come from the
// COM: batch_strides operand. Re-deriving it as base_height * (base_pitch /
// COM: elem_bytes) reads the wrong slice for any non-densely-packed
// COM: descriptor (issue #7882).
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4, 2], repCluster = [1, 1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: @block_load_batch_rank3
  tt.func public @block_load_batch_rank3(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i64, %arg7: i32, %arg8: i32) {
    // CHECK: %[[BOFF:.*]] = llvm.mul %{{.*}}, %arg6 : i64
    // CHECK: %[[BPTR:.*]] = llvm.getelementptr %{{.*}}{{\[}}%[[BOFF]]{{\]}} : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
    // COM: The batch offset is folded into the base pointer, which re-bases the
    // COM: 2D surface and so escapes the hardware's base_width x base_height
    // COM: clamp. It must be bounds-checked against the descriptor's declared
    // COM: extent instead (issue #7922). Signed compares, because a negative
    // COM: descriptor index is out of bounds and must not wrap.
    // CHECK: %[[IDX:.*]] = llvm.add %arg7, %{{.*}} : i32
    // CHECK-DAG: %[[GE0:.*]] = llvm.icmp "sge" %[[IDX]], %{{.*}} : i32
    // CHECK-DAG: %[[LT:.*]] = llvm.icmp "slt" %[[IDX]], %arg8 : i32
    // CHECK: %[[PRED:.*]] = llvm.and %[[GE0]], %[[LT]]
    // COM: A failing check pushes the Y coordinate past base_height, so the
    // COM: hardware returns the zero padding instead of reading the surface.
    // CHECK: llvm.select %[[PRED]]
    // CHECK: triton_gen.2Dblockload %[[BPTR]]
    %0 = ttig.2d_block_load %arg0, %arg1, %arg2, %arg3[%arg4, %arg5] batch_strides[%arg6] batch_offsets[%arg7] batch_shapes[%arg8] {row_major} : !tt.ptr<f16> -> tensor<2x64x32xf16, #dot0>
    tt.return
  }
}

// -----

// COM: Test a rank-4 block load: each leading dim gets its own stride, applied
// COM: outermost-first as a chain of GEPs. A single shared stride (the #7882
// COM: behaviour) would make the outer batch dim off by the inner dim's extent.
#blocked = #ttg.blocked<{sizePerThread = [1, 1, 8, 1], threadsPerWarp = [1, 1, 1, 16], warpsPerCTA = [2, 2, 2, 1], order = [3, 2, 1, 0]}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: @block_load_batch_rank4
  tt.func public @block_load_batch_rank4(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i64, %arg7: i64, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) {
    // CHECK: %[[B0:.*]] = llvm.mul %{{.*}}, %arg6 : i64
    // CHECK: %[[P0:.*]] = llvm.getelementptr %{{.*}}{{\[}}%[[B0]]{{\]}} : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
    // COM: Each batch dim is bounds-checked against ITS OWN declared extent,
    // COM: interleaved with the GEP chain: dim 0 against %arg10, dim 1 against
    // COM: %arg11. Checking both against one extent would be an indexing bug
    // COM: (issue #7922).
    // CHECK: %[[LT0:.*]] = llvm.icmp "slt" %{{.*}}, %arg10 : i32
    // CHECK: %[[PRED0:.*]] = llvm.and %{{.*}}, %[[LT0]] : i1
    // CHECK: %[[B1:.*]] = llvm.mul %{{.*}}, %arg7 : i64
    // CHECK: %[[P1:.*]] = llvm.getelementptr %[[P0]]{{\[}}%[[B1]]{{\]}} : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
    // CHECK: %[[LT1:.*]] = llvm.icmp "slt" %{{.*}}, %arg11 : i32
    // CHECK: %[[PRED1:.*]] = llvm.and %{{.*}}, %[[LT1]] : i1
    // COM: Both dimensions must gate the load, so their predicates are ANDed.
    // CHECK: %[[PRED:.*]] = llvm.and %[[PRED0]], %[[PRED1]] : i1
    // CHECK: llvm.select %[[PRED]]
    // CHECK: triton_gen.2Dblockload %[[P1]]
    %0 = ttig.2d_block_load %arg0, %arg1, %arg2, %arg3[%arg4, %arg5] batch_strides[%arg6, %arg7] batch_offsets[%arg8, %arg9] batch_shapes[%arg10, %arg11] {row_major} : !tt.ptr<f16> -> tensor<2x2x16x16xf16, #blocked>
    tt.return
  }
}

// -----

// COM: Test a rank-3 column-major (transposed) block load. The batch stride is
// COM: independent of the inner-two-dim memory layout.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4, 2], repCluster = [1, 1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: @block_load_batch_rank3_column_major
  tt.func public @block_load_batch_rank3_column_major(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i64, %arg7: i32, %arg8: i32) {
    // CHECK: %[[BOFF:.*]] = llvm.mul %{{.*}}, %arg6 : i64
    // CHECK: %[[BPTR:.*]] = llvm.getelementptr %{{.*}}{{\[}}%[[BOFF]]{{\]}} : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
    // CHECK: triton_gen.2Dblockload %[[BPTR]]
    // CHECK-SAME: transpose = true
    %0 = ttig.2d_block_load %arg0, %arg1, %arg2, %arg3[%arg4, %arg5] batch_strides[%arg6] batch_offsets[%arg7] batch_shapes[%arg8] {column_major} : !tt.ptr<f16> -> tensor<2x32x64xf16, #dot1>
    tt.return
  }
}

// -----

// COM: Test a MIXED rank-reducing block load: the descriptor is rank 4 but the
// COM: result is rank 3, so descriptor batch dim 0 is dropped while dim 1 is still
// COM: spanned by the result layout. Both indices are folded into the base pointer
// COM: and so escape the hardware surface clamp (issue #7922), but they need
// COM: different checks: the dropped dim is uniform across sub-tiles, the retained
// COM: one is not. This is the only case where `batch_offsets`/`batch_shapes` are
// COM: indexed at a non-zero `rankDelta`, so it is what pins that mapping.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4, 2], repCluster = [1, 1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: @block_load_batch_mixed_rank_reducing
  tt.func public @block_load_batch_mixed_rank_reducing(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i64, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) {
    // COM: Dropped dim 0: no sub-tile offset is added, so its index is compared
    // COM: directly against its own extent.
    // CHECK: llvm.icmp "slt" %arg7, %arg9 : i32
    // COM: Retained dim 1 is offset by the sub-tile and checked against ITS extent.
    // COM: Reading either operand list at the wrong `rankDelta` would compare
    // COM: against %arg7/%arg9 here and silently bound the wrong dimension.
    // CHECK: %[[IDX:.*]] = llvm.add %arg8, %{{.*}} : i32
    // CHECK: llvm.icmp "slt" %[[IDX]], %arg10 : i32
    // CHECK: llvm.select
    // CHECK: triton_gen.2Dblockload
    %0 = ttig.2d_block_load %arg0, %arg1, %arg2, %arg3[%arg4, %arg5] batch_strides[%arg6] batch_offsets[%arg7, %arg8] batch_shapes[%arg9, %arg10] {row_major} : !tt.ptr<f16> -> tensor<2x64x32xf16, #dot0>
    tt.return
  }
}

// -----

// COM: Test subgroup-size=32 DotOp-B block load lowers with VNNI transform.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 32, warpsPerCTA = [2, 2], repCluster = [1, 1], A = [16, 16], B = [16, 16], C = [16, 16]}>
#dot = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: @block_load_dot_b_subgroup32_vnni
  tt.func public @block_load_dot_b_subgroup32_vnni(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
    // CHECK: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 32, v_blocks = 1, transpose = false, vnni_transform = true, cache_control = Default}
    %0 = ttig.2d_block_load %arg0, %arg1, %arg2, %arg3[%arg4, %arg5] {row_major} : !tt.ptr<f16> -> tensor<128x32xf16, #dot>
    tt.return
  }
}

// -----

// COM: Test subgroup-size=32 DotOp-B bf8 block load lowers with VNNI transform.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 4, threadsPerWarp = 32, warpsPerCTA = [2, 2], repCluster = [1, 1], A = [16, 32], B = [32, 16], C = [16, 16]}>
#dot = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 4}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: @block_load_dot_b_subgroup32_vnni_bf8
  tt.func public @block_load_dot_b_subgroup32_vnni_bf8(%arg0: !tt.ptr<f8E5M2>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
    // CHECK: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 8, tile_width = 16, tile_height = 32, v_blocks = 1, transpose = false, vnni_transform = true, cache_control = Default}
    %0 = ttig.2d_block_load %arg0, %arg1, %arg2, %arg3[%arg4, %arg5] {row_major} : !tt.ptr<f8E5M2> -> tensor<128x32xf8E5M2, #dot>
    tt.return
  }
}

// -----

// COM: Test subgroup-size=32 DotOp-B column-major (transposed) block load with
// COM: f16/opsPerChan=2. The old width-comparison check rejected this because
// COM: dpasInstShapeB()[1]=16 != threadsPerWarp=32. The new linear-layout check
// COM: correctly accepts it.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 32, warpsPerCTA = [2, 2], repCluster = [1, 1], A = [16, 16], B = [16, 16], C = [16, 16]}>
#dot = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: @block_load_dot_b_subgroup32_transpose
  tt.func public @block_load_dot_b_subgroup32_transpose(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
    // CHECK: triton_gen.2Dblockload {{.*}} transpose = true
    %0 = ttig.2d_block_load %arg0, %arg1, %arg2, %arg3[%arg4, %arg5] {column_major} : !tt.ptr<f16> -> tensor<32x32xf16, #dot>
    tt.return
  }
}

// -----

// COM: Test pad_nan on a rank-3 (batched) block load. The NaN mask must bound the
// COM: batch dim by the descriptor's declared extent at the descriptor's index,
// COM: not by the tile extent at index 0 (issue #7922). All NaN-mask IR is emitted
// COM: before any address IR, so a compare against the declared extent %arg8 that
// COM: precedes the batch-stride multiply can only come from the mask.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4, 2], repCluster = [1, 1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: @block_load_batch_rank3_pad_nan
  tt.func public @block_load_batch_rank3_pad_nan(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i64, %arg7: i32, %arg8: i32) {
    // COM: Bounding by the tile extent instead would never reference %arg8 here.
    // CHECK: %[[EXT:.*]] = llvm.trunc %arg8
    // CHECK: llvm.icmp "slt" %{{.*}}, %[[EXT]] : i32
    // CHECK: triton_gen.2Dblockload
    %0 = ttig.2d_block_load %arg0, %arg1, %arg2, %arg3[%arg4, %arg5] batch_strides[%arg6] batch_offsets[%arg7] batch_shapes[%arg8] {row_major, pad_nan} : !tt.ptr<f16> -> tensor<2x64x32xf16, #dot0>
    tt.return
  }
}

// -----

// COM: Test pad_nan on a rank-reducing block load. The dropped batch dim has no
// COM: result dimension for the NaN mask to iterate, so its bounds check must be
// COM: ANDed into every mask element instead (issue #7922). Without this, an
// COM: out-of-range batch index yields zeros rather than NaN.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: @block_load_batch_rank_reducing_pad_nan
  tt.func public @block_load_batch_rank_reducing_pad_nan(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
    // CHECK: %[[LT:.*]] = llvm.icmp "slt" %arg6, %arg7 : i32
    // CHECK: %[[DROP:.*]] = llvm.and %{{.*}}, %[[LT]] : i1
    // CHECK: llvm.and %[[DROP]], %{{.*}} : i1
    // CHECK: triton_gen.2Dblockload
    %0 = ttig.2d_block_load %arg0, %arg1, %arg2, %arg3[%arg4, %arg5] batch_offsets[%arg6] batch_shapes[%arg7] {row_major, pad_nan} : !tt.ptr<f16> -> tensor<64x32xf16, #dot0>
    tt.return
  }
}
