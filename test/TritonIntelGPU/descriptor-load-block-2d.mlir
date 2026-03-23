// RUN: TRITON_INTEL_ENABLE_BLOCK_IO_ALL_LAYOUTS=0 triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s --check-prefixes=CHECK,DPAS-LAYOUT
// RUN: TRITON_INTEL_ENABLE_BLOCK_IO_ALL_LAYOUTS=1 triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s --check-prefixes=CHECK,ALL-LAYOUT

// Test: MakeTensorDescOp lowers to LLVM struct packing and
// DescriptorLoadOp with dot_op A encoding generates 2D block loads.

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: @descriptor_load_dot_a
  tt.func public @descriptor_load_dot_a(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i32, %arg5: i32) {
    %c1_i64 = arith.constant 1 : i64
    // Verify struct packing for MakeTensorDescOp:
    //   struct { i64 shape[2], i64 stride[2], ptr<1> base }
    // CHECK: llvm.insertvalue {{.*}}[0] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK: llvm.insertvalue {{.*}}[1] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK: llvm.insertvalue {{.*}}[2] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK: llvm.insertvalue {{.*}}[3] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK: llvm.insertvalue {{.*}}[4] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <tensor<64x32xf16>>
    // Verify 2D block loads are generated for dot A operand.
    // For DPAS A with f16: tile_height=8, tile_width=16, v_blocks=2.
    // CHECK-COUNT-2: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 2, transpose = false, vnni_transform = false, cache_control = Default}
    %load = tt.descriptor_load %desc[%arg4, %arg5] {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<64x32xf16>> -> tensor<64x32xf16, #dot0>
    tt.return
  }
}

// -----

// Test: DescriptorLoadOp with dot_op B encoding generates 2D block loads
// with VNNI transform.

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: @descriptor_load_dot_b_vnni
  tt.func public @descriptor_load_dot_b_vnni(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i32, %arg5: i32) {
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <tensor<32x64xf16>>
    // For DPAS B with f16: tile_height=32, tile_width=16, v_blocks=1, vnni_transform=true.
    // CHECK-COUNT-2: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 32, v_blocks = 1, transpose = false, vnni_transform = true, cache_control = Default}
    %load = tt.descriptor_load %desc[%arg4, %arg5] {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<32x64xf16>> -> tensor<32x64xf16, #dot1>
    tt.return
  }
}

// -----

// Test: DescriptorLoadOp with f32 element type and dot_op A encoding.

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [8, 4], repCluster = [1, 1], A = [8, 8], B = [8, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: @descriptor_load_f32_dot_a
  tt.func public @descriptor_load_f32_dot_a(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i32, %arg5: i32) {
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f32>, <tensor<64x32xf32>>
    // For DPAS A with f32: tile_height=8, tile_width=8, v_blocks=2.
    // CHECK-COUNT-2: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 32, tile_width = 8, tile_height = 8, v_blocks = 2, transpose = false, vnni_transform = false, cache_control = Default}
    %load = tt.descriptor_load %desc[%arg4, %arg5] {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<64x32xf32>> -> tensor<64x32xf32, #dot0>
    tt.return
  }
}

// -----

// Negative test: Missing ttig.support_2d_block_io attribute should prevent
// 2D block load generation, falling back to scalar gather loads.

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @descriptor_load_no_2d_block_io_attr
  // CHECK-NOT: triton_gen.2Dblockload
  tt.func public @descriptor_load_no_2d_block_io_attr(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i32, %arg5: i32) {
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <tensor<64x32xf16>>
    %load = tt.descriptor_load %desc[%arg4, %arg5] : !tt.tensordesc<tensor<64x32xf16>> -> tensor<64x32xf16, #dot0>
    tt.return
  }
}

// -----

// Negative test: Blocked encoding (non-DPAS) should prevent 2D block load
// generation, falling back to scalar gather loads.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: @descriptor_load_blocked_no_block_io
  // DPAS-LAYOUT-NOT: triton_gen.2Dblockload
  // ALL-LAYOUT-COUNT-32: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 1, v_blocks = 1, transpose = false, vnni_transform = false, cache_control = Default}
  tt.func public @descriptor_load_blocked_no_block_io(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i32, %arg5: i32) {
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <tensor<64x32xf16>>
    %load = tt.descriptor_load %desc[%arg4, %arg5] {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<64x32xf16>> -> tensor<64x32xf16, #blocked>
    tt.return
  }
}

// -----

// Negative test: Non-contiguous inner stride (stride[1] != 1) should prevent
// 2D block load generation, falling back to scalar gather loads.

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: @descriptor_load_non_unit_inner_stride
  // CHECK-NOT: triton_gen.2Dblockload
  tt.func public @descriptor_load_non_unit_inner_stride(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i32, %arg5: i32) {
    %c2_i64 = arith.constant 2 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c2_i64] : <f16>, <tensor<64x32xf16>>
    %load = tt.descriptor_load %desc[%arg4, %arg5] : !tt.tensordesc<tensor<64x32xf16>> -> tensor<64x32xf16, #dot0>
    tt.return
  }
}

// -----

// Negative test: Dynamic (non-constant) inner stride should prevent
// 2D block load generation, falling back to scalar gather loads.

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: @descriptor_load_dynamic_inner_stride
  // CHECK-NOT: triton_gen.2Dblockload
  tt.func public @descriptor_load_dynamic_inner_stride(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: i32, %arg6: i32) {
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %arg4] : <f16>, <tensor<64x32xf16>>
    %load = tt.descriptor_load %desc[%arg5, %arg6] : !tt.tensordesc<tensor<64x32xf16>> -> tensor<64x32xf16, #dot0>
    tt.return
  }
}

// -----

// Test: DescriptorLoadOp feeds into tt.dot, verifying the full matmul pattern.

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: @descriptor_load_matmul
  tt.func public @descriptor_load_matmul(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i64, %arg6: i64) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %C = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #dpas>
    %descA = tt.make_tensor_descriptor %arg0, [%arg2, %arg4], [%arg5, %c1_i64] : <f16>, <tensor<64x32xf16>>
    %descB = tt.make_tensor_descriptor %arg1, [%arg4, %arg3], [%arg6, %c1_i64] : <f16>, <tensor<32x64xf16>>
    // CHECK-COUNT-2: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 2, transpose = false, vnni_transform = false, cache_control = Default}
    %A = tt.descriptor_load %descA[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<64x32xf16>> -> tensor<64x32xf16, #dot0>
    // CHECK-COUNT-2: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 32, v_blocks = 1, transpose = false, vnni_transform = true, cache_control = Default}
    %B = tt.descriptor_load %descB[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<32x64xf16>> -> tensor<32x64xf16, #dot1>
    // CHECK-COUNT-8: triton_gen.dpas {{.*}} {pa = f16, pb = f16, rc = 8}
    %D = tt.dot %A, %B, %C, inputPrecision = tf32 : tensor<64x32xf16, #dot0> * tensor<32x64xf16, #dot1> -> tensor<64x64xf32, #dpas>
    %0 = ttg.convert_layout %D : tensor<64x64xf32, #dpas> -> tensor<64x64xf32, #blocked>
    tt.return
  }
}

// -----

// Test: DescriptorLoadOp with column_major block_io attribute (result of
// FuseTransWithDescriptorLoad) generates 2D block loads with transpose=true.
// The descriptor is always row-major (stride-1 on last dim). The result type
// dimensions are transposed relative to the descriptor's block shape.

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: @descriptor_load_column_major
  tt.func public @descriptor_load_column_major(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i32, %arg5: i32) {
    %c1_i64 = arith.constant 1 : i64
    // Row-major descriptor: shape [N, K], strides [K_stride, 1], block <64x32>.
    // The load produces a transposed result tensor<32x64, dot1>.
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <tensor<64x32xf16>>
    // For DPAS B with column_major f16: should generate transposed 2D block loads.
    // CHECK-COUNT-2: triton_gen.2Dblockload {{.*}} transpose = true, vnni_transform = false
    %load = tt.descriptor_load %desc[%arg4, %arg5] {ttig.block_io = "column_major"} : !tt.tensordesc<tensor<64x32xf16>> -> tensor<32x64xf16, #dot1>
    tt.return
  }
}

// -----

// Test: DescriptorLoadOp with row_major block_io attribute and dot_op B encoding
// generates standard (non-transposed) 2D block loads with VNNI transform.

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: @descriptor_load_explicit_row_major
  tt.func public @descriptor_load_explicit_row_major(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i32, %arg5: i32) {
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <tensor<32x64xf16>>
    // CHECK-COUNT-2: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 32, v_blocks = 1, transpose = false, vnni_transform = true, cache_control = Default}
    %load = tt.descriptor_load %desc[%arg4, %arg5] {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<32x64xf16>> -> tensor<32x64xf16, #dot1>
    tt.return
  }
}

// -----

// COM: Test that when a column_major tt.descriptor_load is present but descriptor is not traceable to
// COM: a MakeTensorDescOp (passed as function argument) the code generated uses the "gather path".
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: @descriptor_load_column_major_fallback
  tt.func public @descriptor_load_column_major_fallback(%desc: !tt.tensordesc<tensor<64x32xf16>>, %row: i32, %col: i32) {
    // CHECK-NOT: triton_gen.2Dblockload
    // CHECK: llvm.getelementptr
    %load = tt.descriptor_load %desc[%row, %col] {ttig.block_io = "column_major"} : !tt.tensordesc<tensor<64x32xf16>> -> tensor<32x64xf16, #dot1>
    tt.return
  }
}

// -----

// COM: The blocked layout with threadsPerWarp = [8, 2, 1], order = [1, 2, 0] defines a tile on dimensions 0 and 1.
// COM: However, only dimension 2 is contiguous in memory, so a transposed block load is not legal for this layout.
#blocked = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [8, 2, 1], warpsPerCTA = [64, 1, 1], order = [1, 2, 0]}>
module attributes {"ttg.num-warps" = 64 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: llvm.func spir_kernelcc @kernel_trans(
  // CHECK: llvm.load
  // CHECK-NOT: triton_gen.2Dblockload
  tt.func public @kernel_trans(%arg0: !tt.tensordesc<tensor<32x2x16xf32>>, %idx: i32) -> tensor<32x2x16xf32, #blocked> {
    %0 = tt.descriptor_load %arg0[%idx, %idx, %idx] {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<32x2x16xf32>> -> tensor<32x2x16xf32, #blocked>
    tt.return %0 : tensor<32x2x16xf32, #blocked>
  }
}

// -----

// COM: Rank-3 descriptor load with column_major block_io attribute (produced by
// COM: FuseTransWithDescriptorLoad for rank-3 tensors). The descriptor has shape
// COM: [2, 64, 32] and the result has inner dims transposed: [2, 32, 64].
// COM: Since 2D block IO only supports rank-2 tensors, the lowering falls back
// COM: to the gather path using llvm.getelementptr.
#blocked3d = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 16], warpsPerCTA = [1, 8, 1], order = [2, 1, 0]}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: llvm.func spir_kernelcc @descriptor_load_rank3_column_major(
  // CHECK-NOT: triton_gen.2Dblockload
  // CHECK: llvm.getelementptr
  tt.func public @descriptor_load_rank3_column_major(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i64, %arg5: i64, %arg6: i32, %arg7: i32, %arg8: i32) {
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2, %arg3], [%arg4, %arg5, %c1_i64] : <f16>, <tensor<2x64x32xf16>>
    %load = tt.descriptor_load %desc[%arg6, %arg7, %arg8] {ttig.block_io = "column_major"} : !tt.tensordesc<tensor<2x64x32xf16>> -> tensor<2x32x64xf16, #blocked3d>
    tt.return
  }
}
