// RUN: env TRITON_INTEL_ENABLE_BLOCK_IO_ALL_LAYOUTS=0 triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s --implicit-check-not=llvm.inline_asm --check-prefixes=CHECK,DPAS-LAYOUT
// RUN: env TRITON_INTEL_ENABLE_BLOCK_IO_ALL_LAYOUTS=1 triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s --implicit-check-not=llvm.inline_asm --check-prefixes=CHECK,ALL-LAYOUT

// Test that tt.descriptor_store with DPAS encodings is lowered to
// triton_gen.2Dblockstore when the module has "ttig.support_2d_block_io".
// This mirrors test/TritonIntelGPU/blockptr_store.mlir but uses tensor
// descriptors (tt.make_tensor_descriptor / tt.descriptor_store) instead of
// block pointers (tt.make_tensor_ptr / tt.store).

// Test 1: f16 dpas layout store — descriptor store produces 2D block stores.
// Mirrors blockptr_store.mlir "matmul_no_scf_with_advance_kernel" (f16).

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: llvm.func spir_kernelcc @store_f16_dpas_tdesc
  tt.func public @store_f16_dpas_tdesc(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf16, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <tensor<64x64xf16, #dpas>>
    // 4 warps x 2 warps, repCluster [1,1], C shape [8,16] => 4 stores total.
    // CHECK-COUNT-4: triton_gen.2Dblockstore {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default}
    tt.descriptor_store %desc[%c0_i32, %c0_i32], %cst {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<64x64xf16, #dpas>>, tensor<64x64xf16, #dpas>
    tt.return
  }
}

// -----
// Test: Rank-reducing descriptor store with block IO.
// Descriptor rank is 3 while source tensor rank is 2.

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: llvm.func spir_kernelcc @store_rank_reducing_dpas_tdesc
  // CHECK-COUNT-8: triton_gen.2Dblockstore {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default}
  tt.func public @store_rank_reducing_dpas_tdesc(%arg0: !tt.ptr<f16>, %arg1: i32,
                                                  %arg2: i32, %arg3: i64) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf16, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%c1_i32, %arg1, %arg2], [%arg3, %c1_i64, %c1_i64] : <f16>, <tensor<1x32x32xf16, #dpas>>
    tt.descriptor_store %desc[%c0_i32, %c0_i32, %c0_i32], %cst {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<1x32x32xf16, #dpas>>, tensor<32x32xf16, #dpas>
    tt.return
  }
}

// -----
// Test: Rank-reducing descriptor store without block IO attribute uses generic
// gather/scatter lowering (no 2D block store emitted).

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @store_rank_reducing_generic_tdesc
  // CHECK-NOT: triton_gen.2Dblockstore
  tt.func public @store_rank_reducing_generic_tdesc(%arg0: !tt.ptr<f16>,
                                                     %arg1: i32, %arg2: i32,
                                                     %arg3: i64) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf16, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%c1_i32, %arg1, %arg2], [%arg3, %c1_i64, %c1_i64] : <f16>, <tensor<1x32x32xf16, #dpas>>
    tt.descriptor_store %desc[%c0_i32, %c0_i32, %c0_i32], %cst : !tt.tensordesc<tensor<1x32x32xf16, #dpas>>, tensor<32x32xf16, #dpas>
    tt.return
  }
}

// -----
// Test 2: f32 dpas layout store — f32 element type produces different tile dims.
// Mirrors blockptr_store.mlir f32 variant.

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [8, 4], repCluster = [1, 1], A = [8, 8], B = [8, 16], C = [8, 16]}>
module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: llvm.func spir_kernelcc @store_f32_dpas_tdesc
  tt.func public @store_f32_dpas_tdesc(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: i32, %arg3: i64) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f32>, <tensor<64x64xf32, #dpas>>
    // 8 warps x 4 warps, repCluster [1,1], C shape [8,16] => 32 warps tile 64x64 exactly => 1 store per warp.
    // CHECK-COUNT-1: triton_gen.2Dblockstore {{.*}} {elem_size_in_bits = 32, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default}
    tt.descriptor_store %desc[%c0_i32, %c0_i32], %cst {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<64x64xf32, #dpas>>, tensor<64x64xf32, #dpas>
    tt.return
  }
}

// -----
// Test 3: Detailed dpas layout descriptor store — verifies tensor descriptor
// struct unpacking and 2D block store payload construction.
// Mirrors blockptr_store.mlir "dpas_layout_2d_store_rep_cluster_4_2" but with
// descriptor struct layout: !llvm.struct<(i64, i64, i64, i64, ptr<1>)> instead
// of block pointer layout: !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>.

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
// CHECK-LABEL:   llvm.func spir_kernelcc @dpas_tdesc_2d_store(
// CHECK-SAME:    %[[BASE_PTR:.*]]: !llvm.ptr<1>, %[[SHAPE0_I32:.*]]: i32, %[[SHAPE1_I32:.*]]: i32, %[[STRIDE0:.*]]: i64,
  tt.func public @dpas_tdesc_2d_store(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf16, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    // CHECK:           %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[C1:.*]] = llvm.mlir.constant(1 : i64) : i64

    // Shapes are sign-extended from i32 to i64 for the descriptor struct.
    // CHECK:           %[[SHAPE0:.*]] = llvm.sext %[[SHAPE0_I32]] : i32 to i64
    // CHECK:           %[[SHAPE1:.*]] = llvm.sext %[[SHAPE1_I32]] : i32 to i64

    // Descriptor struct packing: {shape0, shape1, stride0, stride1, base_ptr}
    // CHECK:           %[[UNDEF:.*]] = llvm.mlir.undef : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[V0:.*]] = llvm.insertvalue %[[SHAPE0]], %[[UNDEF]][0] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[V1:.*]] = llvm.insertvalue %[[SHAPE1]], %[[V0]][1] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[V2:.*]] = llvm.insertvalue %[[STRIDE0]], %[[V1]][2] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[V3:.*]] = llvm.insertvalue %[[C1]], %[[V2]][3] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[DESC:.*]] = llvm.insertvalue %[[BASE_PTR]], %[[V3]][4] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>

    // Subgroup ID for offset computation (appears before descriptor extraction).
    // CHECK:           %[[SGI_RAW:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id() {no_unwind, will_return} : () -> i32

    // Extraction of descriptor fields.
    // CHECK:           %[[EX_SHAPE0:.*]] = llvm.extractvalue %[[DESC]][0] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[EX_SHAPE1:.*]] = llvm.extractvalue %[[DESC]][1] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[EX_STRIDE0:.*]] = llvm.extractvalue %[[DESC]][2] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[EX_STRIDE1:.*]] = llvm.extractvalue %[[DESC]][3] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[EX_BASE:.*]] = llvm.extractvalue %[[DESC]][4] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>

    // Compute baseWidth (shape1 in bytes): mul in i64 then trunc to i32.
    // CHECK:           %[[ELEM_BYTES_W:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK:           %[[WIDTH_I64:.*]] = llvm.mul %[[EX_SHAPE1]], %[[ELEM_BYTES_W]] : i64
    // CHECK:           %[[BASE_WIDTH:.*]] = llvm.trunc %[[WIDTH_I64]] : i64 to i32

    // Compute baseHeight: trunc shape0 from i64 to i32.
    // CHECK:           %[[BASE_HEIGHT:.*]] = llvm.trunc %[[EX_SHAPE0]] : i64 to i32

    // Compute pitch (stride in bytes): mul in i64 then trunc to i32.
    // CHECK:           %[[ELEM_BYTES_P:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK:           %[[PITCH_I64:.*]] = llvm.mul %[[EX_STRIDE0]], %[[ELEM_BYTES_P]] : i64
    // CHECK:           %[[PITCH:.*]] = llvm.trunc %[[PITCH_I64]] : i64 to i32

    // Offsets from indices (always %c0_i32 here) + linear layout warp offsets.
    // CHECK:           %[[OFF_COL:.*]] = llvm.add %[[C0]], {{.*}} : i32
    // CHECK:           %[[OFF_ROW:.*]] = llvm.add %[[C0]], {{.*}} : i32

    // Compose value vector and insert elements.
    // CHECK:           llvm.mlir.undef : vector<8xf16>
    // CHECK-COUNT-8:   llvm.insertelement %{{[0-9]+}}, %{{[0-9]+}}{{\[}}{{.*}} : i32] : vector<8xf16>

    // The 2D block store: base ptr from descriptor, computed width/height/pitch.
    // CHECK:           triton_gen.2Dblockstore %[[EX_BASE]], %[[BASE_WIDTH]], %[[BASE_HEIGHT]], %[[PITCH]], {{.*}}, %[[OFF_ROW]], {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi16>)

    // Remaining 7 stores (repCluster [4,2] => 8 total stores).
    // CHECK-COUNT-7: triton_gen.2Dblockstore {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default}
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <tensor<32x32xf16, #dpas>>
    tt.descriptor_store %desc[%c0_i32, %c0_i32], %cst {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<32x32xf16, #dpas>>, tensor<32x32xf16, #dpas>
    tt.return
  }
}

// -----
// Test 4: Blocked layout with support_2d_block_io.
// With TRITON_INTEL_ENABLE_BLOCK_IO_ALL_LAYOUTS=0, blocked layout should use
// gather path and avoid triton_gen.2Dblockstore.
// With TRITON_INTEL_ENABLE_BLOCK_IO_ALL_LAYOUTS=1, blocked layout is allowed
// and lowers to triton_gen.2Dblockstore.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @blocked_layout_store_tdesc
  tt.func public @blocked_layout_store_tdesc(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64) {
      %cst = arith.constant dense<0.000000e+00> : tensor<64x16xf16, #blocked>
      %c0_i32 = arith.constant 0 : i32
      %c1_i64 = arith.constant 1 : i64
      %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <tensor<64x16xf16, #blocked>>
      // DPAS-LAYOUT-NOT: triton_gen.2Dblockstore
      // ALL-LAYOUT-COUNT-32: triton_gen.2Dblockstore {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 1, v_blocks = 1, cache_control = Default}
      tt.descriptor_store %desc[%c0_i32, %c0_i32], %cst {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<64x16xf16, #blocked>>, tensor<64x16xf16, #blocked>
      tt.return
  }
}

// -----
// Test 5: Negative — dpas encoding WITHOUT support_2d_block_io module attr.
// Without the module attribute, the block IO conversion should not fire.

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @no_support_2d_block_io_store_tdesc
  tt.func public @no_support_2d_block_io_store_tdesc(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64) {
      %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf16, #dpas>
      %c0_i32 = arith.constant 0 : i32
      %c1_i64 = arith.constant 1 : i64
      %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <tensor<32x32xf16, #dpas>>
      // CHECK-NOT:    triton_gen.2Dblockstore
      tt.descriptor_store %desc[%c0_i32, %c0_i32], %cst : !tt.tensordesc<tensor<32x32xf16, #dpas>>, tensor<32x32xf16, #dpas>
      tt.return
  }
}

// -----
// Test 6: Negative — non-contiguous inner stride (stride[1] != 1) should
// prevent 2D block store generation.

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @non_unit_inner_stride_store_tdesc
  // CHECK-NOT:     triton_gen.2Dblockstore
  tt.func public @non_unit_inner_stride_store_tdesc(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64) {
      %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf16, #dpas>
      %c0_i32 = arith.constant 0 : i32
      %c2_i64 = arith.constant 2 : i64
      %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c2_i64] : <f16>, <tensor<64x64xf16, #dpas>>
      tt.descriptor_store %desc[%c0_i32, %c0_i32], %cst : !tt.tensordesc<tensor<64x64xf16, #dpas>>, tensor<64x64xf16, #dpas>
      tt.return
  }
}

// -----
// Test 7: Full matmul integration — descriptor loads feed tt.dot, result
// stored via descriptor store producing 2D block stores.

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @matmul_descriptor_store
  tt.func public @matmul_descriptor_store(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i64, %arg7: i64, %arg8: i64) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %C = arith.constant dense<0.000000e+00> : tensor<64x64xf16, #dpas>
    %descA = tt.make_tensor_descriptor %arg0, [%arg3, %arg5], [%arg6, %c1_i64] : <f16>, <tensor<64x32xf16>>
    %descB = tt.make_tensor_descriptor %arg1, [%arg5, %arg4], [%arg7, %c1_i64] : <f16>, <tensor<32x64xf16>>
    %descD = tt.make_tensor_descriptor %arg2, [%arg3, %arg4], [%arg8, %c1_i64] : <f16>, <tensor<64x64xf16, #dpas>>
    // Loads for A operand: 4 warps along M, repCluster [1,1], A tile [8,16] => 2 loads.
    // CHECK-COUNT-2: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 2, transpose = false, vnni_transform = false, cache_control = Default}
    %A = tt.descriptor_load %descA[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<64x32xf16>> -> tensor<64x32xf16, #dot0>
    // Loads for B operand: 2 warps along N, repCluster [1,1], B tile [16,16] => 2 loads with VNNI.
    // CHECK-COUNT-2: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 32, v_blocks = 1, transpose = false, vnni_transform = true, cache_control = Default}
    %B = tt.descriptor_load %descB[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<32x64xf16>> -> tensor<32x64xf16, #dot1>
    // DPAS instructions: 8 total.
    // CHECK-COUNT-8: triton_gen.dpas {{.*}} {pa = f16, pb = f16, rc = 8}
    %D = tt.dot %A, %B, %C, inputPrecision = tf32 : tensor<64x32xf16, #dot0> * tensor<32x64xf16, #dot1> -> tensor<64x64xf16, #dpas>
    // Stores for result: 4 warps x 2 warps, repCluster [1,1], C shape [8,16] => 4 stores.
    // CHECK-COUNT-4: triton_gen.2Dblockstore {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default}
    tt.descriptor_store %descD[%c0_i32, %c0_i32], %D {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<64x64xf16, #dpas>>, tensor<64x64xf16, #dpas>
    tt.return
  }
}
