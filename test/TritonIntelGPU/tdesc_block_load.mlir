// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s --implicit-check-not=llvm.inline_asm

// Test that tt.descriptor_load with DPAS/dot_op encodings is lowered to
// triton_gen.2Dblockload when the module has "ttig.support_2d_block_io".
// This mirrors test/TritonIntelGPU/blockptr_load.mlir but uses tensor
// descriptors (tt.make_tensor_descriptor / tt.descriptor_load) instead of
// block pointers (tt.make_tensor_ptr / tt.load).

// Test 1: f16 matmul — dot_op A + B descriptor loads produce 2D block loads.
// Mirrors blockptr_load.mlir "matmul_no_scf_with_advance_kernel" (f16).

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: llvm.func spir_kernelcc @matmul_f16_tdesc
  tt.func public @matmul_f16_tdesc(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i64, %arg6: i64) {
    %C = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %descA = tt.make_tensor_descriptor %arg0, [%arg2, %arg4], [%arg5, %c1_i64] : <f16>, <tensor<64x32xf16, #dot0>>
    %descB = tt.make_tensor_descriptor %arg1, [%arg4, %arg3], [%arg6, %c1_i64] : <f16>, <tensor<32x64xf16, #dot1>>
    // Operand A: 2 loads, each 8 rows x 16 cols x 2 v_blocks, no VNNI.
    // CHECK-COUNT-2: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 2, transpose = false, vnni_transform = false, cache_control = Default}
    // Operand B: 2 loads, 32 rows x 16 cols, VNNI transform for opsPerChan=2.
    // CHECK-COUNT-2: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 32, v_blocks = 1, transpose = false, vnni_transform = true, cache_control = Default}
    // CHECK-COUNT-8: triton_gen.dpas {{.*}} {pa = f16, pb = f16, rc = 8}
    %A = tt.descriptor_load %descA[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<64x32xf16, #dot0>> -> tensor<64x32xf16, #dot0>
    %B = tt.descriptor_load %descB[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<32x64xf16, #dot1>> -> tensor<32x64xf16, #dot1>
    %D = tt.dot %A, %B, %C, inputPrecision = tf32 : tensor<64x32xf16, #dot0> * tensor<32x64xf16, #dot1> -> tensor<64x64xf32, #dpas>
    %0 = ttg.convert_layout %D : tensor<64x64xf32, #dpas> -> tensor<64x64xf32, #blocked>
    tt.return
  }
}

// -----
// Test 2: f32 matmul — dot_op A + B descriptor loads with opsPerChan=1.
// Mirrors blockptr_load.mlir "matmul_no_scf_with_advance_kernel" (f32).

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [8, 4], repCluster = [1, 1], A = [8, 8], B = [8, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=1}>
module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: llvm.func spir_kernelcc @matmul_f32_tdesc
  tt.func public @matmul_f32_tdesc(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i64, %arg6: i64) {
    %C = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %descA = tt.make_tensor_descriptor %arg0, [%arg2, %arg4], [%arg5, %c1_i64] : <f32>, <tensor<64x32xf32, #dot0>>
    %descB = tt.make_tensor_descriptor %arg1, [%arg4, %arg3], [%arg6, %c1_i64] : <f32>, <tensor<32x64xf32, #dot1>>
    // Operand A: 2 loads of 8x8 tiles with 2 v_blocks, no VNNI (f32, opsPerChan=1).
    // CHECK-COUNT-2: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 32, tile_width = 8, tile_height = 8, v_blocks = 2, transpose = false, vnni_transform = false, cache_control = Default}
    // Operand B: 1 load of 32x16 tile, no VNNI (f32, opsPerChan=1).
    // CHECK-COUNT-1: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 32, tile_width = 16, tile_height = 32, v_blocks = 1, transpose = false, vnni_transform = false, cache_control = Default}
    // CHECK-COUNT-4: triton_gen.dpas {{.*}} {pa = tf32, pb = tf32, rc = 8}
    %A = tt.descriptor_load %descA[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<64x32xf32, #dot0>> -> tensor<64x32xf32, #dot0>
    %B = tt.descriptor_load %descB[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<32x64xf32, #dot1>> -> tensor<32x64xf32, #dot1>
    %D = tt.dot %A, %B, %C, inputPrecision = tf32 : tensor<64x32xf32, #dot0> * tensor<32x64xf32, #dot1> -> tensor<64x64xf32, #dpas>
    tt.return
  }
}

// -----
// Test 3: Detailed dot_op A descriptor load — verifies tensor descriptor
// struct unpacking and 2D block load payload construction.
// Mirrors blockptr_load.mlir "dot_op_a_2d_load" but with descriptor struct
// layout: !llvm.struct<(i64, i64, i64, i64, ptr<1>)> instead of block pointer
// layout: !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>.

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
// CHECK-LABEL:   llvm.func spir_kernelcc @dot_op_a_tdesc_2d_load(
// CHECK-SAME:    %[[BASE_PTR:.*]]: !llvm.ptr<1>, %[[SHAPE0_I32:.*]]: i32, %[[SHAPE1_I32:.*]]: i32, %[[STRIDE0:.*]]: i64,
  tt.func public @dot_op_a_tdesc_2d_load(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64) {
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

    // Subgroup ID for offset computation.
    // CHECK:           %[[SGI_RAW:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id() {no_unwind, will_return} : () -> i32

    // Offsets from indices (always %c0_i32 here) + linear layout warp offsets.
    // CHECK:           %[[OFF_COL:.*]] = llvm.add %[[C0]], {{.*}} : i32
    // CHECK:           %[[OFF_ROW:.*]] = llvm.add %[[C0]], {{.*}} : i32
    // CHECK:           %[[PACK:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:           %[[OFF_X:.*]] = llvm.udiv %[[OFF_COL]], %[[PACK]] : i32

    // The 2D block load: base ptr from descriptor, computed width/height/pitch.
    // CHECK:           triton_gen.2Dblockload %[[EX_BASE]], %[[BASE_WIDTH]], %[[BASE_HEIGHT]], %[[PITCH]], %[[OFF_X]], %[[OFF_ROW]] {elem_size_in_bits = 16, tile_width = 16, tile_height = 32, v_blocks = 2, transpose = false, vnni_transform = false, cache_control = Default}
    %descA = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <tensor<32x32xf16, #dot0>>
    %A = tt.descriptor_load %descA[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<32x32xf16, #dot0>> -> tensor<32x32xf16, #dot0>
    %B = arith.constant dense<0.000000e+00> : tensor<32x32xf16, #dot1>
    %C = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #dpas>
    %D = tt.dot %A, %B, %C, inputPrecision = tf32 : tensor<32x32xf16, #dot0> * tensor<32x32xf16, #dot1> -> tensor<32x32xf32, #dpas>
    tt.return
  }
}

// -----
// Test 4: Detailed dot_op B descriptor load — verifies VNNI transform is
// applied for operand B with opsPerChan=2. Descriptor struct is 5-field
// (i64, i64, i64, i64, ptr<1>) unlike block pointer's 7-field struct.
// Mirrors blockptr_load.mlir "dot_op_b_2d_load".

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
// CHECK-LABEL:   llvm.func spir_kernelcc @dot_op_b_tdesc_2d_load(
// CHECK-SAME:    %[[BASE_PTR:.*]]: !llvm.ptr<1>, %[[SHAPE0_I32:.*]]: i32, %[[SHAPE1_I32:.*]]: i32, %[[STRIDE0:.*]]: i64,
  tt.func public @dot_op_b_tdesc_2d_load(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    // CHECK:           %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[C1:.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK:           %[[SHAPE0:.*]] = llvm.sext %[[SHAPE0_I32]] : i32 to i64
    // CHECK:           %[[SHAPE1:.*]] = llvm.sext %[[SHAPE1_I32]] : i32 to i64

    // Descriptor struct: 5 fields (shapes, strides, base) — not 7 like block pointers.
    // CHECK:           %[[UNDEF:.*]] = llvm.mlir.undef : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[V0:.*]] = llvm.insertvalue %[[SHAPE0]], %[[UNDEF]][0] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[V1:.*]] = llvm.insertvalue %[[SHAPE1]], %[[V0]][1] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[V2:.*]] = llvm.insertvalue %[[STRIDE0]], %[[V1]][2] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[V3:.*]] = llvm.insertvalue %[[C1]], %[[V2]][3] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[DESC:.*]] = llvm.insertvalue %[[BASE_PTR]], %[[V3]][4] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>

    // Extraction and 2D block load setup.
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

    // Subgroup ID.
    // CHECK:           %[[SGI_RAW:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id() {no_unwind, will_return} : () -> i32

    // Offsets from indices + warp offset.
    // CHECK:           %[[OFF_COL:.*]] = llvm.add %[[C0]], {{.*}} : i32
    // CHECK:           {{.*}} = llvm.add %[[C0]], {{.*}} : i32
    // CHECK:           %[[PACK:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:           %[[OFF_X:.*]] = llvm.udiv %[[OFF_COL]], %[[PACK]] : i32

    // VNNI transform is true for operand B with opsPerChan=2, f16.
    // CHECK:           triton_gen.2Dblockload %[[EX_BASE]], %[[BASE_WIDTH]], %[[BASE_HEIGHT]], %[[PITCH]], %[[OFF_X]], {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 32, v_blocks = 2, transpose = false, vnni_transform = true, cache_control = Default}
    %descB = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <tensor<32x32xf16, #dot1>>
    %B = tt.descriptor_load %descB[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<32x32xf16, #dot1>> -> tensor<32x32xf16, #dot1>
    %A = arith.constant dense<0.000000e+00> : tensor<32x32xf16, #dot0>
    %C = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #dpas>
    %D = tt.dot %A, %B, %C, inputPrecision = tf32 : tensor<32x32xf16, #dot0> * tensor<32x32xf16, #dot1> -> tensor<32x32xf32, #dpas>
    tt.return
  }
}

// -----
// Test 5: dpas C operand load — accumulator loaded via descriptor then added.
// Mirrors blockptr_load.mlir "matmul_no_scf_with_add_kernel".

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: llvm.func spir_kernelcc @matmul_f16_add_tdesc
  tt.func public @matmul_f16_add_tdesc(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f32>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i64, %arg7: i64, %arg8: i64) {
    %C = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %descA = tt.make_tensor_descriptor %arg0, [%arg3, %arg5], [%arg6, %c1_i64] : <f16>, <tensor<64x32xf16, #dot0>>
    %descB = tt.make_tensor_descriptor %arg1, [%arg5, %arg4], [%arg7, %c1_i64] : <f16>, <tensor<32x64xf16, #dot1>>
    // Operand A + B loads.
    // CHECK-COUNT-2: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 2, transpose = false, vnni_transform = false, cache_control = Default}
    // CHECK-COUNT-2: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 32, v_blocks = 1, transpose = false, vnni_transform = true, cache_control = Default}
    // CHECK-COUNT-8: triton_gen.dpas {{.*}} {pa = f16, pb = f16, rc = 8}
    %A = tt.descriptor_load %descA[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<64x32xf16, #dot0>> -> tensor<64x32xf16, #dot0>
    %B = tt.descriptor_load %descB[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<32x64xf16, #dot1>> -> tensor<32x64xf16, #dot1>
    %D = tt.dot %A, %B, %C, inputPrecision = tf32 : tensor<64x32xf16, #dot0> * tensor<32x64xf16, #dot1> -> tensor<64x64xf32, #dpas>
    // Accumulator load: dpas C operand uses f32, tile 8x16, no VNNI.
    %descX = tt.make_tensor_descriptor %arg2, [%arg3, %arg4], [%arg8, %c1_i64] : <f32>, <tensor<64x64xf32, #dpas>>
    // CHECK-COUNT-4: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 32, tile_width = 16, tile_height = 8, v_blocks = 1, transpose = false, vnni_transform = false, cache_control = Default}
    %X = tt.descriptor_load %descX[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<64x64xf32, #dpas>> -> tensor<64x64xf32, #dpas>
    // CHECK-COUNT-32: llvm.fadd {{.*}}, {{.*}}
    %0 = arith.addf %D, %X : tensor<64x64xf32, #dpas>
    tt.return
  }
}

// -----
// Test 6: i8 dot_op B with opsPerChan=4 — VNNI transform + shufflevector
// unpacking. Mirrors blockptr_load.mlir "dot_op_b_2d_load" (i8).

#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 4, threadsPerWarp = 16, warpsPerCTA = [8, 4], repCluster = [4, 4]}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>
module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @dot_op_b_i8_tdesc_2d_load
  tt.func public @dot_op_b_i8_tdesc_2d_load(%arg0: !tt.ptr<i8>, %arg1: i32, %arg2: i32, %arg3: i64) {
      %c0_i32 = arith.constant 0 : i32
      %c1_i64 = arith.constant 1 : i64
      %descB = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <i8>, <tensor<32x256xi8, #dot_b>>
      // CHECK: %[[LOAD:.*]] = triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 8, tile_width = 16, tile_height = 32, v_blocks = 4, transpose = false, vnni_transform = true, cache_control = Default}
      // CHECK:    %[[VAL_0:.*]] = llvm.shufflevector %[[LOAD]], %[[LOAD]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<32xi32>
      // CHECK:    llvm.bitcast %[[VAL_0]] : vector<8xi32> to vector<32xi8>
      // CHECK:    %[[VAL_1:.*]] = llvm.shufflevector %[[LOAD]], %[[LOAD]] [8, 9, 10, 11, 12, 13, 14, 15] : vector<32xi32>
      // CHECK:    llvm.bitcast %[[VAL_1]] : vector<8xi32> to vector<32xi8>
      // CHECK:    %[[VAL_2:.*]] = llvm.shufflevector %[[LOAD]], %[[LOAD]] [16, 17, 18, 19, 20, 21, 22, 23] : vector<32xi32>
      // CHECK:    llvm.bitcast %[[VAL_2]] : vector<8xi32> to vector<32xi8>
      // CHECK:    %[[VAL_3:.*]] = llvm.shufflevector %[[LOAD]], %[[LOAD]] [24, 25, 26, 27, 28, 29, 30, 31] : vector<32xi32>
      // CHECK:    llvm.bitcast %[[VAL_3]] : vector<8xi32> to vector<32xi8>
      %B = tt.descriptor_load %descB[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<32x256xi8, #dot_b>> -> tensor<32x256xi8, #dot_b>
      tt.return
  }
}

// -----
// Test 7: Negative — blocked layout with support_2d_block_io.
// Block IO requires DPAS/dot_op encoding; blocked layout should NOT produce
// triton_gen.2Dblockload.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @blocked_layout_tdesc
  tt.func public @blocked_layout_tdesc(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64) {
      %c0_i32 = arith.constant 0 : i32
      %c1_i64 = arith.constant 1 : i64
      %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <tensor<64x16xf16, #blocked>>
      // CHECK-NOT:    triton_gen.2Dblockload
      %val = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<64x16xf16, #blocked>> -> tensor<64x16xf16, #blocked>
      tt.return
  }
}

// -----
// Test 8: Negative — dot_op encoding WITHOUT support_2d_block_io module attr.
// Without the module attribute, the block IO conversion should not fire even
// for DPAS-encoded tensor descriptors.

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @no_support_2d_block_io_tdesc
  tt.func public @no_support_2d_block_io_tdesc(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64) {
      %c0_i32 = arith.constant 0 : i32
      %c1_i64 = arith.constant 1 : i64
      %descA = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <tensor<32x32xf16, #dot0>>
      // CHECK-NOT:    triton_gen.2Dblockload
      %A = tt.descriptor_load %descA[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<32x32xf16, #dot0>> -> tensor<32x32xf16, #dot0>
      %B = arith.constant dense<0.000000e+00> : tensor<32x32xf16, #dot1>
      %C = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #dpas>
      %D = tt.dot %A, %B, %C, inputPrecision = tf32 : tensor<32x32xf16, #dot0> * tensor<32x32xf16, #dot1> -> tensor<32x32xf32, #dpas>
      tt.return
  }
}
