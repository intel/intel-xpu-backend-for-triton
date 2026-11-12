// RUN: triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm | FileCheck %s

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @prefetch_tensor_of_pointers
  tt.func public @prefetch_tensor_of_pointers(%arg0: !tt.ptr<f16>) {
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>}>> -> tensor<64x1xi32, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    %2 = arith.constant dense<64> : tensor<64x1xi32, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    %3 = arith.muli %1, %2 : tensor<64x1xi32, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    %4 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>}>> -> tensor<1x32xi32, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    %6 = tt.broadcast %3 : tensor<64x1xi32, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>> -> tensor<64x32xi32, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    %7 = tt.broadcast %5 : tensor<1x32xi32, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>> -> tensor<64x32xi32, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    %8 = arith.addi %6, %7 : tensor<64x32xi32, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    %9 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x32x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    %tensor_of_ptr = tt.addptr %9, %8 : tensor<64x32x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>, tensor<64x32xi32, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>

    // CHECK: %[[ADDR_0:.*]] = llvm.extractvalue {{.*}}[0] : !llvm.struct<(ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>)>
    // CHECK: %[[ADDR_0:.*]] = llvm.extractvalue {{.*}}[0] : !llvm.struct<(ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>)>
    // CHECK: %[[ADDR_1:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>)>
    // CHECK: %[[ADDR_16:.*]] = llvm.extractvalue {{.*}}[16] : !llvm.struct<(ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>)>
    // CHECK: %[[ADDR_32:.*]] = llvm.extractvalue {{.*}}[32] : !llvm.struct<(ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>)>
    // CHECK: %[[ADDR_48:.*]] = llvm.extractvalue {{.*}}[48] : !llvm.struct<(ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>)>
    // CHECK: %[[MASK_0:.*]] = llvm.extractvalue {{.*}}[0] : !llvm.struct<(i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1)>
    // CHECK: %[[MASK_16:.*]] = llvm.extractvalue {{.*}}[16] : !llvm.struct<(i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1)>
    // CHECK: %[[MASK_32:.*]] = llvm.extractvalue {{.*}}[32] : !llvm.struct<(i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1)>
    // CHECK: %[[MASK_48:.*]] = llvm.extractvalue {{.*}}[48] : !llvm.struct<(i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1)>
    // CHECK: %[[PITCH:.*]] = llvm.mlir.constant(128 : i32) : i32
    // CHECK: %[[BASE_HEIGHT:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK: %[[BASE_WIDTH:.*]] = llvm.mlir.constant(64 : i32) : i32

    // CHECK: %[[WARP_ID:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id() {no_unwind, will_return} : () -> i32
    // CHECK: llvm.xor {{.*}}, {{.*}} : i32
    // CHECK: %[[OFFSET_X_TO_TILE:.*]] = llvm.xor {{.*}}, {{.*}} : i32
    // CHECK: %[[EXTRACTED_BASE:.*]] = llvm.ptrtoint %[[ADDR_0]] : !llvm.ptr<1> to i64
    // CHECK: %[[UNIFIED_BASE:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflelj(%[[EXTRACTED_BASE]], {{.*}}) {convergent, no_unwind, will_return} : (i64, i32) -> i64
    // CHECK: %[[VAL_26:.*]] = llvm.inttoptr %[[UNIFIED_BASE]] : i64 to !llvm.ptr<1>
    // CHECK: %[[NEG_OFFSET_X:.*]] = llvm.sub {{.*}}, %[[OFFSET_X_TO_TILE]] : i32
    // CHECK: %[[ADJUSTED_BASE:.*]] = llvm.getelementptr %[[VAL_26]]{{\[}}%[[NEG_OFFSET_X]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f16

    // CHECK: %[[MUL_32:.*]] = llvm.mul %[[OFFSET_X_TO_TILE]], {{.*}} : i32
    // CHECK: %[[ADD_98:.*]] = llvm.add %[[BASE_WIDTH]], %[[MUL_32]] : i32
    // CHECK: %[[MIN_BASE_WIDTH:.*]] = llvm.mlir.constant(64 : i32) : i32
    // CHECK: %[[ADJUSTED_BASE_WIDTH:.*]] = llvm.intr.umax(%[[ADD_98]], %[[MIN_BASE_WIDTH]]) : (i32, i32) -> i32

    // CHECK: %[[EXTRACTED_MASK:.*]] = llvm.zext %[[MASK_0]] : i1 to i8
    // CHECK: %[[UNIFIED_MASK:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflecj(%[[EXTRACTED_MASK]], {{.*}}) {convergent, no_unwind, will_return} : (i8, i32) -> i8
    // CHECK: %[[UNIFIED_MASK_I1:.*]] = llvm.trunc %[[UNIFIED_MASK]] : i8 to i1
    // CHECK: %[[OFFSET_Y:.*]] = llvm.select %[[UNIFIED_MASK_I1]], {{.*}}, %[[BASE_HEIGHT]] : i1, i32
    // CHECK: %[[OFFSET_IN_PACKEDELEM_SIZE:.*]] = llvm.udiv %[[OFFSET_X_TO_TILE]], {{.*}} : i32
    // CHECK: triton_gen.2Dblockprefetch %[[ADJUSTED_BASE]], %[[ADJUSTED_BASE_WIDTH]], %[[BASE_HEIGHT]], %[[PITCH]], %[[OFFSET_IN_PACKEDELEM_SIZE]], %[[OFFSET_Y]] {elem_size_in_bits = 16, tile_width = 32, tile_height = 8, v_blocks = 1, cache_control = L1C_L3C}

    // CHECK: llvm.xor {{.*}}, {{.*}} : i32
    // CHECK: %[[OFFSET_X_TO_TILE:.*]] = llvm.xor {{.*}}, {{.*}} : i32
    // CHECK: %[[EXTRACTED_BASE:.*]] = llvm.ptrtoint %[[ADDR_16]] : !llvm.ptr<1> to i64
    // CHECK: %[[UNIFIED_BASE:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflelj(%[[EXTRACTED_BASE]], {{.*}}) {convergent, no_unwind, will_return} : (i64, i32) -> i64
    // CHECK: %[[VAL_26:.*]] = llvm.inttoptr %[[UNIFIED_BASE]] : i64 to !llvm.ptr<1>
    // CHECK: %[[NEG_OFFSET_X:.*]] = llvm.sub {{.*}}, %[[OFFSET_X_TO_TILE]] : i32
    // CHECK: %[[ADJUSTED_BASE:.*]] = llvm.getelementptr %[[VAL_26]]{{\[}}%[[NEG_OFFSET_X]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f16

    // CHECK: %[[MUL_32:.*]] = llvm.mul %[[OFFSET_X_TO_TILE]], {{.*}} : i32
    // CHECK: %[[ADD_98:.*]] = llvm.add %[[BASE_WIDTH]], %[[MUL_32]] : i32
    // CHECK: %[[MIN_BASE_WIDTH:.*]] = llvm.mlir.constant(64 : i32) : i32
    // CHECK: %[[ADJUSTED_BASE_WIDTH:.*]] = llvm.intr.umax(%[[ADD_98]], %[[MIN_BASE_WIDTH]]) : (i32, i32) -> i32

    // CHECK: %[[EXTRACTED_MASK:.*]] = llvm.zext %[[MASK_16]] : i1 to i8
    // CHECK: %[[UNIFIED_MASK:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflecj(%[[EXTRACTED_MASK]], {{.*}}) {convergent, no_unwind, will_return} : (i8, i32) -> i8
    // CHECK: %[[UNIFIED_MASK_I1:.*]] = llvm.trunc %[[UNIFIED_MASK]] : i8 to i1
    // CHECK: %[[OFFSET_Y:.*]] = llvm.select %[[UNIFIED_MASK_I1]], {{.*}}, %[[BASE_HEIGHT]] : i1, i32
    // CHECK: %[[OFFSET_IN_PACKEDELEM_SIZE:.*]] = llvm.udiv %[[OFFSET_X_TO_TILE]], {{.*}} : i32
    // CHECK: triton_gen.2Dblockprefetch %[[ADJUSTED_BASE]], %[[ADJUSTED_BASE_WIDTH]], %[[BASE_HEIGHT]], %[[PITCH]], %[[OFFSET_IN_PACKEDELEM_SIZE]], %[[OFFSET_Y]] {elem_size_in_bits = 16, tile_width = 32, tile_height = 8, v_blocks = 1, cache_control = L1C_L3C}

    // CHECK: llvm.xor {{.*}}, {{.*}} : i32
    // CHECK: %[[OFFSET_X_TO_TILE:.*]] = llvm.xor {{.*}}, {{.*}} : i32
    // CHECK: %[[EXTRACTED_BASE:.*]] = llvm.ptrtoint %[[ADDR_32]] : !llvm.ptr<1> to i64
    // CHECK: %[[UNIFIED_BASE:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflelj(%[[EXTRACTED_BASE]], {{.*}}) {convergent, no_unwind, will_return} : (i64, i32) -> i64
    // CHECK: %[[VAL_26:.*]] = llvm.inttoptr %[[UNIFIED_BASE]] : i64 to !llvm.ptr<1>
    // CHECK: %[[NEG_OFFSET_X:.*]] = llvm.sub {{.*}}, %[[OFFSET_X_TO_TILE]] : i32
    // CHECK: %[[ADJUSTED_BASE:.*]] = llvm.getelementptr %[[VAL_26]]{{\[}}%[[NEG_OFFSET_X]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f16

    // CHECK: %[[MUL_32:.*]] = llvm.mul %[[OFFSET_X_TO_TILE]], {{.*}} : i32
    // CHECK: %[[ADD_98:.*]] = llvm.add %[[BASE_WIDTH]], %[[MUL_32]] : i32
    // CHECK: %[[MIN_BASE_WIDTH:.*]] = llvm.mlir.constant(64 : i32) : i32
    // CHECK: %[[ADJUSTED_BASE_WIDTH:.*]] = llvm.intr.umax(%[[ADD_98]], %[[MIN_BASE_WIDTH]]) : (i32, i32) -> i32

    // CHECK: %[[EXTRACTED_MASK:.*]] = llvm.zext %[[MASK_32]] : i1 to i8
    // CHECK: %[[UNIFIED_MASK:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflecj(%[[EXTRACTED_MASK]], {{.*}}) {convergent, no_unwind, will_return} : (i8, i32) -> i8
    // CHECK: %[[UNIFIED_MASK_I1:.*]] = llvm.trunc %[[UNIFIED_MASK]] : i8 to i1
    // CHECK: %[[OFFSET_Y:.*]] = llvm.select %[[UNIFIED_MASK_I1]], {{.*}}, %[[BASE_HEIGHT]] : i1, i32
    // CHECK: %[[OFFSET_IN_PACKEDELEM_SIZE:.*]] = llvm.udiv %[[OFFSET_X_TO_TILE]], {{.*}} : i32
    // CHECK: triton_gen.2Dblockprefetch %[[ADJUSTED_BASE]], %[[ADJUSTED_BASE_WIDTH]], %[[BASE_HEIGHT]], %[[PITCH]], %[[OFFSET_IN_PACKEDELEM_SIZE]], %[[OFFSET_Y]] {elem_size_in_bits = 16, tile_width = 32, tile_height = 8, v_blocks = 1, cache_control = L1C_L3C}


    // CHECK: llvm.xor {{.*}}, {{.*}} : i32
    // CHECK: %[[OFFSET_X_TO_TILE:.*]] = llvm.xor {{.*}}, {{.*}} : i32
    // CHECK: %[[EXTRACTED_BASE:.*]] = llvm.ptrtoint %[[ADDR_48]] : !llvm.ptr<1> to i64
    // CHECK: %[[UNIFIED_BASE:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflelj(%[[EXTRACTED_BASE]], {{.*}}) {convergent, no_unwind, will_return} : (i64, i32) -> i64
    // CHECK: %[[VAL_26:.*]] = llvm.inttoptr %[[UNIFIED_BASE]] : i64 to !llvm.ptr<1>
    // CHECK: %[[NEG_OFFSET_X:.*]] = llvm.sub {{.*}}, %[[OFFSET_X_TO_TILE]] : i32
    // CHECK: %[[ADJUSTED_BASE:.*]] = llvm.getelementptr %[[VAL_26]]{{\[}}%[[NEG_OFFSET_X]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f16

    // CHECK: %[[MUL_32:.*]] = llvm.mul %[[OFFSET_X_TO_TILE]], {{.*}} : i32
    // CHECK: %[[ADD_98:.*]] = llvm.add %[[BASE_WIDTH]], %[[MUL_32]] : i32
    // CHECK: %[[MIN_BASE_WIDTH:.*]] = llvm.mlir.constant(64 : i32) : i32
    // CHECK: %[[ADJUSTED_BASE_WIDTH:.*]] = llvm.intr.umax(%[[ADD_98]], %[[MIN_BASE_WIDTH]]) : (i32, i32) -> i32

    // CHECK: %[[EXTRACTED_MASK:.*]] = llvm.zext %[[MASK_48]] : i1 to i8
    // CHECK: %[[UNIFIED_MASK:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflecj(%[[EXTRACTED_MASK]], {{.*}}) {convergent, no_unwind, will_return} : (i8, i32) -> i8
    // CHECK: %[[UNIFIED_MASK_I1:.*]] = llvm.trunc %[[UNIFIED_MASK]] : i8 to i1
    // CHECK: %[[OFFSET_Y:.*]] = llvm.select %[[UNIFIED_MASK_I1]], {{.*}}, %[[BASE_HEIGHT]] : i1, i32
    // CHECK: %[[OFFSET_IN_PACKEDELEM_SIZE:.*]] = llvm.udiv %[[OFFSET_X_TO_TILE]], {{.*}} : i32
    // CHECK: triton_gen.2Dblockprefetch %[[ADJUSTED_BASE]], %[[ADJUSTED_BASE_WIDTH]], %[[BASE_HEIGHT]], %[[PITCH]], %[[OFFSET_IN_PACKEDELEM_SIZE]], %[[OFFSET_Y]] {elem_size_in_bits = 16, tile_width = 32, tile_height = 8, v_blocks = 1, cache_control = L1C_L3C}

    %mask_tensor = arith.constant dense<1> : tensor<64x32xi1, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    ttig.prefetch %tensor_of_ptr, %mask_tensor {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 1, 1>, ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>

    // CHECK-COUNT-4: triton_gen.2Dblockprefetch {{.*}} {elem_size_in_bits = 16, tile_width = 32, tile_height = 8, v_blocks = 1, cache_control = L1C_L3C}

    ttig.prefetch %tensor_of_ptr {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 1, 1>, ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>

    tt.return
  }
}

// -----

// COM: Currently the prefetch operation in this test cannot be lowered correctly, so we check that the test compiles cleanly and not 2D block prefetch operation gets generated.
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 8], B = [8, 16], C = [32, 16]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @kernel
  tt.func public @kernel(%arg0 : tensor<128x32x!tt.ptr<f32>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>) {
    // CHECK-NOT: triton_gen.2Dblockprefetch
    ttig.prefetch %arg0 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>, ttig.block_io = "row_major"} : tensor<128x32x!tt.ptr<f32>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    tt.return
  }
}

// -----

// COM: Prefetch of a tensor-of-pointers whose row stride (`lda`) is a runtime
// COM: scalar, as in grouped GEMM. The pitch can't be a compile-time constant,
// COM: so the lowering recovers the runtime stride and materializes
// COM: pitch = lda * elemSize, then feeds it to the HW 2D block prefetch.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @prefetch_runtime_stride
  tt.func public @prefetch_runtime_stride(%arg0: !tt.ptr<f16>, %lda: i32 {tt.divisibility = 16 : i32}) {
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>}>> -> tensor<64x1xi32, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    %lda_splat = tt.splat %lda : i32 -> tensor<64x1xi32, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    %3 = arith.muli %1, %lda_splat : tensor<64x1xi32, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    %4 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>}>> -> tensor<1x32xi32, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    %6 = tt.broadcast %3 : tensor<64x1xi32, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>> -> tensor<64x32xi32, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    %7 = tt.broadcast %5 : tensor<1x32xi32, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>> -> tensor<64x32xi32, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    %8 = arith.addi %6, %7 : tensor<64x32xi32, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    %9 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x32x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    %tp = tt.addptr %9, %8 : tensor<64x32x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>, tensor<64x32xi32, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    // CHECK: %[[PITCH:.*]] = llvm.mul %arg1, %{{.*}} : i32
    // CHECK: triton_gen.2Dblockprefetch %{{.*}}, %{{.*}}, %{{.*}}, %[[PITCH]],
    ttig.prefetch %tp {cache = 1 : i32, evict = 1 : i32, isVolatile = false, ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    tt.return
  }
}

// -----

// COM: Prefetch of a tensor-of-pointers whose stride is neither a compile-time
// COM: constant nor a recoverable runtime value (the pointer tensor is a bare
// COM: function argument). The lowering must cleanly bail to a no-op rather
// COM: than emit a malformed prefetch.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @prefetch_unknown_stride
  // CHECK-NOT: triton_gen.2Dblockprefetch
  tt.func public @prefetch_unknown_stride(%arg0: tensor<64x32x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>) {
    ttig.prefetch %arg0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false, ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    tt.return
  }
}

// -----

// COM: Prefetch of a column-major tensor-of-pointers with affine pointer
// COM: arithmetic. This should lower to 2D block prefetch ops.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @prefetch_column_major
  // CHECK: %[[PITCH:.*]] = llvm.mlir.constant(128 : i32) : i32
  // CHECK-COUNT-2: triton_gen.2Dblockprefetch {{.*}} {elem_size_in_bits = 32, tile_width = 16, tile_height = 16, v_blocks = 1, cache_control = L1C_L3C}
  tt.func public @prefetch_column_major(%arg0: !tt.ptr<f16>) {
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>}>> -> tensor<64x1xi32, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>
    %2 = tt.broadcast %1 : tensor<64x1xi32, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>> -> tensor<64x32xi32, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>
    %3 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>}>>
    %4 = tt.expand_dims %3 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>}>> -> tensor<1x32xi32, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>
    %5 = arith.constant dense<64> : tensor<1x32xi32, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>
    %6 = arith.muli %4, %5 : tensor<1x32xi32, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>
    %7 = tt.broadcast %6 : tensor<1x32xi32, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>> -> tensor<64x32xi32, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>
    %8 = arith.addi %2, %7 : tensor<64x32xi32, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>
    %9 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x32x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>
    %10 = tt.addptr %9, %8 : tensor<64x32x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>, tensor<64x32xi32, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>
    ttig.prefetch %10 {cache = 1 : i32, evict = 1 : i32, isVolatile = false, ttig.block_io = "column_major"} : tensor<64x32x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>
    tt.return
  }
}

// -----

// COM: Column-major fp8 tensor-of-pointers prefetch.
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @prefetch_column_major_fp8
  // COM: The prefetch only supports <= 64 bytes or 256 bytes. 128 bytes per row are split into two 64 bytes prefetches.
  // COM: The redundant prefetches are dummy prefetches as hit on miss.
  // COM: Check the prefetch tile of
  // COM:                      even Warp            odd Warp         even Warp             odd Warp
  // COM:
  // COM:                 addr0──────────────┬──────────────────addr256────────────┬──────────────────┐
  // COM:                 │                  │                  │                  │                  │
  // COM:                 │                  │                  │                  │                  │
  // COM:                 │                  │                  │                  │                  │
  // COM:       64 bytes  │                  │                  │                  │                  │
  // COM:                 │                  │                  │                  │                  │
  // COM:                 │                  │                  │                  │                  │
  // COM:                 │                  │                  │                  │                  │
  // COM:                 addr128────────────┼──────────────────addr384────────────┼──────────────────┤
  // COM:                 │                  │                  │                  │                  │
  // COM:                 │                  │                  │                  │                  │
  // COM:                 │                  │                  │                  │                  │
  // COM:       64 bytes  │                  │                  │                  │                  │
  // COM:                 │                  │                  │                  │                  │
  // COM:                 │                  │                  │                  │                  │
  // COM:                 │                  │                  │                  │                  │
  // COM:                 └──────────────────┴──────────────────┴──────────────────┴──────────────────┘
  // COM:                        32 row             32 row             32 row              32 row
  // CHECK: llvm.extractvalue {{.*}}[0] : !llvm.struct<(ptr<1>
  // CHECK: %[[ADDR_0:.*]] = llvm.extractvalue {{.*}}[0] : !llvm.struct<(ptr<1>,
  // CHECK: %[[ADDR_128:.*]] = llvm.extractvalue {{.*}}[128] : !llvm.struct<(ptr<1>,
  // CHECK: %[[ADDR_256:.*]] = llvm.extractvalue {{.*}}[256] : !llvm.struct<(ptr<1>,
  // CHECK: %[[ADDR_384:.*]] = llvm.extractvalue {{.*}}[384] : !llvm.struct<(ptr<1>,
  // CHECK: %[[BASE_I64:.*]] = llvm.ptrtoint %[[ADDR_0]] : !llvm.ptr<1> to i64
  // CHECK: %[[SHUF_BASE:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflelj(%[[BASE_I64]], {{.*}})
  // CHECK: %[[BASE_PTR:.*]] = llvm.inttoptr %[[SHUF_BASE]] : i64 to !llvm.ptr<1>
  // CHECK: %[[NEG_X:.*]] = llvm.sub {{.*}}, {{.*}} : i32
  // CHECK: %[[ADJ_BASE:.*]] = llvm.getelementptr %[[BASE_PTR]]{{\[}}%[[NEG_X]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>
  // CHECK: %[[X_PACKED:.*]] = llvm.udiv {{.*}}, {{.*}} : i32
  // CHECK: triton_gen.2Dblockprefetch %[[ADJ_BASE]], {{.*}}, {{.*}}, {{.*}}, %[[X_PACKED]], {{.*}} {elem_size_in_bits = 32, tile_width = 16, tile_height = 32, v_blocks = 1, cache_control = L1C_L3C}
  // CHECK: %[[BASE_I64:.*]] = llvm.ptrtoint %[[ADDR_128]] : !llvm.ptr<1> to i64
  // CHECK: %[[SHUF_BASE:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflelj(%[[BASE_I64]], {{.*}})
  // CHECK: %[[BASE_PTR:.*]] = llvm.inttoptr %[[SHUF_BASE]] : i64 to !llvm.ptr<1>
  // CHECK: %[[NEG_X:.*]] = llvm.sub {{.*}}, {{.*}} : i32
  // CHECK: %[[ADJ_BASE:.*]] = llvm.getelementptr %[[BASE_PTR]]{{\[}}%[[NEG_X]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>
  // CHECK: %[[X_PACKED:.*]] = llvm.udiv {{.*}}, {{.*}} : i32
  // CHECK: triton_gen.2Dblockprefetch %[[ADJ_BASE]], {{.*}}, {{.*}}, {{.*}}, %[[X_PACKED]], {{.*}} {elem_size_in_bits = 32, tile_width = 16, tile_height = 32, v_blocks = 1, cache_control = L1C_L3C}
  // CHECK: %[[BASE_I64:.*]] = llvm.ptrtoint %[[ADDR_256]] : !llvm.ptr<1> to i64
  // CHECK: %[[SHUF_BASE:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflelj(%[[BASE_I64]], {{.*}})
  // CHECK: %[[BASE_PTR:.*]] = llvm.inttoptr %[[SHUF_BASE]] : i64 to !llvm.ptr<1>
  // CHECK: %[[NEG_X:.*]] = llvm.sub {{.*}}, {{.*}} : i32
  // CHECK: %[[ADJ_BASE:.*]] = llvm.getelementptr %[[BASE_PTR]]{{\[}}%[[NEG_X]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>
  // CHECK: %[[X_PACKED:.*]] = llvm.udiv {{.*}}, {{.*}} : i32
  // CHECK: triton_gen.2Dblockprefetch %[[ADJ_BASE]], {{.*}}, {{.*}}, {{.*}}, %[[X_PACKED]], {{.*}} {elem_size_in_bits = 32, tile_width = 16, tile_height = 32, v_blocks = 1, cache_control = L1C_L3C}
  // CHECK: %[[BASE_I64:.*]] = llvm.ptrtoint %[[ADDR_384]] : !llvm.ptr<1> to i64
  // CHECK: %[[SHUF_BASE:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflelj(%[[BASE_I64]], {{.*}})
  // CHECK: %[[BASE_PTR:.*]] = llvm.inttoptr %[[SHUF_BASE]] : i64 to !llvm.ptr<1>
  // CHECK: %[[NEG_X:.*]] = llvm.sub {{.*}}, {{.*}} : i32
  // CHECK: %[[ADJ_BASE:.*]] = llvm.getelementptr %[[BASE_PTR]]{{\[}}%[[NEG_X]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>
  // CHECK: %[[X_PACKED:.*]] = llvm.udiv {{.*}}, {{.*}} : i32
  // CHECK: triton_gen.2Dblockprefetch %[[ADJ_BASE]], {{.*}}, {{.*}}, {{.*}}, %[[X_PACKED]], {{.*}} {elem_size_in_bits = 32, tile_width = 16, tile_height = 32, v_blocks = 1, cache_control = L1C_L3C}
  tt.func public @prefetch_column_major_fp8(%arg0: !tt.ptr<f8E4M3FN>, %arg1: !tt.ptr<f8E4M3FN>, %pred: i1) {
    %cst = arith.constant dense<256> : tensor<1x128xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %k = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>}>>
    %k2d = tt.expand_dims %k {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>}>> -> tensor<128x1xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %kBc = tt.broadcast %k2d : tensor<128x1xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %n = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>}>>
    %n2d = tt.expand_dims %n {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>}>> -> tensor<1x128xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %nScaled = arith.muli %n2d, %cst : tensor<1x128xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %nScaledBc = tt.broadcast %nScaled : tensor<1x128xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %offsets = arith.addi %kBc, %nScaledBc : tensor<128x128xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %base = tt.splat %arg0 : !tt.ptr<f8E4M3FN> -> tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %ptrs = tt.addptr %base, %offsets : tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>, tensor<128x128xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>

    ttig.prefetch %ptrs {cache = 1 : i32, evict = 1 : i32, isVolatile = false, ttig.block_io = "column_major"} : tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>

    tt.return
  }
}


// -----

// COM: Column-major fp8 tensor-of-pointers prefetch.
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_prefetch_256b"} {
  // CHECK-LABEL: llvm.func spir_kernelcc @prefetch_column_major_fp8
  // COM: The prefetch only supports <= 64 bytes or 256 bytes. 128 bytes per row are split into two 64 bytes prefetches.
  // COM: The redundant prefetches are dummy prefetches as hit on miss.
  // COM: Check the prefetch tile of
  // COM:                      even Warp            odd Warp         even Warp             odd Warp
  // COM:
  // COM:                 addr0──────────────┬──────────────────addr512────────────┬──────────────────┐
  // COM:                 │                  │                  │                  │                  │
  // COM:                 │                  │                  │                  │                  │
  // COM:                 │                  │                  │                  │                  │
  // COM:      256 bytes  │                  │                  │                  │                  │
  // COM:                 │                  │                  │                  │                  │
  // COM:                 │                  │                  │                  │                  │
  // COM:                 │                  │                  │                  │                  │
  // COM:                 └──────────────────┴──────────────────┴──────────────────┴──────────────────┘
  // COM:                        32 row             32 row             32 row              32 row
  // CHECK: llvm.extractvalue {{.*}}[0] : !llvm.struct<(ptr<1>
  // CHECK: %[[ADDR_0:.*]] = llvm.extractvalue {{.*}}[0] : !llvm.struct<(ptr<1>,
  // CHECK: %[[ADDR_512:.*]] = llvm.extractvalue {{.*}}[512] : !llvm.struct<(ptr<1>,
  // CHECK: %[[BASE_I64:.*]] = llvm.ptrtoint %[[ADDR_0]] : !llvm.ptr<1> to i64
  // CHECK: %[[SHUF_BASE:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflelj(%[[BASE_I64]], {{.*}})
  // CHECK: %[[BASE_PTR:.*]] = llvm.inttoptr %[[SHUF_BASE]] : i64 to !llvm.ptr<1>
  // CHECK: %[[NEG_X:.*]] = llvm.sub {{.*}}, {{.*}} : i32
  // CHECK: %[[ADJ_BASE:.*]] = llvm.getelementptr %[[BASE_PTR]]{{\[}}%[[NEG_X]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>
  // CHECK: %[[X_PACKED:.*]] = llvm.udiv {{.*}}, {{.*}} : i32
  // CHECK: triton_gen.2Dblockprefetch %[[ADJ_BASE]], {{.*}}, {{.*}}, {{.*}}, %[[X_PACKED]], {{.*}} {elem_size_in_bits = 32, tile_width = 64, tile_height = 32, v_blocks = 1, cache_control = L1C_L3C}
  // CHECK: %[[BASE_I64:.*]] = llvm.ptrtoint %[[ADDR_512]] : !llvm.ptr<1> to i64
  // CHECK: %[[SHUF_BASE:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflelj(%[[BASE_I64]], {{.*}})
  // CHECK: %[[BASE_PTR:.*]] = llvm.inttoptr %[[SHUF_BASE]] : i64 to !llvm.ptr<1>
  // CHECK: %[[NEG_X:.*]] = llvm.sub {{.*}}, {{.*}} : i32
  // CHECK: %[[ADJ_BASE:.*]] = llvm.getelementptr %[[BASE_PTR]]{{\[}}%[[NEG_X]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>
  // CHECK: %[[X_PACKED:.*]] = llvm.udiv {{.*}}, {{.*}} : i32
  // CHECK: triton_gen.2Dblockprefetch %[[ADJ_BASE]], {{.*}}, {{.*}}, {{.*}}, %[[X_PACKED]], {{.*}} {elem_size_in_bits = 32, tile_width = 64, tile_height = 32, v_blocks = 1, cache_control = L1C_L3C}
  tt.func public @prefetch_column_major_fp8(%arg0: !tt.ptr<f8E4M3FN>, %arg1: !tt.ptr<f8E4M3FN>, %pred: i1) {
    %cst = arith.constant dense<256> : tensor<1x128xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %k = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>}>>
    %k2d = tt.expand_dims %k {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>}>> -> tensor<256x1xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %kBc = tt.broadcast %k2d : tensor<256x1xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<256x128xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %n = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>}>>
    %n2d = tt.expand_dims %n {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>}>> -> tensor<1x128xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %nScaled = arith.muli %n2d, %cst : tensor<1x128xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %nScaledBc = tt.broadcast %nScaled : tensor<1x128xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<256x128xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %offsets = arith.addi %kBc, %nScaledBc : tensor<256x128xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %base = tt.splat %arg0 : !tt.ptr<f8E4M3FN> -> tensor<256x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %ptrs = tt.addptr %base, %offsets : tensor<256x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>, tensor<256x128xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>

    ttig.prefetch %ptrs {cache = 1 : i32, evict = 1 : i32, isVolatile = false, ttig.block_io = "column_major"} : tensor<256x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>

    tt.return
  }
}

// -----

// COM: A prefetch row-limit mask ANDed with a uniform loop-drain guard.
// COM: base_height must be computed overflow-safely (widen to i64 before
// COM: subtracting), and the guard must still gate the message.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @prefetch_row_limit_guarded
  tt.func public @prefetch_row_limit_guarded(%arg0: !tt.ptr<f16>, %lda: i32 {tt.divisibility = 16 : i32}, %row_base: i32, %bound: i32, %guard: i1) {
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>> -> tensor<64x1xi32, #dot0>
    %lda_splat = tt.splat %lda : i32 -> tensor<64x1xi32, #dot0>
    %row = arith.muli %1, %lda_splat : tensor<64x1xi32, #dot0>
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>>
    %3 = tt.expand_dims %2 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>> -> tensor<1x32xi32, #dot0>
    %rowb = tt.broadcast %row : tensor<64x1xi32, #dot0> -> tensor<64x32xi32, #dot0>
    %colb = tt.broadcast %3 : tensor<1x32xi32, #dot0> -> tensor<64x32xi32, #dot0>
    %off = arith.addi %rowb, %colb : tensor<64x32xi32, #dot0>
    %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x32x!tt.ptr<f16>, #dot0>
    %ptrs = tt.addptr %base, %off : tensor<64x32x!tt.ptr<f16>, #dot0>, tensor<64x32xi32, #dot0>
    // rows = row_base + arange(0, 64); row_limit = rows < bound.
    %5 = tt.splat %row_base : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>>
    %6 = arith.addi %5, %0 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>>
    %8 = tt.splat %bound : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>>
    %9 = arith.cmpi slt, %6, %8 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>>
    %10 = tt.expand_dims %9 {axis = 1 : i32} : tensor<64xi1, #ttg.slice<{dim = 1, parent = #dot0}>> -> tensor<64x1xi1, #dot0>
    %rowlimit = tt.broadcast %10 : tensor<64x1xi1, #dot0> -> tensor<64x32xi1, #dot0>
    // Uniform loop-drain guard, ANDed on as the extra conjunct.
    %guardb = tt.splat %guard : i1 -> tensor<64x32xi1, #dot0>
    %mask = arith.andi %rowlimit, %guardb : tensor<64x32xi1, #dot0>

    // CHECK: %[[BOUND64:.*]] = llvm.sext %{{.*}} : i32 to i64
    // CHECK: %[[BASE64:.*]] = llvm.sext %{{.*}} : i32 to i64
    // CHECK: %[[DIFF64:.*]] = llvm.sub %[[BOUND64]], %[[BASE64]] : i64
    // CHECK: %[[MAX64:.*]] = llvm.intr.smax(%[[DIFF64]], %{{.*}}) : (i64, i64) -> i64
    // CHECK: %[[CLAMP64:.*]] = llvm.intr.smin(%[[MAX64]], %{{.*}}) : (i64, i64) -> i64
    // CHECK: %[[H:.*]] = llvm.trunc %[[CLAMP64]] : i64 to i32
    // CHECK: %[[LOCAL:.*]] = llvm.sub %[[H]], %{{.*}} : i32
    // CHECK: %[[VALID:.*]] = llvm.icmp "sgt" %[[LOCAL]], %{{.*}} : i32
    // CHECK: %[[PRED:.*]] = llvm.and %[[VALID]], %{{.*}} : i1
    // CHECK: llvm.select %[[PRED]]
    // CHECK: triton_gen.2Dblockprefetch
    ttig.prefetch %ptrs, %mask {cache = 1 : i32, evict = 1 : i32, isVolatile = false, ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #dot0>
    tt.return
  }
}

// -----

// COM: A bare prefetch row-limit mask with no extra conjunct. Height must
// COM: still be computed, and the select must gate directly on the
// COM: row-limit validity predicate with no extraGuard AND involved.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @prefetch_row_limit_bare
  tt.func public @prefetch_row_limit_bare(%arg0: !tt.ptr<f16>, %lda: i32 {tt.divisibility = 16 : i32}, %row_base: i32, %bound: i32) {
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>> -> tensor<64x1xi32, #dot0>
    %lda_splat = tt.splat %lda : i32 -> tensor<64x1xi32, #dot0>
    %row = arith.muli %1, %lda_splat : tensor<64x1xi32, #dot0>
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>>
    %3 = tt.expand_dims %2 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>> -> tensor<1x32xi32, #dot0>
    %rowb = tt.broadcast %row : tensor<64x1xi32, #dot0> -> tensor<64x32xi32, #dot0>
    %colb = tt.broadcast %3 : tensor<1x32xi32, #dot0> -> tensor<64x32xi32, #dot0>
    %off = arith.addi %rowb, %colb : tensor<64x32xi32, #dot0>
    %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x32x!tt.ptr<f16>, #dot0>
    %ptrs = tt.addptr %base, %off : tensor<64x32x!tt.ptr<f16>, #dot0>, tensor<64x32xi32, #dot0>
    %5 = tt.splat %row_base : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>>
    %6 = arith.addi %5, %0 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>>
    %8 = tt.splat %bound : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>>
    %9 = arith.cmpi slt, %6, %8 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>>
    %10 = tt.expand_dims %9 {axis = 1 : i32} : tensor<64xi1, #ttg.slice<{dim = 1, parent = #dot0}>> -> tensor<64x1xi1, #dot0>
    %mask = tt.broadcast %10 : tensor<64x1xi1, #dot0> -> tensor<64x32xi1, #dot0>
    // CHECK: %[[BOUND64:.*]] = llvm.sext %{{.*}} : i32 to i64
    // CHECK: %[[BASE64:.*]] = llvm.sext %{{.*}} : i32 to i64
    // CHECK: %[[DIFF64:.*]] = llvm.sub %[[BOUND64]], %[[BASE64]] : i64
    // CHECK: %[[MAX64:.*]] = llvm.intr.smax(%[[DIFF64]], %{{.*}}) : (i64, i64) -> i64
    // CHECK: %[[CLAMP64:.*]] = llvm.intr.smin(%[[MAX64]], %{{.*}}) : (i64, i64) -> i64
    // CHECK: %[[H:.*]] = llvm.trunc %[[CLAMP64]] : i64 to i32
    // CHECK: %[[LOCAL:.*]] = llvm.sub %[[H]], %{{.*}} : i32
    // CHECK: %[[VALID:.*]] = llvm.icmp "sgt" %[[LOCAL]], %{{.*}} : i32
    // CHECK: llvm.select %[[VALID]]
    // CHECK: triton_gen.2Dblockprefetch
    ttig.prefetch %ptrs, %mask {cache = 1 : i32, evict = 1 : i32, isVolatile = false, ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #dot0>
    tt.return
  }
}

// -----

// COM: i64 counterpart of @prefetch_row_limit_bare (issue #8565). The i64
// COM: operands are not sign-extended, and the height uses the overflow-safe
// COM: `(bound <=s rowBase) ? 0 : umin(bound - rowBase, 64)`.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @prefetch_row_limit_bare_i64
  // CHECK-SAME: %[[ROW_BASE:[^:]*]]: i64, %[[BOUND:[^:]*]]: i64
  tt.func public @prefetch_row_limit_bare_i64(%arg0: !tt.ptr<f16>, %lda: i32 {tt.divisibility = 16 : i32}, %row_base: i64, %bound: i64) {
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>> -> tensor<64x1xi32, #dot0>
    %lda_splat = tt.splat %lda : i32 -> tensor<64x1xi32, #dot0>
    %row = arith.muli %1, %lda_splat : tensor<64x1xi32, #dot0>
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>>
    %3 = tt.expand_dims %2 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>> -> tensor<1x32xi32, #dot0>
    %rowb = tt.broadcast %row : tensor<64x1xi32, #dot0> -> tensor<64x32xi32, #dot0>
    %colb = tt.broadcast %3 : tensor<1x32xi32, #dot0> -> tensor<64x32xi32, #dot0>
    %off = arith.addi %rowb, %colb : tensor<64x32xi32, #dot0>
    %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x32x!tt.ptr<f16>, #dot0>
    %ptrs = tt.addptr %base, %off : tensor<64x32x!tt.ptr<f16>, #dot0>, tensor<64x32xi32, #dot0>
    %r0_64 = arith.extsi %0 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>> to tensor<64xi64, #ttg.slice<{dim = 1, parent = #dot0}>>
    %5 = tt.splat %row_base : i64 -> tensor<64xi64, #ttg.slice<{dim = 1, parent = #dot0}>>
    %6 = arith.addi %5, %r0_64 : tensor<64xi64, #ttg.slice<{dim = 1, parent = #dot0}>>
    %8 = tt.splat %bound : i64 -> tensor<64xi64, #ttg.slice<{dim = 1, parent = #dot0}>>
    %9 = arith.cmpi slt, %6, %8 : tensor<64xi64, #ttg.slice<{dim = 1, parent = #dot0}>>
    %10 = tt.expand_dims %9 {axis = 1 : i32} : tensor<64xi1, #ttg.slice<{dim = 1, parent = #dot0}>> -> tensor<64x1xi1, #dot0>
    %mask = tt.broadcast %10 : tensor<64x1xi1, #dot0> -> tensor<64x32xi1, #dot0>
    // CHECK-NOT: llvm.sext %[[BOUND]]
    // CHECK-NOT: llvm.sext %[[ROW_BASE]]
    // CHECK-DAG: %[[EMPTY:.*]] = llvm.icmp "sle" %[[BOUND]], %[[ROW_BASE]] : i64
    // CHECK-DAG: %[[DIFF64:.*]] = llvm.sub %[[BOUND]], %[[ROW_BASE]] : i64
    // CHECK: %[[MIN64:.*]] = llvm.intr.umin(%[[DIFF64]], %{{.*}}) : (i64, i64) -> i64
    // CHECK: %[[SEL64:.*]] = llvm.select %[[EMPTY]], %{{.*}}, %[[MIN64]] : i1, i64
    // CHECK: %[[H:.*]] = llvm.trunc %[[SEL64]] : i64 to i32
    // CHECK: %[[LOCAL:.*]] = llvm.sub %[[H]], %{{.*}} : i32
    // CHECK: %[[VALID:.*]] = llvm.icmp "sgt" %[[LOCAL]], %{{.*}} : i32
    // CHECK: llvm.select %[[VALID]]
    // CHECK: triton_gen.2Dblockprefetch
    ttig.prefetch %ptrs, %mask {cache = 1 : i32, evict = 1 : i32, isVolatile = false, ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #dot0>
    tt.return
  }
}

// -----

// COM: A prefetch mask ANDing the row-limit pattern with a second conjunct
// COM: that is itself non-uniform (a per-column bound), not a splat scalar
// COM: predicate. matchPrefetchRowLimitMask can't express that as an
// COM: extraGuard, so it abandons the row-limit optimization entirely; the
// COM: resulting mask can't be lowered to a 2D block prefetch at all and the
// COM: op is dropped, mirroring prefetch_unknown_stride's bail-to-no-op path.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @prefetch_row_limit_nonsplat_conjunct
  // CHECK-NOT: triton_gen.2Dblockprefetch
  tt.func public @prefetch_row_limit_nonsplat_conjunct(%arg0: !tt.ptr<f16>, %lda: i32 {tt.divisibility = 16 : i32}, %row_base: i32, %bound: i32, %col_bound: i32) {
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>> -> tensor<64x1xi32, #dot0>
    %lda_splat = tt.splat %lda : i32 -> tensor<64x1xi32, #dot0>
    %row = arith.muli %1, %lda_splat : tensor<64x1xi32, #dot0>
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>>
    %3 = tt.expand_dims %2 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>> -> tensor<1x32xi32, #dot0>
    %rowb = tt.broadcast %row : tensor<64x1xi32, #dot0> -> tensor<64x32xi32, #dot0>
    %colb = tt.broadcast %3 : tensor<1x32xi32, #dot0> -> tensor<64x32xi32, #dot0>
    %off = arith.addi %rowb, %colb : tensor<64x32xi32, #dot0>
    %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x32x!tt.ptr<f16>, #dot0>
    %ptrs = tt.addptr %base, %off : tensor<64x32x!tt.ptr<f16>, #dot0>, tensor<64x32xi32, #dot0>
    %5 = tt.splat %row_base : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>>
    %6 = arith.addi %5, %0 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>>
    %8 = tt.splat %bound : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>>
    %9 = arith.cmpi slt, %6, %8 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>>
    %10 = tt.expand_dims %9 {axis = 1 : i32} : tensor<64xi1, #ttg.slice<{dim = 1, parent = #dot0}>> -> tensor<64x1xi1, #dot0>
    %rowlimit = tt.broadcast %10 : tensor<64x1xi1, #dot0> -> tensor<64x32xi1, #dot0>
    %colbound_splat = tt.splat %col_bound : i32 -> tensor<1x32xi32, #dot0>
    %colcmp = arith.cmpi slt, %3, %colbound_splat : tensor<1x32xi32, #dot0>
    %colmask = tt.broadcast %colcmp : tensor<1x32xi1, #dot0> -> tensor<64x32xi1, #dot0>
    %mask = arith.andi %rowlimit, %colmask : tensor<64x32xi1, #dot0>
    ttig.prefetch %ptrs, %mask {cache = 1 : i32, evict = 1 : i32, isVolatile = false, ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #dot0>
    tt.return
  }
}

// -----

// COM: A row-limit mask on a prefetch whose pitch can't be recovered (the
// COM: pointer tensor is a bare function argument). The lowering bails to a
// COM: no-op, and must do so before materializing the row-limit height, so no
// COM: dead clamp ops are left behind.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @prefetch_row_limit_unknown_stride
  // CHECK-NOT: triton_gen.2Dblockprefetch
  // CHECK-NOT: llvm.intr.smax
  tt.func public @prefetch_row_limit_unknown_stride(%arg0: tensor<64x32x!tt.ptr<f16>, #dot0>, %bound: i32) {
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>>
    %1 = tt.splat %bound : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>>
    %2 = arith.cmpi slt, %0, %1 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>>
    %3 = tt.expand_dims %2 {axis = 1 : i32} : tensor<64xi1, #ttg.slice<{dim = 1, parent = #dot0}>> -> tensor<64x1xi1, #dot0>
    %mask = tt.broadcast %3 : tensor<64x1xi1, #dot0> -> tensor<64x32xi1, #dot0>
    ttig.prefetch %arg0, %mask {cache = 1 : i32, evict = 1 : i32, isVolatile = false, ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #dot0>
    tt.return
  }
}

// -----

// COM: Rank-3 prefetch whose mask is a row-limit ANDed with a batch-only
// COM: predicate (degenerate along M and K). The row half becomes a dynamic
// COM: base_height; the batch half guards each message.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4, 2], repCluster = [1, 1, 1], A = [1, 8, 16], B = [1, 16, 16], C = [1, 8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#sK = #ttg.slice<{dim = 2, parent = #dot0}>
#sM = #ttg.slice<{dim = 1, parent = #dot0}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @prefetch_row_limit_rank3_batch_pred
  tt.func @prefetch_row_limit_rank3_batch_pred(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %row_base: i32 {tt.divisibility = 16 : i32}, %bound: i32 {tt.divisibility = 16 : i32}, %batch_bound: i32 {tt.divisibility = 16 : i32}) {
    %c32 = arith.constant dense<32> : tensor<1x64x1xi32, #dot0>
    // ptr[b, i, j] = arg0 + i * 32 + j; batch addressing is irrelevant here,
    // so every batch reads the same 64x32 slice.
    %rb = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #sK}>>
    %rm = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #sK}>>
    %ema = tt.expand_dims %rm {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #sK}>> -> tensor<1x64xi32, #sK>
    %em = tt.expand_dims %ema {axis = 2 : i32} : tensor<1x64xi32, #sK> -> tensor<1x64x1xi32, #dot0>
    %om = arith.muli %em, %c32 : tensor<1x64x1xi32, #dot0>
    %rk = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #sM}>>
    %eka = tt.expand_dims %rk {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #sM}>> -> tensor<1x32xi32, #sM>
    %ek = tt.expand_dims %eka {axis = 1 : i32} : tensor<1x32xi32, #sM> -> tensor<1x1x32xi32, #dot0>
    %bm = tt.broadcast %om : tensor<1x64x1xi32, #dot0> -> tensor<4x64x32xi32, #dot0>
    %bk = tt.broadcast %ek : tensor<1x1x32xi32, #dot0> -> tensor<4x64x32xi32, #dot0>
    %off = arith.addi %bm, %bk : tensor<4x64x32xi32, #dot0>
    %ps = tt.splat %arg0 : !tt.ptr<f16> -> tensor<4x64x32x!tt.ptr<f16>, #dot0>
    %ptr = tt.addptr %ps, %off : tensor<4x64x32x!tt.ptr<f16>, #dot0>, tensor<4x64x32xi32, #dot0>
    // rows = row_base + arange(0, 64) < bound; batch = arange(0, 4) < batch_bound.
    %rbase = tt.splat %row_base : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #sK}>>
    %rows = arith.addi %rbase, %rm : tensor<64xi32, #ttg.slice<{dim = 0, parent = #sK}>>
    %boundsplat = tt.splat %bound : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #sK}>>
    %rowcmp = arith.cmpi slt, %rows, %boundsplat : tensor<64xi32, #ttg.slice<{dim = 0, parent = #sK}>>
    %rc1 = tt.expand_dims %rowcmp {axis = 0 : i32} : tensor<64xi1, #ttg.slice<{dim = 0, parent = #sK}>> -> tensor<1x64xi1, #sK>
    %rc2 = tt.expand_dims %rc1 {axis = 2 : i32} : tensor<1x64xi1, #sK> -> tensor<1x64x1xi1, #dot0>
    %rowmask = tt.broadcast %rc2 : tensor<1x64x1xi1, #dot0> -> tensor<4x64x32xi1, #dot0>
    %bboundsplat = tt.splat %batch_bound : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #sK}>>
    %batchcmp = arith.cmpi slt, %rb, %bboundsplat : tensor<4xi32, #ttg.slice<{dim = 1, parent = #sK}>>
    %bc1 = tt.expand_dims %batchcmp {axis = 1 : i32} : tensor<4xi1, #ttg.slice<{dim = 1, parent = #sK}>> -> tensor<4x1xi1, #sK>
    %bc2 = tt.expand_dims %bc1 {axis = 2 : i32} : tensor<4x1xi1, #sK> -> tensor<4x1x1xi1, #dot0>
    %batchmask = tt.broadcast %bc2 : tensor<4x1x1xi1, #dot0> -> tensor<4x64x32xi1, #dot0>
    %mask = arith.andi %rowmask, %batchmask : tensor<4x64x32xi1, #dot0>
    // CHECK: llvm.intr.smax
    // CHECK: llvm.icmp "sgt"
    // CHECK: llvm.and {{.*}} : i1
    // CHECK: llvm.select
    // CHECK: triton_gen.2Dblockprefetch
    ttig.prefetch %ptr, %mask {cache = 1 : i32, evict = 1 : i32, isVolatile = false, ttig.block_io = "row_major"} : tensor<4x64x32x!tt.ptr<f16>, #dot0>
    tt.return
  }
}

// -----

// COM: Same as @prefetch_row_limit_rank3_sunk_and, but with the AND below the
// COM: final broadcast, as `tt-reorder-broadcast` leaves it:
// COM: broadcast(and(broadcast(row), broadcast(batch))). The batch predicate is
// COM: lowered at its narrow [4, 64, 1] shape and expanded to the tile's elements.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4, 2], repCluster = [1, 1, 1], A = [1, 8, 16], B = [1, 16, 16], C = [1, 8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#sK = #ttg.slice<{dim = 2, parent = #dot0}>
#sM = #ttg.slice<{dim = 1, parent = #dot0}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @prefetch_row_limit_rank3_sunk_and
  tt.func @prefetch_row_limit_rank3_sunk_and(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %row_base: i32 {tt.divisibility = 16 : i32}, %bound: i32 {tt.divisibility = 16 : i32}, %batch_bound: i32 {tt.divisibility = 16 : i32}) {
    %c32 = arith.constant dense<32> : tensor<1x64x1xi32, #dot0>
    // ptr[b, i, j] = arg0 + i * 32 + j; batch addressing is irrelevant here,
    // so every batch reads the same 64x32 slice.
    %rb = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #sK}>>
    %rm = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #sK}>>
    %ema = tt.expand_dims %rm {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #sK}>> -> tensor<1x64xi32, #sK>
    %em = tt.expand_dims %ema {axis = 2 : i32} : tensor<1x64xi32, #sK> -> tensor<1x64x1xi32, #dot0>
    %om = arith.muli %em, %c32 : tensor<1x64x1xi32, #dot0>
    %rk = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #sM}>>
    %eka = tt.expand_dims %rk {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #sM}>> -> tensor<1x32xi32, #sM>
    %ek = tt.expand_dims %eka {axis = 1 : i32} : tensor<1x32xi32, #sM> -> tensor<1x1x32xi32, #dot0>
    %bm = tt.broadcast %om : tensor<1x64x1xi32, #dot0> -> tensor<4x64x32xi32, #dot0>
    %bk = tt.broadcast %ek : tensor<1x1x32xi32, #dot0> -> tensor<4x64x32xi32, #dot0>
    %off = arith.addi %bm, %bk : tensor<4x64x32xi32, #dot0>
    %ps = tt.splat %arg0 : !tt.ptr<f16> -> tensor<4x64x32x!tt.ptr<f16>, #dot0>
    %ptr = tt.addptr %ps, %off : tensor<4x64x32x!tt.ptr<f16>, #dot0>, tensor<4x64x32xi32, #dot0>
    // rows = row_base + arange(0, 64) < bound; batch = arange(0, 4) < batch_bound.
    %rbase = tt.splat %row_base : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #sK}>>
    %rows = arith.addi %rbase, %rm : tensor<64xi32, #ttg.slice<{dim = 0, parent = #sK}>>
    %boundsplat = tt.splat %bound : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #sK}>>
    %rowcmp = arith.cmpi slt, %rows, %boundsplat : tensor<64xi32, #ttg.slice<{dim = 0, parent = #sK}>>
    %rc1 = tt.expand_dims %rowcmp {axis = 0 : i32} : tensor<64xi1, #ttg.slice<{dim = 0, parent = #sK}>> -> tensor<1x64xi1, #sK>
    %rc2 = tt.expand_dims %rc1 {axis = 2 : i32} : tensor<1x64xi1, #sK> -> tensor<1x64x1xi1, #dot0>
    %rowb = tt.broadcast %rc2 : tensor<1x64x1xi1, #dot0> -> tensor<4x64x1xi1, #dot0>
    %bboundsplat = tt.splat %batch_bound : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #sK}>>
    %batchcmp = arith.cmpi slt, %rb, %bboundsplat : tensor<4xi32, #ttg.slice<{dim = 1, parent = #sK}>>
    %bc1 = tt.expand_dims %batchcmp {axis = 1 : i32} : tensor<4xi1, #ttg.slice<{dim = 1, parent = #sK}>> -> tensor<4x1xi1, #sK>
    %bc2 = tt.expand_dims %bc1 {axis = 2 : i32} : tensor<4x1xi1, #sK> -> tensor<4x1x1xi1, #dot0>
    %batchb = tt.broadcast %bc2 : tensor<4x1x1xi1, #dot0> -> tensor<4x64x1xi1, #dot0>
    %and = arith.andi %rowb, %batchb : tensor<4x64x1xi1, #dot0>
    %mask = tt.broadcast %and : tensor<4x64x1xi1, #dot0> -> tensor<4x64x32xi1, #dot0>
    // CHECK: llvm.intr.smax
    // CHECK: llvm.icmp "sgt"
    // CHECK: llvm.and {{.*}} : i1
    // CHECK: llvm.select
    // CHECK: triton_gen.2Dblockprefetch
    ttig.prefetch %ptr, %mask {cache = 1 : i32, evict = 1 : i32, isVolatile = false, ttig.block_io = "row_major"} : tensor<4x64x32x!tt.ptr<f16>, #dot0>
    tt.return
  }
}

// -----

// COM: Rank-3 prefetch whose mask is a bare row-limit (no batch predicate).
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4, 2], repCluster = [1, 1, 1], A = [1, 8, 16], B = [1, 16, 16], C = [1, 8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#sK = #ttg.slice<{dim = 2, parent = #dot0}>
#sM = #ttg.slice<{dim = 1, parent = #dot0}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @prefetch_row_limit_rank3
  tt.func @prefetch_row_limit_rank3(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %row_base: i32 {tt.divisibility = 16 : i32}, %bound: i32 {tt.divisibility = 16 : i32}) {
    %c32 = arith.constant dense<32> : tensor<1x64x1xi32, #dot0>
    // ptr[b, i, j] = arg0 + i * 32 + j; batch addressing is irrelevant here,
    // so every batch reads the same 64x32 slice.
    %rb = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #sK}>>
    %rm = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #sK}>>
    %ema = tt.expand_dims %rm {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #sK}>> -> tensor<1x64xi32, #sK>
    %em = tt.expand_dims %ema {axis = 2 : i32} : tensor<1x64xi32, #sK> -> tensor<1x64x1xi32, #dot0>
    %om = arith.muli %em, %c32 : tensor<1x64x1xi32, #dot0>
    %rk = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #sM}>>
    %eka = tt.expand_dims %rk {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #sM}>> -> tensor<1x32xi32, #sM>
    %ek = tt.expand_dims %eka {axis = 1 : i32} : tensor<1x32xi32, #sM> -> tensor<1x1x32xi32, #dot0>
    %bm = tt.broadcast %om : tensor<1x64x1xi32, #dot0> -> tensor<4x64x32xi32, #dot0>
    %bk = tt.broadcast %ek : tensor<1x1x32xi32, #dot0> -> tensor<4x64x32xi32, #dot0>
    %off = arith.addi %bm, %bk : tensor<4x64x32xi32, #dot0>
    %ps = tt.splat %arg0 : !tt.ptr<f16> -> tensor<4x64x32x!tt.ptr<f16>, #dot0>
    %ptr = tt.addptr %ps, %off : tensor<4x64x32x!tt.ptr<f16>, #dot0>, tensor<4x64x32xi32, #dot0>
    // rows = row_base + arange(0, 64) < bound; batch = arange(0, 4) < batch_bound.
    %rbase = tt.splat %row_base : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #sK}>>
    %rows = arith.addi %rbase, %rm : tensor<64xi32, #ttg.slice<{dim = 0, parent = #sK}>>
    %boundsplat = tt.splat %bound : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #sK}>>
    %rowcmp = arith.cmpi slt, %rows, %boundsplat : tensor<64xi32, #ttg.slice<{dim = 0, parent = #sK}>>
    %rc1 = tt.expand_dims %rowcmp {axis = 0 : i32} : tensor<64xi1, #ttg.slice<{dim = 0, parent = #sK}>> -> tensor<1x64xi1, #sK>
    %rc2 = tt.expand_dims %rc1 {axis = 2 : i32} : tensor<1x64xi1, #sK> -> tensor<1x64x1xi1, #dot0>
    %mask = tt.broadcast %rc2 : tensor<1x64x1xi1, #dot0> -> tensor<4x64x32xi1, #dot0>
    // CHECK: llvm.intr.smax
    // CHECK: llvm.icmp "sgt"
    // CHECK-NOT: llvm.and {{.*}} : i1
    // CHECK: llvm.select
    // CHECK: triton_gen.2Dblockprefetch
    ttig.prefetch %ptr, %mask {cache = 1 : i32, evict = 1 : i32, isVolatile = false, ttig.block_io = "row_major"} : tensor<4x64x32x!tt.ptr<f16>, #dot0>
    tt.return
  }
}
