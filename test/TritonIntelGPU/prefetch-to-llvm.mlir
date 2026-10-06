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
