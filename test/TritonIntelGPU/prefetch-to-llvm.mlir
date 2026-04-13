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
    // CHECK: %[[PITCH:.*]] = llvm.mlir.constant(128 : i32) : i32
    // CHECK: %[[BASE_HEIGHT:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK: %[[BASE_WIDTH:.*]] = llvm.mlir.constant(64 : i32) : i32
    // CHECK: %[[CST_0_:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: llvm.mlir.constant(0 : i32) : i32

    // CHECK: %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[UNIFIED_MASK:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflecj(%{{.*}}, %[[CST_0]]) {convergent, no_unwind, will_return} : (i8, i32) -> i8
    // CHECK: %[[UNIFIED_MASK_I1:.*]] = llvm.trunc %[[UNIFIED_MASK]] : i8 to i1
    // CHECK: %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[OFFSET_Y:.*]] = llvm.select %[[UNIFIED_MASK_I1]], %[[CST_0]], %[[BASE_HEIGHT]] : i1, i32
    // CHECK: %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[VAL_13:.*]] = llvm.ptrtoint %[[ADDR_0]] : !llvm.ptr<1> to i64
    // CHECK: %[[UNIFIED_BASE:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflelj(%[[VAL_13]], %[[CST_0]]) {convergent, no_unwind, will_return} : (i64, i32) -> i64
    // CHECK: %[[VAL_26:.*]] = llvm.inttoptr %[[UNIFIED_BASE]] : i64 to !llvm.ptr<1>
    // CHECK: triton_gen.2Dblockprefetch %[[VAL_26]], %[[BASE_WIDTH]], %[[BASE_HEIGHT]], %[[PITCH]], %[[CST_0_]], %[[OFFSET_Y]] {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 2, cache_control = L1C_L3C}

    // CHECK: %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[VAL_29:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflecj(%{{.*}}, %[[CST_0]]) {convergent, no_unwind, will_return} : (i8, i32) -> i8
    // CHECK: %[[VAL_30:.*]] = llvm.trunc %[[VAL_29]] : i8 to i1
    // CHECK: %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[VAL_31:.*]] = llvm.select %[[VAL_30]], %[[CST_0]], %[[BASE_HEIGHT]] : i1, i32
    // CHECK: %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[VAL_32:.*]] = llvm.ptrtoint %[[ADDR_16]] : !llvm.ptr<1> to i64
    // CHECK: %[[VAL_33:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflelj(%[[VAL_32]], %[[CST_0]]) {convergent, no_unwind, will_return} : (i64, i32) -> i64
    // CHECK: %[[VAL_34:.*]] = llvm.inttoptr %[[VAL_33]] : i64 to !llvm.ptr<1>
    // CHECK: triton_gen.2Dblockprefetch %[[VAL_34]], %[[BASE_WIDTH]], %[[BASE_HEIGHT]], %[[PITCH]], %[[CST_0_]], %[[VAL_31]] {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 2, cache_control = L1C_L3C}

    // CHECK: %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[VAL_36:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflecj(%{{.*}}, %[[CST_0]]) {convergent, no_unwind, will_return} : (i8, i32) -> i8
    // CHECK: %[[VAL_37:.*]] = llvm.trunc %[[VAL_36]] : i8 to i1
    // CHECK: %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[VAL_38:.*]] = llvm.select %[[VAL_37]], %[[CST_0]], %[[BASE_HEIGHT]] : i1, i32
    // CHECK: %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[VAL_39:.*]] = llvm.ptrtoint %[[ADDR_32]] : !llvm.ptr<1> to i64
    // CHECK: %[[VAL_40:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflelj(%[[VAL_39]], %[[CST_0]]) {convergent, no_unwind, will_return} : (i64, i32) -> i64
    // CHECK: %[[VAL_41:.*]] = llvm.inttoptr %[[VAL_40]] : i64 to !llvm.ptr<1>
    // CHECK: triton_gen.2Dblockprefetch %[[VAL_41]], %[[BASE_WIDTH]], %[[BASE_HEIGHT]], %[[PITCH]], %[[CST_0_]], %[[VAL_38]] {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 2, cache_control = L1C_L3C}

    // CHECK: %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[VAL_43:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflecj(%{{.*}}, %[[CST_0]]) {convergent, no_unwind, will_return} : (i8, i32) -> i8
    // CHECK: %[[VAL_44:.*]] = llvm.trunc %[[VAL_43]] : i8 to i1
    // CHECK: %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[VAL_45:.*]] = llvm.select %[[VAL_44]], %[[CST_0]], %[[BASE_HEIGHT]] : i1, i32
    // CHECK: %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[VAL_46:.*]] = llvm.ptrtoint %[[ADDR_48]] : !llvm.ptr<1> to i64
    // CHECK: %[[VAL_47:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflelj(%[[VAL_46]], %[[CST_0]]) {convergent, no_unwind, will_return} : (i64, i32) -> i64
    // CHECK: %[[VAL_48:.*]] = llvm.inttoptr %[[VAL_47]] : i64 to !llvm.ptr<1>
    // CHECK: triton_gen.2Dblockprefetch %[[VAL_48]], %[[BASE_WIDTH]], %[[BASE_HEIGHT]], %[[PITCH]], %[[CST_0_]], %[[VAL_45]] {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 2, cache_control = L1C_L3C}

    %mask_tensor = arith.constant dense<1> : tensor<64x32xi1, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    ttig.prefetch %tensor_of_ptr, %mask_tensor {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 1, 1>, ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>

    // CHECK-COUNT-4: triton_gen.2Dblockprefetch {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 2, cache_control = L1C_L3C}

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
