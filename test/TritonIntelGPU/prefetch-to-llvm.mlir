// RUN: triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm | FileCheck %s

module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
// CHECK-LABEL:   llvm.func spir_kernelcc @prefetch_block_ptr(
// CHECK-SAME:                                                %[[BASE:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr<1>,
// CHECK-SAME:                                                %[[BASE_HEIGHT:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64,
// CHECK-SAME:                                                %[[BASE_WIDTH:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64,
// CHECK-SAME:                                                %[[ROW_STRIDE:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64) attributes {intel_reqd_sub_group_size = 16 : i32, triton_gen.max_work_group_size = array<i32: 128, 1, 1>} {
  tt.func public @prefetch_block_ptr(%arg0: !tt.ptr<f16>, %arg2: i64, %arg4: i64, %arg5: i64) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64

    // CHECK-DAG: %[[VAL_5:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-DAG: %[[VAL_6:.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK:     %[[VAL_7:.*]] = llvm.mlir.undef : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[VAL_8:.*]] = llvm.insertvalue %[[VAL_5]], %[[VAL_7]][0] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[VAL_9:.*]] = llvm.insertvalue %[[VAL_5]], %[[VAL_8]][1] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[VAL_10:.*]] = llvm.insertvalue %[[BASE_HEIGHT]], %[[VAL_9]][2] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[VAL_11:.*]] = llvm.insertvalue %[[BASE_WIDTH]], %[[VAL_10]][3] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[VAL_12:.*]] = llvm.insertvalue %[[ROW_STRIDE]], %[[VAL_11]][4] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[VAL_13:.*]] = llvm.insertvalue %[[VAL_6]], %[[VAL_12]][5] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[BLOCK_POINTER:.*]] = llvm.insertvalue %[[BASE]], %[[VAL_13]][6] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[SUB_GROUP_ID_RAW:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id() {no_unwind, will_return} : () -> i32
    // CHECK:     %[[SUB_GROUP_ID_EXT:.*]] = llvm.zext %[[SUB_GROUP_ID_RAW]] : i32 to i64
    // CHECK:     %[[SUB_GROUP_ID:.*]] = llvm.trunc %[[SUB_GROUP_ID_EXT]] : i64 to i32
    // CHECK:     %[[VAL_18:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:     %[[VAL_19:.*]] = llvm.urem %[[SUB_GROUP_ID]], %[[VAL_18]] : i32
    // CHECK:     %[[VAL_20:.*]] = llvm.udiv %[[SUB_GROUP_ID]], %[[VAL_18]] : i32
    // CHECK:     %[[CST_8:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK:     %[[VAL_22:.*]] = llvm.urem %[[VAL_20]], %[[CST_8]] : i32
    // CHECK:     %[[VAL_23:.*]] = llvm.udiv %[[VAL_20]], %[[CST_8]] : i32
    // CHECK:     %[[OFFSET_0:.*]] = llvm.extractvalue %[[BLOCK_POINTER]][0] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[OFFSET_1:.*]] = llvm.extractvalue %[[BLOCK_POINTER]][1] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[WIDTH_i64:.*]] = llvm.extractvalue %[[BLOCK_POINTER]][2] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[HEIGHT_i64:.*]] = llvm.extractvalue %[[BLOCK_POINTER]][3] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[ROW_STRIDE_i64:.*]] = llvm.extractvalue %[[BLOCK_POINTER]][4] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[COL_STRIDE_i64:.*]] = llvm.extractvalue %[[BLOCK_POINTER]][5] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[BASE_:.*]] = llvm.extractvalue %[[BLOCK_POINTER]][6] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[CST_2:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK:     %[[VAL_21:.*]] = llvm.mul %[[HEIGHT_i64]], %[[CST_2]] : i64
    // CHECK:     %[[ROW_MAJOR_BASE_WIDTH:.*]] = llvm.trunc %[[VAL_21]] : i64 to i32
    // CHECK:     %[[ROW_MAJOR_BASE_HEIGHT:.*]] = llvm.trunc %[[WIDTH_i64]] : i64 to i32
    // CHECK:     %[[CST_2:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK:     %[[VAL_24:.*]] = llvm.mul %[[ROW_STRIDE_i64]], %[[CST_2]] : i64
    // CHECK:     %[[ROW_MAJOR_PITCH:.*]] = llvm.trunc %[[VAL_24]] : i64 to i32
    // CHECK:     %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:     %[[CST_32:.*]] = llvm.mlir.constant(32 : i32) : i32
    // CHECK:     %[[VAL_26:.*]] = llvm.mul %[[VAL_19]], %[[CST_32]] : i32
    // CHECK:     %[[VAL_27:.*]] = llvm.add %[[VAL_26]], %[[CST_0]] : i32
    // CHECK:     %[[CST_32:.*]] = llvm.mlir.constant(32 : i32) : i32
    // CHECK:     %[[VAL_28:.*]] = llvm.urem %[[VAL_27]], %[[CST_32]] : i32
    // CHECK:     %[[VAL_29:.*]] = llvm.add %[[VAL_28]], %[[OFFSET_1]] : i32
    // CHECK:     %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:     %[[CST_2:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK:     %[[VAL_30:.*]] = llvm.mul %[[VAL_22]], %[[CST_2]] : i32
    // CHECK:     %[[VAL_31:.*]] = llvm.add %[[VAL_30]], %[[CST_0]] : i32
    // CHECK:     %[[CST_16:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:     %[[VAL_32:.*]] = llvm.urem %[[VAL_31]], %[[CST_16]] : i32
    // CHECK:     %[[VAL_33:.*]] = llvm.add %[[VAL_32]], %[[OFFSET_0]] : i32
    // CHECK:     %[[ROW_MAJOR_OFFSET_Y:.*]] = llvm.trunc %[[VAL_33]] : i32 to i32
    // CHECK:     %[[ROW_MAJOR_OFFSET_X:.*]] = llvm.trunc %[[VAL_29]] : i32 to i32
    // CHECK:     triton_gen.2Dblockprefetch %[[BASE_]], %[[ROW_MAJOR_BASE_WIDTH]], %[[ROW_MAJOR_BASE_HEIGHT]], %[[ROW_MAJOR_PITCH]], %[[ROW_MAJOR_OFFSET_X]], %[[ROW_MAJOR_OFFSET_Y]] {elem_size_in_bits = 16, tile_width = 16, tile_height = 2, v_blocks = 2, cache_control = L1C_L3C}
    %rowMajorPtr = tt.make_tensor_ptr %arg0, [%arg2, %arg4], [%arg5, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x32xf16>>
    ttig.prefetch %rowMajorPtr {cache = 1 : i32, evict = 1 : i32, isVolatile = false, ttig.block_io = "row_major"} : !tt.ptr<tensor<16x32xf16>>

    // CHECK:     %[[VAL_7:.*]] = llvm.mlir.undef : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[VAL_8:.*]] = llvm.insertvalue %[[VAL_5]], %[[VAL_7]][0] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[VAL_9:.*]] = llvm.insertvalue %[[VAL_5]], %[[VAL_8]][1] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[VAL_10:.*]] = llvm.insertvalue %[[BASE_WIDTH]], %[[VAL_9]][2] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[VAL_11:.*]] = llvm.insertvalue %[[BASE_HEIGHT]], %[[VAL_10]][3] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[VAL_12:.*]] = llvm.insertvalue %[[VAL_6]], %[[VAL_11]][4] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[VAL_13:.*]] = llvm.insertvalue %[[ROW_STRIDE]], %[[VAL_12]][5] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[BLOCK_POINTER:.*]] = llvm.insertvalue %[[BASE]], %[[VAL_13]][6] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[SUB_GROUP_ID_RAW:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id() {no_unwind, will_return} : () -> i32
    // CHECK:     %[[SUB_GROUP_ID_EXT:.*]] = llvm.zext %[[SUB_GROUP_ID_RAW]] : i32 to i64
    // CHECK:     %[[SUB_GROUP_ID:.*]] = llvm.trunc %[[SUB_GROUP_ID_EXT]] : i64 to i32
    // CHECK:     %[[VAL_18:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK:     %[[VAL_19:.*]] = llvm.urem %[[SUB_GROUP_ID]], %[[VAL_18]] : i32
    // CHECK:     %[[VAL_20:.*]] = llvm.udiv %[[SUB_GROUP_ID]], %[[VAL_18]] : i32
    // CHECK:     %[[CST_8:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK:     %[[VAL_22:.*]] = llvm.urem %[[VAL_20]], %[[CST_8]] : i32
    // CHECK:     %[[VAL_23:.*]] = llvm.udiv %[[VAL_20]], %[[CST_8]] : i32
    // CHECK:     %[[OFFSET_0:.*]] = llvm.extractvalue %[[BLOCK_POINTER]][0] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[OFFSET_1:.*]] = llvm.extractvalue %[[BLOCK_POINTER]][1] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[WIDTH_i64:.*]] = llvm.extractvalue %[[BLOCK_POINTER]][2] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[HEIGHT_i64:.*]] = llvm.extractvalue %[[BLOCK_POINTER]][3] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[ROW_STRIDE_i64:.*]] = llvm.extractvalue %[[BLOCK_POINTER]][4] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[COL_STRIDE_i64:.*]] = llvm.extractvalue %[[BLOCK_POINTER]][5] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[BASE_:.*]] = llvm.extractvalue %[[BLOCK_POINTER]][6] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[CST_2:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK:     %[[VAL_21:.*]] = llvm.mul %[[WIDTH_i64]], %[[CST_2]] : i64
    // CHECK:     %[[COL_MAJOR_BASE_WIDTH:.*]] = llvm.trunc %[[VAL_21]] : i64 to i32
    // CHECK:     %[[COL_MAJOR_BASE_HEIGHT:.*]] = llvm.trunc %[[HEIGHT_i64]] : i64 to i32
    // CHECK:     %[[CST_2:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK:     %[[VAL_24:.*]] = llvm.mul %[[COL_STRIDE_i64]], %[[CST_2]] : i64
    // CHECK:     %[[COL_MAJOR_PITCH:.*]] = llvm.trunc %[[VAL_24]] : i64 to i32
    // CHECK:     %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:     %[[CST_32:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:     %[[VAL_26:.*]] = llvm.mul %[[VAL_19]], %[[CST_32]] : i32
    // CHECK:     %[[VAL_27:.*]] = llvm.add %[[VAL_26]], %[[CST_0]] : i32
    // CHECK:     %[[CST_32:.*]] = llvm.mlir.constant(32 : i32) : i32
    // CHECK:     %[[VAL_28:.*]] = llvm.urem %[[VAL_27]], %[[CST_32]] : i32
    // CHECK:     %[[VAL_29:.*]] = llvm.add %[[VAL_28]], %[[OFFSET_1]] : i32
    // CHECK:     %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:     %[[CST_2:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK:     %[[VAL_30:.*]] = llvm.mul %[[VAL_22]], %[[CST_2]] : i32
    // CHECK:     %[[VAL_31:.*]] = llvm.add %[[VAL_30]], %[[CST_0]] : i32
    // CHECK:     %[[CST_16:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:     %[[VAL_32:.*]] = llvm.urem %[[VAL_31]], %[[CST_16]] : i32
    // CHECK:     %[[VAL_33:.*]] = llvm.add %[[VAL_32]], %[[OFFSET_0]] : i32
    // CHECK:     %[[COL_MAJOR_OFFSET_Y:.*]] = llvm.trunc %[[VAL_33]] : i32 to i32
    // CHECK:     %[[COL_MAJOR_OFFSET_X:.*]] = llvm.trunc %[[VAL_29]] : i32 to i32
    // CHECK:     triton_gen.2Dblockprefetch %[[BASE_]], %[[COL_MAJOR_BASE_WIDTH]], %[[COL_MAJOR_BASE_HEIGHT]], %[[COL_MAJOR_PITCH]], %[[COL_MAJOR_OFFSET_X]], %[[COL_MAJOR_OFFSET_Y]] {elem_size_in_bits = 16, tile_width = 16, tile_height = 4, v_blocks = 1, cache_control = L1C_L3C}
    %columnMajorPtr = tt.make_tensor_ptr %arg0, [%arg4, %arg2], [%c1_i64, %arg5], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<32x16xf16>>
    ttig.prefetch %columnMajorPtr {cache = 1 : i32, evict = 1 : i32, isVolatile = false, ttig.block_io = "column_major"} : !tt.ptr<tensor<32x16xf16>>

    // COM: The memory is not structured densely. Not to prefetch it to the cache.
    // CHECK-NOT: triton_gen.2Dblockprefetch
    %nonContiguousPtr = tt.make_tensor_ptr %arg0, [%arg4, %arg2], [%arg5, %arg5], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x16xf16>>
    ttig.prefetch %nonContiguousPtr {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<32x16xf16>>
    tt.return
  }
}

// -----

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @prefetch_tensor_of_pointers
  tt.func public @prefetch_tensor_of_pointers(%tensor_of_ptr: tensor<64x32x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>) {
    // CHECK: %[[ADDR_0:.*]] = llvm.extractvalue {{.*}}[0] : !llvm.struct<(ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>)>
    // CHECK: %[[ADDR_1:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>)>
    // CHECK: %[[ADDR_16:.*]] = llvm.extractvalue {{.*}}[16] : !llvm.struct<(ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>)>
    // CHECK: %[[ADDR_32:.*]] = llvm.extractvalue {{.*}}[32] : !llvm.struct<(ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>)>
    // CHECK: %[[ADDR_48:.*]] = llvm.extractvalue {{.*}}[48] : !llvm.struct<(ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>)>
    // CHECK: %[[BASE_WIDTH:.*]] = llvm.mlir.constant(64 : i32) : i32
    // CHECK: %[[VAL_13:.*]] = llvm.ptrtoint %[[ADDR_0]] : !llvm.ptr<1> to i64
    // CHECK: %[[VAL_14:.*]] = llvm.ptrtoint %[[ADDR_1]] : !llvm.ptr<1> to i64
    // CHECK: %[[PITCH:.*]] = llvm.sub %[[VAL_14]], %[[VAL_13]] : i64
    // CHECK: %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[UNIFIED_PITCH:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflelj(%[[PITCH]], %[[CST_0]]) {convergent, no_unwind, will_return} : (i64, i32) -> i64
    // CHECK: %[[UNIFIED_PITCH_I32:.*]] = llvm.trunc %[[UNIFIED_PITCH]] : i64 to i32
    // CHECK: %[[PITCH_IN_BYTES_I32:.*]] = llvm.intr.umax(%[[UNIFIED_PITCH_I32]], %[[BASE_WIDTH]]) : (i32, i32) -> i32
    // CHECK-DAG: %[[BASE_HEIGHT:.*]] = llvm.mlir.constant(8 : i32) : i32
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
    // CHECK: triton_gen.2Dblockprefetch %[[VAL_26]], %[[BASE_WIDTH]], %[[BASE_HEIGHT]], %[[PITCH_IN_BYTES_I32]], %[[CST_0_]], %[[OFFSET_Y]] {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 2, cache_control = L1C_L3C}

    // CHECK: %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[VAL_29:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflecj(%{{.*}}, %[[CST_0]]) {convergent, no_unwind, will_return} : (i8, i32) -> i8
    // CHECK: %[[VAL_30:.*]] = llvm.trunc %[[VAL_29]] : i8 to i1
    // CHECK: %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[VAL_31:.*]] = llvm.select %[[VAL_30]], %[[CST_0]], %[[BASE_HEIGHT]] : i1, i32
    // CHECK: %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[VAL_32:.*]] = llvm.ptrtoint %[[ADDR_16]] : !llvm.ptr<1> to i64
    // CHECK: %[[VAL_33:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflelj(%[[VAL_32]], %[[CST_0]]) {convergent, no_unwind, will_return} : (i64, i32) -> i64
    // CHECK: %[[VAL_34:.*]] = llvm.inttoptr %[[VAL_33]] : i64 to !llvm.ptr<1>
    // CHECK: triton_gen.2Dblockprefetch %[[VAL_34]], %[[BASE_WIDTH]], %[[BASE_HEIGHT]], %[[PITCH_IN_BYTES_I32]], %[[CST_0_]], %[[VAL_31]] {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 2, cache_control = L1C_L3C}

    // CHECK: %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[VAL_36:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflecj(%{{.*}}, %[[CST_0]]) {convergent, no_unwind, will_return} : (i8, i32) -> i8
    // CHECK: %[[VAL_37:.*]] = llvm.trunc %[[VAL_36]] : i8 to i1
    // CHECK: %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[VAL_38:.*]] = llvm.select %[[VAL_37]], %[[CST_0]], %[[BASE_HEIGHT]] : i1, i32
    // CHECK: %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[VAL_39:.*]] = llvm.ptrtoint %[[ADDR_32]] : !llvm.ptr<1> to i64
    // CHECK: %[[VAL_40:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflelj(%[[VAL_39]], %[[CST_0]]) {convergent, no_unwind, will_return} : (i64, i32) -> i64
    // CHECK: %[[VAL_41:.*]] = llvm.inttoptr %[[VAL_40]] : i64 to !llvm.ptr<1>
    // CHECK: triton_gen.2Dblockprefetch %[[VAL_41]], %[[BASE_WIDTH]], %[[BASE_HEIGHT]], %[[PITCH_IN_BYTES_I32]], %[[CST_0_]], %[[VAL_38]] {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 2, cache_control = L1C_L3C}

    // CHECK: %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[VAL_43:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflecj(%{{.*}}, %[[CST_0]]) {convergent, no_unwind, will_return} : (i8, i32) -> i8
    // CHECK: %[[VAL_44:.*]] = llvm.trunc %[[VAL_43]] : i8 to i1
    // CHECK: %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[VAL_45:.*]] = llvm.select %[[VAL_44]], %[[CST_0]], %[[BASE_HEIGHT]] : i1, i32
    // CHECK: %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[VAL_46:.*]] = llvm.ptrtoint %[[ADDR_48]] : !llvm.ptr<1> to i64
    // CHECK: %[[VAL_47:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflelj(%[[VAL_46]], %[[CST_0]]) {convergent, no_unwind, will_return} : (i64, i32) -> i64
    // CHECK: %[[VAL_48:.*]] = llvm.inttoptr %[[VAL_47]] : i64 to !llvm.ptr<1>
    // CHECK: triton_gen.2Dblockprefetch %[[VAL_48]], %[[BASE_WIDTH]], %[[BASE_HEIGHT]], %[[PITCH_IN_BYTES_I32]], %[[CST_0_]], %[[VAL_45]] {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 2, cache_control = L1C_L3C}

    %mask_tensor = arith.constant dense<1> : tensor<64x32xi1, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    ttig.prefetch %tensor_of_ptr, %mask_tensor {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 1, 1>, ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>

    // CHECK-COUNT-4: triton_gen.2Dblockprefetch {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 2, cache_control = L1C_L3C}

    ttig.prefetch %tensor_of_ptr {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 1, 1>, ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>

    tt.return
  }
}

// -----

// COM: Check that pitch is a constant calculated by AxisInfo analysis, instead of calculating dynamically.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @prefetch_tensor_of_pointers
  tt.func public @prefetch_tensor_of_pointers(%arg0: i32, %arg1: !tt.ptr<bf16>) {
    %cst_0 = arith.constant dense<512> : tensor<128x1xi32, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    %cst_1 = arith.constant dense<512> : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>}>>
    %c128_i32 = arith.constant 128 : i32
    %0 = arith.muli %arg0, %c128_i32 : i32
    %1 = tt.splat %0 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>}>>
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>}>>
    %3 = arith.addi %1, %2 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>}>>
    %4 = arith.remsi %3, %cst_1 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>}>>
    %5 = tt.expand_dims %4 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>}>> -> tensor<128x1xi32, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    %6 = arith.muli %5, %cst_0 : tensor<128x1xi32, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    %7 = tt.broadcast %6 : tensor<128x1xi32, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>> -> tensor<128x64xi32, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    %8 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<128x64x!tt.ptr<bf16>, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    %9 = tt.addptr %8, %7 : tensor<128x64x!tt.ptr<bf16>, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>, tensor<128x64xi32, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>

    // CHECK-DAG: %[[PITCH:.*]] = llvm.mlir.constant(1024 : i32) : i32
    // CHECK-COUNT-4: triton_gen.2Dblockprefetch {{.*}}, %[[PITCH]], {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 2, cache_control = L1C_L3C}
    ttig.prefetch %9 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 1, 1>, ttig.block_io = "row_major"} : tensor<128x64x!tt.ptr<bf16>, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    tt.return
  }
}

// -----

// COM: Currently the prefetch operation in this test cannot be lowered correctly, so we check that the test compiles cleanly and not 2D block prefetch operation gets generated.
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 8], B = [8, 16], C = [32, 16]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_sg_2d_block, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @kernel
  tt.func public @kernel(%arg0 : tensor<128x32x!tt.ptr<f32>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>) {
    // CHECK-NOT: triton_gen.2Dblockprefetch
    ttig.prefetch %arg0 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>, ttig.block_io = "row_major"} : tensor<128x32x!tt.ptr<f32>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    tt.return
  }
}
