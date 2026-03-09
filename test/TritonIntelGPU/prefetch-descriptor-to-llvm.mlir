// RUN: triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm | FileCheck %s

// COM: Test row-major descriptor prefetch lowering.
// COM: Tensor descriptor struct layout: { shape[rank], stride[rank], base }
// COM: For rank-2: !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
// COM:   [0]: shape0, [1]: shape1, [2]: stride0, [3]: stride1, [4]: base_ptr

module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
// CHECK-LABEL:   llvm.func spir_kernelcc @prefetch_descriptor_row_major(
// CHECK-SAME:                                                           %[[BASE:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr<1>,
// CHECK-SAME:                                                           %[[BASE_HEIGHT:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64,
// CHECK-SAME:                                                           %[[BASE_WIDTH:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64,
// CHECK-SAME:                                                           %[[ROW_STRIDE:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64,
// CHECK-SAME:                                                           %[[PTR_1:.*]]: !llvm.ptr<1>) attributes {intel_reqd_sub_group_size = 16 : i32, reqd_work_group_size = array<i32: 128, 1, 1>} {
  tt.func public @prefetch_descriptor_row_major(%arg0: !tt.ptr<f16>, %arg2: i64, %arg4: i64, %arg5: i64) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32

    // COM: Create a tensor descriptor for a 16x32xf16 tile.
    // COM: make_tensor_descriptor takes (base, [shape0, shape1], [stride0, stride1]).
    %desc = tt.make_tensor_descriptor %arg0, [%c16_i32, %c32_i32], [%arg5, %c1_i64] : <f16>, <tensor<16x32xf16>>

    // COM: The descriptor struct is: (i64, i64, i64, i64, ptr<1>)
    // CHECK:     %[[DESC:.*]] = llvm.insertvalue %{{.*}}, %{{.*}}[4] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>

    // COM: Unpack descriptor fields: shape0, shape1, stride0, stride1, base.
    // CHECK:     %[[SHAPE0:.*]] = llvm.extractvalue %[[DESC]][0] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[SHAPE1:.*]] = llvm.extractvalue %[[DESC]][1] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[STRIDE0:.*]] = llvm.extractvalue %[[DESC]][2] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[STRIDE1:.*]] = llvm.extractvalue %[[DESC]][3] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[BASE_PTR:.*]] = llvm.extractvalue %[[DESC]][4] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>

    // COM: For row-major: baseWidth = shape1 (cols), baseHeight = shape0 (rows), rowStride = stride0.
    // COM: Convert baseWidth to bytes: shape1 * 2 (f16 = 2 bytes).
    // CHECK:     %[[CST_2:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK:     %[[WIDTH_BYTES:.*]] = llvm.mul %[[SHAPE1]], %[[CST_2]] : i64
    // CHECK:     %[[BASE_WIDTH_I32:.*]] = llvm.trunc %[[WIDTH_BYTES]] : i64 to i32
    // CHECK:     %[[BASE_HEIGHT_I32:.*]] = llvm.trunc %[[SHAPE0]] : i64 to i32

    // COM: Convert rowStride (stride0) to bytes.
    // CHECK:     %[[CST_2_:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK:     %[[STRIDE_BYTES:.*]] = llvm.mul %[[STRIDE0]], %[[CST_2_]] : i64
    // CHECK:     %[[PITCH:.*]] = llvm.trunc %[[STRIDE_BYTES]] : i64 to i32

    // COM: Get the sub-group (warp) ID for cooperative prefetching.
    // CHECK:     %[[SUB_GROUP_ID_RAW:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id() {no_unwind, will_return} : () -> i32
    // CHECK:     %[[SUB_GROUP_ID_EXT:.*]] = llvm.zext %[[SUB_GROUP_ID_RAW]] : i32 to i64
    // CHECK:     %[[SUB_GROUP_ID:.*]] = llvm.trunc %[[SUB_GROUP_ID_EXT]] : i64 to i32

    // COM: Compute the tile offset and add the index bases (offsetBaseX=c0, offsetBaseY=c0).
    // CHECK:     %[[OFFSET_X:.*]] = llvm.add {{.*}}, %{{.*}} : i32
    // CHECK:     %[[OFFSET_Y:.*]] = llvm.add {{.*}}, %{{.*}} : i32

    // COM: 16x32xf16 with 8 warps: tile_height=2, tile_width=32 → vBlocks=2, tile_width=16.
    // CHECK:     triton_gen.2Dblockprefetch %[[BASE_PTR]], %[[BASE_WIDTH_I32]], %[[BASE_HEIGHT_I32]], %[[PITCH]], %[[OFFSET_X]], %[[OFFSET_Y]] {elem_size_in_bits = 16, tile_width = 16, tile_height = 2, v_blocks = 2, cache_control = L1C_L3C}
    ttig.descriptor_prefetch %desc[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<16x32xf16>>

    tt.return
  }
}

// -----

// COM: Test that a descriptor prefetch without the block_io attribute is erased (no prefetch generated).
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
// CHECK-LABEL:   llvm.func spir_kernelcc @prefetch_descriptor_no_block_io(
  tt.func public @prefetch_descriptor_no_block_io(%arg0: !tt.ptr<f16>, %arg5: i64) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32

    %desc = tt.make_tensor_descriptor %arg0, [%c16_i32, %c32_i32], [%arg5, %c1_i64] : <f16>, <tensor<16x32xf16>>

    // COM: Without ttig.block_io, the prefetch should be erased with no 2D block prefetch generated.
    // CHECK-NOT: triton_gen.2Dblockprefetch
    ttig.descriptor_prefetch %desc[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<16x32xf16>>
    tt.return
  }
}

// -----

// COM: Test descriptor prefetch with support_prefetch_256b module attribute.
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_prefetch_256b} {
// CHECK-LABEL:   llvm.func spir_kernelcc @prefetch_descriptor_256b(
// CHECK-SAME:                                                      %[[BASE:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr<1>,
// CHECK-SAME:                                                      %[[BASE_HEIGHT:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64,
// CHECK-SAME:                                                      %[[BASE_WIDTH:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64,
// CHECK-SAME:                                                      %[[ROW_STRIDE:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64,
// CHECK-SAME:                                                      %[[PTR_1:.*]]: !llvm.ptr<1>) attributes {intel_reqd_sub_group_size = 16 : i32, reqd_work_group_size = array<i32: 128, 1, 1>} {
  tt.func public @prefetch_descriptor_256b(%arg0: !tt.ptr<f16>, %arg2: i64, %arg4: i64, %arg5: i64) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c16_i32 = arith.constant 16 : i32
    %c256_i32 = arith.constant 256 : i32

    // COM: Create a tensor descriptor for a 16x256xf16 tile (256 cols → 512 bytes/row → uses 256B prefetch path).
    %desc = tt.make_tensor_descriptor %arg0, [%c16_i32, %c256_i32], [%arg5, %c1_i64] : <f16>, <tensor<16x256xf16>>

    // CHECK:     %[[DESC:.*]] = llvm.insertvalue %{{.*}}, %{{.*}}[4] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[SHAPE0:.*]] = llvm.extractvalue %[[DESC]][0] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[SHAPE1:.*]] = llvm.extractvalue %[[DESC]][1] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[STRIDE0:.*]] = llvm.extractvalue %[[DESC]][2] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[STRIDE1:.*]] = llvm.extractvalue %[[DESC]][3] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[BASE_PTR:.*]] = llvm.extractvalue %[[DESC]][4] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[CST_2:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK:     %[[WIDTH_BYTES:.*]] = llvm.mul %[[SHAPE1]], %[[CST_2]] : i64
    // CHECK:     %[[BASE_WIDTH_I32:.*]] = llvm.trunc %[[WIDTH_BYTES]] : i64 to i32
    // CHECK:     %[[BASE_HEIGHT_I32:.*]] = llvm.trunc %[[SHAPE0]] : i64 to i32
    // CHECK:     %[[CST_2_:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK:     %[[STRIDE_BYTES:.*]] = llvm.mul %[[STRIDE0]], %[[CST_2_]] : i64
    // CHECK:     %[[PITCH:.*]] = llvm.trunc %[[STRIDE_BYTES]] : i64 to i32
    // CHECK:     %[[SUB_GROUP_ID_RAW:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id() {no_unwind, will_return} : () -> i32
    // CHECK:     %[[SUB_GROUP_ID_EXT:.*]] = llvm.zext %[[SUB_GROUP_ID_RAW]] : i32 to i64
    // CHECK:     %[[SUB_GROUP_ID:.*]] = llvm.trunc %[[SUB_GROUP_ID_EXT]] : i64 to i32
    // CHECK:     %[[OFFSET_X:.*]] = llvm.add {{.*}}, %{{.*}} : i32
    // CHECK:     %[[OFFSET_Y:.*]] = llvm.add {{.*}}, %{{.*}} : i32

    // COM: 16x256xf16 with 256B support → tile_width=128, tile_height=4, v_blocks=1.
    // CHECK:     triton_gen.2Dblockprefetch %[[BASE_PTR]], %[[BASE_WIDTH_I32]], %[[BASE_HEIGHT_I32]], %[[PITCH]], %[[OFFSET_X]], %[[OFFSET_Y]] {elem_size_in_bits = 16, tile_width = 128, tile_height = 4, v_blocks = 1, cache_control = L1C_L3C}
    ttig.descriptor_prefetch %desc[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<16x256xf16>>

    tt.return
  }
}

// -----

// COM: Test 256B prefetch with fallback to 64 bytes per row (128 bytes/row < 256B threshold).
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_prefetch_256b} {
// CHECK-LABEL:   llvm.func spir_kernelcc @prefetch_descriptor_256b_fallback(
  tt.func public @prefetch_descriptor_256b_fallback(%arg0: !tt.ptr<f16>, %arg5: i64) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c16_i32 = arith.constant 16 : i32
    %c64_i32 = arith.constant 64 : i32

    // COM: 16x64xf16 → 128 bytes/row. With support_prefetch_256b, this falls back to 64 bytes prefetch.
    %desc = tt.make_tensor_descriptor %arg0, [%c16_i32, %c64_i32], [%arg5, %c1_i64] : <f16>, <tensor<16x64xf16>>

    // CHECK:     %[[DESC:.*]] = llvm.insertvalue %{{.*}}, %{{.*}}[4] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[SHAPE0:.*]] = llvm.extractvalue %[[DESC]][0] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[SHAPE1:.*]] = llvm.extractvalue %[[DESC]][1] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[STRIDE0:.*]] = llvm.extractvalue %[[DESC]][2] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[STRIDE1:.*]] = llvm.extractvalue %[[DESC]][3] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[BASE_PTR:.*]] = llvm.extractvalue %[[DESC]][4] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[CST_2:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK:     %[[WIDTH_BYTES:.*]] = llvm.mul %[[SHAPE1]], %[[CST_2]] : i64
    // CHECK:     %[[BASE_WIDTH_I32:.*]] = llvm.trunc %[[WIDTH_BYTES]] : i64 to i32
    // CHECK:     %[[BASE_HEIGHT_I32:.*]] = llvm.trunc %[[SHAPE0]] : i64 to i32
    // CHECK:     %[[CST_2_:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK:     %[[STRIDE_BYTES:.*]] = llvm.mul %[[STRIDE0]], %[[CST_2_]] : i64
    // CHECK:     %[[PITCH:.*]] = llvm.trunc %[[STRIDE_BYTES]] : i64 to i32
    // CHECK:     %[[SUB_GROUP_ID_RAW:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id() {no_unwind, will_return} : () -> i32
    // CHECK:     %[[SUB_GROUP_ID_EXT:.*]] = llvm.zext %[[SUB_GROUP_ID_RAW]] : i32 to i64
    // CHECK:     %[[SUB_GROUP_ID:.*]] = llvm.trunc %[[SUB_GROUP_ID_EXT]] : i64 to i32
    // CHECK:     %[[OFFSET_X:.*]] = llvm.add {{.*}}, %{{.*}} : i32
    // CHECK:     %[[OFFSET_Y:.*]] = llvm.add {{.*}}, %{{.*}} : i32

    // COM: 128 bytes per row falls back to 64 bytes prefetch → tile_width=16, tile_height=4, v_blocks=2.
    // CHECK:     triton_gen.2Dblockprefetch %[[BASE_PTR]], %[[BASE_WIDTH_I32]], %[[BASE_HEIGHT_I32]], %[[PITCH]], %[[OFFSET_X]], %[[OFFSET_Y]] {elem_size_in_bits = 16, tile_width = 16, tile_height = 4, v_blocks = 2, cache_control = L1C_L3C}
    ttig.descriptor_prefetch %desc[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<16x64xf16>>

    tt.return
  }
}
