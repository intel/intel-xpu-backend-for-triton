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
    %c1_i64 = arith.constant 1 : i64
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %c10_i32 = arith.constant 10 : i32
    %c20_i32 = arith.constant 20 : i32

    // COM: Create a tensor descriptor for a 16x32xf16 tile.
    // COM: make_tensor_descriptor takes (base, [shape0, shape1], [stride0, stride1]).
    %desc = tt.make_tensor_descriptor %arg0, [%c16_i32, %c32_i32], [%arg5, %c1_i64] : <f16>, <tensor<16x32xf16>>

    // COM: Capture constants 10 and 20 which will be used as offset bases.
    // CHECK-DAG:     %[[CST_10:.*]] = llvm.mlir.constant(10 : i32) : i32
    // CHECK-DAG:     %[[CST_20:.*]] = llvm.mlir.constant(20 : i32) : i32

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

    // COM: Compute the tile offsets.
    // COM: For row-major with indices [10, 20]: offsetBaseX gets index1 (20), offsetBaseY gets index0 (10).
    // COM: Column-major would use the opposite: offsetBaseX=index0 (10), offsetBaseY=index1 (20).
    // CHECK:     %[[OFFSET_X:.*]] = llvm.add {{.*}}, %[[CST_20]] : i32
    // CHECK:     %[[OFFSET_Y:.*]] = llvm.add {{.*}}, %[[CST_10]] : i32

    // COM: 16x32xf16 with 8 warps: tile_height=2, tile_width=32 → vBlocks=2, tile_width=16.
    // CHECK:     triton_gen.2Dblockprefetch %[[BASE_PTR]], %[[BASE_WIDTH_I32]], %[[BASE_HEIGHT_I32]], %[[PITCH]], %[[OFFSET_X]], %[[OFFSET_Y]] {elem_size_in_bits = 16, tile_width = 16, tile_height = 2, v_blocks = 2, cache_control = L1C_L3C}
    ttig.descriptor_prefetch %desc[%c10_i32, %c20_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<16x32xf16>>

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
// COM: Base width/height/pitch/offset computation is verified by @prefetch_descriptor_row_major;
// COM: here we only verify the base_ptr extraction and the final prefetch tile parameters.
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_prefetch_256b} {
// CHECK-LABEL:   llvm.func spir_kernelcc @prefetch_descriptor_256b(
  tt.func public @prefetch_descriptor_256b(%arg0: !tt.ptr<f16>, %arg2: i64, %arg4: i64, %arg5: i64) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c16_i32 = arith.constant 16 : i32
    %c256_i32 = arith.constant 256 : i32

    // COM: Create a tensor descriptor for a 16x256xf16 tile (256 cols → 512 bytes/row → uses 256B prefetch path).
    %desc = tt.make_tensor_descriptor %arg0, [%c16_i32, %c256_i32], [%arg5, %c1_i64] : <f16>, <tensor<16x256xf16>>

    // CHECK:     %[[DESC:.*]] = llvm.insertvalue %{{.*}}, %{{.*}}[4] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[BASE_PTR:.*]] = llvm.extractvalue %[[DESC]][4] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>

    // COM: 16x256xf16 with 256B support → tile_width=128, tile_height=4, v_blocks=1.
    // CHECK:     triton_gen.2Dblockprefetch %[[BASE_PTR]], {{.*}} {elem_size_in_bits = 16, tile_width = 128, tile_height = 4, v_blocks = 1, cache_control = L1C_L3C}
    ttig.descriptor_prefetch %desc[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<16x256xf16>>

    tt.return
  }
}

// -----

// COM: Test 256B prefetch with fallback to 64 bytes per row (128 bytes/row < 256B threshold).
// COM: Base width/height/pitch/offset computation is verified by @prefetch_descriptor_row_major;
// COM: here we only verify the base_ptr extraction and the final prefetch tile parameters.
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
    // CHECK:     %[[BASE_PTR:.*]] = llvm.extractvalue %[[DESC]][4] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>

    // COM: 128 bytes per row falls back to 64 bytes prefetch → tile_width=16, tile_height=4, v_blocks=2.
    // CHECK:     triton_gen.2Dblockprefetch %[[BASE_PTR]], {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 4, v_blocks = 2, cache_control = L1C_L3C}
    ttig.descriptor_prefetch %desc[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<16x64xf16>>

    tt.return
  }
}

// -----

// COM: Test column-major descriptor prefetch lowering.
// COM: Uses the same 16x32xf16 descriptor as the row-major test, but the prefetch op
// COM: has ttig.block_io = "column_major". The lowering swaps:
// COM:   baseWidth  = shape0 (rows) * elem_bytes   (row-major uses shape1)
// COM:   baseHeight = shape1 (cols)                 (row-major uses shape0)
// COM:   pitch      = stride1 * elem_bytes          (row-major uses stride0)
// COM: The offsetX/offsetY are also swapped relative to row-major.
// COM: Non-zero distinct offsets (10, 20) are used to verify the offset swap.
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
// CHECK-LABEL:   llvm.func spir_kernelcc @prefetch_descriptor_column_major(
  tt.func public @prefetch_descriptor_column_major(%arg0: !tt.ptr<f16>, %arg5: i64) {
    %c1_i64 = arith.constant 1 : i64
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %c10_i32 = arith.constant 10 : i32
    %c20_i32 = arith.constant 20 : i32

    // COM: Same descriptor as row-major: 16x32xf16, stride0=%arg5 (row stride), stride1=1.
    // COM: The column_major attribute is on the prefetch op, not the descriptor.
    %desc = tt.make_tensor_descriptor %arg0, [%c16_i32, %c32_i32], [%arg5, %c1_i64] : <f16>, <tensor<16x32xf16>>

    // COM: Capture constants 10 and 20 which will be used as offset bases.
    // CHECK-DAG:     %[[CST_10:.*]] = llvm.mlir.constant(10 : i32) : i32
    // CHECK-DAG:     %[[CST_20:.*]] = llvm.mlir.constant(20 : i32) : i32

    // COM: The descriptor struct is: (i64, i64, i64, i64, ptr<1>)
    // CHECK:     %[[DESC:.*]] = llvm.insertvalue %{{.*}}, %{{.*}}[4] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>

    // COM: Unpack descriptor fields: shape0, shape1, stride0, stride1, base.
    // CHECK:     %[[SHAPE0:.*]] = llvm.extractvalue %[[DESC]][0] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[SHAPE1:.*]] = llvm.extractvalue %[[DESC]][1] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[STRIDE0:.*]] = llvm.extractvalue %[[DESC]][2] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[STRIDE1:.*]] = llvm.extractvalue %[[DESC]][3] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[BASE_PTR:.*]] = llvm.extractvalue %[[DESC]][4] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>

    // COM: For column-major: baseWidth = shape0 (rows) * elem_bytes.
    // COM: This is swapped from row-major where baseWidth = shape1 (cols) * elem_bytes.
    // CHECK:     %[[CST_2:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK:     %[[WIDTH_BYTES:.*]] = llvm.mul %[[SHAPE0]], %[[CST_2]] : i64
    // CHECK:     %[[BASE_WIDTH_I32:.*]] = llvm.trunc %[[WIDTH_BYTES]] : i64 to i32
    // CHECK:     %[[BASE_HEIGHT_I32:.*]] = llvm.trunc %[[SHAPE1]] : i64 to i32

    // COM: For column-major, pitch = stride1 * elem_bytes.
    // COM: Row-major uses stride0 (row stride) instead.
    // COM: Here stride1=1 (constant), so pitch = 1 * 2 = 2 bytes.
    // CHECK:     %[[CST_2_:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK:     %[[STRIDE_BYTES:.*]] = llvm.mul %[[STRIDE1]], %[[CST_2_]] : i64
    // CHECK:     %[[PITCH:.*]] = llvm.trunc %[[STRIDE_BYTES]] : i64 to i32

    // COM: Get the sub-group (warp) ID for cooperative prefetching.
    // CHECK:     %[[SUB_GROUP_ID_RAW:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id() {no_unwind, will_return} : () -> i32
    // CHECK:     %[[SUB_GROUP_ID_EXT:.*]] = llvm.zext %[[SUB_GROUP_ID_RAW]] : i32 to i64
    // CHECK:     %[[SUB_GROUP_ID:.*]] = llvm.trunc %[[SUB_GROUP_ID_EXT]] : i64 to i32

    // COM: Compute the tile offsets — swapped relative to row-major.
    // COM: For column-major with indices [10, 20]: offsetBaseX gets index0 (10), offsetBaseY gets index1 (20).
    // COM: Row-major would use the opposite: offsetBaseX=index1 (20), offsetBaseY=index0 (10).
    // CHECK:     %[[OFFSET_X:.*]] = llvm.add {{.*}}, %[[CST_10]] : i32
    // CHECK:     %[[OFFSET_Y:.*]] = llvm.add {{.*}}, %[[CST_20]] : i32

    // COM: 16x32xf16 column-major with 8 warps: after column-major dimension swap the
    // COM: effective tiling shape is 32x16, giving tile_height=4, tile_width=16, v_blocks=1.
    // COM: Compare with row-major (16x32): tile_height=2, tile_width=16, v_blocks=2.
    // CHECK:     triton_gen.2Dblockprefetch %[[BASE_PTR]], %[[BASE_WIDTH_I32]], %[[BASE_HEIGHT_I32]], %[[PITCH]], %[[OFFSET_X]], %[[OFFSET_Y]] {elem_size_in_bits = 16, tile_width = 16, tile_height = 4, v_blocks = 1, cache_control = L1C_L3C}
    ttig.descriptor_prefetch %desc[%c10_i32, %c20_i32] {ttig.block_io = "column_major"} : !tt.tensordesc<tensor<16x32xf16>>

    tt.return
  }
}

// -----

// COM: Test column-major descriptor prefetch with support_prefetch_256b.
// COM: Same 16x256xf16 descriptor as @prefetch_descriptor_256b, but with column_major.
// COM: Base width/height/pitch/offset computation is verified by @prefetch_descriptor_column_major;
// COM: here we only verify the base_ptr extraction and the final prefetch tile parameters.
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_prefetch_256b} {
// CHECK-LABEL:   llvm.func spir_kernelcc @prefetch_descriptor_256b_column_major(
  tt.func public @prefetch_descriptor_256b_column_major(%arg0: !tt.ptr<f16>, %arg2: i64, %arg4: i64, %arg5: i64) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c16_i32 = arith.constant 16 : i32
    %c256_i32 = arith.constant 256 : i32

    // COM: Same descriptor as the row-major 256b test: 16x256xf16.
    // COM: The column_major attribute is on the prefetch op; the descriptor uses standard strides.
    %desc = tt.make_tensor_descriptor %arg0, [%c16_i32, %c256_i32], [%arg5, %c1_i64] : <f16>, <tensor<16x256xf16>>

    // CHECK:     %[[DESC:.*]] = llvm.insertvalue %{{.*}}, %{{.*}}[4] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[BASE_PTR:.*]] = llvm.extractvalue %[[DESC]][4] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>

    // COM: 16x256xf16 column-major with 256B support: after the column-major swap the
    // COM: effective tiling shape is 256x16, giving tile_width=16, tile_height=32, v_blocks=1.
    // COM: Compare with row-major (16x256): tile_width=128, tile_height=4, v_blocks=1.
    // CHECK:     triton_gen.2Dblockprefetch %[[BASE_PTR]], {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 32, v_blocks = 1, cache_control = L1C_L3C}
    ttig.descriptor_prefetch %desc[%c0_i32, %c0_i32] {ttig.block_io = "column_major"} : !tt.tensordesc<tensor<16x256xf16>>

    tt.return
  }
}
