// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s

// Test that ttig.descriptor_gather is lowered to LLVM correctly.
// Two lowering paths are covered:
//   1. Default fallback path: generates per-element predicated loads using
//      pointer arithmetic computed from the descriptor's base/shape/stride.
//   2. Fast path: generates triton_gen.sub_group_gather_load for DPAS dot_op A
//      layouts that satisfy block-IO tile constraints.

// Test 1: Default fallback path
//
// A blocked encoding that does not satisfy block-IO tile constraints falls
// through to the default path which emits one predicated llvm.load per result
// element.  The test tracks the offset X data flow end-to-end:
//   %arg1 (x_offsets tensor) → unpack → offsetX i32
//   → zext i64 → bounds-check icmp against shape0
//   → and with predY → cond_br → predicated llvm.load
//   mul by stride0 → add with y linear offset → gep into base ptr

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked_x = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [8], order = [0]}>

module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @descriptor_gather_default_path
  tt.func public @descriptor_gather_default_path(
      %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %arg1: tensor<8xi32, #blocked_x>,
      %arg2: i32) -> tensor<8x16xf16, #blocked> {
    %c1_i64 = arith.constant 1 : i64
    %c256_i32 = arith.constant 256 : i32
    %c16_i64 = arith.constant 16 : i64
    // Descriptor over a 2D matrix; block shape is 1×16 (one row, 16 cols).
    %desc = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c16_i64, %c1_i64] : <f16>, <1x16xf16>

    // Verify descriptor struct is built:
    // CHECK:     llvm.insertvalue {{.*}}[4] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>

    // x_offsets tensor argument is unpacked to a scalar i32 per thread
    // (comes before descriptor field extraction in the lowered output):
    // CHECK:     %[[OFFX_I32:.*]] = llvm.extractvalue {{.*}} : !llvm.struct<(i32)>

    // Descriptor fields are extracted after the x_offsets unpack:
    // CHECK-DAG: %[[SHAPE0:.*]] = llvm.extractvalue {{.*}}[0] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK-DAG: %[[STRIDE0:.*]] = llvm.extractvalue {{.*}}[2] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK-DAG: %[[BASE:.*]] = llvm.extractvalue {{.*}}[4] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>

    // Offset X data flow inside the per-element loop:
    //   1. Widen offsetX from i32 to i64 for pointer arithmetic.
    // CHECK:     %[[OFFX64:.*]] = llvm.zext %[[OFFX_I32]] : i32 to i64
    //   2. Bounds check: offsetX < shape0 (predX).
    // CHECK:     %[[PRED_X:.*]] = llvm.icmp "ult" %[[OFFX64]], %[[SHAPE0]] : i64
    //   3. Combine predX with predY (offsetY < shape1) to form the load predicate.
    // CHECK:     llvm.and %[[PRED_X]], {{.*}} : i1
    //   4. Row linear offset: offsetX * stride0.
    // CHECK:     %[[X_LIN_OFF:.*]] = llvm.mul %[[OFFX64]], %[[STRIDE0]] : i64
    //   5. Combined linear offset (x + y) fed into the final GEP.
    // CHECK:     %[[LIN_OFF:.*]] = llvm.add %[[X_LIN_OFF]], {{.*}} : i64
    // CHECK:     llvm.getelementptr %[[BASE]][%[[LIN_OFF]]] {{.*}} -> !llvm.ptr<1>, f16
    //   6. Predicated load: branch on the combined predicate.
    // CHECK:     llvm.cond_br {{.*}}
    // CHECK:     llvm.load {{.*}} : !llvm.ptr<1> -> f16
    // Verify the fast path is NOT taken for the blocked encoding.
    // CHECK-NOT: triton_gen.sub_group_gather_load

    %result = ttig.descriptor_gather %desc[%arg1, %arg2]
        : (!tt.tensordesc<1x16xf16>, tensor<8xi32, #blocked_x>, i32) -> tensor<8x16xf16, #blocked>
    tt.return %result : tensor<8x16xf16, #blocked>
  }
}

// -----

// Test 2: Fast path (SubGroupGatherLoad)
//
// When the result tensor uses a dot_op A DPAS encoding the fast-path emits
// triton_gen.sub_group_gather_load instead of scalar predicated loads.
// The test tracks the offset X data flow end-to-end:
//   %arg1 (x_offsets tensor) → unpack → shuffleIdx(lane 0 broadcast)
//   → zext i64 → bounds-check icmp against shape0
//   → mul against stride0 → second gep to build row-offset address
//   → address inserted into a vector → sub_group_gather_load

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
// x_offsets live in a slice layout that removes the column (dim1) dimension so
// that the same row index is broadcast to every column-thread in the warp.
#slice_x = #ttg.slice<{dim = 1, parent = #dpas}>

module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @descriptor_gather_fast_path
  tt.func public @descriptor_gather_fast_path(
      %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %arg1: tensor<64xi32, #slice_x>,
      %arg2: i32) -> tensor<64x32xf16, #dot0> {
    %c1_i64 = arith.constant 1 : i64
    %c256_i32 = arith.constant 256 : i32
    %c32_i64 = arith.constant 32 : i64
    // Descriptor block shape is 1×32 (one row, 32 cols).
    %desc = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c32_i64, %c1_i64] : <f16>, <1x32xf16>

    // x_offsets tensor argument is unpacked first (before descriptor fields):
    // CHECK:     %[[OFFX_I32F:.*]] = llvm.extractvalue {{.*}} : !llvm.struct<(i32)>

    // Descriptor fields are extracted after the x_offsets unpack:
    // CHECK-DAG: %[[SHAPE0F:.*]] = llvm.extractvalue {{.*}}[0] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK-DAG: %[[STRIDE0F:.*]] = llvm.extractvalue {{.*}}[2] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK-DAG: %[[BASEF:.*]] = llvm.extractvalue {{.*}}[4] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>

    // Offset X data flow (fast path) inside the per-load inner loop:
    //   1. Broadcast lane 0's row index to all threads in the sub-group.
    // CHECK:     %[[OFFX_UNIF:.*]] = triton_gen.sub_group_shuffle {{.*}} %[[OFFX_I32F]], {{.*}} : i32
    //   2. Widen to i64 for 64-bit pointer arithmetic.
    // CHECK:     %[[OFFX64F:.*]] = llvm.zext %[[OFFX_UNIF]] : i32 to i64
    //   3. Bounds check: offsetX < shape0 (predX).
    // CHECK:     %[[PRED_XF:.*]] = llvm.icmp "ult" %[[OFFX64F]], %[[SHAPE0F]] : i64
    //   4. Row offset: offsetX * stride0.
    // CHECK:     %[[X_OFF64F:.*]] = llvm.mul %[[OFFX64F]], %[[STRIDE0F]] : i64
    //   5. Combine predX with predY (offsetY < shape1) to form the load predicate.
    // CHECK:     llvm.and %[[PRED_XF]], {{.*}} : i1
    //   6. First GEP advances by the constant y sub-offset; second GEP folds in
    //      the row offset (X_OFF64F) derived from offsetX.
    // CHECK:     %[[INNER_PTR:.*]] = llvm.getelementptr {{.*}}[{{.*}}] {{.*}} -> !llvm.ptr<1>, f16
    // CHECK:     llvm.getelementptr %[[INNER_PTR]][%[[X_OFF64F]]] {{.*}} -> !llvm.ptr<1>, f16
    //   7. Gather all 32 pointers and predicates into vectors for the gather load.
    // CHECK:     triton_gen.sub_group_gather_load {{.*}} : vector<{{.*}}>

    %result = ttig.descriptor_gather %desc[%arg1, %arg2]
        : (!tt.tensordesc<1x32xf16>, tensor<64xi32, #slice_x>, i32) -> tensor<64x32xf16, #dot0>
    tt.return %result : tensor<64x32xf16, #dot0>
  }
}
