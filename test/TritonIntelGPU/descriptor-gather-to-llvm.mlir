// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s

// Test that ttig.descriptor_gather is lowered to LLVM correctly.
// The split-file tests below cover three cases:
//   1. Fallback path when the gather cannot use the fast path. (no coalescing)
//   2. Fast path when numPtrsPerLoad == threadsPerWarp, which lowers to
//      per-lane predicated llvm.load operations on lane-mapped addresses. (Could be coalesced)
//   3. Fast path when numPtrsPerLoad != threadsPerWarp, which lowers to
//      triton_gen.sub_group_gather_load, guarded by a column-window check.
//   4. A layout whose lanes span rows, which takes the fallback path.
//   5-7. Fast-path configurations whose byte-to-register mapping does not hold on
//      hardware, which also take the fallback path.

// Test 1: Fallback path
//
// A blocked encoding that does not satisfy block-IO tile constraints falls
// through to the default path which emits one predicated llvm.load per result
// element.  The test tracks the offset X data flow end-to-end:
//   %arg1 (x_offsets tensor) → unpack → offsetX i16
//   → sext i64 → bounds-check icmp against shape0
//   → and with predY → cond_br (zero on the false edge) → predicated llvm.load
//   mul by stride0 → add with y linear offset → gep into base ptr

#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [1, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#slice_x = #ttg.slice<{dim = 1, parent = #blocked}>

module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @descriptor_gather_default_path
  tt.func public @descriptor_gather_default_path(
      %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %arg1: tensor<8xi16, #slice_x>,
      %arg2: i32) -> tensor<8x16xf16, #blocked> {
    %c1_i64 = arith.constant 1 : i64
    %c256_i32 = arith.constant 256 : i32
    %c16_i64 = arith.constant 16 : i64
    // Descriptor over a 2D matrix; block shape is 1×16 (one row, 16 cols).
    %desc = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c16_i64, %c1_i64] : <f16>, <1x16xf16>

    // Verify descriptor struct is built:
    // CHECK:     llvm.insertvalue {{.*}}[4] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>

    // x_offsets tensor argument is unpacked to a scalar i16 per thread
    // (comes before descriptor field extraction in the lowered output):
    // CHECK:     %[[OFFX_I16:.*]] = llvm.extractvalue {{.*}} : !llvm.struct<(i16, i16)>

    // Descriptor fields are extracted after the x_offsets unpack:
    // CHECK-DAG: %[[SHAPE0:.*]] = llvm.extractvalue {{.*}}[0] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK-DAG: %[[STRIDE0:.*]] = llvm.extractvalue {{.*}}[2] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK-DAG: %[[BASE:.*]] = llvm.extractvalue {{.*}}[4] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>

    // Offset X data flow inside the per-element loop:
    //   1. Sign-extend offsetX to i64, so a negative offset fails the check.
    // CHECK:     %[[OFFX64:.*]] = llvm.sext %[[OFFX_I16]] : i16 to i64
    //   2. Bounds check: offsetX < shape0 (predX).
    // CHECK:     %[[PRED_X:.*]] = llvm.icmp "ult" %[[OFFX64]], %[[SHAPE0]] : i64
    //   3. Combine predX with predY (offsetY < shape1) to form the load predicate.
    // CHECK:     llvm.and %[[PRED_X]], {{.*}} : i1
    //   4. Row linear offset: offsetX * stride0.
    // CHECK:     %[[X_LIN_OFF:.*]] = llvm.mul %[[OFFX64]], %[[STRIDE0]] : i64
    //   5. Combined linear offset (x + y) fed into the final GEP.
    // CHECK:     %[[LIN_OFF:.*]] = llvm.add %[[X_LIN_OFF]], {{.*}} : i64
    // CHECK:     llvm.getelementptr %[[BASE]][%[[LIN_OFF]]] {{.*}} -> !llvm.ptr<1>, f16
    //   6. Predicated load: an out-of-bounds element reads zero.
    // CHECK:     %[[ZERO:.*]] = llvm.mlir.zero : vector<1xf16>
    // CHECK:     llvm.cond_br {{.*}}, ^{{.*}}, ^{{.*}}(%[[ZERO]] : vector<1xf16>)
    // CHECK:     llvm.load {{.*}} : !llvm.ptr<1> -> vector<1xf16>
    // Verify the fast path is NOT taken for the blocked encoding.
    // CHECK-NOT: triton_gen.sub_group_gather_load

    %result = ttig.descriptor_gather %desc[%arg1, %arg2]
        : (!tt.tensordesc<1x16xf16>, tensor<8xi16, #slice_x>, i32) -> tensor<8x16xf16, #blocked>
    tt.return %result : tensor<8x16xf16, #blocked>
  }
}

// -----

// Test 2: Fast path with numPtrsPerLoad == threadsPerWarp
//
// This shape satisfies the gather fast-path tile checks, but each pointer can
// be mapped directly to one SIMD lane because numPtrsPerLoad matches
// threadsPerWarp. The lowering therefore uses predicated llvm.load operations
// on non-uniform per-lane addresses instead of triton_gen.sub_group_gather_load.
// The test tracks the offset X data flow end-to-end:
//   %arg1 (x_offsets tensor) → unpack → offsetX i32
//   → sext i64 → bounds-check icmp against shape0
//   → and with predY → cond_br (zero on the false edge) → predicated llvm.load
//   mul by stride0 → add with y linear offset → gep into base ptr
// Each lane loads one element, so no column-window guard is needed.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#slice_x = #ttg.slice<{dim = 1, parent = #blocked}>

module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @descriptor_gather_fast_path_nonuniform_ptrs
  tt.func public @descriptor_gather_fast_path_nonuniform_ptrs(
      %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %arg1: tensor<8xi32, #slice_x>,
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
    //   1. Sign-extend offsetX to i64, so a negative offset fails the check.
    // CHECK:     %[[OFFX64:.*]] = llvm.sext %[[OFFX_I32]] : i32 to i64
    //   2. Bounds check: offsetX < shape0 (predX).
    // CHECK:     %[[PRED_X:.*]] = llvm.icmp "ult" %[[OFFX64]], %[[SHAPE0]] : i64
    //   3. Combine predX with predY (offsetY < shape1) to form the load predicate.
    // CHECK:     llvm.and %[[PRED_X]], {{.*}} : i1
    //   4. Row linear offset: offsetX * stride0.
    // CHECK:     %[[X_LIN_OFF:.*]] = llvm.mul %[[OFFX64]], %[[STRIDE0]] : i64
    //   5. Combined linear offset (x + y) fed into the final GEP.
    // CHECK:     %[[LIN_OFF:.*]] = llvm.add %[[X_LIN_OFF]], {{.*}} : i64
    // CHECK:     llvm.getelementptr %[[BASE]][%[[LIN_OFF]]] {{.*}} -> !llvm.ptr<1>, f16
    //   6. Predicated load: an out-of-bounds lane reads zero.
    // CHECK:     %[[ZERO:.*]] = llvm.mlir.zero : vector<1xf16>
    // CHECK:     llvm.cond_br {{.*}}, ^{{.*}}, ^{{.*}}(%[[ZERO]] : vector<1xf16>)
    // CHECK:     llvm.load {{.*}} : !llvm.ptr<1> -> vector<1xf16>
    // Verify the subgroup gather intrinsic is not used in the per-lane fast path.
    // CHECK-NOT: triton_gen.sub_group_gather_load

    %result = ttig.descriptor_gather %desc[%arg1, %arg2]
        : (!tt.tensordesc<1x16xf16>, tensor<8xi32, #slice_x>, i32) -> tensor<8x16xf16, #blocked>
    tt.return %result : tensor<8x16xf16, #blocked>
  }
}


// -----

// Test 3: Fast path with numPtrsPerLoad != threadsPerWarp
//
// This DPAS dot-op A layout satisfies the gather fast-path tile checks, and it
// requires more number of pointers per load than threadsPerWarp. The lowering packs
// the computed uniform addresses and predicates into vectors and emits
// triton_gen.sub_group_gather_load. The test tracks the offset X data flow
// through the pointer-vector construction.
//   %arg1 (x_offsets tensor) → unpack → shuffleIdx(lane 0 broadcast)
//   → sext i64 → bounds-check icmp against shape0
//   → mul against stride0 → second gep to build row-offset address
//   → address inserted into a vector → sub_group_gather_load

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
// x_offsets live in a slice layout that removes the column (dim1) dimension so
// that the same row index is broadcast to every column-thread in the warp.
#slice_x = #ttg.slice<{dim = 1, parent = #dpas}>

module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @descriptor_gather_fast_path_uniform_ptrs
  tt.func public @descriptor_gather_fast_path_uniform_ptrs(
      %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %arg1: tensor<64xi32, #slice_x>,
      %arg2: i32) -> tensor<64x32xf16, #dot0> {
    %c1_i64 = arith.constant 1 : i64
    %c256_i32 = arith.constant 256 : i32
    %c32_i64 = arith.constant 32 : i64
    // Descriptor block shape is 1×32 (one row, 32 cols).
    %desc = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c32_i64, %c1_i64] : <f16>, <1x32xf16>

    // x_offsets tensor argument is unpacked first (before descriptor fields):
    // CHECK:     %[[OFFX_0_I32F:.*]] = llvm.extractvalue %arg1[0] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)>
    // CHECK:     %[[OFFX_1_I32F:.*]] = llvm.extractvalue %arg1[1] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)>
    // CHECK:     %[[OFFX_2_I32F:.*]] = llvm.extractvalue %arg1[2] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)>
    // CHECK:     %[[OFFX_3_I32F:.*]] = llvm.extractvalue %arg1[3] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)>
    // CHECK:     %[[OFFX_4_I32F:.*]] = llvm.extractvalue %arg1[4] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)>
    // CHECK:     %[[OFFX_5_I32F:.*]] = llvm.extractvalue %arg1[5] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)>
    // CHECK:     %[[OFFX_6_I32F:.*]] = llvm.extractvalue %arg1[6] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)>
    // CHECK:     %[[OFFX_7_I32F:.*]] = llvm.extractvalue %arg1[7] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)>
    // CHECK:     %[[OFFX_8_I32F:.*]] = llvm.extractvalue %arg1[8] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)>
    // CHECK:     %[[OFFX_9_I32F:.*]] = llvm.extractvalue %arg1[9] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)>
    // CHECK:     %[[OFFX_10_I32F:.*]] = llvm.extractvalue %arg1[10] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)>
    // CHECK:     %[[OFFX_11_I32F:.*]] = llvm.extractvalue %arg1[11] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)>
    // CHECK:     %[[OFFX_12_I32F:.*]] = llvm.extractvalue %arg1[12] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)>
    // CHECK:     %[[OFFX_13_I32F:.*]] = llvm.extractvalue %arg1[13] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)>
    // CHECK:     %[[OFFX_14_I32F:.*]] = llvm.extractvalue %arg1[14] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)>
    // CHECK:     %[[OFFX_15_I32F:.*]] = llvm.extractvalue %arg1[15] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)>

    // Descriptor fields are extracted after the x_offsets unpack:
    // CHECK:     %[[SHAPE_X:.*]] = llvm.extractvalue {{.*}}[0] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[SHAPE_Y:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[STRIDE_X:.*]] = llvm.extractvalue {{.*}}[2] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[STRIDE_Y:.*]] = llvm.extractvalue {{.*}}[3] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
    // CHECK:     %[[BASEF:.*]] = llvm.extractvalue {{.*}}[4] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>

    // CHECK:     %[[BASE_OFFSET_Y:.*]] = llvm.add %arg2, {{.*}} : i32

    // Each pointer loads 4 columns but checks only the first, so the fast path
    // runs only when the whole column window [y, y + 32) is in bounds:
    // CHECK:     %[[Y64:.*]] = llvm.sext %arg2 : i32 to i64
    // CHECK:     %[[NCOLS:.*]] = llvm.mlir.constant(32 : i64) : i64
    // CHECK:     %[[Y_END:.*]] = llvm.add %[[Y64]], %[[NCOLS]] : i64
    // CHECK:     %[[Y_LO:.*]] = llvm.icmp "sge" %[[Y64]], {{.*}} : i64
    // CHECK:     %[[Y_HI:.*]] = llvm.icmp "sle" %[[Y_END]], %[[SHAPE_Y]] : i64
    // CHECK:     %[[COLS_OK:.*]] = llvm.and %[[Y_LO]], %[[Y_HI]] : i1
    // CHECK:     llvm.cond_br %[[COLS_OK]], ^{{.*}}, ^[[SLOW:bb[0-9]+]]

    // Offset X data flow (fast path) inside the per-load inner loop:
    // CHECK:     %[[OFFX_UNIF:.*]] = llvm.call spir_funccc @_Z17sub_group_shuffleij(%[[OFFX_0_I32F]], {{.*}}) {convergent, no_unwind, will_return} : (i32, i32) -> i32
    // Offset Y 0 for first row of the 8*4 load (sub-offset Y = 0, 4, 8, 12):
    // CHECK:     %[[SUB_OFF_Y_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:     %[[OFF_Y:.*]] = llvm.add %[[BASE_OFFSET_Y]], %[[SUB_OFF_Y_0]] : i32
    // CHECK:     %[[OFF_X_64:.*]] = llvm.sext %[[OFFX_UNIF]] : i32 to i64
    // CHECK:     %[[PRED_X:.*]] = llvm.icmp "ult" %[[OFF_X_64]], %[[SHAPE_X]] : i64
    // CHECK:     %[[OFF_Y_64:.*]] = llvm.zext %[[OFF_Y]] : i32 to i64
    // CHECK:     %[[PRED_Y:.*]] = llvm.icmp "ult" %[[OFF_Y_64]], %[[SHAPE_Y]] : i64
    // CHECK:     %[[PRED:.*]] = llvm.and %[[PRED_X]], %[[PRED_Y]] : i1
    // CHECK:     %[[LINEAR_OFF_X:.*]] = llvm.mul %[[OFF_X_64]], %[[STRIDE_X]] : i64
    // CHECK:     %[[LINEAR_OFF_Y:.*]] = llvm.mul %[[OFF_Y_64]], %[[STRIDE_Y]] : i64
    // CHECK:     %[[LINEAR_OFF:.*]] = llvm.add %[[LINEAR_OFF_X]], %[[LINEAR_OFF_Y]] : i64
    // CHECK:     %[[PTR_0:.*]] = llvm.getelementptr %[[BASEF]]{{\[}}%[[LINEAR_OFF]]] {{.*}} -> !llvm.ptr<1>, f16
    // CHECK:     %[[PTR_0_I64:.*]] = llvm.ptrtoint %[[PTR_0]] : !llvm.ptr<1> to i64

    // Offset Y 4 for first row of the 8*4 load (sub-offset Y = 0, 4, 8, 12):
    // CHECK:     %[[SUB_OFF_Y_4:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK:     %[[OFF_Y:.*]] = llvm.add %[[BASE_OFFSET_Y]], %[[SUB_OFF_Y_4]] : i32
    // CHECK:     %[[OFF_X_64:.*]] = llvm.sext %[[OFFX_UNIF]] : i32 to i64
    // CHECK:     %[[PRED_X:.*]] = llvm.icmp "ult" %[[OFF_X_64]], %[[SHAPE_X]] : i64
    // CHECK:     %[[OFF_Y_64:.*]] = llvm.zext %[[OFF_Y]] : i32 to i64
    // CHECK:     %[[PRED_Y:.*]] = llvm.icmp "ult" %[[OFF_Y_64]], %[[SHAPE_Y]] : i64
    // CHECK:     %[[PRED:.*]] = llvm.and %[[PRED_X]], %[[PRED_Y]] : i1
    // CHECK:     %[[LINEAR_OFF_X:.*]] = llvm.mul %[[OFF_X_64]], %[[STRIDE_X]] : i64
    // CHECK:     %[[LINEAR_OFF_Y:.*]] = llvm.mul %[[OFF_Y_64]], %[[STRIDE_Y]] : i64
    // CHECK:     %[[LINEAR_OFF:.*]] = llvm.add %[[LINEAR_OFF_X]], %[[LINEAR_OFF_Y]] : i64
    // CHECK:     %[[PTR_1:.*]] = llvm.getelementptr %[[BASEF]]{{\[}}%[[LINEAR_OFF]]] {{.*}} -> !llvm.ptr<1>, f16
    // CHECK:     %[[PTR_1_I64:.*]] = llvm.ptrtoint %[[PTR_1]] : !llvm.ptr<1> to i64

    // Offset Y 8 for first row of the 8*4 load (sub-offset Y = 0, 4, 8, 12):
    // CHECK:     %[[SUB_OFF_Y_8:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK:     %[[OFF_Y:.*]] = llvm.add %[[BASE_OFFSET_Y]], %[[SUB_OFF_Y_8]] : i32
    // CHECK:     %[[OFF_X_64:.*]] = llvm.sext %[[OFFX_UNIF]] : i32 to i64
    // CHECK:     %[[PRED_X:.*]] = llvm.icmp "ult" %[[OFF_X_64]], %[[SHAPE_X]] : i64
    // CHECK:     %[[OFF_Y_64:.*]] = llvm.zext %[[OFF_Y]] : i32 to i64
    // CHECK:     %[[PRED_Y:.*]] = llvm.icmp "ult" %[[OFF_Y_64]], %[[SHAPE_Y]] : i64
    // CHECK:     %[[PRED:.*]] = llvm.and %[[PRED_X]], %[[PRED_Y]] : i1
    // CHECK:     %[[LINEAR_OFF_X:.*]] = llvm.mul %[[OFF_X_64]], %[[STRIDE_X]] : i64
    // CHECK:     %[[LINEAR_OFF_Y:.*]] = llvm.mul %[[OFF_Y_64]], %[[STRIDE_Y]] : i64
    // CHECK:     %[[LINEAR_OFF:.*]] = llvm.add %[[LINEAR_OFF_X]], %[[LINEAR_OFF_Y]] : i64
    // CHECK:     %[[PTR_2:.*]] = llvm.getelementptr %[[BASEF]]{{\[}}%[[LINEAR_OFF]]] {{.*}} -> !llvm.ptr<1>, f16
    // CHECK:     %[[PTR_2_I64:.*]] = llvm.ptrtoint %[[PTR_2]] : !llvm.ptr<1> to i64

    // Offset Y 12 for first row of the 8*4 load (sub-offset Y = 0, 4, 8, 12):
    // CHECK:     %[[SUB_OFF_Y_12:.*]] = llvm.mlir.constant(12 : i32) : i32
    // CHECK:     %[[OFF_Y:.*]] = llvm.add %[[BASE_OFFSET_Y]], %[[SUB_OFF_Y_12]] : i32
    // CHECK:     %[[OFF_X_64:.*]] = llvm.sext %[[OFFX_UNIF]] : i32 to i64
    // CHECK:     %[[PRED_X:.*]] = llvm.icmp "ult" %[[OFF_X_64]], %[[SHAPE_X]] : i64
    // CHECK:     %[[OFF_Y_64:.*]] = llvm.zext %[[OFF_Y]] : i32 to i64
    // CHECK:     %[[PRED_Y:.*]] = llvm.icmp "ult" %[[OFF_Y_64]], %[[SHAPE_Y]] : i64
    // CHECK:     %[[PRED:.*]] = llvm.and %[[PRED_X]], %[[PRED_Y]] : i1
    // CHECK:     %[[LINEAR_OFF_X:.*]] = llvm.mul %[[OFF_X_64]], %[[STRIDE_X]] : i64
    // CHECK:     %[[LINEAR_OFF_Y:.*]] = llvm.mul %[[OFF_Y_64]], %[[STRIDE_Y]] : i64
    // CHECK:     %[[LINEAR_OFF:.*]] = llvm.add %[[LINEAR_OFF_X]], %[[LINEAR_OFF_Y]] : i64
    // CHECK:     %[[PTR_3:.*]] = llvm.getelementptr %[[BASEF]]{{\[}}%[[LINEAR_OFF]]] {{.*}} -> !llvm.ptr<1>, f16
    // CHECK:     %[[PTR_3_I64:.*]] = llvm.ptrtoint %[[PTR_3]] : !llvm.ptr<1> to i64

    // COM: The offset X data flow is repeated for the next 7 rows of the 8*4 load:
    // CHECK:     %[[OFFX_UNIF:.*]] = llvm.call spir_funccc @_Z17sub_group_shuffleij(%[[OFFX_1_I32F]], {{.*}}) {convergent, no_unwind, will_return} : (i32, i32) -> i32
    // CHECK:     %[[OFFX_UNIF:.*]] = llvm.call spir_funccc @_Z17sub_group_shuffleij(%[[OFFX_2_I32F]], {{.*}}) {convergent, no_unwind, will_return} : (i32, i32) -> i32
    // CHECK:     %[[OFFX_UNIF:.*]] = llvm.call spir_funccc @_Z17sub_group_shuffleij(%[[OFFX_3_I32F]], {{.*}}) {convergent, no_unwind, will_return} : (i32, i32) -> i32
    // CHECK:     %[[OFFX_UNIF:.*]] = llvm.call spir_funccc @_Z17sub_group_shuffleij(%[[OFFX_4_I32F]], {{.*}}) {convergent, no_unwind, will_return} : (i32, i32) -> i32
    // CHECK:     %[[OFFX_UNIF:.*]] = llvm.call spir_funccc @_Z17sub_group_shuffleij(%[[OFFX_5_I32F]], {{.*}}) {convergent, no_unwind, will_return} : (i32, i32) -> i32
    // CHECK:     %[[OFFX_UNIF:.*]] = llvm.call spir_funccc @_Z17sub_group_shuffleij(%[[OFFX_6_I32F]], {{.*}}) {convergent, no_unwind, will_return} : (i32, i32) -> i32
    // CHECK:     %[[OFFX_UNIF:.*]] = llvm.call spir_funccc @_Z17sub_group_shuffleij(%[[OFFX_7_I32F]], {{.*}}) {convergent, no_unwind, will_return} : (i32, i32) -> i32

    // COMM:  Gather all 32 pointers and predicates into vectors for the gather load.
    // CHECK:     triton_gen.sub_group_gather_load {{.*}}, {{.*}} :  (vector<32xi64>, vector<32xi1>) -> vector<8xf16>

    // COM: load for sub-offset Y from 16 to 28.
    // CHECK:     triton_gen.sub_group_gather_load {{.*}}, {{.*}} :  (vector<32xi64>, vector<32xi1>) -> vector<8xf16>

    // COM: load for offset X index from 8 to 15, sub-offset Y from 0 to 12.
    // CHECK:     triton_gen.sub_group_gather_load {{.*}}, {{.*}} :  (vector<32xi64>, vector<32xi1>) -> vector<8xf16>

    // COM: load for offset X index from 8 to 15, sub-offset Y from 16 to 28.
    // CHECK:     %[[OFFX_UNIF:.*]] = llvm.call spir_funccc @_Z17sub_group_shuffleij(%[[OFFX_8_I32F]], {{.*}}) {convergent, no_unwind, will_return} : (i32, i32) -> i32
    // Offset Y 16 for first row of the 8*4 load (sub-offset Y = 16, 20, 24, 28):
    // CHECK:     %[[SUB_OFF_Y_0:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:     %[[OFF_Y:.*]] = llvm.add %[[BASE_OFFSET_Y]], %[[SUB_OFF_Y_0]] : i32
    // CHECK:     %[[OFF_X_64:.*]] = llvm.sext %[[OFFX_UNIF]] : i32 to i64
    // CHECK:     %[[PRED_X:.*]] = llvm.icmp "ult" %[[OFF_X_64]], %[[SHAPE_X]] : i64
    // CHECK:     %[[OFF_Y_64:.*]] = llvm.zext %[[OFF_Y]] : i32 to i64
    // CHECK:     %[[PRED_Y:.*]] = llvm.icmp "ult" %[[OFF_Y_64]], %[[SHAPE_Y]] : i64
    // CHECK:     %[[PRED:.*]] = llvm.and %[[PRED_X]], %[[PRED_Y]] : i1
    // CHECK:     %[[LINEAR_OFF_X:.*]] = llvm.mul %[[OFF_X_64]], %[[STRIDE_X]] : i64
    // CHECK:     %[[LINEAR_OFF_Y:.*]] = llvm.mul %[[OFF_Y_64]], %[[STRIDE_Y]] : i64
    // CHECK:     %[[LINEAR_OFF:.*]] = llvm.add %[[LINEAR_OFF_X]], %[[LINEAR_OFF_Y]] : i64
    // CHECK:     %[[PTR_0:.*]] = llvm.getelementptr %[[BASEF]]{{\[}}%[[LINEAR_OFF]]] {{.*}} -> !llvm.ptr<1>, f16
    // CHECK:     %[[PTR_0_I64:.*]] = llvm.ptrtoint %[[PTR_0]] : !llvm.ptr<1> to i64

    // Offset Y 20 for first row of the 8*4 load (sub-offset Y = 16, 20, 24, 28):
    // CHECK:     %[[SUB_OFF_Y_4:.*]] = llvm.mlir.constant(20 : i32) : i32
    // CHECK:     %[[OFF_Y:.*]] = llvm.add %[[BASE_OFFSET_Y]], %[[SUB_OFF_Y_4]] : i32
    // CHECK:     %[[OFF_X_64:.*]] = llvm.sext %[[OFFX_UNIF]] : i32 to i64
    // CHECK:     %[[PRED_X:.*]] = llvm.icmp "ult" %[[OFF_X_64]], %[[SHAPE_X]] : i64
    // CHECK:     %[[OFF_Y_64:.*]] = llvm.zext %[[OFF_Y]] : i32 to i64
    // CHECK:     %[[PRED_Y:.*]] = llvm.icmp "ult" %[[OFF_Y_64]], %[[SHAPE_Y]] : i64
    // CHECK:     %[[PRED:.*]] = llvm.and %[[PRED_X]], %[[PRED_Y]] : i1
    // CHECK:     %[[LINEAR_OFF_X:.*]] = llvm.mul %[[OFF_X_64]], %[[STRIDE_X]] : i64
    // CHECK:     %[[LINEAR_OFF_Y:.*]] = llvm.mul %[[OFF_Y_64]], %[[STRIDE_Y]] : i64
    // CHECK:     %[[LINEAR_OFF:.*]] = llvm.add %[[LINEAR_OFF_X]], %[[LINEAR_OFF_Y]] : i64
    // CHECK:     %[[PTR_1:.*]] = llvm.getelementptr %[[BASEF]]{{\[}}%[[LINEAR_OFF]]] {{.*}} -> !llvm.ptr<1>, f16
    // CHECK:     %[[PTR_1_I64:.*]] = llvm.ptrtoint %[[PTR_1]] : !llvm.ptr<1> to i64

    // Offset Y 24 for first row of the 8*4 load (sub-offset Y = 16, 20, 24, 28):
    // CHECK:     %[[SUB_OFF_Y_8:.*]] = llvm.mlir.constant(24 : i32) : i32
    // CHECK:     %[[OFF_Y:.*]] = llvm.add %[[BASE_OFFSET_Y]], %[[SUB_OFF_Y_8]] : i32
    // CHECK:     %[[OFF_X_64:.*]] = llvm.sext %[[OFFX_UNIF]] : i32 to i64
    // CHECK:     %[[PRED_X:.*]] = llvm.icmp "ult" %[[OFF_X_64]], %[[SHAPE_X]] : i64
    // CHECK:     %[[OFF_Y_64:.*]] = llvm.zext %[[OFF_Y]] : i32 to i64
    // CHECK:     %[[PRED_Y:.*]] = llvm.icmp "ult" %[[OFF_Y_64]], %[[SHAPE_Y]] : i64
    // CHECK:     %[[PRED:.*]] = llvm.and %[[PRED_X]], %[[PRED_Y]] : i1
    // CHECK:     %[[LINEAR_OFF_X:.*]] = llvm.mul %[[OFF_X_64]], %[[STRIDE_X]] : i64
    // CHECK:     %[[LINEAR_OFF_Y:.*]] = llvm.mul %[[OFF_Y_64]], %[[STRIDE_Y]] : i64
    // CHECK:     %[[LINEAR_OFF:.*]] = llvm.add %[[LINEAR_OFF_X]], %[[LINEAR_OFF_Y]] : i64
    // CHECK:     %[[PTR_2:.*]] = llvm.getelementptr %[[BASEF]]{{\[}}%[[LINEAR_OFF]]] {{.*}} -> !llvm.ptr<1>, f16
    // CHECK:     %[[PTR_2_I64:.*]] = llvm.ptrtoint %[[PTR_2]] : !llvm.ptr<1> to i64

    // Offset Y 28 for first row of the 8*4 load (sub-offset Y = 16, 20, 24, 28):
    // CHECK:     %[[SUB_OFF_Y_12:.*]] = llvm.mlir.constant(28 : i32) : i32
    // CHECK:     %[[OFF_Y:.*]] = llvm.add %[[BASE_OFFSET_Y]], %[[SUB_OFF_Y_12]] : i32
    // CHECK:     %[[OFF_X_64:.*]] = llvm.sext %[[OFFX_UNIF]] : i32 to i64
    // CHECK:     %[[PRED_X:.*]] = llvm.icmp "ult" %[[OFF_X_64]], %[[SHAPE_X]] : i64
    // CHECK:     %[[OFF_Y_64:.*]] = llvm.zext %[[OFF_Y]] : i32 to i64
    // CHECK:     %[[PRED_Y:.*]] = llvm.icmp "ult" %[[OFF_Y_64]], %[[SHAPE_Y]] : i64
    // CHECK:     %[[PRED:.*]] = llvm.and %[[PRED_X]], %[[PRED_Y]] : i1
    // CHECK:     %[[LINEAR_OFF_X:.*]] = llvm.mul %[[OFF_X_64]], %[[STRIDE_X]] : i64
    // CHECK:     %[[LINEAR_OFF_Y:.*]] = llvm.mul %[[OFF_Y_64]], %[[STRIDE_Y]] : i64
    // CHECK:     %[[LINEAR_OFF:.*]] = llvm.add %[[LINEAR_OFF_X]], %[[LINEAR_OFF_Y]] : i64
    // CHECK:     %[[PTR_3:.*]] = llvm.getelementptr %[[BASEF]]{{\[}}%[[LINEAR_OFF]]] {{.*}} -> !llvm.ptr<1>, f16
    // CHECK:     %[[PTR_3_I64:.*]] = llvm.ptrtoint %[[PTR_3]] : !llvm.ptr<1> to i64

    // CHECK:     triton_gen.sub_group_gather_load {{.*}}, {{.*}} :  (vector<32xi64>, vector<32xi1>) -> vector<8xf16>
    // CHECK:     llvm.br ^[[JOIN:bb[0-9]+]](

    // Otherwise each element is checked and loaded on its own, reading zero
    // when out of bounds:
    // CHECK:   ^[[SLOW]]:
    // CHECK:     %[[ZERO:.*]] = llvm.mlir.zero : vector<1xf16>
    // CHECK:     llvm.cond_br {{.*}}, ^{{.*}}, ^{{.*}}(%[[ZERO]] : vector<1xf16>)
    // CHECK-NOT: triton_gen.sub_group_gather_load
    // CHECK:   ^[[JOIN]](%[[RESULT:[0-9]+]]: !llvm.struct
    // CHECK:     llvm.return %[[RESULT]]

    %result = ttig.descriptor_gather %desc[%arg1, %arg2]
        : (!tt.tensordesc<1x32xf16>, tensor<64xi32, #slice_x>, i32) -> tensor<64x32xf16, #dot0>
    tt.return %result : tensor<64x32xf16, #dot0>
  }
}

// -----

// Test 4: Lanes span rows
//
// Both fast paths read each row offset from one lane. Here lanes 8-15 hold the
// odd rows, so the gather must take the per-element fallback path.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 8], warpsPerCTA = [1, 4], order = [1, 0]}>
#slice_x = #ttg.slice<{dim = 1, parent = #blocked}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @descriptor_gather_lanes_span_rows
  // CHECK-NOT:   triton_gen.sub_group_gather_load
  // CHECK:       llvm.load {{.*}} : !llvm.ptr<1> -> vector<1xf16>
  // CHECK-NOT:   triton_gen.sub_group_gather_load
  // CHECK:       llvm.return
  tt.func public @descriptor_gather_lanes_span_rows(
      %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %arg1: tensor<32xi32, #slice_x>,
      %arg2: i32) -> tensor<32x64xf16, #blocked> {
    %c1_i64 = arith.constant 1 : i64
    %c256_i32 = arith.constant 256 : i32
    %c64_i64 = arith.constant 64 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c64_i64, %c1_i64] : <f16>, <1x64xf16>
    %result = ttig.descriptor_gather %desc[%arg1, %arg2]
        : (!tt.tensordesc<1x64xf16>, tensor<32xi32, #slice_x>, i32) -> tensor<32x64xf16, #blocked>
    tt.return %result : tensor<32x64xf16, #blocked>
  }
}

// -----

// Test 5: Packed per-lane loads
//
// Each lane would load 4 contiguous columns, but owns one column of 4 rows (the
// vLLM paged-KV layout at sub-group size 32).

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#slice_x = #ttg.slice<{dim = 1, parent = #blocked}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @descriptor_gather_test5
  // CHECK-NOT:   triton_gen.sub_group_gather_load
  // CHECK-NOT:   -> vector<4xf16>
  // CHECK:       llvm.load {{.*}} : !llvm.ptr<1> -> vector<1xf16>
  // CHECK-NOT:   triton_gen.sub_group_gather_load
  // CHECK-NOT:   -> vector<4xf16>
  // CHECK:       llvm.return
  tt.func public @descriptor_gather_test5(
      %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %arg1: tensor<64xi32, #slice_x>,
      %arg2: i32) -> tensor<64x128xf16, #blocked> {
    %c1_i64 = arith.constant 1 : i64
    %c256_i32 = arith.constant 256 : i32
    %cols_i64 = arith.constant 128 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%cols_i64, %c1_i64] : <f16>, <1x128xf16>
    %result = ttig.descriptor_gather %desc[%arg1, %arg2]
        : (!tt.tensordesc<1x128xf16>, tensor<64xi32, #slice_x>, i32) -> tensor<64x128xf16, #blocked>
    tt.return %result : tensor<64x128xf16, #blocked>
  }
}

// -----

// Test 6: Packed sub-group gather load
//
// Each lane owns 2 adjacent columns; the sub-group load would hand lane l columns l
// and l + 16 instead.

#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
#slice_x = #ttg.slice<{dim = 1, parent = #blocked}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @descriptor_gather_test6
  // CHECK-NOT:   triton_gen.sub_group_gather_load
  // CHECK-NOT:   -> vector<2xf16>
  // CHECK:       llvm.load {{.*}} : !llvm.ptr<1> -> vector<1xf16>
  // CHECK-NOT:   triton_gen.sub_group_gather_load
  // CHECK-NOT:   -> vector<2xf16>
  // CHECK:       llvm.return
  tt.func public @descriptor_gather_test6(
      %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %arg1: tensor<64xi32, #slice_x>,
      %arg2: i32) -> tensor<64x64xf16, #blocked> {
    %c1_i64 = arith.constant 1 : i64
    %c256_i32 = arith.constant 256 : i32
    %cols_i64 = arith.constant 64 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%cols_i64, %c1_i64] : <f16>, <1x64xf16>
    %result = ttig.descriptor_gather %desc[%arg1, %arg2]
        : (!tt.tensordesc<1x64xf16>, tensor<64xi32, #slice_x>, i32) -> tensor<64x64xf16, #blocked>
    tt.return %result : tensor<64x64xf16, #blocked>
  }
}

// -----

// Test 7: Two-byte sub-group gather load
//
// Each address would load one 2-byte element, which hangs the sub-group gather
// load on hardware.

#blocked = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice_x = #ttg.slice<{dim = 1, parent = #blocked}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @descriptor_gather_test7
  // CHECK-NOT:   triton_gen.sub_group_gather_load
  // CHECK-NOT:   -> vector<2xf16>
  // CHECK:       llvm.load {{.*}} : !llvm.ptr<1> -> vector<1xf16>
  // CHECK-NOT:   triton_gen.sub_group_gather_load
  // CHECK-NOT:   -> vector<2xf16>
  // CHECK:       llvm.return
  tt.func public @descriptor_gather_test7(
      %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %arg1: tensor<64xi32, #slice_x>,
      %arg2: i32) -> tensor<64x64xf16, #blocked> {
    %c1_i64 = arith.constant 1 : i64
    %c256_i32 = arith.constant 256 : i32
    %cols_i64 = arith.constant 64 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%cols_i64, %c1_i64] : <f16>, <1x64xf16>
    %result = ttig.descriptor_gather %desc[%arg1, %arg2]
        : (!tt.tensordesc<1x64xf16>, tensor<64xi32, #slice_x>, i32) -> tensor<64x64xf16, #blocked>
    tt.return %result : tensor<64x64xf16, #blocked>
  }
}
