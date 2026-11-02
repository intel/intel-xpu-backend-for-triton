// RUN: triton-opt %s -split-input-file --tritonintelgpu-lower-to-2d-block-load | FileCheck %s --implicit-check-not=tt.load

// COM: Row-major pointer-based load with broadcast (stride=0). The pass
// COM: converts this to ttig.2d_block_load_from_ptr, retaining the full
// COM: pointer tensor.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @tensor_of_pointers_load
  tt.func @tensor_of_pointers_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<64x32xf16, #dot0> {
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>> -> tensor<1x32xi32, #dot0>
    %2 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x32x!tt.ptr<f16>, #dot0>
    %3 = tt.addptr %2, %1 : tensor<1x32x!tt.ptr<f16>, #dot0>, tensor<1x32xi32, #dot0>
    %4 = tt.broadcast %3 : tensor<1x32x!tt.ptr<f16>, #dot0> -> tensor<64x32x!tt.ptr<f16>, #dot0>
    // CHECK: %[[P:.*]] = arith.constant 128 : i32
    // CHECK: %[[H:.*]] = arith.constant 1 : i32
    // CHECK: ttig.2d_block_load_from_ptr %4, %[[P]], %[[H]] {row_major} {base_width = 32 : i32}
    %5 = tt.load %4 {ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #dot0>
    tt.return %5 : tensor<64x32xf16, #dot0>
  }
}

// -----

// COM: Column-major pointer load. The memory_layout is set to column_major,
// COM: which tells the LLVM lowering to use transposed block reads.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @pointer_load_column_major
  tt.func @pointer_load_column_major(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<32x64xf16, #dot1> {
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #dot1}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #dot1}>> -> tensor<1x64xi32, #dot1>
    %2 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x64x!tt.ptr<f16>, #dot1>
    %3 = tt.addptr %2, %1 : tensor<1x64x!tt.ptr<f16>, #dot1>, tensor<1x64xi32, #dot1>
    %4 = tt.broadcast %3 : tensor<1x64x!tt.ptr<f16>, #dot1> -> tensor<32x64x!tt.ptr<f16>, #dot1>
    // CHECK: %[[P:.*]] = arith.constant 128 : i32
    // CHECK: %[[H:.*]] = arith.constant 1 : i32
    // CHECK: ttig.2d_block_load_from_ptr %4, %[[P]], %[[H]] {column_major} {base_width = 32 : i32}
    %5 = tt.load %4 {ttig.block_io = "column_major"} : tensor<32x64x!tt.ptr<f16>, #dot1>
    tt.return %5 : tensor<32x64xf16, #dot1>
  }
}

// -----

// COM: Masked pointer load with both mask and other. Both are forwarded
// COM: to the ttig.2d_block_load_from_ptr op. The mask is a constant true
// COM: (uniform constancy) which satisfies the 2D block I/O tile constraints.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @masked_pointer_load_with_other
  tt.func @masked_pointer_load_with_other(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: i1) -> tensor<64x32xf16, #dot0> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf16, #dot0>
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>> -> tensor<1x32xi32, #dot0>
    %2 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x32x!tt.ptr<f16>, #dot0>
    %3 = tt.addptr %2, %1 : tensor<1x32x!tt.ptr<f16>, #dot0>, tensor<1x32xi32, #dot0>
    %4 = tt.broadcast %3 : tensor<1x32x!tt.ptr<f16>, #dot0> -> tensor<64x32x!tt.ptr<f16>, #dot0>
    %mask = tt.splat %arg1 : i1 -> tensor<64x32xi1, #dot0>
    // CHECK: ttig.2d_block_load_from_ptr
    %7 = tt.load %4, %mask, %cst {ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #dot0>
    tt.return %7 : tensor<64x32xf16, #dot0>
  }
}

// -----

// COM: Masked pointer load WITHOUT an explicit 'other' value. The pass must
// COM: synthesize a zero splat so the verifier constraint (other required when
// COM: mask is present) is satisfied. The mask is a uniform splat which has
// COM: sufficient constancy for 2D block I/O.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @masked_pointer_load_no_other
  tt.func @masked_pointer_load_no_other(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: i1) -> tensor<64x32xf16, #dot0> {
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>> -> tensor<1x32xi32, #dot0>
    %2 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x32x!tt.ptr<f16>, #dot0>
    %3 = tt.addptr %2, %1 : tensor<1x32x!tt.ptr<f16>, #dot0>, tensor<1x32xi32, #dot0>
    %4 = tt.broadcast %3 : tensor<1x32x!tt.ptr<f16>, #dot0> -> tensor<64x32x!tt.ptr<f16>, #dot0>
    %mask = tt.splat %arg1 : i1 -> tensor<64x32xi1, #dot0>
    // CHECK: %[[ZERO_CST:.*]] = arith.constant 0.000000e+00 : f16
    // CHECK: %[[ZERO_SPLAT:.*]] = tt.splat %[[ZERO_CST]]
    // CHECK: ttig.2d_block_load_from_ptr
    %7 = tt.load %4, %mask {ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #dot0>
    tt.return %7 : tensor<64x32xf16, #dot0>
  }
}

// -----

// COM: Pointer load with 1D->2D reshape stride attribute. The pass reads the
// COM: stride from ttig.block_io_stride to compute pitch, skips tile validation,
// COM: and propagates the attribute to the output op.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @pointer_load_1d_reshape_stride
  tt.func @pointer_load_1d_reshape_stride(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<64x32xf16, #dot0> {
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>> -> tensor<1x32xi32, #dot0>
    %2 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x32x!tt.ptr<f16>, #dot0>
    %3 = tt.addptr %2, %1 : tensor<1x32x!tt.ptr<f16>, #dot0>, tensor<1x32xi32, #dot0>
    %4 = tt.broadcast %3 : tensor<1x32x!tt.ptr<f16>, #dot0> -> tensor<64x32x!tt.ptr<f16>, #dot0>
    // CHECK: %[[P:.*]] = arith.constant 512 : i32
    // CHECK: %[[H:.*]] = arith.constant 64 : i32
    // CHECK: ttig.2d_block_load_from_ptr %4, %[[P]], %[[H]] {row_major} {base_width = 64 : i32, ttig.block_io_stride = 256 : i64}
    %5 = tt.load %4 {ttig.block_io = "row_major", ttig.block_io_stride = 256 : i64} : tensor<64x32x!tt.ptr<f16>, #dot0>
    tt.return %5 : tensor<64x32xf16, #dot0>
  }
}

// -----

// COM: Rank-3 pointer load where getBlockIOTileSize assigns rowDim to a batch
// COM: dimension (dim 0). The pass must still convert it — the downstream
// COM: lowering handles batch dims by folding them into the base pointer.
#linear = #ttg.linear<{register = [[1, 0, 0], [2, 0, 0], [0, 0, 16], [0, 0, 32], [0, 0, 64]], lane = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8]], warp = [[0, 0, 0]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @rank3_batch_dim_load
  tt.func @rank3_batch_dim_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<4x1x128xf16, #linear> {
    %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<4x1x128x!tt.ptr<f16>, #linear>
    // CHECK: ttig.2d_block_load_from_ptr
    %1 = tt.load %0 {ttig.block_io = "row_major"} : tensor<4x1x128x!tt.ptr<f16>, #linear>
    tt.return %1 : tensor<4x1x128xf16, #linear>
  }
}

// -----

// COM: Pointer load with ttig.one_matrix_per_load attribute. The pass must
// COM: propagate this attribute to the resulting ttig.2d_block_load_from_ptr.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @pointer_load_one_matrix_per_load
  tt.func @pointer_load_one_matrix_per_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<64x32xf16, #dot0> {
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>> -> tensor<1x32xi32, #dot0>
    %2 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x32x!tt.ptr<f16>, #dot0>
    %3 = tt.addptr %2, %1 : tensor<1x32x!tt.ptr<f16>, #dot0>, tensor<1x32xi32, #dot0>
    %4 = tt.broadcast %3 : tensor<1x32x!tt.ptr<f16>, #dot0> -> tensor<64x32x!tt.ptr<f16>, #dot0>
    // CHECK: ttig.2d_block_load_from_ptr
    // CHECK-SAME: ttig.one_matrix_per_load
    %5 = tt.load %4 {ttig.block_io = "row_major", ttig.one_matrix_per_load} : tensor<64x32x!tt.ptr<f16>, #dot0>
    tt.return %5 : tensor<64x32xf16, #dot0>
  }
}

// -----

// COM: Non-broadcast transposed pointer load. Memory is column-major
// COM: (contiguous in dim 0), with a valid pitch along dim 1. This exercises
// COM: the transpose pitch fix: pitch must use colDim (the non-contiguous
// COM: memory direction) instead of rowDim.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @non_broadcast_column_major
  tt.func @non_broadcast_column_major(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<32x64xf16, #dot1> {
    // Build ptr[i, j] = arg0 + i + j * 128  (stride[0]=1, stride[1]=128)
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #dot1}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #dot1}>> -> tensor<32x1xi32, #dot1>
    %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #dot1}>>
    %3 = tt.expand_dims %2 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #dot1}>> -> tensor<1x64xi32, #dot1>
    %cst_stride = arith.constant dense<128> : tensor<1x64xi32, #dot1>
    %4 = arith.muli %3, %cst_stride : tensor<1x64xi32, #dot1>
    %5 = tt.broadcast %1 : tensor<32x1xi32, #dot1> -> tensor<32x64xi32, #dot1>
    %6 = tt.broadcast %4 : tensor<1x64xi32, #dot1> -> tensor<32x64xi32, #dot1>
    %7 = arith.addi %5, %6 : tensor<32x64xi32, #dot1>
    %8 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x64x!tt.ptr<f16>, #dot1>
    %9 = tt.addptr %8, %7 : tensor<32x64x!tt.ptr<f16>, #dot1>, tensor<32x64xi32, #dot1>
    // CHECK: %[[P:.*]] = arith.constant 256 : i32
    // CHECK: ttig.2d_block_load_from_ptr %9, %[[P]], %{{.*}} {column_major}
    %10 = tt.load %9 {ttig.block_io = "column_major"} : tensor<32x64x!tt.ptr<f16>, #dot1>
    tt.return %10 : tensor<32x64xf16, #dot1>
  }
}

// -----

// COM: Row-major pointer load whose row stride (`lda`) is a runtime scalar,
// COM: as in grouped GEMM. StrideAnalysis cannot fold it to a constant, but it
// COM: recovers the SSA value, so the pass feeds `lda * elemSize` as the pitch
// COM: operand (rather than a materialized arith.constant).
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @runtime_row_stride
  tt.func @runtime_row_stride(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %lda: i32 {tt.divisibility = 16 : i32}) -> tensor<64x32xf16, #dot0> {
    // Build ptr[i, j] = arg0 + i * lda + j  (row-major, runtime row stride lda).
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
    %ptr = tt.addptr %base, %off : tensor<64x32x!tt.ptr<f16>, #dot0>, tensor<64x32xi32, #dot0>
    // CHECK: %[[C2:.*]] = arith.constant 2 : i32
    // CHECK: %[[PITCH:.*]] = arith.muli %{{.*}}, %[[C2]] : i32
    // CHECK: %[[H:.*]] = arith.constant 8 : i32
    // CHECK: ttig.2d_block_load_from_ptr %{{.*}}, %[[PITCH]], %[[H]] {row_major} {base_width = 32 : i32}
    %5 = tt.load %ptr {ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #dot0>
    tt.return %5 : tensor<64x32xf16, #dot0>
  }
}

// -----

// COM: Grouped GEMM advances the pointer inside the K-loop, so the load sees a
// COM: loop-carried iter-arg. The runtime stride (`lda`) is loop-invariant, so
// COM: the dataflow fixpoint still recovers it and the in-loop load lowers to a
// COM: block load with the runtime pitch operand.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @loop_carried_runtime_stride
  tt.func @loop_carried_runtime_stride(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %lda: i32 {tt.divisibility = 16 : i32}, %lb: index, %ub: index, %step: index) -> tensor<64x32xf16, #dot0> {
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
    %ptr0 = tt.addptr %base, %off : tensor<64x32x!tt.ptr<f16>, #dot0>, tensor<64x32xi32, #dot0>
    %cst32 = arith.constant dense<32> : tensor<64x32xi32, #dot0>
    // CHECK: scf.for
    // CHECK: %[[PITCH:.*]] = arith.muli %{{.*}}, %{{.*}} : i32
    // CHECK: ttig.2d_block_load_from_ptr %{{.*}}, %[[PITCH]], %{{.*}} {row_major}
    %res = scf.for %i = %lb to %ub step %step iter_args(%p = %ptr0) -> (tensor<64x32x!tt.ptr<f16>, #dot0>) {
      %ld = tt.load %p {ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #dot0>
      %pn = tt.addptr %p, %cst32 : tensor<64x32x!tt.ptr<f16>, #dot0>, tensor<64x32xi32, #dot0>
      scf.yield %pn : tensor<64x32x!tt.ptr<f16>, #dot0>
    }
    %final = tt.load %res {ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #dot0>
    tt.return %final : tensor<64x32xf16, #dot0>
  }
}

// -----

// COM: Regression test for issue #8565. A row-only mask whose bound is a
// COM: runtime scalar -- `rows[:, None] < bound`, constant along K -- no
// COM: longer falls back to a masked gather: MaterializeBlockPointer permits
// COM: block IO for this mask shape (ttgi::matchRowLimitMask), and this pass
// COM: drops the mask/other operands in favor of a dynamic base_height
// COM: operand (`max(0, bound - rowBase)`), since the hardware's own
// COM: out-of-bounds zero-fill does what the mask would have done.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @row_limit_mask
  tt.func @row_limit_mask(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %lda: i32 {tt.divisibility = 16 : i32}, %row_base: i32, %bound: i32) -> tensor<64x32xf16, #dot0> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf16, #dot0>
    // Build ptr[i, j] = arg0 + i * lda + j (row-major, runtime row stride):
    // the row dim must actually vary per row (not be a broadcast pointer),
    // or matchRowLimitMask correctly excludes it to avoid disabling the
    // unrelated broadcast-row-replication mechanism.
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
    %4 = tt.addptr %base, %off : tensor<64x32x!tt.ptr<f16>, #dot0>, tensor<64x32xi32, #dot0>
    // rows = row_base + arange(0, 64); rows_valid = rows < bound.
    %5 = tt.splat %row_base : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>>
    %6 = arith.addi %5, %0 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>>
    %8 = tt.splat %bound : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>>
    %9 = arith.cmpi slt, %6, %8 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>>
    %10 = tt.expand_dims %9 {axis = 1 : i32} : tensor<64xi1, #ttg.slice<{dim = 1, parent = #dot0}>> -> tensor<64x1xi1, #dot0>
    %11 = tt.broadcast %10 : tensor<64x1xi1, #dot0> -> tensor<64x32xi1, #dot0>
    // CHECK: %[[DIFF:.*]] = arith.subi %arg3, %arg2 : i32
    // CHECK: %[[ZERO:.*]] = arith.constant 0 : i32
    // CHECK: %[[H:.*]] = arith.maxsi %[[DIFF]], %[[ZERO]] : i32
    // CHECK: ttig.2d_block_load_from_ptr %{{.*}}, %{{.*}}, %[[H]] {row_major}
    %12 = tt.load %4, %11, %cst {ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #dot0>
    tt.return %12 : tensor<64x32xf16, #dot0>
  }
}

// -----

// COM: Companion negative case for the row-limit mask test above: a mask
// COM: that varies along BOTH dims (also bounded along K) is not a row-limit
// COM: pattern, so it still falls back to a masked tt.load.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @row_and_col_limit_mask
  tt.func @row_and_col_limit_mask(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %lda: i32 {tt.divisibility = 16 : i32}, %row_bound: i32, %col_bound: i32) -> tensor<64x32xf16, #dot0> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf16, #dot0>
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
    %4 = tt.addptr %base, %off : tensor<64x32x!tt.ptr<f16>, #dot0>, tensor<64x32xi32, #dot0>
    %5 = tt.splat %col_bound : i32 -> tensor<1x32xi32, #dot0>
    %6 = arith.cmpi slt, %3, %5 : tensor<1x32xi32, #dot0>
    %7 = tt.broadcast %6 : tensor<1x32xi1, #dot0> -> tensor<64x32xi1, #dot0>
    %9 = tt.splat %row_bound : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>>
    %10 = arith.cmpi slt, %0, %9 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>>
    %11 = tt.expand_dims %10 {axis = 1 : i32} : tensor<64xi1, #ttg.slice<{dim = 1, parent = #dot0}>> -> tensor<64x1xi1, #dot0>
    %12 = tt.broadcast %11 : tensor<64x1xi1, #dot0> -> tensor<64x32xi1, #dot0>
    %mask = arith.andi %7, %12 : tensor<64x32xi1, #dot0>
    // CHECK: tt.load
    %13 = tt.load %4, %mask, %cst {ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #dot0>
    tt.return %13 : tensor<64x32xf16, #dot0>
  }
}

// -----

// COM: Env var TRITON_INTEL_ONE_MATRIX_PER_LOAD_BT=1 forces the attribute on
// COM: all loads, even those without it originally.
// RUN: env TRITON_INTEL_ONE_MATRIX_PER_LOAD_BT=1 triton-opt %s -split-input-file --tritonintelgpu-lower-to-2d-block-load | FileCheck %s --check-prefix=ENV-CHECK
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // ENV-CHECK-LABEL: tt.func @pointer_load_env_override
  tt.func @pointer_load_env_override(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<64x32xf16, #dot0> {
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>> -> tensor<1x32xi32, #dot0>
    %2 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x32x!tt.ptr<f16>, #dot0>
    %3 = tt.addptr %2, %1 : tensor<1x32x!tt.ptr<f16>, #dot0>, tensor<1x32xi32, #dot0>
    %4 = tt.broadcast %3 : tensor<1x32x!tt.ptr<f16>, #dot0> -> tensor<64x32x!tt.ptr<f16>, #dot0>
    // ENV-CHECK: ttig.2d_block_load_from_ptr
    // ENV-CHECK-SAME: ttig.one_matrix_per_load
    %5 = tt.load %4 {ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #dot0>
    tt.return %5 : tensor<64x32xf16, #dot0>
  }
}
