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
    // CHECK: %[[W:.*]] = arith.constant 32 : i32
    // CHECK: ttig.2d_block_load_from_ptr %4, %[[P]], %[[H]], %[[W]] {row_major}
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
    // CHECK: %[[W:.*]] = arith.constant 32 : i32
    // CHECK: ttig.2d_block_load_from_ptr %4, %[[P]], %[[H]], %[[W]] {column_major}
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
    // CHECK: %[[W:.*]] = arith.constant 64 : i32
    // CHECK: ttig.2d_block_load_from_ptr %4, %[[P]], %[[H]], %[[W]] {row_major} {ttig.block_io_stride = 256 : i64}
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
    // CHECK: ttig.2d_block_load_from_ptr %9, %[[P]], %{{.*}}, %{{.*}} {column_major}
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
    // CHECK: %[[W:.*]] = arith.constant 32 : i32
    // CHECK: ttig.2d_block_load_from_ptr %{{.*}}, %[[PITCH]], %[[H]], %[[W]] {row_major}
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
    // CHECK: ttig.2d_block_load_from_ptr %{{.*}}, %[[PITCH]], %{{.*}}, %{{.*}} {row_major}
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

// COM: Loop fusion (tl.range(..., flatten=True)) carries the pointer through an
// COM: iter-arg with a poison init and rebuilds it under `scf.if` on the first
// COM: iteration. The poison is never read, so the runtime pitch survives
// COM: (issue #8687).
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @fused_loop_poison_init_runtime_stride
  // CHECK-SAME: %[[LDA:[a-z0-9_]+]]: i32 {tt.divisibility = 16 : i32}, %{{[a-z0-9_]+}}: i32)
  tt.func @fused_loop_poison_init_runtime_stride(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %lda: i32 {tt.divisibility = 16 : i32}, %n: i32) -> tensor<64x32xf16, #dot0> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %poison = ub.poison : tensor<64x32x!tt.ptr<f16>, #dot0>
    %cst_init = arith.constant dense<0.000000e+00> : tensor<64x32xf16, #dot0>
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
    // CHECK: %[[C2:.*]] = arith.constant 2 : i32
    // CHECK: %[[PITCH:.*]] = arith.muli %[[LDA]], %[[C2]] : i32
    // CHECK: ttig.2d_block_load_from_ptr %{{.*}}, %[[PITCH]], %{{.*}}, %{{.*}} {row_major}
    %res:2 = scf.for %i = %c0_i32 to %n step %c1_i32 iter_args(%p = %poison, %acc = %cst_init) -> (tensor<64x32x!tt.ptr<f16>, #dot0>, tensor<64x32xf16, #dot0>) : i32 {
      %first = arith.cmpi eq, %i, %c0_i32 : i32
      %q = scf.if %first -> (tensor<64x32x!tt.ptr<f16>, #dot0>) {
        scf.yield %ptr0 : tensor<64x32x!tt.ptr<f16>, #dot0>
      } else {
        scf.yield %p : tensor<64x32x!tt.ptr<f16>, #dot0>
      }
      %ld = tt.load %q {ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #dot0>
      %pn = tt.addptr %q, %cst32 : tensor<64x32x!tt.ptr<f16>, #dot0>, tensor<64x32xi32, #dot0>
      scf.yield %pn, %ld : tensor<64x32x!tt.ptr<f16>, #dot0>, tensor<64x32xf16, #dot0>
    }
    tt.return %res#1 : tensor<64x32xf16, #dot0>
  }
}

// -----

// COM: Same shape, but the row stride is loaded in every iteration. The pointer
// COM: is only rebuilt on the first one, so this iteration's stride is not its
// COM: pitch and the load is left alone.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @fused_loop_loop_variant_stride
  tt.func @fused_loop_loop_variant_stride(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %ldas: !tt.ptr<i32>, %n: i32) -> tensor<64x32xf16, #dot0> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %poison = ub.poison : tensor<64x32x!tt.ptr<f16>, #dot0>
    %cst_init = arith.constant dense<0.000000e+00> : tensor<64x32xf16, #dot0>
    // CHECK: scf.for
    // CHECK: tt.load %{{.*}} : !tt.ptr<i32>
    // CHECK-NOT: ttig.2d_block_load_from_ptr
    // CHECK: tt.load %{{.*}} {ttig.block_io = "row_major"}
    %res:2 = scf.for %i = %c0_i32 to %n step %c1_i32 iter_args(%p = %poison, %acc = %cst_init) -> (tensor<64x32x!tt.ptr<f16>, #dot0>, tensor<64x32xf16, #dot0>) : i32 {
      %ldap = tt.addptr %ldas, %i : !tt.ptr<i32>, i32
      %lda = tt.load %ldap : !tt.ptr<i32>
      %first = arith.cmpi eq, %i, %c0_i32 : i32
      %q = scf.if %first -> (tensor<64x32x!tt.ptr<f16>, #dot0>) {
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
        scf.yield %ptr0 : tensor<64x32x!tt.ptr<f16>, #dot0>
      } else {
        scf.yield %p : tensor<64x32x!tt.ptr<f16>, #dot0>
      }
      %ld = tt.load %q {ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #dot0>
      scf.yield %q, %ld : tensor<64x32x!tt.ptr<f16>, #dot0>, tensor<64x32xf16, #dot0>
    }
    tt.return %res#1 : tensor<64x32xf16, #dot0>
  }
}

// -----

// COM: Regression test for issue #8565. A row-only mask `rows[:, None] <
// COM: bound` with a runtime bound no longer falls back to a masked gather:
// COM: this pass drops the mask/other for a dynamic base_height operand
// COM: (`max(0, bound - rowBase)`, unit-tested in ComputeRowLimitHeightTest).
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @row_limit_mask
  tt.func @row_limit_mask(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %row_base: i32, %bound: i32) -> tensor<64x32xf16, #dot0> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf16, #dot0>
    %cst_stride = arith.constant dense<32> : tensor<64x1xi32, #dot0>
    // ptr[i, j] = arg0 + i * 32 (row dim must vary, not be broadcast; the
    // column dim is left degenerate since this pattern doesn't care about it).
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>> -> tensor<64x1xi32, #dot0>
    %row = arith.muli %1, %cst_stride : tensor<64x1xi32, #dot0>
    %rowb = tt.broadcast %row : tensor<64x1xi32, #dot0> -> tensor<64x32xi32, #dot0>
    %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x32x!tt.ptr<f16>, #dot0>
    %4 = tt.addptr %base, %rowb : tensor<64x32x!tt.ptr<f16>, #dot0>, tensor<64x32xi32, #dot0>
    // rows = row_base + arange(0, 64); rows_valid = rows < bound.
    %5 = tt.splat %row_base : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>>
    %6 = arith.addi %5, %0 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>>
    %8 = tt.splat %bound : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>>
    %9 = arith.cmpi slt, %6, %8 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>>
    %10 = tt.expand_dims %9 {axis = 1 : i32} : tensor<64xi1, #ttg.slice<{dim = 1, parent = #dot0}>> -> tensor<64x1xi1, #dot0>
    %11 = tt.broadcast %10 : tensor<64x1xi1, #dot0> -> tensor<64x32xi1, #dot0>
    // CHECK: %[[H:.*]] = arith.trunci %{{.*}} : i64 to i32
    // CHECK: ttig.2d_block_load_from_ptr %{{.*}}, %{{.*}}, %[[H]], %{{.*}} {row_major}
    // CHECK-SAME: ttig.block_io_runtime_row_limit
    %12 = tt.load %4, %11, %cst {ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #dot0>
    tt.return %12 : tensor<64x32xf16, #dot0>
  }
}

// -----

// COM: i64 counterpart of @row_limit_mask (issue #8565, Helion computes the
// COM: row index in int64). `bound - rowBase` may overflow i64, so the height
// COM: is `(bound <=s rowBase) ? 0 : minui(bound - rowBase, 64)`, with no
// COM: sign extension of the already-i64 operands.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @row_limit_mask_i64
  // CHECK-SAME: %[[ROW_BASE:[^:]*]]: i64, %[[BOUND:[^:]*]]: i64
  tt.func @row_limit_mask_i64(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %row_base: i64, %bound: i64) -> tensor<64x32xf16, #dot0> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf16, #dot0>
    %cst_stride = arith.constant dense<32> : tensor<64x1xi32, #dot0>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>> -> tensor<64x1xi32, #dot0>
    %row = arith.muli %1, %cst_stride : tensor<64x1xi32, #dot0>
    %rowb = tt.broadcast %row : tensor<64x1xi32, #dot0> -> tensor<64x32xi32, #dot0>
    %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x32x!tt.ptr<f16>, #dot0>
    %4 = tt.addptr %base, %rowb : tensor<64x32x!tt.ptr<f16>, #dot0>, tensor<64x32xi32, #dot0>
    // rows = row_base + sext(arange(0, 64)); rows_valid = rows < bound.
    %r0_64 = arith.extsi %0 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>> to tensor<64xi64, #ttg.slice<{dim = 1, parent = #dot0}>>
    %5 = tt.splat %row_base : i64 -> tensor<64xi64, #ttg.slice<{dim = 1, parent = #dot0}>>
    %6 = arith.addi %5, %r0_64 : tensor<64xi64, #ttg.slice<{dim = 1, parent = #dot0}>>
    %8 = tt.splat %bound : i64 -> tensor<64xi64, #ttg.slice<{dim = 1, parent = #dot0}>>
    %9 = arith.cmpi slt, %6, %8 : tensor<64xi64, #ttg.slice<{dim = 1, parent = #dot0}>>
    %10 = tt.expand_dims %9 {axis = 1 : i32} : tensor<64xi1, #ttg.slice<{dim = 1, parent = #dot0}>> -> tensor<64x1xi1, #dot0>
    %11 = tt.broadcast %10 : tensor<64x1xi1, #dot0> -> tensor<64x32xi1, #dot0>
    // CHECK-NOT: arith.extsi %[[BOUND]]
    // CHECK-NOT: arith.extsi %[[ROW_BASE]]
    // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : i64
    // CHECK-DAG: %[[EXTENT:.*]] = arith.constant 64 : i64
    // CHECK-DAG: %[[EMPTY:.*]] = arith.cmpi sle, %[[BOUND]], %[[ROW_BASE]] : i64
    // CHECK-DAG: %[[DIFF:.*]] = arith.subi %[[BOUND]], %[[ROW_BASE]] : i64
    // CHECK: %[[MIN:.*]] = arith.minui %[[DIFF]], %[[EXTENT]] : i64
    // CHECK: %[[SEL:.*]] = arith.select %[[EMPTY]], %[[ZERO]], %[[MIN]] : i64
    // CHECK: %[[H:.*]] = arith.trunci %[[SEL]] : i64 to i32
    // CHECK: ttig.2d_block_load_from_ptr %{{.*}}, %{{.*}}, %[[H]], %{{.*}} {row_major}
    // CHECK-SAME: ttig.block_io_runtime_row_limit
    %12 = tt.load %4, %11, %cst {ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #dot0>
    tt.return %12 : tensor<64x32xf16, #dot0>
  }
}

// -----

// COM: An i32 row index zero-extended to i64 after the add is not a row-limit
// COM: pattern: the base would lose its zero extension (0x80000000 is +2^31
// COM: there, but -2^31 sign-extended), and the height would be wrong.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @row_limit_mask_i32_base_zext_i64
  tt.func @row_limit_mask_i32_base_zext_i64(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %row_base: i32, %bound: i64) -> tensor<64x32xf16, #dot0> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf16, #dot0>
    %cst_stride = arith.constant dense<32> : tensor<64x1xi32, #dot0>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>> -> tensor<64x1xi32, #dot0>
    %row = arith.muli %1, %cst_stride : tensor<64x1xi32, #dot0>
    %rowb = tt.broadcast %row : tensor<64x1xi32, #dot0> -> tensor<64x32xi32, #dot0>
    %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x32x!tt.ptr<f16>, #dot0>
    %4 = tt.addptr %base, %rowb : tensor<64x32x!tt.ptr<f16>, #dot0>, tensor<64x32xi32, #dot0>
    %5 = tt.splat %row_base : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>>
    %6 = arith.addi %5, %0 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>>
    %r6_64 = arith.extui %6 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>> to tensor<64xi64, #ttg.slice<{dim = 1, parent = #dot0}>>
    %8 = tt.splat %bound : i64 -> tensor<64xi64, #ttg.slice<{dim = 1, parent = #dot0}>>
    %9 = arith.cmpi slt, %r6_64, %8 : tensor<64xi64, #ttg.slice<{dim = 1, parent = #dot0}>>
    %10 = tt.expand_dims %9 {axis = 1 : i32} : tensor<64xi1, #ttg.slice<{dim = 1, parent = #dot0}>> -> tensor<64x1xi1, #dot0>
    %11 = tt.broadcast %10 : tensor<64x1xi1, #dot0> -> tensor<64x32xi1, #dot0>
    // CHECK-NOT: arith.minui
    // CHECK-NOT: arith.minsi
    // CHECK: %[[ONE:.*]] = arith.constant 1 : i32
    // CHECK: ttig.2d_block_load_from_ptr %{{.*}}, %{{.*}}, %[[ONE]], %{{.*}}, %{{.*}}, %{{.*}} {row_major}
    // CHECK-NOT: ttig.block_io_runtime_row_limit
    %12 = tt.load %4, %11, %cst {ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #dot0>
    tt.return %12 : tensor<64x32xf16, #dot0>
  }
}

// -----

// COM: An unsigned compare on an i64 row index is still not a row-limit
// COM: pattern: no runtime base_height, the mask and `other` are forwarded to
// COM: a per-row (base_height = 1) block load instead.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @row_limit_mask_i64_unsigned
  tt.func @row_limit_mask_i64_unsigned(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %row_base: i64, %bound: i64) -> tensor<64x32xf16, #dot0> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf16, #dot0>
    %cst_stride = arith.constant dense<32> : tensor<64x1xi32, #dot0>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>> -> tensor<64x1xi32, #dot0>
    %row = arith.muli %1, %cst_stride : tensor<64x1xi32, #dot0>
    %rowb = tt.broadcast %row : tensor<64x1xi32, #dot0> -> tensor<64x32xi32, #dot0>
    %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x32x!tt.ptr<f16>, #dot0>
    %4 = tt.addptr %base, %rowb : tensor<64x32x!tt.ptr<f16>, #dot0>, tensor<64x32xi32, #dot0>
    %r0_64 = arith.extsi %0 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>> to tensor<64xi64, #ttg.slice<{dim = 1, parent = #dot0}>>
    %5 = tt.splat %row_base : i64 -> tensor<64xi64, #ttg.slice<{dim = 1, parent = #dot0}>>
    %6 = arith.addi %5, %r0_64 : tensor<64xi64, #ttg.slice<{dim = 1, parent = #dot0}>>
    %8 = tt.splat %bound : i64 -> tensor<64xi64, #ttg.slice<{dim = 1, parent = #dot0}>>
    %9 = arith.cmpi ult, %6, %8 : tensor<64xi64, #ttg.slice<{dim = 1, parent = #dot0}>>
    %10 = tt.expand_dims %9 {axis = 1 : i32} : tensor<64xi1, #ttg.slice<{dim = 1, parent = #dot0}>> -> tensor<64x1xi1, #dot0>
    %11 = tt.broadcast %10 : tensor<64x1xi1, #dot0> -> tensor<64x32xi1, #dot0>
    // CHECK-NOT: arith.minui
    // CHECK: %[[ONE:.*]] = arith.constant 1 : i32
    // CHECK: ttig.2d_block_load_from_ptr %{{.*}}, %{{.*}}, %[[ONE]], %{{.*}}, %{{.*}}, %{{.*}} {row_major}
    // CHECK-NOT: ttig.block_io_runtime_row_limit
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
  tt.func @row_and_col_limit_mask(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %row_bound: i32, %col_bound: i32) -> tensor<64x32xf16, #dot0> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf16, #dot0>
    %cst_stride = arith.constant dense<32> : tensor<64x1xi32, #dot0>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>> -> tensor<64x1xi32, #dot0>
    %row = arith.muli %1, %cst_stride : tensor<64x1xi32, #dot0>
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>>
    %3 = tt.expand_dims %2 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>> -> tensor<1x32xi32, #dot0>
    %rowb = tt.broadcast %row : tensor<64x1xi32, #dot0> -> tensor<64x32xi32, #dot0>
    %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x32x!tt.ptr<f16>, #dot0>
    %4 = tt.addptr %base, %rowb : tensor<64x32x!tt.ptr<f16>, #dot0>, tensor<64x32xi32, #dot0>
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

// COM: Rank-3 row-limit mask ANDed with a batch-only predicate (degenerate
// COM: along M and K): the row half still becomes `base_height`, and the
// COM: batch half survives as `extraPred`, forwarded as mask/other.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4, 2], repCluster = [1, 1, 1], A = [1, 8, 16], B = [1, 16, 16], C = [1, 8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#sK = #ttg.slice<{dim = 2, parent = #dot0}>
#sM = #ttg.slice<{dim = 1, parent = #dot0}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @row_limit_mask_rank3_batch_pred
  tt.func @row_limit_mask_rank3_batch_pred(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %row_base: i32 {tt.divisibility = 16 : i32}, %bound: i32 {tt.divisibility = 16 : i32}, %batch_bound: i32 {tt.divisibility = 16 : i32}) -> tensor<4x64x32xf16, #dot0> {
    %zero = arith.constant dense<0.000000e+00> : tensor<4x64x32xf16, #dot0>
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
    // CHECK: %[[BATCHMASK:.*]] = tt.broadcast %{{.*}} : tensor<4x1x1xi1, #{{.*}}> -> tensor<4x64x32xi1, #{{.*}}>
    %batchmask = tt.broadcast %bc2 : tensor<4x1x1xi1, #dot0> -> tensor<4x64x32xi1, #dot0>
    %mask = arith.andi %rowmask, %batchmask : tensor<4x64x32xi1, #dot0>
    // Runtime height (not fixed), with the batch predicate forwarded as
    // mask/other instead of folded into or blocking base_height.
    // CHECK: %[[H:.*]] = arith.trunci %{{.*}} : i64 to i32
    // CHECK: ttig.2d_block_load_from_ptr %{{.*}}, %{{.*}}, %[[H]], %{{.*}}, %[[BATCHMASK]], %{{.*}} {row_major}
    // CHECK-SAME: ttig.block_io_runtime_row_limit
    %v = tt.load %ptr, %mask, %zero {ttig.block_io = "row_major"} : tensor<4x64x32x!tt.ptr<f16>, #dot0>
    tt.return %v : tensor<4x64x32xf16, #dot0>
  }
}

// -----

// COM: Same as @row_limit_mask_rank3_sunk_and, but with the AND sunk below
// COM: the final broadcast, as `tt-reorder-broadcast` leaves it:
// COM: broadcast(and(broadcast(row), broadcast(batch))). `extraPred` is then
// COM: narrow, and is broadcast to the full shape straight from its innermost
// COM: source before being forwarded as mask/other.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4, 2], repCluster = [1, 1, 1], A = [1, 8, 16], B = [1, 16, 16], C = [1, 8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#sK = #ttg.slice<{dim = 2, parent = #dot0}>
#sM = #ttg.slice<{dim = 1, parent = #dot0}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @row_limit_mask_rank3_sunk_and
  tt.func @row_limit_mask_rank3_sunk_and(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %row_base: i32 {tt.divisibility = 16 : i32}, %bound: i32 {tt.divisibility = 16 : i32}, %batch_bound: i32 {tt.divisibility = 16 : i32}) -> tensor<4x64x32xf16, #dot0> {
    %zero = arith.constant dense<0.000000e+00> : tensor<4x64x32xf16, #dot0>
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
    // CHECK: %[[BATCHMASK:.*]] = tt.broadcast %{{.*}} : tensor<4x1x1xi1, #{{.*}}> -> tensor<4x64x32xi1, #{{.*}}>
    // Runtime height (not fixed), with the batch predicate forwarded as
    // mask/other instead of folded into or blocking base_height.
    // CHECK: %[[H:.*]] = arith.trunci %{{.*}} : i64 to i32
    // CHECK: ttig.2d_block_load_from_ptr %{{.*}}, %{{.*}}, %[[H]], %{{.*}}, %[[BATCHMASK]], %{{.*}} {row_major}
    // CHECK-SAME: ttig.block_io_runtime_row_limit
    %v = tt.load %ptr, %mask, %zero {ttig.block_io = "row_major"} : tensor<4x64x32x!tt.ptr<f16>, #dot0>
    tt.return %v : tensor<4x64x32xf16, #dot0>
  }
}

// -----

// COM: Column-major counterpart of @row_limit_mask: the pitch dim is dim 1
// COM: here, and the pitch operand must still come from `lda`, not the bound.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @row_limit_mask_column_major
  tt.func @row_limit_mask_column_major(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %lda: i32 {tt.divisibility = 16 : i32}, %col_base: i32, %bound: i32) -> tensor<32x64xf16, #dot1> {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x64xf16, #dot1>
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #dot1}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #dot1}>> -> tensor<32x1xi32, #dot1>
    %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #dot1}>>
    %3 = tt.expand_dims %2 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #dot1}>> -> tensor<1x64xi32, #dot1>
    %lda_splat = tt.splat %lda : i32 -> tensor<1x64xi32, #dot1>
    %col = arith.muli %3, %lda_splat : tensor<1x64xi32, #dot1>
    %rowb = tt.broadcast %1 : tensor<32x1xi32, #dot1> -> tensor<32x64xi32, #dot1>
    %colb = tt.broadcast %col : tensor<1x64xi32, #dot1> -> tensor<32x64xi32, #dot1>
    %off = arith.addi %rowb, %colb : tensor<32x64xi32, #dot1>
    %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x64x!tt.ptr<f16>, #dot1>
    %4 = tt.addptr %base, %off : tensor<32x64x!tt.ptr<f16>, #dot1>, tensor<32x64xi32, #dot1>
    %5 = tt.splat %col_base : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #dot1}>>
    %6 = arith.addi %5, %2 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #dot1}>>
    %7 = tt.splat %bound : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #dot1}>>
    %8 = arith.cmpi slt, %6, %7 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #dot1}>>
    %9 = tt.expand_dims %8 {axis = 0 : i32} : tensor<64xi1, #ttg.slice<{dim = 0, parent = #dot1}>> -> tensor<1x64xi1, #dot1>
    %10 = tt.broadcast %9 : tensor<1x64xi1, #dot1> -> tensor<32x64xi1, #dot1>
    // Pitch must still come from `lda`, not get confused with the bound.
    // CHECK: %[[PITCH:.*]] = arith.muli %arg1, %{{.*}} : i32
    // CHECK: %[[H:.*]] = arith.trunci %{{.*}} : i64 to i32
    // CHECK: ttig.2d_block_load_from_ptr %{{.*}}, %[[PITCH]], %[[H]], %{{.*}} {column_major}
    // CHECK-SAME: ttig.block_io_runtime_row_limit
    %11 = tt.load %4, %10, %cst {ttig.block_io = "column_major"} : tensor<32x64x!tt.ptr<f16>, #dot1>
    tt.return %11 : tensor<32x64xf16, #dot1>
  }
}

// -----

// COM: B-operand K-tail mask: `offs_k[:, None] < K - k * BLOCK_K`. offs_k is
// COM: a bare arange (no row base); the bound is a computed scalar, not a
// COM: bare splat argument.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @row_limit_mask_b_operand_k_tail
  tt.func @row_limit_mask_b_operand_k_tail(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %lda: i32 {tt.divisibility = 16 : i32}, %k_total: i32, %k_block_offset: i32) -> tensor<64x32xf16, #dot1> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf16, #dot1>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot1}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot1}>> -> tensor<64x1xi32, #dot1>
    %lda_splat = tt.splat %lda : i32 -> tensor<64x1xi32, #dot1>
    %row = arith.muli %1, %lda_splat : tensor<64x1xi32, #dot1>
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot1}>>
    %3 = tt.expand_dims %2 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot1}>> -> tensor<1x32xi32, #dot1>
    %rowb = tt.broadcast %row : tensor<64x1xi32, #dot1> -> tensor<64x32xi32, #dot1>
    %colb = tt.broadcast %3 : tensor<1x32xi32, #dot1> -> tensor<64x32xi32, #dot1>
    %off = arith.addi %rowb, %colb : tensor<64x32xi32, #dot1>
    %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x32x!tt.ptr<f16>, #dot1>
    %4 = tt.addptr %base, %off : tensor<64x32x!tt.ptr<f16>, #dot1>, tensor<64x32xi32, #dot1>
    %5 = arith.subi %k_total, %k_block_offset : i32
    %6 = tt.splat %5 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot1}>>
    %7 = arith.cmpi slt, %0, %6 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot1}>>
    %8 = tt.expand_dims %7 {axis = 1 : i32} : tensor<64xi1, #ttg.slice<{dim = 1, parent = #dot1}>> -> tensor<64x1xi1, #dot1>
    %9 = tt.broadcast %8 : tensor<64x1xi1, #dot1> -> tensor<64x32xi1, #dot1>
    // Confirms the matcher accepts a computed bound and a bare arange.
    // CHECK: %[[KREM:.*]] = arith.subi %arg2, %arg3 : i32
    // CHECK: %[[H:.*]] = arith.trunci %{{.*}} : i64 to i32
    // CHECK: ttig.2d_block_load_from_ptr %{{.*}}, %{{.*}}, %[[H]], %{{.*}} {row_major}
    // CHECK-SAME: ttig.block_io_runtime_row_limit
    %10 = tt.load %4, %9, %cst {ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #dot1>
    tt.return %10 : tensor<64x32xf16, #dot1>
  }
}

// -----

// COM: Regression test: a rank-3 row-limit mask whose bounded dim (dim 1) is
// COM: NOT the dim getBlockIOLoadTileSize picks as the tile's pitch dim (dim
// COM: 0, a batch dim, with no register packing). Folding the bound into
// COM: base_height would apply it to the wrong dim, so the pass must fall
// COM: back to per-element masking. Dim 0 is left a pure broadcast (no
// COM: address term): the bug is about tile dim selection, not addressing.
#blocked = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 16], warpsPerCTA = [1, 8, 1], order = [2, 1, 0]}>
#s2 = #ttg.slice<{dim = 2, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @row_limit_mask_wrong_pitch_dim
  tt.func @row_limit_mask_wrong_pitch_dim(%p: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %row_base: i32 {tt.divisibility = 16 : i32}, %bound: i32 {tt.divisibility = 16 : i32}) -> tensor<2x8x64xf16, #blocked> {
    %zero = arith.constant dense<0.000000e+00> : tensor<2x8x64xf16, #blocked>
    %c64 = arith.constant dense<64> : tensor<1x8x1xi32, #blocked>
    %r1 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #s2}>>
    %e1a = tt.expand_dims %r1 {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #s2}>> -> tensor<1x8xi32, #s2>
    %e1 = tt.expand_dims %e1a {axis = 2 : i32} : tensor<1x8xi32, #s2> -> tensor<1x8x1xi32, #blocked>
    %o1 = arith.muli %e1, %c64 : tensor<1x8x1xi32, #blocked>
    %b1 = tt.broadcast %o1 : tensor<1x8x1xi32, #blocked> -> tensor<2x8x64xi32, #blocked>
    %ps = tt.splat %p : !tt.ptr<f16> -> tensor<2x8x64x!tt.ptr<f16>, #blocked>
    %ptr = tt.addptr %ps, %b1 : tensor<2x8x64x!tt.ptr<f16>, #blocked>, tensor<2x8x64xi32, #blocked>
    %rb = tt.splat %row_base : i32 -> tensor<8xi32, #ttg.slice<{dim = 0, parent = #s2}>>
    %rows = arith.addi %rb, %r1 : tensor<8xi32, #ttg.slice<{dim = 0, parent = #s2}>>
    %bs = tt.splat %bound : i32 -> tensor<8xi32, #ttg.slice<{dim = 0, parent = #s2}>>
    %cmp = arith.cmpi slt, %rows, %bs : tensor<8xi32, #ttg.slice<{dim = 0, parent = #s2}>>
    %m1 = tt.expand_dims %cmp {axis = 0 : i32} : tensor<8xi1, #ttg.slice<{dim = 0, parent = #s2}>> -> tensor<1x8xi1, #s2>
    %m2 = tt.expand_dims %m1 {axis = 2 : i32} : tensor<1x8xi1, #s2> -> tensor<1x8x1xi1, #blocked>
    %mask = tt.broadcast %m2 : tensor<1x8x1xi1, #blocked> -> tensor<2x8x64xi1, #blocked>
    // CHECK: ttig.2d_block_load_from_ptr
    // CHECK-SAME: %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}
    // CHECK-NOT: ttig.block_io_runtime_row_limit
    %v = tt.load %ptr, %mask, %zero {ttig.block_io = "row_major"} : tensor<2x8x64x!tt.ptr<f16>, #blocked>
    tt.return %v : tensor<2x8x64xf16, #blocked>
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
