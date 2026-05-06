// RUN: triton-opt %s -split-input-file --tritonintelgpu-lower-to-2d-block-load | FileCheck %s --implicit-check-not=tt.descriptor_load

// COM: Descriptor-based load with block_io attribute should be converted to
// COM: ttig.2d_block_load with surface params from make_tensor_descriptor.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @tensor_descriptor_load
  tt.func @tensor_descriptor_load(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64) -> tensor<64x32xf16, #dot0> {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <64x32xf16>
    // CHECK: ttig.extract_desc
    // CHECK: %[[ELEM_BYTES:.*]] = arith.constant 2 : i32
    // CHECK: %[[WIDTH:.*]] = arith.trunci %{{.*}} : i64 to i32
    // CHECK: %[[BW:.*]] = arith.muli %[[WIDTH]], %[[ELEM_BYTES]]
    // CHECK: %[[HEIGHT:.*]] = arith.trunci %{{.*}} : i64 to i32
    // CHECK: %[[STRIDE:.*]] = arith.trunci %{{.*}} : i64 to i32
    // CHECK: %[[BP:.*]] = arith.muli %[[STRIDE]], %[[ELEM_BYTES]]
    // CHECK: ttig.2d_block_load %{{.*}}, %[[BW]], %[[HEIGHT]], %[[BP]][%{{.*}}, %{{.*}}] {row_major}
    %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #dot0>
    tt.return %0 : tensor<64x32xf16, #dot0>
  }
}

// -----

// COM: Column-major descriptor load. The memory_layout attribute is set to
// COM: column_major, which tells the LLVM lowering to use transposed block
// COM: reads. Surface params are the same as row_major (they describe physical
// COM: memory layout).
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @descriptor_load_column_major
  tt.func @descriptor_load_column_major(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64) -> tensor<32x64xf16, #dot1> {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <64x32xf16>
    // CHECK: ttig.extract_desc
    // CHECK: %[[ELEM_BYTES:.*]] = arith.constant 2 : i32
    // CHECK: %[[WIDTH:.*]] = arith.trunci %{{.*}} : i64 to i32
    // CHECK: %[[BW:.*]] = arith.muli %[[WIDTH]], %[[ELEM_BYTES]]
    // CHECK: %[[HEIGHT:.*]] = arith.trunci %{{.*}} : i64 to i32
    // CHECK: %[[STRIDE:.*]] = arith.trunci %{{.*}} : i64 to i32
    // CHECK: %[[BP:.*]] = arith.muli %[[STRIDE]], %[[ELEM_BYTES]]
    // CHECK: ttig.2d_block_load %{{.*}}, %[[BW]], %[[HEIGHT]], %[[BP]][%{{.*}}, %{{.*}}] {column_major}
    %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32] {ttig.block_io = "column_major"} : !tt.tensordesc<64x32xf16> -> tensor<32x64xf16, #dot1>
    tt.return %0 : tensor<32x64xf16, #dot1>
  }
}

// -----

// COM: Descriptor load with NaN padding. The pass sets the pad_nan attribute
// COM: on the ttig.2d_block_load op so the LLVM lowering builds boundary
// COM: masks and fills out-of-bounds elements with NaN.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @descriptor_load_pad_nan
  tt.func @descriptor_load_pad_nan(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64) -> tensor<64x32xf16, #dot0> {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] {padding = 2 : i32} : <f16>, <64x32xf16>
    // CHECK: ttig.2d_block_load
    // CHECK-SAME: {row_major, pad_nan}
    %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32] {ttig.block_io = "row_major", ttig.desc_padding = 2 : i32} : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #dot0>
    tt.return %0 : tensor<64x32xf16, #dot0>
  }
}

// -----

// COM: 3D descriptor load with batch dimension. The leading batch index should
// COM: be folded into the base pointer via tt.addptr, and the inner-2 dims
// COM: produce the 2D block load surface params.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4, 2], repCluster = [1, 1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @descriptor_load_batch
  tt.func @descriptor_load_batch(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i64, %arg5: i64, %batch_idx: i32) -> tensor<2x64x32xf16, #dot0> {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2, %arg3], [%arg4, %arg5, %c1_i64] : <f16>, <2x64x32xf16>
    // CHECK: ttig.extract_desc
    // CHECK: ttig.extract_desc
    // CHECK: %[[BATCH_EXT:.*]] = arith.extsi %arg6 : i32 to i64
    // CHECK: %[[BATCH_OFF:.*]] = arith.muli %[[BATCH_EXT]], %{{.*}}
    // CHECK: %[[ADJ_PTR:.*]] = tt.addptr %{{.*}}, %[[BATCH_OFF]]
    // CHECK: ttig.2d_block_load %[[ADJ_PTR]]
    %0 = tt.descriptor_load %desc[%batch_idx, %c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<2x64x32xf16> -> tensor<2x64x32xf16, #dot0>
    tt.return %0 : tensor<2x64x32xf16, #dot0>
  }
}

// -----

// COM: Rank-reducing descriptor load. The descriptor has rank 3 (with a leading
// COM: size-1 batch dim) but the result tensor has rank 2. The batch index is
// COM: folded into the base pointer via tt.addptr.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @descriptor_load_rank_reducing
  tt.func @descriptor_load_rank_reducing(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64, %batch_idx: i32) -> tensor<64x32xf16, #dot0> {
    %c1_i32 = arith.constant 1 : i32
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%c1_i32, %arg1, %arg2], [%arg3, %c1_i64, %c1_i64] : <f16>, <1x64x32xf16>
    // CHECK: ttig.extract_desc
    // CHECK: ttig.extract_desc
    // CHECK: %[[BATCH_EXT:.*]] = arith.extsi %arg4 : i32 to i64
    // CHECK: %[[BATCH_OFF:.*]] = arith.muli %[[BATCH_EXT]], %{{.*}}
    // CHECK: %[[ADJ_PTR:.*]] = tt.addptr %{{.*}}, %[[BATCH_OFF]]
    // CHECK: ttig.2d_block_load %[[ADJ_PTR]]
    %0 = tt.descriptor_load %desc[%batch_idx, %c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<1x64x32xf16> -> tensor<64x32xf16, #dot0>
    tt.return %0 : tensor<64x32xf16, #dot0>
  }
}

// -----

// COM: Loop-carried descriptor: the descriptor is rebuilt each iteration with
// COM: an advanced base pointer. The pass must use ttig.extract_desc on the
// COM: loop-body descriptor value (%arg5), NOT the pre-loop MakeTensorDescOp.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @descriptor_load_loop_carried
  tt.func @descriptor_load_loop_carried(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64, %N: i32) -> tensor<64x32xf16, #dot0> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16_i64 = arith.constant 16 : i64
    %c1_i64 = arith.constant 1 : i64
    %desc_init = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <64x32xf16>
    // The extract_desc must be on %arg5 (the loop-carried descriptor),
    // not on %desc_init. (%arg5 is the induction var, %arg6 is the iter_arg).
    // CHECK: scf.for
    // CHECK: ttig.extract_desc %arg6
    // CHECK: ttig.2d_block_load
    %result = scf.for %i = %c0_i32 to %N step %c1_i32 iter_args(%desc = %desc_init) -> (!tt.tensordesc<64x32xf16>) : i32 {
      %load = tt.descriptor_load %desc[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #dot0>
      // Advance: rebuild descriptor with a new base pointer.
      %new_base = tt.addptr %arg0, %c16_i64 : !tt.ptr<f16>, i64
      %new_desc = tt.make_tensor_descriptor %new_base, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <64x32xf16>
      scf.yield %new_desc : !tt.tensordesc<64x32xf16>
    }
    // Load from the final descriptor outside the loop.
    %final_load = tt.descriptor_load %result[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #dot0>
    tt.return %final_load : tensor<64x32xf16, #dot0>
  }
}
