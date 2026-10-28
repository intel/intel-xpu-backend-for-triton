// RUN: triton-opt %s -split-input-file --tritonintelgpu-lower-to-2d-block-load --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io", ttig.2d_block_io_base_alignment = 4 : i32} {
  // CHECK-LABEL: @store_signed_x
  tt.func public @store_signed_x(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf16, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %cm8 = arith.constant -8 : i32
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <32x32xf16, #dpas>
    // CHECK: llvm.sdiv
    // CHECK: triton_gen.2Dblockstore
    tt.descriptor_store %desc[%c0_i32, %cm8], %cst {ttig.block_io = "row_major"} : !tt.tensordesc<32x32xf16, #dpas>, tensor<32x32xf16, #dpas>
    tt.return
  }
}

// -----

// COM: The NaN mask is built in surface space from the logical (pre-compensation)
// COM: coordinates: the width bounds the contiguous dimension (dim 0 for a
// COM: column-major load) and the height the other.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: llvm.func spir_kernelcc @nan_mask_logical_surface_coordinates(
  // CHECK-SAME: {{.*}}, [[X:%[a-z0-9]+]]: i32, [[Y:%[a-z0-9]+]]: i32
  tt.func public @nan_mask_logical_surface_coordinates(%arg0: !tt.ptr<f16>, %x: i32, %y: i32) {
    %w = arith.constant 128 : i32
    %h = arith.constant 48 : i32
    %p = arith.constant 256 : i32
    // CHECK-DAG: %[[W:.*]] = llvm.mlir.constant(128 : i32) : i32
    // CHECK-DAG: %[[H:.*]] = llvm.mlir.constant(48 : i32) : i32
    // CHECK: %[[WELEMS:.*]] = llvm.udiv %[[W]], {{.*}} : i32
    // CHECK: llvm.add {{.*}}, [[X]] : i32
    // CHECK-NEXT: llvm.trunc %[[WELEMS]] : i32 to i32
    // CHECK: llvm.add {{.*}}, [[Y]] : i32
    // CHECK-NEXT: llvm.trunc %[[H]] : i32 to i32
    %0 = ttig.2d_block_load %arg0, %w, %h, %p[%x, %y] {column_major, pad_nan} : !tt.ptr<f16> -> tensor<64x32xf16, #dot1>
    tt.return
  }
}

// -----

// COM: A column index that may be negative but is a multiple of the message span
// COM: (32 columns for f16 DPAS A) cannot straddle column 0, so a negative one is
// COM: pushed out of bounds through the row coordinate.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: @load_redirect
  tt.func public @load_redirect(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i32, %arg5: i32 {tt.divisibility = 32 : i32}) {
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <64x32xf16>
    // CHECK: llvm.icmp "sge"
    // CHECK: llvm.select {{.*}} : i1, i32
    // CHECK: triton_gen.2Dblockload
    // CHECK-NOT: : vector<{{[0-9]+}}xi1>
    %load = tt.descriptor_load %desc[%arg4, %arg5] {ttig.block_io = "row_major"} : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #dot0>
    tt.return
  }
}

// -----

// COM: No negative-column handling for a constant non-negative column, or when the
// COM: target does not need the base-alignment compensation.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: @load_constant_column
  tt.func public @load_constant_column(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i32) {
    %c32_i32 = arith.constant 32 : i32
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <64x32xf16>
    // CHECK: llvm.inttoptr
    // CHECK-NOT: llvm.icmp "sge"
    // CHECK: triton_gen.2Dblockload
    // CHECK-NOT: llvm.select
    %load = tt.descriptor_load %desc[%arg4, %c32_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #dot0>
    tt.return
  }
}

// -----

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io", ttig.2d_block_io_base_alignment = 4 : i32} {
  // CHECK-LABEL: @load_no_compensation
  tt.func public @load_no_compensation(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i32, %arg5: i32) {
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <64x32xf16>
    // CHECK-NOT: llvm.icmp "sge"
    // CHECK: llvm.sdiv
    // CHECK: triton_gen.2Dblockload
    // CHECK-NOT: llvm.select
    %load = tt.descriptor_load %desc[%arg4, %arg5] {ttig.block_io = "row_major"} : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #dot0>
    tt.return
  }
}

// -----

// COM: The bounds predicate of a dropped batch dimension composes with the redirect
// COM: and is kept by the data mask.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: @load_rank_reducing_redirect
  tt.func public @load_rank_reducing_redirect(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i32, %arg5: i32 {tt.divisibility = 32 : i32}, %batch: i32) {
    %c1_i32 = arith.constant 1 : i32
    %c1_i64 = arith.constant 1 : i64
    %c32_i64 = arith.constant 32 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%c1_i32, %arg1, %arg2], [%arg3, %c32_i64, %c1_i64] : <f16>, <1x64x32xf16>
    // CHECK: llvm.icmp "sge"
    // CHECK: llvm.icmp "slt"
    // CHECK: [[BATCH:%.*]] = llvm.and
    // CHECK: [[COL:%.*]] = llvm.icmp "sge"
    // CHECK: [[BOTH:%.*]] = llvm.and [[BATCH]], [[COL]]
    // CHECK: llvm.select [[BOTH]], {{.*}} : i1, i32
    %load = tt.descriptor_load %desc[%batch, %arg4, %arg5] {ttig.block_io = "row_major"} : !tt.tensordesc<1x64x32xf16> -> tensor<64x32xf16, #dot0>
    tt.return
  }

  // CHECK-LABEL: @load_rank_reducing_mask
  tt.func public @load_rank_reducing_mask(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i32, %arg5: i32 {tt.divisibility = 8 : i32}, %batch: i32) {
    %c1_i32 = arith.constant 1 : i32
    %c1_i64 = arith.constant 1 : i64
    %c32_i64 = arith.constant 32 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%c1_i32, %arg1, %arg2], [%arg3, %c32_i64, %c1_i64] : <f16>, <1x64x32xf16>
    // CHECK: llvm.icmp "sge"
    // CHECK: llvm.icmp "slt"
    // CHECK: [[BATCH:%.*]] = llvm.and
    // CHECK: llvm.select [[BATCH]], {{.*}} : i1, i32
    // CHECK: triton_gen.2Dblockload
    // CHECK: llvm.select {{.*}} : vector<8xi1>, vector<8xf16>
    %load = tt.descriptor_load %desc[%batch, %arg4, %arg5] {ttig.block_io = "row_major"} : !tt.tensordesc<1x64x32xf16> -> tensor<64x32xf16, #dot0>
    tt.return
  }
}

// -----

// COM: The store span is in elements, so it includes the packing: a blocked layout
// COM: with two f16 per lane packs them into one 32-bit element and a 16-lane
// COM: message spans 32 columns. A 16-aligned index may straddle column 0.
#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: @packed_store_straddle_scatter
  tt.func public @packed_store_straddle_scatter(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64, %arg5: i32 {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<1.000000e+00> : tensor<64x64xf16, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <64x64xf16>
    // CHECK-NOT: triton_gen.2Dblockstore
    tt.descriptor_store %desc[%c0_i32, %arg5], %cst {ttig.block_io = "row_major"} : !tt.tensordesc<64x64xf16>, tensor<64x64xf16, #blocked>
    tt.return
  }

  // CHECK-LABEL: @packed_store_redirect_shift
  tt.func public @packed_store_redirect_shift(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64, %arg5: i32 {tt.divisibility = 32 : i32}) {
    %cst = arith.constant dense<1.000000e+00> : tensor<64x64xf16, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <64x64xf16>
    // CHECK-NOT: llvm.sdiv
    // CHECK: triton_gen.2Dblockstore
    // CHECK-NOT: llvm.sdiv
    tt.descriptor_store %desc[%c0_i32, %arg5], %cst {ttig.block_io = "row_major"} : !tt.tensordesc<64x64xf16>, tensor<64x64xf16, #blocked>
    tt.return
  }
}

// -----

// COM: The packed x of a redirected access is a shift: a negative x is dropped,
// COM: so rounding it differently from a signed division does not matter.
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: @packed_load_redirect_shift
  tt.func public @packed_load_redirect_shift(%arg0: !tt.ptr<f8E4M3FN>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i32 {tt.divisibility = 64 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f8E4M3FN>, <16x64xf8E4M3FN>
    // CHECK-NOT: llvm.sdiv
    // CHECK: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 32, {{.*}} transpose = true
    // CHECK-NOT: llvm.sdiv
    %load = tt.descriptor_load %desc[%c0_i32, %arg4] {ttig.block_io = "column_major"} : !tt.tensordesc<16x64xf8E4M3FN> -> tensor<64x16xf8E4M3FN, #dot1>
    tt.return
  }
}
