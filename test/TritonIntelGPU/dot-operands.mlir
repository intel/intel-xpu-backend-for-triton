// RUN: triton-opt %s -split-input-file -tritonintelgpu-optimize-dot-operands -canonicalize | FileCheck %s


#brow = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#bcol = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {ttig.support_2d_block_io, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @fuse_trans_with_descriptor_load
  // CHECK-NOT: tt.trans
  // CHECK: %[[LD:.*]] = tt.descriptor_load {{.*}} {ttig.block_io = "column_major"} : !tt.tensordesc<64x32xf16> -> tensor<32x64xf16, #[[BCOL:[a-z]+]]>
  // CHECK: ttg.convert_layout %[[LD]] : tensor<32x64xf16, #[[BCOL]]> -> tensor<32x64xf16, {{.*}}>
  tt.func @fuse_trans_with_descriptor_load(%ptr: !tt.ptr<f16>, %a: tensor<64x32xf16, #dot0>) -> tensor<64x64xf32, #dpas> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %c32_i64 = arith.constant 32 : i64
    %c = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #dpas>
    %desc = tt.make_tensor_descriptor %ptr, [%c64_i32, %c32_i32], [%c32_i64, %c1_i64] : <f16>, <64x32xf16>
    %ld = tt.descriptor_load %desc[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #brow>
    %tr = tt.trans %ld {order = array<i32: 1, 0>} : tensor<64x32xf16, #brow> -> tensor<32x64xf16, #bcol>
    %b = ttg.convert_layout %tr : tensor<32x64xf16, #bcol> -> tensor<32x64xf16, #dot1>
    %d = tt.dot %a, %b, %c, inputPrecision = tf32 : tensor<64x32xf16, #dot0> * tensor<32x64xf16, #dot1> -> tensor<64x64xf32, #dpas>
    tt.return %d : tensor<64x64xf32, #dpas>
  }
}

// -----

#brow = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#bcol = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
#bcolf32 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {ttig.support_2d_block_io, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @do_not_fuse_when_trans_has_multiple_uses
  // CHECK: tt.descriptor_load {{.*}} {ttig.block_io = "row_major"}
  // CHECK: tt.trans
  tt.func @do_not_fuse_when_trans_has_multiple_uses(%ptr: !tt.ptr<f16>, %a: tensor<64x32xf16, #dot0>) -> tensor<32x64xf32, #bcolf32> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %c32_i64 = arith.constant 32 : i64
    %c = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #dpas>
    %desc = tt.make_tensor_descriptor %ptr, [%c64_i32, %c32_i32], [%c32_i64, %c1_i64] : <f16>, <64x32xf16>
    %ld = tt.descriptor_load %desc[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #brow>
    %tr = tt.trans %ld {order = array<i32: 1, 0>} : tensor<64x32xf16, #brow> -> tensor<32x64xf16, #bcol>
    %b = ttg.convert_layout %tr : tensor<32x64xf16, #bcol> -> tensor<32x64xf16, #dot1>
    %d = tt.dot %a, %b, %c, inputPrecision = tf32 : tensor<64x32xf16, #dot0> * tensor<32x64xf16, #dot1> -> tensor<64x64xf32, #dpas>
    %use = arith.extf %tr : tensor<32x64xf16, #bcol> to tensor<32x64xf32, #bcolf32>
    tt.return %use : tensor<32x64xf32, #bcolf32>
  }
}

// -----

#brow = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#bcol = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {ttig.support_2d_block_io, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @do_not_fuse_when_load_is_not_row_major
  // CHECK: tt.descriptor_load {{.*}} {ttig.block_io = "column_major"}
  // CHECK: tt.trans
  tt.func @do_not_fuse_when_load_is_not_row_major(%ptr: !tt.ptr<f16>, %a: tensor<64x32xf16, #dot0>) -> tensor<64x64xf32, #dpas> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %c32_i64 = arith.constant 32 : i64
    %c = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #dpas>
    %desc = tt.make_tensor_descriptor %ptr, [%c64_i32, %c32_i32], [%c32_i64, %c1_i64] : <f16>, <64x32xf16>
    %ld = tt.descriptor_load %desc[%c0_i32, %c0_i32] {ttig.block_io = "column_major"} : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #brow>
    %tr = tt.trans %ld {order = array<i32: 1, 0>} : tensor<64x32xf16, #brow> -> tensor<32x64xf16, #bcol>
    %b = ttg.convert_layout %tr : tensor<32x64xf16, #bcol> -> tensor<32x64xf16, #dot1>
    %d = tt.dot %a, %b, %c, inputPrecision = tf32 : tensor<64x32xf16, #dot0> * tensor<32x64xf16, #dot1> -> tensor<64x64xf32, #dpas>
    tt.return %d : tensor<64x64xf32, #dpas>
  }
}
