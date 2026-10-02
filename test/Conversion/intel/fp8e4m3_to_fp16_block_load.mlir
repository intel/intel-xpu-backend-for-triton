// RUN: triton-opt %s --tritonintelgpu-lower-to-2d-block-load --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm --canonicalize | FileCheck %s

// COM: Dot operand B: 32-bit loads read bytes from dwords for K >= 64 and per element for K = 32;
// COM: 8-bit loads with K = 32 stay packed in pairs.
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.target_arch = "spir64"} {
  // CHECK-LABEL: @fp8_upcast_from_block_load
  tt.func public @fp8_upcast_from_block_load(%arg0: !tt.ptr<f8E4M3FN>, %arg1: i32, %arg2: i32) -> (tensor<64x16xf16, #dot1>, tensor<32x16xf16, #dot1>, tensor<32x16xf16, #dot1>) {
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c64_i32 = arith.constant 64 : i32
    %k64_desc = tt.make_tensor_descriptor %arg0, [%c64_i32, %c64_i32], [%c64_i64, %c1_i64] : <f8E4M3FN>, <16x64xf8E4M3FN>
    %k32_desc = tt.make_tensor_descriptor %arg0, [%c64_i32, %c64_i32], [%c64_i64, %c1_i64] : <f8E4M3FN>, <16x32xf8E4M3FN>
    %row_desc = tt.make_tensor_descriptor %arg0, [%c64_i32, %c64_i32], [%c64_i64, %c1_i64] : <f8E4M3FN>, <32x16xf8E4M3FN>
    // CHECK-COUNT-3: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 32, {{.*}} transpose = true, {{.*}} -> vector<8xi32>
    // CHECK: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 8, {{.*}} transpose = false, {{.*}} -> vector<32xi8>
    %k64 = tt.descriptor_load %k64_desc[%arg1, %arg2] {ttig.block_io = "column_major"} : !tt.tensordesc<16x64xf8E4M3FN> -> tensor<64x16xf8E4M3FN, #dot1>
    %k32 = tt.descriptor_load %k32_desc[%arg1, %arg2] {ttig.block_io = "column_major"} : !tt.tensordesc<16x32xf8E4M3FN> -> tensor<32x16xf8E4M3FN, #dot1>
    %row = tt.descriptor_load %row_desc[%arg1, %arg2] {ttig.block_io = "row_major"} : !tt.tensordesc<32x16xf8E4M3FN> -> tensor<32x16xf8E4M3FN, #dot1>
    // CHECK-NOT: llvm.zext {{.*}} : i8 to i16
    // CHECK-COUNT-64: llvm.trunc {{.*}} : i32 to i16
    // CHECK-NOT: llvm.trunc {{.*}} : i32 to i16
    // CHECK-COUNT-32: llvm.zext {{.*}} : i8 to i16
    // CHECK-NOT: llvm.zext {{.*}} : i8 to i16
    // CHECK-COUNT-16: llvm.bitcast {{.*}} : vector<4xi8> to vector<2xi16>
    // CHECK-NOT: llvm.zext {{.*}} : i8 to i16
    %k64_f16 = tt.fp_to_fp %k64 : tensor<64x16xf8E4M3FN, #dot1> -> tensor<64x16xf16, #dot1>
    %k32_f16 = tt.fp_to_fp %k32 : tensor<32x16xf8E4M3FN, #dot1> -> tensor<32x16xf16, #dot1>
    %row_f16 = tt.fp_to_fp %row : tensor<32x16xf8E4M3FN, #dot1> -> tensor<32x16xf16, #dot1>
    tt.return %k64_f16, %k32_f16, %row_f16 : tensor<64x16xf16, #dot1>, tensor<32x16xf16, #dot1>, tensor<32x16xf16, #dot1>
  }
}
