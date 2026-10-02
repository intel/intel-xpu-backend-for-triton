// RUN: triton-opt %s --tritonintelgpu-lower-to-2d-block-load --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm --canonicalize | FileCheck %s

// COM: fp8e4m3 -> fp16 of block loaded dot operands: B bytes are read from its 32-bit load's dwords.
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.target_arch = "spir64"} {
  // CHECK-LABEL: @fp8_upcast_from_block_load
  tt.func public @fp8_upcast_from_block_load(%arg0: !tt.ptr<f8E4M3FN>, %arg1: !tt.ptr<f8E4M3FN>, %arg2: i32, %arg3: i32) -> (tensor<8x32xf16, #dot0>, tensor<32x16xf16, #dot1>) {
    %c1_i64 = arith.constant 1 : i64
    %c32_i64 = arith.constant 32 : i64
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %a_desc = tt.make_tensor_descriptor %arg0, [%c64_i32, %c32_i32], [%c32_i64, %c1_i64] : <f8E4M3FN>, <8x32xf8E4M3FN>
    %b_desc = tt.make_tensor_descriptor %arg1, [%c64_i32, %c32_i32], [%c32_i64, %c1_i64] : <f8E4M3FN>, <16x32xf8E4M3FN>
    // CHECK-DAG: llvm.mlir.constant(-16385 : i16) : i16
    // CHECK-DAG: llvm.mlir.constant(-16512 : i16) : i16
    // CHECK: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 8, {{.*}} transpose = false, {{.*}} -> vector<16xi8>
    // CHECK: %[[B:.*]] = triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 32, {{.*}} transpose = true, {{.*}} -> vector<8xi32>
    %a = tt.descriptor_load %a_desc[%arg2, %arg3] {ttig.block_io = "row_major"} : !tt.tensordesc<8x32xf8E4M3FN> -> tensor<8x32xf8E4M3FN, #dot0>
    %b = tt.descriptor_load %b_desc[%arg2, %arg3] {ttig.block_io = "column_major"} : !tt.tensordesc<16x32xf8E4M3FN> -> tensor<32x16xf8E4M3FN, #dot1>
    // CHECK-COUNT-16: llvm.zext {{.*}} : i8 to i16
    %a16 = tt.fp_to_fp %a : tensor<8x32xf8E4M3FN, #dot0> -> tensor<8x32xf16, #dot0>
    // CHECK: llvm.extractelement %[[B]][{{.*}}] : vector<8xi32>
    // CHECK-COUNT-32: llvm.ashr {{.*}} : i32
    // CHECK-NOT: llvm.zext {{.*}} : i8 to i16
    %b16 = tt.fp_to_fp %b : tensor<32x16xf8E4M3FN, #dot1> -> tensor<32x16xf16, #dot1>
    tt.return %a16, %b16 : tensor<8x32xf16, #dot0>, tensor<32x16xf16, #dot1>
  }
}
