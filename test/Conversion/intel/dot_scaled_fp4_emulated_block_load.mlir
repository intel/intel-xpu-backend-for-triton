// RUN: triton-opt %s --tritonintelgpu-lower-to-2d-block-load --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s
// RUN: triton-opt %s --tritonintelgpu-lower-to-2d-block-load --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s --check-prefix=COUNT

// COM: Packed mxfp4 operands from 2D block loads (A as 16-bit words, B as VNNI dwords): the nibbles are shifted out
// COM: of the loaded words, without extracting and extending each byte.
#linear = #ttg.linear<{register = [[1, 0], [2, 0], [4, 0]], lane = [[0, 0], [0, 0], [0, 0], [0, 1]], warp = [], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1]], lane = [[1, 0], [2, 0], [4, 0], [8, 0]], warp = [], block = []}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 4, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1], A = [8, 32], B = [32, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_2d_block_io} {
  // CHECK-LABEL: llvm.func spir_kernelcc @emulated_fp4_dot_scaled_block_loads(
  // CHECK: [[AV:%.*]] = triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16,
  // CHECK: [[BV:%.*]] = triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 8, {{.*}} vnni_transform = true
  // COM: A of bf16 DPAS 0, row 0: nibble 0 of byte 0 of the first word.
  // CHECK: [[AW:%.*]] = llvm.extractelement [[AV]][{{%.*}} : i32] : vector<8xi16>
  // CHECK-NEXT: [[AW32:%.*]] = llvm.zext [[AW]] : i16 to i32
  // CHECK-NEXT: [[C28:%.*]] = llvm.mlir.constant(28 : i32) : i32
  // CHECK-NEXT: llvm.shl [[AW32]], [[C28]] : i32
  // COM: B of bf16 DPAS 0, dword 0: nibble 0 of byte 0 of the first dword.
  // CHECK: [[BW:%.*]] = llvm.extractelement [[BV]][{{%.*}} : i32] : vector<8xi32>
  // CHECK-NEXT: [[C28B:%.*]] = llvm.mlir.constant(28 : i32) : i32
  // CHECK-NEXT: llvm.shl [[BW]], [[C28B]] : i32
  // COM: A of bf16 DPAS 1, row 0: nibble 1 of byte 0 of the same word.
  // CHECK: [[C24:%.*]] = llvm.mlir.constant(24 : i32) : i32
  // CHECK-NEXT: llvm.shl [[AW32]], [[C24]] : i32
  // COM: A of bf16 DPAS 2, row 0: nibble 0 of byte 1 of the same word.
  // CHECK: [[AWB:%.*]] = llvm.extractelement [[AV]][{{%.*}} : i32] : vector<8xi16>
  // CHECK-NEXT: [[AWB32:%.*]] = llvm.zext [[AWB]] : i16 to i32
  // CHECK-NEXT: [[C20:%.*]] = llvm.mlir.constant(20 : i32) : i32
  // CHECK-NEXT: llvm.shl [[AWB32]], [[C20]] : i32
  // COUNT-LABEL: llvm.func spir_kernelcc @emulated_fp4_dot_scaled_block_loads(
  // COUNT-COUNT-4: triton_gen.dpas {{.*}} {pa = bf16, pb = bf16, rc = 8}
  // COUNT-NOT: triton_gen.dpas
  tt.func public @emulated_fp4_dot_scaled_block_loads(%pa: !tt.ptr<i8>, %pb: !tt.ptr<i8>, %sa: tensor<8x2xi8, #linear>, %sb: tensor<16x2xi8, #linear1>, %c: tensor<8x16xf32, #mma>) -> tensor<8x16xf32, #mma> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c32_i64 = arith.constant 32 : i64
    %c16_i64 = arith.constant 16 : i64
    %c8_i32 = arith.constant 8 : i32
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %da = tt.make_tensor_descriptor %pa, [%c8_i32, %c32_i32], [%c32_i64, %c1_i64] : <i8>, <8x32xi8>
    %db = tt.make_tensor_descriptor %pb, [%c32_i32, %c16_i32], [%c16_i64, %c1_i64] : <i8>, <32x16xi8>
    %a = tt.descriptor_load %da[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<8x32xi8> -> tensor<8x32xi8, #dot0>
    %b = tt.descriptor_load %db[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<32x16xi8> -> tensor<32x16xi8, #dot1>
    %d = tt.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<8x32xi8, #dot0>, tensor<8x2xi8, #linear> * tensor<32x16xi8, #dot1>, tensor<16x2xi8, #linear1> -> tensor<8x16xf32, #mma>
    tt.return %d : tensor<8x16xf32, #mma>
  }
}
