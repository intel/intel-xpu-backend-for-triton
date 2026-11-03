// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.2d_block_io_base_alignment = 64 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: llvm.func spir_kernelcc @i8_d64_store_column_4
  // CHECK-NOT: triton_gen.2Dblockstore
  // CHECK: llvm.return
  tt.func public @i8_d64_store_column_4(%dst: !tt.ptr<i8> {tt.divisibility = 16 : i32}) {
    %val = arith.constant dense<1> : tensor<32x32xi8, #blocked>
    %c1_i64 = arith.constant 1 : i64
    %c96_i64 = arith.constant 96 : i64
    %c96_i32 = arith.constant 96 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %o = tt.make_tensor_descriptor %dst, [%c32_i32, %c96_i32], [%c96_i64, %c1_i64] : <i8>, <32x32xsi8>
    tt.descriptor_store %o[%c0_i32, %c4_i32], %val {ttig.block_io = "row_major"} : !tt.tensordesc<32x32xsi8>, tensor<32x32xi8, #blocked>
    tt.return
  }

  // CHECK-LABEL: llvm.func spir_kernelcc @i8_d64_store_column_8
  // CHECK: triton_gen.2Dblockstore {{.*}} {elem_size_in_bits = 64
  tt.func public @i8_d64_store_column_8(%dst: !tt.ptr<i8> {tt.divisibility = 16 : i32}) {
    %val = arith.constant dense<1> : tensor<32x32xi8, #blocked>
    %c1_i64 = arith.constant 1 : i64
    %c96_i64 = arith.constant 96 : i64
    %c96_i32 = arith.constant 96 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %o = tt.make_tensor_descriptor %dst, [%c32_i32, %c96_i32], [%c96_i64, %c1_i64] : <i8>, <32x32xsi8>
    tt.descriptor_store %o[%c0_i32, %c8_i32], %val {ttig.block_io = "row_major"} : !tt.tensordesc<32x32xsi8>, tensor<32x32xi8, #blocked>
    tt.return
  }

  // CHECK-LABEL: llvm.func spir_kernelcc @i8_d64_store_runtime_column_div8
  // CHECK: triton_gen.2Dblockstore {{.*}} {elem_size_in_bits = 64
  tt.func public @i8_d64_store_runtime_column_div8(%dst: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %col: i32 {tt.divisibility = 8 : i32}) {
    %val = arith.constant dense<1> : tensor<32x32xi8, #blocked>
    %c1_i64 = arith.constant 1 : i64
    %c96_i64 = arith.constant 96 : i64
    %c96_i32 = arith.constant 96 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %o = tt.make_tensor_descriptor %dst, [%c32_i32, %c96_i32], [%c96_i64, %c1_i64] : <i8>, <32x32xsi8>
    tt.descriptor_store %o[%c0_i32, %col], %val {ttig.block_io = "row_major"} : !tt.tensordesc<32x32xsi8>, tensor<32x32xi8, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.2d_block_io_base_alignment = 64 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: llvm.func spir_kernelcc @f32_d64_store_column_1
  // CHECK-NOT: triton_gen.2Dblockstore
  // CHECK: llvm.return
  tt.func public @f32_d64_store_column_1(%dst: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %val = arith.constant dense<1.000000e+00> : tensor<32x8xf32, #blocked>
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %o = tt.make_tensor_descriptor %dst, [%c32_i32, %c64_i32], [%c64_i64, %c1_i64] : <f32>, <32x8xf32>
    tt.descriptor_store %o[%c0_i32, %c1_i32], %val {ttig.block_io = "row_major"} : !tt.tensordesc<32x8xf32>, tensor<32x8xf32, #blocked>
    tt.return
  }
}
