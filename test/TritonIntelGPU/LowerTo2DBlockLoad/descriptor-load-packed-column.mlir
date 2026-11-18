// RUN: triton-opt %s -split-input-file --tritonintelgpu-lower-to-2d-block-load | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.2d_block_io_base_alignment = 64 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @i8_d64_column_4
  // CHECK-NOT: ttig.2d_block_load
  // CHECK: tt.descriptor_load
  tt.func @i8_d64_column_4(%src: !tt.ptr<i8> {tt.divisibility = 16 : i32}) -> tensor<32x32xi8, #blocked> {
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %d = tt.make_tensor_descriptor %src, [%c32_i32, %c64_i32], [%c64_i64, %c1_i64] : <i8>, <32x32xsi8>
    %0 = tt.descriptor_load %d[%c0_i32, %c4_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<32x32xsi8> -> tensor<32x32xi8, #blocked>
    tt.return %0 : tensor<32x32xi8, #blocked>
  }

  // CHECK-LABEL: tt.func @i8_d64_column_8
  // CHECK: ttig.2d_block_load
  // CHECK-NOT: tt.descriptor_load
  tt.func @i8_d64_column_8(%src: !tt.ptr<i8> {tt.divisibility = 16 : i32}) -> tensor<32x32xi8, #blocked> {
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %d = tt.make_tensor_descriptor %src, [%c32_i32, %c64_i32], [%c64_i64, %c1_i64] : <i8>, <32x32xsi8>
    %0 = tt.descriptor_load %d[%c0_i32, %c8_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<32x32xsi8> -> tensor<32x32xi8, #blocked>
    tt.return %0 : tensor<32x32xi8, #blocked>
  }

  // CHECK-LABEL: tt.func @i8_d64_runtime_column_div4
  // CHECK-NOT: ttig.2d_block_load
  // CHECK: tt.descriptor_load
  tt.func @i8_d64_runtime_column_div4(%src: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %col: i32 {tt.divisibility = 4 : i32}) -> tensor<32x32xi8, #blocked> {
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %d = tt.make_tensor_descriptor %src, [%c32_i32, %c64_i32], [%c64_i64, %c1_i64] : <i8>, <32x32xsi8>
    %0 = tt.descriptor_load %d[%c0_i32, %col] {ttig.block_io = "row_major"} : !tt.tensordesc<32x32xsi8> -> tensor<32x32xi8, #blocked>
    tt.return %0 : tensor<32x32xi8, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.2d_block_io_base_alignment = 64 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @i8_d32_column_4
  // CHECK: ttig.2d_block_load
  // CHECK-NOT: tt.descriptor_load
  tt.func @i8_d32_column_4(%src: !tt.ptr<i8> {tt.divisibility = 16 : i32}) -> tensor<32x16xi8, #blocked> {
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %d = tt.make_tensor_descriptor %src, [%c32_i32, %c64_i32], [%c64_i64, %c1_i64] : <i8>, <32x16xsi8>
    %0 = tt.descriptor_load %d[%c0_i32, %c4_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<32x16xsi8> -> tensor<32x16xi8, #blocked>
    tt.return %0 : tensor<32x16xi8, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.2d_block_io_base_alignment = 64 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @f32_d64_column_1
  // CHECK-NOT: ttig.2d_block_load
  // CHECK: tt.descriptor_load
  tt.func @f32_d64_column_1(%src: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> tensor<32x8xf32, #blocked> {
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %d = tt.make_tensor_descriptor %src, [%c32_i32, %c64_i32], [%c64_i64, %c1_i64] : <f32>, <32x8xf32>
    %0 = tt.descriptor_load %d[%c0_i32, %c1_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<32x8xf32> -> tensor<32x8xf32, #blocked>
    tt.return %0 : tensor<32x8xf32, #blocked>
  }
}
