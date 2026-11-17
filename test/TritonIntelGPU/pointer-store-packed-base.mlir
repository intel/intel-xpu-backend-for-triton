// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.2d_block_io_base_alignment = 64 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: llvm.func spir_kernelcc @i8_d64_ptr_store_base_plus_4
  // CHECK-NOT: triton_gen.2Dblockstore
  // CHECK: llvm.return
  tt.func public @i8_d64_ptr_store_base_plus_4(%dst: !tt.ptr<i8> {tt.divisibility = 16 : i32}) {
    %val = arith.constant dense<1> : tensor<32x32xi8, #blocked>
    %cst = arith.constant dense<64> : tensor<32x1xi32, #blocked>
    %c4_i32 = arith.constant 4 : i32
    %r = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %r_1 = tt.expand_dims %r {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %c = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %c_2 = tt.expand_dims %c {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %4 = tt.broadcast %c_2 : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %6 = tt.addptr %dst, %c4_i32 : !tt.ptr<i8>, i32
    %7 = arith.muli %r_1, %cst : tensor<32x1xi32, #blocked>
    %8 = tt.splat %6 : !tt.ptr<i8> -> tensor<32x1x!tt.ptr<i8>, #blocked>
    %9 = tt.addptr %8, %7 : tensor<32x1x!tt.ptr<i8>, #blocked>, tensor<32x1xi32, #blocked>
    %10 = tt.broadcast %9 : tensor<32x1x!tt.ptr<i8>, #blocked> -> tensor<32x32x!tt.ptr<i8>, #blocked>
    %11 = tt.addptr %10, %4 : tensor<32x32x!tt.ptr<i8>, #blocked>, tensor<32x32xi32, #blocked>
    tt.store %11, %val {ttig.block_io = "row_major"} : tensor<32x32x!tt.ptr<i8>, #blocked>
    tt.return
  }

  // CHECK-LABEL: llvm.func spir_kernelcc @i8_d64_ptr_store_base_plus_8
  // CHECK: triton_gen.2Dblockstore {{.*}} {elem_size_in_bits = 64
  tt.func public @i8_d64_ptr_store_base_plus_8(%dst: !tt.ptr<i8> {tt.divisibility = 16 : i32}) {
    %val = arith.constant dense<1> : tensor<32x32xi8, #blocked>
    %cst = arith.constant dense<64> : tensor<32x1xi32, #blocked>
    %c8_i32 = arith.constant 8 : i32
    %r = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %r_1 = tt.expand_dims %r {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %c = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %c_2 = tt.expand_dims %c {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %4 = tt.broadcast %c_2 : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %6 = tt.addptr %dst, %c8_i32 : !tt.ptr<i8>, i32
    %7 = arith.muli %r_1, %cst : tensor<32x1xi32, #blocked>
    %8 = tt.splat %6 : !tt.ptr<i8> -> tensor<32x1x!tt.ptr<i8>, #blocked>
    %9 = tt.addptr %8, %7 : tensor<32x1x!tt.ptr<i8>, #blocked>, tensor<32x1xi32, #blocked>
    %10 = tt.broadcast %9 : tensor<32x1x!tt.ptr<i8>, #blocked> -> tensor<32x32x!tt.ptr<i8>, #blocked>
    %11 = tt.addptr %10, %4 : tensor<32x32x!tt.ptr<i8>, #blocked>, tensor<32x32xi32, #blocked>
    tt.store %11, %val {ttig.block_io = "row_major"} : tensor<32x32x!tt.ptr<i8>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.2d_block_io_base_alignment = 64 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: llvm.func spir_kernelcc @i8_d32_ptr_store_base_plus_4
  // CHECK: triton_gen.2Dblockstore {{.*}} {elem_size_in_bits = 32
  tt.func public @i8_d32_ptr_store_base_plus_4(%dst: !tt.ptr<i8> {tt.divisibility = 16 : i32}) {
    %val = arith.constant dense<1> : tensor<32x16xi8, #blocked>
    %cst = arith.constant dense<64> : tensor<32x1xi32, #blocked>
    %c4_i32 = arith.constant 4 : i32
    %r = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %r_1 = tt.expand_dims %r {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %c = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %c_2 = tt.expand_dims %c {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x16xi32, #blocked>
    %4 = tt.broadcast %c_2 : tensor<1x16xi32, #blocked> -> tensor<32x16xi32, #blocked>
    %6 = tt.addptr %dst, %c4_i32 : !tt.ptr<i8>, i32
    %7 = arith.muli %r_1, %cst : tensor<32x1xi32, #blocked>
    %8 = tt.splat %6 : !tt.ptr<i8> -> tensor<32x1x!tt.ptr<i8>, #blocked>
    %9 = tt.addptr %8, %7 : tensor<32x1x!tt.ptr<i8>, #blocked>, tensor<32x1xi32, #blocked>
    %10 = tt.broadcast %9 : tensor<32x1x!tt.ptr<i8>, #blocked> -> tensor<32x16x!tt.ptr<i8>, #blocked>
    %11 = tt.addptr %10, %4 : tensor<32x16x!tt.ptr<i8>, #blocked>, tensor<32x16xi32, #blocked>
    tt.store %11, %val {ttig.block_io = "row_major"} : tensor<32x16x!tt.ptr<i8>, #blocked>
    tt.return
  }
}
