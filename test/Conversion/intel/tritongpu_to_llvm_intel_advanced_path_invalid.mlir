// RUN: env TRITON_INTEL_ADVANCED_PATH=1 triton-opt %s --convert-triton-intel-gpu-to-llvm --verify-diagnostics --split-input-file

module attributes {"ttig.support_sg_2d_block", "ttig.support_dpas", "ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func public @matmul_kernel_with_block_pointers(%arg0: !tt.ptr<f32>, %arg1: i64, %arg2: i32) {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %22 = tt.make_tensor_ptr %arg0, [%arg1, %arg1], [%arg1, %c1_i64], [%arg2, %c0_i32] {order = array<i32: 1, 0>} : <tensor<2x32xf32>>
    // expected-error @+2 {{expecting elem_size_in_bits * tile_width * v_blocks <= 512}}
    // expected-error @+1 {{failed to legalize operation 'ttig.prefetch'}}
    ttig.prefetch %22 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<2x32xf32>>
    tt.return
  }
}

// -----

module attributes {"ttig.support_sg_2d_block", "ttig.support_dpas", "ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func public @matmul_kernel_with_block_pointers(%arg0: !tt.ptr<f32>, %arg1: i64, %arg2: i32) {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %22 = tt.make_tensor_ptr %arg0, [%arg1, %arg1], [%arg1, %c1_i64], [%arg2, %c0_i32] {order = array<i32: 1, 0>} : <tensor<2x32xf32>>
    // expected-error @+2 {{expecting elem_size_in_bits * tile_width * v_blocks <= 512}}
    // expected-error @+1 {{failed to legalize operation 'tt.load'}}
    %res = tt.load %22 {DotIdx = 0 : i32, boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<2x32xf32>>
    tt.return
  }
}

// -----

module attributes {"ttig.support_sg_2d_block", "ttig.support_dpas", "ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func public @matmul_kernel_with_block_pointers(%arg0: !tt.ptr<f32>, %arg1: i64, %arg2: i32) {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<2x32xf32>
    %22 = tt.make_tensor_ptr %arg0, [%arg1, %arg1], [%arg1, %c1_i64], [%arg2, %c0_i32] {order = array<i32: 1, 0>} : <tensor<2x32xf32>>
    // expected-error @+2 {{expecting tile_width to be between 1 and 16}}
    // expected-error @+1 {{failed to legalize operation 'tt.store'}}
    tt.store %22, %cst {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<2x32xf32>>
    tt.return
  }
}
