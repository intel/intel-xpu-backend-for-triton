// RUN: TRITON_INTEL_ENABLE_BLOCK_PTR=1 triton-opt %s --convert-triton-intel-gpu-to-llvm --verify-diagnostics

module attributes {"triton_gpu.num-warps" = 32 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
  tt.func public @matmul_kernel_with_block_pointers(%arg0: !tt.ptr<f32>, %arg1: i64, %arg2: i32) {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %22 = tt.make_tensor_ptr %arg0, [%arg1, %arg1], [%arg1, %c1_i64], [%arg2, %c0_i32] {order = array<i32: 1, 0>} : <tensor<2x32xf32>>
    // expected-error @+2 {{tile_width for 32 bit elements should be equal to 8 or 16}}
    // expected-error @+1 {{failed to legalize operation 'triton_intel_gpu.prefetch'}}
    triton_intel_gpu.prefetch %22 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<2x32xf32>>
    tt.return
  }
}
