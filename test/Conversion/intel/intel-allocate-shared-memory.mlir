// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory | FileCheck %s

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [0, 1]}>

// Check no scratch memory is allocated for sub-group shuffle-like layout conversions.

// CHECK-LABEL: module attributes
// CHECK-SAME: triton_gpu.shared = 0 : i32
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
  // CHECK: tt.func @test_sub_group_shuffle
  // CHECK-NOT: llvm.ptr<3>
  tt.func @test_sub_group_shuffle(%arg0: tensor<16xf16, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<16xf16, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> {
    %0 = triton_gpu.convert_layout %arg0 : tensor<16xf16, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<16xf16, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    tt.return %0 : tensor<16xf16, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [1, 1], order = [0, 1]}>

// Check scracth memory configuration for different sub-group transpose-like layout conversions.

// CHECK-LABEL: module attributes
// CHECK-SAME: triton_gpu.shared = 512 : i32
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
  tt.func @test_f16(%arg0: tensor<16x16xf16, #blocked>) -> tensor<16x16xf16, #blocked1> {
    %0 = triton_gpu.convert_layout %arg0 : tensor<16x16xf16, #blocked> -> tensor<16x16xf16, #blocked1>
    tt.return %0 : tensor<16x16xf16, #blocked1>
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [1, 1], order = [0, 1]}>

// Check scracth memory configuration for different sub-group transpose-like layout conversions.

// CHECK-LABEL: module attributes
// CHECK-SAME: triton_gpu.shared = 1024 : i32
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
  tt.func @test_f32(%arg0: tensor<16x16xf32, #blocked>) -> tensor<16x16xf32, #blocked1> {
    %0 = triton_gpu.convert_layout %arg0 : tensor<16x16xf32, #blocked> -> tensor<16x16xf32, #blocked1>
    tt.return %0 : tensor<16x16xf32, #blocked1>
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 2], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [4, 2], order = [0, 1]}>

// Check scracth memory configuration for different sub-group transpose-like layout conversions.

// CHECK-LABEL: module attributes
// CHECK-SAME: triton_gpu.shared = 32768 : i32
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
  tt.func @test_f32(%arg0: tensor<128x64xf32, #blocked>) -> tensor<128x64xf32, #blocked1> {
    %0 = triton_gpu.convert_layout %arg0 : tensor<128x64xf32, #blocked> -> tensor<128x64xf32, #blocked1>
    tt.return %0 : tensor<128x64xf32, #blocked1>
  }
}
