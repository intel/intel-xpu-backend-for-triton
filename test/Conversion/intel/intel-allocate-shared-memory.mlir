// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [0, 1]}>

// Check no scratch memory is allocated for sub-group shuffle-like layout conversions.

// CHECK-LABEL: module attributes
// CHECK-SAME: ttg.shared = 0 : i32
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK: tt.func @test_sub_group_shuffle
  // CHECK-NOT: llvm.ptr<3>
  tt.func @test_sub_group_shuffle(%arg0: tensor<16xf16, #ttg.slice<{dim = 1, parent = #blocked}>>) -> tensor<16xf16, #ttg.slice<{dim = 1, parent = #blocked1}>> {
    %0 = ttg.convert_layout %arg0 : tensor<16xf16, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16xf16, #ttg.slice<{dim = 1, parent = #blocked1}>>
    tt.return %0 : tensor<16xf16, #ttg.slice<{dim = 1, parent = #blocked1}>>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [16, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [32, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [0, 1]}>

// Check no scratch memory is allocated for sub-group shuffle-like layout conversions.

// CHECK-LABEL: module attributes
// CHECK-SAME: ttg.shared = 0 : i32
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK: tt.func @test_sub_group_shuffle
  // CHECK-NOT: llvm.ptr<3>
  tt.func @test_sub_group_shuffle(%arg0: tensor<32xf16, #ttg.slice<{dim = 1, parent = #blocked}>>) -> tensor<32xf16, #ttg.slice<{dim = 1, parent = #blocked1}>> {
    %0 = ttg.convert_layout %arg0 : tensor<32xf16, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32xf16, #ttg.slice<{dim = 1, parent = #blocked1}>>
    tt.return %0 : tensor<32xf16, #ttg.slice<{dim = 1, parent = #blocked1}>>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [1, 1], order = [0, 1]}>

// Check scracth memory configuration for different sub-group transpose-like layout conversions.

// CHECK-LABEL: module attributes
// CHECK-SAME: ttg.shared = 544 : i32
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @test_f16(%arg0: tensor<16x16xf16, #blocked>) -> tensor<16x16xf16, #blocked1> {
    %0 = ttg.convert_layout %arg0 : tensor<16x16xf16, #blocked> -> tensor<16x16xf16, #blocked1>
    tt.return %0 : tensor<16x16xf16, #blocked1>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [1, 1], order = [0, 1]}>

// Check scracth memory configuration for different sub-group transpose-like layout conversions.

// CHECK-LABEL: module attributes
// CHECK-SAME: ttg.shared = 1088 : i32
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @test_f32(%arg0: tensor<16x16xf32, #blocked>) -> tensor<16x16xf32, #blocked1> {
    %0 = ttg.convert_layout %arg0 : tensor<16x16xf32, #blocked> -> tensor<16x16xf32, #blocked1>
    tt.return %0 : tensor<16x16xf32, #blocked1>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 2], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [4, 2], order = [0, 1]}>

// Check scracth memory configuration for different sub-group transpose-like layout conversions.

// CHECK-LABEL: module attributes
// CHECK-SAME: ttg.shared = 34816 : i32
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @test_f32(%arg0: tensor<128x64xf32, #blocked>) -> tensor<128x64xf32, #blocked1> {
    %0 = ttg.convert_layout %arg0 : tensor<128x64xf32, #blocked> -> tensor<128x64xf32, #blocked1>
    tt.return %0 : tensor<128x64xf32, #blocked1>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [16, 1], warpsPerCTA = [2, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [16, 2], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [0, 1]}>

// Check scracth memory configuration for different sub-group transpose-like layout conversions.

// CHECK-LABEL: module attributes
// CHECK-SAME: ttg.shared = 17408 : i32
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @test_contiguous(%arg0: tensor<32x128xf32, #blocked>) -> tensor<32x128xf32, #blocked1> {
    %0 = ttg.convert_layout %arg0 : tensor<32x128xf32, #blocked> -> tensor<32x128xf32, #blocked1>
    tt.return %0 : tensor<32x128xf32, #blocked1>
  }
}
