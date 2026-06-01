// RUN: env TRITON_INTEL_ENABLE_BLOCK_IO_ALL_LAYOUTS=0 triton-opt %s -split-input-file -tritonintelgpu-materialize-block-pointer | FileCheck %s --check-prefixes=CHECK,DISABLED
// RUN: env TRITON_INTEL_ENABLE_BLOCK_IO_ALL_LAYOUTS=1 triton-opt %s -split-input-file -tritonintelgpu-materialize-block-pointer | FileCheck %s --check-prefixes=CHECK,ENABLED

// COM: Regression test for issue #6897: verifies that the env var guard
// COM: TRITON_INTEL_ENABLE_BLOCK_IO_ALL_LAYOUTS=0 disables 1D reshape
// COM: optimization for both strided stores and strided loads.

// COM: Test 1: 1D strided STORE with H==1 — the canonical single-row case.
// COM: When enabled, this pattern should be reshaped [32] -> [1, 32] with
// COM: ttig.block_io = "row_major" and ttig.block_io_stride = 96.
// COM: When disabled, the original tt.store must survive unchanged.

#blocked1d = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @test_1d_strided_store_single_row
  tt.func @test_1d_strided_store_single_row(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: tensor<32xf16, #blocked1d>) {
    %idx = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #blocked1d>
    %cst32 = arith.constant dense<32> : tensor<32xi32, #blocked1d>
    %cst96 = arith.constant dense<96> : tensor<32xi32, #blocked1d>
    %rem = arith.remui %idx, %cst32 : tensor<32xi32, #blocked1d>
    %div = arith.divui %idx, %cst32 : tensor<32xi32, #blocked1d>
    %mul = arith.muli %div, %cst96 : tensor<32xi32, #blocked1d>
    %off = arith.addi %rem, %mul : tensor<32xi32, #blocked1d>
    %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>, #blocked1d>
    %ptrs = tt.addptr %base, %off : tensor<32x!tt.ptr<f16>, #blocked1d>, tensor<32xi32, #blocked1d>
    // DISABLED-NOT: tt.reshape
    // DISABLED: tt.store %{{.*}}, %{{.*}} : tensor<32x!tt.ptr<f16>
    // DISABLED-NOT: ttig.block_io
    // ENABLED: tt.reshape
    // ENABLED: tt.store %{{.*}}, %{{.*}} {ttig.block_io = "row_major", ttig.block_io_stride = 96 : i64} : tensor<1x32x!tt.ptr<f16>
    tt.store %ptrs, %arg1 : tensor<32x!tt.ptr<f16>, #blocked1d>
    tt.return
  }
}

// -----

// COM: Test 2: 1D strided LOAD with H > 1 — the canonical multi-row case.
// COM: numElements=1024, W=32, S=96, fp16, numWarps=4, H = 1024/32 = 32.
// COM: When enabled, this pattern should be reshaped [1024] -> [32, 32] with
// COM: ttig.block_io = "row_major" and ttig.block_io_stride = 96, followed by
// COM: a convert_layout and a reshape back to [1024].
// COM: When disabled, the original tt.load must survive unchanged.

#blocked1d_large = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @test_1d_strided_load_multi_row
  tt.func @test_1d_strided_load_multi_row(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<1024xf16, #blocked1d_large> {
    %idx = tt.make_range {start = 0 : i32, end = 1024 : i32} : tensor<1024xi32, #blocked1d_large>
    %c32 = arith.constant dense<32> : tensor<1024xi32, #blocked1d_large>
    %c96 = arith.constant dense<96> : tensor<1024xi32, #blocked1d_large>
    %rem = arith.remui %idx, %c32 : tensor<1024xi32, #blocked1d_large>
    %div = arith.divui %idx, %c32 : tensor<1024xi32, #blocked1d_large>
    %mul = arith.muli %div, %c96 : tensor<1024xi32, #blocked1d_large>
    %off = arith.addi %rem, %mul : tensor<1024xi32, #blocked1d_large>
    %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked1d_large>
    %ptrs = tt.addptr %base, %off : tensor<1024x!tt.ptr<f16>, #blocked1d_large>, tensor<1024xi32, #blocked1d_large>
    %mask = arith.constant dense<true> : tensor<1024xi1, #blocked1d_large>
    // DISABLED-NOT: tt.reshape
    // DISABLED: tt.load %{{.*}} : tensor<1024x!tt.ptr<f16>
    // DISABLED-NOT: ttig.block_io
    // ENABLED: tt.reshape %{{.*}} allow_reorder efficient_layout
    // ENABLED: tt.load %{{.*}} {ttig.block_io = "row_major", ttig.block_io_stride = 96 : i64}
    // ENABLED: ttg.convert_layout
    // ENABLED: tt.reshape %{{.*}} efficient_layout
    %result = tt.load %ptrs, %mask : tensor<1024x!tt.ptr<f16>, #blocked1d_large>
    tt.return %result : tensor<1024xf16, #blocked1d_large>
  }
}
