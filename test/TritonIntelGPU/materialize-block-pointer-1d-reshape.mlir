// RUN: triton-opt %s -split-input-file -tritonintelgpu-materialize-block-pointer | FileCheck %s

// COM: Test 1: Canonical strided pattern — 1D store with W=32, S=96, fp16, 1024 elements.
// COM: The pass should detect the offset pattern:
// COM:   arith.addi(arith.remui(idx, 32), arith.muli(arith.divui(idx, 32), 96))
// COM: where idx = tt.make_range(0, 1024), and reshape the 1D store [1024] -> [32, 32]
// COM: with ttig.block_io = "row_major" and ttig.block_io_stride = 96.

#blocked1d = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @test_canonical_pattern
  tt.func @test_canonical_pattern(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: tensor<1024xf16, #blocked1d>) {
    %idx = tt.make_range {start = 0 : i32, end = 1024 : i32} : tensor<1024xi32, #blocked1d>
    %cst32 = arith.constant dense<32> : tensor<1024xi32, #blocked1d>
    %cst96 = arith.constant dense<96> : tensor<1024xi32, #blocked1d>
    %rem = arith.remui %idx, %cst32 : tensor<1024xi32, #blocked1d>
    %div = arith.divui %idx, %cst32 : tensor<1024xi32, #blocked1d>
    %mul = arith.muli %div, %cst96 : tensor<1024xi32, #blocked1d>
    %off = arith.addi %rem, %mul : tensor<1024xi32, #blocked1d>
    %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked1d>
    %ptrs = tt.addptr %base, %off : tensor<1024x!tt.ptr<f16>, #blocked1d>, tensor<1024xi32, #blocked1d>
    // CHECK: tt.reshape
    // CHECK: tt.reshape
    // CHECK: tt.store %{{.*}}, %{{.*}} {ttig.block_io = "row_major", ttig.block_io_stride = 96 : i64} : tensor<32x32x!tt.ptr<f16>
    tt.store %ptrs, %arg1 : tensor<1024x!tt.ptr<f16>, #blocked1d>
    tt.return
  }
}

// -----

// COM: Test 2: Masked store with dense<true> constant — should still be reshaped.
// COM: The mask is a direct dense<true> tensor constant (not splat(true)).
// COM: matchPattern/m_One recognizes this as provably all-true.

#blocked1d = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @test_dense_true_mask
  tt.func @test_dense_true_mask(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: tensor<1024xf16, #blocked1d>) {
    %idx = tt.make_range {start = 0 : i32, end = 1024 : i32} : tensor<1024xi32, #blocked1d>
    %cst32 = arith.constant dense<32> : tensor<1024xi32, #blocked1d>
    %cst96 = arith.constant dense<96> : tensor<1024xi32, #blocked1d>
    %rem = arith.remui %idx, %cst32 : tensor<1024xi32, #blocked1d>
    %div = arith.divui %idx, %cst32 : tensor<1024xi32, #blocked1d>
    %mul = arith.muli %div, %cst96 : tensor<1024xi32, #blocked1d>
    %off = arith.addi %rem, %mul : tensor<1024xi32, #blocked1d>
    %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked1d>
    %ptrs = tt.addptr %base, %off : tensor<1024x!tt.ptr<f16>, #blocked1d>, tensor<1024xi32, #blocked1d>
    %mask = arith.constant dense<true> : tensor<1024xi1, #blocked1d>
    // CHECK: tt.reshape
    // CHECK: tt.store %{{.*}}, %{{.*}} {ttig.block_io = "row_major", ttig.block_io_stride = 96 : i64} : tensor<32x32x!tt.ptr<f16>
    tt.store %ptrs, %arg1, %mask : tensor<1024x!tt.ptr<f16>, #blocked1d>
    tt.return
  }
}

// -----

// COM: Test 3: Non-canonical index rejection — scaled index.
// COM: The index is 2 * tt.make_range instead of plain tt.make_range.
// COM: The isCanonicalLinearIndex check should reject this because remui's
// COM: LHS is arith.muli, not tt.make_range(0, N).

#blocked1d = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @test_non_canonical_index
  tt.func @test_non_canonical_index(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: tensor<1024xf16, #blocked1d>) {
    %idx = tt.make_range {start = 0 : i32, end = 1024 : i32} : tensor<1024xi32, #blocked1d>
    %cst2 = arith.constant dense<2> : tensor<1024xi32, #blocked1d>
    %scaled_idx = arith.muli %idx, %cst2 : tensor<1024xi32, #blocked1d>
    %cst32 = arith.constant dense<32> : tensor<1024xi32, #blocked1d>
    %cst96 = arith.constant dense<96> : tensor<1024xi32, #blocked1d>
    %rem = arith.remui %scaled_idx, %cst32 : tensor<1024xi32, #blocked1d>
    %div = arith.divui %scaled_idx, %cst32 : tensor<1024xi32, #blocked1d>
    %mul = arith.muli %div, %cst96 : tensor<1024xi32, #blocked1d>
    %off = arith.addi %rem, %mul : tensor<1024xi32, #blocked1d>
    %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked1d>
    %ptrs = tt.addptr %base, %off : tensor<1024x!tt.ptr<f16>, #blocked1d>, tensor<1024xi32, #blocked1d>
    // CHECK-NOT: tt.reshape
    // CHECK: tt.store
    // CHECK-NOT: ttig.block_io
    tt.store %ptrs, %arg1 : tensor<1024x!tt.ptr<f16>, #blocked1d>
    tt.return
  }
}

// -----

// COM: Test 4: Non-trivial mask rejection.
// COM: Same strided offset pattern as Test 1 but the store has a real mask
// COM: argument (not splat(true)). The pass should reject this and leave the
// COM: 1D store unchanged.

#blocked1d = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @test_non_trivial_mask
  tt.func @test_non_trivial_mask(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: tensor<1024xf16, #blocked1d>, %mask: tensor<1024xi1, #blocked1d>) {
    %idx = tt.make_range {start = 0 : i32, end = 1024 : i32} : tensor<1024xi32, #blocked1d>
    %cst32 = arith.constant dense<32> : tensor<1024xi32, #blocked1d>
    %cst96 = arith.constant dense<96> : tensor<1024xi32, #blocked1d>
    %rem = arith.remui %idx, %cst32 : tensor<1024xi32, #blocked1d>
    %div = arith.divui %idx, %cst32 : tensor<1024xi32, #blocked1d>
    %mul = arith.muli %div, %cst96 : tensor<1024xi32, #blocked1d>
    %off = arith.addi %rem, %mul : tensor<1024xi32, #blocked1d>
    %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked1d>
    %ptrs = tt.addptr %base, %off : tensor<1024x!tt.ptr<f16>, #blocked1d>, tensor<1024xi32, #blocked1d>
    // CHECK-NOT: tt.reshape
    // CHECK: tt.store
    // CHECK-NOT: ttig.block_io
    tt.store %ptrs, %arg1, %mask : tensor<1024x!tt.ptr<f16>, #blocked1d>
    tt.return
  }
}
