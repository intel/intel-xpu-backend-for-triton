// RUN: env TRITON_INTEL_ENABLE_BLOCK_IO_ALL_LAYOUTS=1 triton-opt %s -split-input-file -tritonintelgpu-materialize-block-pointer --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s

// COM: Test that the 1D strided store pattern, after MaterializeBlockPointer
// COM: reshapes it to a 2D store with ttig.block_io and ttig.block_io_stride,
// COM: is lowered all the way to triton_gen.2Dblockstore by the LLVM conversion.

// COM: Test 1: Canonical strided store — W=32, S=96, fp16, 1024 elements, H=32.
// COM: H > 1 so MaterializeBlockPointer does NOT reshape the store (PR #6682).
// COM: No triton_gen.2Dblockstore should be emitted.

#blocked1d = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: llvm.func spir_kernelcc @test_1d_reshape_to_2d_blockstore
  // CHECK-NOT: triton_gen.2Dblockstore
  tt.func @test_1d_reshape_to_2d_blockstore(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: tensor<1024xf16, #blocked1d>) {
    %idx = tt.make_range {start = 0 : i32, end = 1024 : i32} : tensor<1024xi32, #blocked1d>
    %cst32 = arith.constant dense<32> : tensor<1024xi32, #blocked1d>
    %cst96 = arith.constant dense<96> : tensor<1024xi32, #blocked1d>
    %rem = arith.remui %idx, %cst32 : tensor<1024xi32, #blocked1d>
    %div = arith.divui %idx, %cst32 : tensor<1024xi32, #blocked1d>
    %mul = arith.muli %div, %cst96 : tensor<1024xi32, #blocked1d>
    %off = arith.addi %rem, %mul : tensor<1024xi32, #blocked1d>
    %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked1d>
    %ptrs = tt.addptr %base, %off : tensor<1024x!tt.ptr<f16>, #blocked1d>, tensor<1024xi32, #blocked1d>
    tt.store %ptrs, %arg1 : tensor<1024x!tt.ptr<f16>, #blocked1d>
    tt.return
  }
}

// -----

// COM: Test 2: Masked store with dense<true> constant — H=32 > 1, so the store
// COM: is NOT reshaped (PR #6682). No triton_gen.2Dblockstore should be emitted.

#blocked1d = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: llvm.func spir_kernelcc @test_1d_reshape_dense_true_mask
  // CHECK-NOT: triton_gen.2Dblockstore
  tt.func @test_1d_reshape_dense_true_mask(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: tensor<1024xf16, #blocked1d>) {
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
    tt.store %ptrs, %arg1, %mask : tensor<1024x!tt.ptr<f16>, #blocked1d>
    tt.return
  }
}

// -----

// COM: Test 3: Non-canonical index — should NOT produce 2D block store.
// COM: The index is 2 * tt.make_range, so MaterializeBlockPointer rejects it,
// COM: and the store falls through to scalar scatter.

#blocked1d = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: llvm.func spir_kernelcc @test_1d_reshape_rejected_non_canonical
  // CHECK-NOT: triton_gen.2Dblockstore
  tt.func @test_1d_reshape_rejected_non_canonical(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: tensor<1024xf16, #blocked1d>) {
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
    tt.store %ptrs, %arg1 : tensor<1024x!tt.ptr<f16>, #blocked1d>
    tt.return
  }
}

// -----

// COM: Test 4: 1D strided load — W=32, S=96, fp16, 1024 elements, H=32.
// COM: MaterializeBlockPointer detects the offset pattern and reshapes the 1D
// COM: load to a 2D block load. The LLVM conversion should emit a
// COM: triton_gen.2Dblockload.

#blocked1d = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: llvm.func spir_kernelcc @test_1d_strided_load_to_2d_blockload
  // CHECK: triton_gen.2Dblockload
  tt.func @test_1d_strided_load_to_2d_blockload(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<1024xf16, #blocked1d> {
    %idx = tt.make_range {start = 0 : i32, end = 1024 : i32} : tensor<1024xi32, #blocked1d>
    %c32 = arith.constant dense<32> : tensor<1024xi32, #blocked1d>
    %c96 = arith.constant dense<96> : tensor<1024xi32, #blocked1d>
    %rem = arith.remui %idx, %c32 : tensor<1024xi32, #blocked1d>
    %div = arith.divui %idx, %c32 : tensor<1024xi32, #blocked1d>
    %mul = arith.muli %div, %c96 : tensor<1024xi32, #blocked1d>
    %off = arith.addi %rem, %mul : tensor<1024xi32, #blocked1d>
    %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked1d>
    %ptrs = tt.addptr %base, %off : tensor<1024x!tt.ptr<f16>, #blocked1d>, tensor<1024xi32, #blocked1d>
    %mask = arith.constant dense<true> : tensor<1024xi1, #blocked1d>
    %result = tt.load %ptrs, %mask : tensor<1024x!tt.ptr<f16>, #blocked1d>
    tt.return %result : tensor<1024xf16, #blocked1d>
  }
}
