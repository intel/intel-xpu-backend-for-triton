// RUN: triton-opt %s -split-input-file -tritonintelgpu-pipeline="num-stages=3" | FileCheck %s

// COM: Test that the pipeline pass creates ttig.prefetch ops for elementwise
// COM: loads (BlockedEncodingAttr, NOT DotOperandEncodingAttr) inside scf.for
// COM: loops when the loads have the "ttig.block_io" attribute and rank >= 2.

#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io", "ttig.support_prefetch_256b"} {
  // CHECK-LABEL: tt.func public @elementwise_prefetch
  tt.func public @elementwise_prefetch(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<32> : tensor<32x32xi32, #blocked>

    // COM: Build pointer tensors for two elementwise loads.
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %2 = arith.constant dense<32> : tensor<32x1xi32, #blocked>
    %3 = arith.muli %1, %2 : tensor<32x1xi32, #blocked>
    %4 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %6 = tt.broadcast %3 : tensor<32x1xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %7 = tt.broadcast %5 : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %8 = arith.addi %6, %7 : tensor<32x32xi32, #blocked>

    %splat_a = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    %ptr_a = tt.addptr %splat_a, %8 : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>

    %splat_b = tt.splat %arg1 : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    %ptr_b = tt.addptr %splat_b, %8 : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>

    %splat_c = tt.splat %arg2 : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    %ptr_c = tt.addptr %splat_c, %8 : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>

    %n_steps = arith.divsi %arg3, %c32_i32 : i32

    // COM: Verify prefetch ops are created before and inside the loop.
    // CHECK: ttig.prefetch {{.*}} : tensor<32x32x!tt.ptr<f16>, #{{.*}}>
    // CHECK: ttig.prefetch {{.*}} : tensor<32x32x!tt.ptr<f16>, #{{.*}}>
    // CHECK: ttig.prefetch {{.*}} : tensor<32x32x!tt.ptr<f16>, #{{.*}}>
    // CHECK: ttig.prefetch {{.*}} : tensor<32x32x!tt.ptr<f16>, #{{.*}}>
    // CHECK: scf.for
    // CHECK:   ttig.prefetch {{.*}} : tensor<32x32x!tt.ptr<f16>, #{{.*}}>
    // CHECK:   ttig.prefetch {{.*}} : tensor<32x32x!tt.ptr<f16>, #{{.*}}>
    // CHECK:   tt.load {{.*}} : tensor<32x32x!tt.ptr<f16>, #{{.*}}>
    // CHECK:   tt.load {{.*}} : tensor<32x32x!tt.ptr<f16>, #{{.*}}>
    // CHECK:   arith.addf
    // CHECK:   tt.store
    // CHECK:   scf.yield
    %9:2 = scf.for %iv = %c0_i32 to %n_steps step %c1_i32 iter_args(%arg_a = %ptr_a, %arg_b = %ptr_b) -> (tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32x!tt.ptr<f16>, #blocked>) : i32 {
      %ld_a = tt.load %arg_a {ttig.block_io = "row_major"} : tensor<32x32x!tt.ptr<f16>, #blocked>
      %ld_b = tt.load %arg_b {ttig.block_io = "row_major"} : tensor<32x32x!tt.ptr<f16>, #blocked>
      %sum = arith.addf %ld_a, %ld_b : tensor<32x32xf16, #blocked>
      tt.store %ptr_c, %sum : tensor<32x32x!tt.ptr<f16>, #blocked>
      %next_a = tt.addptr %arg_a, %cst : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>
      %next_b = tt.addptr %arg_b, %cst : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>
      scf.yield %next_a, %next_b : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32x!tt.ptr<f16>, #blocked>
    }
    tt.return
  }
}

// -----

// COM: Test that loads WITHOUT the "ttig.block_io" attribute do NOT get
// COM: prefetch ops created by the pipeline pass.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: tt.func public @no_blockio_no_prefetch
  tt.func public @no_blockio_no_prefetch(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst = arith.constant dense<64> : tensor<64x64xi32, #blocked>

    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %2 = arith.constant dense<64> : tensor<64x1xi32, #blocked>
    %3 = arith.muli %1, %2 : tensor<64x1xi32, #blocked>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %6 = tt.broadcast %3 : tensor<64x1xi32, #blocked> -> tensor<64x64xi32, #blocked>
    %7 = tt.broadcast %5 : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked>
    %8 = arith.addi %6, %7 : tensor<64x64xi32, #blocked>

    %splat_a = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>, #blocked>
    %ptr_a = tt.addptr %splat_a, %8 : tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64xi32, #blocked>

    %splat_b = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>, #blocked>
    %ptr_b = tt.addptr %splat_b, %8 : tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64xi32, #blocked>

    %n_steps = arith.divsi %arg2, %c64_i32 : i32

    // COM: Without block_io attribute, no prefetch should be inserted.
    // CHECK-NOT: ttig.prefetch
    // CHECK: scf.for
    // CHECK-NOT: ttig.prefetch
    // CHECK:   tt.load
    // CHECK:   tt.load
    // CHECK:   arith.addf
    // CHECK:   scf.yield
    %9:2 = scf.for %iv = %c0_i32 to %n_steps step %c1_i32 iter_args(%arg_a = %ptr_a, %arg_b = %ptr_b) -> (tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64x!tt.ptr<f16>, #blocked>) : i32 {
      %ld_a = tt.load %arg_a : tensor<64x64x!tt.ptr<f16>, #blocked>
      %ld_b = tt.load %arg_b : tensor<64x64x!tt.ptr<f16>, #blocked>
      %sum = arith.addf %ld_a, %ld_b : tensor<64x64xf16, #blocked>
      %next_a = tt.addptr %arg_a, %cst : tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64xi32, #blocked>
      %next_b = tt.addptr %arg_b, %cst : tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64xi32, #blocked>
      scf.yield %next_a, %next_b : tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64x!tt.ptr<f16>, #blocked>
    }
    tt.return
  }
}

// -----

// COM: Test that rank-1 loads with block_io do NOT get prefetch ops.
// COM: The pipeline pass requires rank >= 2.

#blocked1d = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [16], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io", "ttig.support_prefetch_256b"} {
  // CHECK-LABEL: tt.func public @rank1_no_prefetch
  tt.func public @rank1_no_prefetch(%arg0: !tt.ptr<f16>, %arg1: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c256_i32 = arith.constant 256 : i32
    %cst = arith.constant dense<256> : tensor<256xi32, #blocked1d>

    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked1d>
    %splat_a = tt.splat %arg0 : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>, #blocked1d>
    %ptr_a = tt.addptr %splat_a, %0 : tensor<256x!tt.ptr<f16>, #blocked1d>, tensor<256xi32, #blocked1d>

    %n_steps = arith.divsi %arg1, %c256_i32 : i32

    // COM: Rank-1 load should not be prefetched even with block_io.
    // CHECK-NOT: ttig.prefetch
    // CHECK: scf.for
    // CHECK-NOT: ttig.prefetch
    // CHECK:   tt.load
    // CHECK:   scf.yield
    %9 = scf.for %iv = %c0_i32 to %n_steps step %c1_i32 iter_args(%arg_a = %ptr_a) -> (tensor<256x!tt.ptr<f16>, #blocked1d>) : i32 {
      %ld_a = tt.load %arg_a {ttig.block_io = "row_major"} : tensor<256x!tt.ptr<f16>, #blocked1d>
      %next_a = tt.addptr %arg_a, %cst : tensor<256x!tt.ptr<f16>, #blocked1d>, tensor<256xi32, #blocked1d>
      scf.yield %next_a : tensor<256x!tt.ptr<f16>, #blocked1d>
    }
    tt.return
  }
}
