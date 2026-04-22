// RUN: triton-opt %s -split-input-file --tritonintelgpu-materialize-block-pointer | FileCheck %s

// COM: Ensure pointers with strides [0, 1]/[1, 0] are considered row/column major respectively.
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 4], warpsPerCTA = [32, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  tt.func public @tensor_of_ptr(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) {
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %2 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<1x32x!tt.ptr<bf16>, #blocked>
    %3 = tt.addptr %2, %1 : tensor<1x32x!tt.ptr<bf16>, #blocked>, tensor<1x32xi32, #blocked>
    %4 = tt.broadcast %3 : tensor<1x32x!tt.ptr<bf16>, #blocked> -> tensor<256x32x!tt.ptr<bf16>, #blocked>
    // CHECK: tt.load {{.*}} {ttig.block_io = "row_major"}
    tt.load %4 : tensor<256x32x!tt.ptr<bf16>, #blocked>

    %6 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %7 = tt.expand_dims %6 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xi32, #blocked>
    %8 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<256x1x!tt.ptr<bf16>, #blocked>
    %9 = tt.addptr %8, %7 : tensor<256x1x!tt.ptr<bf16>, #blocked>, tensor<256x1xi32, #blocked>
    %10 = tt.broadcast %9 : tensor<256x1x!tt.ptr<bf16>, #blocked> -> tensor<256x32x!tt.ptr<bf16>, #blocked>
    // CHECK: tt.load {{.*}} {ttig.block_io = "column_major"}
    tt.load %10 : tensor<256x32x!tt.ptr<bf16>, #blocked>
    tt.return
  }
}

// -----

// COM: Ensure tt.contiguity hint recovers stride after remsi, enabling block IO.
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 4], warpsPerCTA = [32, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @contiguity_hint_enables_block_io
  tt.func public @contiguity_hint_enables_block_io(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) {
    %range_row = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %c8 = arith.constant dense<8> : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %row_rem = arith.remsi %range_row, %c8 {tt.contiguity = dense<128> : tensor<1xi32>} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>

    %c32 = arith.constant dense<32> : tensor<128x1xi32, #blocked2>
    %row_exp = tt.expand_dims %row_rem {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi32, #blocked2>
    %row_off = arith.muli %row_exp, %c32 : tensor<128x1xi32, #blocked2>
    %row_off_bc = tt.broadcast %row_off : tensor<128x1xi32, #blocked2> -> tensor<128x32xi32, #blocked2>

    %range_col = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %col_exp = tt.expand_dims %range_col {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x32xi32, #blocked2>
    %col_off_bc = tt.broadcast %col_exp : tensor<1x32xi32, #blocked2> -> tensor<128x32xi32, #blocked2>

    %offsets = arith.addi %row_off_bc, %col_off_bc : tensor<128x32xi32, #blocked2>
    %ptr_splat = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<128x32x!tt.ptr<bf16>, #blocked2>
    %ptrs = tt.addptr %ptr_splat, %offsets : tensor<128x32x!tt.ptr<bf16>, #blocked2>, tensor<128x32xi32, #blocked2>

    // CHECK: tt.load {{.*}} {ttig.block_io = "row_major"}
    tt.load %ptrs : tensor<128x32x!tt.ptr<bf16>, #blocked2>
    tt.return
  }
}

// -----

// COM: Row-major masked tensor-of-pointer load should get block_io attribute.
#blocked3 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 4], warpsPerCTA = [32, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @masked_row_major_load
  tt.func public @masked_row_major_load(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) {
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x32xi32, #blocked3>
    %2 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<1x32x!tt.ptr<bf16>, #blocked3>
    %3 = tt.addptr %2, %1 : tensor<1x32x!tt.ptr<bf16>, #blocked3>, tensor<1x32xi32, #blocked3>
    %4 = tt.broadcast %3 : tensor<1x32x!tt.ptr<bf16>, #blocked3> -> tensor<256x32x!tt.ptr<bf16>, #blocked3>
    %mask = arith.constant dense<true> : tensor<256x32xi1, #blocked3>
    // CHECK: tt.load {{.*}} {ttig.block_io = "row_major"}
    tt.load %4, %mask : tensor<256x32x!tt.ptr<bf16>, #blocked3>
    tt.return
  }
}

// -----

// COM: Column-major masked tensor-of-pointer load should get block_io attribute.
#blocked4 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 4], warpsPerCTA = [32, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @masked_column_major_load
  tt.func public @masked_column_major_load(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) {
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked4}>> -> tensor<256x1xi32, #blocked4>
    %2 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<256x1x!tt.ptr<bf16>, #blocked4>
    %3 = tt.addptr %2, %1 : tensor<256x1x!tt.ptr<bf16>, #blocked4>, tensor<256x1xi32, #blocked4>
    %4 = tt.broadcast %3 : tensor<256x1x!tt.ptr<bf16>, #blocked4> -> tensor<256x32x!tt.ptr<bf16>, #blocked4>
    %mask = arith.constant dense<true> : tensor<256x32xi1, #blocked4>
    // CHECK: tt.load {{.*}} {ttig.block_io = "column_major"}
    tt.load %4, %mask : tensor<256x32x!tt.ptr<bf16>, #blocked4>
    tt.return
  }
}

// -----

// COM: Negative case - masked 1D load should NOT get block_io attribute (rank < 2).
#blocked5 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [16], warpsPerCTA = [32], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @masked_1d_load_no_block_io
  tt.func public @masked_1d_load_no_block_io(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) {
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked5>
    %1 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<256x!tt.ptr<bf16>, #blocked5>
    %2 = tt.addptr %1, %0 : tensor<256x!tt.ptr<bf16>, #blocked5>, tensor<256xi32, #blocked5>
    %mask = arith.constant dense<true> : tensor<256xi1, #blocked5>
    // CHECK-NOT: ttig.block_io
    tt.load %2, %mask : tensor<256x!tt.ptr<bf16>, #blocked5>
    tt.return
  }
}

// -----

// COM: Negative case - masked load with non-contiguous pattern should NOT get block_io attribute.
#blocked6 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 4], warpsPerCTA = [32, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @masked_non_contiguous_no_block_io
  tt.func public @masked_non_contiguous_no_block_io(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) {
    // Create strided (non-contiguous) access pattern
    %c2 = arith.constant 2 : i32
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked6}>>
    %c2_tensor = tt.splat %c2 : i32 -> tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked6}>>
    %strided = arith.muli %0, %c2_tensor : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked6}>>
    %1 = tt.expand_dims %strided {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked6}>> -> tensor<1x32xi32, #blocked6>
    %2 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<1x32x!tt.ptr<bf16>, #blocked6>
    %3 = tt.addptr %2, %1 : tensor<1x32x!tt.ptr<bf16>, #blocked6>, tensor<1x32xi32, #blocked6>
    %4 = tt.broadcast %3 : tensor<1x32x!tt.ptr<bf16>, #blocked6> -> tensor<256x32x!tt.ptr<bf16>, #blocked6>
    %mask = arith.constant dense<true> : tensor<256x32xi1, #blocked6>
    // CHECK-NOT: ttig.block_io
    tt.load %4, %mask : tensor<256x32x!tt.ptr<bf16>, #blocked6>
    tt.return
  }
}

// -----

// COM: 3D regular pointer.
#blocked = #ttg.blocked<{sizePerThread = [1, 4, 4], threadsPerWarp = [1, 4, 8], warpsPerCTA = [2, 2, 1], order = [2, 1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  tt.func public @_helion_bmm(%A: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c8_i32 = arith.constant 8 : i32
    %c16_i32 = arith.constant 16 : i32
    %c2_i32 = arith.constant 2 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %cst = arith.constant dense<786432> : tensor<2x1x1xi32, #blocked>
    %cst_0 = arith.constant dense<768> : tensor<1x32x1xi32, #blocked>
    %cst_1 = arith.constant dense<393216> : tensor<2x1x1xi32, #blocked>
    %0 = tt.get_program_id x : i32
    %1 = arith.remsi %0, %c8_i32 : i32
    %2 = arith.divsi %0, %c8_i32 : i32
    %3 = arith.remsi %2, %c16_i32 : i32
    %4 = arith.divsi %0, %c128_i32 : i32
    %5 = arith.muli %1, %c2_i32 : i32
    %6 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>
    %7 = tt.splat %5 : i32 -> tensor<2xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>
    %8 = arith.addi %7, %6 : tensor<2xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>
    %9 = arith.muli %3, %c32_i32 : i32
    %10 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>
    %11 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked}>}>>
    %12 = tt.splat %9 : i32 -> tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>
    %13 = arith.addi %12, %10 : tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>
    %14 = arith.muli %4, %c32_i32 : i32
    %15 = tt.splat %14 : i32 -> tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked}>}>>
    %16 = arith.addi %15, %11 : tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked}>}>>
    %17 = tt.expand_dims %8 {axis = 1 : i32} : tensor<2xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>> -> tensor<2x1xi32, #ttg.slice<{dim = 2, parent = #blocked}>>
    %18 = tt.expand_dims %17 {axis = 2 : i32} : tensor<2x1xi32, #ttg.slice<{dim = 2, parent = #blocked}>> -> tensor<2x1x1xi32, #blocked>
    %19 = arith.muli %18, %cst_1 : tensor<2x1x1xi32, #blocked>
    %20 = tt.expand_dims %13 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>> -> tensor<1x32xi32, #ttg.slice<{dim = 2, parent = #blocked}>>
    %21 = tt.expand_dims %20 {axis = 2 : i32} : tensor<1x32xi32, #ttg.slice<{dim = 2, parent = #blocked}>> -> tensor<1x32x1xi32, #blocked>
    %22 = arith.muli %21, %cst_0 : tensor<1x32x1xi32, #blocked>
    %23 = tt.broadcast %19 : tensor<2x1x1xi32, #blocked> -> tensor<2x32x1xi32, #blocked>
    %24 = tt.broadcast %22 : tensor<1x32x1xi32, #blocked> -> tensor<2x32x1xi32, #blocked>
    %25 = arith.addi %23, %24 : tensor<2x32x1xi32, #blocked>
    %26 = tt.broadcast %25 : tensor<2x32x1xi32, #blocked> -> tensor<2x32x32xi32, #blocked>
    %27 = tt.splat %A : !tt.ptr<f16> -> tensor<2x32x32x!tt.ptr<f16>, #blocked>
    %28 = arith.muli %18, %cst : tensor<2x1x1xi32, #blocked>
    %29 = tt.expand_dims %16 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked}>}>> -> tensor<1x32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %30 = tt.expand_dims %29 {axis = 1 : i32} : tensor<1x32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<1x1x32xi32, #blocked>
    %31 = tt.splat %c0_i32 : i32 -> tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked}>}>>
    %32 = tt.splat %c0_i32 : i32 -> tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>
    %33 = arith.addi %31, %11 : tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked}>}>>
    %34 = arith.addi %32, %10 : tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>
    %35 = tt.expand_dims %33 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked}>}>> -> tensor<1x32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %36 = tt.expand_dims %35 {axis = 1 : i32} : tensor<1x32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<1x1x32xi32, #blocked>
    %37 = tt.broadcast %36 : tensor<1x1x32xi32, #blocked> -> tensor<2x32x32xi32, #blocked>
    %38 = arith.addi %26, %37 : tensor<2x32x32xi32, #blocked>
    %39 = tt.addptr %27, %38 : tensor<2x32x32x!tt.ptr<f16>, #blocked>, tensor<2x32x32xi32, #blocked>
    // CHECK: tt.load {{.*}} {ttig.block_io = "row_major"}
    %40 = tt.load %39 evictionPolicy = evict_last : tensor<2x32x32x!tt.ptr<f16>, #blocked>
    // CHECK: tt.store {{.*}} {ttig.block_io = "row_major"}
    tt.store %39, %40 : tensor<2x32x32x!tt.ptr<f16>, #blocked>
    tt.return
  }
}
