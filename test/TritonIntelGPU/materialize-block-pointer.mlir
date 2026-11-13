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

// COM: The row bound is an opaque runtime scalar (mask constancy 1), but it's
// COM: a pure row-limit pattern (`row < bound`) with a compile-time pointer
// COM: stride (not broadcast), so ttgi::matchRowLimitMask permits block IO
// COM: and the LLVM lowering expresses the bound as base_height (issue #8565).
#blocked7 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 4], warpsPerCTA = [32, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @masked_opaque_row_bound_load
  tt.func public @masked_opaque_row_bound_load(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}) {
    %m_size = tt.load %arg1 : !tt.ptr<i32>
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked7}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked7}>> -> tensor<1x32xi32, #blocked7>
    %5 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked7}>>
    %6 = tt.expand_dims %5 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked7}>> -> tensor<256x1xi32, #blocked7>
    %cst_stride = arith.constant dense<64> : tensor<256x1xi32, #blocked7>
    %row_off = arith.muli %6, %cst_stride : tensor<256x1xi32, #blocked7>
    %row_off_bc = tt.broadcast %row_off : tensor<256x1xi32, #blocked7> -> tensor<256x32xi32, #blocked7>
    %col_off_bc = tt.broadcast %1 : tensor<1x32xi32, #blocked7> -> tensor<256x32xi32, #blocked7>
    %off = arith.addi %row_off_bc, %col_off_bc : tensor<256x32xi32, #blocked7>
    %base = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<256x32x!tt.ptr<bf16>, #blocked7>
    %4 = tt.addptr %base, %off : tensor<256x32x!tt.ptr<bf16>, #blocked7>, tensor<256x32xi32, #blocked7>
    %7 = tt.splat %m_size : i32 -> tensor<256x1xi32, #blocked7>
    %8 = arith.cmpi slt, %6, %7 : tensor<256x1xi32, #blocked7>
    %mask = tt.broadcast %8 : tensor<256x1xi1, #blocked7> -> tensor<256x32xi1, #blocked7>
    // CHECK: tt.load {{.*}} {ttig.block_io = "row_major"}
    tt.load %4, %mask : tensor<256x32x!tt.ptr<bf16>, #blocked7>
    tt.return
  }
}

// -----

// COM: The same opaque row bound ANDed with two scalar flags (`tt.splat`s of
// COM: an i1). Both are uniform, so the mask is still a row-limit pattern and
// COM: block IO is permitted, as LowerTo2DBlockLoad will absorb it.
#blocked7 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 4], warpsPerCTA = [32, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @masked_opaque_row_bound_uniform_conjuncts_load
  tt.func public @masked_opaque_row_bound_uniform_conjuncts_load(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %f0: i1, %f1: i1) {
    %m_size = tt.load %arg1 : !tt.ptr<i32>
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked7}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked7}>> -> tensor<1x32xi32, #blocked7>
    %5 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked7}>>
    %6 = tt.expand_dims %5 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked7}>> -> tensor<256x1xi32, #blocked7>
    %cst_stride = arith.constant dense<64> : tensor<256x1xi32, #blocked7>
    %row_off = arith.muli %6, %cst_stride : tensor<256x1xi32, #blocked7>
    %row_off_bc = tt.broadcast %row_off : tensor<256x1xi32, #blocked7> -> tensor<256x32xi32, #blocked7>
    %col_off_bc = tt.broadcast %1 : tensor<1x32xi32, #blocked7> -> tensor<256x32xi32, #blocked7>
    %off = arith.addi %row_off_bc, %col_off_bc : tensor<256x32xi32, #blocked7>
    %base = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<256x32x!tt.ptr<bf16>, #blocked7>
    %4 = tt.addptr %base, %off : tensor<256x32x!tt.ptr<bf16>, #blocked7>, tensor<256x32xi32, #blocked7>
    %7 = tt.splat %m_size : i32 -> tensor<256x1xi32, #blocked7>
    %8 = arith.cmpi slt, %6, %7 : tensor<256x1xi32, #blocked7>
    %rowmask = tt.broadcast %8 : tensor<256x1xi1, #blocked7> -> tensor<256x32xi1, #blocked7>
    %s0 = tt.splat %f0 : i1 -> tensor<256x32xi1, #blocked7>
    %s1 = tt.splat %f1 : i1 -> tensor<256x32xi1, #blocked7>
    %m0 = arith.andi %rowmask, %s0 : tensor<256x32xi1, #blocked7>
    %mask = arith.andi %m0, %s1 : tensor<256x32xi1, #blocked7>
    // CHECK: tt.load {{.*}} {ttig.block_io = "row_major"}
    tt.load %4, %mask : tensor<256x32x!tt.ptr<bf16>, #blocked7>
    tt.return
  }
}

// -----

// COM: i64 counterpart of @masked_opaque_row_bound_load, the form Helion's
// COM: grouped GEMM emits (issue #8565): rows = (group_start + tile_off) in
// COM: i64 plus a sign-extended arange, compared against an i32 bound
// COM: sign-extended to i64. The row offsets feeding the pointer are i64 too.
#blocked7 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 4], warpsPerCTA = [32, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @masked_opaque_row_bound_load_i64
  tt.func public @masked_opaque_row_bound_load_i64(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %tile_m: i32) {
    %c256_i32 = arith.constant 256 : i32
    %c1_i32 = arith.constant 1 : i32
    %group_start = tt.load %arg1 : !tt.ptr<i32>
    %end_ptr = tt.addptr %arg1, %c1_i32 : !tt.ptr<i32>, i32
    %group_end = tt.load %end_ptr : !tt.ptr<i32>
    %tile_off = arith.muli %tile_m, %c256_i32 : i32
    %tile_off64 = arith.extsi %tile_off : i32 to i64
    %group_start64 = arith.extsi %group_start : i32 to i64
    %row_base64 = arith.addi %group_start64, %tile_off64 : i64
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked7}>>
    %1 = arith.extsi %0 : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked7}>> to tensor<32xi64, #ttg.slice<{dim = 0, parent = #blocked7}>>
    %2 = tt.expand_dims %1 {axis = 0 : i32} : tensor<32xi64, #ttg.slice<{dim = 0, parent = #blocked7}>> -> tensor<1x32xi64, #blocked7>
    %5 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked7}>>
    %r5_64 = arith.extsi %5 : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked7}>> to tensor<256xi64, #ttg.slice<{dim = 1, parent = #blocked7}>>
    %base_splat = tt.splat %row_base64 : i64 -> tensor<256xi64, #ttg.slice<{dim = 1, parent = #blocked7}>>
    %rows = arith.addi %base_splat, %r5_64 : tensor<256xi64, #ttg.slice<{dim = 1, parent = #blocked7}>>
    %6 = tt.expand_dims %rows {axis = 1 : i32} : tensor<256xi64, #ttg.slice<{dim = 1, parent = #blocked7}>> -> tensor<256x1xi64, #blocked7>
    %cst_stride = arith.constant dense<64> : tensor<256x1xi64, #blocked7>
    %row_off = arith.muli %6, %cst_stride : tensor<256x1xi64, #blocked7>
    %row_off_bc = tt.broadcast %row_off : tensor<256x1xi64, #blocked7> -> tensor<256x32xi64, #blocked7>
    %col_off_bc = tt.broadcast %2 : tensor<1x32xi64, #blocked7> -> tensor<256x32xi64, #blocked7>
    %off = arith.addi %row_off_bc, %col_off_bc : tensor<256x32xi64, #blocked7>
    %base = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<256x32x!tt.ptr<bf16>, #blocked7>
    %4 = tt.addptr %base, %off : tensor<256x32x!tt.ptr<bf16>, #blocked7>, tensor<256x32xi64, #blocked7>
    %group_end64 = arith.extsi %group_end : i32 to i64
    %bound = tt.splat %group_end64 : i64 -> tensor<256xi64, #ttg.slice<{dim = 1, parent = #blocked7}>>
    %cmp = arith.cmpi slt, %rows, %bound : tensor<256xi64, #ttg.slice<{dim = 1, parent = #blocked7}>>
    %8 = tt.expand_dims %cmp {axis = 1 : i32} : tensor<256xi1, #ttg.slice<{dim = 1, parent = #blocked7}>> -> tensor<256x1xi1, #blocked7>
    %mask = tt.broadcast %8 : tensor<256x1xi1, #blocked7> -> tensor<256x32xi1, #blocked7>
    // CHECK: tt.load {{.*}} {ttig.block_io = "row_major"}
    tt.load %4, %mask : tensor<256x32x!tt.ptr<bf16>, #blocked7>
    tt.return
  }
}

// -----

// COM: Both dims bounded, as `RewriteTensorDescriptorToPointer` emits: the
// COM: column bound's granularity is provable (`tt.divisibility` on the bound,
// COM: a program-id multiple for the base), and so is the base's sign, so
// COM: ttgi::matchBoundaryCheckMask permits block IO and the lowering
// COM: expresses the bounds as base_height/base_width.
#blocked8 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 4], warpsPerCTA = [32, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @masked_both_dims_bounded_load
  tt.func public @masked_both_dims_bounded_load(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %rows: i32, %cols: i32 {tt.divisibility = 16 : i32}) {
    %pid = tt.get_program_id x : i32
    %c32_i32 = arith.constant 32 : i32
    %offk = arith.muli %pid, %c32_i32 : i32
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked8}>>
    %col_base = tt.splat %offk : i32 -> tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked8}>>
    %col_idx = arith.addi %col_base, %0 : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked8}>>
    %1 = tt.expand_dims %col_idx {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked8}>> -> tensor<1x32xi32, #blocked8>
    %5 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked8}>>
    %6 = tt.expand_dims %5 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked8}>> -> tensor<256x1xi32, #blocked8>
    %cst_stride = arith.constant dense<64> : tensor<256x1xi32, #blocked8>
    %row_off = arith.muli %6, %cst_stride : tensor<256x1xi32, #blocked8>
    %row_off_bc = tt.broadcast %row_off : tensor<256x1xi32, #blocked8> -> tensor<256x32xi32, #blocked8>
    %col_off_bc = tt.broadcast %1 : tensor<1x32xi32, #blocked8> -> tensor<256x32xi32, #blocked8>
    %off = arith.addi %row_off_bc, %col_off_bc : tensor<256x32xi32, #blocked8>
    %base = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<256x32x!tt.ptr<bf16>, #blocked8>
    %4 = tt.addptr %base, %off : tensor<256x32x!tt.ptr<bf16>, #blocked8>, tensor<256x32xi32, #blocked8>
    %7 = tt.splat %rows : i32 -> tensor<256x1xi32, #blocked8>
    %8 = arith.cmpi slt, %6, %7 : tensor<256x1xi32, #blocked8>
    %row_mask = tt.broadcast %8 : tensor<256x1xi1, #blocked8> -> tensor<256x32xi1, #blocked8>
    %9 = tt.splat %cols : i32 -> tensor<1x32xi32, #blocked8>
    %10 = arith.cmpi slt, %1, %9 : tensor<1x32xi32, #blocked8>
    %col_mask = tt.broadcast %10 : tensor<1x32xi1, #blocked8> -> tensor<256x32xi1, #blocked8>
    %mask = arith.andi %row_mask, %col_mask : tensor<256x32xi1, #blocked8>
    // CHECK: tt.load {{.*}} {ttig.block_io = "row_major"}
    tt.load %4, %mask : tensor<256x32x!tt.ptr<bf16>, #blocked8>
    tt.return
  }
}

// -----

// COM: (Opaque bound, so only the exclusion in matchRowLimitMask keeps
// COM: block IO off; if the plumbing broke, this would start matching.)
// COM: Same mask shape, but the pointer is broadcast (stride zero) along the
// COM: bounded dim, so matchRowLimitMask excludes it and the ordinary
// COM: constancy check rejects the opaque bound.
#blocked9 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 4], warpsPerCTA = [32, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @masked_broadcast_row_no_block_io
  tt.func public @masked_broadcast_row_no_block_io(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}) {
    %m_size = tt.load %arg1 : !tt.ptr<i32>
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked9}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked9}>> -> tensor<1x32xi32, #blocked9>
    %2 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<1x32x!tt.ptr<bf16>, #blocked9>
    %3 = tt.addptr %2, %1 : tensor<1x32x!tt.ptr<bf16>, #blocked9>, tensor<1x32xi32, #blocked9>
    %4 = tt.broadcast %3 : tensor<1x32x!tt.ptr<bf16>, #blocked9> -> tensor<256x32x!tt.ptr<bf16>, #blocked9>
    %5 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked9}>>
    %6 = tt.expand_dims %5 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked9}>> -> tensor<256x1xi32, #blocked9>
    %7 = tt.splat %m_size : i32 -> tensor<256x1xi32, #blocked9>
    %8 = arith.cmpi slt, %6, %7 : tensor<256x1xi32, #blocked9>
    %mask = tt.broadcast %8 : tensor<256x1xi1, #blocked9> -> tensor<256x32xi1, #blocked9>
    // CHECK-NOT: ttig.block_io
    tt.load %4, %mask : tensor<256x32x!tt.ptr<bf16>, #blocked9>
    tt.return
  }
}

// -----

// COM: The same mask shape as above, but the row bound is known to be a multiple
// COM: of 16, so the mask is uniform over a 16-row tile and block IO is kept.
#blocked8 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 4], warpsPerCTA = [32, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @masked_divisible_row_bound_load
  tt.func public @masked_divisible_row_bound_load(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %m_size: i32 {tt.divisibility = 16 : i32}) {
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked8}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked8}>> -> tensor<1x32xi32, #blocked8>
    %2 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<1x32x!tt.ptr<bf16>, #blocked8>
    %3 = tt.addptr %2, %1 : tensor<1x32x!tt.ptr<bf16>, #blocked8>, tensor<1x32xi32, #blocked8>
    %4 = tt.broadcast %3 : tensor<1x32x!tt.ptr<bf16>, #blocked8> -> tensor<256x32x!tt.ptr<bf16>, #blocked8>
    %5 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked8}>>
    %6 = tt.expand_dims %5 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked8}>> -> tensor<256x1xi32, #blocked8>
    %7 = tt.splat %m_size : i32 -> tensor<256x1xi32, #blocked8>
    %8 = arith.cmpi slt, %6, %7 : tensor<256x1xi32, #blocked8>
    %mask = tt.broadcast %8 : tensor<256x1xi1, #blocked8> -> tensor<256x32xi1, #blocked8>
    // CHECK: tt.load {{.*}} {ttig.block_io = "row_major"}
    tt.load %4, %mask : tensor<256x32x!tt.ptr<bf16>, #blocked8>
    tt.return
  }
}

// -----

// COM: (Opaque bound, so only the exclusion in matchRowLimitMask keeps
// COM: block IO off; if the plumbing broke, this would start matching.)
// COM: Same mask shape as masked_opaque_row_bound_load, but the mask bounds
// COM: the memory-contiguous (column) dim instead of the pitch (row) dim.
// COM: matchRowLimitMask only accepts a bound along the pitch dim.
#blocked11 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 4], warpsPerCTA = [32, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @masked_contig_bound_no_block_io
  tt.func public @masked_contig_bound_no_block_io(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}) {
    %m_size = tt.load %arg1 : !tt.ptr<i32>
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked11}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked11}>> -> tensor<1x32xi32, #blocked11>
    %5 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked11}>>
    %6 = tt.expand_dims %5 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked11}>> -> tensor<256x1xi32, #blocked11>
    %cst_stride = arith.constant dense<64> : tensor<256x1xi32, #blocked11>
    %row_off = arith.muli %6, %cst_stride : tensor<256x1xi32, #blocked11>
    %row_off_bc = tt.broadcast %row_off : tensor<256x1xi32, #blocked11> -> tensor<256x32xi32, #blocked11>
    %col_off_bc = tt.broadcast %1 : tensor<1x32xi32, #blocked11> -> tensor<256x32xi32, #blocked11>
    %off = arith.addi %row_off_bc, %col_off_bc : tensor<256x32xi32, #blocked11>
    %base = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<256x32x!tt.ptr<bf16>, #blocked11>
    %4 = tt.addptr %base, %off : tensor<256x32x!tt.ptr<bf16>, #blocked11>, tensor<256x32xi32, #blocked11>
    %9 = tt.splat %m_size : i32 -> tensor<1x32xi32, #blocked11>
    %10 = arith.cmpi slt, %1, %9 : tensor<1x32xi32, #blocked11>
    %mask = tt.broadcast %10 : tensor<1x32xi1, #blocked11> -> tensor<256x32xi1, #blocked11>
    // CHECK-NOT: ttig.block_io
    tt.load %4, %mask : tensor<256x32x!tt.ptr<bf16>, #blocked11>
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
    // CHECK: tt.load {{.*}}ttig.block_io = "row_major"
    %40 = tt.load %39 {cachePolicy = #tt.cache_policy<cache_modifier = none, eviction_policy = evict_last>} : tensor<2x32x32x!tt.ptr<f16>, #blocked>
    // CHECK: tt.store {{.*}} {ttig.block_io = "row_major"}
    tt.store %39, %40 : tensor<2x32x32x!tt.ptr<f16>, #blocked>
    tt.return
  }
}


// -----

// COM: Ensure offsets derived from a loop induction variable preserve divisibility for block IO lowering.
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @offset_loop_var
  tt.func public @offset_loop_var(%x_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %w_g_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %w_fc_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %b_g_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %b_fc_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %y_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %M: i32 {tt.divisibility = 16 : i32}, %N: i32 {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %desc_x = arith.extsi %K : i32 to i64
    %desc = tt.make_tensor_descriptor %x_ptr, [%M, %K], [%desc_x, %c1_i64] : <bf16>, <128x64xbf16>
    scf.for %k = %c0_i32 to %K step %c64_i32 : i32 {
      // CHECK: tt.descriptor_load {{.*}} {ttig.block_io = "row_major", ttig.desc_padding = 1 : i32}
      %x = tt.descriptor_load %desc[%c128_i32, %k] : !tt.tensordesc<128x64xbf16> -> tensor<128x64xbf16, #blocked>
    }
    tt.return
  }
}

// -----
// COM: A non-divisible loop step must not enable block IO lowering.
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @offset_loop_var_indivisible
  tt.func public @offset_loop_var_indivisible(%x_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %M: i32 {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c3_i32 = arith.constant 3 : i32
    %c128_i32 = arith.constant 128 : i32
    %desc_x = arith.extsi %K : i32 to i64
    %desc = tt.make_tensor_descriptor %x_ptr, [%M, %K], [%desc_x, %c1_i64] : <bf16>, <128x64xbf16>
    scf.for %k = %c0_i32 to %K step %c3_i32 : i32 {
      // CHECK-NOT: ttig.block_io
      %x = tt.descriptor_load %desc[%c128_i32, %k] : !tt.tensordesc<128x64xbf16> -> tensor<128x64xbf16, #blocked>
    }
    tt.return
  }
}

// -----
// COM: A nested-loop induction variable must not enable block IO lowering.
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @offset_nested_loop_var
  tt.func public @offset_nested_loop_var(%x_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %M: i32 {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %desc_x = arith.extsi %K : i32 to i64
    %desc = tt.make_tensor_descriptor %x_ptr, [%M, %K], [%desc_x, %c1_i64] : <bf16>, <128x64xbf16>
    scf.for %i = %c0_i32 to %M step %c64_i32 : i32 {
      scf.for %k = %c0_i32 to %K step %c64_i32 : i32 {
        // CHECK-NOT: ttig.block_io
        %x = tt.descriptor_load %desc[%c128_i32, %k] : !tt.tensordesc<128x64xbf16> -> tensor<128x64xbf16, #blocked>
      }
    }
    tt.return
  }
}

// -----
// COM: A loop-carried offset stepped by a non-divisible amount must not enable block IO:
// COM: the index is odd on every other iteration, and the 2D block message
// COM: requires a 2-element aligned X for 16-bit elements (issue #7990).
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @offset_loop_carried_indivisible
  tt.func public @offset_loop_carried_indivisible(%x_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %M: i32 {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c3_i32 = arith.constant 3 : i32
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %desc_x = arith.extsi %K : i32 to i64
    %desc = tt.make_tensor_descriptor %x_ptr, [%M, %K], [%desc_x, %c1_i64] : <bf16>, <128x64xbf16>
    scf.for %i = %c0_i32 to %K step %c64_i32 iter_args(%off = %c0_i32) -> (i32) : i32 {
      // CHECK: tt.descriptor_load
      // CHECK-NOT: ttig.block_io
      // CHECK-SAME: !tt.tensordesc
      %x = tt.descriptor_load %desc[%c128_i32, %off] : !tt.tensordesc<128x64xbf16> -> tensor<128x64xbf16, #blocked>
      %next = arith.addi %off, %c3_i32 : i32
      scf.yield %next : i32
    }
    tt.return
  }
}

// -----
// COM: A loop-carried offset stepped by a divisible amount must keep block IO lowering.
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @offset_loop_carried_divisible
  tt.func public @offset_loop_carried_divisible(%x_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %M: i32 {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %desc_x = arith.extsi %K : i32 to i64
    %desc = tt.make_tensor_descriptor %x_ptr, [%M, %K], [%desc_x, %c1_i64] : <bf16>, <128x64xbf16>
    scf.for %i = %c0_i32 to %K step %c64_i32 iter_args(%off = %c0_i32) -> (i32) : i32 {
      // CHECK: tt.descriptor_load {{.*}} {ttig.block_io = "row_major", ttig.desc_padding = 1 : i32}
      %x = tt.descriptor_load %desc[%c128_i32, %off] : !tt.tensordesc<128x64xbf16> -> tensor<128x64xbf16, #blocked>
      %next = arith.addi %off, %c32_i32 : i32
      scf.yield %next : i32
    }
    tt.return
  }
}

// -----
// COM: The stream-K shape (gemm_streamk_benchmark.py): a loop-carried offset whose
// COM: initial value is itself a multiple of the step must keep block IO lowering.
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @offset_loop_carried_divisible_mul_init
  tt.func public @offset_loop_carried_divisible_mul_init(%x_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %M: i32 {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32}, %remain_iters: i32) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %desc_x = arith.extsi %K : i32 to i64
    %desc = tt.make_tensor_descriptor %x_ptr, [%M, %K], [%desc_x, %c1_i64] : <bf16>, <128x64xbf16>
    %init_off = arith.muli %remain_iters, %c32_i32 : i32
    scf.for %i = %c0_i32 to %K step %c64_i32 iter_args(%off = %init_off) -> (i32) : i32 {
      // CHECK: tt.descriptor_load {{.*}} {ttig.block_io = "row_major", ttig.desc_padding = 1 : i32}
      %x = tt.descriptor_load %desc[%c128_i32, %off] : !tt.tensordesc<128x64xbf16> -> tensor<128x64xbf16, #blocked>
      %next = arith.addi %off, %c32_i32 : i32
      scf.yield %next : i32
    }
    tt.return
  }
}

// -----
// COM: The persistent / stream-K shape: a K loop nested in a tile loop must keep block IO
// COM: lowering for its loop-carried offset. Unlike @offset_nested_loop_var the index is an
// COM: iteration argument, not the induction variable, and the nested-loop refusal only
// COM: applies to the latter.
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @offset_loop_carried_nested_loop
  tt.func public @offset_loop_carried_nested_loop(%x_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %M: i32 {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %desc_x = arith.extsi %K : i32 to i64
    %desc = tt.make_tensor_descriptor %x_ptr, [%M, %K], [%desc_x, %c1_i64] : <bf16>, <128x64xbf16>
    scf.for %tile = %c0_i32 to %M step %c64_i32 : i32 {
      scf.for %i = %c0_i32 to %K step %c64_i32 iter_args(%off = %c0_i32) -> (i32) : i32 {
        // CHECK: tt.descriptor_load {{.*}} {ttig.block_io = "row_major", ttig.desc_padding = 1 : i32}
        %x = tt.descriptor_load %desc[%c128_i32, %off] : !tt.tensordesc<128x64xbf16> -> tensor<128x64xbf16, #blocked>
        %next = arith.addi %off, %c32_i32 : i32
        scf.yield %next : i32
      }
    }
    tt.return
  }
}

// -----
// COM: A descriptor rebuilt in the loop with a loop-carried stride-one extent that starts at 63
// COM: and advances by 32: the extent is odd on every iteration, so block IO must be refused
// COM: although the increment is divisible. Unlike the index, the extent is checked by
// COM: ttgi::isDivisible alone, with no axis-info check behind it.
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @extent_loop_carried_odd_init
  tt.func public @extent_loop_carried_odd_init(%x_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %M: i32 {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c32_i32 = arith.constant 32 : i32
    %c63_i32 = arith.constant 63 : i32
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %desc_x = arith.extsi %K : i32 to i64
    scf.for %i = %c0_i32 to %K step %c64_i32 iter_args(%w = %c63_i32) -> (i32) : i32 {
      %desc = tt.make_tensor_descriptor %x_ptr, [%M, %w], [%desc_x, %c1_i64] : <bf16>, <128x64xbf16>
      // CHECK: tt.descriptor_load
      // CHECK-NOT: ttig.block_io
      // CHECK-SAME: !tt.tensordesc
      %x = tt.descriptor_load %desc[%c128_i32, %c0_i32] : !tt.tensordesc<128x64xbf16> -> tensor<128x64xbf16, #blocked>
      %next = arith.addi %w, %c32_i32 : i32
      scf.yield %next : i32
    }
    tt.return
  }
}

// -----
// COM: 8-bit elements need a 4-element aligned index, so a loop-carried offset stepped
// COM: by 2 must not enable block IO lowering even though 2 is even.
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @offset_loop_carried_i8_indivisible
  tt.func public @offset_loop_carried_i8_indivisible(%x_ptr: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %M: i32 {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c2_i32 = arith.constant 2 : i32
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %desc_x = arith.extsi %K : i32 to i64
    %desc = tt.make_tensor_descriptor %x_ptr, [%M, %K], [%desc_x, %c1_i64] : <i8>, <128x64xi8>
    scf.for %i = %c0_i32 to %K step %c64_i32 iter_args(%off = %c0_i32) -> (i32) : i32 {
      // CHECK: tt.descriptor_load
      // CHECK-NOT: ttig.block_io
      // CHECK-SAME: !tt.tensordesc
      %x = tt.descriptor_load %desc[%c128_i32, %off] : !tt.tensordesc<128x64xi8> -> tensor<128x64xi8, #blocked>
      %next = arith.addi %off, %c2_i32 : i32
      scf.yield %next : i32
    }
    tt.return
  }
}

// -----
// COM: A loop-carried offset stepped by a function argument carrying a tt.divisibility
// COM: hint must keep block IO lowering.
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @offset_loop_carried_step_divisible_arg
  tt.func public @offset_loop_carried_step_divisible_arg(%x_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %M: i32 {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %desc_x = arith.extsi %K : i32 to i64
    %desc = tt.make_tensor_descriptor %x_ptr, [%M, %K], [%desc_x, %c1_i64] : <bf16>, <128x64xbf16>
    scf.for %i = %c0_i32 to %M step %c64_i32 iter_args(%off = %c0_i32) -> (i32) : i32 {
      // CHECK: tt.descriptor_load {{.*}} {ttig.block_io = "row_major", ttig.desc_padding = 1 : i32}
      %x = tt.descriptor_load %desc[%c128_i32, %off] : !tt.tensordesc<128x64xbf16> -> tensor<128x64xbf16, #blocked>
      %next = arith.addi %off, %K : i32
      scf.yield %next : i32
    }
    tt.return
  }
}

// -----
// COM: A loop-carried offset stepped by an unconstrained runtime value is not provably
// COM: aligned, so block IO must be refused. This is intentional: a kernel that knows
// COM: the step is aligned can say so with tl.multiple_of.
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @offset_loop_carried_runtime_step
  tt.func public @offset_loop_carried_runtime_step(%x_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %M: i32 {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32}, %step: i32) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %desc_x = arith.extsi %K : i32 to i64
    %desc = tt.make_tensor_descriptor %x_ptr, [%M, %K], [%desc_x, %c1_i64] : <bf16>, <128x64xbf16>
    scf.for %i = %c0_i32 to %M step %c64_i32 iter_args(%off = %c0_i32) -> (i32) : i32 {
      // CHECK: tt.descriptor_load
      // CHECK-NOT: ttig.block_io
      // CHECK-SAME: !tt.tensordesc
      %x = tt.descriptor_load %desc[%c128_i32, %off] : !tt.tensordesc<128x64xbf16> -> tensor<128x64xbf16, #blocked>
      %next = arith.addi %off, %step : i32
      scf.yield %next : i32
    }
    tt.return
  }
}

// -----

// COM: Rank-3 row-limit mask ANDed with a batch-only predicate, with the AND
// COM: sunk below the final broadcast, as `tt-reorder-broadcast` leaves it:
// COM: broadcast(and(broadcast(row), broadcast(batch))). matchRowLimitMask
// COM: looks through the narrow AND, so the load gets block IO and its mask is
// COM: left untouched.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4, 2], repCluster = [1, 1, 1], A = [1, 8, 16], B = [1, 16, 16], C = [1, 8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#sK = #ttg.slice<{dim = 2, parent = #dot0}>
#sM = #ttg.slice<{dim = 1, parent = #dot0}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @row_limit_rank3_sunk_and
  tt.func public @row_limit_rank3_sunk_and(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %bound: i32, %batch_bound: i32) {
    %zero = arith.constant dense<0.000000e+00> : tensor<4x64x32xf16, #dot0>
    %c32 = arith.constant dense<32> : tensor<1x64x1xi32, #dot0>
    %rb = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #sK}>>
    %rm = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #sK}>>
    %ema = tt.expand_dims %rm {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #sK}>> -> tensor<1x64xi32, #sK>
    %em = tt.expand_dims %ema {axis = 2 : i32} : tensor<1x64xi32, #sK> -> tensor<1x64x1xi32, #dot0>
    %om = arith.muli %em, %c32 : tensor<1x64x1xi32, #dot0>
    %rk = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #sM}>>
    %eka = tt.expand_dims %rk {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #sM}>> -> tensor<1x32xi32, #sM>
    %ek = tt.expand_dims %eka {axis = 1 : i32} : tensor<1x32xi32, #sM> -> tensor<1x1x32xi32, #dot0>
    %bm = tt.broadcast %om : tensor<1x64x1xi32, #dot0> -> tensor<4x64x32xi32, #dot0>
    %bk = tt.broadcast %ek : tensor<1x1x32xi32, #dot0> -> tensor<4x64x32xi32, #dot0>
    %off = arith.addi %bm, %bk : tensor<4x64x32xi32, #dot0>
    %ps = tt.splat %arg0 : !tt.ptr<f16> -> tensor<4x64x32x!tt.ptr<f16>, #dot0>
    %ptr = tt.addptr %ps, %off : tensor<4x64x32x!tt.ptr<f16>, #dot0>, tensor<4x64x32xi32, #dot0>
    %boundsplat = tt.splat %bound : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #sK}>>
    %rowcmp = arith.cmpi slt, %rm, %boundsplat : tensor<64xi32, #ttg.slice<{dim = 0, parent = #sK}>>
    %rc1 = tt.expand_dims %rowcmp {axis = 0 : i32} : tensor<64xi1, #ttg.slice<{dim = 0, parent = #sK}>> -> tensor<1x64xi1, #sK>
    %rc2 = tt.expand_dims %rc1 {axis = 2 : i32} : tensor<1x64xi1, #sK> -> tensor<1x64x1xi1, #dot0>
    %bboundsplat = tt.splat %batch_bound : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #sK}>>
    %batchcmp = arith.cmpi slt, %rb, %bboundsplat : tensor<4xi32, #ttg.slice<{dim = 1, parent = #sK}>>
    %bc1 = tt.expand_dims %batchcmp {axis = 1 : i32} : tensor<4xi1, #ttg.slice<{dim = 1, parent = #sK}>> -> tensor<4x1xi1, #sK>
    %bc2 = tt.expand_dims %bc1 {axis = 2 : i32} : tensor<4x1xi1, #sK> -> tensor<4x1x1xi1, #dot0>
    // The AND is on [4, 64, 1], below the final broadcast.
    %rowb = tt.broadcast %rc2 : tensor<1x64x1xi1, #dot0> -> tensor<4x64x1xi1, #dot0>
    %batchb = tt.broadcast %bc2 : tensor<4x1x1xi1, #dot0> -> tensor<4x64x1xi1, #dot0>
    %and = arith.andi %rowb, %batchb : tensor<4x64x1xi1, #dot0>
    %mask = tt.broadcast %and : tensor<4x64x1xi1, #dot0> -> tensor<4x64x32xi1, #dot0>
    // CHECK: %[[AND:.*]] = arith.andi {{.*}} : tensor<4x64x1xi1, #{{.*}}>
    // CHECK: %[[MASK:.*]] = tt.broadcast %[[AND]] : tensor<4x64x1xi1, #{{.*}}> -> tensor<4x64x32xi1, #{{.*}}>
    // CHECK: tt.load %{{.*}}, %[[MASK]], %{{.*}} {ttig.block_io = "row_major"}
    %v = tt.load %ptr, %mask, %zero : tensor<4x64x32x!tt.ptr<f16>, #dot0>
    tt.return
  }
}

// -----

// COM: A row-limit mask, but the compile-time row pitch is 32 bytes (< 64), so
// COM: LowerTo2DBlockLoad would reject the load: the bypass must not stamp it.
#blocked10 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 2], warpsPerCTA = [32, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @row_limit_narrow_const_pitch
  tt.func public @row_limit_narrow_const_pitch(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}) {
    %m_size = tt.load %arg1 : !tt.ptr<i32>
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked10}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked10}>> -> tensor<1x16xi32, #blocked10>
    %5 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked10}>>
    %6 = tt.expand_dims %5 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked10}>> -> tensor<256x1xi32, #blocked10>
    %cst_stride = arith.constant dense<16> : tensor<256x1xi32, #blocked10>
    %row_off = arith.muli %6, %cst_stride : tensor<256x1xi32, #blocked10>
    %row_off_bc = tt.broadcast %row_off : tensor<256x1xi32, #blocked10> -> tensor<256x16xi32, #blocked10>
    %col_off_bc = tt.broadcast %1 : tensor<1x16xi32, #blocked10> -> tensor<256x16xi32, #blocked10>
    %off = arith.addi %row_off_bc, %col_off_bc : tensor<256x16xi32, #blocked10>
    %base = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<256x16x!tt.ptr<bf16>, #blocked10>
    %4 = tt.addptr %base, %off : tensor<256x16x!tt.ptr<bf16>, #blocked10>, tensor<256x16xi32, #blocked10>
    %7 = tt.splat %m_size : i32 -> tensor<256x1xi32, #blocked10>
    %8 = arith.cmpi slt, %6, %7 : tensor<256x1xi32, #blocked10>
    %mask = tt.broadcast %8 : tensor<256x1xi1, #blocked10> -> tensor<256x16xi1, #blocked10>
    // CHECK-NOT: ttig.block_io
    tt.load %4, %mask : tensor<256x16x!tt.ptr<bf16>, #blocked10>
    tt.return
  }
}
