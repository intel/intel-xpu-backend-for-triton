// RUN: triton-opt %s -split-input-file -tritonintelgpu-remove-layout-conversions | FileCheck %s

// COM: A load carrying `ttig.block_io` is not an expensive load, so it is not a layout
// COM: anchor and gets rematerialized into its consumer's encoding. That trade is only
// COM: worth it when the consumer is a dot, where the 2D block message delivers the DPAS
// COM: operand directly. A masked load with any other consumer stays anchored: it gives
// COM: up its coalesced encoding for nothing, and the mask bounds the resulting tile
// COM: besides. The attribute is kept either way, so the load can still lower to a 2D
// COM: block message -- just in the encoding the memory access itself wants.

// COM: Unmasked load: not an anchor, so it is rematerialized into the store's layout and
// COM: the convert disappears.

#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-DAG: #[[$STORE_LAYOUT:.+]] = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
  // CHECK-LABEL: tt.func public @unmasked_block_io_load_is_remat
  tt.func public @unmasked_block_io_load_is_remat(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<3> : tensor<64x64xi32, #blocked1>
    %cst_0 = arith.constant dense<64> : tensor<64x1xi32, #blocked>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %2 = arith.muli %1, %cst_0 : tensor<64x1xi32, #blocked>
    %3 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %4 = tt.expand_dims %3 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %5 = tt.broadcast %2 : tensor<64x1xi32, #blocked> -> tensor<64x64xi32, #blocked>
    %6 = tt.broadcast %4 : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked>
    %7 = arith.addi %5, %6 : tensor<64x64xi32, #blocked>
    %8 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>, #blocked>
    %9 = tt.addptr %8, %7 : tensor<64x64x!tt.ptr<f32>, #blocked>, tensor<64x64xi32, #blocked>
    // CHECK: tt.load {{.*}} {ttig.block_io = "row_major"} : tensor<64x64x!tt.ptr<f32>, #[[$STORE_LAYOUT]]>
    // CHECK-NOT: ttg.convert_layout
    %10 = tt.load %9 {ttig.block_io = "row_major"} : tensor<64x64x!tt.ptr<f32>, #blocked>
    %11 = ttg.convert_layout %10 : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked1>
    %12 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>, #blocked1>
    %13 = tt.addptr %12, %cst : tensor<64x64x!tt.ptr<f32>, #blocked1>, tensor<64x64xi32, #blocked1>
    tt.store %13, %11 : tensor<64x64x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

// COM: Same load, now masked, and no dot downstream: it stays an anchor, keeping the
// COM: layout the memory access wants, and the convert to the store's layout remains.

#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-DAG: #[[$LOAD_LAYOUT:.+]] = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
  // CHECK-DAG: #[[$STORE_LAYOUT:.+]] = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
  // CHECK-LABEL: tt.func public @masked_block_io_load_without_dot_stays_anchored
  tt.func public @masked_block_io_load_without_dot_stays_anchored(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32) {
    %cst = arith.constant dense<3> : tensor<64x64xi32, #blocked1>
    %cst_0 = arith.constant dense<64> : tensor<64x1xi32, #blocked>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %2 = arith.muli %1, %cst_0 : tensor<64x1xi32, #blocked>
    %3 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %4 = tt.expand_dims %3 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %5 = tt.broadcast %2 : tensor<64x1xi32, #blocked> -> tensor<64x64xi32, #blocked>
    %6 = tt.broadcast %4 : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked>
    %7 = arith.addi %5, %6 : tensor<64x64xi32, #blocked>
    %8 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>, #blocked>
    %9 = tt.addptr %8, %7 : tensor<64x64x!tt.ptr<f32>, #blocked>, tensor<64x64xi32, #blocked>
    %10 = tt.splat %arg2 : i32 -> tensor<64x1xi32, #blocked>
    %11 = arith.cmpi slt, %1, %10 : tensor<64x1xi32, #blocked>
    %12 = tt.broadcast %11 : tensor<64x1xi1, #blocked> -> tensor<64x64xi1, #blocked>
    // CHECK: %[[LOAD:.*]] = tt.load {{.*}} {ttig.block_io = "row_major"} : tensor<64x64x!tt.ptr<f32>, #[[$LOAD_LAYOUT]]>
    // CHECK: ttg.convert_layout %[[LOAD]] : tensor<64x64xf32, #[[$LOAD_LAYOUT]]> -> tensor<64x64xf32, #[[$STORE_LAYOUT]]>
    %13 = tt.load %9, %12 {ttig.block_io = "row_major"} : tensor<64x64x!tt.ptr<f32>, #blocked>
    %14 = ttg.convert_layout %13 : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked1>
    %15 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>, #blocked1>
    %16 = tt.addptr %15, %cst : tensor<64x64x!tt.ptr<f32>, #blocked1>, tensor<64x64xi32, #blocked1>
    tt.store %16, %14 : tensor<64x64x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

// COM: A masked load feeding a dot is still rematerialized: the load ends up in the dot
// COM: operand layout and the convert disappears, so the 2D block message loads the DPAS
// COM: operand straight into registers.

#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [1, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate} {
  // CHECK-DAG: #[[$DPAS:.+]] = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
  // CHECK-LABEL: tt.func public @masked_block_io_load_feeding_dot_is_remat
  tt.func public @masked_block_io_load_feeding_dot_is_remat(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: i32) -> tensor<64x16xf32, #dpas> {
    %cst = arith.constant dense<1.000000e+00> : tensor<32x16xbf16, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x16xf32, #dpas>
    %cst_1 = arith.constant dense<32> : tensor<64x1xi32, #blocked>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %2 = arith.muli %1, %cst_1 : tensor<64x1xi32, #blocked>
    %3 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %4 = tt.expand_dims %3 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %5 = tt.broadcast %2 : tensor<64x1xi32, #blocked> -> tensor<64x32xi32, #blocked>
    %6 = tt.broadcast %4 : tensor<1x32xi32, #blocked> -> tensor<64x32xi32, #blocked>
    %7 = arith.addi %5, %6 : tensor<64x32xi32, #blocked>
    %8 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<64x32x!tt.ptr<bf16>, #blocked>
    %9 = tt.addptr %8, %7 : tensor<64x32x!tt.ptr<bf16>, #blocked>, tensor<64x32xi32, #blocked>
    %10 = tt.splat %arg1 : i32 -> tensor<64x1xi32, #blocked>
    %11 = arith.cmpi slt, %1, %10 : tensor<64x1xi32, #blocked>
    %12 = tt.broadcast %11 : tensor<64x1xi1, #blocked> -> tensor<64x32xi1, #blocked>
    // CHECK: %[[LOAD:.*]] = tt.load {{.*}} {ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<bf16>, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>
    // CHECK-NOT: ttg.convert_layout
    // CHECK: tt.dot %[[LOAD]]
    %13 = tt.load %9, %12 {ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<bf16>, #blocked>
    %14 = ttg.convert_layout %13 : tensor<64x32xbf16, #blocked> -> tensor<64x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    %15 = tt.dot %14, %cst, %cst_0, inputPrecision = tf32 : tensor<64x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>> * tensor<32x16xbf16, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>> -> tensor<64x16xf32, #dpas>
    tt.return %15 : tensor<64x16xf32, #dpas>
  }
}
