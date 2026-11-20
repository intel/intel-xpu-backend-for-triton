// RUN: triton-opt %s -split-input-file -tritonintelgpu-remove-layout-conversions | FileCheck %s

// COM: Issue #8819: a load whose pointer is broadcast along a dimension, such as an addmm
// COM: bias with strides (0, 1), reads each distinct address once per row. When it feeds a
// COM: DPAS value it is not an anchor, however many logical elements it has: the load is
// COM: rematerialized into the DPAS accumulator's encoding and the blocked -> DPAS convert,
// COM: which goes through shared memory, disappears. The store shares the load's mask, so it
// COM: follows into DPAS and the DPAS -> blocked convert of the output disappears too.

#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [1, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate} {
  // CHECK-DAG: #[[$DPAS:.+]] = #ttig.dpas<{{.*}}warpsPerCTA = [4, 1]
  // CHECK-LABEL: tt.func public @broadcast_bias_load_is_remat
  tt.func public @broadcast_bias_load_is_remat(%acc: tensor<64x32xf32, #dpas>, %bias_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %c_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %m: i32, %n: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf16, #blocked>
    %cst_ld = arith.constant dense<32> : tensor<64x1xi32, #blocked>
    %rm = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %rn = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %m_splat = tt.splat %m : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %n_splat = tt.splat %n : i32 -> tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %mask_m = arith.cmpi slt, %rm, %m_splat : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %mask_m_2d = tt.expand_dims %mask_m {axis = 1 : i32} : tensor<64xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi1, #blocked>
    %mask_n = arith.cmpi slt, %rn, %n_splat : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %mask_n_2d = tt.expand_dims %mask_n {axis = 0 : i32} : tensor<32xi1, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi1, #blocked>
    %mask_mb = tt.broadcast %mask_m_2d : tensor<64x1xi1, #blocked> -> tensor<64x32xi1, #blocked>
    %mask_nb = tt.broadcast %mask_n_2d : tensor<1x32xi1, #blocked> -> tensor<64x32xi1, #blocked>
    %mask = arith.andi %mask_mb, %mask_nb : tensor<64x32xi1, #blocked>
    %rn_2d = tt.expand_dims %rn {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %bias_base = tt.splat %bias_ptr : !tt.ptr<f16> -> tensor<1x32x!tt.ptr<f16>, #blocked>
    %bias_row = tt.addptr %bias_base, %rn_2d : tensor<1x32x!tt.ptr<f16>, #blocked>, tensor<1x32xi32, #blocked>
    %bias_ptrs = tt.broadcast %bias_row : tensor<1x32x!tt.ptr<f16>, #blocked> -> tensor<64x32x!tt.ptr<f16>, #blocked>
    // CHECK: %[[BIAS:.*]] = tt.load {{.*}} {ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #[[$DPAS]]>
    // CHECK-NOT: ttg.convert_layout %[[BIAS]]
    // CHECK: arith.extf %[[BIAS]] : tensor<64x32xf16, #[[$DPAS]]> to tensor<64x32xf32, #[[$DPAS]]>
    %bias = tt.load %bias_ptrs, %mask, %cst {ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #blocked>
    %bias_dpas = ttg.convert_layout %bias : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #dpas>
    %bias_f32 = arith.extf %bias_dpas : tensor<64x32xf16, #dpas> to tensor<64x32xf32, #dpas>
    %sum = arith.addf %acc, %bias_f32 : tensor<64x32xf32, #dpas>
    %res = arith.truncf %sum : tensor<64x32xf32, #dpas> to tensor<64x32xf16, #dpas>
    %rm_2d = tt.expand_dims %rm {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %rm_off = arith.muli %rm_2d, %cst_ld : tensor<64x1xi32, #blocked>
    %c_base = tt.splat %c_ptr : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #blocked>
    %c_rows = tt.addptr %c_base, %rm_off : tensor<64x1x!tt.ptr<f16>, #blocked>, tensor<64x1xi32, #blocked>
    %c_rows_b = tt.broadcast %c_rows : tensor<64x1x!tt.ptr<f16>, #blocked> -> tensor<64x32x!tt.ptr<f16>, #blocked>
    %rn_b = tt.broadcast %rn_2d : tensor<1x32xi32, #blocked> -> tensor<64x32xi32, #blocked>
    %c_ptrs = tt.addptr %c_rows_b, %rn_b : tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<64x32xi32, #blocked>
    // CHECK: tt.store {{.*}} {ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #[[$DPAS]]>
    // CHECK-NOT: ttg.convert_layout
    // CHECK: tt.return
    %res_blocked = ttg.convert_layout %res : tensor<64x32xf16, #dpas> -> tensor<64x32xf16, #blocked>
    tt.store %c_ptrs, %res_blocked, %mask {ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

// COM: Control: the same masked load with a pointer that varies along both dimensions (a
// COM: full bias matrix) still has 2048 distinct addresses, so it stays an anchor in the
// COM: encoding the memory access wants and the convert to DPAS remains (#8112).

#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [1, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate} {
  // CHECK-DAG: #[[$BLOCKED:.+]] = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
  // CHECK-DAG: #[[$DPAS:.+]] = #ttig.dpas<{{.*}}warpsPerCTA = [4, 1]
  // CHECK-LABEL: tt.func public @full_bias_load_stays_anchored
  tt.func public @full_bias_load_stays_anchored(%acc: tensor<64x32xf32, #dpas>, %bias_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %m: i32) -> tensor<64x32xf32, #dpas> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf16, #blocked>
    %cst_ld = arith.constant dense<32> : tensor<64x1xi32, #blocked>
    %rm = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %rn = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %m_splat = tt.splat %m : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %mask_m = arith.cmpi slt, %rm, %m_splat : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %mask_m_2d = tt.expand_dims %mask_m {axis = 1 : i32} : tensor<64xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi1, #blocked>
    %mask = tt.broadcast %mask_m_2d : tensor<64x1xi1, #blocked> -> tensor<64x32xi1, #blocked>
    %rm_2d = tt.expand_dims %rm {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %rm_off = arith.muli %rm_2d, %cst_ld : tensor<64x1xi32, #blocked>
    %rn_2d = tt.expand_dims %rn {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %rm_b = tt.broadcast %rm_off : tensor<64x1xi32, #blocked> -> tensor<64x32xi32, #blocked>
    %rn_b = tt.broadcast %rn_2d : tensor<1x32xi32, #blocked> -> tensor<64x32xi32, #blocked>
    %off = arith.addi %rm_b, %rn_b : tensor<64x32xi32, #blocked>
    %base = tt.splat %bias_ptr : !tt.ptr<f16> -> tensor<64x32x!tt.ptr<f16>, #blocked>
    %ptrs = tt.addptr %base, %off : tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<64x32xi32, #blocked>
    // CHECK: %[[BIAS:.*]] = tt.load {{.*}} {ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #[[$BLOCKED]]>
    // CHECK: ttg.convert_layout %[[BIAS]] : tensor<64x32xf16, #[[$BLOCKED]]> -> tensor<64x32xf16, #[[$DPAS]]>
    %bias = tt.load %ptrs, %mask, %cst {ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #blocked>
    %bias_dpas = ttg.convert_layout %bias : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #dpas>
    %bias_f32 = arith.extf %bias_dpas : tensor<64x32xf16, #dpas> to tensor<64x32xf32, #dpas>
    %sum = arith.addf %acc, %bias_f32 : tensor<64x32xf32, #dpas>
    tt.return %sum : tensor<64x32xf32, #dpas>
  }
}

// -----

// COM: A broadcast bias row with more distinct addresses than threads (here 128 > 64) feeding
// COM: a DPAS value is not an anchor either: there is no thread threshold for such loads.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [1, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate} {
  // CHECK-DAG: #[[$DPAS:.+]] = #ttig.dpas<{{.*}}warpsPerCTA = [4, 1]
  // CHECK-LABEL: tt.func public @wide_broadcast_bias_load_is_remat
  tt.func public @wide_broadcast_bias_load_is_remat(%acc: tensor<64x128xf32, #dpas>, %bias_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %n: i32) -> tensor<64x128xf32, #dpas> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf16, #blocked>
    %rn = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %n_splat = tt.splat %n : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %mask_n = arith.cmpi slt, %rn, %n_splat : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %mask_n_2d = tt.expand_dims %mask_n {axis = 0 : i32} : tensor<128xi1, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi1, #blocked>
    %mask = tt.broadcast %mask_n_2d : tensor<1x128xi1, #blocked> -> tensor<64x128xi1, #blocked>
    %rn_2d = tt.expand_dims %rn {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %base = tt.splat %bias_ptr : !tt.ptr<f16> -> tensor<1x128x!tt.ptr<f16>, #blocked>
    %row = tt.addptr %base, %rn_2d : tensor<1x128x!tt.ptr<f16>, #blocked>, tensor<1x128xi32, #blocked>
    %ptrs = tt.broadcast %row : tensor<1x128x!tt.ptr<f16>, #blocked> -> tensor<64x128x!tt.ptr<f16>, #blocked>
    // CHECK: %[[BIAS:.*]] = tt.load {{.*}} : tensor<64x128x!tt.ptr<f16>, #[[$DPAS]]>
    // CHECK-NOT: ttg.convert_layout %[[BIAS]]
    %bias = tt.load %ptrs, %mask, %cst {ttig.block_io = "row_major"} : tensor<64x128x!tt.ptr<f16>, #blocked>
    %bias_dpas = ttg.convert_layout %bias : tensor<64x128xf16, #blocked> -> tensor<64x128xf16, #dpas>
    %bias_f32 = arith.extf %bias_dpas : tensor<64x128xf16, #dpas> to tensor<64x128xf32, #dpas>
    %sum = arith.addf %acc, %bias_f32 : tensor<64x128xf32, #dpas>
    tt.return %sum : tensor<64x128xf32, #dpas>
  }
}

// -----

// COM: An offset built from a chain of `x = x * x` steps reaches each step through two
// COM: operands; the uniformity walk caches per value, so each step is analysed once. This
// COM: case only checks the result (the chain keeps the row broadcast, so the load is still
// COM: rematerialized into the DPAS encoding); it does not detect an uncached walk, which the
// COM: depth limit keeps fast enough for lit.

#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [1, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate} {
  // CHECK-DAG: #[[$DPAS:.+]] = #ttig.dpas<{{.*}}warpsPerCTA = [4, 1]
  // CHECK-LABEL: tt.func public @broadcast_load_through_shared_chain
  tt.func public @broadcast_load_through_shared_chain(%acc: tensor<64x32xf32, #dpas>, %bias_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<64x32xf32, #dpas> {
    %rn = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %x0 = tt.expand_dims %rn {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %x1 = arith.muli %x0, %x0 : tensor<1x32xi32, #blocked>
    %x2 = arith.muli %x1, %x1 : tensor<1x32xi32, #blocked>
    %x3 = arith.muli %x2, %x2 : tensor<1x32xi32, #blocked>
    %x4 = arith.muli %x3, %x3 : tensor<1x32xi32, #blocked>
    %x5 = arith.muli %x4, %x4 : tensor<1x32xi32, #blocked>
    %x6 = arith.muli %x5, %x5 : tensor<1x32xi32, #blocked>
    %x7 = arith.muli %x6, %x6 : tensor<1x32xi32, #blocked>
    %x8 = arith.muli %x7, %x7 : tensor<1x32xi32, #blocked>
    %x9 = arith.muli %x8, %x8 : tensor<1x32xi32, #blocked>
    %x10 = arith.muli %x9, %x9 : tensor<1x32xi32, #blocked>
    %x11 = arith.muli %x10, %x10 : tensor<1x32xi32, #blocked>
    %x12 = arith.muli %x11, %x11 : tensor<1x32xi32, #blocked>
    %x13 = arith.muli %x12, %x12 : tensor<1x32xi32, #blocked>
    %x14 = arith.muli %x13, %x13 : tensor<1x32xi32, #blocked>
    %x15 = arith.muli %x14, %x14 : tensor<1x32xi32, #blocked>
    %base = tt.splat %bias_ptr : !tt.ptr<f16> -> tensor<1x32x!tt.ptr<f16>, #blocked>
    %row = tt.addptr %base, %x15 : tensor<1x32x!tt.ptr<f16>, #blocked>, tensor<1x32xi32, #blocked>
    %ptrs = tt.broadcast %row : tensor<1x32x!tt.ptr<f16>, #blocked> -> tensor<64x32x!tt.ptr<f16>, #blocked>
    // CHECK: %[[BIAS:.*]] = tt.load {{.*}} : tensor<64x32x!tt.ptr<f16>, #[[$DPAS]]>
    // CHECK-NOT: ttg.convert_layout %[[BIAS]]
    // CHECK: arith.extf %[[BIAS]] : tensor<64x32xf16, #[[$DPAS]]>
    %bias = tt.load %ptrs : tensor<64x32x!tt.ptr<f16>, #blocked>
    %bias_dpas = ttg.convert_layout %bias : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #dpas>
    %bias_f32 = arith.extf %bias_dpas : tensor<64x32xf16, #dpas> to tensor<64x32xf32, #dpas>
    %sum = arith.addf %acc, %bias_f32 : tensor<64x32xf32, #dpas>
    tt.return %sum : tensor<64x32xf32, #dpas>
  }
}

// -----

// COM: A fully uniform tile (a load of a splat pointer) keeps the logical element count:
// COM: it is not a broadcast row or column, so it stays an anchor even next to a DPAS value.

#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [1, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate} {
  // CHECK-DAG: #[[$BLOCKED:.+]] = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
  // CHECK-LABEL: tt.func public @splat_pointer_load_stays_anchored
  tt.func public @splat_pointer_load_stays_anchored(%acc: tensor<64x32xf32, #dpas>, %bias_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<64x32xf32, #dpas> {
    %ptrs = tt.splat %bias_ptr : !tt.ptr<f16> -> tensor<64x32x!tt.ptr<f16>, #blocked>
    // CHECK: %[[BIAS:.*]] = tt.load {{.*}} : tensor<64x32x!tt.ptr<f16>, #[[$BLOCKED]]>
    // CHECK: ttg.convert_layout %[[BIAS]] : tensor<64x32xf16, #[[$BLOCKED]]>
    %bias = tt.load %ptrs : tensor<64x32x!tt.ptr<f16>, #blocked>
    %bias_dpas = ttg.convert_layout %bias : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #dpas>
    %bias_f32 = arith.extf %bias_dpas : tensor<64x32xf16, #dpas> to tensor<64x32xf32, #dpas>
    %sum = arith.addf %acc, %bias_f32 : tensor<64x32xf32, #dpas>
    tt.return %sum : tensor<64x32xf32, #dpas>
  }
}

// -----

// COM: Stores keep the logical element count; only tt.load is costed by distinct addresses.
// COM: A store has no result, so that count does not make it an anchor: a store through a
// COM: broadcast pointer follows its value into the DPAS encoding, exactly as before.

#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [1, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate} {
  // CHECK-DAG: #[[$DPAS:.+]] = #ttig.dpas<{{.*}}warpsPerCTA = [4, 1]
  // CHECK-LABEL: tt.func public @broadcast_pointer_store_unchanged
  tt.func public @broadcast_pointer_store_unchanged(%acc: tensor<64x32xf32, #dpas>, %out_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %rn = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %rn_2d = tt.expand_dims %rn {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %base = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<1x32x!tt.ptr<f32>, #blocked>
    %row = tt.addptr %base, %rn_2d : tensor<1x32x!tt.ptr<f32>, #blocked>, tensor<1x32xi32, #blocked>
    %ptrs = tt.broadcast %row : tensor<1x32x!tt.ptr<f32>, #blocked> -> tensor<64x32x!tt.ptr<f32>, #blocked>
    // CHECK-NOT: ttg.convert_layout
    // CHECK: tt.store {{.*}} : tensor<64x32x!tt.ptr<f32>, #[[$DPAS]]>
    %val = ttg.convert_layout %acc : tensor<64x32xf32, #dpas> -> tensor<64x32xf32, #blocked>
    tt.store %ptrs, %val : tensor<64x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// COM: A column broadcast (one value per row, Mx1 -> MxN) feeding a DPAS value is
// COM: rematerialized like the row broadcast.

#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [1, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate} {
  // CHECK-DAG: #[[$DPAS:.+]] = #ttig.dpas<{{.*}}warpsPerCTA = [4, 1]
  // CHECK-LABEL: tt.func public @column_broadcast_load_is_remat
  tt.func public @column_broadcast_load_is_remat(%acc: tensor<32x64xf32, #dpas>, %scale_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<32x64xf32, #dpas> {
    %rm = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %rm_2d = tt.expand_dims %rm {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %base = tt.splat %scale_ptr : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>, #blocked>
    %col = tt.addptr %base, %rm_2d : tensor<32x1x!tt.ptr<f16>, #blocked>, tensor<32x1xi32, #blocked>
    %ptrs = tt.broadcast %col : tensor<32x1x!tt.ptr<f16>, #blocked> -> tensor<32x64x!tt.ptr<f16>, #blocked>
    // CHECK: %[[SCALE:.*]] = tt.load {{.*}} : tensor<32x64x!tt.ptr<f16>, #[[$DPAS]]>
    // CHECK-NOT: ttg.convert_layout %[[SCALE]]
    // CHECK: arith.extf %[[SCALE]] : tensor<32x64xf16, #[[$DPAS]]>
    %scale = tt.load %ptrs : tensor<32x64x!tt.ptr<f16>, #blocked>
    %scale_dpas = ttg.convert_layout %scale : tensor<32x64xf16, #blocked> -> tensor<32x64xf16, #dpas>
    %scale_f32 = arith.extf %scale_dpas : tensor<32x64xf16, #dpas> to tensor<32x64xf32, #dpas>
    %prod = arith.mulf %acc, %scale_f32 : tensor<32x64xf32, #dpas>
    tt.return %prod : tensor<32x64xf32, #dpas>
  }
}

// -----

// COM: A broadcast load that does not meet a DPAS value keeps the logical element count
// COM: (#8112), so a masked load whose consumer is a store in another blocked encoding
// COM: stays an anchor and the convert to the store's encoding remains. If every broadcast
// COM: load counted as feeding DPAS, this load would be rematerialized into the store's
// COM: encoding instead.

#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [16, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-DAG: #[[$LOAD_LAYOUT:.+]] = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
  // CHECK-DAG: #[[$STORE_LAYOUT:.+]] = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [16, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
  // CHECK-LABEL: tt.func public @broadcast_load_without_dpas_stays_anchored
  tt.func public @broadcast_load_without_dpas_stays_anchored(%bias_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %out_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %n: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf16, #blocked>
    %offs = arith.constant dense<3> : tensor<64x32xi32, #blocked1>
    %rn = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %n_splat = tt.splat %n : i32 -> tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %mask_n = arith.cmpi slt, %rn, %n_splat : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %mask_2d = tt.expand_dims %mask_n {axis = 0 : i32} : tensor<32xi1, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi1, #blocked>
    %mask = tt.broadcast %mask_2d : tensor<1x32xi1, #blocked> -> tensor<64x32xi1, #blocked>
    %rn_2d = tt.expand_dims %rn {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %base = tt.splat %bias_ptr : !tt.ptr<f16> -> tensor<1x32x!tt.ptr<f16>, #blocked>
    %row = tt.addptr %base, %rn_2d : tensor<1x32x!tt.ptr<f16>, #blocked>, tensor<1x32xi32, #blocked>
    %ptrs = tt.broadcast %row : tensor<1x32x!tt.ptr<f16>, #blocked> -> tensor<64x32x!tt.ptr<f16>, #blocked>
    // CHECK: %[[BIAS:.*]] = tt.load {{.*}} : tensor<64x32x!tt.ptr<f16>, #[[$LOAD_LAYOUT]]>
    // CHECK: ttg.convert_layout %[[BIAS]] : tensor<64x32xf16, #[[$LOAD_LAYOUT]]> -> tensor<64x32xf16, #[[$STORE_LAYOUT]]>
    %bias = tt.load %ptrs, %mask, %cst : tensor<64x32x!tt.ptr<f16>, #blocked>
    %bias_out = ttg.convert_layout %bias : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #blocked1>
    %out_base = tt.splat %out_ptr : !tt.ptr<f16> -> tensor<64x32x!tt.ptr<f16>, #blocked1>
    %out_ptrs = tt.addptr %out_base, %offs : tensor<64x32x!tt.ptr<f16>, #blocked1>, tensor<64x32xi32, #blocked1>
    tt.store %out_ptrs, %bias_out : tensor<64x32x!tt.ptr<f16>, #blocked1>
    tt.return
  }
}

// -----

// COM: At the old threshold: a broadcast bias row with exactly as many distinct addresses as
// COM: threads (64 = 4 warps x 16) feeding a DPAS value is rematerialized as well.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [1, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate} {
  // CHECK-DAG: #[[$DPAS:.+]] = #ttig.dpas<{{.*}}warpsPerCTA = [4, 1]
  // CHECK-LABEL: tt.func public @threshold_broadcast_bias_load_is_remat
  tt.func public @threshold_broadcast_bias_load_is_remat(%acc: tensor<64x64xf32, #dpas>, %bias_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<64x64xf32, #dpas> {
    %rn = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %rn_2d = tt.expand_dims %rn {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %base = tt.splat %bias_ptr : !tt.ptr<f16> -> tensor<1x64x!tt.ptr<f16>, #blocked>
    %row = tt.addptr %base, %rn_2d : tensor<1x64x!tt.ptr<f16>, #blocked>, tensor<1x64xi32, #blocked>
    %ptrs = tt.broadcast %row : tensor<1x64x!tt.ptr<f16>, #blocked> -> tensor<64x64x!tt.ptr<f16>, #blocked>
    // CHECK: %[[BIAS:.*]] = tt.load {{.*}} : tensor<64x64x!tt.ptr<f16>, #[[$DPAS]]>
    // CHECK-NOT: ttg.convert_layout %[[BIAS]]
    %bias = tt.load %ptrs : tensor<64x64x!tt.ptr<f16>, #blocked>
    %bias_dpas = ttg.convert_layout %bias : tensor<64x64xf16, #blocked> -> tensor<64x64xf16, #dpas>
    %bias_f32 = arith.extf %bias_dpas : tensor<64x64xf16, #dpas> to tensor<64x64xf32, #dpas>
    %sum = arith.addf %acc, %bias_f32 : tensor<64x64xf32, #dpas>
    tt.return %sum : tensor<64x64xf32, #dpas>
  }
}

// -----

// COM: As AccelerateMatmul emits a GEMM epilogue: the accumulator is converted to the blocked
// COM: encoding and the bias is added there. The bias pointer is built as Inductor's template
// COM: does, addptr(splat, broadcast(offsets)). The load is combined with a value converted from
// COM: DPAS, so it is not an anchor: it is rematerialized in the DPAS encoding together with the
// COM: add and the store, and no convert remains.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [1, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate} {
  // CHECK-DAG: #[[$DPAS:.+]] = #ttig.dpas<{{.*}}warpsPerCTA = [4, 1]
  // CHECK-LABEL: tt.func public @inductor_bias_epilogue_in_blocked_is_remat
  tt.func public @inductor_bias_epilogue_in_blocked_is_remat(%acc_dpas: tensor<64x128xf32, #dpas>, %bias_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %c_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %m: i32, %n: i32) {
    %cst_ld = arith.constant dense<128> : tensor<64x1xi32, #blocked>
    %acc = ttg.convert_layout %acc_dpas : tensor<64x128xf32, #dpas> -> tensor<64x128xf32, #blocked>
    %rm = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %rn = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %idx_m = tt.expand_dims %rm {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %idx_n = tt.expand_dims %rn {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %m_splat = tt.splat %m : i32 -> tensor<64x1xi32, #blocked>
    %n_splat = tt.splat %n : i32 -> tensor<1x128xi32, #blocked>
    %mask_m = arith.cmpi slt, %idx_m, %m_splat : tensor<64x1xi32, #blocked>
    %mask_n = arith.cmpi slt, %idx_n, %n_splat : tensor<1x128xi32, #blocked>
    %mask_mb = tt.broadcast %mask_m : tensor<64x1xi1, #blocked> -> tensor<64x128xi1, #blocked>
    %mask_nb = tt.broadcast %mask_n : tensor<1x128xi1, #blocked> -> tensor<64x128xi1, #blocked>
    %mask = arith.andi %mask_mb, %mask_nb : tensor<64x128xi1, #blocked>
    %offs = tt.broadcast %idx_n : tensor<1x128xi32, #blocked> -> tensor<64x128xi32, #blocked>
    %bias_base = tt.splat %bias_ptr : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #blocked>
    %bias_ptrs = tt.addptr %bias_base, %offs : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
    // CHECK: %[[BIAS:.*]] = tt.load {{.*}} : tensor<64x128x!tt.ptr<f16>, #[[$DPAS]]>
    // CHECK: arith.extf %[[BIAS]] : tensor<64x128xf16, #[[$DPAS]]>
    // CHECK: tt.store {{.*}} : tensor<64x128x!tt.ptr<f16>, #[[$DPAS]]>
    // CHECK-NOT: ttg.convert_layout
    // CHECK: tt.return
    %bias = tt.load %bias_ptrs, %mask {evictionPolicy = 3 : i32, ttig.block_io = "row_major"} : tensor<64x128x!tt.ptr<f16>, #blocked>
    %bias_f32 = arith.extf %bias : tensor<64x128xf16, #blocked> to tensor<64x128xf32, #blocked>
    %sum = arith.addf %bias_f32, %acc : tensor<64x128xf32, #blocked>
    %res = arith.truncf %sum : tensor<64x128xf32, #blocked> to tensor<64x128xf16, #blocked>
    %c_rows = arith.muli %idx_m, %cst_ld : tensor<64x1xi32, #blocked>
    %c_rows_b = tt.broadcast %c_rows : tensor<64x1xi32, #blocked> -> tensor<64x128xi32, #blocked>
    %rn_c = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %idx_n_c = tt.expand_dims %rn_c {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %c_cols_b = tt.broadcast %idx_n_c : tensor<1x128xi32, #blocked> -> tensor<64x128xi32, #blocked>
    %c_off = arith.addi %c_rows_b, %c_cols_b : tensor<64x128xi32, #blocked>
    %c_base = tt.splat %c_ptr : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #blocked>
    %c_ptrs = tt.addptr %c_base, %c_off : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
    tt.store %c_ptrs, %res, %mask : tensor<64x128x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

// COM: Only the DPAS-consumer rule decides here. An unmasked broadcast load without `ttig.block_io`
// COM: whose consumer is a store in another blocked encoding keeps the logical element count and
// COM: stays an anchor, so the convert to the store's encoding remains. Costed by its distinct
// COM: addresses it would be cheap, and the load would be rematerialized into the store's encoding.

#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [16, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-DAG: #[[$LOAD_LAYOUT:.+]] = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
  // CHECK-DAG: #[[$STORE_LAYOUT:.+]] = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [16, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
  // CHECK-LABEL: tt.func public @unmasked_broadcast_load_without_dpas_stays_anchored
  tt.func public @unmasked_broadcast_load_without_dpas_stays_anchored(%bias_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %out_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %offs = arith.constant dense<3> : tensor<64x64xi32, #blocked1>
    %rn = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %rn_2d = tt.expand_dims %rn {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %base = tt.splat %bias_ptr : !tt.ptr<f16> -> tensor<1x64x!tt.ptr<f16>, #blocked>
    %row = tt.addptr %base, %rn_2d : tensor<1x64x!tt.ptr<f16>, #blocked>, tensor<1x64xi32, #blocked>
    %ptrs = tt.broadcast %row : tensor<1x64x!tt.ptr<f16>, #blocked> -> tensor<64x64x!tt.ptr<f16>, #blocked>
    // CHECK: %[[BIAS:.*]] = tt.load {{.*}} : tensor<64x64x!tt.ptr<f16>, #[[$LOAD_LAYOUT]]>
    // CHECK: ttg.convert_layout %[[BIAS]] : tensor<64x64xf16, #[[$LOAD_LAYOUT]]> -> tensor<64x64xf16, #[[$STORE_LAYOUT]]>
    %bias = tt.load %ptrs : tensor<64x64x!tt.ptr<f16>, #blocked>
    %bias_out = ttg.convert_layout %bias : tensor<64x64xf16, #blocked> -> tensor<64x64xf16, #blocked1>
    %out_base = tt.splat %out_ptr : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>, #blocked1>
    %out_ptrs = tt.addptr %out_base, %offs : tensor<64x64x!tt.ptr<f16>, #blocked1>, tensor<64x64xi32, #blocked1>
    tt.store %out_ptrs, %bias_out : tensor<64x64x!tt.ptr<f16>, #blocked1>
    tt.return
  }
}
