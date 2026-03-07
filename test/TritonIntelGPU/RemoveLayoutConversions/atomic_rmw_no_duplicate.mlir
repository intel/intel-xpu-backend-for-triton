// RUN: triton-opt %s -tritonintelgpu-remove-layout-conversions | FileCheck %s

// COM: Verify that the remove-layout-conversions pass does not duplicate
//      tt.atomic_rmw operations. Duplicating side-effecting atomics causes
//      incorrect results (e.g., double-counting with atomic_add).

// CHECK-LABEL: @atomic_rmw_no_duplicate
// CHECK: tt.atomic_rmw
// CHECK-NOT: tt.atomic_rmw

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate} {
  tt.func public @atomic_rmw_no_duplicate(%dy_ptr: !tt.ptr<f32>, %x_ptr: !tt.ptr<f32>, %row_off: i32, %col_off: i32) {
    %cst = arith.constant dense<true> : tensor<32xi1, #blocked>
    %cst_0 = arith.constant dense<128> : tensor<32x1xi32, #blocked1>
    %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #blocked>
    %row_splat = tt.splat %row_off : i32 -> tensor<32xi32, #blocked>
    // %rows is shared between the 2D load path and the atomic pointer path.
    %rows = arith.addi %row_splat, %range : tensor<32xi32, #blocked>
    %col_splat = tt.splat %col_off : i32 -> tensor<32xi32, #blocked>
    %cols = arith.addi %col_splat, %range : tensor<32xi32, #blocked>

    // 2D load path: rows → slice → expand_dims → broadcast → load → reduce.
    %r_slice = ttg.convert_layout %rows : tensor<32xi32, #blocked> -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %r_exp = tt.expand_dims %r_slice {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<32x1xi32, #blocked2>
    %r_cvt = ttg.convert_layout %r_exp : tensor<32x1xi32, #blocked2> -> tensor<32x1xi32, #blocked1>
    %r_mul = arith.muli %r_cvt, %cst_0 : tensor<32x1xi32, #blocked1>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>, #blocked1>
    %x_addr = tt.addptr %x_splat, %r_mul : tensor<32x1x!tt.ptr<f32>, #blocked1>, tensor<32x1xi32, #blocked1>
    %c_slice = ttg.convert_layout %cols : tensor<32xi32, #blocked> -> tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %c_exp = tt.expand_dims %c_slice {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x32xi32, #blocked3>
    %c_cvt = ttg.convert_layout %c_exp : tensor<1x32xi32, #blocked3> -> tensor<1x32xi32, #blocked4>
    %x_bcast = tt.broadcast %x_addr : tensor<32x1x!tt.ptr<f32>, #blocked1> -> tensor<32x32x!tt.ptr<f32>, #blocked1>
    %x_cvt = ttg.convert_layout %x_bcast : tensor<32x32x!tt.ptr<f32>, #blocked1> -> tensor<32x32x!tt.ptr<f32>, #blocked4>
    %c_bcast = tt.broadcast %c_cvt : tensor<1x32xi32, #blocked4> -> tensor<32x32xi32, #blocked4>
    %ld_addr = tt.addptr %x_cvt, %c_bcast : tensor<32x32x!tt.ptr<f32>, #blocked4>, tensor<32x32xi32, #blocked4>
    %ld_cvt = ttg.convert_layout %ld_addr : tensor<32x32x!tt.ptr<f32>, #blocked4> -> tensor<32x32x!tt.ptr<f32>, #blocked5>
    %loaded = tt.load %ld_cvt : tensor<32x32x!tt.ptr<f32>, #blocked5>
    %ld_back = ttg.convert_layout %loaded : tensor<32x32xf32, #blocked5> -> tensor<32x32xf32, #blocked4>
    %reduced = "tt.reduce"(%ld_back) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) : (tensor<32x32xf32, #blocked4>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %val = ttg.convert_layout %reduced : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked4}>> -> tensor<32xf32, #blocked>

    // Atomic path: uses %rows for pointer computation.
    %dy_splat = tt.splat %dy_ptr : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #blocked>
    %dy_addr = tt.addptr %dy_splat, %rows : tensor<32x!tt.ptr<f32>, #blocked>, tensor<32xi32, #blocked>

    // Identity convert_layouts inserted by coalesce.
    %ptr_id = ttg.convert_layout %dy_addr : tensor<32x!tt.ptr<f32>, #blocked> -> tensor<32x!tt.ptr<f32>, #blocked>
    %val_id = ttg.convert_layout %val : tensor<32xf32, #blocked> -> tensor<32xf32, #blocked>
    %msk_id = ttg.convert_layout %cst : tensor<32xi1, #blocked> -> tensor<32xi1, #blocked>

    // This atomic must NOT be duplicated by the pass.
    %result = tt.atomic_rmw fadd, relaxed, gpu, %ptr_id, %val_id, %msk_id : (tensor<32x!tt.ptr<f32>, #blocked>, tensor<32xf32, #blocked>, tensor<32xi1, #blocked>) -> tensor<32xf32, #blocked>
    tt.return
  }
}
