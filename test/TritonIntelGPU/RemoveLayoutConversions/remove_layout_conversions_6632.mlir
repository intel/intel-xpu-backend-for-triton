// RUN: triton-opt %s -split-input-file -tritonintelgpu-remove-layout-conversions 2>&1 | FileCheck %s --enable-var-scope

// COM: Regression test for https://github.com/intel/intel-xpu-backend-for-triton/issues/6632
// COM:
// COM: When hoistConvertOnTopOfExtOrBroadcast rewrites a backward slice of a
// COM: convert_layout, it reuses cached rematerialisation values from earlier
// COM: iterations.  If the cached value (e.g. a convert_layout registered for a
// COM: previous convert op) is defined *after* the ops being cloned, the clone
// COM: would reference an operand that does not dominate it, violating MLIR
// COM: dominance requirements.
// COM:
// COM: The fix verifies that a cached remat value properly dominates the
// COM: defining op of the value it replaces before reusing it.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>

// CHECK-LABEL: @remat_dominance_cross_entropy
// COM: The key check is that the pass does not crash with a dominance error.
// COM: Verify that sitofp and divf appear with a consistent layout.
// CHECK: arith.sitofp
// CHECK: arith.divf
// CHECK: tt.store
// CHECK: tt.store
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_bfloat16_arithmetic, ttig.support_bfloat16_conversion, ttig.support_predicated_io, ttig.support_subgroup_matrix_multiply_accumulate, ttig.target_arch = "spir64"} {
  tt.func public @remat_dominance_cross_entropy(
      %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %arg1: !tt.ptr<i64> {tt.divisibility = 16 : i32},
      %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst_5 = arith.constant dense<5> : tensor<1x512xi64, #blocked>
    %cst_0 = arith.constant dense<0> : tensor<1x512xi64, #blocked>
    %0 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<512xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x512xi32, #blocked>
    %2 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<1x512x!tt.ptr<i64>, #blocked>
    %3 = tt.addptr %2, %1 : tensor<1x512x!tt.ptr<i64>, #blocked>, tensor<1x512xi32, #blocked>
    %4 = tt.load %3 {ttig.block_io = "row_major"} : tensor<1x512x!tt.ptr<i64>, #blocked>
    %11 = arith.cmpi ne, %4, %cst_5 : tensor<1x512xi64, #blocked>
    %12 = arith.extui %11 : tensor<1x512xi1, #blocked> to tensor<1x512xi64, #blocked>
    // i64 reduce + expand_dims → feeds sitofp
    %13 = "tt.reduce"(%12) <{axis = 1 : i32}> ({
    ^bb0(%arg7: i64, %arg8: i64):
      %r = arith.addi %arg7, %arg8 : i64
      tt.reduce.return %r : i64
    }) : (tensor<1x512xi64, #blocked>) -> tensor<1xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
    %14 = tt.expand_dims %13 {axis = 1 : i32} : tensor<1xi64, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<1x1xi64, #blocked>

    // f32 reduce + expand_dims → feeds divf
    %sel = arith.select %11, %4, %cst_0 : tensor<1x512xi1, #blocked>, tensor<1x512xi64, #blocked>
    %sitofp_sel = arith.sitofp %sel : tensor<1x512xi64, #blocked> to tensor<1x512xf32, #blocked>
    %33 = "tt.reduce"(%sitofp_sel) <{axis = 1 : i32}> ({
    ^bb0(%arg7: f32, %arg8: f32):
      %r = arith.addf %arg7, %arg8 : f32
      tt.reduce.return %r : f32
    }) : (tensor<1x512xf32, #blocked>) -> tensor<1xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %34 = tt.expand_dims %33 {axis = 1 : i32} : tensor<1xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<1x1xf32, #blocked>

    // sitofp of the i64 reduce result → used by divf AND by store (via convert)
    %35 = arith.sitofp %14 : tensor<1x1xi64, #blocked> to tensor<1x1xf32, #blocked>
    %36 = arith.divf %34, %35 : tensor<1x1xf32, #blocked>

    // First convert+store of sitofp result: this convert gets cached as a
    // remat value for (%35, #blocked1).
    %37 = tt.splat %arg5 : !tt.ptr<f32> -> tensor<1x1x!tt.ptr<f32>, #blocked1>
    %38 = ttg.convert_layout %35 : tensor<1x1xf32, #blocked> -> tensor<1x1xf32, #blocked1>
    tt.store %37, %38 {ttig.block_io = "column_major"} : tensor<1x1x!tt.ptr<f32>, #blocked1>

    ttg.barrier all

    // Second convert+store of divf result: the backward slice of this convert
    // would try to reuse %38 for %35, but %38 is defined *after* the divf.
    %39 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x1x!tt.ptr<f32>, #blocked1>
    %40 = ttg.convert_layout %36 : tensor<1x1xf32, #blocked> -> tensor<1x1xf32, #blocked1>
    tt.store %39, %40 {ttig.block_io = "column_major"} : tensor<1x1x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}
