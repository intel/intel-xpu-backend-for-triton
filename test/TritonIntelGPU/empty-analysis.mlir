// RUN: triton-opt %s -tritonintelgpu-empty-analysis | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.2d_block_io_base_alignment = 64 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_bfloat16_arithmetic, ttig.support_bfloat16_conversion, ttig.support_predicated_io, ttig.support_rounded_divide_sqrt, ttig.support_subgroup_matrix_multiply_accumulate, ttig.target_arch = "spir64"} {

  // CHECK-LABEL: tt.func public @empty_analysis_smoke
  tt.func public @empty_analysis_smoke(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %3 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<1024xi32, #blocked>
    %5 = tt.splat %arg3 : i32 -> tensor<1024xi32, #blocked>
    %6 = arith.cmpi slt, %4, %5 : tensor<1024xi32, #blocked>
    %7 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked>
    %8 = tt.addptr %7, %4 : tensor<1024x!tt.ptr<f16>, #blocked>, tensor<1024xi32, #blocked>
    %9 = tt.load %8, %6 cacheModifier = cg : tensor<1024x!tt.ptr<f16>, #blocked>
    %10 = tt.extern_elementwise %9 {libname = "", libpath = "", pure = true, symbol = "__imf_log2f16"} : (tensor<1024xf16, #blocked>) -> tensor<1024xf16, #blocked>
    %11 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked>
    %12 = tt.addptr %11, %4 : tensor<1024x!tt.ptr<f16>, #blocked>, tensor<1024xi32, #blocked>
    tt.store %12, %10, %6 : tensor<1024x!tt.ptr<f16>, #blocked>
    tt.return
  }
}
