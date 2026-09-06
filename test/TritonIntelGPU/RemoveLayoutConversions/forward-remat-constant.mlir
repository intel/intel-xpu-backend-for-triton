// RUN: triton-opt %s --pass-pipeline='builtin.module(convert-triton-to-tritongpu{enable-source-remat=false num-ctas=1 num-warps=1 target=xpu threads-per-warp=16}, tritonintelgpu-remove-layout-conversions{max-backward-remat-iterations=1}, tritonintelgpu-accelerate-matmul, tritonintelgpu-remove-layout-conversions{max-backward-remat-iterations=1})' -verify-each | FileCheck %s

// COM: Regression test: forward layout propagation used to relabel an
// COM: arith.constant to a DPAS encoding in place without rebuilding the
// COM: value attribute, failing the arith verifier (value/result type
// COM: mismatch). The constant below feeds both a dot chain and a masked
// COM: load's `other` operand, which is what drives the conflicting layout
// COM: demands.

// CHECK: #[[$MMA:.+]] = #ttig.dpas
// CHECK-LABEL: tt.func public @constant_relabel_forward
// CHECK: arith.constant dense<0.000000e+00> : tensor<16x16xf32, #[[$MMA]]>
// CHECK: tt.dot {{.*}} -> tensor<16x16xf32, #[[$MMA]]>
// CHECK: tt.return

module attributes {"ttg.threads-per-warp" = 16 : i32, ttig.2d_block_io_base_alignment = 64 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_bfloat16_arithmetic, ttig.support_bfloat16_conversion, ttig.support_predicated_io, ttig.support_rounded_divide_sqrt, ttig.support_subgroup_matrix_multiply_accumulate, ttig.target_arch = "spir64"} {
  tt.func public @constant_relabel_forward(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
    %cst_0 = arith.constant dense<96> : tensor<16x1xi64>
    %c32_i64 = arith.constant 32 : i64
    %cst_1 = arith.constant dense<16> : tensor<16xi32>
    %c3_i64 = arith.constant 3 : i64
    %0 = tt.get_program_id x : i32
    %1 = arith.extsi %0 : i32 to i64
    %2 = tt.get_program_id y : i32
    %3 = arith.extsi %2 : i32 to i64
    %4 = arith.divui %3, %c3_i64 : i64
    %5 = arith.remui %3, %c3_i64 : i64
    %6 = arith.extsi %arg2 : i32 to i64
    %7 = arith.muli %4, %6 : i64
    %8 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %9 = arith.addi %8, %cst_1 : tensor<16xi32>
    %10 = tt.expand_dims %8 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %11 = tt.expand_dims %8 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %12 = tt.broadcast %10 : tensor<16x1xi32> -> tensor<16x16xi32>
    %13 = tt.broadcast %11 : tensor<1x16xi32> -> tensor<16x16xi32>
    %14 = arith.cmpi sgt, %12, %13 : tensor<16x16xi32>
    %15 = arith.muli %1, %c32_i64 : i64
    %16 = arith.extsi %8 : tensor<16xi32> to tensor<16xi64>
    %17 = tt.splat %15 : i64 -> tensor<16xi64>
    %18 = arith.addi %17, %16 : tensor<16xi64>
    %19 = tt.splat %6 : i64 -> tensor<16xi64>
    %20 = arith.cmpi slt, %18, %19 : tensor<16xi64>
    %21 = arith.muli %7, %c3_i64 : i64
    %22 = arith.addi %21, %5 : i64
    %23 = arith.muli %22, %c32_i64 : i64
    %24 = tt.addptr %arg0, %23 : !tt.ptr<f32>, i64
    %25 = tt.expand_dims %18 {axis = 1 : i32} : tensor<16xi64> -> tensor<16x1xi64>
    %26 = arith.muli %25, %cst_0 : tensor<16x1xi64>
    %27 = tt.splat %24 : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>>
    %28 = tt.addptr %27, %26 : tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi64>
    %29 = tt.addptr %arg1, %23 : !tt.ptr<f32>, i64
    %30 = tt.splat %29 : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>>
    %31 = tt.addptr %30, %26 : tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi64>
    %32 = tt.expand_dims %20 {axis = 1 : i32} : tensor<16xi1> -> tensor<16x1xi1>
    %33 = tt.broadcast %28 : tensor<16x1x!tt.ptr<f32>> -> tensor<16x16x!tt.ptr<f32>>
    %34 = tt.addptr %33, %13 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    %35 = tt.broadcast %32 : tensor<16x1xi1> -> tensor<16x16xi1>
    %36 = tt.load %34, %35, %cst : tensor<16x16x!tt.ptr<f32>>
    %37 = arith.select %14, %36, %cst : tensor<16x16xi1>, tensor<16x16xf32>
    %38 = arith.truncf %37 : tensor<16x16xf32> to tensor<16x16xf16>
    %39 = arith.cmpi eq, %12, %13 : tensor<16x16xi32>
    %40 = arith.uitofp %39 : tensor<16x16xi1> to tensor<16x16xf16>
    %41 = tt.dot %38, %38, %cst : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
    %42 = arith.truncf %41 : tensor<16x16xf32> to tensor<16x16xf16>
    %43 = tt.dot %42, %42, %cst : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
    %44 = arith.truncf %43 : tensor<16x16xf32> to tensor<16x16xf16>
    %45 = tt.dot %44, %44, %cst : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
    %46 = arith.truncf %45 : tensor<16x16xf32> to tensor<16x16xf16>
    %47 = arith.addf %40, %38 : tensor<16x16xf16>
    %48 = arith.addf %40, %42 : tensor<16x16xf16>
    %49 = tt.dot %47, %48, %cst : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
    %50 = arith.truncf %49 : tensor<16x16xf32> to tensor<16x16xf16>
    %51 = arith.addf %40, %44 : tensor<16x16xf16>
    %52 = tt.dot %50, %51, %cst : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
    %53 = arith.truncf %52 : tensor<16x16xf32> to tensor<16x16xf16>
    %54 = arith.addf %40, %46 : tensor<16x16xf16>
    %55 = tt.dot %53, %54, %cst : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
    %56 = tt.broadcast %31 : tensor<16x1x!tt.ptr<f32>> -> tensor<16x16x!tt.ptr<f32>>
    %57 = tt.addptr %56, %13 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    tt.store %57, %55, %35 : tensor<16x16x!tt.ptr<f32>>
    %67 = tt.expand_dims %9 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %69 = tt.broadcast %67 : tensor<1x16xi32> -> tensor<16x16xi32>
    %101 = tt.addptr %56, %69 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    tt.store %101, %cst, %35 : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}
