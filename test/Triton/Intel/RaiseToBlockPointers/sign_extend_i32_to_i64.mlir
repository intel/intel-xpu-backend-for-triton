// RUN: triton-opt.exe %s -triton-raise-block-pointer -canonicalize | FileCheck %s

// IR from python/examples/sign_extend.py
module {
  tt.func public @sign_extend(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) attributes {noinline = false} {
    %cst = arith.constant dense<1.100000e+01> : tensor<4xf32>
    %0 = tt.load %arg0 : !tt.ptr<i32>
    %1 = arith.extsi %0 : i32 to i64
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = arith.extsi %2 : tensor<4xi32> to tensor<4xi64>
    %4 = tt.splat %1 : i64 -> tensor<4xi64>
    %5 = arith.addi %4, %3 : tensor<4xi64>
    %6 = arith.extsi %arg3 : i32 to i64
    %7 = tt.splat %6 : i64 -> tensor<4xi64>
    %8 = arith.cmpi slt, %5, %7 : tensor<4xi64>
    %9 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %10 = tt.addptr %9, %5 : tensor<4x!tt.ptr<f32>>, tensor<4xi64>
    %11 = tt.load %10 : tensor<4x!tt.ptr<f32>>
    // TODO: uncomment once masked loads are supported
    // %11 = tt.load %10, %8, %cst : tensor<4x!tt.ptr<f32>>
    %12 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %13 = tt.addptr %12, %2 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %13, %11 : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK:         tt.func public @sign_extend([[PARAM_0_:%.+]]: !tt.ptr<i32>, [[PARAM_1_:%.+]]: !tt.ptr<f32>, [[PARAM_2_:%.+]]: !tt.ptr<f32>, [[PARAM_3_:%.+]]: i32) attributes {noinline = false} {
// CHECK-DAG:       [[CST_0_i64:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[CST_1_i64:%.+]] = arith.constant 1 : i64
// CHECK-DAG:       [[CST_0_i32:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[VAR_0_:%.+]] = tt.load [[PARAM_0_]] : !tt.ptr<i32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tt.make_tensor_ptr [[PARAM_1_]], {{\[}}[[CST_0_i64]]], {{\[}}[[CST_1_i64]]], {{\[}}[[VAR_0_]]] {{.*}} : <tensor<4xf32>>
// CHECK-DAG:       [[VAR_2_:%.+]] = tt.load [[VAR_1_]] : !tt.ptr<tensor<4xf32>>
// CHECK:           [[VAR_3_:%.+]] = tt.make_tensor_ptr [[PARAM_2_]], {{\[}}[[CST_0_i64]]], {{\[}}[[CST_1_i64]]], {{\[}}[[CST_0_i32]]] {{.*}} : <tensor<4xf32>>
// CHECK:           tt.store [[VAR_3_]], [[VAR_2_]] : !tt.ptr<tensor<4xf32>>
// CHECK:           tt.return
// CHECK:         }
