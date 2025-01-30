// RUN: triton-opt %s -triton-raise-block-pointer -canonicalize | FileCheck %s

module {
  tt.func public @test_1(%arg0: !tt.ptr<f32>) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<4> : tensor<4xi32>
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %2:2 = scf.for %arg1 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg2 = %0, %arg3 = %0) -> (tensor<4xi32>, tensor<4xi32>)  : i32 {
      %3 = tt.addptr %1, %arg2 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      %4 = arith.sitofp %arg3 : tensor<4xi32> to tensor<4xf32>
      tt.store %3, %4 : tensor<4x!tt.ptr<f32>>
      %5 = arith.addi %arg2, %cst : tensor<4xi32>
      %6 = arith.addi %arg3, %cst : tensor<4xi32>
      scf.yield %5, %6 : tensor<4xi32>, tensor<4xi32>
    }
    tt.return
  }
}

// CHECK:         tt.func public @test_1([[PARAM_0_:.+]]: !tt.ptr<f32>) attributes {noinline = false} {
// CHECK-DAG:       [[CST_1_i64:%.+]] = arith.constant 1 : i64
// CHECK-DAG:       [[CST_0_i64:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[CST_1_i32:%.+]] = arith.constant 1 : i32
// CHECK-DAG:       [[CST_2_i32:%.+]] = arith.constant 2 : i32
// CHECK-DAG:       [[CST_0_i32:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<4> : tensor<4xi32>
// CHECK-DAG:       [[VAR_0_:%.+]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK:           [[VAR_1_:%.+]]:2 = scf.for [[VAR_arg1_:%.+]] = {{.*}} iter_args([[VAR_arg2_:%.+]] = [[VAR_0_]], [[VAR_arg3_:%.+]] = [[VAR_0_]]) -> (tensor<4xi32>, tensor<4xi32>)  : i32 {
// CHECK-DAG:         [[VAR_2_:%.+]] = tt.make_tensor_ptr [[PARAM_0_]], {{\[}}[[CST_0_i64]]], {{\[}}[[CST_1_i64]]], {{\[}}[[CST_0_i32]]] {{.*}} : <tensor<4xf32>>
// CHECK-DAG:         [[VAR_3_:%.+]] = arith.sitofp [[VAR_arg3_]] : tensor<4xi32> to tensor<4xf32>
// CHECK:             tt.store [[VAR_2_]], [[VAR_3_]] : !tt.ptr<tensor<4xf32>>
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.addi [[VAR_arg2_]], [[VAR_cst_]] : tensor<4xi32>
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.addi [[VAR_arg3_]], [[VAR_cst_]] : tensor<4xi32>
// CHECK:             scf.yield [[VAR_4_]], [[VAR_5_]] : tensor<4xi32>, tensor<4xi32>
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
