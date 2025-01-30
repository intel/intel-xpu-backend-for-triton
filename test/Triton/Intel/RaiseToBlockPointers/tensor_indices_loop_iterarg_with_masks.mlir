// RUN: triton-opt %s -triton-raise-block-pointer -canonicalize | FileCheck %s

module {
  tt.func public @addptr_with_masks(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) attributes {noinline = false} {
    %cst = arith.constant dense<-1.100000e+01> : tensor<4xf32>
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<4> : tensor<4xi32>
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %1 = tt.splat %arg2 : i32 -> tensor<4xi32>
    %2 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %3 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %4:2 = scf.for %arg3 = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%arg4 = %0, %arg5 = %0) -> (tensor<4xi32>, tensor<4xi32>)  : i32 {
      %5 = arith.cmpi slt, %arg4, %1 : tensor<4xi32>
      %6 = tt.addptr %2, %arg4 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      // TODO: replace with the following line when masked loads are supported.
      // %7 = tt.load %6, %5, %cst : tensor<4x!tt.ptr<f32>>
      %7 = tt.load %6 : tensor<4x!tt.ptr<f32>>
      %8 = tt.addptr %3, %arg5 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      tt.store %8, %7 : tensor<4x!tt.ptr<f32>>
      %9 = arith.addi %arg4, %cst_0 : tensor<4xi32>
      %10 = arith.addi %arg5, %cst_0 : tensor<4xi32>
      scf.yield %9, %10 : tensor<4xi32>, tensor<4xi32>
    }
    tt.return
  }
}

// CHECK:         tt.func public @addptr_with_masks([[PARAM_0_:%.+]]: !tt.ptr<f32>, [[PARAM_1_:%.+]]: !tt.ptr<f32>, [[PARAM_2_:%.+]]: i32) attributes {noinline = false} {
// CHECK-DAG:       [[CST_1_i64:%.+]] = arith.constant 1 : i64
// CHECK-DAG:       [[CST_0_i64:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[CST_1_i32:%.+]] = arith.constant 1 : i32
// CHECK-DAG:       [[CST_4_i32:%.+]] = arith.constant 4 : i32
// CHECK-DAG:       [[CST_0_i32:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<4> : tensor<4xi32>
// CHECK-DAG:       [[VAR_0_:%.+]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK:           [[VAR_1_:%.+]]:2 = scf.for [[VAR_arg3_:%.+]] = {{.*}} iter_args([[VAR_arg4_:%.+]] = [[VAR_0_]], [[VAR_arg5_:%.+]] = [[VAR_0_]]) -> (tensor<4xi32>, tensor<4xi32>)  : i32 {
// CHECK:             [[VAR_2_:%.+]] = tt.make_tensor_ptr [[PARAM_0_]], {{\[}}[[CST_0_i64]]], {{\[}}[[CST_1_i64]]], {{\[}}[[CST_0_i32]]] {{.*}} : <tensor<4xf32>>
// CHECK-DAG:         [[VAR_3_:%.+]] = tt.load [[VAR_2_]] : !tt.ptr<tensor<4xf32>>
// CHECK-DAG:         [[VAR_4_:%.+]] = tt.make_tensor_ptr [[PARAM_1_]], {{\[}}[[CST_0_i64]]], {{\[}}[[CST_1_i64]]], {{\[}}[[CST_0_i32]]] {{.*}} : <tensor<4xf32>>
// CHECK:             tt.store [[VAR_4_]], [[VAR_3_]] : !tt.ptr<tensor<4xf32>>
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.addi [[VAR_arg4_]], [[VAR_cst_]] : tensor<4xi32>
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.addi [[VAR_arg5_]], [[VAR_cst_]] : tensor<4xi32>
// CHECK:             scf.yield [[VAR_5_]], [[VAR_6_]] : tensor<4xi32>, tensor<4xi32>
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
