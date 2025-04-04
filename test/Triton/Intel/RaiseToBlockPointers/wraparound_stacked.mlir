// RUN: triton-opt %s -triton-raise-block-pointer -canonicalize | FileCheck %s
// XFAIL: *
// Note: Cannot generate tt.advance because block ptrs are created with unknown strides

module {
  tt.func public @wrap_stacked_masked_loop_01234567(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
    %cst = arith.constant dense<-9.900000e+01> : tensor<4x4xf32>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %cst_0 = arith.constant dense<3> : tensor<1x4xi32>
    %cst_1 = arith.constant dense<3> : tensor<4xi32>
    %cst_2 = arith.constant dense<2> : tensor<4xi32>
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %1 = arith.addi %0, %cst_2 : tensor<4xi32>
    %2 = tt.splat %arg2 : i32 -> tensor<4xi32>
    %3 = arith.remsi %1, %2 : tensor<4xi32>
    %4 = arith.addi %0, %cst_1 : tensor<4xi32>
    %5 = tt.expand_dims %3 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %6 = tt.splat %arg4 : i32 -> tensor<4x1xi32>
    %7 = arith.muli %5, %6 : tensor<4x1xi32>
    %8 = tt.expand_dims %4 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %9 = tt.splat %arg5 : i32 -> tensor<1x4xi32>
    %10 = arith.muli %8, %9 : tensor<1x4xi32>
    %11 = tt.broadcast %7 : tensor<4x1xi32> -> tensor<4x4xi32>
    %12 = tt.broadcast %10 : tensor<1x4xi32> -> tensor<4x4xi32>
    %13 = arith.addi %11, %12 : tensor<4x4xi32>
    %14 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
    %15 = tt.addptr %14, %13 : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    %16 = tt.expand_dims %0 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %17 = tt.splat %arg6 : i32 -> tensor<4x1xi32>
    %18 = arith.muli %17, %16 : tensor<4x1xi32>
    %19 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x1x!tt.ptr<f32>>
    %20 = tt.addptr %19, %18 : tensor<4x1x!tt.ptr<f32>>, tensor<4x1xi32>
    %21 = tt.expand_dims %0 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %22 = tt.splat %arg7 : i32 -> tensor<1x4xi32>
    %23 = arith.muli %22, %21 : tensor<1x4xi32>
    %24 = tt.broadcast %20 : tensor<4x1x!tt.ptr<f32>> -> tensor<4x4x!tt.ptr<f32>>
    %25 = tt.broadcast %23 : tensor<1x4xi32> -> tensor<4x4xi32>
    %26 = tt.addptr %24, %25 : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    %27 = arith.cmpi slt, %21, %cst_0 : tensor<1x4xi32>
    %28 = tt.broadcast %27 : tensor<1x4xi1> -> tensor<4x4xi1>
    %29 = arith.muli %arg5, %c4_i32 : i32
    %30 = tt.splat %29 : i32 -> tensor<4x4xi32>
    %31:2 = scf.for %arg8 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg9 = %15, %arg10 = %26) -> (tensor<4x4x!tt.ptr<f32>>, tensor<4x4x!tt.ptr<f32>>)  : i32 {
      %32 = tt.load %arg9, %28, %cst : tensor<4x4x!tt.ptr<f32>>
      tt.store %arg10, %32 : tensor<4x4x!tt.ptr<f32>>
      %33 = tt.addptr %arg9, %30 : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
      %34 = tt.addptr %arg10, %30 : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
      scf.yield %33, %34 : tensor<4x4x!tt.ptr<f32>>, tensor<4x4x!tt.ptr<f32>>
    }
    tt.return
  }
}

// CHECK:         tt.func public @wrap_stacked_masked_loop_01234567([[PARAM_0_:%.+]]: !tt.ptr<f32>, [[PARAM_1_:%.+]]: !tt.ptr<f32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32) {
// CHECK-DAG:       [[CST_:%.+]] = arith.constant dense<-9.900000e+01> : tensor<4x4xf32>
// CHECK-DAG:       [[CST_1_i32:%.+]] = arith.constant 1 : i32
// CHECK-DAG:       [[CST_0_i32:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_0_i64:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[CST_2_i32:%.+]] = arith.constant 2 : i32
// CHECK-DAG:       [[CST_4_i32:%.+]] = arith.constant 4 : i32
// CHECK-DAG:       [[CST_3_i32:%.+]] = arith.constant 3 : i32
// CHECK:           [[VAR_0_:%.+]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK:           [[VAR_2_:%.+]] = arith.index_cast [[PARAM_4_]] : i32 to index
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.muli [[PARAM_4_]], {{.*}} : i32
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.index_cast [[VAR_2_]] : index to i64
// CHECK-DAG:       [[VAR_5_:%.+]] = arith.extsi [[PARAM_4_]] : i32 to i64
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.index_cast [[VAR_1_]] : index to i64
// CHECK-DAG:       [[VAR_7_:%.+]] = arith.muli [[VAR_6_]], [[VAR_4_]] : i64
// CHECK-DAG:       [[VAR_8_:%.+]] = arith.muli [[PARAM_5_]], [[CST_3_i32]] : i32
// CHECK-DAG:       [[VAR_9_:%.+]] = arith.extsi [[PARAM_5_]] : i32 to i64
// CHECK:           [[VAR_10_:%.+]] = arith.divui [[VAR_3_]], [[PARAM_4_]] : i32
// CHECK:           [[VAR_11_:%.+]] = arith.divui [[VAR_8_]], [[PARAM_5_]] : i32
// CHECK:           [[VAR_12_:%.+]] = tt.make_tensor_ptr [[PARAM_0_]], {{\[}}[[VAR_7_]], [[CST_0_i64]]], {{\[}}[[VAR_5_]], [[VAR_9_]]], {{\[}}[[VAR_10_]], [[VAR_11_]]] {{.*}} : <tensor<4x4xf32>>
// CHECK-DAG:       [[VAR_13_:%.+]] = arith.extsi [[PARAM_6_]] : i32 to i64
// CHECK-DAG:       [[VAR_14_:%.+]] = tt.expand_dims [[VAR_0_]] {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
// CHECK-DAG:       [[VAR_15_:%.+]] = arith.extsi [[PARAM_7_]] : i32 to i64
// CHECK:           [[VAR_16_:%.+]] = tt.make_tensor_ptr [[PARAM_1_]], {{\[}}[[CST_0_i64]], [[CST_0_i64]]], {{\[}}[[VAR_13_]], [[VAR_15_]]], {{\[}}[[CST_0_i32]], [[CST_0_i32]]] {{.*}} : <tensor<4x4xf32>>
// CHECK:           [[VAR_17_:%.+]] = arith.cmpi slt, [[VAR_14_]], {{.*}} : tensor<1x4xi32>
// CHECK:           [[VAR_18_:%.+]] = tt.broadcast [[VAR_17_]] : tensor<1x4xi1> -> tensor<4x4xi1>
// CHECK:           [[VAR_19_:%.+]] = arith.muli [[PARAM_5_]], [[CST_4_i32]] : i32
// CHECK:           [[VAR_20_:%.+]]:2 = scf.for [[VAR_arg8_:%.+]] = {{.*}} iter_args([[VAR_arg9_:%.+]] = [[VAR_12_]], [[VAR_arg10_:%.+]] = [[VAR_16_]]) -> (!tt.ptr<tensor<4x4xf32>>, !tt.ptr<tensor<4x4xf32>>) : i32 {
// CHECK:             [[VAR_21_:%.+]] = tt.load [[VAR_arg9_]], [[VAR_18_]], [[CST_]] : !tt.ptr<tensor<4x4xf32>>
// CHECK:             tt.store [[VAR_arg10_]], [[VAR_21_]] : !tt.ptr<tensor<4x4xf32>>
// CHECK:             [[VAR_22_:%.+]] = arith.divui [[VAR_19_]], [[PARAM_4_]] : i32
// CHECK:             [[VAR_23_:%.+]] = tt.advance [[VAR_arg9_]], {{\[}}[[VAR_22_]], [[CST_0_i32]]] : <tensor<4x4xf32>>
// CHECK:             [[VAR_24_:%.+]] = arith.divui [[VAR_19_]], [[PARAM_6_]] : i32
// CHECK-DAG:         [[VAR_25_:%.+]] = tt.advance [[VAR_arg10_]], {{\[}}[[VAR_24_]], [[CST_0_i32]]] : <tensor<4x4xf32>>
// CHECK:             scf.yield [[VAR_23_]], [[VAR_25_]] : !tt.ptr<tensor<4x4xf32>>, !tt.ptr<tensor<4x4xf32>>
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
