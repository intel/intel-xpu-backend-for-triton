// RUN: triton-opt %s -triton-raise-block-pointer -canonicalize | FileCheck %s

module {
  tt.func public @wrap_side_by_side_masked_loop_01234567(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
    %cst = arith.constant dense<-9.900000e+01> : tensor<4x4xf32>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %cst_0 = arith.constant dense<2> : tensor<4x1xi32>
    %cst_1 = arith.constant dense<6> : tensor<4xi32>
    %cst_2 = arith.constant dense<2> : tensor<4xi32>
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %1 = arith.addi %0, %cst_2 : tensor<4xi32>
    %2 = arith.addi %0, %cst_1 : tensor<4xi32>
    %3 = tt.splat %arg3 : i32 -> tensor<4xi32>
    %4 = arith.remsi %2, %3 : tensor<4xi32>
    %5 = tt.expand_dims %1 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
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
    %27 = arith.cmpi slt, %16, %cst_0 : tensor<4x1xi32>
    %28 = tt.broadcast %27 : tensor<4x1xi1> -> tensor<4x4xi1>
    %29 = arith.muli %arg4, %c4_i32 : i32
    %30 = tt.splat %29 : i32 -> tensor<4x4xi32>
    %31 = arith.muli %arg5, %c4_i32 : i32
    %32 = tt.splat %31 : i32 -> tensor<4x4xi32>
    %33:2 = scf.for %arg8 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg9 = %15, %arg10 = %26) -> (tensor<4x4x!tt.ptr<f32>>, tensor<4x4x!tt.ptr<f32>>)  : i32 {
      %34 = tt.load %arg9, %28, %cst : tensor<4x4x!tt.ptr<f32>>
      tt.store %arg10, %34 : tensor<4x4x!tt.ptr<f32>>
      %35 = tt.addptr %arg9, %30 : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
      %36 = tt.addptr %arg10, %32 : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
      scf.yield %35, %36 : tensor<4x4x!tt.ptr<f32>>, tensor<4x4x!tt.ptr<f32>>
    }
    tt.return
  }
}

// CHECK:         tt.func public @wrap_side_by_side_masked_loop_01234567([[PARAM_0_:%.+]]: !tt.ptr<f32>, [[PARAM_1_:%.+]]: !tt.ptr<f32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32) {
// CHECK-DAG:       [[CST_:%.+]] = arith.constant dense<-9.900000e+01> : tensor<4x4xf32>
// CHECK-DAG:       [[CST_1_i32:%.+]] = arith.constant 1 : i32
// CHECK-DAG:       [[CST_0_i32:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_0_i64:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[CST_2_i32:%.+]] = arith.constant 2 : i32
// CHECK-DAG:       [[CST_4_i32:%.+]] = arith.constant 4 : i32
// CHECK-DAG:       [[CST_6_i32:%.+]] = arith.constant 6 : i32
// CHECK:           [[VAR_0_:%.+]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[PARAM_4_]] : i32 to index
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.muli [[PARAM_4_]], {{.*}} : i32
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[VAR_1_]] : index to i64
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.index_cast [[PARAM_3_]] : i32 to index
// CHECK-DAG:       [[VAR_5_:%.+]] = arith.index_cast [[PARAM_5_]] : i32 to index
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.index_cast [[VAR_5_]] : index to i64
// CHECK-DAG:       [[VAR_7_:%.+]] = arith.muli [[PARAM_5_]], [[CST_6_i32]] : i32
// CHECK-DAG:       [[VAR_8_:%.+]] = arith.index_cast [[VAR_4_]] : index to i64
// CHECK-DAG:       [[VAR_9_:%.+]] = arith.muli [[VAR_8_]], [[VAR_6_]] : i64
// CHECK:           [[VAR_10_:%.+]] = arith.trunci [[VAR_3_]] : i64 to i32
// CHECK:           [[VAR_11_:%.+]] = arith.divui [[VAR_2_]], [[VAR_10_]] : i32
// CHECK:           [[VAR_12_:%.+]] = arith.trunci [[VAR_6_]] : i64 to i32
// CHECK:           [[VAR_13_:%.+]] = arith.divui [[VAR_7_]], [[VAR_12_]] : i32
// CHECK:           [[VAR_14_:%.+]] = tt.make_tensor_ptr [[PARAM_0_]], {{\[}}[[CST_0_i64]], [[VAR_9_]]], {{\[}}[[VAR_3_]], [[VAR_6_]]], {{\[}}[[VAR_11_]], [[VAR_13_]]] {{.*}} : <tensor<4x4xf32>>
// CHECK-DAG:       [[VAR_15_:%.+]] = tt.expand_dims [[VAR_0_]] {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
// CHECK-DAG:       [[VAR_16_:%.+]] = arith.index_cast [[PARAM_6_]] : i32 to index
// CHECK-DAG:       [[VAR_17_:%.+]] = arith.index_cast [[VAR_16_]] : index to i64
// CHECK-DAG:       [[VAR_18_:%.+]] = arith.index_cast [[PARAM_7_]] : i32 to index
// CHECK-DAG:       [[VAR_19_:%.+]] = arith.index_cast [[VAR_18_]] : index to i64
// CHECK-DAG:       [[VAR_20_:%.+]] = arith.trunci [[VAR_17_]] : i64 to i32
// CHECK-DAG:       [[VAR_21_:%.+]] = arith.divui [[CST_0_i32]], [[VAR_20_]] : i32
// CHECK-DAG:       [[VAR_22_:%.+]] = arith.trunci [[VAR_19_]] : i64 to i32
// CHECK-DAG:       [[VAR_23_:%.+]] = arith.divui [[CST_0_i32]], [[VAR_22_]] : i32
// CHECK:           [[VAR_24_:%.+]] = tt.make_tensor_ptr [[PARAM_1_]], {{\[}}[[CST_0_i64]], [[CST_0_i64]]], {{\[}}[[VAR_17_]], [[VAR_19_]]], {{\[}}[[VAR_21_]], [[VAR_23_]]] {{.*}} : <tensor<4x4xf32>>
// CHECK:           [[VAR_25_:%.+]] = arith.cmpi slt, [[VAR_15_]], {{.*}} : tensor<4x1xi32>
// CHECK:           [[VAR_26_:%.+]] = tt.broadcast [[VAR_25_]] : tensor<4x1xi1> -> tensor<4x4xi1>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_27_:%.+]] = arith.muli [[PARAM_4_]], [[CST_4_i32]] : i32
// CHECK-DAG:       [[VAR_28_:%.+]] = arith.muli [[PARAM_5_]], [[CST_4_i32]] : i32
// CHECK-DAG:       [[VAR_12_:%.+]]:2 = scf.for [[VAR_arg8_:%.+]] = {{.*}} iter_args([[VAR_arg9_:%.+]] = [[VAR_14_]], [[VAR_arg10_:%.+]] = [[VAR_24_]]) -> (!tt.ptr<tensor<4x4xf32>>, !tt.ptr<tensor<4x4xf32>>) : i32 {
// CHECK:             [[VAR_30_:%.+]] = tt.load [[VAR_arg9_]], [[VAR_26_]], [[CST_]] : !tt.ptr<tensor<4x4xf32>>
// CHECK:             tt.store [[VAR_arg10_]], [[VAR_30_]] : !tt.ptr<tensor<4x4xf32>>
// CHECK-DAG:         [[VAR_31_:%.+]] = tt.advance [[VAR_arg9_]], {{\[}}[[CST_0_i32]], [[VAR_27_]]] : <tensor<4x4xf32>>
// CHECK-DAG:         [[VAR_32_:%.+]] = tt.advance [[VAR_arg10_]], {{\[}}[[CST_0_i32]], [[VAR_28_]]] : <tensor<4x4xf32>>
// CHECK:             scf.yield [[VAR_31_]], [[VAR_32_]] : !tt.ptr<tensor<4x4xf32>>, !tt.ptr<tensor<4x4xf32>>
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
