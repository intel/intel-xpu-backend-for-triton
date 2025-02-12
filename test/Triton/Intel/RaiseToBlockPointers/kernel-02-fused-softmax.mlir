// RUN: triton-opt %s -triton-raise-block-pointer=ignore-masks=true -canonicalize | FileCheck %s

module {
  tt.func public @softmax_kernel_012345(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32, %arg4: i32) {
    %cst = arith.constant 0xFF800000 : f32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %arg2 : i32
    %2 = tt.addptr %arg1, %1 : !tt.ptr<f32>, i32
    %3 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %4 = tt.splat %2 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %5 = tt.addptr %4, %3 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %6 = tt.splat %arg4 : i32 -> tensor<128xi32>
    %7 = arith.cmpi slt, %3, %6 : tensor<128xi32>
    %8 = tt.splat %cst : f32 -> tensor<128xf32>
    %9 = tt.load %5, %7, %8 : tensor<128x!tt.ptr<f32>>
    %10 = "tt.reduce"(%9) ({
    ^bb0(%arg5: f32, %arg6: f32):
      %21 = arith.cmpf ogt, %arg5, %arg6 : f32
      %22 = arith.select %21, %arg5, %arg6 : f32
      tt.reduce.return %22 : f32
    }) {axis = 0 : i32} : (tensor<128xf32>) -> f32
    %11 = tt.splat %10 : f32 -> tensor<128xf32>
    %12 = arith.subf %9, %11 : tensor<128xf32>
    %13 = math.exp %12 : tensor<128xf32>
    %14 = "tt.reduce"(%13) ({
    ^bb0(%arg5: f32, %arg6: f32):
      %21 = arith.addf %arg5, %arg6 : f32
      tt.reduce.return %21 : f32
    }) {axis = 0 : i32} : (tensor<128xf32>) -> f32
    %15 = tt.splat %14 : f32 -> tensor<128xf32>
    %16 = arith.divf %13, %15 : tensor<128xf32>
    %17 = arith.muli %0, %arg3 : i32
    %18 = tt.addptr %arg0, %17 : !tt.ptr<f32>, i32
    %19 = tt.splat %18 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %20 = tt.addptr %19, %3 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    tt.store %20, %16, %7 : tensor<128x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK:         tt.func public @softmax_kernel_012345([[PARAM_0_:%.+]]: !tt.ptr<f32>, [[PARAM_1_:%.+]]: !tt.ptr<f32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32) {
// CHECK-DAG:       [[CST_0_i64:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[CST_1_i64:%.+]] = arith.constant 1 : i64
// CHECK-DAG:       [[VAR_0_:%.+]] = tt.get_program_id x : i32
// CHECK:           [[VAR_1_:%.+]] = arith.muli [[VAR_0_]], [[PARAM_2_]] : i32
// CHECK:           [[VAR_2_:%.+]] = tt.make_tensor_ptr [[PARAM_1_]], {{\[}}[[CST_0_i64]]], {{\[}}[[CST_1_i64]]], {{\[}}[[VAR_1_]]] {{.*}} : <tensor<128xf32>>
// CHECK:           [[VAR_3_:%.+]] = tt.load [[VAR_2_]] : !tt.ptr<tensor<128xf32>>
// CHECK:           [[VAR_4_:%.+]] = "tt.reduce"([[VAR_3_]]) <{axis = 0 : i32}> ({
// CHECK:           ^bb0([[IN_0_:%.+]]: f32, [[IN_1_:%.+]]: f32):
// CHECK:             [[VAR_13_:%.+]] = arith.cmpf ogt, [[IN_0_]], [[IN_1_]] : f32
// CHECK:             [[VAR_14_:%.+]] = arith.select [[VAR_13_]], [[IN_0_]], [[IN_1_]] : f32
// CHECK:             tt.reduce.return [[VAR_14_]] : f32
// CHECK:           }) : (tensor<128xf32>) -> f32
// CHECK:           [[VAR_5_:%.+]] = tt.splat [[VAR_4_]] : f32 -> tensor<128xf32>
// CHECK:           [[VAR_6_:%.+]] = arith.subf [[VAR_3_]], [[VAR_5_]] : tensor<128xf32>
// CHECK:           [[VAR_7_:%.+]] = math.exp [[VAR_6_]] : tensor<128xf32>
// CHECK:           [[VAR_8_:%.+]] = "tt.reduce"([[VAR_7_]]) <{axis = 0 : i32}> ({
// CHECK:           ^bb0([[IN_2_:%.+]]: f32, [[IN_3_:%.+]]: f32):
// CHECK:             [[VAR_13_1_:%.+]] = arith.addf [[IN_2_]], [[IN_3_]] : f32
// CHECK:             tt.reduce.return [[VAR_13_1_]] : f32
// CHECK:           }) : (tensor<128xf32>) -> f32
// CHECK:           [[VAR_9_:%.+]] = tt.splat [[VAR_8_]] : f32 -> tensor<128xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = arith.divf [[VAR_7_]], [[VAR_9_]] : tensor<128xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = arith.muli [[VAR_0_]], [[PARAM_3_]] : i32
// CHECK:           [[VAR_12_:%.+]] = tt.make_tensor_ptr [[PARAM_0_]], {{\[}}[[CST_0_i64]]], {{\[}}[[CST_1_i64]]], {{\[}}[[VAR_11_]]] {{.*}} : <tensor<128xf32>>
// CHECK:           tt.store [[VAR_12_]], [[VAR_10_]] : !tt.ptr<tensor<128xf32>>
// CHECK:           tt.return
// CHECK:         }
