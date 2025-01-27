// RUN: triton-opt %s -triton-raise-block-pointer -canonicalize | FileCheck %s

module {
  tt.func @kernel (%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32) {
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %arg2 : i32
    %2 = tt.addptr %arg1, %1 : !tt.ptr<f32>, i32
    // source = arg1, offset = %1, size = 1, strides = 0
    %3 = tt.splat %2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    // source = arg1, offset = %1, size = 1024, strides = 0
    %4 = tt.expand_dims %3 {axis = 1 : i32} : tensor<1024x!tt.ptr<f32>> -> tensor<1024x1x!tt.ptr<f32>>
    // source = arg1, offset = [%1, 0], size = [1024, 1], strides = [0, 0]
    %5 = tt.broadcast %4 : tensor<1024x1x!tt.ptr<f32>> -> tensor<1024x1024x!tt.ptr<f32>>
    // source = arg1, offset = [%1, 0], size = [1024, 1024], strides = [0, 0]
    %6 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    // offset = 0, size = 1024, strides = 1
    %7 = tt.expand_dims %6 {axis = 0 : i32} : tensor<1024xi32> -> tensor<1x1024xi32>
    // offset = [0, 0], size = [1, 1024], strides = [0, 1]
    %8 = tt.broadcast %7 : tensor<1x1024xi32> -> tensor<1024x1024xi32>
    // offset = [0, 0], size = [1024, 1024], strides = [0, 1]
    %9 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    // offset = 0, size = 1024, strides = 1
    %10 = tt.expand_dims %9 {axis = 1 : i32} : tensor<1024xi32> -> tensor<1024x1xi32>
    // offset = [0, 0], size = [1024, 1], strides = [1, 0]
    %11 = tt.broadcast %10 : tensor<1024x1xi32> -> tensor<1024x1024xi32>
    // offset = [0, 0], size = [1024, 1024], strides = [1, 0]
    %12 = arith.addi %8, %11 : tensor<1024x1024xi32>
    // offset = [0, 0], size = [1024, 1024], strides = [1, 1]
    %13 = tt.addptr %5, %12 : tensor<1024x1024x!tt.ptr<f32>>, tensor<1024x1024xi32>
    // source = arg1, offset = [pid * %arg2, 0], size = [1024, 1024], strides = [1, 1]
    %14 = tt.load %13 : tensor<1024x1024x!tt.ptr<f32>>
    %17 = math.exp %14 : tensor<1024x1024xf32>
    %18 = arith.muli %0, %arg3 : i32
    %19 = tt.addptr %arg0, %18 : !tt.ptr<f32>, i32
    // source = arg0, offset = pid+arg3, size = 1, strides = 0
    %20 = tt.splat %19 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    // source = arg0, offset = pid+arg3, size = 1024, strides = 0
    %21 = tt.expand_dims %20 {axis = 1 : i32} : tensor<1024x!tt.ptr<f32>> -> tensor<1024x1x!tt.ptr<f32>>
    // source = arg0, offset = [pid+arg3, 0], size = [1024, 1], strides = [0, 0]
    %22 = tt.broadcast %21 : tensor<1024x1x!tt.ptr<f32>> -> tensor<1024x1024x!tt.ptr<f32>>
    // source = arg0, offset = [pid+arg3, 0], size = [1024, 1024], strides = [0, 0]
    %23 = tt.addptr %22, %12 : tensor<1024x1024x!tt.ptr<f32>>, tensor<1024x1024xi32>
    // source = arg0, offset = [pid+arg3, 0], size = [1024, 1024], strides = [1, 1]
    tt.store %23, %17 : tensor<1024x1024x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK:         tt.func @kernel([[PARAM_0_:%.+]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}, [[PARAM_1_:%.+]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32) {
// CHECK-DAG:       [[CST_0_i64:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[CST_0_i32:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_1_i64:%.+]] = arith.constant 1 : i64
// CHECK:           [[VAR_0_:%.+]] = tt.get_program_id x : i32
// CHECK:           [[VAR_1_:%.+]] = arith.muli [[VAR_0_]], [[PARAM_2_]] : i32
// CHECK:           [[VAR_2_:%.+]] = tt.make_tensor_ptr [[PARAM_1_]], {{\[}}[[CST_0_i64]], [[CST_0_i64]]], {{\[}}[[CST_1_i64]], [[CST_1_i64]]], {{\[}}[[VAR_1_]], [[CST_0_i32]]] {{.*}} : <tensor<1024x1024xf32>>
// CHECK:           [[VAR_3_:%.+]] = tt.load [[VAR_2_]] : !tt.ptr<tensor<1024x1024xf32>>
// CHECK-DAG:       [[VAR_4_:%.+]] = math.exp [[VAR_3_]] : tensor<1024x1024xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = arith.muli [[VAR_0_]], [[PARAM_3_]] : i32
// CHECK:           [[VAR_6_:%.+]] = tt.make_tensor_ptr [[PARAM_0_]], {{\[}}[[CST_0_i64]], [[CST_0_i64]]], {{\[}}[[CST_1_i64]], [[CST_1_i64]]], {{\[}}[[VAR_5_]], [[CST_0_i32]]] {{.*}} : <tensor<1024x1024xf32>>
// CHECK:           tt.store [[VAR_6_]], [[VAR_4_]] : !tt.ptr<tensor<1024x1024xf32>>
// CHECK:           tt.return
// CHECK:         }
