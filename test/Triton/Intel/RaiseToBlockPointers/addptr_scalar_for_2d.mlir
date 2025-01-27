// RUN: triton-opt.exe %s -triton-raise-block-pointer -canonicalize | FileCheck %s

module {
  tt.func @kernel (%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32) {
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %arg2 : i32
    %2 = tt.addptr %arg1, %1 : !tt.ptr<f32>, i32
    %cf0 = arith.constant 0.000000e+00 : f32
    %tensor_cf0 = tt.splat %cf0 : f32 -> tensor<128x128xf32>
    %c0 = arith.constant 0 : index
    %c12 = arith.constant 12 : index
    %c3 = arith.constant 3 : index
    %sum_out, %_ptr = scf.for %i = %c0 to %c12 step %c3 iter_args(%sum_iter = %tensor_cf0,  %ptr_iter = %2) ->  (tensor<128x128xf32>, !tt.ptr<f32> ) {
      %3 = tt.splat %ptr_iter : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>>
      // source = %arg1, offset = [%1, 0], size = [128, 128], strides = [0, 0]
      %4 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
      %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
      %6 = tt.broadcast %5 : tensor<1x128xi32> -> tensor<128x128xi32>
      // offset = [0, 0], size = [128, 128], strides = [0, 1]
      %7 = tt.make_range {end = 256 : i32, start = 128 : i32} : tensor<128xi32>
      %8 = tt.expand_dims %7 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
      %9 = tt.broadcast %8 : tensor<128x1xi32> -> tensor<128x128xi32>
      // offset = [128, 0], size = [128, 128], strides = [1, 0]
      %10 = arith.addi %6, %9 : tensor<128x128xi32>
      // offset = [128, 0], size = [128, 128], strides = [1, 1]
      %11 = tt.addptr %3, %10 : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>
      // source = %arg1, offset = [%1 + 128, 0], size = [128, 128], strides = [1, 1]
      %12 = tt.load %11 : tensor<128x128x!tt.ptr<f32>>
      %17 = math.exp %12 : tensor<128x128xf32>
      %sum_next = arith.addf %sum_iter, %17 : tensor<128x128xf32>
      %cast_i = arith.index_cast %i : index to i32
      %ptr_next = tt.addptr %ptr_iter, %cast_i : !tt.ptr<f32>, i32
      // source = %arg1, offset = %1 + %i, size = 1, strides = 0
      scf.yield %sum_next, %ptr_next : tensor<128x128xf32>, !tt.ptr<f32>
    }
    %4 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %6 = tt.broadcast %5 : tensor<1x128xi32> -> tensor<128x128xi32>
    // offset = [0, 0], size = [128, 128], strides = [0, 1]
    %7 = tt.make_range {end = 256 : i32, start = 128 : i32} : tensor<128xi32>
    %8 = tt.expand_dims %7 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    %9 = tt.broadcast %8 : tensor<128x1xi32> -> tensor<128x128xi32>
    // offset = [128, 0], size = [128, 128], strides = [1, 0]
    %10 = arith.addi %6, %9 : tensor<128x128xi32>
    // offset = [128, 0], size = [128, 128], strides = [1, 1]
    %18 = arith.muli %0, %arg3 : i32
    %19 = tt.addptr %arg0, %18 : !tt.ptr<f32>, i32
    // source = arg0, offset = %18, size = 1, strides = 0
    %20 = tt.splat %19 : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>>
    // source = arg0, offset = [%18, 0], size = [128, 128], strides = [0, 0]
    %21 = tt.addptr %20, %10 : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>
    // source = %arg0, offset = [%18 + 128, 0], size = [128, 128], strides = [1, 1]
    tt.store %21, %sum_out : tensor<128x128x!tt.ptr<f32>>
    tt.return
  }
}
// CHECK:         tt.func @kernel([[PARAM_0_:%.+]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}, [[PARAM_1_:%.+]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32) {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
// CHECK-DAG:       [[CST_0_i64:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[CST_1_i64:%.+]] = arith.constant 1 : i64
// CHECK-DAG:       [[CST_0_i32:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_128_i32:%.+]] = arith.constant 128 : i32
// CHECK-DAG:       [[VAR_0_:%.+]] = tt.get_program_id x : i32
// CHECK:           [[VAR_1_:%.+]] = arith.muli [[VAR_0_]], [[PARAM_2_]] : i32
// CHECK:           [[VAR_2_:%.+]] = tt.addptr [[PARAM_1_]], [[VAR_1_]] : !tt.ptr<f32>, i32
// CHECK-DAG:       [[VAR_3_:%.+]]:2 = scf.for [[VAR_arg5_:%.+]] = {{.*}} iter_args([[VAR_arg6_:%.+]] = [[CST_0_]], [[VAR_arg7_:%.+]] = [[VAR_2_]]) -> (tensor<128x128xf32>, !tt.ptr<f32>) {
// CHECK-NOT:         tt.make_tensor_ptr
// CHECK-NOT:         tt.advance
// CHECK-DAG:         [[VAR_20_:%.+]] = tt.addptr [[VAR_arg7_]], {{.*}} : !tt.ptr<f32>, i32
// CHECK:             scf.yield {{.*}}, [[VAR_20_]] : tensor<128x128xf32>, !tt.ptr<f32>
// CHECK:           }
// CHECK:           [[VAR_4_:%.+]] = arith.muli [[VAR_0_]], [[PARAM_3_]] : i32
// CHECK:           [[VAR_5_:%.+]] = arith.addi [[VAR_4_]], [[CST_128_i32]] : i32
// CHECK:           [[VAR_6_:%.+]] = tt.make_tensor_ptr [[PARAM_0_]], {{\[}}[[CST_0_i64]], [[CST_0_i64]]], {{\[}}[[CST_1_i64]], [[CST_1_i64]]], {{\[}}[[VAR_5_]], [[CST_0_i32]]] {{.*}} : <tensor<128x128xf32>>
// CHECK:           tt.store [[VAR_6_]], [[VAR_3_]]#0 : !tt.ptr<tensor<128x128xf32>>
// CHECK:           tt.return
// CHECK:         }
