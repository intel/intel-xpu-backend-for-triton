// RUN: triton-opt %s -triton-intel-tdesc-to-block-pointer  | FileCheck %s

module {
  tt.func public @test_load(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: i32) {
    %c1_i64 = arith.constant 1 : i64
    %c64_i32 = arith.constant 64 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = arith.extsi %arg2 : i32 to i64
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%0, %c1_i64] : <f32>, <tensor<16x128xf32>>
    %load = tt.descriptor_load %desc[%c8_i32, %c64_i32] : !tt.tensordesc<tensor<16x128xf32>> -> tensor<16x128xf32>
    tt.return
  }
  // CHECK:      tt.func public @test_load([[PARAM_0:%.+]]: !tt.ptr<f32>, [[PARAM_1:%.+]]: i32, [[PARAM_2:%.+]]: i32) {
  // CHECK-NOT:    tt.make_tensor_descriptor
  // CHECK-NOT:    tt.descriptor_load
  // CHECK-DAG:    [[CST_1_i64:%.+]] = arith.constant 1 : i64
  // CHECK-DAG:    [[CST_64_i32:%.+]] = arith.constant 64 : i32
  // CHECK-DAG:    [[CST_8_i32:%.+]] = arith.constant 8 : i32
  // CHECK-DAG:    [[EXTSI_PARAM_1:%.+]] = arith.extsi [[PARAM_1]] : i32 to i64
  // CHECK-DAG:    [[EXTSI_PARAM_2:%.+]] = arith.extsi [[PARAM_2]] : i32 to i64
  // CHECK:        [[TENSOR_PTR:%.+]] = tt.make_tensor_ptr [[PARAM_0]], {{\[}}[[EXTSI_PARAM_1]], [[EXTSI_PARAM_2]]], {{\[}}[[EXTSI_PARAM_2]], [[CST_1_i64]]], {{\[}}[[CST_8_i32]], [[CST_64_i32]]] {{.*}} : <tensor<16x128xf32>>
  // CHECK:        [[LOAD:%.+]] = tt.load [[TENSOR_PTR]] : !tt.ptr<tensor<16x128xf32>>
  // CHECK:        tt.return
  // CHECK:      }

  tt.func public @test_store(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: i32) {
    %c1_i64 = arith.constant 1 : i64
    %c64_i32 = arith.constant 64 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst = arith.constant dense<1.000000e+00> : tensor<16x128xf32>
    %0 = arith.extsi %arg2 : i32 to i64
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%0, %c1_i64] : <f32>, <tensor<16x128xf32>>
    tt.descriptor_store %desc[%c8_i32, %c64_i32], %cst : !tt.tensordesc<tensor<16x128xf32>>, tensor<16x128xf32>
    tt.return
  }
  // CHECK:      tt.func public @test_store([[PARAM_0:%.+]]: !tt.ptr<f32>, [[PARAM_1:%.+]]: i32, [[PARAM_2:%.+]]: i32) {
  // CHECK-NOT:    tt.make_tensor_descriptor
  // CHECK-NOT:    tt.descriptor_store
  // CHECK-DAG:    [[CST_1_i64:%.+]] = arith.constant 1 : i64
  // CHECK-DAG:    [[CST_64_i32:%.+]] = arith.constant 64 : i32
  // CHECK-DAG:    [[CST_8_i32:%.+]] = arith.constant 8 : i32
  // CHECK-DAG:    [[CST:%.+]] = arith.constant dense<1.000000e+00> : tensor<16x128xf32>
  // CHECK-DAG:    [[EXTSI_PARAM_1:%.+]] = arith.extsi [[PARAM_1]] : i32 to i64
  // CHECK-DAG:    [[EXTSI_PARAM_2:%.+]] = arith.extsi [[PARAM_2]] : i32 to i64
  // CHECK:        [[TENSOR_PTR:%.+]] = tt.make_tensor_ptr [[PARAM_0]], {{\[}}[[EXTSI_PARAM_1]], [[EXTSI_PARAM_2]]], {{\[}}[[EXTSI_PARAM_2]], [[CST_1_i64]]], {{\[}}[[CST_8_i32]], [[CST_64_i32]]] {{.*}} : <tensor<16x128xf32>>
  // CHECK:        tt.store [[TENSOR_PTR]], [[CST]] {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<16x128xf32>>
  // CHECK:        tt.return
  // CHECK:      }
}
