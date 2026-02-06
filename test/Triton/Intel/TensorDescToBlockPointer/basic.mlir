// RUN: triton-opt %s -triton-intel-tdesc-to-block-pointer  | FileCheck %s

module {
  tt.func public @test_load(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: i32) {
    %c1_i64 = arith.constant 1 : i64
    %c64_i32 = arith.constant 64 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = arith.extsi %arg2 : i32 to i64
    %desc1 = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%0, %c1_i64] : <f32>, <tensor<16x128xf32>>
    %load1 = tt.descriptor_load %desc1[%c8_i32, %c64_i32] : !tt.tensordesc<tensor<16x128xf32>> -> tensor<16x128xf32>
    tt.return
  }
  // CHECK:      tt.func public @test_load([[PARAM_0:%.+]]: !tt.ptr<f32>, [[PARAM_1:%.+]]: i32, [[PARAM_2:%.+]]: i32) {
  // CHECK-NOT:    tt.make_tensor_descriptor
  // CHECK-NOT:    tt.descriptor_load
  // CHECK-DAG:    [[CST_0_i32:%.+]] = arith.constant 0 : i32
  // CHECK-DAG:    [[CST_1_i64:%.+]] = arith.constant 1 : i64
  // CHECK-DAG:    [[CST_64_i32:%.+]] = arith.constant 64 : i32
  // CHECK-DAG:    [[CST_8_i32:%.+]] = arith.constant 8 : i32
  // CHECK-DAG:    [[EXTSI_PARAM_1:%.+]] = arith.extsi [[PARAM_1]] : i32 to i64
  // CHECK-DAG:    [[EXTSI_PARAM_2:%.+]] = arith.extsi [[PARAM_2]] : i32 to i64
  // CHECK:        [[TENSOR_PTR:%.+]] = tt.make_tensor_ptr [[PARAM_0]], {{\[}}[[EXTSI_PARAM_1]], [[EXTSI_PARAM_2]]], {{\[}}[[EXTSI_PARAM_2]], [[CST_1_i64]]], {{\[}}[[CST_0_i32]], [[CST_0_i32]]] {{.*}} : <tensor<16x128xf32>>
  // CHECK:        [[TENSOR_PTR1:%.+]] = tt.advance [[TENSOR_PTR]], {{\[}}[[CST_8_i32]], [[CST_64_i32]]] : <tensor<16x128xf32>>
  // CHECK:        [[LOAD1:%.+]] = tt.load [[TENSOR_PTR1]] {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x128xf32>>
  // CHECK:        tt.return
  // CHECK:      }

  tt.func public @test_load_padding_nan(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: i32) {
    %c1_i64 = arith.constant 1 : i64
    %c64_i32 = arith.constant 64 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = arith.extsi %arg2 : i32 to i64
    %desc1 = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%0, %c1_i64] {padding = 2 : i32} : <f32>, <tensor<16x128xf32>>
    %load1 = tt.descriptor_load %desc1[%c8_i32, %c64_i32] : !tt.tensordesc<tensor<16x128xf32>> -> tensor<16x128xf32>
    tt.return
  }
  // CHECK:      tt.func public @test_load_padding_nan([[PARAM_0:%.+]]: !tt.ptr<f32>, [[PARAM_1:%.+]]: i32, [[PARAM_2:%.+]]: i32) {
  // CHECK-NOT:    tt.make_tensor_descriptor
  // CHECK-NOT:    tt.descriptor_load
  // CHECK-DAG:    [[CST_0_i32:%.+]] = arith.constant 0 : i32
  // CHECK-DAG:    [[CST_1_i64:%.+]] = arith.constant 1 : i64
  // CHECK-DAG:    [[CST_64_i32:%.+]] = arith.constant 64 : i32
  // CHECK-DAG:    [[CST_8_i32:%.+]] = arith.constant 8 : i32
  // CHECK-DAG:    [[EXTSI_PARAM_1:%.+]] = arith.extsi [[PARAM_1]] : i32 to i64
  // CHECK-DAG:    [[EXTSI_PARAM_2:%.+]] = arith.extsi [[PARAM_2]] : i32 to i64
  // CHECK:        [[TENSOR_PTR:%.+]] = tt.make_tensor_ptr [[PARAM_0]], {{\[}}[[EXTSI_PARAM_1]], [[EXTSI_PARAM_2]]], {{\[}}[[EXTSI_PARAM_2]], [[CST_1_i64]]], {{\[}}[[CST_0_i32]], [[CST_0_i32]]] {{.*}} : <tensor<16x128xf32>>
  // CHECK:        [[TENSOR_PTR1:%.+]] = tt.advance [[TENSOR_PTR]], {{\[}}[[CST_8_i32]], [[CST_64_i32]]] : <tensor<16x128xf32>>
  // CHECK:        [[LOAD1:%.+]] = tt.load [[TENSOR_PTR1]] {boundaryCheck = array<i32: 0, 1>, padding = 2 : i32} : !tt.ptr<tensor<16x128xf32>>
  // CHECK:        tt.return
  // CHECK:      }

  tt.func public @test_load_res_type_contracted(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.extsi %arg4 : i32 to i64
    %1 = arith.extsi %arg5 : i32 to i64
    %2 = arith.extsi %arg6 : i32 to i64
    %3 = arith.extsi %arg7 : i32 to i64
    %4 = tt.make_tensor_descriptor %arg1, [%c1_i32, %c1_i32, %c1_i32, %arg2, %arg3], [%0, %1, %2, %3, %c1_i64] : <i8>, <tensor<1x1x1x8x128xui8>>
    %5 = tt.descriptor_load %4[%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<tensor<1x1x1x8x128xui8>> -> tensor<8x128xi8>
    %6 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %7 = tt.expand_dims %6 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %8 = tt.splat %arg7 : i32 -> tensor<8x1xi32>
    %9 = arith.muli %7, %8 : tensor<8x1xi32>
    %10 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %11 = tt.expand_dims %10 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %12 = tt.broadcast %9 : tensor<8x1xi32> -> tensor<8x128xi32>
    %13 = tt.broadcast %11 : tensor<1x128xi32> -> tensor<8x128xi32>
    %14 = arith.addi %12, %13 : tensor<8x128xi32>
    %15 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<8x128x!tt.ptr<i8>>
    %16 = tt.addptr %15, %14 : tensor<8x128x!tt.ptr<i8>>, tensor<8x128xi32>
    tt.store %16, %5 : tensor<8x128x!tt.ptr<i8>>
    tt.return
  }
  // CHECK:      tt.func public @test_load_res_type_contracted([[PARAM_0:%.+]]: !tt.ptr<i8> {tt.divisibility = 16 : i32}, [[PARAM_1:%.+]]: !tt.ptr<i8> {tt.divisibility = 16 : i32}
  // CHECK-NOT:    tt.make_tensor_descriptor
  // CHECK-NOT:    tt.descriptor_load
  // CHECK:        [[CST_0:%.+]] = arith.constant 0 : i32
  // CHECK:        [[TENSOR_PTR:%.+]] = tt.make_tensor_ptr [[PARAM_1]], {{\[}}{{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}{{\]}}, {{.*}}, {{.*}}, {{.*}} : <tensor<1x1x1x8x128xi8>>
  // CHECK:        [[TENSOR_PTR1:%.+]] = tt.advance [[TENSOR_PTR]], {{\[}}[[CST_0]], [[CST_0]], [[CST_0]], [[CST_0]], [[CST_0]]{{\]}} : <tensor<1x1x1x8x128xi8>>
  // CHECK:        [[LOAD:%.+]] = tt.load [[TENSOR_PTR1]] {boundaryCheck = array<i32: 0, 1, 2, 3, 4>, padding = 1 : i32} : !tt.ptr<tensor<1x1x1x8x128xi8>>
  // CHECK:        [[RESHAPE:%.+]] = tt.reshape [[LOAD]] : tensor<1x1x1x8x128xi8> -> tensor<8x128xi8>
  // CHECK:        tt.store {{.*}}, [[RESHAPE]] : tensor<8x128x!tt.ptr<i8>>
  // CHECK:        tt.return
  // CHECK:      }

  tt.func public @test_store(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: i32) {
    %c1_i64 = arith.constant 1 : i64
    %c64_i32 = arith.constant 64 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst = arith.constant dense<1.000000e+00> : tensor<16x128xf32>
    %0 = arith.extsi %arg2 : i32 to i64
    %desc1 = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%0, %c1_i64] : <f32>, <tensor<16x128xf32>>
    tt.descriptor_store %desc1[%c8_i32, %c64_i32], %cst : !tt.tensordesc<tensor<16x128xf32>>, tensor<16x128xf32>
    tt.return
  }
  // CHECK:      tt.func public @test_store([[PARAM_0:%.+]]: !tt.ptr<f32>, [[PARAM_1:%.+]]: i32, [[PARAM_2:%.+]]: i32) {
  // CHECK-NOT:    tt.make_tensor_descriptor
  // CHECK-NOT:    tt.descriptor_store
  // CHECK-DAG:    [[CST_0_i32:%.+]] = arith.constant 0 : i32
  // CHECK-DAG:    [[CST_1_i64:%.+]] = arith.constant 1 : i64
  // CHECK-DAG:    [[CST_64_i32:%.+]] = arith.constant 64 : i32
  // CHECK-DAG:    [[CST_8_i32:%.+]] = arith.constant 8 : i32
  // CHECK-DAG:    [[CST:%.+]] = arith.constant dense<1.000000e+00> : tensor<16x128xf32>
  // CHECK-DAG:    [[EXTSI_PARAM_1:%.+]] = arith.extsi [[PARAM_1]] : i32 to i64
  // CHECK-DAG:    [[EXTSI_PARAM_2:%.+]] = arith.extsi [[PARAM_2]] : i32 to i64
  // CHECK:        [[TENSOR_PTR:%.+]] = tt.make_tensor_ptr [[PARAM_0]], {{\[}}[[EXTSI_PARAM_1]], [[EXTSI_PARAM_2]]], {{\[}}[[EXTSI_PARAM_2]], [[CST_1_i64]]], {{\[}}[[CST_0_i32]], [[CST_0_i32]]] {{.*}} : <tensor<16x128xf32>>
  // CHECK:        [[TENSOR_PTR1:%.+]] = tt.advance [[TENSOR_PTR]], {{\[}}[[CST_8_i32]], [[CST_64_i32]]] : <tensor<16x128xf32>>
  // CHECK:        tt.store [[TENSOR_PTR1]], [[CST]]  {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<16x128xf32>>
  // CHECK:        tt.return
  // CHECK:      }
}
