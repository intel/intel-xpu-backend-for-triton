// RUN: triton-opt %s -triton-intel-tdesc-to-block-pointer  | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func public @test_load(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: i32) {
    %c1_i64 = arith.constant 1 : i64
    %c64_i32 = arith.constant 64 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = arith.extsi %arg2 : i32 to i64
    %desc1 = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%0, %c1_i64] : <f32>, <tensor<16x128xf32>>
    %load1 = tt.descriptor_load %desc1[%c8_i32, %c64_i32] : !tt.tensordesc<tensor<16x128xf32>> -> tensor<16x128xf32>
    %load2 = tt.descriptor_load %desc1[%c8_i32, %c64_i32] : !tt.tensordesc<tensor<16x128xf32>> -> tensor<16x128xf32, #blocked>
    tt.return
  }
  // CHECK:      #[[$BLOCKED:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
  // CHECK:      tt.func public @test_load([[PARAM_0:%.+]]: !tt.ptr<f32>, [[PARAM_1:%.+]]: i32, [[PARAM_2:%.+]]: i32) {
  // CHECK-NOT:    tt.make_tensor_descriptor
  // CHECK-NOT:    tt.descriptor_load
  // CHECK-DAG:    [[CST_1_i64:%.+]] = arith.constant 1 : i64
  // CHECK-DAG:    [[CST_64_i32:%.+]] = arith.constant 64 : i32
  // CHECK-DAG:    [[CST_8_i32:%.+]] = arith.constant 8 : i32
  // CHECK-DAG:    [[EXTSI_PARAM_1:%.+]] = arith.extsi [[PARAM_1]] : i32 to i64
  // CHECK-DAG:    [[EXTSI_PARAM_2:%.+]] = arith.extsi [[PARAM_2]] : i32 to i64
  // CHECK:        [[TENSOR_PTR1:%.+]] = tt.make_tensor_ptr [[PARAM_0]], {{\[}}[[EXTSI_PARAM_1]], [[EXTSI_PARAM_2]]], {{\[}}[[EXTSI_PARAM_2]], [[CST_1_i64]]], {{\[}}[[CST_8_i32]], [[CST_64_i32]]] {{.*}} : <tensor<16x128xf32>>
  // CHECK:        [[LOAD1:%.+]] = tt.load [[TENSOR_PTR1]] : !tt.ptr<tensor<16x128xf32>>
  // CHECK:        [[TENSOR_PTR2:%.+]] = tt.make_tensor_ptr [[PARAM_0]], {{\[}}[[EXTSI_PARAM_1]], [[EXTSI_PARAM_2]]], {{\[}}[[EXTSI_PARAM_2]], [[CST_1_i64]]], {{\[}}[[CST_8_i32]], [[CST_64_i32]]] {{.*}} : <tensor<16x128xf32, #[[$BLOCKED]]>>
  // CHECK:        [[LOAD2:%.+]] = tt.load [[TENSOR_PTR2]] : !tt.ptr<tensor<16x128xf32, #[[$BLOCKED]]>>
  // CHECK:        tt.return
  // CHECK:      }

  tt.func public @test_store(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: i32) {
    %c1_i64 = arith.constant 1 : i64
    %c64_i32 = arith.constant 64 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst = arith.constant dense<1.000000e+00> : tensor<16x128xf32>
    %cst1 = arith.constant dense<1.000000e+00> : tensor<16x128xf32, #blocked>
    %0 = arith.extsi %arg2 : i32 to i64
    %desc1 = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%0, %c1_i64] : <f32>, <tensor<16x128xf32>>
    tt.descriptor_store %desc1[%c8_i32, %c64_i32], %cst : !tt.tensordesc<tensor<16x128xf32>>, tensor<16x128xf32>
    tt.descriptor_store %desc1[%c8_i32, %c64_i32], %cst1 : !tt.tensordesc<tensor<16x128xf32>>, tensor<16x128xf32, #blocked>
    tt.return
  }
  // CHECK:      tt.func public @test_store([[PARAM_0:%.+]]: !tt.ptr<f32>, [[PARAM_1:%.+]]: i32, [[PARAM_2:%.+]]: i32) {
  // CHECK-NOT:    tt.make_tensor_descriptor
  // CHECK-NOT:    tt.descriptor_store
  // CHECK-DAG:    [[CST_1_i64:%.+]] = arith.constant 1 : i64
  // CHECK-DAG:    [[CST_64_i32:%.+]] = arith.constant 64 : i32
  // CHECK-DAG:    [[CST_8_i32:%.+]] = arith.constant 8 : i32
  // CHECK-DAG:    [[CST:%.+]] = arith.constant dense<1.000000e+00> : tensor<16x128xf32>
  // CHECK-DAG:    [[CST1:%.+]] = arith.constant dense<1.000000e+00> : tensor<16x128xf32, #[[$BLOCKED]]>
  // CHECK-DAG:    [[EXTSI_PARAM_1:%.+]] = arith.extsi [[PARAM_1]] : i32 to i64
  // CHECK-DAG:    [[EXTSI_PARAM_2:%.+]] = arith.extsi [[PARAM_2]] : i32 to i64
  // CHECK:        [[TENSOR_PTR1:%.+]] = tt.make_tensor_ptr [[PARAM_0]], {{\[}}[[EXTSI_PARAM_1]], [[EXTSI_PARAM_2]]], {{\[}}[[EXTSI_PARAM_2]], [[CST_1_i64]]], {{\[}}[[CST_8_i32]], [[CST_64_i32]]] {{.*}} : <tensor<16x128xf32>>
  // CHECK:        tt.store [[TENSOR_PTR1]], [[CST]] : !tt.ptr<tensor<16x128xf32>>
  // CHECK:        [[TENSOR_PTR2:%.+]] = tt.make_tensor_ptr [[PARAM_0]], {{\[}}[[EXTSI_PARAM_1]], [[EXTSI_PARAM_2]]], {{\[}}[[EXTSI_PARAM_2]], [[CST_1_i64]]], {{\[}}[[CST_8_i32]], [[CST_64_i32]]] {{.*}} : <tensor<16x128xf32, #[[$BLOCKED]]>>
  // CHECK:        tt.store [[TENSOR_PTR2]], [[CST1]] : !tt.ptr<tensor<16x128xf32, #[[$BLOCKED]]>>
  // CHECK:        tt.return
  // CHECK:      }
}
