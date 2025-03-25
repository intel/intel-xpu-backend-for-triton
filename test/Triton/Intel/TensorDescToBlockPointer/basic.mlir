// RUN: triton-opt %s -triton-intel-tdesc-to-block-pointer  | FileCheck %s

module {
  tt.func public @test1(%arg0: !tt.ptr<i16>, %arg1: i32, %arg2: i32) {
    %c1_i64 = arith.constant 1 : i64
    %c64_i32 = arith.constant 64 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = arith.extsi %arg2 : i32 to i64
    %1 = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%0, %c1_i64] : <i16>, <tensor<8x32xi16>>
    %2 = tt.descriptor_load %1[%c8_i32, %c64_i32] : !tt.tensordesc<tensor<8x32xi16>> -> tensor<8x32xi16>
    tt.return
  }

  // CHECK:      tt.func public @test1([[PARAM_0:%.+]]: !tt.ptr<i16>, [[PARAM_1:%.+]]: i32, [[PARAM_2:%.+]]: i32) {
  // CHECK-DAG:     [[CST_1_i64:%.+]] = arith.constant 1 : i64
  // CHECK-DAG:     [[CST_64_i32:%.+]] = arith.constant 64 : i32
  // CHECK-DAG:     [[CST_8_i32:%.+]] = arith.constant 8 : i32
  // CHECK-NOT: separator of consecutive DAGs
  // CHECK-DAG:     [[EXTSI_PARAM_1:%.+]] = arith.extsi [[PARAM_1]] : i32 to i64
  // CHECK-DAG:     [[EXTSI_PARAM_2:%.+]] = arith.extsi [[PARAM_2]] : i32 to i64
  // CHECK:         [[TENSOR_PTR:%.+]] = tt.make_tensor_ptr [[PARAM_0]], {{\[}}[[EXTSI_PARAM_1]], [[EXTSI_PARAM_2]]], {{\[}}[[EXTSI_PARAM_2]], [[CST_1_i64]]], {{\[}}[[CST_8_i32]], [[CST_64_i32]]] {{.*}} : <tensor<8x32xi16>>
  // CHECK:         [[LOAD:%.+]] = tt.load [[TENSOR_PTR]] : !tt.ptr<tensor<8x32xi16>>
  // CHECK:         tt.return
  // CHECK:       }
}
