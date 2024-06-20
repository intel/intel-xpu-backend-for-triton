// RUN: triton-opt %s -triton-raise-block-pointer | FileCheck %s

// CHECK-LABEL:   tt.func @test_addptr_splat_make_range(
// CHECK-SAME:                                          %[[VAL_0:.*]]: !tt.ptr<f32>) {
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : i64
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 128 : i32
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 1 : i64
// CHECK-DAG:       %[[VAL_7:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_8:.*]] = arith.addi %[[VAL_3]], %[[VAL_5]] : i32
// CHECK:           %[[VAL_9:.*]] = arith.addi %[[VAL_4]], %[[VAL_6]] : i64
// CHECK:           %[[VAL_10:.*]] = tt.make_tensor_ptr %[[VAL_0]], {{\[}}%[[VAL_4]]], {{\[}}%[[VAL_9]]], {{\[}}%[[VAL_8]]] {order = array<i32>} : <tensor<128xf32>>
tt.func @test_addptr_splat_make_range(%arg0 : !tt.ptr<f32>) {
  %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
  %1 = tt.make_range {start = 128 : i32, end = 256 : i32} : tensor<128xi32>
  %2 = tt.addptr %0, %1 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
  tt.return
}

// CHECK-LABEL:   tt.func @test_addptr_splat_splat_i32(
// CHECK-SAME:                                         %[[VAL_0:.*]]: !tt.ptr<f32>,
// CHECK-SAME:                                         %[[VAL_1:.*]]: i32) {
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 0 : i64
// CHECK-DAG:       %[[VAL_6:.*]] = arith.index_cast %[[VAL_1]] : i32 to index
// CHECK-DAG:       %[[VAL_8:.*]] = arith.constant 0 : i64
// CHECK-DAG:       %[[VAL_9:.*]] = arith.index_cast %[[VAL_6]] : index to i32
// CHECK:           %[[VAL_10:.*]] = arith.addi %[[VAL_4]], %[[VAL_9]] : i32
// CHECK:           %[[VAL_11:.*]] = arith.addi %[[VAL_5]], %[[VAL_8]] : i64
// CHECK:           %[[VAL_12:.*]] = tt.make_tensor_ptr %[[VAL_0]], {{\[}}%[[VAL_5]]], {{\[}}%[[VAL_11]]], {{\[}}%[[VAL_10]]] {order = array<i32>} : <tensor<128xf32>>
tt.func @test_addptr_splat_splat_i32(%arg0 : !tt.ptr<f32>, %arg1: i32) {
  %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
  %1 = tt.splat %arg1 : i32 -> tensor<128xi32>
  %2 = tt.addptr %0, %1 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
  tt.return
}

// CHECK-LABEL:   tt.func @test_addptr_splat_splat_i64(
// CHECK-SAME:                                         %[[VAL_0:.*]]: !tt.ptr<f32>,
// CHECK-SAME:                                         %[[VAL_1:.*]]: i64) {
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 0 : i64
// CHECK-DAG:       %[[VAL_6:.*]] = arith.index_cast %[[VAL_1]] : i64 to index
// CHECK-DAG:       %[[VAL_8:.*]] = arith.constant 0 : i64
// CHECK-DAG:       %[[VAL_9:.*]] = arith.index_cast %[[VAL_6]] : index to i32
// CHECK:           %[[VAL_10:.*]] = arith.addi %[[VAL_4]], %[[VAL_9]] : i32
// CHECK:           %[[VAL_11:.*]] = arith.addi %[[VAL_5]], %[[VAL_8]] : i64
// CHECK:           %[[VAL_12:.*]] = tt.make_tensor_ptr %[[VAL_0]], {{\[}}%[[VAL_5]]], {{\[}}%[[VAL_11]]], {{\[}}%[[VAL_10]]] {order = array<i32>} : <tensor<128xf32>>
tt.func @test_addptr_splat_splat_i64(%arg0 : !tt.ptr<f32>, %arg1: i64) {
  %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
  %1 = tt.splat %arg1 : i64 -> tensor<128xi64>
  %2 = tt.addptr %0, %1 : tensor<128x!tt.ptr<f32>>, tensor<128xi64>
  tt.return
}

// CHECK-LABEL:   tt.func @test_addptr_splat_splat_2d(
// CHECK-SAME:                                        %[[VAL_0:.*]]: !tt.ptr<f32>,
// CHECK-SAME:                                        %[[VAL_1:.*]]: i64) {
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 0 : i64
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[VAL_7:.*]] = arith.constant 0 : i64
// CHECK-DAG:       %[[VAL_8:.*]] = arith.index_cast %[[VAL_1]] : i64 to index
// CHECK-DAG:       %[[VAL_9:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[VAL_10:.*]] = arith.constant 0 : i64
// CHECK-DAG:       %[[VAL_11:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[VAL_12:.*]] = arith.constant 0 : i64
// CHECK-DAG:       %[[VAL_13:.*]] = arith.index_cast %[[VAL_8]] : index to i32
// CHECK:           %[[VAL_14:.*]] = arith.addi %[[VAL_4]], %[[VAL_13]] : i32
// CHECK:           %[[VAL_15:.*]] = arith.addi %[[VAL_5]], %[[VAL_10]] : i64
// CHECK:           %[[VAL_16:.*]] = arith.addi %[[VAL_6]], %[[VAL_11]] : i32
// CHECK:           %[[VAL_17:.*]] = arith.addi %[[VAL_7]], %[[VAL_12]] : i64
// CHECK:           %[[VAL_18:.*]] = tt.make_tensor_ptr %[[VAL_0]], {{\[}}%[[VAL_5]], %[[VAL_7]]], {{\[}}%[[VAL_15]], %[[VAL_17]]], {{\[}}%[[VAL_14]], %[[VAL_16]]] {order = array<i32>} : <tensor<2x128xf32>>
tt.func @test_addptr_splat_splat_2d(%arg0 : !tt.ptr<f32>, %arg1: i64) {
  %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x128x!tt.ptr<f32>>
  %1 = tt.splat %arg1 : i64 -> tensor<2x128xi64>
  %2 = tt.addptr %0, %1 : tensor<2x128x!tt.ptr<f32>>, tensor<2x128xi64>
  tt.return
}
