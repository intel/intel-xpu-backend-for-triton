// RUN: triton-opt %s -triton-raise-block-pointer -canonicalize | FileCheck %s

// CHECK-LABEL:   tt.func @test_addptr_splat_make_range(
// CHECK-SAME:                                          %[[VAL_0:.*]]: !tt.ptr<f32>) -> tensor<128xf32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_2:.*]] = arith.constant 128 : i32
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_4:.*]] = tt.make_tensor_ptr %[[VAL_0]], {{\[}}%[[VAL_1]]], {{\[}}%[[VAL_3]]], {{\[}}%[[VAL_2]]] {order = array<i32>} : <tensor<128xf32>>
// CHECK:           %[[VAL_5:.*]] = tt.load %[[VAL_4]] : !tt.ptr<tensor<128xf32>>
// CHECK:           tt.return %[[VAL_5]] : tensor<128xf32>
tt.func @test_addptr_splat_make_range(%arg0 : !tt.ptr<f32>) -> tensor<128xf32> {
  %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
  %1 = tt.make_range {start = 128 : i32, end = 256 : i32} : tensor<128xi32>
  %2 = tt.addptr %0, %1 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
  %3 = tt.load %2 : tensor<128x!tt.ptr<f32>>
  tt.return %3 : tensor<128xf32>
}

// CHECK-LABEL:   tt.func @test_addptr_splat_splat_i32(
// CHECK-SAME:                                         %[[VAL_0:.*]]: !tt.ptr<f32>,
// CHECK-SAME:                                         %[[VAL_1:.*]]: i32,
// CHECK-SAME:                                         %[[VAL_2:.*]]: tensor<128xi1>) -> tensor<128xf32> {
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_4:.*]] = tt.make_tensor_ptr %[[VAL_0]], {{\[}}%[[VAL_3]]], {{\[}}%[[VAL_3]]], {{\[}}%[[VAL_1]]] {order = array<i32>} : <tensor<128xf32>>
// CHECK:           %[[VAL_5:.*]] = tt.load %[[VAL_4]], %[[VAL_2]] cacheModifier = ca evictionPolicy = evict_first {isVolatile = true} : !tt.ptr<tensor<128xf32>>
// CHECK:           tt.return %[[VAL_5]] : tensor<128xf32>
tt.func @test_addptr_splat_splat_i32(%arg0 : !tt.ptr<f32>, %arg1: i32, %arg2: tensor<128xi1>) -> tensor<128xf32> {
  %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
  %1 = tt.splat %arg1 : i32 -> tensor<128xi32>
  %2 = tt.addptr %0, %1 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
  %3 = tt.load %2, %arg2 cacheModifier = ca evictionPolicy = evict_first {isVolatile = true} : tensor<128x!tt.ptr<f32>>
  tt.return %3 : tensor<128xf32>
}

// CHECK-LABEL:   tt.func @test_addptr_splat_splat_i64(
// CHECK-SAME:                                         %[[VAL_0:.*]]: !tt.ptr<f32>,
// CHECK-SAME:                                         %[[VAL_1:.*]]: i64) -> tensor<128xf32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_3:.*]] = arith.index_cast %[[VAL_1]] : i64 to index
// CHECK:           %[[VAL_4:.*]] = arith.index_cast %[[VAL_3]] : index to i32
// CHECK:           %[[VAL_5:.*]] = tt.make_tensor_ptr %[[VAL_0]], {{\[}}%[[VAL_2]]], {{\[}}%[[VAL_2]]], {{\[}}%[[VAL_4]]] {order = array<i32>} : <tensor<128xf32>>
// CHECK:           %[[VAL_6:.*]] = tt.load %[[VAL_5]] : !tt.ptr<tensor<128xf32>>
// CHECK:           tt.return %[[VAL_6]] : tensor<128xf32>
tt.func @test_addptr_splat_splat_i64(%arg0 : !tt.ptr<f32>, %arg1: i64) -> tensor<128xf32> {
  %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
  %1 = tt.splat %arg1 : i64 -> tensor<128xi64>
  %2 = tt.addptr %0, %1 : tensor<128x!tt.ptr<f32>>, tensor<128xi64>
  %3 = tt.load %2 : tensor<128x!tt.ptr<f32>>
  tt.return %3 : tensor<128xf32>
}

// CHECK-LABEL:   tt.func @test_addptr_splat_splat_2d(
// CHECK-SAME:                                        %[[VAL_0:.*]]: !tt.ptr<f32>,
// CHECK-SAME:                                        %[[VAL_1:.*]]: i64,
// CHECK-SAME:                                        %[[VAL_2:.*]]: tensor<2x128xi1>,
// CHECK-SAME:                                        %[[VAL_3:.*]]: tensor<2x128xf32>) -> tensor<2x128xf32> {
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_6:.*]] = arith.index_cast %[[VAL_1]] : i64 to index
// CHECK:           %[[VAL_7:.*]] = arith.index_cast %[[VAL_6]] : index to i32
// CHECK:           %[[VAL_8:.*]] = tt.make_tensor_ptr %[[VAL_0]], {{\[}}%[[VAL_4]], %[[VAL_4]]], {{\[}}%[[VAL_4]], %[[VAL_4]]], {{\[}}%[[VAL_7]], %[[VAL_5]]] {order = array<i32>} : <tensor<2x128xf32>>
// CHECK:           %[[VAL_9:.*]] = tt.load %[[VAL_8]], %[[VAL_2]], %[[VAL_3]] : !tt.ptr<tensor<2x128xf32>>
// CHECK:           tt.return %[[VAL_9]] : tensor<2x128xf32>
tt.func @test_addptr_splat_splat_2d(%arg0 : !tt.ptr<f32>, %arg1: i64, %arg2: tensor<2x128xi1>, %arg3: tensor<2x128xf32>) -> tensor<2x128xf32> {
  %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x128x!tt.ptr<f32>>
  %1 = tt.splat %arg1 : i64 -> tensor<2x128xi64>
  %2 = tt.addptr %0, %1 : tensor<2x128x!tt.ptr<f32>>, tensor<2x128xi64>
  %3 = tt.load %2, %arg2, %arg3 : tensor<2x128x!tt.ptr<f32>>
  tt.return %3 : tensor<2x128xf32>
}
