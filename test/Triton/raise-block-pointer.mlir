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

// CHECK-LABEL:   tt.func @test_addptr_splat_splat_2d_store(
// CHECK-SAME:                                              %[[VAL_0:.*]]: !tt.ptr<f32>,
// CHECK-SAME:                                              %[[VAL_1:.*]]: i64,
// CHECK-SAME:                                              %[[VAL_2:.*]]: tensor<2x128xi1>,
// CHECK-SAME:                                              %[[VAL_3:.*]]: tensor<2x128xf32>) {
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_6:.*]] = arith.index_cast %[[VAL_1]] : i64 to index
// CHECK:           %[[VAL_7:.*]] = arith.index_cast %[[VAL_6]] : index to i32
// CHECK:           %[[VAL_8:.*]] = tt.make_tensor_ptr %[[VAL_0]], {{\[}}%[[VAL_4]], %[[VAL_4]]], {{\[}}%[[VAL_4]], %[[VAL_4]]], {{\[}}%[[VAL_7]], %[[VAL_5]]] {order = array<i32>} : <tensor<2x128xf32>>
// CHECK:           tt.store %[[VAL_8]], %[[VAL_3]], %[[VAL_2]] : !tt.ptr<tensor<2x128xf32>>
tt.func @test_addptr_splat_splat_2d_store(%arg0 : !tt.ptr<f32>, %arg1: i64, %arg2: tensor<2x128xi1>, %arg3: tensor<2x128xf32>) {
  %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x128x!tt.ptr<f32>>
  %1 = tt.splat %arg1 : i64 -> tensor<2x128xi64>
  %2 = tt.addptr %0, %1 : tensor<2x128x!tt.ptr<f32>>, tensor<2x128xi64>
  tt.store %2, %arg3, %arg2 : tensor<2x128x!tt.ptr<f32>>
  tt.return
}

// CHECK-LABEL:   tt.func @test_addptr_splat_make_range_add(
// CHECK-SAME:                                              %[[VAL_0:.*]]: !tt.ptr<f32>) -> tensor<128xf32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_3:.*]] = arith.constant 2 : i64
// CHECK:           %[[VAL_4:.*]] = tt.make_tensor_ptr %[[VAL_0]], {{\[}}%[[VAL_1]]], {{\[}}%[[VAL_3]]], {{\[}}%[[VAL_2]]] {order = array<i32>} : <tensor<128xf32>>
// CHECK:           %[[VAL_5:.*]] = tt.load %[[VAL_4]] : !tt.ptr<tensor<128xf32>>
// CHECK:           tt.return %[[VAL_5]] : tensor<128xf32>
tt.func @test_addptr_splat_make_range_add(%arg0 : !tt.ptr<f32>) -> tensor<128xf32> {
  %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
  %1 = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
  %2 = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
  %3 = arith.addi %1, %2 : tensor<128xi32>
  %4 = tt.addptr %0, %3 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
  %5 = tt.load %4 : tensor<128x!tt.ptr<f32>>
  tt.return %5 : tensor<128xf32>
}

// CHECK-LABEL:   tt.func @test_addptr_splat_make_range_mul(
// CHECK-SAME:                                              %[[VAL_0:.*]]: !tt.ptr<f32>,
// CHECK-SAME:                                              %[[VAL_1:.*]]: i32) -> tensor<128xf32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_4:.*]] = arith.index_cast %[[VAL_1]] : i32 to index
// CHECK:           %[[VAL_5:.*]] = arith.index_cast %[[VAL_4]] : index to i64
// CHECK:           %[[VAL_6:.*]] = tt.make_tensor_ptr %[[VAL_0]], {{\[}}%[[VAL_2]]], {{\[}}%[[VAL_5]]], {{\[}}%[[VAL_3]]] {order = array<i32>} : <tensor<128xf32>>
// CHECK:           %[[VAL_7:.*]] = tt.load %[[VAL_6]] : !tt.ptr<tensor<128xf32>>
// CHECK:           tt.return %[[VAL_7]] : tensor<128xf32>
tt.func @test_addptr_splat_make_range_mul(%arg0 : !tt.ptr<f32>, %arg1: i32) -> tensor<128xf32> {
  %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
  %1 = tt.splat %arg1 : i32 -> tensor<128xi32>
  %2 = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
  %3 = arith.muli %1, %2 : tensor<128xi32>
  %4 = tt.addptr %0, %3 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
  %5 = tt.load %4 : tensor<128x!tt.ptr<f32>>
  tt.return %5 : tensor<128xf32>
}

// CHECK-LABEL:   tt.func @test_const_splat_addptr(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !tt.ptr<f32>) -> tensor<128xf32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_2:.*]] = arith.constant 512 : i32
// CHECK:           %[[VAL_3:.*]] = tt.make_tensor_ptr %[[VAL_0]], {{\[}}%[[VAL_1]]], {{\[}}%[[VAL_1]]], {{\[}}%[[VAL_2]]] {order = array<i32>} : <tensor<128xf32>>
// CHECK:           %[[VAL_4:.*]] = tt.load %[[VAL_3]] : !tt.ptr<tensor<128xf32>>
// CHECK:           tt.return %[[VAL_4]] : tensor<128xf32>
tt.func @test_const_splat_addptr(%arg0 : !tt.ptr<f32>) -> tensor<128xf32> {
  %cst = arith.constant dense<512> : tensor<128xi32>
  %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
  %1 = tt.addptr %0, %cst : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
  %2 = tt.load %1 : tensor<128x!tt.ptr<f32>>
  tt.return %2 : tensor<128xf32>
}

// CHECK-LABEL:   tt.func @test_expand_dims(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !tt.ptr<f32>) -> tensor<1x128xf32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_3:.*]] = arith.constant 512 : i32
// CHECK:           %[[VAL_4:.*]] = tt.make_tensor_ptr %[[VAL_0]], {{\[}}%[[VAL_1]], %[[VAL_1]]], {{\[}}%[[VAL_1]], %[[VAL_1]]], {{\[}}%[[VAL_2]], %[[VAL_3]]] {order = array<i32>} : <tensor<1x128xf32>>
// CHECK:           %[[VAL_5:.*]] = tt.load %[[VAL_4]] : !tt.ptr<tensor<1x128xf32>>
// CHECK:           tt.return %[[VAL_5]] : tensor<1x128xf32>
tt.func @test_expand_dims(%arg0 : !tt.ptr<f32>) -> tensor<1x128xf32> {
  %cst = arith.constant dense<512> : tensor<128xi32>
  %0 = tt.expand_dims %cst {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
  %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x128x!tt.ptr<f32>>
  %2 = tt.addptr %1, %0 : tensor<1x128x!tt.ptr<f32>>, tensor<1x128xi32>
  %3 = tt.load %2 : tensor<1x128x!tt.ptr<f32>>
  tt.return %3 : tensor<1x128xf32>
}

// CHECK-LABEL:   tt.func @test_const_splat_addptr_2d(
// CHECK-SAME:                                        %[[VAL_0:.*]]: !tt.ptr<f32>) -> tensor<2x128xf32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_2:.*]] = arith.constant 512 : i32
// CHECK:           %[[VAL_3:.*]] = tt.make_tensor_ptr %[[VAL_0]], {{\[}}%[[VAL_1]], %[[VAL_1]]], {{\[}}%[[VAL_1]], %[[VAL_1]]], {{\[}}%[[VAL_2]], %[[VAL_2]]] {order = array<i32>} : <tensor<2x128xf32>>
// CHECK:           %[[VAL_4:.*]] = tt.load %[[VAL_3]] : !tt.ptr<tensor<2x128xf32>>
// CHECK:           tt.return %[[VAL_4]] : tensor<2x128xf32>
tt.func @test_const_splat_addptr_2d(%arg0 : !tt.ptr<f32>) -> tensor<2x128xf32> {
  %cst = arith.constant dense<512> : tensor<2x128xi32>
  %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x128x!tt.ptr<f32>>
  %1 = tt.addptr %0, %cst : tensor<2x128x!tt.ptr<f32>>, tensor<2x128xi32>
  %2 = tt.load %1 : tensor<2x128x!tt.ptr<f32>>
  tt.return %2 : tensor<2x128xf32>
}

// CHECK-LABEL:   tt.func @test_addptr_broadcast(
// CHECK-SAME:                                   %[[VAL_0:.*]]: !tt.ptr<f32>) -> tensor<2x128xf32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_3:.*]] = tt.make_tensor_ptr %[[VAL_0]], {{\[}}%[[VAL_1]], %[[VAL_1]]], {{\[}}%[[VAL_1]], %[[VAL_1]]], {{\[}}%[[VAL_2]], %[[VAL_2]]] {order = array<i32>} : <tensor<2x128xf32>>
// CHECK:           %[[VAL_4:.*]] = tt.load %[[VAL_3]] : !tt.ptr<tensor<2x128xf32>>
// CHECK:           tt.return %[[VAL_4]] : tensor<2x128xf32>
tt.func @test_addptr_broadcast(%arg0 : !tt.ptr<f32>) -> tensor<2x128xf32> {
  %cst = arith.constant dense<1> : tensor<1x128xi32>
  %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x128x!tt.ptr<f32>>
  %1 = tt.broadcast %cst : tensor<1x128xi32> -> tensor<2x128xi32>
  %2 = tt.addptr %0, %1 : tensor<2x128x!tt.ptr<f32>>, tensor<2x128xi32>
  %3 = tt.load %2 : tensor<2x128x!tt.ptr<f32>>
  tt.return %3 : tensor<2x128xf32>
}

// CHECK-LABEL:   tt.func @test_addptr_broadcast_rank(
// CHECK-SAME:                                   %[[VAL_0:.*]]: !tt.ptr<f32>) -> tensor<2x128xf32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_3:.*]] = tt.make_tensor_ptr %[[VAL_0]], {{\[}}%[[VAL_1]], %[[VAL_1]]], {{\[}}%[[VAL_1]], %[[VAL_1]]], {{\[}}%[[VAL_2]], %[[VAL_2]]] {order = array<i32>} : <tensor<2x128xf32>>
// CHECK:           %[[VAL_4:.*]] = tt.load %[[VAL_3]] : !tt.ptr<tensor<2x128xf32>>
// CHECK:           tt.return %[[VAL_4]] : tensor<2x128xf32>
tt.func @test_addptr_broadcast_rank(%arg0 : !tt.ptr<f32>) -> tensor<2x128xf32> {
  %cst = arith.constant dense<1> : tensor<128xi32>
  %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x128x!tt.ptr<f32>>
  %1 = tt.broadcast %cst : tensor<128xi32> -> tensor<2x128xi32>
  %2 = tt.addptr %0, %1 : tensor<2x128x!tt.ptr<f32>>, tensor<2x128xi32>
  %3 = tt.load %2 : tensor<2x128x!tt.ptr<f32>>
  tt.return %3 : tensor<2x128xf32>
}

// CHECK-LABEL:   tt.func @test_addptr_broadcast_rank_2(
// CHECK-SAME:                                   %[[VAL_0:.*]]: !tt.ptr<f32>) -> tensor<128x2x128xf32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_3:.*]] = tt.make_tensor_ptr %[[VAL_0]], {{\[}}%[[VAL_1]], %[[VAL_1]], %[[VAL_1]]], {{\[}}%[[VAL_1]], %[[VAL_1]], %[[VAL_1]]], {{\[}}%[[VAL_2]], %[[VAL_2]], %[[VAL_2]]] {order = array<i32>} : <tensor<128x2x128xf32>>
// CHECK:           %[[VAL_4:.*]] = tt.load %[[VAL_3]] : !tt.ptr<tensor<128x2x128xf32>>
// CHECK:           tt.return %[[VAL_4]] : tensor<128x2x128xf32>
tt.func @test_addptr_broadcast_rank_2(%arg0 : !tt.ptr<f32>) -> tensor<128x2x128xf32> {
  %cst = arith.constant dense<1> : tensor<128x128xi32>
  %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x2x128x!tt.ptr<f32>>
  %1 = tt.broadcast %cst : tensor<128x128xi32> -> tensor<128x2x128xi32>
  %2 = tt.addptr %0, %1 : tensor<128x2x128x!tt.ptr<f32>>, tensor<128x2x128xi32>
  %3 = tt.load %2 : tensor<128x2x128x!tt.ptr<f32>>
  tt.return %3 : tensor<128x2x128xf32>
}

// CHECK-LABEL:   tt.func @test_addptr_broadcast_rank_3(
// CHECK-SAME:                                   %[[VAL_0:.*]]: !tt.ptr<f32>) -> tensor<128x2x128xf32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_3:.*]] = tt.make_tensor_ptr %[[VAL_0]], {{\[}}%[[VAL_1]], %[[VAL_1]], %[[VAL_1]]], {{\[}}%[[VAL_1]], %[[VAL_1]], %[[VAL_1]]], {{\[}}%[[VAL_2]], %[[VAL_2]], %[[VAL_2]]] {order = array<i32>} : <tensor<128x2x128xf32>>
// CHECK:           %[[VAL_4:.*]] = tt.load %[[VAL_3]] : !tt.ptr<tensor<128x2x128xf32>>
// CHECK:           tt.return %[[VAL_4]] : tensor<128x2x128xf32>
tt.func @test_addptr_broadcast_rank_3(%arg0 : !tt.ptr<f32>) -> tensor<128x2x128xf32> {
  %cst = arith.constant dense<1> : tensor<128xi32>
  %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x2x128x!tt.ptr<f32>>
  %1 = tt.broadcast %cst : tensor<128xi32> -> tensor<128x2x128xi32>
  %2 = tt.addptr %0, %1 : tensor<128x2x128x!tt.ptr<f32>>, tensor<128x2x128xi32>
  %3 = tt.load %2 : tensor<128x2x128x!tt.ptr<f32>>
  tt.return %3 : tensor<128x2x128xf32>
}


// CHECK:         tt.func public @wrap_side_by_side_masked([[PARAM_0_:%.+]]: !tt.ptr<f32>, [[PARAM_1_:%.+]]: !tt.ptr<f32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32) {
// CHECK:       [[CST_6_i32:%.+]] = arith.constant 6 : i32
// CHECK:       [[CST_0_i64:%.+]] = arith.constant 0 : i64
// CHECK:       [[CST_0_i32:%.+]] = arith.constant 0 : i32
// CHECK:       [[CST_2_i32:%.+]] = arith.constant 2 : i32
// CHECK:       [[VAR_1_:%.+]] = arith.index_cast [[PARAM_3_]] : i32 to index
// CHECK:       [[VAR_2_:%.+]] = arith.index_cast [[VAR_1_]] : index to i64	
// CHECK:       [[VAR_3_:%.+]] = arith.muli [[PARAM_3_]], [[CST_2_i32]] : i32
// CHECK:       [[VAR_4_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK:       [[VAR_5_:%.+]] = arith.index_cast [[PARAM_4_]] : i32 to index
// CHECK:       [[VAR_6_:%.+]] = arith.index_cast [[VAR_5_]] : index to i64	
// CHECK:       [[VAR_7_:%.+]] = arith.muli [[PARAM_4_]], [[CST_6_i32]] : i32
// CHECK:       [[VAR_8_:%.+]] = arith.index_cast [[VAR_4_]] : index to i64	
// CHECK:       [[VAR_9_:%.+]] = arith.muli [[VAR_8_]], [[VAR_6_]] : i64
// CHECK:       [[VAR_10:%.+]] = tt.make_tensor_ptr [[PARAM_0_]], {{\[}}[[CST_0_i64]], [[VAR_9_]]], {{\[}}[[VAR_2_]], [[VAR_6_]]], {{\[}}[[VAR_3_]], [[VAR_7_]]] {order = array<i32>} : <tensor<4x4xf32>>
// CHECK:       [[VAR_11_:%.+]] = arith.index_cast [[PARAM_5_]] : i32 to index
// CHECK:       [[VAR_12_:%.+]] = arith.index_cast [[VAR_11_]] : index to i64	
// CHECK:       [[VAR_13_:%.+]] = arith.index_cast [[PARAM_6_]] : i32 to index
// CHECK:       [[VAR_14_:%.+]] = arith.index_cast [[VAR_13_]] : index to i64	
// CHECK:       [[VAR_15:%.+]] = tt.make_tensor_ptr [[PARAM_1_]], {{\[}}[[CST_0_i64]], [[CST_0_i64]]], {{\[}}[[VAR_12_]], [[VAR_14_]]], {{\[}}[[CST_0_i32]], [[CST_0_i32]]] {order = array<i32>} : <tensor<4x4xf32>>
// CHECK:       [[VAR_16:%.+]] = tt.load [[VAR_10]] : !tt.ptr<tensor<4x4xf32>>
// CHECK:	tt.store [[VAR_15]], [[VAR_16]] : !tt.ptr<tensor<4x4xf32>>
// CHECK:       tt.return
module {
tt.func public @wrap_side_by_side_masked(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) {
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
    %3 = tt.splat %arg2 : i32 -> tensor<4xi32>
    %4 = arith.remsi %2, %3 : tensor<4xi32>
    %5 = tt.expand_dims %1 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %6 = tt.splat %arg3 : i32 -> tensor<4x1xi32>
    %7 = arith.muli %5, %6 : tensor<4x1xi32>
    %8 = tt.expand_dims %4 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %9 = tt.splat %arg4 : i32 -> tensor<1x4xi32>
    %10 = arith.muli %8, %9 : tensor<1x4xi32>
    %11 = tt.broadcast %7 : tensor<4x1xi32> -> tensor<4x4xi32>
    %12 = tt.broadcast %10 : tensor<1x4xi32> -> tensor<4x4xi32>
    %13 = arith.addi %11, %12 : tensor<4x4xi32>
    %14 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
    %15 = tt.addptr %14, %13 : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    %16 = tt.expand_dims %0 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %17 = tt.splat %arg5 : i32 -> tensor<4x1xi32>
    %18 = arith.muli %17, %16 : tensor<4x1xi32>
    %19 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x1x!tt.ptr<f32>>
    %20 = tt.addptr %19, %18 : tensor<4x1x!tt.ptr<f32>>, tensor<4x1xi32>
    %21 = tt.expand_dims %0 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %22 = tt.splat %arg6 : i32 -> tensor<1x4xi32>
    %23 = arith.muli %22, %21 : tensor<1x4xi32>
    %24 = tt.broadcast %20 : tensor<4x1x!tt.ptr<f32>> -> tensor<4x4x!tt.ptr<f32>>
    %25 = tt.broadcast %23 : tensor<1x4xi32> -> tensor<4x4xi32>
    %26 = tt.addptr %24, %25 : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    %27 = arith.cmpi slt, %16, %cst_0 : tensor<4x1xi32>
    %28 = tt.broadcast %27 : tensor<4x1xi1> -> tensor<4x4xi1>
    %29 = arith.muli %arg3, %c4_i32 : i32
    %30 = tt.splat %29 : i32 -> tensor<4x4xi32>
    %31 = arith.muli %arg4, %c4_i32 : i32
    %32 = tt.splat %31 : i32 -> tensor<4x4xi32>
    %34 = tt.load %15 : tensor<4x4x!tt.ptr<f32>>
    tt.store %26, %34 : tensor<4x4x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK:       tt.func public @wrap_stacked_masked_loop([[PARAM_0_:%.+]]: !tt.ptr<f32>, [[PARAM_1_:%.+]]: !tt.ptr<f32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32) {
// CHECK:       [[CST_3_i32:%.+]] = arith.constant 3 : i32
// CHECK:       [[CST_2_i32:%.+]] = arith.constant 2 : i32
// CHECK:       [[CST_0_i64:%.+]] = arith.constant 0 : i64
// CHECK:       [[CST_0_i32:%.+]] = arith.constant 0 : i32
// CHECK:       [[VAR_0_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK:       [[VAR_1_:%.+]] = arith.index_cast [[PARAM_3_]] : i32 to index
// CHECK:       [[VAR_2_:%.+]] = arith.index_cast [[VAR_1_]] : index to i64	
// CHECK:       [[VAR_3_:%.+]] = arith.muli [[PARAM_3_]], [[CST_2_i32]] : i32
// CHECK:       [[VAR_4_:%.+]] = arith.index_cast [[VAR_0_]] : index to i64	
// CHECK:       [[VAR_5_:%.+]] = arith.muli [[VAR_4_]], [[VAR_2_]] : i64
// CHECK:       [[VAR_6_:%.+]] = arith.index_cast [[PARAM_4_]] : i32 to index
// CHECK:       [[VAR_7_:%.+]] = arith.index_cast [[VAR_6_]] : index to i64	
// CHECK:       [[VAR_8_:%.+]] = arith.muli [[PARAM_4_]], [[CST_3_i32]] : i32
// CHECK:       [[VAR_9:%.+]] = tt.make_tensor_ptr [[PARAM_0_]], {{\[}}[[VAR_5_]], [[CST_0_i64]]], {{\[}}[[VAR_2_]], [[VAR_7_]]], {{\[}}[[VAR_3_]], [[VAR_8_]]] {order = array<i32>} : <tensor<4x4xf32>>
// CHECK:       [[VAR_10_:%.+]] = arith.index_cast [[PARAM_5_]] : i32 to index
// CHECK:       [[VAR_11_:%.+]] = arith.index_cast [[VAR_10_]] : index to i64	
// CHECK:       [[VAR_12_:%.+]] = arith.index_cast [[PARAM_6_]] : i32 to index
// CHECK:       [[VAR_13_:%.+]] = arith.index_cast [[VAR_12_]] : index to i64	
// CHECK:       [[VAR_14:%.+]] = tt.make_tensor_ptr [[PARAM_1_]], {{\[}}[[CST_0_i64]], [[CST_0_i64]]], {{\[}}[[VAR_11_]], [[VAR_13_]]], {{\[}}[[CST_0_i32]], [[CST_0_i32]]] {order = array<i32>} : <tensor<4x4xf32>>
// CHECK:       [[VAR_15:%.+]] = tt.load [[VAR_9]] : !tt.ptr<tensor<4x4xf32>>
// CHECK:	tt.store [[VAR_14]], [[VAR_15]] : !tt.ptr<tensor<4x4xf32>>
// CHECK:       tt.return
module {
  tt.func public @wrap_stacked_masked_loop(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %cst_0 = arith.constant dense<3> : tensor<1x4xi32>
    %cst_1 = arith.constant dense<3> : tensor<4xi32>
    %cst_2 = arith.constant dense<2> : tensor<4xi32>
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %1 = arith.addi %0, %cst_2 : tensor<4xi32>
    %2 = tt.splat %arg2 : i32 -> tensor<4xi32>
    %3 = arith.remui %1, %2 : tensor<4xi32>
    %4 = arith.addi %0, %cst_1 : tensor<4xi32>
    %5 = tt.expand_dims %3 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %6 = tt.splat %arg3 : i32 -> tensor<4x1xi32>
    %7 = arith.muli %5, %6 : tensor<4x1xi32>
    %8 = tt.expand_dims %4 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %9 = tt.splat %arg4 : i32 -> tensor<1x4xi32>
    %10 = arith.muli %8, %9 : tensor<1x4xi32>
    %11 = tt.broadcast %7 : tensor<4x1xi32> -> tensor<4x4xi32>
    %12 = tt.broadcast %10 : tensor<1x4xi32> -> tensor<4x4xi32>
    %13 = arith.addi %11, %12 : tensor<4x4xi32>
    %14 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
    %15 = tt.addptr %14, %13 : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    %16 = tt.expand_dims %0 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %17 = tt.splat %arg5 : i32 -> tensor<4x1xi32>
    %18 = arith.muli %17, %16 : tensor<4x1xi32>
    %19 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x1x!tt.ptr<f32>>
    %20 = tt.addptr %19, %18 : tensor<4x1x!tt.ptr<f32>>, tensor<4x1xi32>
    %21 = tt.expand_dims %0 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %22 = tt.splat %arg6 : i32 -> tensor<1x4xi32>
    %23 = arith.muli %22, %21 : tensor<1x4xi32>
    %24 = tt.broadcast %20 : tensor<4x1x!tt.ptr<f32>> -> tensor<4x4x!tt.ptr<f32>>
    %25 = tt.broadcast %23 : tensor<1x4xi32> -> tensor<4x4xi32>
    %26 = tt.addptr %24, %25 : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    %27 = arith.cmpi slt, %21, %cst_0 : tensor<1x4xi32>
    %28 = tt.broadcast %27 : tensor<1x4xi1> -> tensor<4x4xi1>
    %29 = arith.muli %arg4, %c4_i32 : i32
    %30 = tt.splat %29 : i32 -> tensor<4x4xi32>
    %32 = tt.load %15 : tensor<4x4x!tt.ptr<f32>>
    tt.store %26, %32 : tensor<4x4x!tt.ptr<f32>>
    tt.return
  }
}
