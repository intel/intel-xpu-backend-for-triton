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

// CHECK-LABEL:   tt.func @test_addptr_load_with_mask(
// CHECK-SAME:                                         %[[VAL_0:.*]]: !tt.ptr<f32>,
// CHECK-SAME:                                         %[[VAL_1:.*]]: i32,
// CHECK-SAME:                                         %[[VAL_2:.*]]: tensor<128xi1>) -> tensor<128xf32> {
// CHECK:           %[[VAL_3:.*]] = tt.addptr
// CHECK:           %[[VAL_4:.*]] = tt.load %[[VAL_3]], %[[VAL_2]] cacheModifier = ca evictionPolicy = evict_first {isVolatile = true} : tensor<128x!tt.ptr<f32>>
// CHECK:           tt.return %[[VAL_4]] : tensor<128xf32>
tt.func @test_addptr_load_with_mask(%arg0 : !tt.ptr<f32>, %arg1: i32, %arg2: tensor<128xi1>) -> tensor<128xf32> {
  %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
  %1 = tt.splat %arg1 : i32 -> tensor<128xi32>
  %2 = tt.addptr %0, %1 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
  %3 = tt.load %2, %arg2 cacheModifier = ca evictionPolicy = evict_first {isVolatile = true} : tensor<128x!tt.ptr<f32>>
  tt.return %3 : tensor<128xf32>
}

// CHECK-LABEL:   tt.func @test_addptr_splat_splat_i32(
// CHECK-SAME:                                         %[[VAL_0:.*]]: !tt.ptr<f32>,
// CHECK-SAME:                                         %[[VAL_1:.*]]: i32) -> tensor<128xf32> {
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_4:.*]] = tt.make_tensor_ptr %[[VAL_0]], {{\[}}%[[VAL_3]]], {{\[}}%[[VAL_3]]], {{\[}}%[[VAL_1]]] {order = array<i32>} : <tensor<128xf32>>
// CHECK:           %[[VAL_5:.*]] = tt.load %[[VAL_4]] cacheModifier = ca evictionPolicy = evict_first {isVolatile = true} : !tt.ptr<tensor<128xf32>>
// CHECK:           tt.return %[[VAL_5]] : tensor<128xf32>
tt.func @test_addptr_splat_splat_i32(%arg0 : !tt.ptr<f32>, %arg1: i32) -> tensor<128xf32> {
  %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
  %1 = tt.splat %arg1 : i32 -> tensor<128xi32>
  %2 = tt.addptr %0, %1 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
  %3 = tt.load %2 cacheModifier = ca evictionPolicy = evict_first {isVolatile = true} : tensor<128x!tt.ptr<f32>>
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
// CHECK-SAME:                                        %[[VAL_1:.*]]: i64) -> tensor<2x128xf32> {
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_6:.*]] = arith.index_cast %[[VAL_1]] : i64 to index
// CHECK:           %[[VAL_7:.*]] = arith.index_cast %[[VAL_6]] : index to i32
// CHECK:           %[[VAL_8:.*]] = tt.make_tensor_ptr %[[VAL_0]], {{\[}}%[[VAL_4]], %[[VAL_4]]], {{\[}}%[[VAL_4]], %[[VAL_4]]], {{\[}}%[[VAL_7]], %[[VAL_5]]] {order = array<i32>} : <tensor<2x128xf32>>
// CHECK:           %[[VAL_9:.*]] = tt.load %[[VAL_8]] : !tt.ptr<tensor<2x128xf32>>
// CHECK:           tt.return %[[VAL_9]] : tensor<2x128xf32>
tt.func @test_addptr_splat_splat_2d(%arg0 : !tt.ptr<f32>, %arg1: i64) -> tensor<2x128xf32> {
  %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x128x!tt.ptr<f32>>
  %1 = tt.splat %arg1 : i64 -> tensor<2x128xi64>
  %2 = tt.addptr %0, %1 : tensor<2x128x!tt.ptr<f32>>, tensor<2x128xi64>
  %3 = tt.load %2 : tensor<2x128x!tt.ptr<f32>>
  tt.return %3 : tensor<2x128xf32>
}

// CHECK-LABEL:   tt.func @test_addptr_splat_splat_2d_store(
// CHECK-SAME:                                              %[[VAL_0:.*]]: !tt.ptr<f32>,
// CHECK-SAME:                                              %[[VAL_1:.*]]: i64,
// CHECK-SAME:                                              %[[VAL_2:.*]]: tensor<2x128xf32>) {
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_5:.*]] = arith.index_cast %[[VAL_1]] : i64 to index
// CHECK:           %[[VAL_6:.*]] = arith.index_cast %[[VAL_5]] : index to i32
// CHECK:           %[[VAL_7:.*]] = tt.make_tensor_ptr %[[VAL_0]], {{\[}}%[[VAL_3]], %[[VAL_3]]], {{\[}}%[[VAL_3]], %[[VAL_3]]], {{\[}}%[[VAL_6]], %[[VAL_4]]] {order = array<i32>} : <tensor<2x128xf32>>
// CHECK:           tt.store %[[VAL_7]], %[[VAL_2]] : !tt.ptr<tensor<2x128xf32>>
tt.func @test_addptr_splat_splat_2d_store(%arg0 : !tt.ptr<f32>, %arg1: i64, %arg2: tensor<2x128xf32>) {
  %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x128x!tt.ptr<f32>>
  %1 = tt.splat %arg1 : i64 -> tensor<2x128xi64>
  %2 = tt.addptr %0, %1 : tensor<2x128x!tt.ptr<f32>>, tensor<2x128xi64>
  tt.store %2, %arg2 : tensor<2x128x!tt.ptr<f32>>
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
  %cst = arith.constant dense<1> : tensor<1x128xi32>
  %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x128x!tt.ptr<f32>>
  %1 = tt.broadcast %cst : tensor<1x128xi32> -> tensor<2x128xi32>
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
  %cst = arith.constant dense<1> : tensor<128x1x128xi32>
  %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x2x128x!tt.ptr<f32>>
  %1 = tt.broadcast %cst : tensor<128x1x128xi32> -> tensor<128x2x128xi32>
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
  %cst = arith.constant dense<1> : tensor<128x1x1xi32>
  %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x2x128x!tt.ptr<f32>>
  %1 = tt.broadcast %cst : tensor<128x1x1xi32> -> tensor<128x2x128xi32>
  %2 = tt.addptr %0, %1 : tensor<128x2x128x!tt.ptr<f32>>, tensor<128x2x128xi32>
  %3 = tt.load %2 : tensor<128x2x128x!tt.ptr<f32>>
  tt.return %3 : tensor<128x2x128xf32>
}


// CHECK:         tt.func public @wrap_side_by_side_masked([[PARAM_0_:%.+]]: !tt.ptr<f32>, [[PARAM_1_:%.+]]: !tt.ptr<f32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32) {
// CHECK-DAG:   [[CST_6_i32:%.+]] = arith.constant 6 : i32
// CHECK-DAG:   [[CST_0_i64:%.+]] = arith.constant 0 : i64
// CHECK-DAG:   [[CST_0_i32:%.+]] = arith.constant 0 : i32
// CHECK-DAG:   [[CST_2_i32:%.+]] = arith.constant 2 : i32
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
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c4_i32 = arith.constant 4 : i32
    %cst_0 = arith.constant dense<2> : tensor<4x1xi32>
    %cst_1 = arith.constant dense<6> : tensor<4xi32>
    %cst_2 = arith.constant dense<2> : tensor<4xi32>
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


// CHECK:         tt.func @test_addptr_for_accumulation([[PARAM_0_:%.+]]: !tt.ptr<bf16>, [[PARAM_1_:%.+]]: !tt.ptr<bf16>, [[PARAM_2_:%.+]]: !tt.ptr<bf16>, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32) {
// CHECK:       [[CST_3_i32:%.+]] = arith.constant 3 : i32
// CHECK:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK:       [[CST_12_:%.+]] = arith.constant 12 : index
// CHECK:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:       [[CST_1_i64:%.+]] = arith.constant 1 : i64
// CHECK:       [[CST_0_i64:%.+]] = arith.constant 0 : i64
// CHECK:       [[CST_0_i32:%.+]] = arith.constant 0 : i32
// CHECK:       [[CST_5_i64:%.+]] = arith.constant 5 : i64
// CHECK:       [[VAR_1_:%.+]] = tt.make_tensor_ptr [[PARAM_0_]], {{\[}}[[CST_0_i64]], [[CST_0_i64]]], {{\[}}[[CST_1_i64]], [[CST_5_i64]]], {{\[}}[[PARAM_3_]], [[CST_0_i32]]] {order = array<i32>} : <tensor<4x256xbf16>>
// CHECK:       [[VAR_2_:%.+]] = tt.load [[VAR_1_]] : !tt.ptr<tensor<4x256xbf16>>
// CHECK:       [[VAR_4_:%.+]]:2 = scf.for [[VAR_arg5_:%.+]] = [[CST_0_]] to [[CST_12_]] step [[CST_3_]] iter_args([[VAR_arg6_:%.+]] = [[VAR_2_]], [[VAR_arg7_:%.+]] = [[PARAM_3_]]) -> (tensor<4x256xbf16>, i32) {
// CHECK:         [[VAR_7_:%.+]] = tt.make_tensor_ptr [[PARAM_1_]], {{\[}}[[CST_0_i64]], [[CST_0_i64]]], {{\[}}[[CST_1_i64]], [[CST_5_i64]]], {{\[}}[[VAR_arg7_]], [[CST_0_i32]]] {order = array<i32>} : <tensor<4x256xbf16>>
// CHECK:         [[VAR_8_:%.+]] = tt.load [[VAR_7_]] : !tt.ptr<tensor<4x256xbf16>>
// CHECK:         [[VAR_9_:%.+]] = arith.addf [[VAR_arg6_]], [[VAR_8_]] : tensor<4x256xbf16>
// CHECK:         [[VAR_10_:%.+]] = arith.addi [[VAR_arg7_]], [[CST_3_i32]] : i32
// CHECK:         scf.yield [[VAR_9_]], [[VAR_10_]] : tensor<4x256xbf16>, i32
// CHECK:       }
// CHECK:       [[VAR_6_:%.+]] = tt.make_tensor_ptr [[PARAM_2_]], {{\[}}[[CST_0_i64]], [[CST_0_i64]]], {{\[}}[[CST_1_i64]], [[CST_5_i64]]], {{\[}}[[PARAM_3_]], [[CST_0_i32]]] {order = array<i32>} : <tensor<4x256xbf16>>
// CHECK:       tt.store [[VAR_6_]], [[VAR_4_]]#0 : !tt.ptr<tensor<4x256xbf16>>
// CHECK:       tt.return
// CHECK:       }
module {
  tt.func @test_addptr_for_accumulation(
    %arg0 : !tt.ptr<bf16>,
    %arg1 : !tt.ptr<bf16>,
    %arg2 : !tt.ptr<bf16>,
    %arg3 : i32,
    %arg4 : i32
  )
  {
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32}:tensor<4xi32>
    // offset = 0, size = 4, stride = 1
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    // offset = [0,0], size = [4,1], stride = [1,0]
    %2 = tt.broadcast %1 : tensor<4x1xi32> -> tensor<4x256xi32>
    // offset = [0,0], size = [4,256], stride = [1,0]
    %arg3splat = tt.splat %arg3 : i32 -> tensor<4x256xi32>
    %offset3 = arith.addi %2, %arg3splat : tensor<4x256xi32>
    // offset = [%arg3,0], size = [4,256], stride = [1,0]
    %3 = tt.make_range {end = 256 : i32, start = 0 : i32}:tensor<256xi32>
    // offset = 0, size = 256, stride = 1
    %4 = tt.expand_dims %3 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    // offset = [0,0], size = [1,256], stride = [0,1]
    %5 = tt.broadcast %4 : tensor<1x256xi32> -> tensor<4x256xi32>
    // offset = [0,0], size = [4,256], stride = [0,1]
    %c5 = arith.constant 5 : i32
    %splat6 = tt.splat %c5 : i32 -> tensor<4x256xi32>
    // scalar = 5
    %scale5 = arith.muli %5, %splat6 : tensor<4x256xi32> // Why we never called the conversion function for the inputs here?
    // offset = [0,0], size = [4,256], stride = [0,5]
    %7 = arith.addi %offset3, %scale5: tensor<4x256xi32> // Why we never called the conversion function for the inputs here?
    // offset = [%arg3, 0], size = [4, 256], stride = [1, 5]
    %8 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<4x256x!tt.ptr<bf16>> // Why is the input unknown
    %9 = tt.addptr %8, %7 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
    // source: %arg0, offset = [%arg3, 0], size = [4, 256], stride = [1, 5]
    %19 = tt.load %9 : tensor<4x256x!tt.ptr<bf16>> // this will be replaced with a memref.copy
    %11 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<4x256x!tt.ptr<bf16>>
    %12 = tt.addptr %11, %7 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
    // source: %arg1, offset = [%arg3, 0], size = [4, 256], stride = [1, 5]
    %c0 = arith.constant 0 : index
    %c12 = arith.constant 12 : index
    %c3 = arith.constant 3 : index
    %i_c3 = arith.constant 3 : i32
    %sum_out, %_ptr = scf.for %i = %c0 to %c12 step %c3 iter_args(%sum_iter = %19, %ptr_iter = %12) -> (tensor<4x256xbf16>, tensor<4x256x!tt.ptr<bf16>>) {
        %20 = tt.load %ptr_iter : tensor<4x256x!tt.ptr<bf16>>
        %sum = arith.addf %sum_iter, %20 : tensor<4x256xbf16>
        // pointer updates
        %17 = tt.splat %i_c3 : i32 -> tensor<4x256xi32>
        // offset: [3, 0], size = [4, 256], stride [0, 0]
        %ptr = tt.addptr %ptr_iter, %17 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
        // source: %arg1, offset = [%arg3+%i, 0], size = [4, 256], stride = [1, 5]
        scf.yield %sum, %ptr : tensor<4x256xbf16>, tensor<4x256x!tt.ptr<bf16>>
    }
    %15 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<4x256x!tt.ptr<bf16>>
    %16 = tt.addptr %15, %7 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
    // source: %arg2, offset = [%arg3, 0], size = [4, 256], stride = [1, 5]
    tt.store %16, %sum_out : tensor<4x256x!tt.ptr<bf16>>
    tt.return
  }
}


// CHECK:       tt.func public @wrap_stacked_masked_loop([[PARAM_0_:%.+]]: !tt.ptr<f32>, [[PARAM_1_:%.+]]: !tt.ptr<f32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32) {
// CHECK-DAG:   [[CST_3_i32:%.+]] = arith.constant 3 : i32
// CHECK-DAG:   [[CST_2_i32:%.+]] = arith.constant 2 : i32
// CHECK-DAG:   [[CST_0_i64:%.+]] = arith.constant 0 : i64
// CHECK-DAG:   [[CST_0_i32:%.+]] = arith.constant 0 : i32
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
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c4_i32 = arith.constant 4 : i32
    %cst_0 = arith.constant dense<3> : tensor<1x4xi32>
    %cst_1 = arith.constant dense<3> : tensor<4xi32>
    %cst_2 = arith.constant dense<2> : tensor<4xi32>
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


// CHECK:         tt.func @test_addptr_for_more_init_args([[PARAM_0_:%.+]]: !tt.ptr<bf16>, [[PARAM_1_:%.+]]: !tt.ptr<bf16>) {
// CHECK:       [[CST_3_i32:%.+]] = arith.constant 3 : i32
// CHECK:       [[CST_0_i64:%.+]] = arith.constant 0 : i64
// CHECK:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK:       [[CST_12_:%.+]] = arith.constant 12 : index
// CHECK:       [[CST_1024_i32:%.+]] = arith.constant 1024 : i32
// CHECK:       [[CST_1_i64:%.+]] = arith.constant 1 : i64
// CHECK:       [[VAR_0_:%.+]]:5 = scf.for [[VAR_arg2_:%.+]] = [[CST_0_]] to [[CST_12_]] step [[CST_3_]] iter_args([[VAR_arg3_:%.+]] = [[CST_1_]], [[VAR_arg4_:%.+]] = [[CST_2_]], [[VAR_arg5_:%.+]] = [[CST_3_]], [[VAR_arg6_:%.+]] = [[CST_1024_i32]], [[VAR_arg7_:%.+]] = [[CST_1024_i32]]) -> (index, index, index, i32, i32) {
// CHECK:           [[VAR_1_:%.+]] = tt.make_tensor_ptr [[PARAM_1_]], {{\[}}[[CST_0_i64]]], {{\[}}[[CST_1_i64]]], {{\[}}[[VAR_arg7_]]] {order = array<i32>} : <tensor<256xbf16>>
// CHECK:           [[VAR_2_:%.+]] = tt.make_tensor_ptr [[PARAM_0_]], {{\[}}[[CST_0_i64]]], {{\[}}[[CST_1_i64]]], {{\[}}[[VAR_arg6_]]] {order = array<i32>} : <tensor<256xbf16>>
// CHECK:           [[VAR_3_:%.+]] = tt.load [[VAR_2_]] : !tt.ptr<tensor<256xbf16>>
// CHECK:           tt.store [[VAR_1_]], [[VAR_3_]] : !tt.ptr<tensor<256xbf16>>
// CHECK:           [[VAR_4_:%.+]] = arith.addi [[VAR_arg6_]], [[CST_3_i32]] : i32
// CHECK:           [[VAR_5_:%.+]] = arith.addi [[VAR_arg3_]], [[CST_3_]] : index
// CHECK:           [[VAR_6_:%.+]] = arith.addi [[VAR_arg4_]], [[CST_3_]] : index
// CHECK:           [[VAR_7_:%.+]] = arith.addi [[VAR_arg5_]], [[CST_3_]] : index
// CHECK:           [[VAR_8_:%.+]] = arith.addi [[VAR_5_]], [[VAR_6_]] : index
// CHECK:           [[VAR_9_:%.+]] = arith.addi [[VAR_8_]], [[VAR_7_]] : index
// CHECK:           [[VAR_10_:%.+]] = arith.index_cast [[VAR_9_]] : index to i32
// CHECK:           [[VAR_11_:%.+]] = arith.addi [[VAR_arg7_]], [[VAR_10_]] : i32
// CHECK:           scf.yield [[VAR_5_]], [[VAR_6_]], [[VAR_7_]], [[VAR_4_]], [[VAR_11_]] : index, index, index, i32, i32
// CHECK:       }
// CHECK:       tt.return
// CHECK:       }
module {
  tt.func @test_addptr_for_more_init_args(
    %arg0 : !tt.ptr<bf16>,
    %arg1 : !tt.ptr<bf16>
  )
  {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c12 = arith.constant 12 : index
    %0 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<256x!tt.ptr<bf16>>
    %1 = tt.make_range {end = 1280 : i32, start = 1024 : i32}:tensor<256xi32>
    %2 = tt.addptr %0, %1 : tensor<256x!tt.ptr<bf16>>, tensor<256xi32>
    %3 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<256x!tt.ptr<bf16>>
    %4 = tt.addptr %3, %1 : tensor<256x!tt.ptr<bf16>>, tensor<256xi32>
    %_arg2, %_ptr_ld, %_arg3, %_ptr_st, %_arg4 = scf.for %i = %c0 to %c12 step %c3 iter_args(%arg2 = %c1, %ptr_ld = %2, %arg3 = %c2, %ptr_st = %4, %arg4 = %c3) -> (index, tensor<256x!tt.ptr<bf16>>, index, tensor<256x!tt.ptr<bf16>>, index) {
        %5 = tt.load %ptr_ld {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<256x!tt.ptr<bf16>>
        tt.store %ptr_st, %5 : tensor<256x!tt.ptr<bf16>>
        %cast3 = arith.index_cast %c3 : index to i32
        %6 = tt.splat %cast3 : i32 -> tensor<256xi32>
        %ptr_ld_iter = tt.addptr %ptr_ld, %6 : tensor<256x!tt.ptr<bf16>>, tensor<256xi32>
        %arg2_iter = arith.addi %arg2, %c3 : index
        %arg3_iter = arith.addi %arg3, %c3 : index
        %arg4_iter = arith.addi %arg4, %c3 : index
        %7 = arith.addi %arg2_iter, %arg3_iter : index
        %8 = arith.addi %7, %arg4_iter : index
        %cast8 = arith.index_cast %8 : index to i32
        %9 = tt.splat %cast8 : i32 -> tensor<256xi32>
        %ptr_st_iter = tt.addptr %ptr_st, %9 : tensor<256x!tt.ptr<bf16>>, tensor<256xi32>
        scf.yield %arg2_iter, %ptr_ld_iter, %arg3_iter, %ptr_st_iter, %arg4_iter : index, tensor<256x!tt.ptr<bf16>>, index, tensor<256x!tt.ptr<bf16>>, index
    }
    tt.return
  }
}


// CHECK:         tt.func @test_addptr_for_used_after_update([[PARAM_0_:%.+]]: !tt.ptr<bf16>) {
// CHECK:       [[CST_3_i32:%.+]] = arith.constant 3 : i32
// CHECK:       [[CST_0_i64:%.+]] = arith.constant 0 : i64
// CHECK:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:       [[CST_12_:%.+]] = arith.constant 12 : index
// CHECK:       [[CST_1024_i32:%.+]] = arith.constant 1024 : i32
// CHECK:       [[CST_1_i64:%.+]] = arith.constant 1 : i64
// CHECK:       [[VAR_0_:%.+]] = scf.for [[VAR_arg1_:%.+]] = [[CST_0_]] to [[CST_12_]] step [[CST_3_]] iter_args([[VAR_arg2_:%.+]] = [[CST_1024_i32]]) -> (i32) {
// CHECK:         [[VAR_1_:%.+]] = arith.addi [[VAR_arg2_]], [[CST_3_i32]] : i32
// CHECK:         [[VAR_2_:%.+]] = tt.make_tensor_ptr [[PARAM_0_]], {{\[}}[[CST_0_i64]]], {{\[}}[[CST_1_i64]]], {{\[}}[[VAR_1_]]] {order = array<i32>} : <tensor<256xbf16>>
// CHECK:         [[VAR_3_:%.+]] = tt.load [[VAR_2_]] : !tt.ptr<tensor<256xbf16>>
// CHECK:         tt.store [[VAR_2_]], [[VAR_3_]] : !tt.ptr<tensor<256xbf16>>
// CHECK:         scf.yield [[VAR_1_]] : i32
// CHECK:         }
// CHECK:         tt.return
// CHECK:      }
module {
  tt.func @test_addptr_for_used_after_update(
    %arg0 : !tt.ptr<bf16>
  )
  {
    %c0 = arith.constant 0 : index
    %c12 = arith.constant 12 : index
    %c3 = arith.constant 3 : index
    %i_c3 = arith.constant 3 : i32
    %0 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<256x!tt.ptr<bf16>>
    %1 = tt.make_range {end = 1280 : i32, start = 1024 : i32}:tensor<256xi32>
    %2 = tt.addptr %0, %1 : tensor<256x!tt.ptr<bf16>>, tensor<256xi32>
    %_ptr = scf.for %i = %c0 to %c12 step %c3 iter_args(%ptr = %2) -> (tensor<256x!tt.ptr<bf16>>) {
        %4 = tt.splat %i_c3 : i32 -> tensor<256xi32>
        %ptr_iter = tt.addptr %ptr, %4 : tensor<256x!tt.ptr<bf16>>, tensor<256xi32>
        %3 = tt.load %ptr_iter {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<256x!tt.ptr<bf16>>
        tt.store %ptr_iter, %3 : tensor<256x!tt.ptr<bf16>>
        scf.yield %ptr_iter : tensor<256x!tt.ptr<bf16>>
    }
    tt.return
  }
}


// CHECK:       tt.func @test_addptr_for_used_before_update([[PARAM_0_:%.+]]: !tt.ptr<bf16>) {
// CHECK:       [[CST_3_i32:%.+]] = arith.constant 3 : i32
// CHECK:       [[CST_0_i64:%.+]] = arith.constant 0 : i64
// CHECK:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:       [[CST_12_:%.+]] = arith.constant 12 : index
// CHECK:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK:       [[CST_1024_i32:%.+]] = arith.constant 1024 : i32
// CHECK:       [[CST_1_i64:%.+]] = arith.constant 1 : i64
// CHECK:       [[VAR_0_:%.+]] = scf.for [[VAR_arg1_:%.+]] = [[CST_0_]] to [[CST_12_]] step [[CST_3_]] iter_args([[VAR_arg2_:%.+]] = [[CST_1024_i32]]) -> (i32) {

// CHECK:           [[VAR_2_:%.+]] = tt.make_tensor_ptr [[PARAM_0_]], {{\[}}[[CST_0_i64]]], {{\[}}[[CST_1_i64]]], {{\[}}[[VAR_arg2_]]] {order = array<i32>} : <tensor<256xbf16>>
// CHECK:           [[VAR_3_:%.+]] = tt.load [[VAR_2_]] : !tt.ptr<tensor<256xbf16>>
// CHECK:           tt.store [[VAR_2_]], [[VAR_3_]] : !tt.ptr<tensor<256xbf16>>
// CHECK:           [[VAR_3_:%.+]] = arith.addi [[VAR_arg2_]], [[CST_3_i32]] : i32
// CHECK:           scf.yield [[VAR_3_]] : i32
// CHECK:       }
// CHECK:       tt.return
// CHECK:       }
module {
  tt.func @test_addptr_for_used_before_update(
    %arg0 : !tt.ptr<bf16>
  )
  {
    %c0 = arith.constant 0 : index
    %c12 = arith.constant 12 : index
    %c3 = arith.constant 3 : index
    %i_c3 = arith.constant 3 : i32
    %0 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<256x!tt.ptr<bf16>>
    %1 = tt.make_range {end = 1280 : i32, start = 1024 : i32}:tensor<256xi32>
    %2 = tt.addptr %0, %1 : tensor<256x!tt.ptr<bf16>>, tensor<256xi32>
    %_ptr2 = scf.for %i = %c0 to %c12 step %c3 iter_args(%ptr = %2) -> (tensor<256x!tt.ptr<bf16>>) {
        %3 = tt.load %ptr {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<256x!tt.ptr<bf16>>
        tt.store %ptr, %3 : tensor<256x!tt.ptr<bf16>>
        %4 = tt.splat %i_c3 : i32 -> tensor<256xi32>
        %ptr_iter = tt.addptr %ptr, %4 : tensor<256x!tt.ptr<bf16>>, tensor<256xi32>
        scf.yield %ptr_iter : tensor<256x!tt.ptr<bf16>>
    }
    tt.return
  }
}

// `triton::ExpandDims` ops on tensor of pointers are currently not supported in for loops.
// Consequently, the pass should fail cleanly.
// CHECK:       tt.func @test_fail_addptr_for_expand_ptr([[PARAM_0_:%.+]]: !tt.ptr<bf16>) {
// CHECK-NOT:       tt.make_tensor_ptr
module {
  tt.func @test_fail_addptr_for_expand_ptr(
    %arg0 : !tt.ptr<bf16>
  )
  {
    %c0 = arith.constant 0 : index
    %c12 = arith.constant 12 : index
    %c3 = arith.constant 3 : index
    %i_c3 = arith.constant 3 : i32
    %0 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<256x!tt.ptr<bf16>>
    %1 = tt.make_range {end = 1280 : i32, start = 1024 : i32}:tensor<256xi32>
    %2 = tt.addptr %0, %1 : tensor<256x!tt.ptr<bf16>>, tensor<256xi32>
    %_ptr = scf.for %i = %c0 to %c12 step %c3 iter_args(%ptr = %2) -> (tensor<256x!tt.ptr<bf16>>) {
      %6 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
      %7 = tt.expand_dims %6 {axis = 1 : i32} : tensor<256xi32> -> tensor<256x1xi32>
      %8 = tt.broadcast %7 : tensor<256x1xi32> -> tensor<256x256xi32>
      %9 = tt.make_range {end = 512 : i32, start = 256 : i32} : tensor<256xi32>
      %10 = tt.expand_dims %9 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
      %11 = tt.broadcast %10 : tensor<1x256xi32> -> tensor<256x256xi32>
      %12 = arith.addi %8, %11 : tensor<256x256xi32>
      %13 = tt.expand_dims %ptr {axis = 1 : i32} : tensor<256x!tt.ptr<bf16>> -> tensor<256x1x!tt.ptr<bf16>>
      %14 = tt.broadcast %13 : tensor<256x1x!tt.ptr<bf16>> -> tensor<256x256x!tt.ptr<bf16>>
      %15 = tt.addptr %14, %12 : tensor<256x256x!tt.ptr<bf16>>, tensor<256x256xi32>
      %16 = tt.load %15 {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<256x256x!tt.ptr<bf16>>
      tt.store %15, %16 : tensor<256x256x!tt.ptr<bf16>>
      %17 = tt.splat %i_c3 : i32 -> tensor<256xi32>
      %ptr_iter = tt.addptr %ptr, %17 : tensor<256x!tt.ptr<bf16>>, tensor<256xi32>
      scf.yield %ptr_iter : tensor<256x!tt.ptr<bf16>>
    }
    tt.return
  }
}
