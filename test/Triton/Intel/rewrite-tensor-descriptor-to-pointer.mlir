// RUN: triton-opt %s --triton-intel-rewrite-tensor-descriptor-to-pointer --canonicalize --cse --split-input-file | FileCheck %s

module {
  tt.func public @load(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32, %arg2: i32) -> (tensor<128x128xf32>) {
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c1_i64, %c256_i64] {order = array<i32: 0>} : <f32>, <tensor<128x128xf32>>
    %3 = tt.descriptor_load %0[%arg1, %arg2] : !tt.tensordesc<tensor<128x128xf32>> -> tensor<128x128xf32>
    tt.return %3 : tensor<128x128xf32>
  }
}

// CHECK-LABEL: @load
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor
// CHECK: tt.descriptor_load [[DESC]]

// -----

module {
  tt.func public @store(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32, %arg2: i32, %arg3: tensor<128x128xf32>) {
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c1_i64, %c256_i64] {order = array<i32: 0>} : <f32>, <tensor<128x128xf32>>
    tt.descriptor_store %0[%arg1, %arg2], %arg3 : !tt.tensordesc<tensor<128x128xf32>>, tensor<128x128xf32>
    tt.return
  }
}

// CHECK-LABEL: @store
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor
// CHECK: tt.descriptor_store [[DESC]]

// -----

module {
  tt.func public @callee(%tensordesc: !tt.tensordesc<tensor<128x128xf32>>) -> !tt.tensordesc<tensor<128x128xf32>> {
    tt.return %tensordesc : !tt.tensordesc<tensor<128x128xf32>>
  }

  tt.func public @caller(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %c1_i64 = arith.constant 1 : i64
    %c256_i32 = arith.constant 256 : i32
    %c256_i64 = arith.constant 256 : i64
    %0 = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c256_i64, %c1_i64] {order = array<i32: 0>} : <f32>, <tensor<128x128xf32>>
    %1 = tt.call @callee(%0) : (!tt.tensordesc<tensor<128x128xf32>>) -> !tt.tensordesc<tensor<128x128xf32>>
    tt.return
  }
}

// CHECK-LABEL: @callee
// CHECK-SAME: %[[PTR:[^:]*]]
// CHECK-SAME: %[[SHAPE0:[^:]*]]
// CHECK-SAME: %[[SHAPE1:[^:]*]]
// CHECK-SAME: %[[STRIDE0:[^:]*]]
// CHECK-SAME: %[[STRIDE1:[^:]*]]
// CHECK-SAME: %[[PAD:[^:]*]]
// CHECK-SAME: %[[ROUND:[^:]*]]
// CHECK-NEXT: tt.return %[[PTR]], %[[SHAPE0]], %[[SHAPE1]], %[[STRIDE0]], %[[STRIDE1]], %[[PAD]], %[[ROUND]]

// CHECK-LABEL: @caller
// CHECK-SAME: %[[PTR:[^:]*]]
// CHECK-DAG: %[[c1:.*]] = arith.constant 1 : i64
// CHECK-DAG: %[[c256:.*]] = arith.constant 256 : i64
// CHECK: %{{.*}}:7 = tt.call @callee(%[[PTR]], %[[c256]], %[[c256]], %[[c256]], %[[c1]], %false, %false)
// CHECK-SAME -> (!tt.ptr<f32>, i64, i64, i64, i64, i1, i1)

// -----

module {
  tt.func public @arg_attr(%arg0: !tt.tensordesc<tensor<128x128xf32>>, %arg1: i32 {tt.divisibility = 16 : i32}) {
    tt.return
  }
}

// CHECK-LABEL: @arg_attr
// CHECK-SAME: %arg7: i32 {tt.divisibility = 16 : i32}) {

// -----

module {
  tt.func public @gather(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> (tensor<32x128xf32>) {
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c1_i64, %c256_i64] {order = array<i32: 0>} : <f32>, <tensor<1x128xf32>>
    %cst = arith.constant dense<1> : tensor<32xi32>
    %3 = tt.descriptor_gather %0[%cst, %c0_i32] : (!tt.tensordesc<tensor<1x128xf32>>, tensor<32xi32>, i32) -> tensor<32x128xf32>
    tt.return %3 : tensor<32x128xf32>
  }
}

// CHECK-LABEL: @gather
// CHECK-SAME: %[[ARG0:[^:]*]]
// CHECK-DAG: %[[CST:.*]] = arith.constant dense<0> : tensor<1x128xi64>
// CHECK-DAG: %[[CST_0:.*]] = arith.constant dense<256> : tensor<1x128xi64>
// CHECK-DAG: %[[CST_1:.*]] = arith.constant dense<1> : tensor<32x128xi64>
// CHECK-DAG: %[[CST_2:.*]] = arith.constant dense<0.000000e+00> : tensor<32x128xf32>

// CHECK-DAG: %[[VAL0:.*]] = tt.make_range {end = 128 : i32, start = 0 : i32}
// CHECK-DAG: %[[VAL1:.*]] = arith.extsi %[[VAL0]] :
// CHECK-DAG: %[[VAL2:.*]] = tt.expand_dims %[[VAL1]] {axis = 0 : i32}
// CHECK-DAG: %[[VAL3:.*]] = tt.splat %[[ARG0]] :
// CHECK-DAG: %[[VAL4:.*]] = tt.addptr %[[VAL3]], %[[CST_1]] :
// CHECK-DAG: %[[VAL5:.*]] = arith.muli %[[VAL2]], %[[CST_0]] :
// CHECK-DAG: %[[VAL6:.*]] = tt.broadcast %[[VAL5]] : tensor<1x128xi64> -> tensor<32x128xi64>
// CHECK-DAG: %[[VAL7:.*]] = tt.addptr %[[VAL4]], %[[VAL6]] :

// CHECK-DAG: %[[VAL8:.*]] = arith.cmpi sge, %[[VAL2]], %[[CST]]
// CHECK-DAG: %[[VAL9:.*]] = arith.cmpi slt, %[[VAL2]], %[[CST_0]]
// CHECK-DAG: %[[VAL10:.*]] = arith.andi %[[VAL8]], %[[VAL9]]
// CHECK-DAG: %[[VAL11:.*]] = tt.broadcast %[[VAL10]] : tensor<1x128xi1> -> tensor<32x128xi1>

// CHECK-DAG: %[[VAL12:.*]] = tt.load %[[VAL7]], %[[VAL11]], %[[CST_2]]
// CHECK: tt.return %[[VAL12]] :

// -----

module {
  tt.func public @multi_users(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32, %arg2: i32, %arg3: tensor<1x128xf32>) {
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c1_i64, %c256_i64] {order = array<i32: 0>} : <f32>, <tensor<1x128xf32>>
    %cst = arith.constant dense<1> : tensor<32xi32>
    %1 = tt.descriptor_gather %0[%cst, %c0_i32] : (!tt.tensordesc<tensor<1x128xf32>>, tensor<32xi32>, i32) -> tensor<32x128xf32>
    tt.descriptor_store %0[%arg1, %arg2], %arg3 : !tt.tensordesc<tensor<1x128xf32>>, tensor<1x128xf32>
    tt.return
  }
}

// CHECK-LABEL: @multi_users
// CHECK-SAME: %[[ARG0:[^:]*]]
// CHECK-SAME: %[[ARG1:[^:]*]]
// CHECK-SAME: %[[ARG2:[^:]*]]
// CHECK-SAME: %[[ARG3:[^:]*]]
// CHECK-DAG: %[[CST:.*]] = arith.constant dense<256> : tensor<1x128xi64>
// CHECK-DAG: %[[CST_0:.*]] = arith.constant dense<0> : tensor<1x128xi64>
// CHECK-DAG: %[[CST_1:.*]] = arith.constant dense<256> : tensor<1x1xi64>
// CHECK-DAG: %[[CST_2:.*]] = arith.constant dense<0> : tensor<1x1xi64>

// CHECK-DAG: %[[VAL0:.*]] = arith.extsi %[[ARG1]] : i32 to i64
// CHECK-DAG: %[[VAL1:.*]] = arith.extsi %[[ARG2]] : i32 to i64
// CHECK-DAG: %[[VAL2:.*]] = tt.splat %[[VAL0]] :
// CHECK-DAG: %[[VAL3:.*]] = tt.splat %[[VAL1]] :
// CHECK-DAG: %[[VAL4:.*]] = tt.make_range {end = 128 : i32, start = 0 : i32}
// CHECK-DAG: %[[VAL5:.*]] = arith.extsi %[[VAL4]] :
// CHECK-DAG: %[[VAL6:.*]] = arith.addi %[[VAL3]], %[[VAL5]] :
// CHECK-DAG: %[[VAL7:.*]] = tt.expand_dims %[[VAL6]] {axis = 0 : i32}

// CHECK-DAG: %[[VAL8:.*]] = arith.cmpi sge, %[[VAL2]], %[[CST_2:.*]]
// CHECK-DAG: %[[VAL9:.*]] = arith.cmpi slt, %[[VAL2]], %[[CST_1:.*]]
// CHECK-DAG: %[[VAL10:.*]] = arith.andi %[[VAL8]], %[[VAL9]]
// CHECK-DAG: %[[VAL11:.*]] = tt.broadcast %[[VAL10]] : tensor<1x1xi1> -> tensor<1x128xi1>
// CHECK-DAG: %[[VAL12:.*]] = arith.cmpi sge, %[[VAL7]], %[[CST_0:.*]]
// CHECK-DAG: %[[VAL13:.*]] = arith.cmpi slt, %[[VAL7]], %[[CST:.*]]
// CHECK-DAG: %[[VAL14:.*]] = arith.andi %[[VAL12]], %[[VAL13]]
// CHECK-DAG: %[[VAL15:.*]] = arith.andi %[[VAL11]], %[[VAL14]]
// CHECK-DAG: %[[VAL16:.*]] = tt.splat %[[ARG0]] :
// CHECK-DAG: %[[VAL17:.*]] = tt.splat %[[VAL0]] :
// CHECK-DAG: %[[VAL18:.*]] = tt.addptr %[[VAL16]], %[[VAL17]] :
// CHECK-DAG: %[[VAL19:.*]] = arith.muli %[[VAL7]], %[[CST:.*]]
// CHECK-DAG: %[[VAL20:.*]] = tt.addptr %[[VAL18]], %[[VAL19]] :

// CHECK-DAG: tt.store %[[VAL20]], %[[ARG3]], %[[VAL15]]
