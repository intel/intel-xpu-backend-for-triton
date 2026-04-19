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
// CHECK-DAG: %[[c256_i32:.*]] = arith.constant 256 : i32
// CHECK-DAG: %[[c256_i64:.*]] = arith.constant 256 : i64
// CHECK: %{{.*}}:7 = tt.call @callee(%[[PTR]], %[[c256_i32]], %[[c256_i32]], %[[c256_i64]], %[[c1]], %false, %false)
// CHECK-SAME -> (!tt.ptr<f32>, i32, i32, i64, i64, i1, i1)

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
// CHECK-DAG: %[[CST_I32_256:.*]] = arith.constant dense<256> : tensor<1x128xi32>
// CHECK-DAG: %[[CST_I32_0:.*]] = arith.constant dense<0> : tensor<1x128xi32>
// CHECK-DAG: %[[CST_I64_256:.*]] = arith.constant dense<256> : tensor<1x128xi64>
// CHECK-DAG: %[[CST_I64_1:.*]] = arith.constant dense<1> : tensor<32x128xi64>
// CHECK-DAG: %[[CST_ZERO:.*]] = arith.constant dense<0.000000e+00> : tensor<32x128xf32>

// CHECK-DAG: %[[RANGE:.*]] = tt.make_range {end = 128 : i32, start = 0 : i32}
// CHECK-DAG: %[[RANGE_EXP:.*]] = tt.expand_dims %[[RANGE]] {axis = 0 : i32}
// CHECK-DAG: %[[SPLAT_PTR:.*]] = tt.splat %[[ARG0]] :
// CHECK-DAG: %[[PTR1:.*]] = tt.addptr %[[SPLAT_PTR]], %[[CST_I64_1]] :
// Offset range extended to i64 only at stride multiply
// CHECK-DAG: %[[EXT:.*]] = arith.extsi %[[RANGE_EXP]] : tensor<1x128xi32> to tensor<1x128xi64>
// CHECK-DAG: %[[MUL:.*]] = arith.muli %[[EXT]], %[[CST_I64_256]] :
// CHECK-DAG: %[[BCAST:.*]] = tt.broadcast %[[MUL]] : tensor<1x128xi64> -> tensor<32x128xi64>
// CHECK-DAG: %[[PTR2:.*]] = tt.addptr %[[PTR1]], %[[BCAST]] :

// Mask comparisons in i32 (not i64)
// CHECK-DAG: %[[CMP_LO:.*]] = arith.cmpi sge, %[[RANGE_EXP]], %[[CST_I32_0]] : tensor<1x128xi32>
// CHECK-DAG: %[[CMP_HI:.*]] = arith.cmpi slt, %[[RANGE_EXP]], %[[CST_I32_256]] : tensor<1x128xi32>
// CHECK-DAG: %[[AND:.*]] = arith.andi %[[CMP_LO]], %[[CMP_HI]]
// CHECK-DAG: %[[MASK:.*]] = tt.broadcast %[[AND]] : tensor<1x128xi1> -> tensor<32x128xi1>

// CHECK-DAG: %[[LOAD:.*]] = tt.load %[[PTR2]], %[[MASK]], %[[CST_ZERO]]
// CHECK: tt.return %[[LOAD]] :

// -----

module {
  tt.func public @multi_users(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32, %arg2: i32, %arg3: tensor<1x128xf32>) -> (tensor<32x128xf32>) {
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c1_i64, %c256_i64] {order = array<i32: 0>} : <f32>, <tensor<1x128xf32>>
    %cst = arith.constant dense<1> : tensor<32xi32>
    %1 = tt.descriptor_gather %0[%cst, %c0_i32] : (!tt.tensordesc<tensor<1x128xf32>>, tensor<32xi32>, i32) -> tensor<32x128xf32>
    tt.descriptor_store %0[%arg1, %arg2], %arg3 : !tt.tensordesc<tensor<1x128xf32>>, tensor<1x128xf32>
    tt.return %1 : tensor<32x128xf32>
  }
}

// COM: Descriptor has two users: gather (-> tt.load) and store (-> tt.store).
// COM: Both are lowered to the pointer fallback path since the gather makes
// COM: the descriptor "unhandled".
// CHECK-LABEL: @multi_users
// CHECK-SAME: %[[ARG0:[^:]*]]: !tt.ptr<f32>
// CHECK-SAME: %[[ARG1:[^:]*]]: i32
// CHECK-SAME: %[[ARG2:[^:]*]]: i32
// CHECK-SAME: %[[ARG3:[^:]*]]: tensor<1x128xf32>

// COM: Gather path: lowered to tt.load with pointer arithmetic.
// CHECK: tt.load {{.*}} : tensor<32x128x!tt.ptr<f32>>

// COM: Store path: lowered to tt.store with pointer arithmetic.
// CHECK: tt.store {{.*}}, %[[ARG3]], {{.*}} : tensor<1x128x!tt.ptr<f32>>
