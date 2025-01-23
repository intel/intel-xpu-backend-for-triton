// RUN: triton-opt %s -triton-raise-block-pointer -canonicalize | FileCheck %s

module {
  tt.func @kernel(
    %arg0 : !tt.ptr<bf16>,
    %arg1 : !tt.ptr<bf16>,
    %arg2 : i32
  )
  {
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {end = 1024 : i32, start = 0 : i32}:tensor<1024xi32>
    %2 = tt.splat %0 : i32 -> tensor<1024xi32>
    %3 = arith.addi %2, %1 : tensor<1024xi32>
    //%3: splat(%0) + range(0, 1024)
    //%3: offset = %0, size = 1024, stride = 1
    // vector is constant, scalar is value
    %4 = tt.make_range {end = 3072 : i32, start = 2048 : i32}:tensor<1024xi32>
    %5 = tt.splat %arg2 : i32 -> tensor<1024xi32>
    %6 = arith.muli %5, %4 : tensor<1024xi32>
    //%6: splat(%arg2)*range(2048, 3072);
    //%6: offset = %arg2*2048, size = 1024, stride = %arg2*1
    %7 = arith.addi %3, %6 : tensor<1024xi32>
    //%7: offset = %arg2*2048 + %0, size = 1024, stride = %arg2*1+1
    %8 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>>
    %9 = tt.addptr %8, %7 : tensor<1024x!tt.ptr<bf16>>, tensor<1024xi32>
    //source=%arg0: offset = %arg2*2048 + pid0, size = 1024, stride = %arg2*1+1
    %10 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>>
    %11 = tt.addptr %10, %3 : tensor<1024x!tt.ptr<bf16>>, tensor<1024xi32>
    //source=%arg1: offset = pid0, size = 1024, stride = 1
    %16 = tt.load %9 : tensor<1024x!tt.ptr<bf16>>
    tt.store %11, %16 : tensor<1024x!tt.ptr<bf16>>
    tt.return
  }
}

// CHECK:         tt.func @kernel([[PARAM_0_:%.+]]: !tt.ptr<bf16>, [[PARAM_1_:%.+]]: !tt.ptr<bf16>, [[PARAM_2_:%.+]]: i32) {
// CHECK-DAG:       [[CST_2048_i32:%.+]] = arith.constant 2048 : i32
// CHECK-DAG:       [[CST_1_i64:%.+]] = arith.constant 1 : i64
// CHECK-DAG:       [[CST_0_i64:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[VAR_0_:%.+]] = tt.get_program_id x : i32
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK:           [[VAR_2_:%.+]] = arith.muli [[PARAM_2_]], [[CST_2048_i32]] : i32
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[VAR_1_]] : index to i64
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.addi [[VAR_0_]], [[VAR_2_]] : i32
// CHECK-DAG:       [[VAR_5_:%.+]] = arith.addi [[VAR_3_]], [[CST_1_i64]] : i64
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.trunci [[VAR_5_]] : i64 to i32
// CHECK-DAG:       [[VAR_7_:%.+]] = arith.divui [[VAR_4_]], [[VAR_6_]] : i32
// CHECK-DAG:       [[VAR_8_:%.+]] = tt.make_tensor_ptr [[PARAM_0_]], {{\[}}[[CST_0_i64]]], {{\[}}[[VAR_5_]]], {{\[}}[[VAR_7_]]] {{.*}} : <tensor<1024xbf16>>
// CHECK-DAG:       [[VAR_9_:%.+]] = tt.make_tensor_ptr [[PARAM_1_]], {{\[}}[[CST_0_i64]]], {{\[}}[[CST_1_i64]]], {{\[}}[[VAR_0_]]] {{.*}} : <tensor<1024xbf16>>
// CHECK:           [[VAR_10_:%.+]] = tt.load [[VAR_8_]] : !tt.ptr<tensor<1024xbf16>>
// CHECK:           tt.store [[VAR_9_]], [[VAR_10_]] : !tt.ptr<tensor<1024xbf16>>
// CHECK:           tt.return
// CHECK:         }
