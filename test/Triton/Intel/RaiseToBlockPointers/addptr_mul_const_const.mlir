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
    // vector and scalar are both constant
    %4 = tt.make_range {end = 3072 : i32, start = 2048 : i32}:tensor<1024xi32>
    %c10 = arith.constant 10 : i32
    %5 = tt.splat %c10 : i32 -> tensor<1024xi32>
    %6 = arith.muli %5, %4 : tensor<1024xi32>
    //%6: splat(%c10)*range(2048, 4096);
    //%6: offset = %c10*2048, size = 1024, stride = %c10*1
    %7 = arith.addi %3, %6 : tensor<1024xi32>
    //%7: offset = %c10*2048 + %0, size = 1024, stride = %c10*1+1
    %8 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>>
    %9 = tt.addptr %8, %7 : tensor<1024x!tt.ptr<bf16>>, tensor<1024xi32>
    //source=%arg0 offset = %c10*2048 + pid0, size = 1024, stride = %c10*1+1
    %10 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>>
    %11 = tt.addptr %10, %3 : tensor<1024x!tt.ptr<bf16>>, tensor<1024xi32>
    //source=%arg1, offset = pid0, size = 1024, stride = 1
    %16 = tt.load %9 : tensor<1024x!tt.ptr<bf16>>
    tt.store %11, %16 : tensor<1024x!tt.ptr<bf16>>
    tt.return
  }
}

// CHECK:         tt.func @kernel([[PARAM_0_:%.+]]: !tt.ptr<bf16>, [[PARAM_1_:%.+]]: !tt.ptr<bf16>, [[PARAM_2_:%.+]]: i32) {
// CHECK-DAG:       [[CST_11_i32:%.+]] = arith.constant 11 : i32
// CHECK-DAG:       [[CST_20480_i32:%.+]] = arith.constant 20480 : i32
// CHECK-DAG:       [[CST_0_i64:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[CST_1_i64:%.+]] = arith.constant 1 : i64
// CHECK-DAG:       [[CST_11_i64:%.+]] = arith.constant 11 : i64
// CHECK-DAG:       [[VAR_0_:%.+]] = tt.get_program_id x : i32
// CHECK:           [[VAR_1_:%.+]] = arith.addi [[VAR_0_]], [[CST_20480_i32]] : i32
// CHECK:           [[VAR_2_:%.+]] = arith.divui [[VAR_1_]], [[CST_11_i32]] : i32
// CHECK-DAG:       [[VAR_3_:%.+]] = tt.make_tensor_ptr [[PARAM_0_]], {{\[}}[[CST_0_i64]]], {{\[}}[[CST_11_i64]]], {{\[}}[[VAR_2_]]] {{.*}} : <tensor<1024xbf16>>
// CHECK-DAG:       [[VAR_4_:%.+]] = tt.make_tensor_ptr [[PARAM_1_]], {{\[}}[[CST_0_i64]]], {{\[}}[[CST_1_i64]]], {{\[}}[[VAR_0_]]] {{.*}} : <tensor<1024xbf16>>
// CHECK:           [[VAR_5_:%.+]] = tt.load [[VAR_3_]] : !tt.ptr<tensor<1024xbf16>>
// CHECK:           tt.store [[VAR_4_]], [[VAR_5_]] : !tt.ptr<tensor<1024xbf16>>
// CHECK:           tt.return
// CHECK:         }
