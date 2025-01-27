// RUN: triton-opt.exe %s -triton-raise-block-pointer -canonicalize | FileCheck %s

module {
  tt.func @kernel(
    %a : !tt.ptr<i32>,
    %b : !tt.ptr<i32>
  ) -> () {
        // offset calculations
        %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
        // a pointer
        %8 = tt.splat %a : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
        %9 = tt.addptr %8, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>
        // b pointer
        %18 = tt.splat %b : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
        %19 = tt.addptr %18, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>
        %am = tt.load %9 : tensor<1024x!tt.ptr<i32>>
        %bm = tt.load %19 : tensor<1024x!tt.ptr<i32>>
        %5 = arith.addi %am, %bm : tensor<1024xi32>
        tt.store %19, %5 : tensor<1024x!tt.ptr<i32>>
        tt.return
    }
}

// CHECK:         tt.func @kernel([[PARAM_0_:%.+]]: !tt.ptr<i32>, [[PARAM_1_:%.+]]: !tt.ptr<i32>) {
// CHECK-DAG:       [[CST_0_i64:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[CST_1_i64:%.+]] = arith.constant 1 : i64
// CHECK-DAG:       [[CST_0_i32:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[VAR_0_:%.+]] = tt.make_tensor_ptr [[PARAM_0_]], {{\[}}[[CST_0_i64]]], {{\[}}[[CST_1_i64]]], {{\[}}[[CST_0_i32]]] {{.*}} : <tensor<1024xi32>>
// CHECK-DAG:       [[VAR_1_:%.+]] = tt.make_tensor_ptr [[PARAM_1_]], {{\[}}[[CST_0_i64]]], {{\[}}[[CST_1_i64]]], {{\[}}[[CST_0_i32]]] {{.*}} : <tensor<1024xi32>>
// CHECK-DAG:       [[VAR_2_:%.+]] = tt.load [[VAR_0_]] : !tt.ptr<tensor<1024xi32>>
// CHECK-DAG:       [[VAR_3_:%.+]] = tt.load [[VAR_1_]] : !tt.ptr<tensor<1024xi32>>
// CHECK:           [[VAR_4_:%.+]] = arith.addi [[VAR_2_]], [[VAR_3_]] : tensor<1024xi32>
// CHECK:           tt.store [[VAR_1_]], [[VAR_4_]] : !tt.ptr<tensor<1024xi32>>
// CHECK:           tt.return
// CHECK:         }
