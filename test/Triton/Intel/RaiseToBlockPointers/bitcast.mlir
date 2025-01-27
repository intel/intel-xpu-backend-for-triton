// RUN: triton-opt.exe %s -triton-raise-block-pointer -canonicalize | FileCheck %s

module {
  tt.func @kernel(%a : !tt.ptr<i32>, %b : !tt.ptr<f32>) -> () {
    // offset calculations
    %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>

    // a pointer
    %8 = tt.splat %a : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
    %9 = tt.addptr %8, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>

    // b pointer
    %18 = tt.splat %b : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %19 = tt.addptr %18, %0 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>

    %am = tt.load %9 : tensor<1024x!tt.ptr<i32>>

    // cast result before doing float add
    %am_bitcast = tt.bitcast %am : tensor<1024xi32> -> tensor<1024xf32>


    tt.store %19, %am_bitcast : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK:         tt.func @kernel([[PARAM_0_:%.+]]: !tt.ptr<i32>, [[PARAM_1_:%.+]]: !tt.ptr<f32>) {
// CHECK-DAG:       [[CST_0_i64:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[CST_1_i64:%.+]] = arith.constant 1 : i64
// CHECK-DAG:       [[CST_0_i32:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[VAR_0_:%.+]] = tt.make_tensor_ptr [[PARAM_0_]], {{\[}}[[CST_0_i64]]], {{\[}}[[CST_1_i64]]], {{\[}}[[CST_0_i32]]] {{.*}} : <tensor<1024xi32>>
// CHECK-DAG:       [[VAR_1_:%.+]] = tt.make_tensor_ptr [[PARAM_1_]], {{\[}}[[CST_0_i64]]], {{\[}}[[CST_1_i64]]], {{\[}}[[CST_0_i32]]] {{.*}} : <tensor<1024xf32>>
// CHECK:           [[VAR_2_:%.+]] = tt.load [[VAR_0_]] : !tt.ptr<tensor<1024xi32>>
// CHECK:           [[VAR_3_:%.+]] = tt.bitcast [[VAR_2_]] : tensor<1024xi32> -> tensor<1024xf32>
// CHECK:           tt.store [[VAR_1_]], [[VAR_3_]] : !tt.ptr<tensor<1024xf32>>
// CHECK:           tt.return
// CHECK:         }
