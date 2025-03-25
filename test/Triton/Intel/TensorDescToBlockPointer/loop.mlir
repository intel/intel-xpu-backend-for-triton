// RUN: triton-opt %s -triton-intel-tdesc-to-block-pointer  | FileCheck %s

module {
  tt.func public @load_in_loop(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index    
    %c10 = arith.constant 10 : index
    %c1_i64 = arith.constant 1 : i64
    %c8_i32 = arith.constant 8 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<16x32xf16>
    %0 = arith.extsi %arg2 : i32 to i64
    %tdesc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%0, %c1_i64] : <f16>, <tensor<16x32xf16>>
    %tdesc_out, %sum_out = scf.for %i = %c0 to %c10 step %c1 iter_args(%ptr_iter = %tdesc, %sum_iter = %cst) -> (!tt.tensordesc<tensor<16x32xf16>>, tensor<16x32xf16>) {
      %cast_i = arith.index_cast %i : index to i32
      %load1 = tt.descriptor_load %ptr_iter[%c8_i32, %cast_i] : !tt.tensordesc<tensor<16x32xf16>> -> tensor<16x32xf16>
      %sum_next = arith.addf %sum_iter, %load1 : tensor<16x32xf16>
      scf.yield %ptr_iter, %sum_next : !tt.tensordesc<tensor<16x32xf16>>, tensor<16x32xf16>
    }
    tt.return
  }

  // CHECK:      tt.func public @load_in_loop([[PARAM_0:%.+]]: !tt.ptr<f16>, [[PARAM_1:%.+]]: i32, [[PARAM_2:%.+]]: i32) {
  // CHECK-DAG:     [[CST_1_i64:%.+]] = arith.constant 1 : i64
  // CHECK-DAG:     [[CST_8_i32:%.+]] = arith.constant 8 : i32
  // CHECK-DAG:     [[CST:%.+]] = arith.constant dense<0.000000e+00> : tensor<16x32xf16>  
  // CHECK-DAG:     [[EXTSI_PARAM_2a:%.+]] = arith.extsi [[PARAM_2]] : i32 to i64
  // CHECK:         [[FOR_RES:%.+]]:2 = scf.for [[IV:%.+]] = {{.*}} iter_args([[VAR_arg1:%.+]] = {{.*}}, [[VAR_arg2:%.+]] = [[CST]]) -> (!tt.tensordesc<tensor<16x32xf16>>, tensor<16x32xf16>) {  
  // CHECK-DAG:       [[IDX_CAST_1:%.+]] = arith.index_cast [[IV]] : index to i32
  // CHECK-DAG:       [[EXTSI_PARAM_1:%.+]] = arith.extsi [[PARAM_1]] : i32 to i64
  // CHECK-DAG:       [[EXTSI_PARAM_2b:%.+]] = arith.extsi [[PARAM_2]] : i32 to i64
  // CHECK:           [[TENSOR_PTR:%.+]] = tt.make_tensor_ptr [[PARAM_0]], {{\[}}[[EXTSI_PARAM_1]], [[EXTSI_PARAM_2b]]], {{\[}}[[EXTSI_PARAM_2a]], [[CST_1_i64]]], {{\[}}[[CST_8_i32]], [[IDX_CAST_1]]] {{.*}} : <tensor<16x32xf16>>
  // CHECK:           [[LOAD:%.+]] = tt.load [[TENSOR_PTR]] : !tt.ptr<tensor<16x32xf16>>
  // CHECK:           [[ADD:%.+]] = arith.addf [[VAR_arg2]], [[LOAD]] : tensor<16x32xf16>  
  // CHECK:           scf.yield {{.*}}, [[ADD]] : !tt.tensordesc<tensor<16x32xf16>>, tensor<16x32xf16>
  // CHECK:         }
  // CHECK:         tt.return
  // CHECK:       }
}
