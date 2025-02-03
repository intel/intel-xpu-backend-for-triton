// RUN: triton-opt %s -triton-raise-block-pointer -canonicalize | FileCheck %s

module {
  tt.func @kernel(
    %arg0 : !tt.ptr<bf16>
  )
  {
    %c0 = arith.constant 0 : index
    %c12 = arith.constant 12 : index
    %c3 = arith.constant 3 : index
    %i_c3 = arith.constant 3 : i32
    %0 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<256x!tt.ptr<bf16>>
    %1 = tt.make_range {end = 1280 : i32, start = 1024 : i32}:tensor<256xi32>
    // source: null, sizes: 256, offsets: 1024, strides: 1
    %2 = tt.addptr %0, %1 : tensor<256x!tt.ptr<bf16>>, tensor<256xi32>
    // source: arg0, sizes: 256, offsets: 1024, strides: 1
    // Example 2, gep operand is another gep's output, which is passed into the loop as varible, used before update
    %_ptr2 = scf.for %i = %c0 to %c12 step %c3 iter_args(%ptr = %2) -> (tensor<256x!tt.ptr<bf16>>) {
        // perform load
        %3 = tt.load %ptr {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<256x!tt.ptr<bf16>>
        tt.store %ptr, %3 : tensor<256x!tt.ptr<bf16>>
        // pointer updates
        %4 = tt.splat %i_c3 : i32 -> tensor<256xi32>
        %ptr_iter = tt.addptr %ptr, %4 : tensor<256x!tt.ptr<bf16>>, tensor<256xi32>
        scf.yield %ptr_iter : tensor<256x!tt.ptr<bf16>>
    }
    tt.return
  }
}

// CHECK:         tt.func @kernel([[PARAM_0_:%.+]]: !tt.ptr<bf16>) {
// CHECK-DAG:       [[CST_1_i64:%.+]] = arith.constant 1 : i64
// CHECK-DAG:       [[CST_1024_i32:%.+]] = arith.constant 1024 : i32
// CHECK-DAG:       [[CST_0_i64:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[CST_3_i32:%.+]] = arith.constant 3 : i32
// CHECK:           [[VAR_0_:%.+]] = tt.make_tensor_ptr [[PARAM_0_]], {{\[}}[[CST_0_i64]]], {{\[}}[[CST_1_i64]]], {{\[}}[[CST_1024_i32]]] {{.*}} : <tensor<256xbf16>>
// CHECK:           [[VAR_1_:%.+]] = scf.for [[VAR_arg1_:%.+]] = {{.*}} iter_args([[VAR_arg2_:%.+]] = [[VAR_0_]]) -> (!tt.ptr<tensor<256xbf16>>) {
// CHECK:             [[VAR_2_:%.+]] = tt.load [[VAR_arg2_]] : !tt.ptr<tensor<256xbf16>>
// CHECK:             tt.store [[VAR_arg2_]], [[VAR_2_]] : !tt.ptr<tensor<256xbf16>>
// CHECK:             [[VAR_3_:%.+]] = tt.advance [[VAR_arg2_]], {{\[}}[[CST_3_i32]]] : <tensor<256xbf16>>
// CHECK:             scf.yield [[VAR_3_]] : !tt.ptr<tensor<256xbf16>>
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
