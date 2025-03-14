// RUN: triton-opt %s -triton-raise-block-pointer --split-input-file -canonicalize | FileCheck %s

module {
  tt.func @kernel(
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
    // source: null, sizes: 256, offsets: 1024, strides: 1
    %2 = tt.addptr %0, %1 : tensor<256x!tt.ptr<bf16>>, tensor<256xi32>
    // source: arg0, sizes: 256, offsets: 1024, strides: 1
    %3 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<256x!tt.ptr<bf16>>
    %4 = tt.addptr %3, %1 : tensor<256x!tt.ptr<bf16>>, tensor<256xi32>
    // source: arg1, sizes: 256, offsets: 1024, strides: 1
    %_arg2, %_ptr_ld, %_arg3, %_ptr_st, %_arg4 = scf.for %i = %c0 to %c12 step %c3 iter_args(%arg2 = %c1, %ptr_ld = %2, %arg3 = %c2, %ptr_st = %4, %arg4 = %c3) -> (index, tensor<256x!tt.ptr<bf16>>, index, tensor<256x!tt.ptr<bf16>>, index) {
        // perform load
        %5 = tt.load %ptr_ld {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<256x!tt.ptr<bf16>>
        tt.store %ptr_st, %5 : tensor<256x!tt.ptr<bf16>>
        // pointer updates
        %cast3 = arith.index_cast %c3 : index to i32
        %6 = tt.splat %cast3 : i32 -> tensor<256xi32>
        %ptr_ld_iter = tt.addptr %ptr_ld, %6 : tensor<256x!tt.ptr<bf16>>, tensor<256xi32>
        // source: arg0, sizes: 256, offsets: 1024 + i*3, strides: 1
        %arg2_iter = arith.addi %arg2, %c3 : index
        %arg3_iter = arith.addi %arg3, %c3 : index
        %arg4_iter = arith.addi %arg4, %c3 : index
        %7 = arith.addi %arg2_iter, %arg3_iter : index
        %8 = arith.addi %7, %arg4_iter : index
        %cast8 = arith.index_cast %8 : index to i32
        %9 = tt.splat %cast8 : i32 -> tensor<256xi32>
        %ptr_st_iter = tt.addptr %ptr_st, %9 : tensor<256x!tt.ptr<bf16>>, tensor<256xi32>
        // source: arg1, sizes: 256, offsets: 1024 + loop-carry variable*i, strides: 1
        scf.yield %arg2_iter, %ptr_ld_iter, %arg3_iter, %ptr_st_iter, %arg4_iter : index, tensor<256x!tt.ptr<bf16>>, index, tensor<256x!tt.ptr<bf16>>, index
    }
    tt.return
  }
}

// CHECK:         tt.func @kernel([[PARAM_0_:%.+]]: !tt.ptr<bf16>, [[PARAM_1_:%.+]]: !tt.ptr<bf16>) {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : i64
// CHECK-DAG:       [[CST_1024_:%.+]] = arith.constant 1024 : i32
// CHECK-DAG:       [[CST_1_1:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_3_i32:%.+]] = arith.constant 3 : i32
// CHECK-DAG:       [[CST_12_:%.+]] = arith.constant 12 : index
// CHECK-DAG:       [[VAR_1_:%.+]] = tt.make_tensor_ptr [[PARAM_0_]], {{\[}}[[CST_0_]]], {{\[}}[[CST_1_]]], {{\[}}[[CST_1024_]]] {{.*}} : <tensor<256xbf16>
// CHECK-DAG:       [[VAR_2_:%.+]] = tt.make_tensor_ptr [[PARAM_1_]], {{\[}}[[CST_0_]]], {{\[}}[[CST_1_]]], {{\[}}[[CST_1024_]]] {{.*}} : <tensor<256xbf16>
// CHECK:           [[VAR_0_:%.+]]:5 = scf.for [[VAR_arg2_:%.+]] = {{.*}} iter_args([[VAR_arg3_:%.+]] = [[CST_1_1]], [[VAR_arg4_:%.+]] = [[VAR_1_]], [[VAR_arg5_:%.+]] = [[CST_2_]], [[VAR_arg6_:%.+]] = [[VAR_2_]], [[VAR_arg7_:%.+]] = [[CST_3_]]) -> (index, !tt.ptr<tensor<256xbf16>>, index, !tt.ptr<tensor<256xbf16>>, index) {
// CHECK:             [[VAR_3_:%.+]] = tt.load [[VAR_arg4_]] : !tt.ptr<tensor<256xbf16>>
// CHECK:             tt.store [[VAR_arg6_]], [[VAR_3_]] : !tt.ptr<tensor<256xbf16>>
// CHECK-DAG:         [[VAR_4_:%.+]] = tt.advance [[VAR_arg4_]], {{\[}}[[CST_3_i32]]] : <tensor<256xbf16>>
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.addi [[VAR_arg3_]], [[CST_3_]] : index
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.addi [[VAR_arg5_]], [[CST_3_]] : index
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.addi [[VAR_arg7_]], [[CST_3_]] : index
// CHECK:             [[VAR_8_:%.+]] = arith.addi [[VAR_5_]], [[VAR_6_]] : index
// CHECK:             [[VAR_9_:%.+]] = arith.addi [[VAR_8_]], [[VAR_7_]] : index
// CHECK:             [[VAR_10_:%.+]] = arith.index_cast [[VAR_9_]] : index to i32
// CHECK:             [[VAR_11_:%.+]] = tt.advance [[VAR_arg6_]], {{\[}}[[VAR_10_]]] : <tensor<256xbf16>>
// CHECK:             scf.yield [[VAR_5_]], [[VAR_4_]], [[VAR_6_]], [[VAR_11_]], [[VAR_7_]] : index, !tt.ptr<tensor<256xbf16>>, index, !tt.ptr<tensor<256xbf16>>, index
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
