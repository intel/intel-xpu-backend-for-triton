// RUN: triton-opt.exe %s -triton-raise-block-pointer -canonicalize | FileCheck %s

module {
  tt.func @kernel(
    %arg0 : !tt.ptr<bf16>,
    %arg1 : !tt.ptr<bf16>,
    %arg2 : !tt.ptr<bf16>
  )
  {
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %c64 = arith.constant 128 : i32
    %1 = tt.splat %c64 : i32 -> tensor<128xi32>
    %2 = arith.muli %0, %1 : tensor<128xi32>
    %3 = tt.expand_dims %2 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    %4 = tt.broadcast %3 : tensor<128x1xi32> -> tensor<128x64xi32>
    %5 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %6 = tt.expand_dims %5 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %7 = tt.broadcast %6 : tensor<1x64xi32> -> tensor<128x64xi32>
    %8 = arith.addi %4, %7 : tensor<128x64xi32>
    %10 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %11 = tt.expand_dims %10 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    %12 = tt.broadcast %11 : tensor<1x256xi32> -> tensor<64x256xi32>
    %13 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %c256 = arith.constant 256 : i32
    %14 = tt.splat %c256 : i32 -> tensor<64xi32>
    %15 = arith.muli %13, %14 : tensor<64xi32>
    %16 = tt.expand_dims %15 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %17 = tt.broadcast %16 : tensor<64x1xi32> -> tensor<64x256xi32>
    %18 = arith.addi %12, %17 : tensor<64x256xi32>
    %20 = tt.splat %c256 : i32 -> tensor<128xi32>
    %21 = arith.muli %0, %20 : tensor<128xi32>
    %22 = tt.expand_dims %21 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    %23 = tt.broadcast %22 : tensor<128x1xi32> -> tensor<128x256xi32>
    %24 = tt.expand_dims %10 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    %25 = tt.broadcast %24 {axis = 0 : i32} : tensor<1x256xi32> -> tensor<128x256xi32>
    %26 = arith.addi %23, %25 : tensor<128x256xi32>
    %30 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<128x64x!tt.ptr<bf16>>
    %31 = tt.addptr %30, %8 : tensor<128x64x!tt.ptr<bf16>>, tensor<128x64xi32>
    %32 = tt.load %31 {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<128x64x!tt.ptr<bf16>>
    %40 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<64x256x!tt.ptr<bf16>>
    %41 = tt.addptr %40, %18 : tensor<64x256x!tt.ptr<bf16>>, tensor<64x256xi32>
    %42 = tt.load %41 {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<64x256x!tt.ptr<bf16>>
    %50 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<128x256x!tt.ptr<bf16>>
    %51 = tt.addptr %50, %26 : tensor<128x256x!tt.ptr<bf16>>, tensor<128x256xi32>
    %cf0 = arith.constant 0.0 : bf16
    %71 = tt.splat %cf0 : bf16 -> tensor<128x256xbf16>
    %60 = tt.dot %32, %42, %71 {inputPrecision = 2 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<128x64xbf16> * tensor<64x256xbf16> -> tensor<128x256xbf16>
    tt.store %51, %60 : tensor<128x256x!tt.ptr<bf16>>
    tt.store %51, %71 : tensor<128x256x!tt.ptr<bf16>>
    tt.return
  }
}

// CHECK:         tt.func @kernel([[PARAM_0_:%.+]]: !tt.ptr<bf16>, [[PARAM_1_:%.+]]: !tt.ptr<bf16>, [[PARAM_2_:%.+]]: !tt.ptr<bf16>) {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<0.000000e+00> : tensor<128x256xbf16>
// CHECK-DAG:       [[CST_256_i64:%.+]] = arith.constant 256 : i64
// CHECK-DAG:       [[CST_128_i64:%.+]] = arith.constant 128 : i64
// CHECK-DAG:       [[CST_0_i64:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[CST_1_i64:%.+]] = arith.constant 1 : i64
// CHECK-DAG:       [[CST_0_i32:%.+]] = arith.constant 0 : i32
// CHECK:           [[VAR_0_:%.+]] = tt.make_tensor_ptr [[PARAM_0_]], {{\[}}[[CST_0_i64]], [[CST_0_i64]]], {{\[}}[[CST_128_i64]], [[CST_1_i64]]], {{\[}}[[CST_0_i32]], [[CST_0_i32]]] {{.*}} : <tensor<128x64xbf16>>
// CHECK-DAG:       [[VAR_1_:%.+]] = tt.load [[VAR_0_]] : !tt.ptr<tensor<128x64xbf16>>
// CHECK-DAG:       [[VAR_2_:%.+]] = tt.make_tensor_ptr [[PARAM_1_]], {{\[}}[[CST_0_i64]], [[CST_0_i64]]], {{\[}}[[CST_256_i64]], [[CST_1_i64]]], {{\[}}[[CST_0_i32]], [[CST_0_i32]]] {{.*}} : <tensor<64x256xbf16>>
// CHECK:           [[VAR_3_:%.+]] = tt.load [[VAR_2_]] : !tt.ptr<tensor<64x256xbf16>>
// CHECK-DAG:       [[VAR_4_:%.+]] = tt.make_tensor_ptr [[PARAM_2_]], {{\[}}[[CST_0_i64]], [[CST_0_i64]]], {{\[}}[[CST_256_i64]], [[CST_1_i64]]], {{\[}}[[CST_0_i32]], [[CST_0_i32]]] {{.*}} : <tensor<128x256xbf16>>
// CHECK-DAG:       [[VAR_5_:%.+]] = tt.dot [[VAR_1_]], [[VAR_3_]], [[VAR_cst_]] : tensor<128x64xbf16> * tensor<64x256xbf16> -> tensor<128x256xbf16>
// CHECK:           tt.store [[VAR_4_]], [[VAR_5_]] : !tt.ptr<tensor<128x256xbf16>>
// CHECK-NEXT:      tt.store [[VAR_4_]], [[VAR_cst_]] : !tt.ptr<tensor<128x256xbf16>>
// CHECK:           tt.return
// CHECK:         }
