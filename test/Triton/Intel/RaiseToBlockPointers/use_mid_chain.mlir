// RUN: triton-opt.exe %s -triton-raise-block-pointer -canonicalize | FileCheck %s

module {
  tt.func @kernel(
  %arg0 : !tt.ptr<bf16>,
  %arg1 : !tt.ptr<bf16>,
  %arg2 : !tt.ptr<i32>
  )
  {
  %0 = tt.make_range {end = 768 : i32, start = 512 : i32}:tensor<256xi32>
  // offset = [512] size = 256, stride = 1
  %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<256xi32> -> tensor<256x1xi32>
  // offset = [512,0], size = [256,1], stride = [1,0]
  %2 = tt.broadcast %1 : tensor<256x1xi32> -> tensor<256x128xi32>
  // offset = [512,0], size = [256,128], stride = [1,0]
  // mixed use
  %5 = tt.make_range {end = 1152 : i32, start = 1024 : i32}:tensor<128xi32>
  // offset = 1024, size = 128, stride = 1
  %6 = tt.expand_dims %5 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
  // offset = [0,1024], size = [1,128], stride = [0,1]
  %7 = tt.broadcast %6 : tensor<1x128xi32> -> tensor<256x128xi32>
  // offset = [0,1024], size = [256,128], stride = [0,1]
  %c6 = arith.constant 6 : i32
  %splat6 = tt.splat %c6 : i32 -> tensor<256x128xi32>
  %scale7 = arith.muli %7, %splat6 : tensor<256x128xi32>
  // offset = [0,6144], size = [256,128], stride = [0,6]
  %14 = arith.addi %2, %scale7 : tensor<256x128xi32>
  // offset = [512,6144], size = [256,128], stride = [1,6]
  %17 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<256x128x!tt.ptr<bf16>>
  %18 = tt.addptr %17, %14 : tensor<256x128x!tt.ptr<bf16>>, tensor<256x128xi32>
  %19 = tt.load %18 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256x128x!tt.ptr<bf16>>
  tt.store %18, %19 : tensor<256x128x!tt.ptr<bf16>>
  %20 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<256x128x!tt.ptr<i32>>
  %21 = tt.addptr %20, %14 : tensor<256x128x!tt.ptr<i32>>, tensor<256x128xi32>
  tt.store %21, %2 : tensor<256x128x!tt.ptr<i32>>
  tt.return
  }
}

// CHECK:         tt.func @kernel([[PARAM_0_:.+]]: !tt.ptr<bf16>, [[PARAM_1_:.+]]: !tt.ptr<bf16>, [[PARAM_2_:.+]]: !tt.ptr<i32>) {
// CHECK-DAG:       [[CST_1024_i32:%.+]] = arith.constant 1024 : i32
// CHECK-DAG:       [[CST_6_i64:%.+]] = arith.constant 6 : i64
// CHECK-DAG:       [[CST_1_i64:%.+]] = arith.constant 1 : i64
// CHECK-DAG:       [[CST_512_i32:%.+]] = arith.constant 512 : i32
// CHECK-DAG:       [[CST_0_i64:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[VAR_0_:%.+]] = tt.make_range {end = 768 : i32, start = 512 : i32} : tensor<256xi32>
// CHECK:           [[VAR_1_:%.+]] = tt.expand_dims [[VAR_0_]] {axis = 1 : i32} : tensor<256xi32> -> tensor<256x1xi32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tt.broadcast [[VAR_1_]] : tensor<256x1xi32> -> tensor<256x128xi32>
// CHECK-DAG:       [[VAR_3_:%.+]] = tt.make_tensor_ptr [[PARAM_1_]], {{\[}}[[CST_0_i64]], [[CST_0_i64]]], {{\[}}[[CST_1_i64]], [[CST_6_i64]]], {{\[}}[[CST_512_i32]], [[CST_1024_i32]]] {{.*}} : <tensor<256x128xbf16>>
// CHECK:           [[VAR_4_:%.+]] = tt.load [[VAR_3_]] : !tt.ptr<tensor<256x128xbf16>>
// CHECK:           tt.store [[VAR_3_]], [[VAR_4_]] : !tt.ptr<tensor<256x128xbf16>>
// CHECK-DAG:       [[VAR_5_:%.+]] = tt.make_tensor_ptr [[PARAM_2_]], {{\[}}[[CST_0_i64]], [[CST_0_i64]]], {{\[}}[[CST_1_i64]], [[CST_6_i64]]], {{\[}}[[CST_512_i32]], [[CST_1024_i32]]] {{.*}} : <tensor<256x128xi32>>
// CHECK:           tt.store [[VAR_5_]], [[VAR_2_]] : !tt.ptr<tensor<256x128xi32>>
// CHECK:           tt.return
// CHECK:         }
