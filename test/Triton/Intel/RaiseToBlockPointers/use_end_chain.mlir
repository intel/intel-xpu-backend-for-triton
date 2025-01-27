// RUN: triton-opt.exe %s -triton-raise-block-pointer -canonicalize | FileCheck %s

module {
  tt.func @kernel(
  %arg0 : !tt.ptr<bf16>,
  %arg1 : !tt.ptr<bf16>
  )
  {
  %0 = tt.make_range {end = 768 : i32, start = 512 : i32}:tensor<256xi32>
  // offset = [512] size = 256, stride = 1
  %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<256xi32> -> tensor<256x1xi32>
  // offset = [512,0], size = [256,1], stride = [1,0]
  %2 = tt.broadcast %1 : tensor<256x1xi32> -> tensor<256x128xi32>
  // offset = [512,0], size = [256,128], stride = [1,0]
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
  // mixed use
  %17 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<256x128x!tt.ptr<bf16>>
  %18 = tt.addptr %17, %14 : tensor<256x128x!tt.ptr<bf16>>, tensor<256x128xi32>
  %19 = tt.load %18 : tensor<256x128x!tt.ptr<bf16>>
  tt.store %18, %19 : tensor<256x128x!tt.ptr<bf16>>
  %20 = arith.sitofp %14 : tensor<256x128xi32> to tensor<256x128xbf16>
  tt.store %18, %20 : tensor<256x128x!tt.ptr<bf16>>
  tt.return
  }
}

// CHECK:         tt.func @kernel([[PARAM_0_:%.+]]: !tt.ptr<bf16>, [[PARAM_1_:%.+]]: !tt.ptr<bf16>) {
// CHECK-DAG:       [[CST_1024_i32:%.+]] = arith.constant 1024 : i32
// CHECK-DAG:       [[CST_512_i32:%.+]] = arith.constant 512 : i32
// CHECK-DAG:       [[CST_0_i64:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[CST_1_i64:%.+]] = arith.constant 1 : i64
// CHECK-DAG:       [[CST_6_i64:%.+]] = arith.constant 6 : i64
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<6> : tensor<256x128xi32>
// CHECK-DAG:       [[VAR_0_:%.+]] = tt.make_range {end = 768 : i32, start = 512 : i32} : tensor<256xi32>
// CHECK:           [[VAR_1_:%.+]] = tt.expand_dims [[VAR_0_]] {axis = 1 : i32} : tensor<256xi32> -> tensor<256x1xi32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tt.broadcast [[VAR_1_]] : tensor<256x1xi32> -> tensor<256x128xi32>
// CHECK-DAG:       [[VAR_3_:%.+]] = tt.make_range {end = 1152 : i32, start = 1024 : i32} : tensor<128xi32>
// CHECK:           [[VAR_4_:%.+]] = tt.expand_dims [[VAR_3_]] {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
// CHECK:           [[VAR_5_:%.+]] = tt.broadcast [[VAR_4_]] : tensor<1x128xi32> -> tensor<256x128xi32>
// CHECK:           [[VAR_6_:%.+]] = arith.muli [[VAR_5_]], [[VAR_cst_]] : tensor<256x128xi32>
// CHECK-DAG:       [[VAR_7_:%.+]] = arith.addi [[VAR_2_]], [[VAR_6_]] : tensor<256x128xi32>
// CHECK-DAG:       [[VAR_8_:%.+]] = tt.make_tensor_ptr [[PARAM_1_]], {{\[}}[[CST_0_i64]], [[CST_0_i64]]], {{\[}}[[CST_1_i64]], [[CST_6_i64]]], {{\[}}[[CST_512_i32]], [[CST_1024_i32]]] {{.*}} : <tensor<256x128xbf16>>
// CHECK:           [[VAR_9_:%.+]] = tt.load [[VAR_8_]] : !tt.ptr<tensor<256x128xbf16>>
// CHECK:           tt.store [[VAR_8_]], [[VAR_9_]] : !tt.ptr<tensor<256x128xbf16>>
// CHECK:           [[VAR_10_:%.+]] = arith.sitofp [[VAR_7_]] : tensor<256x128xi32> to tensor<256x128xbf16>
// CHECK:           tt.store [[VAR_8_]], [[VAR_10_]] : !tt.ptr<tensor<256x128xbf16>>
// CHECK:           tt.return
// CHECK:         }
