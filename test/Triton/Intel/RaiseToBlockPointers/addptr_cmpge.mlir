// RUN: triton-opt %s -triton-raise-block-pointer --split-input-file -canonicalize | FileCheck %s

// These tests check that loads/stores that exhibit a cmp ge against 0 work
// correctly with the pointer analysis pass

// Example of the triton kernel that generates the loads/stores with cmp ge 0.
//
//  def kernel(in_ptr0, out_ptr0, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
//     yoffset = tl.program_id(1) * YBLOCK
//     xoffset = tl.program_id(0) * XBLOCK
//     tmp0 = tl.load(tl.make_block_ptr(in_ptr0, shape=[16640, 10],
//                     strides=[1, 16640], block_shape=[XBLOCK, YBLOCK],
//                     order=[1, 0], offsets=[xoffset, yoffset]),
//                     boundary_check=[0, 1])
//     tl.store(tl.make_block_ptr(out_ptr0, shape=[16640, 10],
//                     strides=[1, 16640], block_shape=[XBLOCK, YBLOCK],
//                     order=[1, 0], offsets=[xoffset, yoffset]),
//                     tl.broadcast_to(tmp0, [XBLOCK, YBLOCK]).to(tl.float16),
//                     boundary_check=[0, 1])

tt.func public @test_masked_load(%arg0: !tt.ptr<f16>) -> tensor<16x16xf16> {
  %cst = arith.constant dense<0> : tensor<1x16xi64>
  %c16_i32 = arith.constant 16 : i32
  %0 = tt.get_program_id y : i32
  %1 = arith.muli %0, %c16_i32 : i32
  %2 = arith.extsi %1 : i32 to i64
  %3 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
  %4 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %5 = arith.extsi %4 : tensor<16xi32> to tensor<16xi64>
  %6 = tt.expand_dims %5 {axis = 1 : i32} : tensor<16xi64> -> tensor<16x1xi64>
  %7 = tt.broadcast %6 : tensor<16x1xi64> -> tensor<16x16xi64>
  %8 = tt.addptr %3, %7 : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi64>
  %9 = tt.splat %2 : i64 -> tensor<16xi64>
  %10 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %11 = arith.extsi %10 : tensor<16xi32> to tensor<16xi64>
  %12 = arith.addi %9, %11 : tensor<16xi64>
  %13 = tt.expand_dims %12 {axis = 0 : i32} : tensor<16xi64> -> tensor<1x16xi64>
  %14 = arith.cmpi sge, %13, %cst : tensor<1x16xi64>
  %15 = tt.broadcast %14 : tensor<1x16xi1> -> tensor<16x16xi1>
  %16 = tt.load %8 evictionPolicy = evict_last : tensor<16x16x!tt.ptr<f16>>
  // TODO: Replace above with below once support for masked loads is complete.
  // %16 = tt.load %8, %15 evictionPolicy = evict_last : tensor<16x16x!tt.ptr<f16>>
  tt.return %16 : tensor<16x16xf16>
}

// CHECK:         tt.func public @test_masked_load([[arg0:%.+]]: !tt.ptr<f16>) -> tensor<16x16xf16> {
// CHECK-DAG:       [[CST_0_i64:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[CST_1_i64:%.+]] = arith.constant 1 : i64
// CHECK-DAG:       [[CST_0_i32:%.+]] = arith.constant 0 : i32
// CHECK:           [[VAR_0:%.+]] = tt.make_tensor_ptr [[arg0]], {{\[}}[[CST_0_i64]], [[CST_0_i64]]], {{\[}}[[CST_1_i64]], [[CST_0_i64]]], {{\[}}[[CST_0_i32]], [[CST_0_i32]]] {{.*}} : <tensor<16x16xf16>>
// CHECK:           [[VAR_1:%.+]] = tt.load [[VAR_0]] evictionPolicy = evict_last : !tt.ptr<tensor<16x16xf16>>
// CHECK:           tt.return [[VAR_1]] : tensor<16x16xf16>
// CHECK:         }

// -----

tt.func public @test_masked_store(%arg0: !tt.ptr<f16>) {
  %cst = arith.constant dense<0> : tensor<16x1xi64>
  %cst_0 = arith.constant dense<1.500000e+01> : tensor<16x16xf16>
  %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
  %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %2 = arith.extsi %1 : tensor<16xi32> to tensor<16xi64>
  %3 = tt.expand_dims %2 {axis = 1 : i32} : tensor<16xi64> -> tensor<16x1xi64>
  %4 = tt.broadcast %3 : tensor<16x1xi64> -> tensor<16x16xi64>
  %5 = tt.addptr %0, %4 : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi64>
  %6 = arith.cmpi sge, %3, %cst : tensor<16x1xi64>
  %7 = tt.broadcast %6 : tensor<16x1xi1> -> tensor<16x16xi1>
  // TODO: Replace above with below once support for masked stores is complete.
  //  tt.store %5, %cst_0, %7 : tensor<16x16x!tt.ptr<f16>>
  tt.store %5, %cst_0 : tensor<16x16x!tt.ptr<f16>>
  tt.return
}

// CHECK:         tt.func public @test_masked_store([[arg0:%.+]]: !tt.ptr<f16>) {
// CHECK-DAG:       [[VAR_cst:%.+]] = arith.constant dense<1.500000e+01> : tensor<16x16xf16>
// CHECK-DAG:       [[CST_0_i64:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[CST_1_i64:%.+]] = arith.constant 1 : i64
// CHECK-DAG:       [[CST_0_i32:%.+]] = arith.constant 0 : i32
// CHECK:           [[VAR_0:%.+]] = tt.make_tensor_ptr [[arg0]], {{\[}}[[CST_0_i64]], [[CST_0_i64]]], {{\[}}[[CST_1_i64]], [[CST_0_i64]]], {{\[}}[[CST_0_i32]], [[CST_0_i32]]] {{.*}} : <tensor<16x16xf16>>
// CHECK:           tt.store [[VAR_0]], [[VAR_cst]] : !tt.ptr<tensor<16x16xf16>>
// CHECK:         }
