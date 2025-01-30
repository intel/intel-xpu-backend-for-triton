// RUN: triton-opt %s -triton-raise-block-pointer -canonicalize | FileCheck %s

module {
  tt.func public @tensor_indices_nested(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c3_i32 = arith.constant 3 : i32
    %cst = arith.constant dense<4> : tensor<4xi32>
    %c2_i32 = arith.constant 2 : i32
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %2 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %3:2 = scf.for %arg2 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg3 = %0, %arg4 = %0) -> (tensor<4xi32>, tensor<4xi32>)  : i32 {
      %4 = arith.muli %arg2, %c2_i32 : i32
      %5 = tt.splat %4 : i32 -> tensor<4xi32>
      %6 = arith.addi %arg3, %5 : tensor<4xi32>
      %7 = tt.addptr %1, %6 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      %8 = tt.load %7 : tensor<4x!tt.ptr<f32>>
      %9 = tt.addptr %2, %arg4 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      tt.store %9, %8 : tensor<4x!tt.ptr<f32>>
      %10 = arith.addi %6, %cst : tensor<4xi32>
      %11 = arith.addi %arg4, %cst : tensor<4xi32>
      %12:2 = scf.for %arg5 = %c0_i32 to %c3_i32 step %c1_i32 iter_args(%arg6 = %10, %arg7 = %11) -> (tensor<4xi32>, tensor<4xi32>)  : i32 {
        %13 = arith.muli %arg5, %c3_i32 : i32
        %14 = tt.splat %13 : i32 -> tensor<4xi32>
        %15 = arith.addi %arg6, %14 : tensor<4xi32>
        %16 = tt.addptr %1, %15 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
        %17 = tt.load %16 : tensor<4x!tt.ptr<f32>>
        %18 = tt.addptr %2, %arg7 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
        tt.store %18, %17 : tensor<4x!tt.ptr<f32>>
        %19 = arith.addi %15, %cst : tensor<4xi32>
        %20 = arith.addi %arg7, %cst : tensor<4xi32>
        scf.yield %19, %20 : tensor<4xi32>, tensor<4xi32>
      }
      scf.yield %12#0, %12#1 : tensor<4xi32>, tensor<4xi32>
    }
    tt.return
  }
}

// CHECK:         tt.func public @tensor_indices_nested([[arg0_:.+]]: !tt.ptr<f32>, [[arg1_:.+]]: !tt.ptr<f32>) attributes {noinline = false} {
// CHECK:           [[VAR_0_:%.+]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK-DAG:       [[VAR_3_:%.+]]:2 = scf.for [[VAR_arg2_:%.+]] = {{.*}} iter_args([[VAR_arg3_:%.+]] = [[VAR_0_]], [[VAR_arg4_:%.+]] = [[VAR_0_]]) -> (tensor<4xi32>, tensor<4xi32>)  : i32 {
// CHECK-NOT:         tt.make_tensor_ptr
// CHECK-NOT:         tt.advance
// CHECK:             [[VAR_12_:%.+]]:2 = scf.for [[VAR_arg5_:%.+]] = {{.*}} iter_args([[VAR_arg6_:%.+]] = {{.*}}, [[VAR_arg7_:%.+]] = {{.*}}) -> (tensor<4xi32>, tensor<4xi32>)  : i32 {
// CHECK-NOT:           tt.make_tensor_ptr
// CHECK-NOT:           tt.advance
// CHECK:               scf.yield {{.*}}, {{.*}} : tensor<4xi32>, tensor<4xi32>
// CHECK:             }
// CHECK:             scf.yield [[VAR_12_]]#0, [[VAR_12_]]#1 : tensor<4xi32>, tensor<4xi32>
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
