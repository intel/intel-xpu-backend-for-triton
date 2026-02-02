// RUN: triton-opt %s -split-input-file -triton-intel-stride-versioning | FileCheck %s

module {
  tt.func public @version_for_loop_bptr(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: i64, %arg3: i64, %arg4: i64 {tt.divisibility = 16 : i32}, %arg5: i64) {
    %c64_i32 = arith.constant 64 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32>
    %c256_i32 = arith.constant 256 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %c4096_i32 = arith.constant 4096 : i32
    %c8192_i32 = arith.constant 8192 : i32
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.divsi %0, %c64_i32 : i32
    %2 = arith.muli %1, %c4_i32 : i32
    %3 = arith.subi %c32_i32, %2 : i32
    %4 = arith.minsi %3, %c4_i32 : i32
    %5 = arith.remsi %0, %c64_i32 : i32
    %6 = arith.remsi %5, %4 : i32
    %7 = arith.addi %2, %6 : i32
    %8 = arith.divsi %5, %4 : i32
    %9 = arith.extsi %c8192_i32 : i32 to i64
    %10 = arith.extsi %c4096_i32 : i32 to i64
    %11 = tt.make_tensor_ptr %arg0, [%9, %10], [%arg2, %arg3], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<256x32xbf16>>
    %12 = tt.make_tensor_ptr %arg1, [%10, %10], [%arg4, %arg5], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x256xbf16>>
    %13 = arith.muli %7, %c256_i32 : i32
    %14 = arith.muli %8, %c256_i32 : i32
    %15:2 = scf.for %arg9 = %c0_i32 to %c4096_i32 step %c32_i32 iter_args(%arg10 = %cst, %arg11 = %c0_i32) -> (tensor<256x256xf32>, i32)  : i32 {
      %20 = tt.advance %11, [%13, %arg11] : <tensor<256x32xbf16>>
      %21 = tt.load %20 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<256x32xbf16>>
      %22 = tt.advance %12, [%arg11, %14] : <tensor<32x256xbf16>>
      %23 = tt.load %22 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<32x256xbf16>>
      %24 = tt.dot %21, %23, %cst, inputPrecision = tf32 : tensor<256x32xbf16> * tensor<32x256xbf16> -> tensor<256x256xf32>
      %25 = arith.addf %arg10, %24 : tensor<256x256xf32>
      %26 = arith.addi %arg11, %c32_i32 : i32
      scf.yield %25, %26 : tensor<256x256xf32>, i32
    }
    tt.return
  }

  // CHECK: tt.func public @version_for_loop_bptr([[ARG0:%.+]]: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, [[ARG1:%.+]]: !tt.ptr<bf16> {tt.divisibility = 16 : i32},
  // CHECK-SAME:                                  [[ARG2:%.+]]: i64, [[ARG3:%.+]]: i64, [[ARG4:%.+]]: i64 {tt.divisibility = 16 : i32}, [[ARG5:%.+]]: i64)
  // CHECK:     [[CST_1_i64:%.+]] = arith.constant 1 : i64
  // CHECK-DAG: [[NEW_PTR1:%.+]] = tt.make_tensor_ptr [[ARG0]], {{.*}}, [[[ARG2]], [[CST_1_i64]]], {{.*}} {order = array<i32: 1, 0>} : <tensor<256x32xbf16>>
  // CHECK-DAG: [[ORIG_PTR1:%.+]] = tt.make_tensor_ptr [[ARG0]], {{.*}}, [[[ARG2]], [[ARG3]]], {{.*}} {order = array<i32: 1, 0>} : <tensor<256x32xbf16>>
  // CHECK:     [[NEW_PTR2:%.+]] = tt.make_tensor_ptr [[ARG1]], {{.*}}, [[[ARG4]], [[CST_1_i64]]], {{.*}} {order = array<i32: 1, 0>} : <tensor<32x256xbf16>>
  // CHECK:     [[ORIG_PTR2:%.+]] = tt.make_tensor_ptr [[ARG1]], {{.*}}, [[[ARG4]], [[ARG5]]], {{.*}} {order = array<i32: 1, 0>} : <tensor<32x256xbf16>>
  // CHECK-DAG: [[CMP1:%.+]] = arith.cmpi eq, [[ARG3]], [[CST_1_i64]] : i64
  // CHECK-DAG: [[CMP2:%.+]] = arith.cmpi eq, [[ARG5]], [[CST_1_i64]] : i64
  // CHECK:     [[VER_COND:%.+]] = arith.andi [[CMP1]], [[CMP2]] : i1
  // CHECK:     [[LOOP_VER:%.+]]:2 = scf.if [[VER_COND]]
  // CHECK:       scf.for
  // CHECK:         tt.advance [[NEW_PTR1]]
  // CHECK:         tt.advance [[NEW_PTR2]]
  // CHECK:     } else {
  // CHECK:       scf.for
  // CHECK:         tt.advance [[ORIG_PTR1]]
  // CHECK:         tt.advance [[ORIG_PTR2]]
  // CHECK:     }
}

// -----

module {
  tt.func public @version_for_loop_tdesc(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: i64, %arg3: i64, %arg4: i64 {tt.divisibility = 16 : i32}, %arg5: i64) {
    %c64_i32 = arith.constant 64 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32>
    %c256_i32 = arith.constant 256 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %c4096_i32 = arith.constant 4096 : i32
    %c8192_i32 = arith.constant 8192 : i32
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.divsi %0, %c64_i32 : i32
    %2 = arith.muli %1, %c4_i32 : i32
    %3 = arith.subi %c32_i32, %2 : i32
    %4 = arith.minsi %3, %c4_i32 : i32
    %5 = arith.remsi %0, %c64_i32 : i32
    %6 = arith.remsi %5, %4 : i32
    %7 = arith.addi %2, %6 : i32
    %8 = arith.divsi %5, %4 : i32
    %9 = arith.extsi %c8192_i32 : i32 to i64
    %10 = arith.trunci %9 : i64 to i32
    %11 = arith.extsi %c4096_i32 : i32 to i64
    %12 = arith.trunci %11 : i64 to i32
    %13 = tt.make_tensor_descriptor %arg0, [%10, %12], [%arg2, %arg3] : <bf16>, <tensor<256x32xbf16>>
    %14 = tt.make_tensor_descriptor %arg1, [%12, %12], [%arg4, %arg5] : <bf16>, <tensor<32x256xbf16>>
    %15 = arith.muli %7, %c256_i32 : i32
    %16 = arith.muli %8, %c256_i32 : i32
    %17:2 = scf.for %arg6 = %c0_i32 to %c4096_i32 step %c32_i32 iter_args(%arg7 = %cst, %arg8 = %c0_i32) -> (tensor<256x256xf32>, i32)  : i32 {
      %18 = tt.descriptor_load %13[%15, %arg8] : !tt.tensordesc<tensor<256x32xbf16>> -> tensor<256x32xbf16>
      %19 = tt.descriptor_load %14[%arg8, %16] : !tt.tensordesc<tensor<32x256xbf16>> -> tensor<32x256xbf16>
      %20 = tt.dot %18, %19, %cst, inputPrecision = tf32 : tensor<256x32xbf16> * tensor<32x256xbf16> -> tensor<256x256xf32>
      %21 = arith.addf %arg7, %20 : tensor<256x256xf32>
      %22 = arith.addi %arg8, %c32_i32 : i32
      scf.yield %21, %22 : tensor<256x256xf32>, i32
    }
    tt.return
  }

  // CHECK: tt.func public @version_for_loop_tdesc([[ARG0:%.+]]: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, [[ARG1:%.+]]: !tt.ptr<bf16> {tt.divisibility = 16 : i32},
  // CHECK-SAME:                                   [[ARG2:%.+]]: i64, [[ARG3:%.+]]: i64, [[ARG4:%.+]]: i64 {tt.divisibility = 16 : i32}, [[ARG5:%.+]]: i64)
  // CHECK:     [[CST_1_i64:%.+]] = arith.constant 1 : i64
  // CHECK-DAG: [[NEW_PTR1:%.+]] = tt.make_tensor_descriptor [[ARG0]], {{.*}}, [[[ARG2]], [[CST_1_i64]]] : <bf16>, <tensor<256x32xbf16>>
  // CHECK-DAG: [[ORIG_PTR1:%.+]] = tt.make_tensor_descriptor [[ARG0]], {{.*}}, [[[ARG2]], [[ARG3]]] : <bf16>, <tensor<256x32xbf16>>
  // CHECK-NOT: separator of consecutive DAGs
  // CHECK-DAG: [[NEW_PTR2:%.+]] = tt.make_tensor_descriptor [[ARG1]], {{.*}}, [[[ARG4]], [[CST_1_i64]]] : <bf16>, <tensor<32x256xbf16>>
  // CHECK-DAG: [[ORIG_PTR2:%.+]] = tt.make_tensor_descriptor [[ARG1]], {{.*}}, [[[ARG4]], [[ARG5]]] : <bf16>, <tensor<32x256xbf16>>
  // CHECK-NOT: separator of consecutive DAGs
  // CHECK-DAG: [[CMP1:%.+]] = arith.cmpi eq, [[ARG3]], [[CST_1_i64]] : i64
  // CHECK-DAG: [[CMP2:%.+]] = arith.cmpi eq, [[ARG5]], [[CST_1_i64]] : i64
  // CHECK:     [[VER_COND:%.+]] = arith.andi [[CMP1]], [[CMP2]] : i1
  // CHECK:     [[LOOP_VER:%.+]]:2 = scf.if [[VER_COND]]
  // CHECK:       scf.for
  // CHECK-DAG:     tt.descriptor_load [[NEW_PTR1]]
  // CHECK-DAG:     tt.descriptor_load [[NEW_PTR2]]
  // CHECK:     } else {
  // CHECK:       scf.for
  // CHECK-DAG:     tt.descriptor_load [[ORIG_PTR1]]
  // CHECK-DAG:     tt.descriptor_load [[ORIG_PTR2]]
  // CHECK:     }

}

// -----

module {
  tt.func public @do_not_version_bptr(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) {
    %c64_i32 = arith.constant 64 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32>
    %c256_i32 = arith.constant 256 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %c4096_i32 = arith.constant 4096 : i32
    %c8192_i32 = arith.constant 8192 : i32
    %c4_i32 = arith.constant 4 : i32
    %c2_i64 = arith.constant 2 : i64
    %c4_i64 = arith.constant 4 : i64
    %0 = tt.get_program_id x : i32
    %1 = arith.divsi %0, %c64_i32 : i32
    %2 = arith.muli %1, %c4_i32 : i32
    %3 = arith.subi %c32_i32, %2 : i32
    %4 = arith.minsi %3, %c4_i32 : i32
    %5 = arith.remsi %0, %c64_i32 : i32
    %6 = arith.remsi %5, %4 : i32
    %7 = arith.addi %2, %6 : i32
    %8 = arith.divsi %5, %4 : i32
    %9 = arith.extsi %c8192_i32 : i32 to i64
    %10 = arith.extsi %c4096_i32 : i32 to i64
    %11 = tt.make_tensor_ptr %arg0, [%9, %10], [%c4_i64, %c2_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<256x32xbf16>>
    %12 = tt.make_tensor_ptr %arg1, [%10, %10], [%c2_i64, %c4_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x256xbf16>>
    %13 = arith.muli %7, %c256_i32 : i32
    %14 = arith.muli %8, %c256_i32 : i32
    %15:2 = scf.for %arg9 = %c0_i32 to %c4096_i32 step %c32_i32 iter_args(%arg10 = %cst, %arg11 = %c0_i32) -> (tensor<256x256xf32>, i32)  : i32 {
      %20 = tt.advance %11, [%13, %arg11] : <tensor<256x32xbf16>>
      %21 = tt.load %20 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<256x32xbf16>>
      %22 = tt.advance %12, [%arg11, %14] : <tensor<32x256xbf16>>
      %23 = tt.load %22 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<32x256xbf16>>
      %24 = tt.dot %21, %23, %cst, inputPrecision = tf32 : tensor<256x32xbf16> * tensor<32x256xbf16> -> tensor<256x256xf32>
      %25 = arith.addf %arg10, %24 : tensor<256x256xf32>
      %26 = arith.addi %arg11, %c32_i32 : i32
      scf.yield %25, %26 : tensor<256x256xf32>, i32
    }
    tt.return
  }

  // CHECK: tt.func public @do_not_version_bptr([[ARG0:%.+]]: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, [[ARG1:%.+]]: !tt.ptr<bf16> {tt.divisibility = 16 : i32})
  // CHECK-DAG: [[PTR1:%.+]] = tt.make_tensor_ptr [[ARG0]]
  // CHECK-DAG: [[PTR2:%.+]] = tt.make_tensor_ptr [[ARG1]]
  // CHECK-NOT: tt.make_tensor_ptr
  // CHECK-NOT: scf.if
  // CHECK:     scf.for
  // CHECK:       tt.advance [[PTR1]]
  // CHECK:       tt.advance [[PTR2]]
}

// -----

module {
  tt.func public @do_not_version_tdesc(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) {
    %c64_i32 = arith.constant 64 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32>
    %c256_i32 = arith.constant 256 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %c4096_i32 = arith.constant 4096 : i32
    %c8192_i32 = arith.constant 8192 : i32
    %c4_i32 = arith.constant 4 : i32
    %c2_i64 = arith.constant 2 : i64
    %c4_i64 = arith.constant 4 : i64
    %0 = tt.get_program_id x : i32
    %1 = arith.divsi %0, %c64_i32 : i32
    %2 = arith.muli %1, %c4_i32 : i32
    %3 = arith.subi %c32_i32, %2 : i32
    %4 = arith.minsi %3, %c4_i32 : i32
    %5 = arith.remsi %0, %c64_i32 : i32
    %6 = arith.remsi %5, %4 : i32
    %7 = arith.addi %2, %6 : i32
    %8 = arith.divsi %5, %4 : i32
    %9 = arith.extsi %c8192_i32 : i32 to i64
    %10 = arith.extsi %c4096_i32 : i32 to i64
    %11 = tt.make_tensor_descriptor %arg0, [%c8192_i32, %c4096_i32], [%c4_i64, %c2_i64] : <bf16>, <tensor<256x32xbf16>>
    %12 = tt.make_tensor_descriptor %arg1, [%c4096_i32, %c4096_i32], [%c2_i64, %c4_i64] : <bf16>, <tensor<32x256xbf16>>
    %13 = arith.muli %7, %c256_i32 : i32
    %14 = arith.muli %8, %c256_i32 : i32
    %15:2 = scf.for %arg9 = %c0_i32 to %c4096_i32 step %c32_i32 iter_args(%arg10 = %cst, %arg11 = %c0_i32) -> (tensor<256x256xf32>, i32)  : i32 {
      %21 = tt.descriptor_load %11[%13, %arg11] : !tt.tensordesc<tensor<256x32xbf16>> -> tensor<256x32xbf16>
      %23 = tt.descriptor_load %12[%arg11, %14] : !tt.tensordesc<tensor<32x256xbf16>> -> tensor<32x256xbf16>
      %24 = tt.dot %21, %23, %cst, inputPrecision = tf32 : tensor<256x32xbf16> * tensor<32x256xbf16> -> tensor<256x256xf32>
      %25 = arith.addf %arg10, %24 : tensor<256x256xf32>
      %26 = arith.addi %arg11, %c32_i32 : i32
      scf.yield %25, %26 : tensor<256x256xf32>, i32
    }
    tt.return
  }

  // CHECK: tt.func public @do_not_version_tdesc([[ARG0:%.+]]: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, [[ARG1:%.+]]: !tt.ptr<bf16> {tt.divisibility = 16 : i32})
  // CHECK-DAG: [[PTR1:%.+]] = tt.make_tensor_descriptor [[ARG0]]
  // CHECK-DAG: [[PTR2:%.+]] = tt.make_tensor_descriptor [[ARG1]]
  // CHECK-NOT: tt.make_tensor_descriptor
  // CHECK-NOT: scf.if
  // CHECK:     scf.for
  // CHECK:       tt.descriptor_load [[PTR1]]
  // CHECK:       tt.descriptor_load [[PTR2]]
}
