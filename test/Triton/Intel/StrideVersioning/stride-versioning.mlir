// RUN: triton-opt %s -triton-intel-stride-versioning | FileCheck %s

module {
  tt.func public @matmul_kernel_with_tensor_descriptors(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i64 {tt.divisibility = 16 : i32}, %arg4: i64, %arg5: i64 {tt.divisibility = 16 : i32}, %arg6: i64, %arg7: i64 {tt.divisibility = 16 : i32}, %arg8: i64) attributes {noinline = false} {
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
    %11 = tt.make_tensor_ptr %arg0, [%9, %10], [%arg3, %arg4], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<256x32xbf16>>
    %12 = tt.make_tensor_ptr %arg1, [%10, %10], [%arg5, %arg6], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x256xbf16>>
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
    %16 = tt.make_tensor_ptr %arg2, [%9, %10], [%arg7, %arg8], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<256x256xf32>>
    %17 = arith.muli %7, %c256_i32 : i32
    %18 = arith.muli %8, %c256_i32 : i32
    %19 = tt.advance %16, [%17, %18] : <tensor<256x256xf32>>
    tt.store %19, %15#0 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<256x256xf32>>
    tt.return
  }

  // CHECK: tt.func public @matmul_kernel_with_tensor_descriptors
  // CHECK:   [[CST_1_i64:%.+]] = arith.constant 1 : i64
  // CHECK:   [[NEW_TDESC1:%.+]] = tt.make_tensor_ptr %arg0, {{.*}}, [%arg3, %c1_i64], {{.*}} {order = array<i32: 1, 0>} : <tensor<256x32xbf16>>
  // CHECK:   [[ORIG_TDESC1:%.+]] = tt.make_tensor_ptr %arg0, {{.*}}, [%arg3, %arg4], {{.*}} {order = array<i32: 1, 0>} : <tensor<256x32xbf16>>
  // CHECK:   [[NEW_TDESC2:%.+]] = tt.make_tensor_ptr %arg1, {{.*}}, [%arg5, %c1_i64], {{.*}} {order = array<i32: 1, 0>} : <tensor<32x256xbf16>>
  // CHECK:   [[ORIG_TDESC2:%.+]] = tt.make_tensor_ptr %arg1, {{.*}}, [%arg5, %arg6], {{.*}} {order = array<i32: 1, 0>} : <tensor<32x256xbf16>>
  // CHECK:   [[CMP1:%.+]] = arith.cmpi eq, %arg4, [[CST_1_i64]] : i64
  // CHECK:   [[CMP2:%.+]] = arith.cmpi eq, %arg6, [[CST_1_i64]] : i64
  // CHECK:   [[VER_COND:%.+]] = arith.andi [[CMP1]], [[CMP2]] : i1
  // CHECK:   [[LOOP_VER:%.+]]:2 = scf.if [[VER_COND]]
  // CHECK:     [[THEN_LOOP_RES:%.+]]:2 = scf.for {{.*}}
  // CHECK:       [[ADV_1:%.+]] = tt.advance [[NEW_TDESC1]]
  // CHECK:       [[ADV_2:%.+]] = tt.advance [[NEW_TDESC2]]
  // CHECK:   } else {
  // CHECK:     [[ELSE_LOOP_RES:%.+]]:2 = scf.for {{.*}}
  // CHECK:       [[ADV_1:%.+]] = tt.advance [[ORIG_TDESC1]]
  // CHECK:       [[ADV_2:%.+]] = tt.advance [[ORIG_TDESC2]]
  // CHECK:   }
}
