// RUN: triton-opt %s --mlir-disable-threading --test-liveness 2>&1 | FileCheck %s
module attributes {"triton_gpu.num-warps" = 8 : i32} {
  tt.func public @test1(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>) {
    // CHECK-LABEL: test1    
    %c48_i32 = arith.constant 48 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %c2_i32 = arith.constant 2 : i32
    %c16_i32 = arith.constant 16 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1024_i64 = arith.constant 1024 : i64
    %c64_i64 = arith.constant 64 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %c3145728_i64 = arith.constant 3145728 : i64
    %c65536_i64 = arith.constant 65536 : i64
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<8x16xf32>
    %0 = gpu.subgroup_id : index
    %1 = arith.index_cast %0 : index to i32
    %2 = tt.get_program_id z : i32
    %3 = tt.get_program_id x : i32
    %4 = tt.get_program_id y : i32
    %5 = arith.extsi %3 : i32 to i64
    %6 = arith.muli %5, %c3145728_i64 : i64
    %7 = arith.extsi %4 : i32 to i64
    %8 = arith.muli %7, %c65536_i64 : i64
    %9 = arith.addi %6, %8 : i64
    %10 = tt.addptr %arg0, %9 : !tt.ptr<f16>, i64
    %11 = arith.muli %2, %c128_i32 : i32
    %12 = arith.muli %1, %c16_i32 : i32
    %13 = arith.addi %12, %11 : i32
    %14 = tt.make_tensor_ptr %10, [%c1024_i64, %c64_i64], [%c64_i64, %c1_i64], [%13, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x32xf16>>
    %15 = tt.make_tensor_ptr %10, [%c1024_i64, %c64_i64], [%c64_i64, %c1_i64], [%13, %c32_i32] {order = array<i32: 1, 0>} : <tensor<16x32xf16>>
    %58 = tt.load %14 {DotIdx = 0 : i32} : !tt.ptr<tensor<16x32xf16>>
    %59 = tt.load %15 {DotIdx = 0 : i32} : !tt.ptr<tensor<16x32xf16>>
    %28 = tt.addptr %arg1, %9 : !tt.ptr<f16>, i64
    %31 = tt.make_tensor_ptr %28, [%c64_i64, %c1024_i64], [%c1_i64, %c64_i64], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<16x16xf16>>
    %35 = tt.make_tensor_ptr %28, [%c64_i64, %c1024_i64], [%c1_i64, %c64_i64], [%c0_i32, %c16_i32] {order = array<i32: 0, 1>} : <tensor<16x16xf16>>
    %62:4 = scf.for %arg6 = %c0_i32 to %c1024_i32 step %c64_i32 iter_args(%arg8 = %cst_2, %arg10 = %cst_2, %arg21 = %31, %arg25 = %35)
           -> (tensor<8x16xf32>, tensor<8x16xf32>, !tt.ptr<tensor<16x16xf16>>, !tt.ptr<tensor<16x16xf16>>) : i32 {
      // CHECK: LiveIntervals for block: ^bb0
      // CHECK-NEXT: [%22, %28] for value: %c0_i32
      // CHECK-NEXT: [%22, %26] for value: %cst
      // CHECK-NEXT: [%22, %24] for value: %16
      // CHECK-NEXT: [%22, %28] for value: %c64_i32
      // CHECK-NEXT: [%22, %25] for value: %22
      // CHECK-NEXT: [%23, %26] for value: %23
      // CHECK-NEXT: [%24, %26] for value: %24
      // CHECK-NEXT: [%25, scf.yield] for value: %25
      // CHECK-NEXT: [%26, scf.yield] for value: %26
      // CHECK-NEXT: [%27, scf.yield] for value: %27
      // CHECK-NEXT: [%28, scf.yield] for value: %28

      // CHECK:      %22 = tt.load %arg5 {DotIdx = 1 : i32} : !tt.ptr<tensor<16x16xf16>>
      // CHECK-NEXT: %23 = tt.load %arg6 {DotIdx = 1 : i32} : !tt.ptr<tensor<16x16xf16>>
      // CHECK-NEXT: %24 = triton_intel_gpu.extract %16[0] : tensor<16x32xf16> -> tensor<8x16xf16>
      // CHECK-NEXT: %25 = tt.dot %24, %22, %cst, inputPrecision = tf32 : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      // CHECK-NEXT: %26 = tt.dot %24, %23, %cst, inputPrecision = tf32 : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      // CHECK-NEXT: %27 = tt.advance %arg5, [%c0_i32, %c64_i32] : <tensor<16x16xf16>>
      // CHECK-NEXT: %28 = tt.advance %arg6, [%c0_i32, %c64_i32] : <tensor<16x16xf16>>
      // CHECK-NEXT: scf.yield %25, %26, %27, %28 : tensor<8x16xf32>, tensor<8x16xf32>, !tt.ptr<tensor<16x16xf16>>, !tt.ptr<tensor<16x16xf16>>

      %75 = tt.load %arg21 {DotIdx = 1 : i32} : !tt.ptr<tensor<16x16xf16>>
      %79 = tt.load %arg25 {DotIdx = 1 : i32} : !tt.ptr<tensor<16x16xf16>>
      %91 = triton_intel_gpu.extract %58[0] : tensor<16x32xf16> -> tensor<8x16xf16>
      %92 = tt.dot %91, %75, %cst_2, inputPrecision = tf32 : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %107 = tt.dot %91, %79, %cst_2, inputPrecision = tf32 : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %321 = tt.advance %arg21, [%c0_i32, %c64_i32] : <tensor<16x16xf16>>
      %325 = tt.advance %arg25, [%c0_i32, %c64_i32] : <tensor<16x16xf16>>
      scf.yield %92, %107, %321, %325 : tensor<8x16xf32>, tensor<8x16xf32>, !tt.ptr<tensor<16x16xf16>>, !tt.ptr<tensor<16x16xf16>>
    }
    tt.return
  }
}
