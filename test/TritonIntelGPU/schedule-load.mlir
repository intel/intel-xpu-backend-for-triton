// RUN: TRITON_INTEL_ENABLE_INSTR_SCHED=1 triton-opt %s -split-input-file -tritonintelgpu-schedule-load | FileCheck %s
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
  tt.func public @_attn_fwd(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: f32, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c48_i32 = arith.constant 48 : i32
    %c1_i32 = arith.constant 1 : i32
    %c3_i32 = arith.constant 3 : i32
    %c8_i32 = arith.constant 8 : i32
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
    %16 = tt.addptr %arg2, %9 : !tt.ptr<f16>, i64
    %17 = arith.divsi %1, %c2_i32 : i32
    %18 = arith.andi %17, %c3_i32 : i32
    %19 = arith.muli %18, %c16_i32 : i32
    %20 = arith.andi %1, %c1_i32 : i32
    %21 = arith.muli %20, %c32_i32 : i32
    %28 = tt.addptr %arg1, %9 : !tt.ptr<f16>, i64
    %31 = tt.make_tensor_ptr %28, [%c64_i64, %c1024_i64], [%c1_i64, %c64_i64], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<16x16xf16>>
    %32 = tt.make_tensor_ptr %28, [%c64_i64, %c1024_i64], [%c1_i64, %c64_i64], [%c16_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<16x16xf16>>
    %33 = tt.make_tensor_ptr %28, [%c64_i64, %c1024_i64], [%c1_i64, %c64_i64], [%c32_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<16x16xf16>>
    %34 = tt.make_tensor_ptr %28, [%c64_i64, %c1024_i64], [%c1_i64, %c64_i64], [%c48_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<16x16xf16>>
    %35 = tt.make_tensor_ptr %28, [%c64_i64, %c1024_i64], [%c1_i64, %c64_i64], [%c0_i32, %c16_i32] {order = array<i32: 0, 1>} : <tensor<16x16xf16>>
    %36 = tt.make_tensor_ptr %28, [%c64_i64, %c1024_i64], [%c1_i64, %c64_i64], [%c16_i32, %c16_i32] {order = array<i32: 0, 1>} : <tensor<16x16xf16>>
    %37 = tt.make_tensor_ptr %28, [%c64_i64, %c1024_i64], [%c1_i64, %c64_i64], [%c32_i32, %c16_i32] {order = array<i32: 0, 1>} : <tensor<16x16xf16>>
    %38 = tt.make_tensor_ptr %28, [%c64_i64, %c1024_i64], [%c1_i64, %c64_i64], [%c48_i32, %c16_i32] {order = array<i32: 0, 1>} : <tensor<16x16xf16>>
    %39 = tt.make_tensor_ptr %28, [%c64_i64, %c1024_i64], [%c1_i64, %c64_i64], [%c0_i32, %c32_i32] {order = array<i32: 0, 1>} : <tensor<16x16xf16>>
    %40 = tt.make_tensor_ptr %28, [%c64_i64, %c1024_i64], [%c1_i64, %c64_i64], [%c16_i32, %c32_i32] {order = array<i32: 0, 1>} : <tensor<16x16xf16>>
    %41 = tt.make_tensor_ptr %28, [%c64_i64, %c1024_i64], [%c1_i64, %c64_i64], [%c32_i32, %c32_i32] {order = array<i32: 0, 1>} : <tensor<16x16xf16>>
    %42 = tt.make_tensor_ptr %28, [%c64_i64, %c1024_i64], [%c1_i64, %c64_i64], [%c48_i32, %c32_i32] {order = array<i32: 0, 1>} : <tensor<16x16xf16>>
    %43 = tt.make_tensor_ptr %28, [%c64_i64, %c1024_i64], [%c1_i64, %c64_i64], [%c0_i32, %c48_i32] {order = array<i32: 0, 1>} : <tensor<16x16xf16>>
    %44 = tt.make_tensor_ptr %28, [%c64_i64, %c1024_i64], [%c1_i64, %c64_i64], [%c16_i32, %c48_i32] {order = array<i32: 0, 1>} : <tensor<16x16xf16>>
    %45 = tt.make_tensor_ptr %28, [%c64_i64, %c1024_i64], [%c1_i64, %c64_i64], [%c32_i32, %c48_i32] {order = array<i32: 0, 1>} : <tensor<16x16xf16>>
    %46 = tt.make_tensor_ptr %28, [%c64_i64, %c1024_i64], [%c1_i64, %c64_i64], [%c48_i32, %c48_i32] {order = array<i32: 0, 1>} : <tensor<16x16xf16>>
    %47 = tt.addptr %arg5, %9 : !tt.ptr<f32>, i64
    %48 = tt.make_tensor_ptr %47, [%c1024_i64, %c64_i64], [%c64_i64, %c1_i64], [%13, %c0_i32] {order = array<i32: 1, 0>} : <tensor<8x16xf32>>
    %49 = arith.addi %13, %c8_i32 : i32
    %50 = tt.make_tensor_ptr %47, [%c1024_i64, %c64_i64], [%c64_i64, %c1_i64], [%49, %c0_i32] {order = array<i32: 1, 0>} : <tensor<8x16xf32>>
    %51 = tt.make_tensor_ptr %47, [%c1024_i64, %c64_i64], [%c64_i64, %c1_i64], [%13, %c16_i32] {order = array<i32: 1, 0>} : <tensor<8x16xf32>>
    %52 = tt.make_tensor_ptr %47, [%c1024_i64, %c64_i64], [%c64_i64, %c1_i64], [%49, %c16_i32] {order = array<i32: 1, 0>} : <tensor<8x16xf32>>
    %53 = tt.make_tensor_ptr %47, [%c1024_i64, %c64_i64], [%c64_i64, %c1_i64], [%13, %c32_i32] {order = array<i32: 1, 0>} : <tensor<8x16xf32>>
    %54 = tt.make_tensor_ptr %47, [%c1024_i64, %c64_i64], [%c64_i64, %c1_i64], [%49, %c32_i32] {order = array<i32: 1, 0>} : <tensor<8x16xf32>>
    %55 = tt.make_tensor_ptr %47, [%c1024_i64, %c64_i64], [%c64_i64, %c1_i64], [%13, %c48_i32] {order = array<i32: 1, 0>} : <tensor<8x16xf32>>
    %56 = tt.make_tensor_ptr %47, [%c1024_i64, %c64_i64], [%c64_i64, %c1_i64], [%49, %c48_i32] {order = array<i32: 1, 0>} : <tensor<8x16xf32>>
    // CHECK-LABEL: @_attn_fwd
    // CHECK: scf.for
    // CHECK-COUNT-2: tt.load {{.*}} : !tt.ptr<tensor<16x32xf16>>
    // CHECK-COUNT-4: tt.load {{.*}} : !tt.ptr<tensor<16x16xf16>>
    // CHECK-COUNT-8: tt.dot
    // CHECK-COUNT-4: tt.load {{.*}} : !tt.ptr<tensor<16x16xf16>>
    // CHECK-COUNT-8: tt.dot
    // CHECK-COUNT-4: tt.load {{.*}} : !tt.ptr<tensor<16x16xf16>>
    // CHECK-COUNT-8: tt.dot
    // CHECK-COUNT-4: tt.load {{.*}} : !tt.ptr<tensor<16x16xf16>>
    // CHECK-COUNT-8: tt.dot
    %58 = tt.load %14 {DotIdx = 0 : i32} : !tt.ptr<tensor<16x32xf16>>
    %59 = tt.load %15 {DotIdx = 0 : i32} : !tt.ptr<tensor<16x32xf16>>
    %62:24 = scf.for %arg6 = %c0_i32 to %c1024_i32 step %c64_i32 iter_args(%arg8 = %cst_2, %arg9 = %cst_2, %arg10 = %cst_2, %arg11 = %cst_2, %arg12 = %cst_2, %arg13 = %cst_2, %arg14 = %cst_2, %arg15 = %cst_2, %arg21 = %31, %arg22 = %32, %arg23 = %33, %arg24 = %34, %arg25 = %35, %arg26 = %36, %arg27 = %37, %arg28 = %38, %arg29 = %39, %arg30 = %40, %arg31 = %41, %arg32 = %42, %arg33 = %43, %arg34 = %44, %arg35 = %45, %arg36 = %46) -> (tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>, !tt.ptr<tensor<16x16xf16>>, !tt.ptr<tensor<16x16xf16>>, !tt.ptr<tensor<16x16xf16>>, !tt.ptr<tensor<16x16xf16>>, !tt.ptr<tensor<16x16xf16>>, !tt.ptr<tensor<16x16xf16>>, !tt.ptr<tensor<16x16xf16>>, !tt.ptr<tensor<16x16xf16>>, !tt.ptr<tensor<16x16xf16>>, !tt.ptr<tensor<16x16xf16>>, !tt.ptr<tensor<16x16xf16>>, !tt.ptr<tensor<16x16xf16>>, !tt.ptr<tensor<16x16xf16>>, !tt.ptr<tensor<16x16xf16>>, !tt.ptr<tensor<16x16xf16>>, !tt.ptr<tensor<16x16xf16>>)  : i32 {
      %75 = tt.load %arg21 {DotIdx = 1 : i32} : !tt.ptr<tensor<16x16xf16>>
      %76 = tt.load %arg22 {DotIdx = 1 : i32} : !tt.ptr<tensor<16x16xf16>>
      %77 = tt.load %arg23 {DotIdx = 1 : i32} : !tt.ptr<tensor<16x16xf16>>
      %78 = tt.load %arg24 {DotIdx = 1 : i32} : !tt.ptr<tensor<16x16xf16>>
      %79 = tt.load %arg25 {DotIdx = 1 : i32} : !tt.ptr<tensor<16x16xf16>>
      %80 = tt.load %arg26 {DotIdx = 1 : i32} : !tt.ptr<tensor<16x16xf16>>
      %81 = tt.load %arg27 {DotIdx = 1 : i32} : !tt.ptr<tensor<16x16xf16>>
      %82 = tt.load %arg28 {DotIdx = 1 : i32} : !tt.ptr<tensor<16x16xf16>>
      %83 = tt.load %arg29 {DotIdx = 1 : i32} : !tt.ptr<tensor<16x16xf16>>
      %84 = tt.load %arg30 {DotIdx = 1 : i32} : !tt.ptr<tensor<16x16xf16>>
      %85 = tt.load %arg31 {DotIdx = 1 : i32} : !tt.ptr<tensor<16x16xf16>>
      %86 = tt.load %arg32 {DotIdx = 1 : i32} : !tt.ptr<tensor<16x16xf16>>
      %87 = tt.load %arg33 {DotIdx = 1 : i32} : !tt.ptr<tensor<16x16xf16>>
      %88 = tt.load %arg34 {DotIdx = 1 : i32} : !tt.ptr<tensor<16x16xf16>>
      %89 = tt.load %arg35 {DotIdx = 1 : i32} : !tt.ptr<tensor<16x16xf16>>
      %90 = tt.load %arg36 {DotIdx = 1 : i32} : !tt.ptr<tensor<16x16xf16>>
      %91 = triton_intel_gpu.extract %58[0] : tensor<16x32xf16> -> tensor<8x16xf16>
      %92 = tt.dot %91, %75, %cst_2, inputPrecision = tf32 {"schedule-group" = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %93 = triton_intel_gpu.extract %58[2] : tensor<16x32xf16> -> tensor<8x16xf16>
      %94 = tt.dot %93, %76, %92, inputPrecision = tf32 {"schedule-group" = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %95 = triton_intel_gpu.extract %59[0] : tensor<16x32xf16> -> tensor<8x16xf16>
      %96 = tt.dot %95, %77, %94, inputPrecision = tf32 {"schedule-group" = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %97 = triton_intel_gpu.extract %59[2] : tensor<16x32xf16> -> tensor<8x16xf16>
      %98 = tt.dot %97, %78, %96, inputPrecision = tf32 {"schedule-group" = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %99 = triton_intel_gpu.extract %58[1] : tensor<16x32xf16> -> tensor<8x16xf16>
      %100 = tt.dot %99, %75, %cst_2, inputPrecision = tf32 {"schedule-group" = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %101 = triton_intel_gpu.extract %58[3] : tensor<16x32xf16> -> tensor<8x16xf16>
      %102 = tt.dot %101, %76, %100, inputPrecision = tf32 {"schedule-group" = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %103 = triton_intel_gpu.extract %59[1] : tensor<16x32xf16> -> tensor<8x16xf16>
      %104 = tt.dot %103, %77, %102, inputPrecision = tf32 {"schedule-group" = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %105 = triton_intel_gpu.extract %59[3] : tensor<16x32xf16> -> tensor<8x16xf16>
      %106 = tt.dot %105, %78, %104, inputPrecision = tf32 {"schedule-group" = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %107 = tt.dot %91, %79, %cst_2, inputPrecision = tf32 {"schedule-group" = 1 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %108 = tt.dot %93, %80, %107, inputPrecision = tf32 {"schedule-group" = 1 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %109 = tt.dot %95, %81, %108, inputPrecision = tf32 {"schedule-group" = 1 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %110 = tt.dot %97, %82, %109, inputPrecision = tf32 {"schedule-group" = 1 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %111 = tt.dot %99, %79, %cst_2, inputPrecision = tf32 {"schedule-group" = 1 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %112 = tt.dot %101, %80, %111, inputPrecision = tf32 {"schedule-group" = 1 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %113 = tt.dot %103, %81, %112, inputPrecision = tf32 {"schedule-group" = 1 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %114 = tt.dot %105, %82, %113, inputPrecision = tf32 {"schedule-group" = 1 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %115 = tt.dot %91, %83, %cst_2, inputPrecision = tf32 {"schedule-group" = 2 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %116 = tt.dot %93, %84, %115, inputPrecision = tf32 {"schedule-group" = 2 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %117 = tt.dot %95, %85, %116, inputPrecision = tf32 {"schedule-group" = 2 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %118 = tt.dot %97, %86, %117, inputPrecision = tf32 {"schedule-group" = 2 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %119 = tt.dot %99, %83, %cst_2, inputPrecision = tf32 {"schedule-group" = 2 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %120 = tt.dot %101, %84, %119, inputPrecision = tf32 {"schedule-group" = 2 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %121 = tt.dot %103, %85, %120, inputPrecision = tf32 {"schedule-group" = 2 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %122 = tt.dot %105, %86, %121, inputPrecision = tf32 {"schedule-group" = 2 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %123 = tt.dot %91, %87, %cst_2, inputPrecision = tf32 {"schedule-group" = 3 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %124 = tt.dot %93, %88, %123, inputPrecision = tf32 {"schedule-group" = 3 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %125 = tt.dot %95, %89, %124, inputPrecision = tf32 {"schedule-group" = 3 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %126 = tt.dot %97, %90, %125, inputPrecision = tf32 {"schedule-group" = 3 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %127 = tt.dot %99, %87, %cst_2, inputPrecision = tf32 {"schedule-group" = 3 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %128 = tt.dot %101, %88, %127, inputPrecision = tf32 {"schedule-group" = 3 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %129 = tt.dot %103, %89, %128, inputPrecision = tf32 {"schedule-group" = 3 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %130 = tt.dot %105, %90, %129, inputPrecision = tf32 {"schedule-group" = 3 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>

      %cst_3 = arith.constant dense<1.000000e+00> : tensor<16x16xf16>
      %131 = tt.dot %91, %cst_3, %cst_2, inputPrecision = tf32 {"schedule-group" = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>

      %321 = tt.advance %arg21, [%c0_i32, %c64_i32] : <tensor<16x16xf16>>
      %322 = tt.advance %arg22, [%c0_i32, %c64_i32] : <tensor<16x16xf16>>
      %323 = tt.advance %arg23, [%c0_i32, %c64_i32] : <tensor<16x16xf16>>
      %324 = tt.advance %arg24, [%c0_i32, %c64_i32] : <tensor<16x16xf16>>
      %325 = tt.advance %arg25, [%c0_i32, %c64_i32] : <tensor<16x16xf16>>
      %326 = tt.advance %arg26, [%c0_i32, %c64_i32] : <tensor<16x16xf16>>
      %327 = tt.advance %arg27, [%c0_i32, %c64_i32] : <tensor<16x16xf16>>
      %328 = tt.advance %arg28, [%c0_i32, %c64_i32] : <tensor<16x16xf16>>
      %329 = tt.advance %arg29, [%c0_i32, %c64_i32] : <tensor<16x16xf16>>
      %330 = tt.advance %arg30, [%c0_i32, %c64_i32] : <tensor<16x16xf16>>
      %331 = tt.advance %arg31, [%c0_i32, %c64_i32] : <tensor<16x16xf16>>
      %332 = tt.advance %arg32, [%c0_i32, %c64_i32] : <tensor<16x16xf16>>
      %333 = tt.advance %arg33, [%c0_i32, %c64_i32] : <tensor<16x16xf16>>
      %334 = tt.advance %arg34, [%c0_i32, %c64_i32] : <tensor<16x16xf16>>
      %335 = tt.advance %arg35, [%c0_i32, %c64_i32] : <tensor<16x16xf16>>
      %336 = tt.advance %arg36, [%c0_i32, %c64_i32] : <tensor<16x16xf16>>
      scf.yield %98, %106, %110, %114, %118, %122, %126, %130, %321, %322, %323, %324, %325, %326, %327, %328, %329, %330, %331, %332, %333, %334, %335, %336 : tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>, !tt.ptr<tensor<16x16xf16>>, !tt.ptr<tensor<16x16xf16>>, !tt.ptr<tensor<16x16xf16>>, !tt.ptr<tensor<16x16xf16>>, !tt.ptr<tensor<16x16xf16>>, !tt.ptr<tensor<16x16xf16>>, !tt.ptr<tensor<16x16xf16>>, !tt.ptr<tensor<16x16xf16>>, !tt.ptr<tensor<16x16xf16>>, !tt.ptr<tensor<16x16xf16>>, !tt.ptr<tensor<16x16xf16>>, !tt.ptr<tensor<16x16xf16>>, !tt.ptr<tensor<16x16xf16>>, !tt.ptr<tensor<16x16xf16>>, !tt.ptr<tensor<16x16xf16>>, !tt.ptr<tensor<16x16xf16>>
    }
    %cst_1 = arith.constant dense<1.000000e+00> : tensor<8x16xf32>
    // CHECK-COUNT-8: arith.divf {{.*}} fastmath<fast>
    %67 = arith.divf %62#1, %cst_1 : tensor<8x16xf32>
    %68 = arith.divf %62#2, %cst_1 : tensor<8x16xf32>
    %69 = arith.divf %62#3, %cst_1 : tensor<8x16xf32>
    %70 = arith.divf %62#4, %cst_1 : tensor<8x16xf32>
    %71 = arith.divf %62#5, %cst_1 : tensor<8x16xf32>
    %72 = arith.divf %62#6, %cst_1 : tensor<8x16xf32>
    %73 = arith.divf %62#7, %cst_1 : tensor<8x16xf32>
    %74 = arith.divf %62#0, %cst_1 : tensor<8x16xf32>
    tt.store %48, %67 : !tt.ptr<tensor<8x16xf32>>
    tt.store %50, %68 : !tt.ptr<tensor<8x16xf32>>
    tt.store %51, %69 : !tt.ptr<tensor<8x16xf32>>
    tt.store %52, %70 : !tt.ptr<tensor<8x16xf32>>
    tt.store %53, %71 : !tt.ptr<tensor<8x16xf32>>
    tt.store %54, %72 : !tt.ptr<tensor<8x16xf32>>
    tt.store %55, %73 : !tt.ptr<tensor<8x16xf32>>
    tt.store %56, %74 : !tt.ptr<tensor<8x16xf32>>
    tt.return
  }
}
