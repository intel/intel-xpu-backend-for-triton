// RUN: env TRITON_INTEL_ENABLE_INSTR_SCHED=1 triton-opt %s -split-input-file -tritonintelgpu-schedule-load | FileCheck %s --check-prefixes=CHECK,SINK-ACROSS-REGIONS
// RUN: env TRITON_INTEL_ENABLE_INSTR_SCHED=1 TRITON_INTEL_DO_NOT_SINK_INSTR_ACROSS_RGN=1 triton-opt %s -split-input-file -tritonintelgpu-schedule-load | FileCheck %s --check-prefixes=CHECK,DO-NOT-SINK-ACROSS-REGIONS

// -----
// COM: Inst Schedule for Flash Attention case

module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func public @_attn_fwd(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: f32, %arg4: !tt.ptr<f32>, %arg5: !tt.ptr<f32>) {
    // CHECK-LABEL: @_attn_fwd
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
    // DO-NOT-SINK-ACROSS-REGIONS-COUNT-2: tt.load {{.*}} : !tt.ptr<tensor<16x32xf16>>
    // CHECK: scf.for
    // SINK-ACROSS-REGIONS-COUNT-2: tt.load {{.*}} : !tt.ptr<tensor<16x32xf16>>
    // CHECK-COUNT-4: tt.load {{.*}} : !tt.ptr<tensor<16x16xf16>>
    // CHECK-COUNT-8: tt.dot {{.*}} {"schedule-group" = 0 : i32}
    // CHECK-COUNT-4: tt.load {{.*}} : !tt.ptr<tensor<16x16xf16>>
    // CHECK-COUNT-8: tt.dot {{.*}} {"schedule-group" = 1 : i32}
    // CHECK-COUNT-4: tt.load {{.*}} : !tt.ptr<tensor<16x16xf16>>
    // CHECK-COUNT-8: tt.dot {{.*}} {"schedule-group" = 2 : i32}
    // CHECK-COUNT-4: tt.load {{.*}} : !tt.ptr<tensor<16x16xf16>>
    // CHECK-COUNT-8: tt.dot {{.*}} {"schedule-group" = 3 : i32}
    // CHECK-COUNT-16: tt.advance {{.*}} : <tensor<16x16xf16>>
    %58 = tt.load %14 {DotIdx = 0 : i32} : !tt.ptr<tensor<16x32xf16>>
    %59 = tt.load %15 {DotIdx = 0 : i32} : !tt.ptr<tensor<16x32xf16>>
    %62:24 = scf.for %arg6 = %c0_i32 to %c1024_i32 step %c64_i32 iter_args(%arg8 = %cst_2, %arg9 = %cst_2, %arg10 = %cst_2, %arg11 = %cst_2, %arg12 = %cst_2, %arg13 = %cst_2, %arg14 = %cst_2, %arg15 = %cst_2, %arg21 = %31, %arg22 = %32, %arg23 = %33, %arg24 = %34, %arg25 = %35, %arg26 = %36, %arg27 = %37, %arg28 = %38, %arg29 = %39, %arg30 = %40, %arg31 = %41, %arg32 = %42, %arg33 = %43, %arg34 = %44, %arg35 = %45, %arg36 = %46)
           -> (tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>, !tt.ptr<tensor<16x16xf16>>, !tt.ptr<tensor<16x16xf16>>, !tt.ptr<tensor<16x16xf16>>, !tt.ptr<tensor<16x16xf16>>, !tt.ptr<tensor<16x16xf16>>, !tt.ptr<tensor<16x16xf16>>, !tt.ptr<tensor<16x16xf16>>, !tt.ptr<tensor<16x16xf16>>, !tt.ptr<tensor<16x16xf16>>, !tt.ptr<tensor<16x16xf16>>, !tt.ptr<tensor<16x16xf16>>, !tt.ptr<tensor<16x16xf16>>, !tt.ptr<tensor<16x16xf16>>, !tt.ptr<tensor<16x16xf16>>, !tt.ptr<tensor<16x16xf16>>, !tt.ptr<tensor<16x16xf16>>) : i32 {
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
      %91 = ttig.extract %58[0] : tensor<16x32xf16> -> tensor<8x16xf16>
      %92 = tt.dot %91, %75, %cst_2, inputPrecision = tf32 {"schedule-group" = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %93 = ttig.extract %58[2] : tensor<16x32xf16> -> tensor<8x16xf16>
      %94 = tt.dot %93, %76, %92, inputPrecision = tf32 {"schedule-group" = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %95 = ttig.extract %59[0] : tensor<16x32xf16> -> tensor<8x16xf16>
      %96 = tt.dot %95, %77, %94, inputPrecision = tf32 {"schedule-group" = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %97 = ttig.extract %59[2] : tensor<16x32xf16> -> tensor<8x16xf16>
      %98 = tt.dot %97, %78, %96, inputPrecision = tf32 {"schedule-group" = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %99 = ttig.extract %58[1] : tensor<16x32xf16> -> tensor<8x16xf16>
      %100 = tt.dot %99, %75, %cst_2, inputPrecision = tf32 {"schedule-group" = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %101 = ttig.extract %58[3] : tensor<16x32xf16> -> tensor<8x16xf16>
      %102 = tt.dot %101, %76, %100, inputPrecision = tf32 {"schedule-group" = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %103 = ttig.extract %59[1] : tensor<16x32xf16> -> tensor<8x16xf16>
      %104 = tt.dot %103, %77, %102, inputPrecision = tf32 {"schedule-group" = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %105 = ttig.extract %59[3] : tensor<16x32xf16> -> tensor<8x16xf16>
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
    %67 = arith.divf %62#1, %cst_1 : tensor<8x16xf32>
    %68 = arith.divf %62#2, %cst_1 : tensor<8x16xf32>
    %69 = arith.divf %62#3, %cst_1 : tensor<8x16xf32>
    %70 = arith.divf %62#4, %cst_1 : tensor<8x16xf32>
    %71 = arith.divf %62#5, %cst_1 : tensor<8x16xf32>
    %72 = arith.divf %62#6, %cst_1 : tensor<8x16xf32>
    %73 = arith.divf %62#7, %cst_1 : tensor<8x16xf32>
    %74 = arith.divf %62#0, %cst_1 : tensor<8x16xf32>
    tt.return
  }
}

// -----
// COM: Inst Schedule for GEMM case

module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func public @gemm_with_block_pointers(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2: !tt.ptr<f32>) {
    // CHECK-LABEL: @gemm_with_block_pointers
    %c3_i32 = arith.constant 3 : i32
    %c7_i32 = arith.constant 7 : i32
    %c63_i32 = arith.constant 63 : i32
    %c48_i32 = arith.constant 48 : i32
    %c24_i32 = arith.constant 24 : i32
    %c64_i32 = arith.constant 64 : i32
    %c16_i32 = arith.constant 16 : i32
    %c4096_i32 = arith.constant 4096 : i32
    %c8_i32 = arith.constant 8 : i32
    %c4_i32 = arith.constant 4 : i32
    %c256_i32 = arith.constant 256 : i32
    %c4096_i64 = arith.constant 4096 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<8x16xf32>
    %0 = gpu.subgroup_id : index
    %1 = arith.index_cast %0 : index to i32
    %2 = tt.get_program_id x : i32
    %3 = arith.divsi %2, %c64_i32 : i32
    %4 = arith.muli %3, %c4_i32 : i32
    %5 = arith.subi %c16_i32, %4 : i32
    %6 = arith.minsi %5, %c4_i32 : i32
    %7 = arith.remsi %2, %6 : i32
    %8 = arith.addi %4, %7 : i32
    %9 = arith.andi %2, %c63_i32 : i32
    %10 = arith.divsi %9, %6 : i32
    %11 = arith.muli %8, %c256_i32 : i32
    %12 = arith.muli %1, %c8_i32 : i32
    %13 = arith.addi %12, %11 : i32
    %14 = tt.make_tensor_ptr %arg0, [%c4096_i64, %c4096_i64], [%c4096_i64, %c1_i64], [%13, %c0_i32] {order = array<i32: 1, 0>} : <tensor<8x32xbf16>>
    ttig.prefetch %14 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<8x32xbf16>>
    %15 = tt.advance %14, [%c0_i32, %c32_i32] : <tensor<8x32xbf16>>
    ttig.prefetch %15 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<8x32xbf16>>
    %16 = tt.advance %15, [%c0_i32, %c32_i32] : <tensor<8x32xbf16>>
    %17 = arith.divsi %1, %c4_i32 : i32
    %18 = arith.andi %17, %c7_i32 : i32
    %19 = arith.muli %18, %c32_i32 : i32
    %20 = arith.addi %19, %11 : i32
    %21 = tt.make_tensor_ptr %arg0, [%c4096_i64, %c4096_i64], [%c4096_i64, %c1_i64], [%20, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x32xbf16>>
    %22 = arith.muli %10, %c256_i32 : i32
    %23 = arith.divsi %1, %c8_i32 : i32
    %24 = arith.andi %23, %c3_i32 : i32
    %25 = arith.muli %24, %c8_i32 : i32
    %26 = arith.andi %1, %c7_i32 : i32
    %27 = arith.muli %26, %c32_i32 : i32
    %28 = arith.addi %27, %22 : i32
    %29 = tt.make_tensor_ptr %arg1, [%c4096_i64, %c4096_i64], [%c4096_i64, %c1_i64], [%25, %28] {order = array<i32: 1, 0>} : <tensor<8x32xbf16>>
    ttig.prefetch %29 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<8x32xbf16>>
    %30 = tt.advance %29, [%c32_i32, %c0_i32] : <tensor<8x32xbf16>>
    ttig.prefetch %30 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<8x32xbf16>>
    %31 = tt.advance %30, [%c32_i32, %c0_i32] : <tensor<8x32xbf16>>
    %32 = arith.andi %1, %c3_i32 : i32
    %33 = arith.muli %32, %c64_i32 : i32
    %34 = arith.addi %33, %22 : i32
    %35 = tt.make_tensor_ptr %arg1, [%c4096_i64, %c4096_i64], [%c4096_i64, %c1_i64], [%c0_i32, %34] {order = array<i32: 1, 0>} : <tensor<32x32xbf16>>
    %36 = arith.addi %34, %c32_i32 : i32
    %37 = tt.make_tensor_ptr %arg1, [%c4096_i64, %c4096_i64], [%c4096_i64, %c1_i64], [%c0_i32, %36] {order = array<i32: 1, 0>} : <tensor<32x32xbf16>>
    // CHECK: scf.for
    // CHECK-COUNT-2: ttig.prefetch {{.*}} : !tt.ptr<tensor<8x32xbf16>>
    // CHECK: tt.load {{.*}} : !tt.ptr<tensor<32x32xbf16>>
    // CHECK-COUNT-8: ttig.extract {{.*}} : tensor<32x32xbf16> -> tensor<8x16xbf16>
    // CHECK: tt.load {{.*}} : !tt.ptr<tensor<32x32xbf16>>
    // CHECK-COUNT-2: ttig.extract {{.*}} : tensor<32x32xbf16> -> tensor<16x16xbf16>
    // CHECK-COUNT-8: tt.dot {{.*}} {"schedule-group" = 0 : i32}
    // CHECK-COUNT-2: ttig.extract {{.*}} : tensor<32x32xbf16> -> tensor<16x16xbf16>
    // CHECK-COUNT-8: tt.dot {{.*}} {"schedule-group" = 1 : i32}
    // CHECK-COUNT-1: tt.load {{.*}} : !tt.ptr<tensor<32x32xbf16>>
    // CHECK-COUNT-2: ttig.extract {{.*}} : tensor<32x32xbf16> -> tensor<16x16xbf16>
    // CHECK-COUNT-8: tt.dot {{.*}} {"schedule-group" = 2 : i32}
    // CHECK-COUNT-2: ttig.extract {{.*}} : tensor<32x32xbf16> -> tensor<16x16xbf16>
    // CHECK-COUNT-8: tt.dot {{.*}} {"schedule-group" = 3 : i32}
    %38:21 = scf.for %arg3 = %c0_i32 to %c4096_i32 step %c32_i32 iter_args(%arg4 = %cst, %arg5 = %cst, %arg6 = %cst, %arg7 = %cst, %arg8 = %cst, %arg9 = %cst, %arg10 = %cst, %arg11 = %cst, %arg12 = %cst, %arg13 = %cst, %arg14 = %cst, %arg15 = %cst, %arg16 = %cst, %arg17 = %cst, %arg18 = %cst, %arg19 = %cst, %arg20 = %21, %arg21 = %35, %arg22 = %37, %arg23 = %16, %arg24 = %31) -> (tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>, !tt.ptr<tensor<32x32xbf16>>, !tt.ptr<tensor<32x32xbf16>>, !tt.ptr<tensor<32x32xbf16>>, !tt.ptr<tensor<8x32xbf16>>, !tt.ptr<tensor<8x32xbf16>>)  : i32 {
      %60 = tt.load %arg20 {DotIdx = 0 : i32, boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<32x32xbf16>>
      %61 = tt.load %arg21 {DotIdx = 1 : i32, boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<32x32xbf16>>
      %62 = tt.load %arg22 {DotIdx = 1 : i32, boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<32x32xbf16>>
      ttig.prefetch %arg23 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<8x32xbf16>>
      ttig.prefetch %arg24 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<8x32xbf16>>
      %63 = ttig.extract %60[0] : tensor<32x32xbf16> -> tensor<8x16xbf16>
      %64 = ttig.extract %61[0] : tensor<32x32xbf16> -> tensor<16x16xbf16>
      %65 = tt.dot %63, %64, %arg4, inputPrecision = tf32 {"schedule-group" = 0 : i32} : tensor<8x16xbf16> * tensor<16x16xbf16> -> tensor<8x16xf32>
      %66 = ttig.extract %60[4] : tensor<32x32xbf16> -> tensor<8x16xbf16>
      %67 = ttig.extract %61[1] : tensor<32x32xbf16> -> tensor<16x16xbf16>
      %68 = tt.dot %66, %67, %65, inputPrecision = tf32 {"schedule-group" = 0 : i32} : tensor<8x16xbf16> * tensor<16x16xbf16> -> tensor<8x16xf32>
      %69 = ttig.extract %60[1] : tensor<32x32xbf16> -> tensor<8x16xbf16>
      %70 = tt.dot %69, %64, %arg5, inputPrecision = tf32 {"schedule-group" = 0 : i32} : tensor<8x16xbf16> * tensor<16x16xbf16> -> tensor<8x16xf32>
      %71 = ttig.extract %60[5] : tensor<32x32xbf16> -> tensor<8x16xbf16>
      %72 = tt.dot %71, %67, %70, inputPrecision = tf32 {"schedule-group" = 0 : i32} : tensor<8x16xbf16> * tensor<16x16xbf16> -> tensor<8x16xf32>
      %73 = ttig.extract %60[2] : tensor<32x32xbf16> -> tensor<8x16xbf16>
      %74 = tt.dot %73, %64, %arg6, inputPrecision = tf32 {"schedule-group" = 0 : i32} : tensor<8x16xbf16> * tensor<16x16xbf16> -> tensor<8x16xf32>
      %75 = ttig.extract %60[6] : tensor<32x32xbf16> -> tensor<8x16xbf16>
      %76 = tt.dot %75, %67, %74, inputPrecision = tf32 {"schedule-group" = 0 : i32} : tensor<8x16xbf16> * tensor<16x16xbf16> -> tensor<8x16xf32>
      %77 = ttig.extract %60[3] : tensor<32x32xbf16> -> tensor<8x16xbf16>
      %78 = tt.dot %77, %64, %arg7, inputPrecision = tf32 {"schedule-group" = 0 : i32} : tensor<8x16xbf16> * tensor<16x16xbf16> -> tensor<8x16xf32>
      %79 = ttig.extract %60[7] : tensor<32x32xbf16> -> tensor<8x16xbf16>
      %80 = tt.dot %79, %67, %78, inputPrecision = tf32 {"schedule-group" = 0 : i32} : tensor<8x16xbf16> * tensor<16x16xbf16> -> tensor<8x16xf32>
      %81 = ttig.extract %61[2] : tensor<32x32xbf16> -> tensor<16x16xbf16>
      %82 = tt.dot %63, %81, %arg8, inputPrecision = tf32 {"schedule-group" = 1 : i32} : tensor<8x16xbf16> * tensor<16x16xbf16> -> tensor<8x16xf32>
      %83 = ttig.extract %61[3] : tensor<32x32xbf16> -> tensor<16x16xbf16>
      %84 = tt.dot %66, %83, %82, inputPrecision = tf32 {"schedule-group" = 1 : i32} : tensor<8x16xbf16> * tensor<16x16xbf16> -> tensor<8x16xf32>
      %85 = tt.dot %69, %81, %arg9, inputPrecision = tf32 {"schedule-group" = 1 : i32} : tensor<8x16xbf16> * tensor<16x16xbf16> -> tensor<8x16xf32>
      %86 = tt.dot %71, %83, %85, inputPrecision = tf32 {"schedule-group" = 1 : i32} : tensor<8x16xbf16> * tensor<16x16xbf16> -> tensor<8x16xf32>
      %87 = tt.dot %73, %81, %arg10, inputPrecision = tf32 {"schedule-group" = 1 : i32} : tensor<8x16xbf16> * tensor<16x16xbf16> -> tensor<8x16xf32>
      %88 = tt.dot %75, %83, %87, inputPrecision = tf32 {"schedule-group" = 1 : i32} : tensor<8x16xbf16> * tensor<16x16xbf16> -> tensor<8x16xf32>
      %89 = tt.dot %77, %81, %arg11, inputPrecision = tf32 {"schedule-group" = 1 : i32} : tensor<8x16xbf16> * tensor<16x16xbf16> -> tensor<8x16xf32>
      %90 = tt.dot %79, %83, %89, inputPrecision = tf32 {"schedule-group" = 1 : i32} : tensor<8x16xbf16> * tensor<16x16xbf16> -> tensor<8x16xf32>
      %91 = ttig.extract %62[0] : tensor<32x32xbf16> -> tensor<16x16xbf16>
      %92 = tt.dot %63, %91, %arg12, inputPrecision = tf32 {"schedule-group" = 2 : i32} : tensor<8x16xbf16> * tensor<16x16xbf16> -> tensor<8x16xf32>
      %93 = ttig.extract %62[1] : tensor<32x32xbf16> -> tensor<16x16xbf16>
      %94 = tt.dot %66, %93, %92, inputPrecision = tf32 {"schedule-group" = 2 : i32} : tensor<8x16xbf16> * tensor<16x16xbf16> -> tensor<8x16xf32>
      %95 = tt.dot %69, %91, %arg13, inputPrecision = tf32 {"schedule-group" = 2 : i32} : tensor<8x16xbf16> * tensor<16x16xbf16> -> tensor<8x16xf32>
      %96 = tt.dot %71, %93, %95, inputPrecision = tf32 {"schedule-group" = 2 : i32} : tensor<8x16xbf16> * tensor<16x16xbf16> -> tensor<8x16xf32>
      %97 = tt.dot %73, %91, %arg14, inputPrecision = tf32 {"schedule-group" = 2 : i32} : tensor<8x16xbf16> * tensor<16x16xbf16> -> tensor<8x16xf32>
      %98 = tt.dot %75, %93, %97, inputPrecision = tf32 {"schedule-group" = 2 : i32} : tensor<8x16xbf16> * tensor<16x16xbf16> -> tensor<8x16xf32>
      %99 = tt.dot %77, %91, %arg15, inputPrecision = tf32 {"schedule-group" = 2 : i32} : tensor<8x16xbf16> * tensor<16x16xbf16> -> tensor<8x16xf32>
      %100 = tt.dot %79, %93, %99, inputPrecision = tf32 {"schedule-group" = 2 : i32} : tensor<8x16xbf16> * tensor<16x16xbf16> -> tensor<8x16xf32>
      %101 = ttig.extract %62[2] : tensor<32x32xbf16> -> tensor<16x16xbf16>
      %102 = tt.dot %63, %101, %arg16, inputPrecision = tf32 {"schedule-group" = 3 : i32} : tensor<8x16xbf16> * tensor<16x16xbf16> -> tensor<8x16xf32>
      %103 = ttig.extract %62[3] : tensor<32x32xbf16> -> tensor<16x16xbf16>
      %104 = tt.dot %66, %103, %102, inputPrecision = tf32 {"schedule-group" = 3 : i32} : tensor<8x16xbf16> * tensor<16x16xbf16> -> tensor<8x16xf32>
      %105 = tt.dot %69, %101, %arg17, inputPrecision = tf32 {"schedule-group" = 3 : i32} : tensor<8x16xbf16> * tensor<16x16xbf16> -> tensor<8x16xf32>
      %106 = tt.dot %71, %103, %105, inputPrecision = tf32 {"schedule-group" = 3 : i32} : tensor<8x16xbf16> * tensor<16x16xbf16> -> tensor<8x16xf32>
      %107 = tt.dot %73, %101, %arg18, inputPrecision = tf32 {"schedule-group" = 3 : i32} : tensor<8x16xbf16> * tensor<16x16xbf16> -> tensor<8x16xf32>
      %108 = tt.dot %75, %103, %107, inputPrecision = tf32 {"schedule-group" = 3 : i32} : tensor<8x16xbf16> * tensor<16x16xbf16> -> tensor<8x16xf32>
      %109 = tt.dot %77, %101, %arg19, inputPrecision = tf32 {"schedule-group" = 3 : i32} : tensor<8x16xbf16> * tensor<16x16xbf16> -> tensor<8x16xf32>
      %110 = tt.dot %79, %103, %109, inputPrecision = tf32 {"schedule-group" = 3 : i32} : tensor<8x16xbf16> * tensor<16x16xbf16> -> tensor<8x16xf32>
      %111 = tt.advance %arg23, [%c0_i32, %c32_i32] : <tensor<8x32xbf16>>
      %112 = tt.advance %arg20, [%c0_i32, %c32_i32] {DotIdx = 0 : i32} : <tensor<32x32xbf16>>
      %113 = tt.advance %arg24, [%c32_i32, %c0_i32] : <tensor<8x32xbf16>>
      %114 = tt.advance %arg21, [%c32_i32, %c0_i32] {DotIdx = 1 : i32} : <tensor<32x32xbf16>>
      %115 = tt.advance %arg22, [%c32_i32, %c0_i32] {DotIdx = 1 : i32} : <tensor<32x32xbf16>>
      scf.yield %68, %72, %76, %80, %84, %86, %88, %90, %94, %96, %98, %100, %104, %106, %108, %110, %112, %114, %115, %111, %113 : tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>, !tt.ptr<tensor<32x32xbf16>>, !tt.ptr<tensor<32x32xbf16>>, !tt.ptr<tensor<32x32xbf16>>, !tt.ptr<tensor<8x32xbf16>>, !tt.ptr<tensor<8x32xbf16>>
    }
    tt.return
  }
}

// -----

tt.func public @test(%arg0: !tt.ptr<tensor<16x16xf16>>, %arg1: !tt.ptr<tensor<8x32xf16>>) {
  %lb = arith.constant 0 : i32
  %ub = tt.get_program_id x : i32
  %st = arith.constant 32 : i32
  %zero = arith.constant dense<0.000000e+00> : tensor<8x16xf32>
  %common = tt.load %arg1 {DotIdx = 0 : i32} : !tt.ptr<tensor<8x32xf16>>
  // COM: Check %common is not moved in the loop.
  // CHECK: tt.load %arg1
  // CHECK-COUNT-2: scf.for
  scf.for %iv0 = %lb to %ub step %st : i32 {
    %load1 = tt.load %arg0 {DotIdx = 1 : i32} : !tt.ptr<tensor<16x16xf16>>
    %extract1 = ttig.extract %common[0] : tensor<8x32xf16> -> tensor<8x16xf16>
    %dot1 = tt.dot %extract1, %load1, %zero, inputPrecision = tf32 {"schedule-group" = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
  }
  scf.for %iv1 = %lb to %ub step %st : i32 {
    %load2 = tt.load %arg0 {DotIdx = 1 : i32} : !tt.ptr<tensor<16x16xf16>>
    %extract2 = ttig.extract %common[0] : tensor<8x32xf16> -> tensor<8x16xf16>
    %dot2 = tt.dot %extract2, %load2, %zero, inputPrecision = tf32 {"schedule-group" = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
  }
  tt.return
}
