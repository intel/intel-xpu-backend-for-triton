// RUN: triton-opt %s -split-input-file -tritonintelgpu-reduce-variable-liveness -cse | FileCheck %s

// CHECK: #[[$DPAS:.+]] = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {ttig.support_2d_block_io, "ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @matmul_kernel_small_tensor(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    // CHECK-LABEL:   tt.func @matmul_kernel_small_tensor
    // COM: This test verifies that that tensor whose size is under the defined threshold are not moved.
    %cst = arith.constant dense<0.000000e+00> : tensor<16x256xf32, #dpas>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    %0 = tt.make_tensor_descriptor %arg0, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <f16>, <tensor<16x64xf16>>
    // CHECK:  tt.descriptor_load {{.*}} : !tt.tensordesc<tensor<16x64xf16>> -> tensor<16x64xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>
    %1 = tt.make_tensor_descriptor %arg1, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <f16>, <tensor<64x256xf16>>
    %2 = tt.descriptor_load %0[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<16x64xf16>> -> tensor<16x64xf16, #dot0>
    %3 = arith.muli %c64_i32, %c0_i32 : i32
    ttig.descriptor_prefetch %1[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<64x256xf16>>
    %4:2 = scf.for %arg2 = %c0_i32 to %c64_i32 step %c64_i32 iter_args(%arg3 = %cst, %arg4 = %c0_i32) -> (tensor<16x256xf32, #dpas>, i32)  : i32 {
      // CHECK:  scf.for
      // CHECK-NOT:  tt.descriptor_load {{.*}} : !tt.tensordesc<tensor<16x64xf16>> -> tensor<16x64xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>
      %7 = arith.addi %arg4, %c64_i32 : i32
      ttig.descriptor_prefetch %1[%7, %c0_i32] : !tt.tensordesc<tensor<64x256xf16>>
      %8 = tt.descriptor_load %1[%arg4, %c0_i32] : !tt.tensordesc<tensor<64x256xf16>> -> tensor<64x256xf16, #dot1>
      %9 = tt.dot %2, %8, %arg3, inputPrecision = tf32 : tensor<16x64xf16, #dot0> * tensor<64x256xf16, #dot1> -> tensor<16x256xf32, #dpas>
      scf.yield %9, %7 : tensor<16x256xf32, #dpas>, i32
    }
    tt.return
  }
}

// -----

// CHECK: #[[$DPAS:.+]] = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {ttig.support_2d_block_io, "ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @matmul_kernel_no_candidate_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    // CHECK-LABEL:   tt.func @matmul_kernel_no_candidate_load
    // COM: This test checks that loads are not moved if the total size of "in variables" are under the defined threshold.
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #dpas>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    %0 = tt.make_tensor_descriptor %arg0, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <f16>, <tensor<128x256xf16>>
    // CHECK:  tt.descriptor_load {{.*}} : !tt.tensordesc<tensor<128x256xf16>> -> tensor<128x256xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>
    %1 = tt.make_tensor_descriptor %arg1, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <f16>, <tensor<256x256xf16>>
    %2 = tt.descriptor_load %0[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<128x256xf16>> -> tensor<128x256xf16, #dot0>
    %3 = arith.muli %c64_i32, %c0_i32 : i32
    ttig.descriptor_prefetch %1[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<256x256xf16>>
    %4:2 = scf.for %arg2 = %c0_i32 to %c64_i32 step %c64_i32 iter_args(%arg3 = %cst, %arg4 = %c0_i32) -> (tensor<128x256xf32, #dpas>, i32)  : i32 {
      // CHECK:  scf.for
      // CHECK-NOT:  tt.descriptor_load {{.*}} : !tt.tensordesc<tensor<128x256xf16>> -> tensor<128x256xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>
      %7 = arith.addi %arg4, %c64_i32 : i32
      ttig.descriptor_prefetch %1[%7, %c0_i32] : !tt.tensordesc<tensor<256x256xf16>>
      %8 = tt.descriptor_load %1[%arg4, %c0_i32] : !tt.tensordesc<tensor<256x256xf16>> -> tensor<256x256xf16, #dot1>
      %9 = tt.dot %2, %8, %arg3, inputPrecision = tf32 : tensor<128x256xf16, #dot0> * tensor<256x256xf16, #dot1> -> tensor<128x256xf32, #dpas>
      scf.yield %9, %7 : tensor<128x256xf32, #dpas>, i32
    }
    tt.return
  }
}


// -----

// CHECK: #[[$DPAS:.+]] = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 2], repCluster = [2, 2], A = [16, 16], B = [16, 32], C = [16, 32]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 2], repCluster = [2, 2], A = [16, 16], B = [16, 32], C = [16, 32]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bfloat16_conversion, ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_2d_block_io, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
  tt.func @_attn_fwd(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: f32, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    // CHECK-LABEL:   tt.func @_attn_fwd
    // COM: This test checks that the Q matrix load is moved inside the loop for the attention kernel.
    %c8192_i64 = arith.constant 8192 : i64
    %c128_i32 = arith.constant 128 : i32
    %c128_i64 = arith.constant 128 : i64
    %c64_i64 = arith.constant 64 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 1.44269502 : f32
    %c64_i32 = arith.constant 64 : i32
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %cst_1 = arith.constant dense<0xFF800000> : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<512x128xf32, #mma>
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id z : i32
    %2 = arith.extsi %1 : i32 to i64
    %3 = arith.muli %2, %c8192_i64 : i64
    %4 = tt.addptr %arg0, %3 : !tt.ptr<f16>, i64
    %5 = arith.muli %0, %c128_i32 : i32
    // Q descriptor (row_major)
    %6 = tt.make_tensor_descriptor %4, [%c128_i32, %c64_i32], [%c64_i64, %c1_i64] : <f16>, <tensor<512x128xf16>>
    %7 = tt.addptr %arg2, %3 : !tt.ptr<f16>, i64
    // V descriptor (row_major)
    %8 = tt.make_tensor_descriptor %7, [%c128_i32, %c64_i32], [%c64_i64, %c1_i64] : <f16>, <tensor<128x128xf16>>
    %9 = tt.addptr %arg1, %3 : !tt.ptr<f16>, i64
    // K descriptor (row_major descriptor, loaded with column_major attribute for transpose)
    %10 = tt.make_tensor_descriptor %9, [%c128_i32, %c64_i32], [%c64_i64, %c1_i64] : <f16>, <tensor<128x128xf16>>
    %11 = tt.addptr %arg5, %3 : !tt.ptr<f32>, i64
    // Output descriptor (row_major): shape [128, 64], strides [64, 1]
    %12 = tt.make_tensor_descriptor %11, [%c128_i32, %c64_i32], [%c64_i64, %c1_i64] : <f32>, <tensor<512x128xf32>>
    %13 = arith.mulf %arg3, %cst : f32
    // CHECK:      %[[OFFSET0:.*]] = arith.muli {{.*}}, {{.*}} : i32
    // CHECK:      ttig.descriptor_prefetch %{{.*}}[%[[OFFSET0]], %{{.*}}] {{.*}} : !tt.tensordesc<tensor<512x128xf16>>
    // CHECK-NOT:  tt.descriptor_load {{.*}} : !tt.tensordesc<tensor<512x128xf16>>
    %14 = tt.descriptor_load %6[%5, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<512x128xf16>> -> tensor<512x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %15 = tt.splat %13 : f32 -> tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %16 = tt.splat %13 : f32 -> tensor<512x128xf32, #mma>
    %17:4 = scf.for %arg6 = %c0_i32 to %c128_i32 step %c64_i32 iter_args(%arg7 = %cst_0, %arg8 = %cst_2, %arg9 = %cst_1, %arg10 = %c0_i32) -> (tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<512x128xf32, #mma>, tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>, i32)  : i32 {
      // CHECK:  scf.for
      // CHECK:  tt.descriptor_load %{{.*}}[%[[OFFSET0]], %{{.*}}] {{.*}} : !tt.tensordesc<tensor<512x128xf16>> -> tensor<512x128xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>
      %21 = tt.descriptor_load %10[%arg10, %c0_i32] {ttig.block_io = "column_major"} : !tt.tensordesc<tensor<128x128xf16>> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %22 = tt.dot %14, %21, %cst_2, inputPrecision = tf32 : tensor<512x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<512x128xf32, #mma>
      %23 = "tt.reduce"(%22) <{axis = 1 : i32}> ({
      ^bb0(%arg12: f32, %arg13: f32):
        %45 = arith.maxnumf %arg12, %arg13 : f32
        tt.reduce.return %45 : f32
      }) : (tensor<512x128xf32, #mma>) -> tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %24 = arith.mulf %23, %15 : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %25 = arith.maxnumf %arg9, %24 : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %26 = arith.mulf %22, %16 : tensor<512x128xf32, #mma>
      %27 = tt.expand_dims %25 {axis = 1 : i32} : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<512x1xf32, #mma>
      %28 = tt.broadcast %27 : tensor<512x1xf32, #mma> -> tensor<512x128xf32, #mma>
      %29 = arith.subf %26, %28 : tensor<512x128xf32, #mma>
      %30 = math.exp2 %29 : tensor<512x128xf32, #mma>
      %31 = "tt.reduce"(%30) <{axis = 1 : i32}> ({
      ^bb0(%arg12: f32, %arg13: f32):
        %45 = arith.addf %arg12, %arg13 : f32
        tt.reduce.return %45 : f32
      }) : (tensor<512x128xf32, #mma>) -> tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %32 = arith.subf %arg9, %25 : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %33 = math.exp2 %32 : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %34 = arith.mulf %arg7, %33 : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %35 = arith.addf %34, %31 : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %36 = tt.expand_dims %33 {axis = 1 : i32} : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<512x1xf32, #mma>
      %37 = tt.broadcast %36 : tensor<512x1xf32, #mma> -> tensor<512x128xf32, #mma>
      %38 = arith.mulf %arg8, %37 : tensor<512x128xf32, #mma>
      %39 = tt.descriptor_load %8[%arg10, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<128x128xf16>> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %40 = arith.truncf %30 : tensor<512x128xf32, #mma> to tensor<512x128xf16, #mma>
      %41 = ttg.convert_layout %40 : tensor<512x128xf16, #mma> -> tensor<512x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %42 = tt.dot %41, %39, %38, inputPrecision = tf32 : tensor<512x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<512x128xf32, #mma>
      %43 = arith.addi %arg10, %c64_i32 : i32
      scf.yield %35, %42, %25, %43 : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<512x128xf32, #mma>, tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>, i32
    }
    %18 = tt.expand_dims %17#0 {axis = 1 : i32} : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<512x1xf32, #mma>
    %19 = tt.broadcast %18 : tensor<512x1xf32, #mma> -> tensor<512x128xf32, #mma>
    %20 = arith.divf %17#1, %19 : tensor<512x128xf32, #mma>
    tt.descriptor_store %12[%5, %c0_i32], %20 : !tt.tensordesc<tensor<512x128xf32>>, tensor<512x128xf32, #mma>
    tt.return
  }
}

// -----

// CHECK: #[[$DPAS1:.+]] = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 2], repCluster = [2, 2], A = [16, 16], B = [16, 32], C = [16, 32]}>
#mma1 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 2], repCluster = [2, 2], A = [16, 16], B = [16, 32], C = [16, 32]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bfloat16_conversion, ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_2d_block_io, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
  tt.func @_attn_fwd_other_use_before_loop(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: f32, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    // CHECK-LABEL:   tt.func @_attn_fwd_other_use_before_loop
    // COM: This test checks that a load is not moved
    // COM: if the data is used by a second user before the loop.
    %c8192_i64 = arith.constant 8192 : i64
    %c128_i32 = arith.constant 128 : i32
    %c128_i64 = arith.constant 128 : i64
    %c64_i64 = arith.constant 64 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 1.44269502 : f32
    %c64_i32 = arith.constant 64 : i32
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma1}>>
    %cst_1 = arith.constant dense<0xFF800000> : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma1}>>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<512x128xf32, #mma1>
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id z : i32
    %2 = arith.extsi %1 : i32 to i64
    %3 = arith.muli %2, %c8192_i64 : i64
    %4 = tt.addptr %arg0, %3 : !tt.ptr<f16>, i64
    %5 = arith.muli %0, %c128_i32 : i32
    // Q descriptor (row_major): original shape [128, 64], strides [64, 1]
    %6 = tt.make_tensor_descriptor %4, [%c128_i32, %c64_i32], [%c64_i64, %c1_i64] : <f16>, <tensor<512x128xf16>>
    %7 = tt.addptr %arg2, %3 : !tt.ptr<f16>, i64
    // V descriptor (row_major): original shape [128, 64], strides [64, 1]
    %8 = tt.make_tensor_descriptor %7, [%c128_i32, %c64_i32], [%c64_i64, %c1_i64] : <f16>, <tensor<128x128xf16>>
    %9 = tt.addptr %arg1, %3 : !tt.ptr<f16>, i64
    // K descriptor (row_major descriptor, loaded with column_major for transpose)
    // Original col-major: shape [64, 128], strides [1, 64] → transposed row-major: shape [128, 64], strides [64, 1]
    %10 = tt.make_tensor_descriptor %9, [%c128_i32, %c64_i32], [%c64_i64, %c1_i64] : <f16>, <tensor<128x128xf16>>
    %11 = tt.addptr %arg5, %3 : !tt.ptr<f32>, i64
    // Output descriptor (row_major): shape [128, 64], strides [64, 1]
    %12 = tt.make_tensor_descriptor %11, [%c128_i32, %c64_i32], [%c64_i64, %c1_i64] : <f32>, <tensor<512x128xf32>>
    %13 = arith.mulf %arg3, %cst : f32
    // CHECK:  tt.descriptor_load {{.*}} : !tt.tensordesc<tensor<512x128xf16>> -> tensor<512x128xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS1]], kWidth = 1}>>
    %14 = tt.descriptor_load %6[%5, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<512x128xf16>> -> tensor<512x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 1}>>
    %15 = tt.splat %13 : f32 -> tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma1}>>
    %16 = tt.splat %13 : f32 -> tensor<512x128xf32, #mma1>
    %100 = tt.descriptor_load %10[%c0_i32, %c0_i32] {ttig.block_io = "column_major"} : !tt.tensordesc<tensor<128x128xf16>> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 2}>>
    %101 = tt.dot %14, %100, %cst_2, inputPrecision = tf32 : tensor<512x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 1}>> * tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 2}>> -> tensor<512x128xf32, #mma1>
    tt.descriptor_store %12[%5, %c0_i32], %101 : !tt.tensordesc<tensor<512x128xf32>>, tensor<512x128xf32, #mma1>
    %17:4 = scf.for %arg6 = %c0_i32 to %c128_i32 step %c64_i32 iter_args(%arg7 = %cst_0, %arg8 = %cst_2, %arg9 = %cst_1, %arg10 = %c0_i32) -> (tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma1}>>, tensor<512x128xf32, #mma1>, tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma1}>>, i32)  : i32 {
      // CHECK:  scf.for
      // CHECK-NOT:  tt.descriptor_load {{.*}} : !tt.tensordesc<tensor<512x128xf16>> -> tensor<512x128xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS1]], kWidth = 1}>>
      %21 = tt.descriptor_load %10[%arg10, %c0_i32] {ttig.block_io = "column_major"} : !tt.tensordesc<tensor<128x128xf16>> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 2}>>
      %22 = tt.dot %14, %21, %cst_2, inputPrecision = tf32 : tensor<512x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 1}>> * tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 2}>> -> tensor<512x128xf32, #mma1>
      %23 = "tt.reduce"(%22) <{axis = 1 : i32}> ({
      ^bb0(%arg12: f32, %arg13: f32):
        %45 = arith.maxnumf %arg12, %arg13 : f32
        tt.reduce.return %45 : f32
      }) : (tensor<512x128xf32, #mma1>) -> tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma1}>>
      %24 = arith.mulf %23, %15 : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma1}>>
      %25 = arith.maxnumf %arg9, %24 : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma1}>>
      %26 = arith.mulf %22, %16 : tensor<512x128xf32, #mma1>
      %27 = tt.expand_dims %25 {axis = 1 : i32} : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma1}>> -> tensor<512x1xf32, #mma1>
      %28 = tt.broadcast %27 : tensor<512x1xf32, #mma1> -> tensor<512x128xf32, #mma1>
      %29 = arith.subf %26, %28 : tensor<512x128xf32, #mma1>
      %30 = math.exp2 %29 : tensor<512x128xf32, #mma1>
      %31 = "tt.reduce"(%30) <{axis = 1 : i32}> ({
      ^bb0(%arg12: f32, %arg13: f32):
        %45 = arith.addf %arg12, %arg13 : f32
        tt.reduce.return %45 : f32
      }) : (tensor<512x128xf32, #mma1>) -> tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma1}>>
      %32 = arith.subf %arg9, %25 : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma1}>>
      %33 = math.exp2 %32 : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma1}>>
      %34 = arith.mulf %arg7, %33 : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma1}>>
      %35 = arith.addf %34, %31 : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma1}>>
      %36 = tt.expand_dims %33 {axis = 1 : i32} : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma1}>> -> tensor<512x1xf32, #mma1>
      %37 = tt.broadcast %36 : tensor<512x1xf32, #mma1> -> tensor<512x128xf32, #mma1>
      %38 = arith.mulf %arg8, %37 : tensor<512x128xf32, #mma1>
      %39 = tt.descriptor_load %8[%arg10, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<128x128xf16>> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 2}>>
      %40 = arith.truncf %30 : tensor<512x128xf32, #mma1> to tensor<512x128xf16, #mma1>
      %41 = ttg.convert_layout %40 : tensor<512x128xf16, #mma1> -> tensor<512x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 1}>>
      %42 = tt.dot %41, %39, %38, inputPrecision = tf32 : tensor<512x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 1}>> * tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 2}>> -> tensor<512x128xf32, #mma1>
      %43 = arith.addi %arg10, %c64_i32 : i32
      scf.yield %35, %42, %25, %43 : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma1}>>, tensor<512x128xf32, #mma1>, tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma1}>>, i32
    }
    %18 = tt.expand_dims %17#0 {axis = 1 : i32} : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma1}>> -> tensor<512x1xf32, #mma1>
    %19 = tt.broadcast %18 : tensor<512x1xf32, #mma1> -> tensor<512x128xf32, #mma1>
    %20 = arith.divf %17#1, %19 : tensor<512x128xf32, #mma1>
    tt.descriptor_store %12[%5, %c0_i32], %20 : !tt.tensordesc<tensor<512x128xf32>>, tensor<512x128xf32, #mma1>
    tt.return
  }
}


// -----

// CHECK: #[[$DPAS2:.+]] = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 2], repCluster = [2, 2], A = [16, 16], B = [16, 32], C = [16, 32]}>
#mma2 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 2], repCluster = [2, 2], A = [16, 16], B = [16, 32], C = [16, 32]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bfloat16_conversion, ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_2d_block_io, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
  tt.func @_attn_fwd_other_use_after_loop(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: f32, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    // CHECK-LABEL:   tt.func @_attn_fwd_other_use_after_loop
    // COM: This test checks that a load is moved inside the loop and after it,
    // COM: if the data is used by a second user after the loop.
    %c8192_i64 = arith.constant 8192 : i64
    %c128_i32 = arith.constant 128 : i32
    %c128_i64 = arith.constant 128 : i64
    %c64_i64 = arith.constant 64 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 1.44269502 : f32
    %c64_i32 = arith.constant 64 : i32
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma2}>>
    %cst_1 = arith.constant dense<0xFF800000> : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma2}>>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<512x128xf32, #mma2>
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id z : i32
    %2 = arith.extsi %1 : i32 to i64
    %3 = arith.muli %2, %c8192_i64 : i64
    %4 = tt.addptr %arg0, %3 : !tt.ptr<f16>, i64
    %5 = arith.muli %0, %c128_i32 : i32
    // Q descriptor (row_major): original shape [128, 64], strides [64, 1]
    %6 = tt.make_tensor_descriptor %4, [%c128_i32, %c64_i32], [%c64_i64, %c1_i64] : <f16>, <tensor<512x128xf16>>
    %7 = tt.addptr %arg2, %3 : !tt.ptr<f16>, i64
    // V descriptor (row_major): original shape [128, 64], strides [64, 1]
    %8 = tt.make_tensor_descriptor %7, [%c128_i32, %c64_i32], [%c64_i64, %c1_i64] : <f16>, <tensor<128x128xf16>>
    %9 = tt.addptr %arg1, %3 : !tt.ptr<f16>, i64
    // K descriptor (row_major descriptor, loaded with column_major for transpose)
    %10 = tt.make_tensor_descriptor %9, [%c128_i32, %c64_i32], [%c64_i64, %c1_i64] : <f16>, <tensor<128x128xf16>>
    %11 = tt.addptr %arg5, %3 : !tt.ptr<f32>, i64
    // Output descriptor (row_major): shape [128, 64], strides [64, 1]
    %12 = tt.make_tensor_descriptor %11, [%c128_i32, %c64_i32], [%c64_i64, %c1_i64] : <f32>, <tensor<512x128xf32>>
    %13 = arith.mulf %arg3, %cst : f32
    // CHECK:      ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<tensor<512x128xf16>>
    // CHECK-NOT:  tt.descriptor_load {{.*}} : !tt.tensordesc<tensor<512x128xf16>>
    %14 = tt.descriptor_load %6[%5, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<512x128xf16>> -> tensor<512x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma2, kWidth = 1}>>
    %15 = tt.splat %13 : f32 -> tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma2}>>
    %16 = tt.splat %13 : f32 -> tensor<512x128xf32, #mma2>
    %17:4 = scf.for %arg6 = %c0_i32 to %c128_i32 step %c64_i32 iter_args(%arg7 = %cst_0, %arg8 = %cst_2, %arg9 = %cst_1, %arg10 = %c0_i32) -> (tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma2}>>, tensor<512x128xf32, #mma2>, tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma2}>>, i32)  : i32 {
      // CHECK:  scf.for
      // CHECK:  tt.descriptor_load {{.*}} : !tt.tensordesc<tensor<512x128xf16>> -> tensor<512x128xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS2]], kWidth = 1}>>
      %21 = tt.descriptor_load %10[%arg10, %c0_i32] {ttig.block_io = "column_major"} : !tt.tensordesc<tensor<128x128xf16>> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma2, kWidth = 2}>>
      %22 = tt.dot %14, %21, %cst_2, inputPrecision = tf32 : tensor<512x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma2, kWidth = 1}>> * tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma2, kWidth = 2}>> -> tensor<512x128xf32, #mma2>
      %23 = "tt.reduce"(%22) <{axis = 1 : i32}> ({
      ^bb0(%arg12: f32, %arg13: f32):
        %45 = arith.maxnumf %arg12, %arg13 : f32
        tt.reduce.return %45 : f32
      }) : (tensor<512x128xf32, #mma2>) -> tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma2}>>
      %24 = arith.mulf %23, %15 : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma2}>>
      %25 = arith.maxnumf %arg9, %24 : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma2}>>
      %26 = arith.mulf %22, %16 : tensor<512x128xf32, #mma2>
      %27 = tt.expand_dims %25 {axis = 1 : i32} : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma2}>> -> tensor<512x1xf32, #mma2>
      %28 = tt.broadcast %27 : tensor<512x1xf32, #mma2> -> tensor<512x128xf32, #mma2>
      %29 = arith.subf %26, %28 : tensor<512x128xf32, #mma2>
      %30 = math.exp2 %29 : tensor<512x128xf32, #mma2>
      %31 = "tt.reduce"(%30) <{axis = 1 : i32}> ({
      ^bb0(%arg12: f32, %arg13: f32):
        %45 = arith.addf %arg12, %arg13 : f32
        tt.reduce.return %45 : f32
      }) : (tensor<512x128xf32, #mma2>) -> tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma2}>>
      %32 = arith.subf %arg9, %25 : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma2}>>
      %33 = math.exp2 %32 : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma2}>>
      %34 = arith.mulf %arg7, %33 : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma2}>>
      %35 = arith.addf %34, %31 : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma2}>>
      %36 = tt.expand_dims %33 {axis = 1 : i32} : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma2}>> -> tensor<512x1xf32, #mma2>
      %37 = tt.broadcast %36 : tensor<512x1xf32, #mma2> -> tensor<512x128xf32, #mma2>
      %38 = arith.mulf %arg8, %37 : tensor<512x128xf32, #mma2>
      %39 = tt.descriptor_load %8[%arg10, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<128x128xf16>> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma2, kWidth = 2}>>
      %40 = arith.truncf %30 : tensor<512x128xf32, #mma2> to tensor<512x128xf16, #mma2>
      %41 = ttg.convert_layout %40 : tensor<512x128xf16, #mma2> -> tensor<512x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma2, kWidth = 1}>>
      %42 = tt.dot %41, %39, %38, inputPrecision = tf32 : tensor<512x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma2, kWidth = 1}>> * tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma2, kWidth = 2}>> -> tensor<512x128xf32, #mma2>
      %43 = arith.addi %arg10, %c64_i32 : i32
      scf.yield %35, %42, %25, %43 : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma2}>>, tensor<512x128xf32, #mma2>, tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma2}>>, i32
    }
    // CHECK:  scf.yield
    // CHECK:  tt.descriptor_load {{.*}} : !tt.tensordesc<tensor<512x128xf16>> -> tensor<512x128xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS2]], kWidth = 1}>>
    %100 = tt.descriptor_load %10[%c0_i32, %c0_i32] {ttig.block_io = "column_major"} : !tt.tensordesc<tensor<128x128xf16>> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma2, kWidth = 2}>>
    %101 = tt.dot %14, %100, %cst_2, inputPrecision = tf32 : tensor<512x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma2, kWidth = 1}>> * tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma2, kWidth = 2}>> -> tensor<512x128xf32, #mma2>
    tt.descriptor_store %12[%5, %c0_i32], %101 : !tt.tensordesc<tensor<512x128xf32>>, tensor<512x128xf32, #mma2>
    %18 = tt.expand_dims %17#0 {axis = 1 : i32} : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma2}>> -> tensor<512x1xf32, #mma2>
    %19 = tt.broadcast %18 : tensor<512x1xf32, #mma2> -> tensor<512x128xf32, #mma2>
    %20 = arith.divf %17#1, %19 : tensor<512x128xf32, #mma2>
    tt.descriptor_store %12[%5, %c0_i32], %20 : !tt.tensordesc<tensor<512x128xf32>>, tensor<512x128xf32, #mma2>
    tt.return
  }
}


// -----

// CHECK: #[[$DPAS3:.+]] = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 1], repCluster = [2, 2], A = [16, 16], B = [16, 32], C = [16, 32]}>
#mma3 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 1], repCluster = [2, 2], A = [16, 16], B = [16, 32], C = [16, 32]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bfloat16_conversion, ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_2d_block_io, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
  tt.func @_attn_fwd_with_block_pointers_causal(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: f32, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    // CHECK-LABEL:   tt.func @_attn_fwd_with_block_pointers_causal
    // COM: This test checks that the Q matrix load is moved inside the two loop bodies
    // COM: when it is used by two users in two different loops
    %cst = arith.constant dense<1.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma3}>>
    %cst_0 = arith.constant dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma3}>>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma3>
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %c16777216_i64 = arith.constant 16777216 : i64
    %c1048576_i64 = arith.constant 1048576 : i64
    %c128_i32 = arith.constant 128 : i32
    %c8192_i32 = arith.constant 8192 : i32
    %c8192_i64 = arith.constant 8192 : i64
    %c128_i64 = arith.constant 128 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant 1.44269502 : f32
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma3>
    %cst_4 = arith.constant dense<-1.000000e+06> : tensor<128x128xf32, #mma3>
    %0 = tt.get_program_id z : i32
    %1 = tt.get_program_id x : i32
    %2 = tt.get_program_id y : i32
    %3 = arith.extsi %1 : i32 to i64
    %4 = arith.muli %3, %c16777216_i64 : i64
    %5 = arith.extsi %2 : i32 to i64
    %6 = arith.muli %5, %c1048576_i64 : i64
    %7 = arith.addi %4, %6 : i64
    %8 = tt.addptr %arg0, %7 : !tt.ptr<f16>, i64
    %9 = arith.muli %0, %c128_i32 : i32
    // Q descriptor (row_major): original shape [8192, 128], strides [128, 1]
    %10 = tt.make_tensor_descriptor %8, [%c8192_i32, %c128_i32], [%c128_i64, %c1_i64] : <f16>, <tensor<128x128xf16>>
    %11 = tt.addptr %arg2, %7 : !tt.ptr<f16>, i64
    // V descriptor (row_major): original shape [8192, 128], strides [128, 1]
    %12 = tt.make_tensor_descriptor %11, [%c8192_i32, %c128_i32], [%c128_i64, %c1_i64] : <f16>, <tensor<128x128xf16>>
    %13 = tt.addptr %arg1, %7 : !tt.ptr<f16>, i64
    // K descriptor (row_major descriptor, loaded with column_major for transpose)
    // Original col-major: shape [128, 8192], strides [1, 128] → transposed row-major: shape [8192, 128], strides [128, 1]
    %14 = tt.make_tensor_descriptor %13, [%c8192_i32, %c128_i32], [%c128_i64, %c1_i64] : <f16>, <tensor<128x128xf16>>
    %15 = tt.addptr %arg5, %7 : !tt.ptr<f32>, i64
    // Output descriptor (row_major): shape [8192, 128], strides [128, 1]
    %16 = tt.make_tensor_descriptor %15, [%c8192_i32, %c128_i32], [%c128_i64, %c1_i64] : <f32>, <tensor<128x128xf32>>
    %17 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma3}>>
    %18 = tt.splat %9 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma3}>>
    %19 = arith.addi %18, %17 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma3}>>
    %20 = arith.mulf %arg3, %cst_2 : f32
    %21 = tt.descriptor_load %10[%9, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<128x128xf16>> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma3, kWidth = 1}>>
    // CHECK:      ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<tensor<128x128xf16>>
    // CHECK-NOT:  tt.descriptor_load {{.*}} : !tt.tensordesc<tensor<128x128xf16>> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS3]], kWidth = 1}>>
    %22 = tt.splat %20 : f32 -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma3}>>
    %23 = tt.splat %20 : f32 -> tensor<128x128xf32, #mma3>
    %24 = arith.cmpi sgt, %9, %c0_i32 : i32
    scf.if %24 {
      ttig.descriptor_prefetch %14[%c0_i32, %c0_i32] {ttig.block_io = "column_major"} : !tt.tensordesc<tensor<128x128xf16>>
      ttig.descriptor_prefetch %12[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<128x128xf16>>
    }
    %27:4 = scf.for %arg6 = %c0_i32 to %9 step %c64_i32 iter_args(%arg7 = %cst, %arg8 = %cst_1, %arg9 = %cst_0, %arg10 = %c0_i32) -> (tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma3}>>, tensor<128x128xf32, #mma3>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma3}>>, i32)  : i32 {
      // CHECK:  scf.for
      // CHECK:  tt.descriptor_load {{.*}} : !tt.tensordesc<tensor<128x128xf16>> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS3]], kWidth = 1}>>
      %44 = arith.subi %9, %c64_i32 : i32
      %45 = arith.cmpi slt, %arg6, %44 : i32
      %next_idx = arith.addi %arg10, %c64_i32 : i32
      scf.if %45 {
        ttig.descriptor_prefetch %14[%next_idx, %c0_i32] {ttig.block_io = "column_major"} : !tt.tensordesc<tensor<128x128xf16>>
        ttig.descriptor_prefetch %12[%next_idx, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<128x128xf16>>
      }
      %50 = tt.descriptor_load %14[%arg10, %c0_i32] {ttig.block_io = "column_major"} : !tt.tensordesc<tensor<128x128xf16>> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma3, kWidth = 2}>>
      %51 = tt.descriptor_load %12[%arg10, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<128x128xf16>> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma3, kWidth = 2}>>
      %52 = tt.dot %21, %50, %cst_3, inputPrecision = tf32 : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma3, kWidth = 1}>> * tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma3, kWidth = 2}>> -> tensor<128x128xf32, #mma3>
      %53 = "tt.reduce"(%52) <{axis = 1 : i32}> ({
      ^bb0(%arg12: f32, %arg13: f32):
        %72 = arith.maxnumf %arg12, %arg13 : f32
        tt.reduce.return %72 : f32
      }) : (tensor<128x128xf32, #mma3>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma3}>>
      %54 = arith.mulf %53, %22 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma3}>>
      %55 = arith.maxnumf %arg9, %54 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma3}>>
      %56 = arith.mulf %52, %23 : tensor<128x128xf32, #mma3>
      %57 = tt.expand_dims %55 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma3}>> -> tensor<128x1xf32, #mma3>
      %58 = tt.broadcast %57 : tensor<128x1xf32, #mma3> -> tensor<128x128xf32, #mma3>
      %59 = arith.subf %56, %58 : tensor<128x128xf32, #mma3>
      %60 = math.exp2 %59 : tensor<128x128xf32, #mma3>
      %61 = "tt.reduce"(%60) <{axis = 1 : i32}> ({
      ^bb0(%arg12: f32, %arg13: f32):
        %72 = arith.addf %arg12, %arg13 : f32
        tt.reduce.return %72 : f32
      }) : (tensor<128x128xf32, #mma3>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma3}>>
      %62 = arith.subf %arg9, %55 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma3}>>
      %63 = math.exp2 %62 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma3}>>
      %64 = arith.mulf %arg7, %63 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma3}>>
      %65 = arith.addf %64, %61 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma3}>>
      %66 = tt.expand_dims %63 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma3}>> -> tensor<128x1xf32, #mma3>
      %67 = tt.broadcast %66 : tensor<128x1xf32, #mma3> -> tensor<128x128xf32, #mma3>
      %68 = arith.mulf %arg8, %67 : tensor<128x128xf32, #mma3>
      %69 = arith.truncf %60 : tensor<128x128xf32, #mma3> to tensor<128x128xf16, #mma3>
      %70 = ttg.convert_layout %69 : tensor<128x128xf16, #mma3> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma3, kWidth = 1}>>
      %71 = tt.dot %70, %51, %68, inputPrecision = tf32 : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma3, kWidth = 1}>> * tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma3, kWidth = 2}>> -> tensor<128x128xf32, #mma3>
      scf.yield %65, %71, %55, %next_idx : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma3}>>, tensor<128x128xf32, #mma3>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma3}>>, i32
    }
    %28 = arith.muli %0, %c128_i32 {tt.divisibility = dense<128> : tensor<1xi32>} : i32
    %29 = arith.addi %0, %c1_i32 : i32
    %30 = arith.muli %29, %c128_i32 : i32
    // Between loops: K and V start at row index %28
    %33 = tt.expand_dims %19 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma3}>> -> tensor<128x1xi32, #mma3>
    %34 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #mma3}>>
    %35 = tt.expand_dims %34 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #mma3}>> -> tensor<1x128xi32, #mma3>
    %36 = tt.broadcast %33 : tensor<128x1xi32, #mma3> -> tensor<128x128xi32, #mma3>
    %37 = arith.cmpi slt, %28, %30 : i32
    scf.if %37 {
      ttig.descriptor_prefetch %14[%28, %c0_i32] {ttig.block_io = "column_major"} : !tt.tensordesc<tensor<128x128xf16>>
      ttig.descriptor_prefetch %12[%28, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<128x128xf16>>
    }
    %40:4 = scf.for %arg6 = %28 to %30 step %c64_i32 iter_args(%arg7 = %27#0, %arg8 = %27#1, %arg9 = %27#2, %arg10 = %28) -> (tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma3}>>, tensor<128x128xf32, #mma3>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma3}>>, i32)  : i32 {
      // CHECK:  scf.for
      // CHECK:  tt.descriptor_load {{.*}} : !tt.tensordesc<tensor<128x128xf16>> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS3]], kWidth = 1}>>
      %44 = arith.subi %30, %c64_i32 : i32
      %45 = arith.cmpi slt, %arg6, %44 : i32
      %next_idx2 = arith.addi %arg10, %c64_i32 : i32
      scf.if %45 {
        ttig.descriptor_prefetch %14[%next_idx2, %c0_i32] {ttig.block_io = "column_major"} : !tt.tensordesc<tensor<128x128xf16>>
        ttig.descriptor_prefetch %12[%next_idx2, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<128x128xf16>>
      }
      %50 = tt.descriptor_load %14[%arg10, %c0_i32] {ttig.block_io = "column_major"} : !tt.tensordesc<tensor<128x128xf16>> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma3, kWidth = 2}>>
      %51 = tt.descriptor_load %12[%arg10, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<128x128xf16>> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma3, kWidth = 2}>>
      %52 = tt.dot %21, %50, %cst_3, inputPrecision = tf32 : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma3, kWidth = 1}>> * tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma3, kWidth = 2}>> -> tensor<128x128xf32, #mma3>
      %53 = tt.splat %arg6 : i32 -> tensor<1x128xi32, #mma3>
      %54 = arith.addi %53, %35 : tensor<1x128xi32, #mma3>
      %55 = tt.broadcast %54 : tensor<1x128xi32, #mma3> -> tensor<128x128xi32, #mma3>
      %56 = arith.cmpi sge, %36, %55 : tensor<128x128xi32, #mma3>
      %57 = arith.mulf %52, %23 : tensor<128x128xf32, #mma3>
      %58 = arith.select %56, %cst_3, %cst_4 : tensor<128x128xi1, #mma3>, tensor<128x128xf32, #mma3>
      %59 = arith.addf %57, %58 : tensor<128x128xf32, #mma3>
      %60 = "tt.reduce"(%59) <{axis = 1 : i32}> ({
      ^bb0(%arg12: f32, %arg13: f32):
        %77 = arith.maxnumf %arg12, %arg13 : f32
        tt.reduce.return %77 : f32
      }) : (tensor<128x128xf32, #mma3>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma3}>>
      %61 = arith.maxnumf %arg9, %60 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma3}>>
      %62 = tt.expand_dims %61 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma3}>> -> tensor<128x1xf32, #mma3>
      %63 = tt.broadcast %62 : tensor<128x1xf32, #mma3> -> tensor<128x128xf32, #mma3>
      %64 = arith.subf %59, %63 : tensor<128x128xf32, #mma3>
      %65 = math.exp2 %64 : tensor<128x128xf32, #mma3>
      %66 = "tt.reduce"(%65) <{axis = 1 : i32}> ({
      ^bb0(%arg12: f32, %arg13: f32):
        %77 = arith.addf %arg12, %arg13 : f32
        tt.reduce.return %77 : f32
      }) : (tensor<128x128xf32, #mma3>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma3}>>
      %67 = arith.subf %arg9, %61 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma3}>>
      %68 = math.exp2 %67 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma3}>>
      %69 = arith.mulf %arg7, %68 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma3}>>
      %70 = arith.addf %69, %66 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma3}>>
      %71 = tt.expand_dims %68 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma3}>> -> tensor<128x1xf32, #mma3>
      %72 = tt.broadcast %71 : tensor<128x1xf32, #mma3> -> tensor<128x128xf32, #mma3>
      %73 = arith.mulf %arg8, %72 : tensor<128x128xf32, #mma3>
      %74 = arith.truncf %65 : tensor<128x128xf32, #mma3> to tensor<128x128xf16, #mma3>
      %75 = ttg.convert_layout %74 : tensor<128x128xf16, #mma3> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma3, kWidth = 1}>>
      %76 = tt.dot %75, %51, %73, inputPrecision = tf32 : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma3, kWidth = 1}>> * tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma3, kWidth = 2}>> -> tensor<128x128xf32, #mma3>
      scf.yield %70, %76, %61, %next_idx2 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma3}>>, tensor<128x128xf32, #mma3>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma3}>>, i32
    }
    tt.return
  }
}
