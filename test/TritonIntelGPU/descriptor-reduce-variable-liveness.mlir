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
    %12 = tt.make_tensor_ptr %11, [%c128_i64, %c64_i64], [%c64_i64, %c1_i64], [%5, %c0_i32] {order = array<i32: 1, 0>} : <tensor<512x128xf32, #mma>>
    %13 = arith.mulf %arg3, %cst : f32
    // CHECK:      ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<tensor<512x128xf16>>
    // CHECK-NOT:  tt.descriptor_load {{.*}} : !tt.tensordesc<tensor<512x128xf16>>
    %14 = tt.descriptor_load %6[%5, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<512x128xf16>> -> tensor<512x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %15 = tt.splat %13 : f32 -> tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %16 = tt.splat %13 : f32 -> tensor<512x128xf32, #mma>
    %17:4 = scf.for %arg6 = %c0_i32 to %c128_i32 step %c64_i32 iter_args(%arg7 = %cst_0, %arg8 = %cst_2, %arg9 = %cst_1, %arg10 = %c0_i32) -> (tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<512x128xf32, #mma>, tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>, i32)  : i32 {
      // CHECK:  scf.for
      // CHECK:  tt.descriptor_load {{.*}} : !tt.tensordesc<tensor<512x128xf16>> -> tensor<512x128xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>
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
    } {tt.divisibility_arg1 = dense<64> : tensor<1xi32>}
    %18 = tt.expand_dims %17#0 {axis = 1 : i32} : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<512x1xf32, #mma>
    %19 = tt.broadcast %18 : tensor<512x1xf32, #mma> -> tensor<512x128xf32, #mma>
    %20 = arith.divf %17#1, %19 : tensor<512x128xf32, #mma>
    tt.store %12, %20 : !tt.ptr<tensor<512x128xf32, #mma>>
    tt.return
  }
}

// -----
// TODO: Test @_attn_fwd_other_use_before_loop uses column_major attribute which is not yet implemented for tensor_descriptor.


// // _CHECK: #[[$DPAS:.+]] = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 2], repCluster = [2, 2], A = [16, 16], B = [16, 32], C = [16, 32]}>
// #mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 2], repCluster = [2, 2], A = [16, 16], B = [16, 32], C = [16, 32]}>
// module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bfloat16_conversion, ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_2d_block_io, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
//   tt.func @_attn_fwd_other_use_before_loop(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: f32, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
//     // _CHECK-LABEL:   tt.func @_attn_fwd_other_use_before_loop
//     // COM: This test checks that a load is not moved
//     // COM: if the data is used by a second user before the loop.
//     %c8192_i64 = arith.constant 8192 : i64
//     %c128_i32 = arith.constant 128 : i32
//     %c128_i64 = arith.constant 128 : i64
//     %c64_i64 = arith.constant 64 : i64
//     %c1_i64 = arith.constant 1 : i64
//     %c0_i32 = arith.constant 0 : i32
//     %cst = arith.constant 1.44269502 : f32
//     %c64_i32 = arith.constant 64 : i32
//     %cst_0 = arith.constant dense<1.000000e+00> : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//     %cst_1 = arith.constant dense<0xFF800000> : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//     %cst_2 = arith.constant dense<0.000000e+00> : tensor<512x128xf32, #mma>
//     %0 = tt.get_program_id x : i32
//     %1 = tt.get_program_id z : i32
//     %2 = arith.extsi %1 : i32 to i64
//     %3 = arith.muli %2, %c8192_i64 : i64
//     %4 = tt.addptr %arg0, %3 : !tt.ptr<f16>, i64
//     %5 = arith.muli %0, %c128_i32 : i32
//     %6 = tt.make_tensor_ptr %4, [%c128_i64, %c64_i64], [%c64_i64, %c1_i64], [%5, %c0_i32] {order = array<i32: 1, 0>} : <tensor<512x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>>
//     %7 = tt.addptr %arg2, %3 : !tt.ptr<f16>, i64
//     %8 = tt.make_tensor_ptr %7, [%c128_i64, %c64_i64], [%c64_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//     %9 = tt.addptr %arg1, %3 : !tt.ptr<f16>, i64
//     %10 = tt.make_tensor_ptr %9, [%c64_i64, %c128_i64], [%c1_i64, %c64_i64], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//     %11 = tt.addptr %arg5, %3 : !tt.ptr<f32>, i64
//     %12 = tt.make_tensor_ptr %11, [%c128_i64, %c64_i64], [%c64_i64, %c1_i64], [%5, %c0_i32] {order = array<i32: 1, 0>} : <tensor<512x128xf32, #mma>>
//     %13 = arith.mulf %arg3, %cst : f32
//     // _CHECK:  tt.load {{.*}} : !tt.ptr<tensor<512x128xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>>
//     %14 = tt.load %6 {ttig.block_io = "row_major"} : !tt.ptr<tensor<512x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>>
//     %15 = tt.splat %13 : f32 -> tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//     %16 = tt.splat %13 : f32 -> tensor<512x128xf32, #mma>
//     %100 = tt.load %10 {ttig.block_io = "column_major"} : !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//     %101 = tt.dot %14, %100, %cst_2, inputPrecision = tf32 : tensor<512x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<512x128xf32, #mma>
//     tt.store %12, %101 : !tt.ptr<tensor<512x128xf32, #mma>>
//     %17:5 = scf.for %arg6 = %c0_i32 to %c128_i32 step %c64_i32 iter_args(%arg7 = %cst_0, %arg8 = %cst_2, %arg9 = %cst_1, %arg10 = %10, %arg11 = %8) -> (tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<512x128xf32, #mma>, tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>, !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>, !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>)  : i32 {
//       // _CHECK:  scf.for
//       // _CHECK-NOT:  tt.load {{.*}} : !tt.ptr<tensor<512x128xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>>
//       %21 = tt.load %arg10 {ttig.block_io = "column_major"} : !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//       %22 = tt.dot %14, %21, %cst_2, inputPrecision = tf32 : tensor<512x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<512x128xf32, #mma>
//       %23 = "tt.reduce"(%22) <{axis = 1 : i32}> ({
//       ^bb0(%arg12: f32, %arg13: f32):
//         %45 = arith.maxnumf %arg12, %arg13 : f32
//         tt.reduce.return %45 : f32
//       }) : (tensor<512x128xf32, #mma>) -> tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//       %24 = arith.mulf %23, %15 : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//       %25 = arith.maxnumf %arg9, %24 : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//       %26 = arith.mulf %22, %16 : tensor<512x128xf32, #mma>
//       %27 = tt.expand_dims %25 {axis = 1 : i32} : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<512x1xf32, #mma>
//       %28 = tt.broadcast %27 : tensor<512x1xf32, #mma> -> tensor<512x128xf32, #mma>
//       %29 = arith.subf %26, %28 : tensor<512x128xf32, #mma>
//       %30 = math.exp2 %29 : tensor<512x128xf32, #mma>
//       %31 = "tt.reduce"(%30) <{axis = 1 : i32}> ({
//       ^bb0(%arg12: f32, %arg13: f32):
//         %45 = arith.addf %arg12, %arg13 : f32
//         tt.reduce.return %45 : f32
//       }) : (tensor<512x128xf32, #mma>) -> tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//       %32 = arith.subf %arg9, %25 : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//       %33 = math.exp2 %32 : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//       %34 = arith.mulf %arg7, %33 : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//       %35 = arith.addf %34, %31 : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//       %36 = tt.expand_dims %33 {axis = 1 : i32} : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<512x1xf32, #mma>
//       %37 = tt.broadcast %36 : tensor<512x1xf32, #mma> -> tensor<512x128xf32, #mma>
//       %38 = arith.mulf %arg8, %37 : tensor<512x128xf32, #mma>
//       %39 = tt.load %arg11 {ttig.block_io = "row_major"} : !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//       %40 = arith.truncf %30 : tensor<512x128xf32, #mma> to tensor<512x128xf16, #mma>
//       %41 = ttg.convert_layout %40 : tensor<512x128xf16, #mma> -> tensor<512x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
//       %42 = tt.dot %41, %39, %38, inputPrecision = tf32 : tensor<512x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<512x128xf32, #mma>
//       %43 = tt.advance %arg11, [%c64_i32, %c0_i32] : <tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//       %44 = tt.advance %arg10, [%c0_i32, %c64_i32] : <tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//       scf.yield %35, %42, %25, %44, %43 : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<512x128xf32, #mma>, tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>, !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>, !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//     } {tt.divisibility_arg1 = dense<64> : tensor<1xi32>}
//     %18 = tt.expand_dims %17#0 {axis = 1 : i32} : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<512x1xf32, #mma>
//     %19 = tt.broadcast %18 : tensor<512x1xf32, #mma> -> tensor<512x128xf32, #mma>
//     %20 = arith.divf %17#1, %19 : tensor<512x128xf32, #mma>
//     tt.store %12, %20 : !tt.ptr<tensor<512x128xf32, #mma>>
//     tt.return
//   }
// }


// -----
// TODO: Test @_attn_fwd_other_use_after_loop uses column_major attribute which is not yet implemented for tensor_descriptor.


// // _CHECK: #[[$DPAS:.+]] = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 2], repCluster = [2, 2], A = [16, 16], B = [16, 32], C = [16, 32]}>
// #mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 2], repCluster = [2, 2], A = [16, 16], B = [16, 32], C = [16, 32]}>
// module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bfloat16_conversion, ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_2d_block_io, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
//   tt.func @_attn_fwd_other_use_after_loop(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: f32, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
//     // _CHECK-LABEL:   tt.func @_attn_fwd_other_use_after_loop
//     // COM: This test checks that a load is moved inside the loop and after it,
//     // COM: if the data is used by a second user after the loop.
//     %c8192_i64 = arith.constant 8192 : i64
//     %c128_i32 = arith.constant 128 : i32
//     %c128_i64 = arith.constant 128 : i64
//     %c64_i64 = arith.constant 64 : i64
//     %c1_i64 = arith.constant 1 : i64
//     %c0_i32 = arith.constant 0 : i32
//     %cst = arith.constant 1.44269502 : f32
//     %c64_i32 = arith.constant 64 : i32
//     %cst_0 = arith.constant dense<1.000000e+00> : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//     %cst_1 = arith.constant dense<0xFF800000> : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//     %cst_2 = arith.constant dense<0.000000e+00> : tensor<512x128xf32, #mma>
//     %0 = tt.get_program_id x : i32
//     %1 = tt.get_program_id z : i32
//     %2 = arith.extsi %1 : i32 to i64
//     %3 = arith.muli %2, %c8192_i64 : i64
//     %4 = tt.addptr %arg0, %3 : !tt.ptr<f16>, i64
//     %5 = arith.muli %0, %c128_i32 : i32
//     %6 = tt.make_tensor_ptr %4, [%c128_i64, %c64_i64], [%c64_i64, %c1_i64], [%5, %c0_i32] {order = array<i32: 1, 0>} : <tensor<512x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>>
//     %7 = tt.addptr %arg2, %3 : !tt.ptr<f16>, i64
//     %8 = tt.make_tensor_ptr %7, [%c128_i64, %c64_i64], [%c64_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//     %9 = tt.addptr %arg1, %3 : !tt.ptr<f16>, i64
//     %10 = tt.make_tensor_ptr %9, [%c64_i64, %c128_i64], [%c1_i64, %c64_i64], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//     %11 = tt.addptr %arg5, %3 : !tt.ptr<f32>, i64
//     %12 = tt.make_tensor_ptr %11, [%c128_i64, %c64_i64], [%c64_i64, %c1_i64], [%5, %c0_i32] {order = array<i32: 1, 0>} : <tensor<512x128xf32, #mma>>
//     %13 = arith.mulf %arg3, %cst : f32
//     // _CHECK:      ttig.prefetch {{.*}} : !tt.ptr<tensor<512x128xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>>
//     // _CHECK-NOT:  tt.load {{.*}} : !tt.ptr<tensor<512x128xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>>
//     %14 = tt.load %6 {ttig.block_io = "row_major"} : !tt.ptr<tensor<512x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>>
//     %15 = tt.splat %13 : f32 -> tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//     %16 = tt.splat %13 : f32 -> tensor<512x128xf32, #mma>
//     %17:5 = scf.for %arg6 = %c0_i32 to %c128_i32 step %c64_i32 iter_args(%arg7 = %cst_0, %arg8 = %cst_2, %arg9 = %cst_1, %arg10 = %10, %arg11 = %8) -> (tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<512x128xf32, #mma>, tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>, !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>, !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>)  : i32 {
//       // _CHECK:  scf.for
//       // _CHECK:  tt.load {{.*}} : !tt.ptr<tensor<512x128xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>>
//       %21 = tt.load %arg10 {ttig.block_io = "column_major"} : !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//       %22 = tt.dot %14, %21, %cst_2, inputPrecision = tf32 : tensor<512x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<512x128xf32, #mma>
//       %23 = "tt.reduce"(%22) <{axis = 1 : i32}> ({
//       ^bb0(%arg12: f32, %arg13: f32):
//         %45 = arith.maxnumf %arg12, %arg13 : f32
//         tt.reduce.return %45 : f32
//       }) : (tensor<512x128xf32, #mma>) -> tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//       %24 = arith.mulf %23, %15 : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//       %25 = arith.maxnumf %arg9, %24 : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//       %26 = arith.mulf %22, %16 : tensor<512x128xf32, #mma>
//       %27 = tt.expand_dims %25 {axis = 1 : i32} : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<512x1xf32, #mma>
//       %28 = tt.broadcast %27 : tensor<512x1xf32, #mma> -> tensor<512x128xf32, #mma>
//       %29 = arith.subf %26, %28 : tensor<512x128xf32, #mma>
//       %30 = math.exp2 %29 : tensor<512x128xf32, #mma>
//       %31 = "tt.reduce"(%30) <{axis = 1 : i32}> ({
//       ^bb0(%arg12: f32, %arg13: f32):
//         %45 = arith.addf %arg12, %arg13 : f32
//         tt.reduce.return %45 : f32
//       }) : (tensor<512x128xf32, #mma>) -> tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//       %32 = arith.subf %arg9, %25 : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//       %33 = math.exp2 %32 : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//       %34 = arith.mulf %arg7, %33 : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//       %35 = arith.addf %34, %31 : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//       %36 = tt.expand_dims %33 {axis = 1 : i32} : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<512x1xf32, #mma>
//       %37 = tt.broadcast %36 : tensor<512x1xf32, #mma> -> tensor<512x128xf32, #mma>
//       %38 = arith.mulf %arg8, %37 : tensor<512x128xf32, #mma>
//       %39 = tt.load %arg11 {ttig.block_io = "row_major"} : !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//       %40 = arith.truncf %30 : tensor<512x128xf32, #mma> to tensor<512x128xf16, #mma>
//       %41 = ttg.convert_layout %40 : tensor<512x128xf16, #mma> -> tensor<512x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
//       %42 = tt.dot %41, %39, %38, inputPrecision = tf32 : tensor<512x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<512x128xf32, #mma>
//       %43 = tt.advance %arg11, [%c64_i32, %c0_i32] : <tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//       %44 = tt.advance %arg10, [%c0_i32, %c64_i32] : <tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//       scf.yield %35, %42, %25, %44, %43 : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<512x128xf32, #mma>, tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>, !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>, !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//     } {tt.divisibility_arg1 = dense<64> : tensor<1xi32>}
//     // _CHECK:  scf.yield
//     // _CHECK:  tt.load {{.*}} : !tt.ptr<tensor<512x128xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>>
//     %100 = tt.load %10 {ttig.block_io = "column_major"} : !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//     %101 = tt.dot %14, %100, %cst_2, inputPrecision = tf32 : tensor<512x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<512x128xf32, #mma>
//     tt.store %12, %101 : !tt.ptr<tensor<512x128xf32, #mma>>
//     %18 = tt.expand_dims %17#0 {axis = 1 : i32} : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<512x1xf32, #mma>
//     %19 = tt.broadcast %18 : tensor<512x1xf32, #mma> -> tensor<512x128xf32, #mma>
//     %20 = arith.divf %17#1, %19 : tensor<512x128xf32, #mma>
//     tt.store %12, %20 : !tt.ptr<tensor<512x128xf32, #mma>>
//     tt.return
//   }
// }


// -----
// TODO: Test @_attn_fwd_with_block_pointers_causal uses column_major attribute which is not yet implemented for tensor_descriptor.


// // _CHECK: #[[$DPAS:.+]] = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 1], repCluster = [2, 2], A = [16, 16], B = [16, 32], C = [16, 32]}>
// #mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 1], repCluster = [2, 2], A = [16, 16], B = [16, 32], C = [16, 32]}>
// module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bfloat16_conversion, ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_2d_block_io, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
//   tt.func @_attn_fwd_with_block_pointers_causal(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: f32, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
//     // _CHECK-LABEL:   tt.func @_attn_fwd_with_block_pointers_causal
//     // COM: This test checks that the Q matrix load is moved inside the two loop bodies
//     // COM: when it is used by two users in two different loops
//     %cst = arith.constant dense<1.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//     %cst_0 = arith.constant dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//     %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
//     %c1_i32 = arith.constant 1 : i32
//     %c64_i32 = arith.constant 64 : i32
//     %c16777216_i64 = arith.constant 16777216 : i64
//     %c1048576_i64 = arith.constant 1048576 : i64
//     %c128_i32 = arith.constant 128 : i32
//     %c8192_i64 = arith.constant 8192 : i64
//     %c128_i64 = arith.constant 128 : i64
//     %c1_i64 = arith.constant 1 : i64
//     %c0_i32 = arith.constant 0 : i32
//     %cst_2 = arith.constant 1.44269502 : f32
//     %cst_3 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
//     %cst_4 = arith.constant dense<-1.000000e+06> : tensor<128x128xf32, #mma>
//     %0 = tt.get_program_id z : i32
//     %1 = tt.get_program_id x : i32
//     %2 = tt.get_program_id y : i32
//     %3 = arith.extsi %1 : i32 to i64
//     %4 = arith.muli %3, %c16777216_i64 : i64
//     %5 = arith.extsi %2 : i32 to i64
//     %6 = arith.muli %5, %c1048576_i64 : i64
//     %7 = arith.addi %4, %6 : i64
//     %8 = tt.addptr %arg0, %7 : !tt.ptr<f16>, i64
//     %9 = arith.muli %0, %c128_i32 : i32
//     %10 = tt.make_tensor_ptr %8, [%c8192_i64, %c128_i64], [%c128_i64, %c1_i64], [%9, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>>
//     %11 = tt.addptr %arg2, %7 : !tt.ptr<f16>, i64
//     %12 = tt.make_tensor_ptr %11, [%c8192_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//     %13 = tt.addptr %arg1, %7 : !tt.ptr<f16>, i64
//     %14 = tt.make_tensor_ptr %13, [%c128_i64, %c8192_i64], [%c1_i64, %c128_i64], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//     %15 = tt.addptr %arg5, %7 : !tt.ptr<f32>, i64
//     %16 = tt.make_tensor_ptr %15, [%c8192_i64, %c128_i64], [%c128_i64, %c1_i64], [%9, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf32, #mma>>
//     %17 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>>
//     %18 = tt.splat %9 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>>
//     %19 = arith.addi %18, %17 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>>
//     %20 = arith.mulf %arg3, %cst_2 : f32
//     %21 = tt.load %10 {ttig.block_io = "row_major"} : !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>>
//     // _CHECK:      ttig.prefetch {{.*}} : !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>>
//     // _CHECK-NOT:  tt.load {{.*}} : !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>>
//     %22 = tt.splat %20 : f32 -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//     %23 = tt.splat %20 : f32 -> tensor<128x128xf32, #mma>
//     %24 = arith.cmpi sgt, %9, %c0_i32 : i32
//     %25 = tt.splat %24 : i1 -> tensor<128x128xi1, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
//     ttig.prefetch %14, %25 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>, ttig.block_io = "column_major"} : !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//     %26 = tt.splat %24 : i1 -> tensor<128x128xi1, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
//     ttig.prefetch %12, %26 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>, ttig.block_io = "row_major"} : !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//     %27:5 = scf.for %arg6 = %c0_i32 to %9 step %c64_i32 iter_args(%arg7 = %cst, %arg8 = %cst_1, %arg9 = %cst_0, %arg10 = %14, %arg11 = %12) -> (tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128x128xf32, #mma>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>, !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>)  : i32 {
//       // _CHECK:  scf.for
//       // _CHECK:  tt.load {{.*}} : !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>>
//       %44 = arith.subi %9, %c64_i32 : i32
//       %45 = arith.cmpi slt, %arg6, %44 : i32
//       %46 = tt.advance %arg11, [%c64_i32, %c0_i32] : <tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//       %47 = tt.advance %arg10, [%c0_i32, %c64_i32] : <tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//       %48 = tt.splat %45 : i1 -> tensor<128x128xi1, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
//       ttig.prefetch %47, %48 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>, ttig.block_io = "column_major"} : !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//       %49 = tt.splat %45 : i1 -> tensor<128x128xi1, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
//       ttig.prefetch %46, %49 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>, ttig.block_io = "row_major"} : !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//       %50 = tt.load %arg10 {ttig.block_io = "column_major"} : !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//       %51 = tt.load %arg11 {ttig.block_io = "row_major"} : !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//       %52 = tt.dot %21, %50, %cst_3, inputPrecision = tf32 : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
//       %53 = "tt.reduce"(%52) <{axis = 1 : i32}> ({
//       ^bb0(%arg12: f32, %arg13: f32):
//         %72 = arith.maxnumf %arg12, %arg13 : f32
//         tt.reduce.return %72 : f32
//       }) : (tensor<128x128xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//       %54 = arith.mulf %53, %22 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//       %55 = arith.maxnumf %arg9, %54 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//       %56 = arith.mulf %52, %23 : tensor<128x128xf32, #mma>
//       %57 = tt.expand_dims %55 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
//       %58 = tt.broadcast %57 : tensor<128x1xf32, #mma> -> tensor<128x128xf32, #mma>
//       %59 = arith.subf %56, %58 : tensor<128x128xf32, #mma>
//       %60 = math.exp2 %59 : tensor<128x128xf32, #mma>
//       %61 = "tt.reduce"(%60) <{axis = 1 : i32}> ({
//       ^bb0(%arg12: f32, %arg13: f32):
//         %72 = arith.addf %arg12, %arg13 : f32
//         tt.reduce.return %72 : f32
//       }) : (tensor<128x128xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//       %62 = arith.subf %arg9, %55 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//       %63 = math.exp2 %62 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//       %64 = arith.mulf %arg7, %63 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//       %65 = arith.addf %64, %61 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//       %66 = tt.expand_dims %63 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
//       %67 = tt.broadcast %66 : tensor<128x1xf32, #mma> -> tensor<128x128xf32, #mma>
//       %68 = arith.mulf %arg8, %67 : tensor<128x128xf32, #mma>
//       %69 = arith.truncf %60 : tensor<128x128xf32, #mma> to tensor<128x128xf16, #mma>
//       %70 = ttg.convert_layout %69 : tensor<128x128xf16, #mma> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
//       %71 = tt.dot %70, %51, %68, inputPrecision = tf32 : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
//       scf.yield %65, %71, %55, %47, %46 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128x128xf32, #mma>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>, !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//     }
//     %28 = arith.muli %0, %c128_i32 {tt.divisibility = dense<128> : tensor<1xi32>} : i32
//     %29 = arith.addi %0, %c1_i32 : i32
//     %30 = arith.muli %29, %c128_i32 : i32
//     %31 = tt.advance %14, [%c0_i32, %28] : <tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//     %32 = tt.advance %12, [%28, %c0_i32] : <tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//     %33 = tt.expand_dims %19 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xi32, #mma>
//     %34 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #mma}>>
//     %35 = tt.expand_dims %34 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x128xi32, #mma>
//     %36 = tt.broadcast %33 : tensor<128x1xi32, #mma> -> tensor<128x128xi32, #mma>
//     %37 = arith.cmpi slt, %28, %30 : i32
//     %38 = tt.splat %37 : i1 -> tensor<128x128xi1, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
//     ttig.prefetch %31, %38 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>, ttig.block_io = "column_major"} : !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//     %39 = tt.splat %37 : i1 -> tensor<128x128xi1, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
//     ttig.prefetch %32, %39 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>, ttig.block_io = "row_major"} : !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//     %40:5 = scf.for %arg6 = %28 to %30 step %c64_i32 iter_args(%arg7 = %27#0, %arg8 = %27#1, %arg9 = %27#2, %arg10 = %31, %arg11 = %32) -> (tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128x128xf32, #mma>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>, !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>)  : i32 {
//       // _CHECK:  scf.for
//       // _CHECK:  tt.load {{.*}} : !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>>
//       %44 = arith.subi %30, %c64_i32 : i32
//       %45 = arith.cmpi slt, %arg6, %44 : i32
//       %46 = tt.advance %arg11, [%c64_i32, %c0_i32] : <tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//       %47 = tt.advance %arg10, [%c0_i32, %c64_i32] : <tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//       %48 = tt.splat %45 : i1 -> tensor<128x128xi1, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
//       ttig.prefetch %47, %48 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>, ttig.block_io = "column_major"} : !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//       %49 = tt.splat %45 : i1 -> tensor<128x128xi1, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
//       ttig.prefetch %46, %49 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>, ttig.block_io = "row_major"} : !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//       %50 = tt.load %arg10 {ttig.block_io = "column_major"} : !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//       %51 = tt.load %arg11 {ttig.block_io = "row_major"} : !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//       %52 = tt.dot %21, %50, %cst_3, inputPrecision = tf32 : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
//       %53 = tt.splat %arg6 : i32 -> tensor<1x128xi32, #mma>
//       %54 = arith.addi %53, %35 : tensor<1x128xi32, #mma>
//       %55 = tt.broadcast %54 : tensor<1x128xi32, #mma> -> tensor<128x128xi32, #mma>
//       %56 = arith.cmpi sge, %36, %55 : tensor<128x128xi32, #mma>
//       %57 = arith.mulf %52, %23 : tensor<128x128xf32, #mma>
//       %58 = arith.select %56, %cst_3, %cst_4 : tensor<128x128xi1, #mma>, tensor<128x128xf32, #mma>
//       %59 = arith.addf %57, %58 : tensor<128x128xf32, #mma>
//       %60 = "tt.reduce"(%59) <{axis = 1 : i32}> ({
//       ^bb0(%arg12: f32, %arg13: f32):
//         %77 = arith.maxnumf %arg12, %arg13 : f32
//         tt.reduce.return %77 : f32
//       }) : (tensor<128x128xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//       %61 = arith.maxnumf %arg9, %60 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//       %62 = tt.expand_dims %61 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
//       %63 = tt.broadcast %62 : tensor<128x1xf32, #mma> -> tensor<128x128xf32, #mma>
//       %64 = arith.subf %59, %63 : tensor<128x128xf32, #mma>
//       %65 = math.exp2 %64 : tensor<128x128xf32, #mma>
//       %66 = "tt.reduce"(%65) <{axis = 1 : i32}> ({
//       ^bb0(%arg12: f32, %arg13: f32):
//         %77 = arith.addf %arg12, %arg13 : f32
//         tt.reduce.return %77 : f32
//       }) : (tensor<128x128xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//       %67 = arith.subf %arg9, %61 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//       %68 = math.exp2 %67 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//       %69 = arith.mulf %arg7, %68 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//       %70 = arith.addf %69, %66 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//       %71 = tt.expand_dims %68 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
//       %72 = tt.broadcast %71 : tensor<128x1xf32, #mma> -> tensor<128x128xf32, #mma>
//       %73 = arith.mulf %arg8, %72 : tensor<128x128xf32, #mma>
//       %74 = arith.truncf %65 : tensor<128x128xf32, #mma> to tensor<128x128xf16, #mma>
//       %75 = ttg.convert_layout %74 : tensor<128x128xf16, #mma> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
//       %76 = tt.dot %75, %51, %73, inputPrecision = tf32 : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
//       scf.yield %70, %76, %61, %47, %46 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128x128xf32, #mma>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>, !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//     }
//     %41 = tt.expand_dims %40#0 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
//     %42 = tt.broadcast %41 : tensor<128x1xf32, #mma> -> tensor<128x128xf32, #mma>
//     %43 = arith.divf %40#1, %42 : tensor<128x128xf32, #mma>
//     tt.store %16, %43 : !tt.ptr<tensor<128x128xf32, #mma>>
//     tt.return
//   }
// }

// -----
// TODO: Test @_attn_fwd_with_block_pointers_other_users_in_loop_before uses column_major attribute which is not yet implemented for tensor_descriptor.


// // _CHECK: #[[$BLOCKED:.+]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
// // _CHECK: #[[$DPAS:.+]] = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 1], repCluster = [2, 2], A = [16, 16], B = [16, 32], C = [16, 32]}>
// #blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
// #mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 1], repCluster = [2, 2], A = [16, 16], B = [16, 32], C = [16, 32]}>
// module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bfloat16_conversion, ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_2d_block_io, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
//   tt.func @_attn_fwd_with_block_pointers_other_users_in_loop_before(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: f32, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
//     // _CHECK-LABEL:   tt.func @_attn_fwd_with_block_pointers_other_users_in_loop_before
//     // COM: This test checks that in case of multiple users in the same loop
//     // COM: the loadOp is moved before the first user. (case where the other user is before the dotOp)
//     %c1984_i32 = arith.constant 1984 : i32
//     %c4194304_i64 = arith.constant 4194304 : i64
//     %c131072_i64 = arith.constant 131072 : i64
//     %c128_i32 = arith.constant 128 : i32
//     %c2048_i64 = arith.constant 2048 : i64
//     %c64_i64 = arith.constant 64 : i64
//     %c1_i64 = arith.constant 1 : i64
//     %c0_i32 = arith.constant 0 : i32
//     %cst = arith.constant 1.44269502 : f32
//     %c64_i32 = arith.constant 64 : i32
//     %c2048_i32 = arith.constant 2048 : i32
//     %cst_0 = arith.constant dense<1.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//     %cst_1 = arith.constant dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//     %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
//     %0 = tt.get_program_id z : i32
//     %1 = tt.get_program_id x : i32
//     %2 = tt.get_program_id y : i32
//     %3 = arith.extsi %1 : i32 to i64
//     %4 = arith.muli %3, %c4194304_i64 : i64
//     %5 = arith.extsi %2 : i32 to i64
//     %6 = arith.muli %5, %c131072_i64 : i64
//     %7 = arith.addi %4, %6 : i64
//     %8 = tt.addptr %arg0, %7 : !tt.ptr<f16>, i64
//     %9 = arith.muli %0, %c128_i32 : i32
//     %10 = tt.make_tensor_ptr %8, [%c2048_i64, %c64_i64], [%c64_i64, %c1_i64], [%9, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>>
//     %11 = tt.make_tensor_ptr %8, [%c2048_i64, %c64_i64], [%c64_i64, %c1_i64], [%9, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf16, #blocked>>
//     %12 = tt.addptr %arg2, %7 : !tt.ptr<f16>, i64
//     %13 = tt.make_tensor_ptr %12, [%c2048_i64, %c64_i64], [%c64_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//     %14 = tt.addptr %arg1, %7 : !tt.ptr<f16>, i64
//     %15 = tt.make_tensor_ptr %14, [%c64_i64, %c2048_i64], [%c1_i64, %c64_i64], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//     %16 = tt.addptr %arg5, %7 : !tt.ptr<f32>, i64
//     %17 = tt.make_tensor_ptr %16, [%c2048_i64, %c64_i64], [%c64_i64, %c1_i64], [%9, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf32, #mma>>
//     %18 = arith.mulf %arg3, %cst : f32
//     %20 = tt.load %11 {ttig.block_io = "row_major"} : !tt.ptr<tensor<128x128xf16, #blocked>>
//     // _CHECK:      ttig.prefetch {{.*}} : !tt.ptr<tensor<128x128xf16,  #[[$BLOCKED]]>>
//     // _CHECK-NOT:  tt.load {{.*}} : !tt.ptr<tensor<128x128xf16, #[[$BLOCKED]]>>
//     %21 = tt.splat %18 : f32 -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//     %22 = tt.splat %18 : f32 -> tensor<128x128xf32, #mma>
//     ttig.prefetch %15 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>, ttig.block_io = "column_major"} : !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//     ttig.prefetch %13 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>, ttig.block_io = "row_major"} : !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//     %23:5 = scf.for %arg6 = %c0_i32 to %c2048_i32 step %c64_i32 iter_args(%arg7 = %cst_0, %arg8 = %cst_2, %arg9 = %cst_1, %arg10 = %15, %arg11 = %13) -> (tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128x128xf32, #mma>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>, !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>)  : i32 {
//       // _CHECK:  scf.for
//       // _CHECK:  %[[LOAD1:.*]] = tt.load {{.*}} : !tt.ptr<tensor<128x128xf16, #[[$BLOCKED]]>>
//       %27 = arith.cmpi slt, %arg6, %c1984_i32 : i32
//       %28 = tt.advance %arg11, [%c64_i32, %c0_i32] : <tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//       %29 = tt.advance %arg10, [%c0_i32, %c64_i32] : <tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//       %30 = tt.splat %27 : i1 -> tensor<128x128xi1, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
//       ttig.prefetch %29, %30 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>, ttig.block_io = "column_major"} : !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//       ttig.prefetch %28, %30 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>, ttig.block_io = "row_major"} : !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//       %31 = tt.load %arg10 {ttig.block_io = "column_major"} : !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//       %32 = tt.load %arg11 {ttig.block_io = "row_major"} : !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//       // _CHECK: tt.store {{.*}}, %[[LOAD1]] : !tt.ptr<tensor<128x128xf16, #[[$BLOCKED]]>>
//       tt.store %11, %20 : !tt.ptr<tensor<128x128xf16, #blocked>>
//       %100 = ttg.convert_layout %20 : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
//       %33 = tt.dot %100, %31, %cst_2, inputPrecision = tf32 : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
//       %34 = "tt.reduce"(%33) <{axis = 1 : i32}> ({
//       ^bb0(%arg12: f32, %arg13: f32):
//         %53 = arith.maxnumf %arg12, %arg13 : f32
//         tt.reduce.return %53 : f32
//       }) : (tensor<128x128xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//       %35 = arith.mulf %34, %21 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//       %36 = arith.maxnumf %arg9, %35 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//       %37 = arith.mulf %33, %22 : tensor<128x128xf32, #mma>
//       %38 = tt.expand_dims %36 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
//       %39 = tt.broadcast %38 : tensor<128x1xf32, #mma> -> tensor<128x128xf32, #mma>
//       %40 = arith.subf %37, %39 : tensor<128x128xf32, #mma>
//       %41 = math.exp2 %40 : tensor<128x128xf32, #mma>
//       %42 = "tt.reduce"(%41) <{axis = 1 : i32}> ({
//       ^bb0(%arg12: f32, %arg13: f32):
//         %53 = arith.addf %arg12, %arg13 : f32
//         tt.reduce.return %53 : f32
//       }) : (tensor<128x128xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//       %43 = arith.subf %arg9, %36 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//       %44 = math.exp2 %43 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//       %45 = arith.mulf %arg7, %44 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//       %46 = arith.addf %45, %42 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//       %47 = tt.expand_dims %44 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
//       %48 = tt.broadcast %47 : tensor<128x1xf32, #mma> -> tensor<128x128xf32, #mma>
//       %49 = arith.mulf %arg8, %48 : tensor<128x128xf32, #mma>
//       %50 = arith.truncf %41 : tensor<128x128xf32, #mma> to tensor<128x128xf16, #mma>
//       %51 = ttg.convert_layout %50 : tensor<128x128xf16, #mma> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
//       %52 = tt.dot %51, %32, %49, inputPrecision = tf32 : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
//       scf.yield %46, %52, %36, %29, %28 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128x128xf32, #mma>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>, !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//     }
//     %24 = tt.expand_dims %23#0 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
//     %25 = tt.broadcast %24 : tensor<128x1xf32, #mma> -> tensor<128x128xf32, #mma>
//     %26 = arith.divf %23#1, %25 : tensor<128x128xf32, #mma>
//     tt.store %17, %26 : !tt.ptr<tensor<128x128xf32, #mma>>
//     tt.return
//   }
// }


// -----
// TODO: Test @_attn_fwd_with_block_pointers_other_users_in_loop_after uses column_major attribute which is not yet implemented for tensor_descriptor.


// // _CHECK: #[[$BLOCKED:.+]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
// // _CHECK: #[[$DPAS:.+]] = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 1], repCluster = [2, 2], A = [16, 16], B = [16, 32], C = [16, 32]}>
// #blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
// #mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 1], repCluster = [2, 2], A = [16, 16], B = [16, 32], C = [16, 32]}>
// module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bfloat16_conversion, ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_2d_block_io, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
//   tt.func @_attn_fwd_with_block_pointers_other_users_in_loop_after(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: f32, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
//     // _CHECK-LABEL:   tt.func @_attn_fwd_with_block_pointers_other_users_in_loop_after
//     // COM: This test checks that in case of multiple users in the same loop
//     // COM: the loadOp is moved before the first user. (case where the other user is after the dotOp)
//     %c1984_i32 = arith.constant 1984 : i32
//     %c4194304_i64 = arith.constant 4194304 : i64
//     %c131072_i64 = arith.constant 131072 : i64
//     %c128_i32 = arith.constant 128 : i32
//     %c2048_i64 = arith.constant 2048 : i64
//     %c64_i64 = arith.constant 64 : i64
//     %c1_i64 = arith.constant 1 : i64
//     %c0_i32 = arith.constant 0 : i32
//     %cst = arith.constant 1.44269502 : f32
//     %c64_i32 = arith.constant 64 : i32
//     %c2048_i32 = arith.constant 2048 : i32
//     %cst_0 = arith.constant dense<1.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//     %cst_1 = arith.constant dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//     %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
//     %0 = tt.get_program_id z : i32
//     %1 = tt.get_program_id x : i32
//     %2 = tt.get_program_id y : i32
//     %3 = arith.extsi %1 : i32 to i64
//     %4 = arith.muli %3, %c4194304_i64 : i64
//     %5 = arith.extsi %2 : i32 to i64
//     %6 = arith.muli %5, %c131072_i64 : i64
//     %7 = arith.addi %4, %6 : i64
//     %8 = tt.addptr %arg0, %7 : !tt.ptr<f16>, i64
//     %9 = arith.muli %0, %c128_i32 : i32
//     %10 = tt.make_tensor_ptr %8, [%c2048_i64, %c64_i64], [%c64_i64, %c1_i64], [%9, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>>
//     %11 = tt.make_tensor_ptr %8, [%c2048_i64, %c64_i64], [%c64_i64, %c1_i64], [%9, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf16, #blocked>>
//     %12 = tt.addptr %arg2, %7 : !tt.ptr<f16>, i64
//     %13 = tt.make_tensor_ptr %12, [%c2048_i64, %c64_i64], [%c64_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//     %14 = tt.addptr %arg1, %7 : !tt.ptr<f16>, i64
//     %15 = tt.make_tensor_ptr %14, [%c64_i64, %c2048_i64], [%c1_i64, %c64_i64], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//     %16 = tt.addptr %arg5, %7 : !tt.ptr<f32>, i64
//     %17 = tt.make_tensor_ptr %16, [%c2048_i64, %c64_i64], [%c64_i64, %c1_i64], [%9, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf32, #mma>>
//     %18 = arith.mulf %arg3, %cst : f32
//     %20 = tt.load %11 {ttig.block_io = "row_major"} : !tt.ptr<tensor<128x128xf16, #blocked>>
//     // _CHECK:      ttig.prefetch {{.*}} : !tt.ptr<tensor<128x128xf16,  #[[$BLOCKED]]>>
//     // _CHECK-NOT:  tt.load {{.*}} : !tt.ptr<tensor<128x128xf16, #[[$BLOCKED]]>>
//     %21 = tt.splat %18 : f32 -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//     %22 = tt.splat %18 : f32 -> tensor<128x128xf32, #mma>
//     ttig.prefetch %15 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>, ttig.block_io = "column_major"} : !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//     ttig.prefetch %13 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>, ttig.block_io = "row_major"} : !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//     %23:5 = scf.for %arg6 = %c0_i32 to %c2048_i32 step %c64_i32 iter_args(%arg7 = %cst_0, %arg8 = %cst_2, %arg9 = %cst_1, %arg10 = %15, %arg11 = %13) -> (tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128x128xf32, #mma>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>, !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>)  : i32 {
//       // _CHECK:  scf.for
//       // _CHECK:  %[[LOAD1:.*]] = tt.load {{.*}} : !tt.ptr<tensor<128x128xf16, #[[$BLOCKED]]>>
//       %27 = arith.cmpi slt, %arg6, %c1984_i32 : i32
//       %28 = tt.advance %arg11, [%c64_i32, %c0_i32] : <tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//       %29 = tt.advance %arg10, [%c0_i32, %c64_i32] : <tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//       %30 = tt.splat %27 : i1 -> tensor<128x128xi1, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
//       ttig.prefetch %29, %30 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>, ttig.block_io = "column_major"} : !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//       ttig.prefetch %28, %30 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>, ttig.block_io = "row_major"} : !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//       %31 = tt.load %arg10 {ttig.block_io = "column_major"} : !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//       %32 = tt.load %arg11 {ttig.block_io = "row_major"} : !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//       %100 = ttg.convert_layout %20 : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
//       %33 = tt.dot %100, %31, %cst_2, inputPrecision = tf32 : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
//       %34 = "tt.reduce"(%33) <{axis = 1 : i32}> ({
//       ^bb0(%arg12: f32, %arg13: f32):
//         %53 = arith.maxnumf %arg12, %arg13 : f32
//         tt.reduce.return %53 : f32
//       }) : (tensor<128x128xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//       %35 = arith.mulf %34, %21 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//       %36 = arith.maxnumf %arg9, %35 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//       %37 = arith.mulf %33, %22 : tensor<128x128xf32, #mma>
//       %38 = tt.expand_dims %36 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
//       %39 = tt.broadcast %38 : tensor<128x1xf32, #mma> -> tensor<128x128xf32, #mma>
//       %40 = arith.subf %37, %39 : tensor<128x128xf32, #mma>
//       %41 = math.exp2 %40 : tensor<128x128xf32, #mma>
//       %42 = "tt.reduce"(%41) <{axis = 1 : i32}> ({
//       ^bb0(%arg12: f32, %arg13: f32):
//         %53 = arith.addf %arg12, %arg13 : f32
//         tt.reduce.return %53 : f32
//       }) : (tensor<128x128xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//       %43 = arith.subf %arg9, %36 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//       %44 = math.exp2 %43 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//       %45 = arith.mulf %arg7, %44 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//       %46 = arith.addf %45, %42 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
//       %47 = tt.expand_dims %44 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
//       %48 = tt.broadcast %47 : tensor<128x1xf32, #mma> -> tensor<128x128xf32, #mma>
//       %49 = arith.mulf %arg8, %48 : tensor<128x128xf32, #mma>
//       %50 = arith.truncf %41 : tensor<128x128xf32, #mma> to tensor<128x128xf16, #mma>
//       %51 = ttg.convert_layout %50 : tensor<128x128xf16, #mma> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
//       %52 = tt.dot %51, %32, %49, inputPrecision = tf32 : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
//       // _CHECK: tt.store {{.*}}, %[[LOAD1]] : !tt.ptr<tensor<128x128xf16, #[[$BLOCKED]]>>
//       tt.store %11, %20 : !tt.ptr<tensor<128x128xf16, #blocked>>
//       scf.yield %46, %52, %36, %29, %28 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128x128xf32, #mma>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>, !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
//     }
//     %24 = tt.expand_dims %23#0 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
//     %25 = tt.broadcast %24 : tensor<128x1xf32, #mma> -> tensor<128x128xf32, #mma>
//     %26 = arith.divf %23#1, %25 : tensor<128x128xf32, #mma>
//     tt.store %17, %26 : !tt.ptr<tensor<128x128xf32, #mma>>
//     tt.return
//   }
// }
