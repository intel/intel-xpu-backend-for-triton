// RUN: triton-opt %s -tritonintelgpu-prefetch | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 8], warpsPerCTA = [16, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 2], warpsPerCTA = [1, 16], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [16, 1], order = [1, 0]}>
#mma = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 2], repCluster = [2, 2], A = [16, 16], B = [16, 32], C = [16, 32]}>
module attributes {triton_intel_gpu.min_sg_size = 16 : i32, triton_intel_gpu.support_bf16_conversion, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block, triton_intel_gpu.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
  tt.func public @_attn_fwd(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: f32, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c8192_i64 = arith.constant 8192 : i64
    %c128_i32 = arith.constant 128 : i32
    %c128_i64 = arith.constant 128 : i64
    %c64_i64 = arith.constant 64 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 1.44269502 : f32
    %c64_i32 = arith.constant 64 : i32
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %cst_1 = arith.constant dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id z : i32
    %2 = arith.extsi %1 : i32 to i64
    %3 = arith.muli %2, %c8192_i64 : i64
    %4 = tt.addptr %arg0, %3 : !tt.ptr<f16>, i64
    %5 = arith.muli %0, %c128_i32 : i32
    %6 = tt.make_tensor_ptr %4, [%c128_i64, %c64_i64], [%c64_i64, %c1_i64], [%5, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x64xf16, #blocked>>
    %7 = tt.addptr %arg2, %3 : !tt.ptr<f16>, i64
    %8 = tt.make_tensor_ptr %7, [%c128_i64, %c64_i64], [%c64_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x64xf16, #blocked>>
    %9 = tt.addptr %arg1, %3 : !tt.ptr<f16>, i64
    %10 = tt.make_tensor_ptr %9, [%c64_i64, %c128_i64], [%c1_i64, %c64_i64], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<64x64xf16, #blocked1>>
    %11 = tt.addptr %arg5, %3 : !tt.ptr<f32>, i64
    %12 = tt.make_tensor_ptr %11, [%c128_i64, %c64_i64], [%c64_i64, %c1_i64], [%5, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x64xf32, #blocked2>>
    %13 = arith.mulf %arg3, %cst : f32
    %14 = tt.load %6 {triton_intel_gpu.block_io = "row_major"} : !tt.ptr<tensor<128x64xf16, #blocked>>
    %15 = tt.splat %13 : f32 -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %16 = tt.splat %13 : f32 -> tensor<128x64xf32, #mma>
    %17:5 = scf.for %arg6 = %c0_i32 to %c128_i32 step %c64_i32 iter_args(%arg7 = %cst_0, %arg8 = %cst_2, %arg9 = %cst_1, %arg10 = %8, %arg11 = %10) -> (tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128x64xf32, #mma>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, !tt.ptr<tensor<64x64xf16, #blocked>>, !tt.ptr<tensor<64x64xf16, #blocked1>>)  : i32 {
      %22 = tt.load %arg11 {triton_intel_gpu.block_io = "column_major"} : !tt.ptr<tensor<64x64xf16, #blocked1>>
      %23 = ttg.convert_layout %14 : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %24 = ttg.convert_layout %22 : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %25 = tt.dot %23, %24, %cst_2, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x64xf32, #mma>
      %26 = "tt.reduce"(%25) <{axis = 1 : i32}> ({
      ^bb0(%arg12: f32, %arg13: f32):
        %49 = arith.maxnumf %arg12, %arg13 : f32
        tt.reduce.return %49 : f32
      }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %27 = arith.mulf %26, %15 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %28 = arith.maxnumf %arg9, %27 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %29 = arith.mulf %25, %16 : tensor<128x64xf32, #mma>
      %30 = tt.expand_dims %28 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
      %31 = tt.broadcast %30 : tensor<128x1xf32, #mma> -> tensor<128x64xf32, #mma>
      %32 = arith.subf %29, %31 : tensor<128x64xf32, #mma>
      %33 = math.exp2 %32 : tensor<128x64xf32, #mma>
      %34 = "tt.reduce"(%33) <{axis = 1 : i32}> ({
      ^bb0(%arg12: f32, %arg13: f32):
        %49 = arith.addf %arg12, %arg13 : f32
        tt.reduce.return %49 : f32
      }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %35 = arith.subf %arg9, %28 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %36 = math.exp2 %35 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %37 = arith.mulf %arg7, %36 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %38 = arith.addf %37, %34 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %39 = tt.expand_dims %36 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
      %40 = tt.broadcast %39 : tensor<128x1xf32, #mma> -> tensor<128x64xf32, #mma>
      %41 = arith.mulf %arg8, %40 : tensor<128x64xf32, #mma>
      %42 = tt.load %arg10 {triton_intel_gpu.block_io = "row_major"} : !tt.ptr<tensor<64x64xf16, #blocked>>
      %43 = arith.truncf %33 : tensor<128x64xf32, #mma> to tensor<128x64xf16, #mma>
      %44 = ttg.convert_layout %43 : tensor<128x64xf16, #mma> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %45 = ttg.convert_layout %42 : tensor<64x64xf16, #blocked> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %46 = tt.dot %44, %45, %41, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x64xf32, #mma>
      %47 = tt.advance %arg10, [%c64_i32, %c0_i32] : <tensor<64x64xf16, #blocked>>
      %48 = tt.advance %arg11, [%c0_i32, %c64_i32] : <tensor<64x64xf16, #blocked1>>
      scf.yield %38, %46, %28, %47, %48 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128x64xf32, #mma>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, !tt.ptr<tensor<64x64xf16, #blocked>>, !tt.ptr<tensor<64x64xf16, #blocked1>>
    } {tt.divisibility_arg1 = dense<64> : tensor<1xi32>}
    %18 = tt.expand_dims %17#0 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
    %19 = tt.broadcast %18 : tensor<128x1xf32, #mma> -> tensor<128x64xf32, #mma>
    %20 = arith.divf %17#1, %19 : tensor<128x64xf32, #mma>
    %21 = ttg.convert_layout %20 : tensor<128x64xf32, #mma> -> tensor<128x64xf32, #blocked2>
    tt.store %12, %21 : !tt.ptr<tensor<128x64xf32, #blocked2>>
    tt.return
  }
}
