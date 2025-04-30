// RUN: triton-opt %s -split-input-file -tritonintelgpu-reduce-variable-liveness -cse | FileCheck %s

// CHECK: #[[$DPAS:.+]] = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {triton_intel_gpu.support_sg_2d_block, "ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func public @matmul_kernel_small_tensor(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    // CHECK-LABEL:   tt.func public @matmul_kernel_small_tensor
    %cst = arith.constant dense<0.000000e+00> : tensor<16x256xf32, #dpas>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    %0 = tt.make_tensor_ptr %arg0, [%c0_i64, %c0_i64], [%c0_i64, %c0_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x64xf16, #dot0>>
    // CHECK:  tt.load {{.*}} : !tt.ptr<tensor<16x64xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>>
    %1 = tt.make_tensor_ptr %arg1, [%c0_i64, %c0_i64], [%c0_i64, %c0_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x256xf16, #dot1>>
    %2 = tt.load %0 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<16x64xf16, #dot0>>
    %3 = arith.muli %c64_i32, %c0_i32 : i32
    triton_intel_gpu.prefetch %1 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>} : !tt.ptr<tensor<64x256xf16, #dot1>>
    %4:2 = scf.for %arg2 = %c0_i32 to %c64_i32 step %c64_i32 iter_args(%arg3 = %cst, %arg4 = %1) -> (tensor<16x256xf32, #dpas>, !tt.ptr<tensor<64x256xf16, #dot1>>)  : i32 {
      // CHECK:  scf.for
      // CHECK-NOT:  tt.load {{.*}} : !tt.ptr<tensor<16x64xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>>
      %7 = tt.advance %arg4, [%c64_i32, %c0_i32] : <tensor<64x256xf16, #dot1>>
      triton_intel_gpu.prefetch %7 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>} : !tt.ptr<tensor<64x256xf16, #dot1>>
      %8 = tt.load %arg4 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x256xf16, #dot1>>
      %9 = tt.dot %2, %8, %arg3, inputPrecision = tf32 : tensor<16x64xf16, #dot0> * tensor<64x256xf16, #dot1> -> tensor<16x256xf32, #dpas>
      scf.yield %9, %7 : tensor<16x256xf32, #dpas>, !tt.ptr<tensor<64x256xf16, #dot1>>
    }
    tt.return
  }
}

// -----

// CHECK: #[[$DPAS:.+]] = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {triton_intel_gpu.support_sg_2d_block, "ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func public @matmul_kernel_no_candidate_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    // CHECK-LABEL:   tt.func public @matmul_kernel_no_candidate_load
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #dpas>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    %0 = tt.make_tensor_ptr %arg0, [%c0_i64, %c0_i64], [%c0_i64, %c0_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x256xf16, #dot0>>
    // CHECK:  tt.load {{.*}} : !tt.ptr<tensor<128x256xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>>
    %1 = tt.make_tensor_ptr %arg1, [%c0_i64, %c0_i64], [%c0_i64, %c0_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<256x256xf16, #dot1>>
    %2 = tt.load %0 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<128x256xf16, #dot0>>
    %3 = arith.muli %c64_i32, %c0_i32 : i32
    triton_intel_gpu.prefetch %1 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>} : !tt.ptr<tensor<256x256xf16, #dot1>>
    %4:2 = scf.for %arg2 = %c0_i32 to %c64_i32 step %c64_i32 iter_args(%arg3 = %cst, %arg4 = %1) -> (tensor<128x256xf32, #dpas>, !tt.ptr<tensor<256x256xf16, #dot1>>)  : i32 {
      // CHECK:  scf.for
      // CHECK-NOT:  tt.load {{.*}} : !tt.ptr<tensor<128x256xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>>
      %7 = tt.advance %arg4, [%c64_i32, %c0_i32] : <tensor<256x256xf16, #dot1>>
      triton_intel_gpu.prefetch %7 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>} : !tt.ptr<tensor<256x256xf16, #dot1>>
      %8 = tt.load %arg4 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<256x256xf16, #dot1>>
      %9 = tt.dot %2, %8, %arg3, inputPrecision = tf32 : tensor<128x256xf16, #dot0> * tensor<256x256xf16, #dot1> -> tensor<128x256xf32, #dpas>
      scf.yield %9, %7 : tensor<128x256xf32, #dpas>, !tt.ptr<tensor<256x256xf16, #dot1>>
    }
    tt.return
  }
}


// -----

// CHECK: #[[$DPAS:.+]] = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 2], repCluster = [2, 2], A = [16, 16], B = [16, 32], C = [16, 32]}>
#mma = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 2], repCluster = [2, 2], A = [16, 16], B = [16, 32], C = [16, 32]}>
module attributes {triton_intel_gpu.min_sg_size = 16 : i32, triton_intel_gpu.support_bf16_conversion, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block, triton_intel_gpu.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
  tt.func public @_attn_fwd(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: f32, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    // CHECK-LABEL:   tt.func public @_attn_fwd
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
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<512x64xf32, #mma>
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id z : i32
    %2 = arith.extsi %1 : i32 to i64
    %3 = arith.muli %2, %c8192_i64 : i64
    %4 = tt.addptr %arg0, %3 : !tt.ptr<f16>, i64
    %5 = arith.muli %0, %c128_i32 : i32
    %6 = tt.make_tensor_ptr %4, [%c128_i64, %c64_i64], [%c64_i64, %c1_i64], [%5, %c0_i32] {order = array<i32: 1, 0>} : <tensor<512x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>>
    %7 = tt.addptr %arg2, %3 : !tt.ptr<f16>, i64
    %8 = tt.make_tensor_ptr %7, [%c128_i64, %c64_i64], [%c64_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
    %9 = tt.addptr %arg1, %3 : !tt.ptr<f16>, i64
    %10 = tt.make_tensor_ptr %9, [%c64_i64, %c128_i64], [%c1_i64, %c64_i64], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
    %11 = tt.addptr %arg5, %3 : !tt.ptr<f32>, i64
    %12 = tt.make_tensor_ptr %11, [%c128_i64, %c64_i64], [%c64_i64, %c1_i64], [%5, %c0_i32] {order = array<i32: 1, 0>} : <tensor<512x64xf32, #mma>>
    %13 = arith.mulf %arg3, %cst : f32
    // CHECK:      triton_intel_gpu.prefetch {{.*}} : !tt.ptr<tensor<512x64xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>>
    // CHECK-NOT:  tt.load {{.*}} : !tt.ptr<tensor<512x64xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>>
    %14 = tt.load %6 {triton_intel_gpu.block_io = "row_major"} : !tt.ptr<tensor<512x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>>
    %15 = tt.splat %13 : f32 -> tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %16 = tt.splat %13 : f32 -> tensor<512x64xf32, #mma>
    %17:5 = scf.for %arg6 = %c0_i32 to %c128_i32 step %c64_i32 iter_args(%arg7 = %cst_0, %arg8 = %cst_2, %arg9 = %cst_1, %arg10 = %10, %arg11 = %8) -> (tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<512x64xf32, #mma>, tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>, !tt.ptr<tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>, !tt.ptr<tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>)  : i32 {
      // CHECK:  scf.for
      // CHECK:  tt.load {{.*}} : !tt.ptr<tensor<512x64xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>>
      %21 = tt.load %arg10 {triton_intel_gpu.block_io = "column_major"} : !tt.ptr<tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
      %22 = tt.dot %14, %21, %cst_2, inputPrecision = tf32 : tensor<512x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<512x64xf32, #mma>
      %23 = "tt.reduce"(%22) <{axis = 1 : i32}> ({
      ^bb0(%arg12: f32, %arg13: f32):
        %45 = arith.maxnumf %arg12, %arg13 : f32
        tt.reduce.return %45 : f32
      }) : (tensor<512x64xf32, #mma>) -> tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %24 = arith.mulf %23, %15 : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %25 = arith.maxnumf %arg9, %24 : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %26 = arith.mulf %22, %16 : tensor<512x64xf32, #mma>
      %27 = tt.expand_dims %25 {axis = 1 : i32} : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<512x1xf32, #mma>
      %28 = tt.broadcast %27 : tensor<512x1xf32, #mma> -> tensor<512x64xf32, #mma>
      %29 = arith.subf %26, %28 : tensor<512x64xf32, #mma>
      %30 = math.exp2 %29 : tensor<512x64xf32, #mma>
      %31 = "tt.reduce"(%30) <{axis = 1 : i32}> ({
      ^bb0(%arg12: f32, %arg13: f32):
        %45 = arith.addf %arg12, %arg13 : f32
        tt.reduce.return %45 : f32
      }) : (tensor<512x64xf32, #mma>) -> tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %32 = arith.subf %arg9, %25 : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %33 = math.exp2 %32 : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %34 = arith.mulf %arg7, %33 : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %35 = arith.addf %34, %31 : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %36 = tt.expand_dims %33 {axis = 1 : i32} : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<512x1xf32, #mma>
      %37 = tt.broadcast %36 : tensor<512x1xf32, #mma> -> tensor<512x64xf32, #mma>
      %38 = arith.mulf %arg8, %37 : tensor<512x64xf32, #mma>
      %39 = tt.load %arg11 {triton_intel_gpu.block_io = "row_major"} : !tt.ptr<tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
      %40 = arith.truncf %30 : tensor<512x64xf32, #mma> to tensor<512x64xf16, #mma>
      %41 = ttg.convert_layout %40 : tensor<512x64xf16, #mma> -> tensor<512x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %42 = tt.dot %41, %39, %38, inputPrecision = tf32 : tensor<512x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<512x64xf32, #mma>
      %43 = tt.advance %arg11, [%c64_i32, %c0_i32] : <tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
      %44 = tt.advance %arg10, [%c0_i32, %c64_i32] : <tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
      scf.yield %35, %42, %25, %44, %43 : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<512x64xf32, #mma>, tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>>, !tt.ptr<tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>, !tt.ptr<tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
    } {tt.divisibility_arg1 = dense<64> : tensor<1xi32>}
    %18 = tt.expand_dims %17#0 {axis = 1 : i32} : tensor<512xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<512x1xf32, #mma>
    %19 = tt.broadcast %18 : tensor<512x1xf32, #mma> -> tensor<512x64xf32, #mma>
    %20 = arith.divf %17#1, %19 : tensor<512x64xf32, #mma>
    tt.store %12, %20 : !tt.ptr<tensor<512x64xf32, #mma>>
    tt.return
  }
}
