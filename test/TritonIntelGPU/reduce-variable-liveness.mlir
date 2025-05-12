// RUN: triton-opt %s -split-input-file -tritonintelgpu-reduce-variable-liveness -cse -cse | FileCheck %s

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

// -----

// CHECK: #[[$DPAS:.+]] = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 1], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [16], warpsPerCTA = [8], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 1], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
module attributes {triton_intel_gpu.min_sg_size = 16 : i32, triton_intel_gpu.support_bf16_conversion, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block, triton_intel_gpu.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
  tt.func public @_attn_fwd_tensor_pointers(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: f32, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    // CHECK-LABEL:   tt.func public @_attn_fwd_tensor_pointers
    %c256_i32 = arith.constant 256 : i32
    %c64_i64 = arith.constant 64 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 1.44269502 : f32
    %c64_i32 = arith.constant 64 : i32
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<256xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %cst_1 = arith.constant dense<0xFF800000> : tensor<256xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<256x64xf32, #mma>
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.divsi %1, %arg19 : i32
    %3 = arith.remsi %1, %arg19 : i32
    %4 = arith.extsi %2 : i32 to i64
    %5 = arith.extsi %arg6 : i32 to i64
    %6 = arith.muli %4, %5 : i64
    %7 = arith.extsi %3 : i32 to i64
    %8 = arith.extsi %arg7 : i32 to i64
    %9 = arith.muli %7, %8 : i64
    %10 = arith.addi %6, %9 : i64
    %11 = tt.addptr %arg5, %10 : !tt.ptr<f16>, i64
    %12 = arith.muli %0, %c256_i32 : i32
    %13 = arith.extsi %arg20 : i32 to i64
    %14 = arith.extsi %arg17 : i32 to i64
    %15 = tt.make_tensor_ptr %11, [%13, %c64_i64], [%14, %c1_i64], [%12, %c0_i32] {order = array<i32: 1, 0>} : <tensor<256x64xf16, #blocked>>
    %16 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %17 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked1>
    %18 = tt.splat %12 : i32 -> tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %19 = tt.splat %12 : i32 -> tensor<256xi32, #blocked1>
    %20 = arith.addi %18, %16 : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %21 = arith.addi %19, %17 : tensor<256xi32, #blocked1>
    %22 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %23 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %24 = tt.addptr %arg0, %10 : !tt.ptr<f16>, i64
    %25 = tt.expand_dims %20 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xi32, #blocked>
    %26 = tt.splat %arg8 : i32 -> tensor<256x1xi32, #blocked>
    %27 = arith.muli %25, %26 : tensor<256x1xi32, #blocked>
    %28 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %29 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %30 = tt.expand_dims %28 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %31 = tt.expand_dims %29 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x64xi32, #blocked2>
    %32 = tt.broadcast %27 : tensor<256x1xi32, #blocked> -> tensor<256x64xi32, #blocked>
    %33 = tt.broadcast %30 : tensor<1x64xi32, #blocked> -> tensor<256x64xi32, #blocked>
    %34 = arith.addi %32, %33 : tensor<256x64xi32, #blocked>
    %35 = tt.splat %24 : !tt.ptr<f16> -> tensor<256x64x!tt.ptr<f16>, #blocked>
    %36 = tt.addptr %35, %34 : tensor<256x64x!tt.ptr<f16>, #blocked>, tensor<256x64xi32, #blocked>
    // CHECK:      triton_intel_gpu.prefetch {{.*}} : tensor<256x64x!tt.ptr<f16>, #blocked>
    // CHECK-NOT:  tt.load {{.*}} {triton_intel_gpu.block_io = "row_major"} : tensor<256x64x!tt.ptr<f16>, #blocked>
    %37 = tt.addptr %arg1, %10 : !tt.ptr<f16>, i64
    %38 = tt.addptr %arg2, %10 : !tt.ptr<f16>, i64
    %39 = arith.mulf %arg3, %cst : f32
    %40 = tt.load %36 {triton_intel_gpu.block_io = "row_major"} : tensor<256x64x!tt.ptr<f16>, #blocked>
    %41 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %42 = tt.expand_dims %41 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1xi32, #blocked2>
    %43 = tt.splat %arg11 : i32 -> tensor<1x64xi32, #blocked2>
    %44 = tt.broadcast %42 : tensor<64x1xi32, #blocked2> -> tensor<64x64xi32, #blocked2>
    %45 = tt.splat %37 : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>, #blocked2>
    %46 = tt.splat %39 : f32 -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %47 = tt.splat %39 : f32 -> tensor<256x64xf32, #mma>
    %48 = tt.splat %arg14 : i32 -> tensor<1x64xi32, #blocked2>
    %49 = arith.muli %31, %48 : tensor<1x64xi32, #blocked2>
    %50 = tt.broadcast %49 : tensor<1x64xi32, #blocked2> -> tensor<64x64xi32, #blocked2>
    %51 = tt.splat %38 : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>, #blocked2>
    %52:3 = scf.for %arg21 = %c0_i32 to %arg20 step %c64_i32 iter_args(%arg22 = %cst_0, %arg23 = %cst_2, %arg24 = %cst_1) -> (tensor<256xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<256x64xf32, #mma>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #mma}>>)  : i32 {
      // CHECK:  scf.for
      // CHECK:  tt.load {{.*}} {triton_intel_gpu.block_io = "row_major"} : tensor<256x64x!tt.ptr<f16>, #blocked>
      %65 = arith.muli %arg21, %c64_i32 : i32
      %66 = tt.splat %65 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
      %67 = tt.splat %65 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
      %68 = arith.addi %66, %22 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
      %69 = arith.addi %67, %23 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
      %70 = tt.expand_dims %68 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x64xi32, #blocked2>
      %71 = arith.muli %70, %43 : tensor<1x64xi32, #blocked2>
      %72 = tt.broadcast %71 : tensor<1x64xi32, #blocked2> -> tensor<64x64xi32, #blocked2>
      %73 = arith.addi %44, %72 : tensor<64x64xi32, #blocked2>
      %74 = tt.addptr %45, %73 : tensor<64x64x!tt.ptr<f16>, #blocked2>, tensor<64x64xi32, #blocked2>
      %75 = tt.load %74 {triton_intel_gpu.block_io = "column_major"} : tensor<64x64x!tt.ptr<f16>, #blocked2>
      %76 = ttg.convert_layout %40 : tensor<256x64xf16, #blocked> -> tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %77 = ttg.convert_layout %75 : tensor<64x64xf16, #blocked2> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %78 = tt.dot %76, %77, %cst_2, inputPrecision = tf32 : tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<256x64xf32, #mma>
      %79 = "tt.reduce"(%78) <{axis = 1 : i32}> ({
      ^bb0(%arg25: f32, %arg26: f32):
        %104 = arith.maxnumf %arg25, %arg26 : f32
        tt.reduce.return %104 : f32
      }) : (tensor<256x64xf32, #mma>) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %80 = arith.mulf %79, %46 : tensor<256xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %81 = arith.maxnumf %arg24, %80 : tensor<256xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %82 = arith.mulf %78, %47 : tensor<256x64xf32, #mma>
      %83 = tt.expand_dims %81 {axis = 1 : i32} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<256x1xf32, #mma>
      %84 = tt.broadcast %83 : tensor<256x1xf32, #mma> -> tensor<256x64xf32, #mma>
      %85 = arith.subf %82, %84 : tensor<256x64xf32, #mma>
      %86 = math.exp2 %85 : tensor<256x64xf32, #mma>
      %87 = "tt.reduce"(%86) <{axis = 1 : i32}> ({
      ^bb0(%arg25: f32, %arg26: f32):
        %104 = arith.addf %arg25, %arg26 : f32
        tt.reduce.return %104 : f32
      }) : (tensor<256x64xf32, #mma>) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %88 = arith.subf %arg24, %81 : tensor<256xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %89 = math.exp2 %88 : tensor<256xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %90 = arith.mulf %arg22, %89 : tensor<256xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %91 = arith.addf %90, %87 : tensor<256xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %92 = tt.expand_dims %89 {axis = 1 : i32} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<256x1xf32, #mma>
      %93 = tt.broadcast %92 : tensor<256x1xf32, #mma> -> tensor<256x64xf32, #mma>
      %94 = arith.mulf %arg23, %93 : tensor<256x64xf32, #mma>
      %95 = tt.expand_dims %69 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1xi32, #blocked2>
      %96 = tt.broadcast %95 : tensor<64x1xi32, #blocked2> -> tensor<64x64xi32, #blocked2>
      %97 = arith.addi %96, %50 : tensor<64x64xi32, #blocked2>
      %98 = tt.addptr %51, %97 : tensor<64x64x!tt.ptr<f16>, #blocked2>, tensor<64x64xi32, #blocked2>
      %99 = tt.load %98 {triton_intel_gpu.block_io = "row_major"} : tensor<64x64x!tt.ptr<f16>, #blocked2>
      %100 = arith.truncf %86 : tensor<256x64xf32, #mma> to tensor<256x64xf16, #mma>
      %101 = ttg.convert_layout %100 : tensor<256x64xf16, #mma> -> tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %102 = ttg.convert_layout %99 : tensor<64x64xf16, #blocked2> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %103 = tt.dot %101, %102, %94, inputPrecision = tf32 : tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<256x64xf32, #mma>
      scf.yield %91, %103, %81 : tensor<256xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<256x64xf32, #mma>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    } {tt.divisibility_arg1 = dense<64> : tensor<1xi32>}
    %53 = math.log2 %52#0 : tensor<256xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %54 = arith.addf %52#2, %53 : tensor<256xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %55 = tt.expand_dims %52#0 {axis = 1 : i32} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<256x1xf32, #mma>
    %56 = tt.broadcast %55 : tensor<256x1xf32, #mma> -> tensor<256x64xf32, #mma>
    %57 = arith.divf %52#1, %56 : tensor<256x64xf32, #mma>
    %58 = arith.muli %1, %arg20 : i32
    %59 = tt.addptr %arg4, %58 : !tt.ptr<f32>, i32
    %60 = tt.splat %59 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked1>
    %61 = tt.addptr %60, %21 : tensor<256x!tt.ptr<f32>, #blocked1>, tensor<256xi32, #blocked1>
    %62 = ttg.convert_layout %54 : tensor<256xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<256xf32, #blocked1>
    tt.store %61, %62 : tensor<256x!tt.ptr<f32>, #blocked1>
    %63 = arith.truncf %57 : tensor<256x64xf32, #mma> to tensor<256x64xf16, #mma>
    %64 = ttg.convert_layout %63 : tensor<256x64xf16, #mma> -> tensor<256x64xf16, #blocked>
    tt.store %15, %64 : !tt.ptr<tensor<256x64xf16, #blocked>>
    tt.return
  }
}
