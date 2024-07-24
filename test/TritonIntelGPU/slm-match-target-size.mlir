// RUN: env TRITON_INTEL_ENABLE_SLM=1 triton-opt %s -split-input-file -tritonintelgpu-match-target-size | FileCheck %s --check-prefix=SLM-CHECK

#warp = #triton_intel_gpu.warp<{sizePerThread = [32, 64], threadsPerWarp = [1, 1], order = [1, 0]}>
#dot0_ = #triton_gpu.dot_op<{opIdx = 0, parent = #warp}>
#dot1_ = #triton_gpu.dot_op<{opIdx = 1, parent = #warp}>

// COM: Test codegen in match-target-size for SLM path
// SLM-CHECK: module
// SLM-CHECK-SAME: triton_gpu.shared = 4096
module attributes {"triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
  // SLM-CHECK-LABEL: @matmul_with_fixed_a
  tt.func public @matmul_with_fixed_a(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32} loc(unknown), %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32} loc(unknown), %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32} loc(unknown), %arg3: f32 loc(unknown), %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc(unknown), %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc(unknown)) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<32x64xf32, #warp>
    %c65536_i64 = arith.constant 65536 : i64
    %c3145728_i64 = arith.constant 3145728 : i64
    %cst_2 = arith.constant 1.44269502 : f32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id z : i32
    %1 = tt.get_program_id x : i32
    %2 = tt.get_program_id y : i32
    %3 = arith.extsi %1 : i32 to i64
    %4 = arith.muli %3, %c3145728_i64 : i64
    %5 = arith.extsi %2 : i32 to i64
    %6 = arith.muli %5, %c65536_i64 : i64
    %7 = arith.addi %4, %6 : i64
    %8 = tt.addptr %arg0, %7 : !tt.ptr<f16>, i64
    %9 = arith.muli %0, %c128_i32 : i32
    %10 = tt.make_tensor_ptr %8, [%c1024_i64, %c64_i64], [%c64_i64, %c1_i64], [%9, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x64xf16, #dot0_>>
    %13 = tt.addptr %arg1, %7 : !tt.ptr<f16>, i64
    %14 = tt.make_tensor_ptr %13, [%c64_i64, %c1024_i64], [%c1_i64, %c64_i64], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<64x64xf16, #dot1_>>
    %15 = tt.addptr %arg5, %7 : !tt.ptr<f32>, i64
    %16 = tt.make_tensor_ptr %15, [%c1024_i64, %c64_i64], [%c64_i64, %c1_i64], [%9, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x64xf32, #warp>>
    %17 = arith.mulf %arg3, %cst_2 : f32
    %18 = tt.load %10 : !tt.ptr<tensor<32x64xf16, #dot0_>>
    // SLM-CHECK: [[subA1:%.*]] = tt.load {{.*}} {DotIdx = 0 : i32} : !tt.ptr<tensor<32x32xf16>>
    // SLM-CHECK: [[subA2:%.*]] = tt.load {{.*}} {DotIdx = 0 : i32} : !tt.ptr<tensor<32x32xf16>>
    // SLM-CHECK: [[glueA:%.*]] = triton_intel_gpu.glue [[subA1]], [[subA2]] : (tensor<32x32xf16>, tensor<32x32xf16>) -> tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #warp}>>
    // SLM-CHECK: [[extracA1:%.*]] = triton_intel_gpu.extract [[glueA]][0] : tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #warp}>> -> tensor<16x64xf16>
    // SLM-CHECK: tt.store {{.*}}, [[extracA1]] : !tt.ptr<tensor<16x64xf16>, 3>
    // SLM-CHECK: [[extracA2:%.*]] = triton_intel_gpu.extract [[glueA]][1] : tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #warp}>> -> tensor<16x64xf16>
    // SLM-CHECK: tt.store {{.*}}, [[extracA2]] : !tt.ptr<tensor<16x64xf16>, 3>
    %21:3 = scf.for %arg6 = %c0_i32 to %c1024_i32 step %c64_i32 iter_args(%arg8 = %cst_1, %arg10 = %10, %arg11 = %14) -> (tensor<32x64xf32, #warp>, !tt.ptr<tensor<32x64xf16, #dot0_>>, !tt.ptr<tensor<64x64xf16, #dot1_>>)  : i32 {
      // SLM-CHECK: [[loadA1:%.*]] = tt.load {{.*}} {DotIdx = 0 : i32} : !tt.ptr<tensor<16x64xf16>, 3>
      // SLM-CHECK: [[loadA2:%.*]] = tt.load {{.*}} {DotIdx = 0 : i32} : !tt.ptr<tensor<16x64xf16>, 3>
      // SLM-CHECK: [[extractDotA:%.*]] = triton_intel_gpu.extract [[loadA1]][0] : tensor<16x64xf16> -> tensor<8x16xf16>
      // SLM-CHECK: [[dot1:%.*]] = tt.dot [[extractDotA]], {{.*}}, {{.*}}, inputPrecision = tf32 : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %25 = tt.load %arg11 : !tt.ptr<tensor<64x64xf16, #dot1_>>
      %26 = tt.dot %18, %25, %cst_1, inputPrecision = tf32 : tensor<32x64xf16, #dot0_> * tensor<64x64xf16, #dot1_> -> tensor<32x64xf32, #warp>
      %27 = tt.advance %arg10, [%c128_i32, %c0_i32] : <tensor<32x64xf16, #dot0_>>
      %28 = tt.advance %arg11, [%c0_i32, %c64_i32] : <tensor<64x64xf16, #dot1_>>
      scf.yield %26, %27, %28 : tensor<32x64xf32, #warp>, !tt.ptr<tensor<32x64xf16, #dot0_>>, !tt.ptr<tensor<64x64xf16, #dot1_>>
    } {tt.divisibility_arg1 = dense<64> : tensor<1xi32>}
    tt.store %16, %21#0 : !tt.ptr<tensor<32x64xf32, #warp>>
    tt.return
  }
}

