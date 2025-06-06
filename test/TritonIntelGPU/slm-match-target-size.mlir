// RUN: env TRITON_INTEL_ENABLE_FIRST_LOAD_TO_SLM=1 triton-opt %s -tritonintelgpu-match-target-size | FileCheck %s

#warp = #ttig.warp<{sizePerThread = [32, 64], threadsPerWarp = [1, 1], order = [1, 0]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #warp}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #warp}>

// COM: Test codegen in match-target-size for SLM path
// CHECK: module attributes {"ttg.num-warps" = 1 : i32, ttg.shared = 4096 : i32, "ttg.threads-per-warp" = 16 : i32} {
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @matmul_with_fixed_a
  tt.func @matmul_with_fixed_a(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: f32, %arg4: !tt.ptr<f32>, %arg5: !tt.ptr<f32>)  {
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
    %10 = tt.make_tensor_ptr %8, [%c1024_i64, %c64_i64], [%c64_i64, %c1_i64], [%9, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x64xf16, #dot0>>
    %13 = tt.addptr %arg1, %7 : !tt.ptr<f16>, i64
    %14 = tt.make_tensor_ptr %13, [%c64_i64, %c1024_i64], [%c1_i64, %c64_i64], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<64x64xf16, #dot1>>
    %15 = tt.addptr %arg5, %7 : !tt.ptr<f32>, i64
    %16 = tt.make_tensor_ptr %15, [%c1024_i64, %c64_i64], [%c64_i64, %c1_i64], [%9, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x64xf32, #warp>>
    %17 = arith.mulf %arg3, %cst_2 : f32
    %18 = tt.load %10 : !tt.ptr<tensor<32x64xf16, #dot0>>
    // CHECK: [[subA1:%.*]] = tt.load {{.*}} {DotIdx = 0 : i32} : !tt.ptr<tensor<32x32xf16>>
    // CHECK: [[subA2:%.*]] = tt.load {{.*}} {DotIdx = 0 : i32} : !tt.ptr<tensor<32x32xf16>>
    // CHECK: [[glueA:%.*]] = ttig.glue [[subA1]], [[subA2]] : (tensor<32x32xf16>, tensor<32x32xf16>) -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 0, parent = #warp}>>
    // CHECK: [[extracA1:%.*]] = ttig.extract [[glueA]][0] : tensor<32x64xf16, #ttg.dot_op<{opIdx = 0, parent = #warp}>> -> tensor<16x64xf16>
    // CHECK: tt.store {{.*}}, [[extracA1]] : !tt.ptr<tensor<16x64xf16>, 3>
    // CHECK: [[extracA2:%.*]] = ttig.extract [[glueA]][1] : tensor<32x64xf16, #ttg.dot_op<{opIdx = 0, parent = #warp}>> -> tensor<16x64xf16>
    // CHECK: tt.store {{.*}}, [[extracA2]] : !tt.ptr<tensor<16x64xf16>, 3>
    %21:3 = scf.for %arg6 = %c0_i32 to %c1024_i32 step %c64_i32 iter_args(%arg8 = %cst_1, %arg10 = %10, %arg11 = %14) -> (tensor<32x64xf32, #warp>, !tt.ptr<tensor<32x64xf16, #dot0>>, !tt.ptr<tensor<64x64xf16, #dot1>>) : i32 {
      // CHECK: [[loadA1:%.*]] = tt.load {{.*}} {DotIdx = 0 : i32} : !tt.ptr<tensor<16x64xf16>, 3>
      // CHECK: [[loadA2:%.*]] = tt.load {{.*}} {DotIdx = 0 : i32} : !tt.ptr<tensor<16x64xf16>, 3>
      // CHECK: [[extractDotA:%.*]] = ttig.extract [[loadA1]][0] : tensor<16x64xf16> -> tensor<8x16xf16>
      // CHECK: [[dot1:%.*]] = tt.dot [[extractDotA]], {{.*}}, {{.*}}, inputPrecision = tf32 : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %25 = tt.load %arg11 : !tt.ptr<tensor<64x64xf16, #dot1>>
      %26 = tt.dot %18, %25, %cst_1, inputPrecision = tf32 : tensor<32x64xf16, #dot0> * tensor<64x64xf16, #dot1> -> tensor<32x64xf32, #warp>
      %27 = tt.advance %arg10, [%c128_i32, %c0_i32] : <tensor<32x64xf16, #dot0>>
      %28 = tt.advance %arg11, [%c0_i32, %c64_i32] : <tensor<64x64xf16, #dot1>>
      scf.yield %26, %27, %28 : tensor<32x64xf32, #warp>, !tt.ptr<tensor<32x64xf16, #dot0>>, !tt.ptr<tensor<64x64xf16, #dot1>>
    }
    tt.store %16, %21#0 : !tt.ptr<tensor<32x64xf32, #warp>>
    tt.return
  }
}
