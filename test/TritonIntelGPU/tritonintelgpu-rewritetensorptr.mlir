// RUN: triton-opt %s -split-input-file --tritonintelgpu-rewrite-tensor-pointer=device-architecture=PVC | FileCheck %s

// CHECK: #[[$BLOCKED:.+]] = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 16], order = [1, 0]}>
// CHECK: #[[$MMA:.+]] = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [16, 4], A = [8, 16], B = [16, 16], C = [8, 16]}>
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 16], order = [1, 0]}>
#mma = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [16, 4], A = [8, 16], B = [16, 16], C = [8, 16]}>
module attributes {"triton_gpu.compute-capability" = 0 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 64 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
  tt.func public @matmul_kernel_with_block_pointers(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c4_i32 = arith.constant 4 : i32
    %c256_i32 = arith.constant 256 : i32
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c255_i32 = arith.constant 255 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg3, %c255_i32 : i32
    %2 = arith.divsi %1, %c256_i32 : i32
    %3 = arith.addi %arg4, %c255_i32 : i32
    %4 = arith.divsi %3, %c256_i32 : i32
    %5 = arith.muli %4, %c4_i32 : i32
    %6 = arith.divsi %0, %5 : i32
    %7 = arith.muli %6, %c4_i32 : i32
    %8 = arith.subi %2, %7 : i32
    %9 = arith.minsi %8, %c4_i32 : i32
    %10 = arith.remsi %0, %9 : i32
    %11 = arith.addi %7, %10 : i32
    %12 = arith.remsi %0, %5 : i32
    %13 = arith.divsi %12, %9 : i32
    %14 = arith.muli %11, %c256_i32 : i32
    %15 = arith.extsi %arg3 : i32 to i64
    %16 = arith.extsi %arg5 : i32 to i64
    %17 = arith.extsi %arg6 : i32 to i64
    // CHECK:  tt.make_tensor_ptr {{.*}}, {{\[}}{{.*}}, {{.*}}], {{\[}}{{.*}}, {{.*}}], {{\[}}{{.*}}, {{.*}}] {order = array<i32: 1, 0>} : <tensor<256x32xf16, {{.*}}<{opIdx = 0, parent = #[[$MMA]]}>>>
    %18 = tt.make_tensor_ptr %arg0, [%15, %16], [%17, %c1_i64], [%14, %c0_i32] {order = array<i32: 1, 0>} : <tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>>>
    %19 = arith.muli %13, %c256_i32 : i32
    %20 = arith.extsi %arg4 : i32 to i64
    %21 = arith.extsi %arg7 : i32 to i64
    // CHECK:  tt.make_tensor_ptr {{.*}}, {{\[}}{{.*}}, {{.*}}], {{\[}}{{.*}}, {{.*}}], {{\[}}{{.*}}, {{.*}}] {order = array<i32: 1, 0>} : <tensor<32x256xf16, {{.*}}<{opIdx = 1, parent = #[[$MMA]]}>>>
    %22 = tt.make_tensor_ptr %arg1, [%16, %20], [%21, %c1_i64], [%c0_i32, %19] {order = array<i32: 1, 0>} : <tensor<32x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma}>>>
    %23:3 = scf.for %arg9 = %c0_i32 to %arg5 step %c32_i32 iter_args(%arg10 = %cst, %arg11 = %18, %arg12 = %22) -> (tensor<256x256xf32, #mma>, !tt.ptr<tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>>>, !tt.ptr<tensor<32x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma}>>>)  : i32 {
      // CHECK:  tt.load {{.*}} {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<256x32xf16, {{.*}}<{opIdx = 0, parent = #[[$MMA]]}>>>
      // CHECK:  tt.load {{.*}} {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<32x256xf16, {{.*}}<{opIdx = 1, parent = #[[$MMA]]}>>>
      // CHECK:  tt.fp_to_fp {{.*}} : tensor<256x32xf16, {{.*}}<{opIdx = 0, parent = #[[$MMA]]}>> -> tensor<256x32xf32, {{.*}}<{opIdx = 0, parent = #[[$MMA]]}>>
      // CHECK:  tt.fp_to_fp {{.*}} : tensor<32x256xf16, {{.*}}<{opIdx = 1, parent = #[[$MMA]]}>> -> tensor<32x256xf32, {{.*}}<{opIdx = 1, parent = #[[$MMA]]}>>
      // CHECK:  tt.dot {{.*}}, {{.*}}, {{.*}}, inputPrecision = tf32 : tensor<256x32xf32, {{.*}}<{opIdx = 0, parent = #[[$MMA]]}>> * tensor<32x256xf32, {{.*}}<{opIdx = 1, parent = #[[$MMA]]}>> -> tensor<256x256xf32, #[[$MMA]]>
      // CHECK:  tt.advance {{.*}}, {{\[}}{{.*}}, {{.*}}] : <tensor<256x32xf16, {{.*}}<{opIdx = 0, parent = #[[$MMA]]}>>>
      // CHECK:  tt.advance {{.*}}, {{\[}}{{.*}}, {{.*}}] : <tensor<32x256xf16, {{.*}}<{opIdx = 1, parent = #[[$MMA]]}>>>
      %28 = tt.load %arg11 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>>>
      %29 = tt.load %arg12 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<32x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma}>>>
      %30 = tt.fp_to_fp %28 : tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>> -> tensor<256x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>>
      %31 = tt.fp_to_fp %29 : tensor<32x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma}>> -> tensor<32x256xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma}>>
      %32 = tt.dot %30, %31, %arg10, inputPrecision = tf32 : tensor<256x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>> * tensor<32x256xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma}>> -> tensor<256x256xf32, #mma>
      %33 = tt.advance %arg11, [%c0_i32, %c32_i32] : <tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>>>
      %34 = tt.advance %arg12, [%c32_i32, %c0_i32] : <tensor<32x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma}>>>
      scf.yield %32, %33, %34 : tensor<256x256xf32, #mma>, !tt.ptr<tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>>>, !tt.ptr<tensor<32x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma}>>>
    }
    // CHECK:  arith.truncf {{.*}}#0 : tensor<256x256xf32, #[[$MMA]]> to tensor<256x256xf16, #[[$MMA]]>
    // CHECK:  triton_gpu.convert_layout {{.*}} : tensor<256x256xf16, #[[$MMA]]> -> tensor<256x256xf16, #[[$BLOCKED]]>
    // CHECK:  arith.extsi {{.*}} : i32 to i64
    // CHECK:  arith.extsi {{.*}} : i32 to i64
    // CHECK:  arith.extsi {{.*}} : i32 to i64
    // CHECK:  tt.splat {{.*}} : !tt.ptr<f16> -> tensor<256x256x!tt.ptr<f16>, #[[$BLOCKED]]>
    // CHECK:  tt.splat {{.*}} : i64 -> tensor<256xi64, {{.*}}<{dim = 1, parent = #[[$BLOCKED]]}>>
    // CHECK:  tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, {{.*}}<{dim = 1, parent = #[[$BLOCKED]]}>>
    // CHECK:  arith.extsi {{.*}} : tensor<256xi32, {{.*}}<{dim = 1, parent = #[[$BLOCKED]]}>> to tensor<256xi64, {{.*}}<{dim = 1, parent = #[[$BLOCKED]]}>>
    // CHECK:  arith.addi {{.*}}, {{.*}} : tensor<256xi64, {{.*}}<{dim = 1, parent = #[[$BLOCKED]]}>>
    // CHECK:  tt.expand_dims {{.*}} {axis = 1 : i32} : tensor<256xi64, {{.*}}<{dim = 1, parent = #[[$BLOCKED]]}>> -> tensor<256x1xi64, #[[$BLOCKED]]>
    // CHECK:  tt.splat {{.*}} : i64 -> tensor<256x1xi64, #[[$BLOCKED]]>
    // CHECK:  arith.muli {{.*}}, {{.*}} : tensor<256x1xi64, #[[$BLOCKED]]>
    // CHECK:  tt.broadcast {{.*}} : tensor<256x1xi64, #[[$BLOCKED]]> -> tensor<256x256xi64, #[[$BLOCKED]]>
    // CHECK:  tt.addptr {{.*}}, {{.*}} : tensor<256x256x!tt.ptr<f16>, #[[$BLOCKED]]>, tensor<256x256xi64, #[[$BLOCKED]]>
    // CHECK:  tt.splat {{.*}} : i64 -> tensor<256xi64, {{.*}}<{dim = 0, parent = #[[$BLOCKED]]}>>
    // CHECK:  tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, {{.*}}<{dim = 0, parent = #[[$BLOCKED]]}>>
    // CHECK:  arith.extsi {{.*}} : tensor<256xi32, {{.*}}<{dim = 0, parent = #[[$BLOCKED]]}>> to tensor<256xi64, {{.*}}<{dim = 0, parent = #[[$BLOCKED]]}>>
    // CHECK:  arith.addi {{.*}}, {{.*}} : tensor<256xi64, {{.*}}<{dim = 0, parent = #[[$BLOCKED]]}>>
    // CHECK:  tt.expand_dims {{.*}} {axis = 0 : i32} : tensor<256xi64, {{.*}}<{dim = 0, parent = #[[$BLOCKED]]}>> -> tensor<1x256xi64, #[[$BLOCKED]]>
    // CHECK:  tt.splat {{.*}} : i64 -> tensor<1x256xi64, #[[$BLOCKED]]>
    // CHECK:  arith.muli {{.*}}, {{.*}} : tensor<1x256xi64, #[[$BLOCKED]]>
    // CHECK:  tt.broadcast {{.*}} : tensor<1x256xi64, #[[$BLOCKED]]> -> tensor<256x256xi64, #[[$BLOCKED]]>
    // CHECK:  tt.addptr {{.*}}, {{.*}} : tensor<256x256x!tt.ptr<f16>, #[[$BLOCKED]]>, tensor<256x256xi64, #[[$BLOCKED]]>
    // CHECK:  arith.constant 0 : i64
    // CHECK:  tt.splat {{.*}} : i64 -> tensor<256x1xi64, #[[$BLOCKED]]>
    // CHECK:  arith.cmpi sge, {{.*}}, {{.*}} : tensor<256x1xi64, #[[$BLOCKED]]>
    // CHECK:  tt.splat {{.*}} : i64 -> tensor<256x1xi64, #[[$BLOCKED]]>
    // CHECK:  arith.cmpi slt, {{.*}}, {{.*}} : tensor<256x1xi64, #[[$BLOCKED]]>
    // CHECK:  arith.andi {{.*}}, {{.*}} : tensor<256x1xi1, #[[$BLOCKED]]>
    // CHECK:  tt.broadcast {{.*}} : tensor<256x1xi1, #[[$BLOCKED]]> -> tensor<256x256xi1, #[[$BLOCKED]]>
    // CHECK:  arith.constant 0 : i64
    // CHECK:  tt.splat {{.*}} : i64 -> tensor<1x256xi64, #[[$BLOCKED]]>
    // CHECK:  arith.cmpi sge, {{.*}}, {{.*}} : tensor<1x256xi64, #[[$BLOCKED]]>
    // CHECK:  tt.splat {{.*}} : i64 -> tensor<1x256xi64, #[[$BLOCKED]]>
    // CHECK:  arith.cmpi slt, {{.*}}, {{.*}} : tensor<1x256xi64, #[[$BLOCKED]]>
    // CHECK:  arith.andi {{.*}}, {{.*}} : tensor<1x256xi1, #[[$BLOCKED]]>
    // CHECK:  tt.broadcast {{.*}} : tensor<1x256xi1, #[[$BLOCKED]]> -> tensor<256x256xi1, #[[$BLOCKED]]>
    // CHECK:  arith.andi {{.*}}, {{.*}} : tensor<256x256xi1, #[[$BLOCKED]]>
    // CHECK:  tt.store {{.*}}, {{.*}}, {{.*}} : tensor<256x256x!tt.ptr<f16>, #[[$BLOCKED]]>
    %24 = arith.truncf %23#0 : tensor<256x256xf32, #mma> to tensor<256x256xf16, #mma>
    %25 = triton_gpu.convert_layout %24 : tensor<256x256xf16, #mma> -> tensor<256x256xf16, #blocked>
    %26 = arith.extsi %arg8 : i32 to i64
    %27 = tt.make_tensor_ptr %arg2, [%15, %20], [%26, %c1_i64], [%14, %19] {order = array<i32: 1, 0>} : <tensor<256x256xf16, #blocked>>
    tt.store %27, %25 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<256x256xf16, #blocked>>
    tt.return
  }
}
