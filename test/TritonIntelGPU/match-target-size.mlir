// RUN: triton-opt %s -split-input-file -tritonintelgpu-match-target-size -canonicalize -cse | FileCheck %s

#warp = #triton_intel_gpu.warp<{sizePerThread = [32, 64], threadsPerWarp = [1, 1], order = [1, 0]}>
#dot0_ = #triton_gpu.dot_op<{opIdx = 0, parent = #warp}>
#dot1_ = #triton_gpu.dot_op<{opIdx = 1, parent = #warp}>
module attributes {"triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 32 : i32, "triton_gpu.threads-per-warp" = 1 : i32} {
  tt.func public @matmul_kernel_with_block_pointers_without_convertlayout(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg4: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg5: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg6: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg7: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg8: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) attributes {noinline = false} {
    %c64_i32 = arith.constant 64 : i32
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c127_i32 = arith.constant 127 : i32
    %c1_i64 = arith.constant 1 : i64
    %c128_i32 = arith.constant 128 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<32x64xf32, #warp>
    %0 = gpu.subgroup_id : index
    %1 = arith.index_cast %0 : index to i32
    %2 = tt.get_program_id x : i32
    %3 = arith.addi %arg3, %c127_i32 : i32
    %4 = arith.divsi %3, %c128_i32 : i32
    %5 = arith.addi %arg4, %c127_i32 : i32
    %6 = arith.divsi %5, %c128_i32 : i32
    %7 = arith.muli %6, %c8_i32 : i32
    %8 = arith.divsi %2, %7 : i32
    %9 = arith.muli %8, %c8_i32 : i32
    %10 = arith.subi %4, %9 : i32
    %11 = arith.minsi %10, %c8_i32 : i32
    %12 = arith.remsi %2, %11 : i32
    %13 = arith.addi %9, %12 : i32
    %14 = arith.remsi %2, %7 : i32
    %15 = arith.divsi %14, %11 : i32
    %16 = arith.muli %13, %c128_i32 : i32
    %17 = arith.extsi %arg3 : i32 to i64
    %18 = arith.extsi %arg5 : i32 to i64
    %19 = arith.extsi %arg6 : i32 to i64
    %20 = arith.divsi %1, %c4_i32 : i32
    %21 = arith.remsi %20, %c8_i32 : i32
    %22 = arith.muli %21, %c32_i32 : i32
    %23 = arith.addi %22, %16 : i32
    %24 = tt.make_tensor_ptr %arg0, [%17, %18], [%19, %c1_i64], [%23, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x32xf16, #dot0_>>
    %25 = arith.muli %15, %c128_i32 : i32
    %26 = arith.extsi %arg4 : i32 to i64
    %27 = arith.extsi %arg7 : i32 to i64
    %28 = arith.remsi %1, %c4_i32 : i32
    %29 = arith.remsi %28, %c4_i32 : i32
    %30 = arith.muli %29, %c64_i32 : i32
    %31 = arith.addi %30, %25 : i32
    %32 = tt.make_tensor_ptr %arg1, [%18, %26], [%27, %c1_i64], [%c0_i32, %31] {order = array<i32: 1, 0>} : <tensor<32x64xf16, #dot1_>>
    %33:3 = scf.for %arg9 = %c0_i32 to %arg5 step %c32_i32 iter_args(%arg10 = %cst, %arg11 = %24, %arg12 = %32) -> (tensor<32x64xf32, #warp>, !tt.ptr<tensor<32x32xf16, #dot0_>>, !tt.ptr<tensor<32x64xf16, #dot1_>>)  : i32 {
      %37 = tt.load %arg11 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<32x32xf16, #dot0_>>
      %38 = tt.load %arg12 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<32x64xf16, #dot1_>>
      %39 = tt.dot %37, %38, %arg10 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<32x32xf16, #dot0_> * tensor<32x64xf16, #dot1_> -> tensor<32x64xf32, #warp>
      // CHECK-LABEL: @matmul_kernel_with_block_pointers_without_convertlayout
      // CHECK: [[A:%.*]] = tt.load {{.*}} : !tt.ptr<tensor<32x32xf16>>
      // CHECK: [[B0:%.*]] = tt.load {{.*}} : !tt.ptr<tensor<32x32xf16>>
      // CHECK: [[B1:%.*]] = tt.load {{.*}} : !tt.ptr<tensor<32x32xf16>>
      // CHECK: [[subA0:%.*]] = triton_intel_gpu.extract [[A]][0] : tensor<32x32xf16> -> tensor<8x16xf16>
      // CHECK: [[subB0:%.*]] = triton_intel_gpu.extract [[B0]][0] : tensor<32x32xf16> -> tensor<16x16xf16>
      // CHECK: [[subC0:%.*]] = tt.dot [[subA0]], [[subB0]], {{.*}} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      // CHECK: [[subA1:%.*]] = triton_intel_gpu.extract [[A]][4] : tensor<32x32xf16> -> tensor<8x16xf16>
      // CHECK: [[subB1:%.*]] = triton_intel_gpu.extract [[B0]][1] : tensor<32x32xf16> -> tensor<16x16xf16>
      // CHECK: [[subC1:%.*]] = tt.dot [[subA1]], [[subB1]], [[subC0]], {{.*}} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %40 = tt.advance %arg11, [%c0_i32, %c32_i32] : <tensor<32x32xf16, #dot0_>>
      %41 = tt.advance %arg12, [%c32_i32, %c0_i32] : <tensor<32x64xf16, #dot1_>>
      scf.yield %39, %40, %41 : tensor<32x64xf32, #warp>, !tt.ptr<tensor<32x32xf16, #dot0_>, 1>, !tt.ptr<tensor<32x64xf16, #dot1_>>
    }
    %34 = arith.truncf %33#0 : tensor<32x64xf32, #warp> to tensor<32x64xf16, #warp>
    %35 = arith.extsi %arg8 : i32 to i64
    %36 = tt.make_tensor_ptr %arg2, [%17, %26], [%35, %c1_i64], [%23, %31] {order = array<i32: 1, 0>} : <tensor<32x64xf16, #warp>>
    tt.store %36, %34 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32} : !tt.ptr<tensor<32x64xf16, #warp>>
    tt.return
  }
}
