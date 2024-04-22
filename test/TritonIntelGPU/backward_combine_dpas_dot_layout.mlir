// RUN: triton-opt %s -split-input-file -tritonintelgpu-remove-layout-conversions 2>&1 | FileCheck %s

// COM: Checks that the loads for the operands of a tt.dot operation are placed
// COM: into registers (have dot layout) rather than shared local memory (via a
// COM: triton_gpu.convert_layout operation).

// CHECK: #{{.*}} = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
// CHECK: #[[DPAS:.+]] = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4], A = [8, 16], B = [16, 16], C = [8, 16]}>
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4], A = [8, 16], B = [16, 16], C = [8, 16]}>
module attributes {"triton_gpu.compute-capability" = 2 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
  tt.func public @matmul_kernel_with_block_pointers(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c8_i32 = arith.constant 8 : i32
    %c64_i32 = arith.constant 64 : i32
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %c32_i32 = arith.constant 32 : i32
    %c255_i32 = arith.constant 255 : i32
    %c63_i32 = arith.constant 63 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x256xf32, #dpas>
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg3, %c63_i32 : i32
    %2 = arith.divsi %1, %c64_i32 : i32
    %3 = arith.addi %arg4, %c255_i32 : i32
    %4 = arith.divsi %3, %c256_i32 : i32
    %5 = arith.muli %4, %c8_i32 : i32
    %6 = arith.divsi %0, %5 : i32
    %7 = arith.muli %6, %c8_i32 : i32
    %8 = arith.subi %2, %7 : i32
    %9 = arith.minsi %8, %c8_i32 : i32
    %10 = arith.remsi %0, %9 : i32
    %11 = arith.addi %7, %10 : i32
    %12 = arith.remsi %0, %5 : i32
    %13 = arith.divsi %12, %9 : i32
    %14 = arith.muli %11, %c64_i32 : i32
    %15 = arith.extsi %arg3 : i32 to i64
    %16 = arith.extsi %arg5 : i32 to i64
    %17 = arith.extsi %arg6 : i32 to i64
    // CHECK: %[[VAL_36:.*]] = tt.make_tensor_ptr %{{.*}}, {{\[}}%{{.*}}, %{{.*}}, {{\[}}%{{.*}}, %{{.*}}], {{\[}}%{{.*}}, %{{.*}}] {order = array<i32: 1, 0>} : <tensor<64x32xf16, #{{.*}}<{opIdx = 0, parent = #[[DPAS]]}>>>
    %18 = tt.make_tensor_ptr %arg0, [%15, %16], [%17, %c1_i64], [%14, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x32xf16, #blocked>>
    %19 = arith.muli %13, %c256_i32 : i32
    %20 = arith.extsi %arg4 : i32 to i64
    %21 = arith.extsi %arg7 : i32 to i64
    // CHECK: %[[VAL_40:.*]] = tt.make_tensor_ptr %{{.*}}, {{\[}}%{{.*}}, %{{.*}}], {{\[}}%{{.*}}, %{{.*}}], {{\[}}%{{.*}}, %{{.*}}] {order = array<i32: 1, 0>} : <tensor<32x256xf16, #{{.*}}<{opIdx = 1, parent = #[[DPAS]]}>>>
    %22 = tt.make_tensor_ptr %arg1, [%16, %20], [%21, %c1_i64], [%c0_i32, %19] {order = array<i32: 1, 0>} : <tensor<32x256xf16, #blocked1>>
    // CHECK: %[[VAL_41:.*]]:3 = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %[[VAL_36]], %{{.*}} = %[[VAL_40]]) -> (tensor<64x256xf32, #[[DPAS]]>, !tt.ptr<tensor<64x32xf16, #{{.*}}<{opIdx = 0, parent = #[[DPAS]]}>>>, !tt.ptr<tensor<32x256xf16, #{{.*}}<{opIdx = 1, parent = #[[DPAS]]}>>>)  : i32 {
    // CHECK: %[[VAL_46:.*]] = tt.load %{{.*}} {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x32xf16, #{{.*}}<{opIdx = 0, parent = #[[DPAS]]}>>>
    // CHECK: %[[VAL_47:.*]] = tt.load %{{.*}} {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<32x256xf16, #{{.*}}<{opIdx = 1, parent = #[[DPAS]]}>>>
    // CHECK-NEXT: %[[VAL_48:.*]] = tt.dot %[[VAL_46]], %[[VAL_47]], %{{.*}}, inputPrecision = tf32 : tensor<64x32xf16, #{{.*}}<{opIdx = 0, parent = #[[DPAS]]}>> * tensor<32x256xf16, #{{.*}}<{opIdx = 1, parent = #[[DPAS]]}>> -> tensor<64x256xf32, #[[DPAS]]>
    // CHECK: %[[VAL_49:.*]] = tt.advance %{{.*}}, {{\[}}%{{.*}}, %{{.*}}] : <tensor<64x32xf16, #{{.*}}<{opIdx = 0, parent = #[[DPAS]]}>>>
    // CHECK: %[[VAL_50:.*]] = tt.advance %{{.*}}, {{\[}}%{{.*}}, %{{.*}}] : <tensor<32x256xf16, #{{.*}}<{opIdx = 1, parent = #[[DPAS]]}>>>
    // CHECK: scf.yield %{{.*}}, %{{.*}}, %{{.*}} : tensor<64x256xf32, #[[DPAS]]>, !tt.ptr<tensor<64x32xf16, #{{.*}}<{opIdx = 0, parent = #[[DPAS]]}>>>, !tt.ptr<tensor<32x256xf16, #{{.*}}<{opIdx = 1, parent = #[[DPAS]]}>>>
    %23:3 = scf.for %arg9 = %c0_i32 to %arg5 step %c32_i32 iter_args(%arg10 = %cst, %arg11 = %18, %arg12 = %22) -> (tensor<64x256xf32, #dpas>, !tt.ptr<tensor<64x32xf16, #blocked>>, !tt.ptr<tensor<32x256xf16, #blocked1>>)  : i32 {
      %28 = tt.load %arg11 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x32xf16, #blocked>>
      %29 = tt.load %arg12 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<32x256xf16, #blocked1>>
      %30 = triton_gpu.convert_layout %28 : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas}>>
      %31 = triton_gpu.convert_layout %29 : tensor<32x256xf16, #blocked1> -> tensor<32x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #dpas}>>
      %32 = tt.dot %30, %31, %arg10, inputPrecision = tf32 : tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas}>> * tensor<32x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #dpas}>> -> tensor<64x256xf32, #dpas>
      %33 = tt.advance %arg11, [%c0_i32, %c32_i32] : <tensor<64x32xf16, #blocked>>
      %34 = tt.advance %arg12, [%c32_i32, %c0_i32] : <tensor<32x256xf16, #blocked1>>
      scf.yield %32, %33, %34 : tensor<64x256xf32, #dpas>, !tt.ptr<tensor<64x32xf16, #blocked>>, !tt.ptr<tensor<32x256xf16, #blocked1>>
    }
    %24 = arith.truncf %23#0 : tensor<64x256xf32, #dpas> to tensor<64x256xf16, #dpas>
    %25 = triton_gpu.convert_layout %24 : tensor<64x256xf16, #dpas> -> tensor<64x256xf16, #blocked1>
    %26 = arith.extsi %arg8 : i32 to i64
    %27 = tt.make_tensor_ptr %arg2, [%15, %20], [%26, %c1_i64], [%14, %19] {order = array<i32: 1, 0>} : <tensor<64x256xf16, #blocked1>>
    tt.store %27, %25 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x256xf16, #blocked1>>
    tt.return
  }
}
