// RUN: triton-opt %s -split-input-file -tritonintelgpu-rewrite-tensor-pointer | FileCheck %s

// COM: Case 0:
// COM: Check that operations using block pointers satisfying the following conditions are not rewritten:
// COM: - the block pointer has the "dot" layout attribute (with dpas parent layout) or has a dpas layout (for store op)
// COM: - the block pointers is advanced in row major order: strides[1] == 1
// COM: - the block pointer pitch is divisible by QW: strides[0] % (64 / elemTypeBitWidth) == 0
// CHECK: #[[DPAS:.+]] = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [16, 4], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [16, 4], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #triton_gpu.dot_op<{opIdx = 0, parent = #dpas, kWidth=2}>
#dot1 = #triton_gpu.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {"triton_gpu.num-warps" = 64 : i32, "triton_gpu.threads-per-warp" = 16 : i32, "triton_intel_gpu.support_sg_2d_block"} {
  tt.func public @matmul_kernel_with_block_pointers(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) {
    // CHECK:  @matmul_kernel_with_block_pointers
    %c4_i32 = arith.constant 4 : i32
    %c256_i32 = arith.constant 256 : i32
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c255_i32 = arith.constant 255 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #dpas>
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
    // CHECK: tt.make_tensor_ptr {{.*}}, {{\[}}{{.*}}, {{.*}}], {{\[}}{{.*}}, {{.*}}], {{\[}}{{.*}}, {{.*}}] {order = array<i32: 1, 0>} : <tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #[[DPAS]], kWidth = 2}>>>
    %18 = tt.make_tensor_ptr %arg0, [%15, %16], [%17, %c1_i64], [%14, %c0_i32] {order = array<i32: 1, 0>} : <tensor<256x32xf16, #dot0>>
    %19 = arith.muli %13, %c256_i32 : i32
    %20 = arith.extsi %arg4 : i32 to i64
    %21 = arith.extsi %arg7 : i32 to i64
    // CHECK: tt.make_tensor_ptr {{.*}}, {{\[}}{{.*}}, {{.*}}], {{\[}}{{.*}}, {{.*}}], {{\[}}{{.*}}, {{.*}}] {order = array<i32: 1, 0>} : <tensor<32x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[DPAS]], kWidth = 2}>>>
    %22 = tt.make_tensor_ptr %arg1, [%16, %20], [%21, %c1_i64], [%c0_i32, %19] {order = array<i32: 1, 0>} : <tensor<32x256xf16, #dot1>>
    %23:3 = scf.for %arg9 = %c0_i32 to %arg5 step %c32_i32 iter_args(%arg10 = %cst, %arg11 = %18, %arg12 = %22) -> (tensor<256x256xf32, #dpas>, !tt.ptr<tensor<256x32xf16, #dot0>>, !tt.ptr<tensor<32x256xf16, #dot1>>)  : i32 {
      // CHECK: tt.load {{.*}} {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #[[DPAS]], kWidth = 2}>>>
      // CHECK: tt.load {{.*}} {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<32x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[DPAS]], kWidth = 2}>>>
      %28 = tt.load %arg11 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<256x32xf16, #dot0>>
      %29 = tt.load %arg12 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<32x256xf16, #dot1>>
      // CHECK:  tt.dot {{.*}}, {{.*}}, {{.*}}, inputPrecision = tf32 : tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #[[DPAS]], kWidth = 2}>> * tensor<32x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[DPAS]], kWidth = 2}>> -> tensor<256x256xf32, #[[DPAS]]>
      // CHECK:  tt.advance {{.*}}, {{\[}}{{.*}}, {{.*}}] : <tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #[[DPAS]], kWidth = 2}>>>
      // CHECK:  tt.advance {{.*}}, {{\[}}{{.*}}, {{.*}}] : <tensor<32x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[DPAS]], kWidth = 2}>>>
      %30 = tt.dot %28, %29, %arg10, inputPrecision = tf32 : tensor<256x32xf16, #dot0> * tensor<32x256xf16, #dot1> -> tensor<256x256xf32, #dpas>
      %31 = tt.advance %arg11, [%c0_i32, %c32_i32] : <tensor<256x32xf16, #dot0>>
      %32 = tt.advance %arg12, [%c32_i32, %c0_i32] : <tensor<32x256xf16, #dot1>>
      scf.yield %30, %31, %32 : tensor<256x256xf32, #dpas>, !tt.ptr<tensor<256x32xf16, #dot0>>, !tt.ptr<tensor<32x256xf16, #dot1>>
    }
    %24 = arith.truncf %23#0 : tensor<256x256xf32, #dpas> to tensor<256x256xf16, #dpas>
    %26 = arith.extsi %arg8 : i32 to i64
    // CHECK: tt.make_tensor_ptr {{.*}}, {{\[}}{{.*}}, {{.*}}], {{\[}}{{.*}}, {{.*}}], {{\[}}{{.*}}, {{.*}}] {order = array<i32: 1, 0>} : <tensor<256x256xf16, #[[DPAS]]>>
    %27 = tt.make_tensor_ptr %arg2, [%15, %20], [%26, %c1_i64], [%14, %19] {order = array<i32: 1, 0>} : <tensor<256x256xf16, #dpas>>
    // CHECK: tt.store {{.*}}, {{.*}} {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<256x256xf16, #[[DPAS]]>>
    tt.store %27, %24 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<256x256xf16, #dpas>>
    tt.return
  }
}

// -----

// COM: Case 1:
// COM: Check that operations using block pointers satisfying the following conditions are not rewritten:
// COM: - the block pointer has the "dot" layout attribute (with dpas parent layout)
// COM: - the block pointers is advanced in row major order: strides[order[0]] == 1
// COM: - the block pointer pitch is divisible by QW: strides[order[1]] % (64 / elemTypeBitWidth) == 0
// COM: Check that store operations using block pointers with non Dpas layout is rewritten
// CHECK: #[[BLOCKED:.+]] = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 16], order = [1, 0]}>
// CHECK: #[[DPAS:.+]] = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [16, 4], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 16], order = [1, 0]}>
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [16, 4], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #triton_gpu.dot_op<{opIdx = 0, parent = #dpas, kWidth=2}>
#dot1 = #triton_gpu.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {"triton_gpu.num-warps" = 64 : i32, "triton_gpu.threads-per-warp" = 16 : i32, "triton_intel_gpu.support_sg_2d_block"} {
  tt.func public @matmul_kernel_with_block_pointers(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                                    %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32},
                                                    %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32},
                                                    %arg8: i32 {tt.divisibility = 16 : i32}) {
    // CHECK:  @matmul_kernel_with_block_pointers
    %c4_i32 = arith.constant 4 : i32
    %c256_i32 = arith.constant 256 : i32
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c255_i32 = arith.constant 255 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #dpas>
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
    // CHECK: tt.make_tensor_ptr {{.*}}, {{\[}}{{.*}}, {{.*}}], {{\[}}{{.*}}, {{.*}}], {{\[}}{{.*}}, {{.*}}] {order = array<i32: 1, 0>} : <tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #[[DPAS]], kWidth = 2}>>>
    %18 = tt.make_tensor_ptr %arg0, [%15, %16], [%17, %c1_i64], [%14, %c0_i32] {order = array<i32: 1, 0>} : <tensor<256x32xf16, #dot0>>
    %19 = arith.muli %13, %c256_i32 : i32
    %20 = arith.extsi %arg4 : i32 to i64
    %21 = arith.extsi %arg7 : i32 to i64
    // CHECK: tt.make_tensor_ptr {{.*}}, {{\[}}{{.*}}, {{.*}}], {{\[}}{{.*}}, {{.*}}], {{\[}}{{.*}}, {{.*}}] {order = array<i32: 1, 0>} : <tensor<32x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[DPAS]], kWidth = 2}>>>
    %22 = tt.make_tensor_ptr %arg1, [%16, %20], [%21, %c1_i64], [%c0_i32, %19] {order = array<i32: 1, 0>} : <tensor<32x256xf16, #dot1>>
    %23:3 = scf.for %arg9 = %c0_i32 to %arg5 step %c32_i32 iter_args(%arg10 = %cst, %arg11 = %18, %arg12 = %22) -> (tensor<256x256xf32, #dpas>, !tt.ptr<tensor<256x32xf16, #dot0>>, !tt.ptr<tensor<32x256xf16, #dot1>>) : i32 {
      // CHECK: tt.load {{.*}} {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #[[DPAS]], kWidth = 2}>>>
      // CHECK: tt.load {{.*}} {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<32x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[DPAS]], kWidth = 2}>>>
      %28 = tt.load %arg11 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<256x32xf16, #dot0>>
      %29 = tt.load %arg12 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<32x256xf16, #dot1>>
      // CHECK:  tt.dot {{.*}}, {{.*}}, {{.*}}, inputPrecision = tf32 : tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #[[DPAS]], kWidth = 2}>> * tensor<32x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[DPAS]], kWidth = 2}>> -> tensor<256x256xf32, #[[DPAS]]>
      // CHECK:  tt.advance {{.*}}, {{\[}}{{.*}}, {{.*}}] : <tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #[[DPAS]], kWidth = 2}>>>
      // CHECK:  tt.advance {{.*}}, {{\[}}{{.*}}, {{.*}}] : <tensor<32x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[DPAS]], kWidth = 2}>>>
      %30 = tt.dot %28, %29, %arg10, inputPrecision = tf32 : tensor<256x32xf16, #dot0> * tensor<32x256xf16, #dot1> -> tensor<256x256xf32, #dpas>
      %31 = tt.advance %arg11, [%c0_i32, %c32_i32] : <tensor<256x32xf16, #dot0>>
      %32 = tt.advance %arg12, [%c32_i32, %c0_i32] : <tensor<32x256xf16, #dot1>>
      scf.yield %30, %31, %32 : tensor<256x256xf32, #dpas>, !tt.ptr<tensor<256x32xf16, #dot0>>, !tt.ptr<tensor<32x256xf16, #dot1>>
    }
    %24 = arith.truncf %23#0 : tensor<256x256xf32, #dpas> to tensor<256x256xf16, #dpas>
    %25 = triton_gpu.convert_layout %24 : tensor<256x256xf16, #dpas> -> tensor<256x256xf16, #blocked>
    %26 = arith.extsi %arg8 : i32 to i64
    // CHECK-NOT: tt.make_tensor_ptr
    %27 = tt.make_tensor_ptr %arg2, [%15, %20], [%26, %c1_i64], [%14, %19] {order = array<i32: 1, 0>} : <tensor<256x256xf16, #blocked>>
    // CHECK: tt.store {{.*}}, {{.*}}, {{.*}} : tensor<256x256x!tt.ptr<f16>, #[[BLOCKED]]>
    tt.store %27, %25 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<256x256xf16, #blocked>>
    tt.return
  }
}

// -----

// COM: Case 2:
// COM: Check that operations using block pointers without divisibility attribute are rewritten to use a legacy pointer.
// CHECK: #[[DPAS:.+]] = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [16, 4], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 16], order = [1, 0]}>
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [16, 4], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #triton_gpu.dot_op<{opIdx = 0, parent = #dpas, kWidth=2}>
#dot1 = #triton_gpu.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {"triton_gpu.num-warps" = 64 : i32, "triton_gpu.threads-per-warp" = 16 : i32, "triton_intel_gpu.support_sg_2d_block"} {
  tt.func public @matmul_kernel_with_block_pointers_indivisible(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32 {tt.divisibility = 16 : i32}) {
    // CHECK:  @matmul_kernel_with_block_pointers_indivisible
    %c4_i32 = arith.constant 4 : i32
    %c256_i32 = arith.constant 256 : i32
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c255_i32 = arith.constant 255 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #dpas>
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
    // CHECK-NOT: tt.make_tensor_ptr
    %18 = tt.make_tensor_ptr %arg0, [%15, %16], [%17, %c1_i64], [%14, %c0_i32] {order = array<i32: 1, 0>} : <tensor<256x32xf16, #dot0>>
    %19 = arith.muli %13, %c256_i32 : i32
    %20 = arith.extsi %arg4 : i32 to i64
    %21 = arith.extsi %arg7 : i32 to i64
    // CHECK-NOT: tt.make_tensor_ptr
    %22 = tt.make_tensor_ptr %arg1, [%16, %20], [%21, %c1_i64], [%c0_i32, %19] {order = array<i32: 1, 0>} : <tensor<32x256xf16, #dot1>>
    %23:3 = scf.for %arg9 = %c0_i32 to %arg5 step %c32_i32 iter_args(%arg10 = %cst, %arg11 = %18, %arg12 = %22) -> (tensor<256x256xf32, #dpas>, !tt.ptr<tensor<256x32xf16, #dot0>>, !tt.ptr<tensor<32x256xf16, #dot1>>) : i32 {
      // CHECK: tt.load {{.*}}, {{.*}} : tensor<256x32x!tt.ptr<f16>, #triton_gpu.dot_op<{opIdx = 0, parent = #[[DPAS]], kWidth = 2}>>
      // CHECK: tt.load {{.*}}, {{.*}} : tensor<32x256x!tt.ptr<f16>, #triton_gpu.dot_op<{opIdx = 1, parent = #[[DPAS]], kWidth = 2}>>
      %28 = tt.load %arg11 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<256x32xf16, #dot0>>
      %29 = tt.load %arg12 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<32x256xf16, #dot1>>
      %30 = tt.dot %28, %29, %arg10, inputPrecision = tf32 : tensor<256x32xf16, #dot0> * tensor<32x256xf16, #dot1> -> tensor<256x256xf32, #dpas>
      // CHECK-NOT: tt.advance
      %31 = tt.advance %arg11, [%c0_i32, %c32_i32] : <tensor<256x32xf16, #dot0>>
      // CHECK-NOT: tt.advance
      %32 = tt.advance %arg12, [%c32_i32, %c0_i32] : <tensor<32x256xf16, #dot1>>
      scf.yield %30, %31, %32 : tensor<256x256xf32, #dpas>, !tt.ptr<tensor<256x32xf16, #dot0>>, !tt.ptr<tensor<32x256xf16, #dot1>>
    }
    %24 = arith.truncf %23#0 : tensor<256x256xf32, #dpas> to tensor<256x256xf16, #dpas>
    %25 = triton_gpu.convert_layout %24 : tensor<256x256xf16, #dpas> -> tensor<256x256xf16, #blocked>
    %26 = arith.extsi %arg8 : i32 to i64
    // CHECK-NOT: tt.make_tensor_ptr
    %27 = tt.make_tensor_ptr %arg2, [%15, %20], [%26, %c1_i64], [%14, %19] {order = array<i32: 1, 0>} : <tensor<256x256xf16, #blocked>>
    // CHECK: tt.store {{.*}}, {{.*}}, {{.*}} : tensor<256x256x!tt.ptr<f16>, #[[BLOCKED]]>
    tt.store %27, %25 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<256x256xf16, #blocked>>
    tt.return
  }
}

// -----

// COM: Case 3:
// COM: Check that operations using block pointers without a layout attribute are rewritten to use a legacy pointer.
module attributes {"triton_intel_gpu.support_sg_2d_block"} {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
    %c31_i32 = arith.constant 31 : i32
    %c127_i32 = arith.constant 127 : i32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0.000000e+00> : tensor<128x32xf32>
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c32_i32 = arith.constant 32 : i32
    %c128_i32 = arith.constant 128 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.addi %arg3, %c127_i32 : i32
    %3 = arith.divsi %2, %c128_i32 : i32
    %4 = arith.addi %arg4, %c31_i32 : i32
    %5 = arith.divsi %4, %c32_i32 : i32
    %6 = arith.muli %5, %c8_i32 : i32
    %7 = arith.divsi %0, %6 : i32
    %8 = arith.muli %7, %c8_i32 : i32
    %9 = arith.subi %3, %8 : i32
    %10 = arith.cmpi slt, %9, %c8_i32 : i32
    %11 = arith.select %10, %9, %c8_i32 : i32
    %12 = arith.remsi %0, %11 : i32
    %13 = arith.addi %8, %12 : i32
    %14 = arith.remsi %0, %6 : i32
    %15 = arith.divsi %14, %11 : i32
    %16 = arith.muli %13, %c128_i32 : i32
    %17 = arith.muli %1, %c32_i32 : i32
    %18 = arith.extsi %arg3 : i32 to i64
    %19 = arith.extsi %arg5 : i32 to i64
    %20 = arith.extsi %arg6 : i32 to i64
    // CHECK-NOT: tt.make_tensor_ptr
    %21 = tt.make_tensor_ptr %arg0, [%18, %19], [%20, %c1_i64], [%16, %17] {order = array<i32: 1, 0>} : !tt.ptr<tensor<128x32xf16>>
    %22 = arith.muli %15, %c32_i32 : i32
    %23 = arith.extsi %arg4 : i32 to i64
    %24 = arith.extsi %arg7 : i32 to i64
    // CHECK-NOT: tt.make_tensor_ptr
    %25 = tt.make_tensor_ptr %arg1, [%19, %23], [%24, %c1_i64], [%17, %22] {order = array<i32: 1, 0>} : !tt.ptr<tensor<32x32xf16>>
    %26 = arith.addi %arg5, %c31_i32 : i32
    %27 = arith.divsi %26, %c32_i32 : i32
    %28 = arith.index_cast %27 : i32 to index
    // CHECK: scf.for
    %29:3 = scf.for %arg9 = %c0 to %28 step %c1 iter_args(%arg10 = %cst, %arg11 = %21, %arg12 = %25) -> (tensor<128x32xf32>, !tt.ptr<tensor<128x32xf16>>, !tt.ptr<tensor<32x32xf16>>) {
      // CHECK: tt.load %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}} : tensor<128x32x!tt.ptr<f16>>
      %55 = tt.load %arg11 {boundaryCheck = array<i32: 1>, padding = 2 : i32} : !tt.ptr<tensor<128x32xf16>>
      // CHECK: tt.load %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}} : tensor<32x32x!tt.ptr<f16>>
      %56 = tt.load %arg12 {boundaryCheck = array<i32: 0>, padding = 2 : i32} : !tt.ptr<tensor<32x32xf16>>
      %57 = tt.dot %55, %56, %arg10 : tensor<128x32xf16> * tensor<32x32xf16> -> tensor<128x32xf32>
      // CHECK-NOT: tt.advance
      %58 = tt.advance %arg11, [%c0_i32, %c32_i32] : !tt.ptr<tensor<128x32xf16>>
      // CHECK-NOT: tt.advance
      %59 = tt.advance %arg12, [%c32_i32, %c0_i32] : !tt.ptr<tensor<32x32xf16>>
      // CHECK: scf.yield
      scf.yield %57, %58, %59 : tensor<128x32xf32>, !tt.ptr<tensor<128x32xf16>>, !tt.ptr<tensor<32x32xf16>>
    }
    tt.return
  }
}
