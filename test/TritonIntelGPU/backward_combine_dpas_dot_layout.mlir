// RUN: triton-opt %s -split-input-file -tritonintelgpu-remove-layout-conversions 2>&1 | FileCheck %s

// COM: Case 1:
// COM: Checks that the loads for the operands of a tt.dot operation are placed
// COM: into registers (have dot layout) rather than shared local memory (via a
// COM: ttg.convert_layout operation).

// CHECK: #[[DPAS:.+]] = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_sg_2d_block"} {
  tt.func public @matmul_kernel_with_block_pointers(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i64, %arg7: i32, %arg8: i64) {
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
    // CHECK: %[[VAL_36:.*]] = tt.make_tensor_ptr %{{.*}}, {{\[}}%{{.*}}, %{{.*}}, {{\[}}%{{.*}}, %{{.*}}], {{\[}}%{{.*}}, %{{.*}}] {order = array<i32: 1, 0>} : <tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #[[DPAS]], kWidth = 1}>>>
    %18 = tt.make_tensor_ptr %arg0, [%15, %16], [%arg6, %c1_i64], [%14, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x32xf16, #blocked>>
    %19 = arith.muli %13, %c256_i32 : i32
    %20 = arith.extsi %arg4 : i32 to i64
    %21 = arith.extsi %arg7 : i32 to i64
    // CHECK: %[[VAL_40:.*]] = tt.make_tensor_ptr %{{.*}}, {{\[}}%{{.*}}, %{{.*}}], {{\[}}%{{.*}}, %{{.*}}], {{\[}}%{{.*}}, %{{.*}}] {order = array<i32: 1, 0>} : <tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #[[DPAS]], kWidth = 2}>>>
    %22 = tt.make_tensor_ptr %arg1, [%16, %20], [%21, %c1_i64], [%c0_i32, %19] {order = array<i32: 1, 0>} : <tensor<32x256xf16, #blocked1>>
    // CHECK: %[[VAL_41:.*]]:3 = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %[[VAL_36]], %{{.*}} = %[[VAL_40]]) -> (tensor<64x256xf32, #[[DPAS]]>, !tt.ptr<tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #[[DPAS]], kWidth = 1}>>>, !tt.ptr<tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #[[DPAS]], kWidth = 2}>>>)  : i32 {
    // CHECK: %[[VAL_46:.*]] = tt.load %{{.*}} {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #[[DPAS]], kWidth = 1}>>>
    // CHECK: %[[VAL_47:.*]] = tt.load %{{.*}} {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #[[DPAS]], kWidth = 2}>>>
    // CHECK-NOT: ttg.convert_layout
    // CHECK-NEXT: %[[VAL_48:.*]] = tt.dot %[[VAL_46]], %[[VAL_47]], %{{.*}}, inputPrecision = tf32 : tensor<64x32xf16, #{{.*}}<{opIdx = 0, parent = #[[DPAS]], kWidth = 1}>> * tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #[[DPAS]], kWidth = 2}>> -> tensor<64x256xf32, #[[DPAS]]>
    // CHECK: %[[VAL_49:.*]] = tt.advance %{{.*}}, {{\[}}%{{.*}}, %{{.*}}] : <tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #[[DPAS]], kWidth = 1}>>>
    // CHECK: %[[VAL_50:.*]] = tt.advance %{{.*}}, {{\[}}%{{.*}}, %{{.*}}] : <tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #[[DPAS]], kWidth = 2}>>>
    // CHECK: scf.yield %{{.*}}, %{{.*}}, %{{.*}} : tensor<64x256xf32, #[[DPAS]]>, !tt.ptr<tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #[[DPAS]], kWidth = 1}>>>, !tt.ptr<tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #[[DPAS]], kWidth = 2}>>>
    %23:3 = scf.for %arg9 = %c0_i32 to %arg5 step %c32_i32 iter_args(%arg10 = %cst, %arg11 = %18, %arg12 = %22) -> (tensor<64x256xf32, #dpas>, !tt.ptr<tensor<64x32xf16, #blocked>>, !tt.ptr<tensor<32x256xf16, #blocked1>>)  : i32 {
      %28 = tt.load %arg11 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<64x32xf16, #blocked>>
      %29 = tt.load %arg12 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<32x256xf16, #blocked1>>
      %30 = ttg.convert_layout %28 : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #dot0>
      %31 = ttg.convert_layout %29 : tensor<32x256xf16, #blocked1> -> tensor<32x256xf16, #dot1>
      %32 = tt.dot %30, %31, %arg10, inputPrecision = tf32 : tensor<64x32xf16, #dot0> * tensor<32x256xf16, #dot1> -> tensor<64x256xf32, #dpas>
      %33 = tt.advance %arg11, [%c0_i32, %c32_i32] : <tensor<64x32xf16, #blocked>>
      %34 = tt.advance %arg12, [%c32_i32, %c0_i32] : <tensor<32x256xf16, #blocked1>>
      scf.yield %32, %33, %34 : tensor<64x256xf32, #dpas>, !tt.ptr<tensor<64x32xf16, #blocked>>, !tt.ptr<tensor<32x256xf16, #blocked1>>
    }
    %24 = arith.truncf %23#0 : tensor<64x256xf32, #dpas> to tensor<64x256xf16, #dpas>
    %25 = ttg.convert_layout %24 : tensor<64x256xf16, #dpas> -> tensor<64x256xf16, #blocked1>
    %27 = tt.make_tensor_ptr %arg2, [%15, %20], [%arg8, %c1_i64], [%14, %19] {order = array<i32: 1, 0>} : <tensor<64x256xf16, #blocked1>>
    tt.store %27, %25 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x256xf16, #blocked1>>
    tt.return
  }
}


// -----

// COM: Case 2:
// COM: Checks that DPAS encoding has been forwarded to the store op
// COM: and the ttg.convert_layout operation has been removed
// CHECK: #[[DPAS:.+]] = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_sg_2d_block"} {
  tt.func public @matmul_kernel_with_block_pointers(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
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
    %18 = tt.make_tensor_ptr %arg0, [%15, %16], [%17, %c1_i64], [%14, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x32xf16, #blocked>>
    %19 = arith.muli %13, %c256_i32 : i32
    %20 = arith.extsi %arg4 : i32 to i64
    %21 = arith.extsi %arg7 : i32 to i64
    %22 = tt.make_tensor_ptr %arg1, [%16, %20], [%21, %c1_i64], [%c0_i32, %19] {order = array<i32: 1, 0>} : <tensor<32x256xf16, #blocked1>>
    %23:3 = scf.for %arg9 = %c0_i32 to %arg5 step %c32_i32 iter_args(%arg10 = %cst, %arg11 = %18, %arg12 = %22) -> (tensor<64x256xf32, #dpas>, !tt.ptr<tensor<64x32xf16, #blocked>>, !tt.ptr<tensor<32x256xf16, #blocked1>>)  : i32 {
      %28 = tt.load %arg11 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x32xf16, #blocked>>
      %29 = tt.load %arg12 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<32x256xf16, #blocked1>>
      %30 = ttg.convert_layout %28 : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #dot0>
      %31 = ttg.convert_layout %29 : tensor<32x256xf16, #blocked1> -> tensor<32x256xf16, #dot1>
      %32 = tt.dot %30, %31, %arg10, inputPrecision = tf32 : tensor<64x32xf16, #dot0> * tensor<32x256xf16, #dot1> -> tensor<64x256xf32, #dpas>
      %33 = tt.advance %arg11, [%c0_i32, %c32_i32] : <tensor<64x32xf16, #blocked>>
      %34 = tt.advance %arg12, [%c32_i32, %c0_i32] : <tensor<32x256xf16, #blocked1>>
      scf.yield %32, %33, %34 : tensor<64x256xf32, #dpas>, !tt.ptr<tensor<64x32xf16, #blocked>>, !tt.ptr<tensor<32x256xf16, #blocked1>>
    }
    %24 = arith.truncf %23#0 : tensor<64x256xf32, #dpas> to tensor<64x256xf16, #dpas>
    %25 = ttg.convert_layout %24 : tensor<64x256xf16, #dpas> -> tensor<64x256xf16, #blocked1>
    %26 = arith.extsi %arg8 : i32 to i64
    // CHECK: tt.make_tensor_ptr {{.*}}, {{\[}}{{.*}}, {{.*}}], {{\[}}{{.*}}, {{.*}}], {{\[}}{{.*}}, {{.*}}] {order = array<i32: 1, 0>} : <tensor<64x256xf16, #[[DPAS]]>>
    %27 = tt.make_tensor_ptr %arg2, [%15, %20], [%26, %c1_i64], [%14, %19] {order = array<i32: 1, 0>} : <tensor<64x256xf16, #blocked1>>
    // CHECK: tt.store {{.*}}, {{.*}} {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x256xf16, #[[DPAS]]>>
    tt.store %27, %25 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x256xf16, #blocked1>>
    tt.return
  }
}

// -----

// COM: Case 3:
// COM: Checks that DPAS encoding has been forwarded to the store op
// COM: The `tt.make_tensor_ptr` has multiple users (the storeOp + another OP)
// COM: The initial `tt.make_tensor_ptr` with non-DPAS encoding must be kept.
// CHECK: #[[BLOCKED:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
// CHECK: #[[DPAS:.+]] = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_sg_2d_block"} {
  tt.func public @matmul_kernel_with_block_pointers(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg13: !tt.ptr<f16>, %arg14: !tt.ptr<f32>) {
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
    %18 = tt.make_tensor_ptr %arg0, [%15, %16], [%17, %c1_i64], [%14, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x32xf16, #blocked>>
    %19 = arith.muli %13, %c256_i32 : i32
    %20 = arith.extsi %arg4 : i32 to i64
    %21 = arith.extsi %arg7 : i32 to i64
    %22 = tt.make_tensor_ptr %arg1, [%16, %20], [%21, %c1_i64], [%c0_i32, %19] {order = array<i32: 1, 0>} : <tensor<32x256xf16, #blocked1>>
    %23:3 = scf.for %arg9 = %c0_i32 to %arg5 step %c32_i32 iter_args(%arg10 = %cst, %arg11 = %18, %arg12 = %22) -> (tensor<64x256xf32, #dpas>, !tt.ptr<tensor<64x32xf16, #blocked>>, !tt.ptr<tensor<32x256xf16, #blocked1>>)  : i32 {
      %28 = tt.load %arg11 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<64x32xf16, #blocked>>
      %29 = tt.load %arg12 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<32x256xf16, #blocked1>>
      %30 = ttg.convert_layout %28 : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #dot0>
      %31 = ttg.convert_layout %29 : tensor<32x256xf16, #blocked1> -> tensor<32x256xf16, #dot1>
      %32 = tt.dot %30, %31, %arg10, inputPrecision = tf32 : tensor<64x32xf16, #dot0> * tensor<32x256xf16, #dot1> -> tensor<64x256xf32, #dpas>
      %33 = tt.advance %arg11, [%c0_i32, %c32_i32] : <tensor<64x32xf16, #blocked>>
      %34 = tt.advance %arg12, [%c32_i32, %c0_i32] : <tensor<32x256xf16, #blocked1>>
      scf.yield %32, %33, %34 : tensor<64x256xf32, #dpas>, !tt.ptr<tensor<64x32xf16, #blocked>>, !tt.ptr<tensor<32x256xf16, #blocked1>>
    }
    %24 = arith.truncf %23#0 : tensor<64x256xf32, #dpas> to tensor<64x256xf16, #dpas>
    %25 = ttg.convert_layout %24 : tensor<64x256xf16, #dpas> -> tensor<64x256xf16, #blocked1>
    %26 = arith.extsi %arg8 : i32 to i64
    // CHECK: tt.make_tensor_ptr {{.*}}, {{\[}}{{.*}}, {{.*}}], {{\[}}{{.*}}, {{.*}}], {{\[}}{{.*}}, {{.*}}] {order = array<i32: 1, 0>} : <tensor<64x256xf16, #[[DPAS]]>>
    // CHECK: tt.make_tensor_ptr {{.*}}, {{\[}}{{.*}}, {{.*}}], {{\[}}{{.*}}, {{.*}}], {{\[}}{{.*}}, {{.*}}] {order = array<i32: 1, 0>} : <tensor<64x256xf16, #[[BLOCKED]]>>
    %27 = tt.make_tensor_ptr %arg2, [%15, %20], [%26, %c1_i64], [%14, %19] {order = array<i32: 1, 0>} : <tensor<64x256xf16, #blocked1>>
    // CHECK: tt.store {{.*}}, {{.*}} {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x256xf16, #[[DPAS]]>>
    tt.store %27, %25 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x256xf16, #blocked1>>
    %35 = tt.load %27 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x256xf16, #blocked1>>
    %36 = tt.make_tensor_ptr %arg13, [%15, %16], [%17, %c1_i64], [%14, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x64xf16, #blocked>>
    %37 = tt.load %36 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x64xf16, #blocked>>
    %38 = ttg.convert_layout %37 : tensor<64x64xf16, #blocked> -> tensor<64x64xf16, #dot0>
    %39 = ttg.convert_layout %35 : tensor<64x256xf16, #blocked1> -> tensor<64x256xf16, #dot1>
    %40 = tt.dot %38, %39, %cst, inputPrecision = tf32 : tensor<64x64xf16, #dot0> * tensor<64x256xf16, #dot1> -> tensor<64x256xf32, #dpas>
    %41 = ttg.convert_layout %40 : tensor<64x256xf32, #dpas> -> tensor<64x256xf32, #blocked1>
    // CHECK: tt.make_tensor_ptr {{.*}}, {{\[}}{{.*}}, {{.*}}], {{\[}}{{.*}}, {{.*}}], {{\[}}{{.*}}, {{.*}}] {order = array<i32: 1, 0>} : <tensor<64x256xf32, #[[DPAS]]>>
    %42 = tt.make_tensor_ptr %arg14, [%15, %20], [%26, %c1_i64], [%14, %19] {order = array<i32: 1, 0>} : <tensor<64x256xf32, #blocked1>>
    // CHECK: tt.store {{.*}}, {{.*}} {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x256xf32, #[[DPAS]]>>
    tt.store %42, %41 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x256xf32, #blocked1>>
    tt.return
  }
}


// -----

// COM: Case 4:
// COM: Checks that DPAS encoding has been forwarded to the store op
// COM: and the ttg.convert_layout operation in the loop has been removed
// CHECK: #[[DPAS:.+]] = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_sg_2d_block"} {
  tt.func public @matmul_kernel_with_block_pointers(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x256xf32, #blocked1>
    %18 = tt.make_tensor_ptr %arg0, [%c0_i64, %c0_i64], [%c0_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x32xf16, #blocked>>
    %22 = tt.make_tensor_ptr %arg1, [%c0_i64, %c0_i64], [%c0_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x256xf16, #blocked1>>
    %23:3 = scf.for %arg9 = %c0_i32 to %arg5 step %c32_i32 iter_args(%arg10 = %cst, %arg11 = %18, %arg12 = %22) -> (tensor<64x256xf32, #blocked1>, !tt.ptr<tensor<64x32xf16, #blocked>>, !tt.ptr<tensor<32x256xf16, #blocked1>>)  : i32 {
      %28 = tt.load %arg11 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major" } : !tt.ptr<tensor<64x32xf16, #blocked>>
      %29 = tt.load %arg12 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<32x256xf16, #blocked1>>
      %36 = ttg.convert_layout %arg10 : tensor<64x256xf32, #blocked1> -> tensor<64x256xf32, #dpas>
      %30 = ttg.convert_layout %28 : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #dot0>
      %31 = ttg.convert_layout %29 : tensor<32x256xf16, #blocked1> -> tensor<32x256xf16, #dot1>
      %32 = tt.dot %30, %31, %36, inputPrecision = tf32 : tensor<64x32xf16, #dot0> * tensor<32x256xf16, #dot1> -> tensor<64x256xf32, #dpas>
      %33 = tt.advance %arg11, [%c0_i32, %c32_i32] : <tensor<64x32xf16, #blocked>>
      %34 = tt.advance %arg12, [%c32_i32, %c0_i32] : <tensor<32x256xf16, #blocked1>>
      // CHECK-NOT: ttg.convert_layout
      %35 = ttg.convert_layout %32 : tensor<64x256xf32, #dpas> -> tensor<64x256xf32, #blocked1>
      scf.yield %35, %33, %34 : tensor<64x256xf32, #blocked1>, !tt.ptr<tensor<64x32xf16, #blocked>>, !tt.ptr<tensor<32x256xf16, #blocked1>>
    }
    %24 = arith.truncf %23#0 : tensor<64x256xf32, #blocked1> to tensor<64x256xf16, #blocked1>
    // CHECK: tt.make_tensor_ptr {{.*}}, {{\[}}{{.*}}, {{.*}}], {{\[}}{{.*}}, {{.*}}], {{\[}}{{.*}}, {{.*}}] {order = array<i32: 1, 0>} : <tensor<64x256xf16, #[[DPAS]]>>
    %27 = tt.make_tensor_ptr %arg2, [%c0_i64, %c0_i64], [%c0_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x256xf16, #blocked1>>
    // CHECK: tt.store {{.*}}, {{.*}} {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x256xf16, #[[DPAS]]>>
    tt.store %27, %24 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x256xf16, #blocked1>>
    tt.return
  }
}

// -----

// COM: Case 5:
// COM: Checks that block encoding has been forwarded to the store op
// COM: and the ttg.convert_layout operation has been removed
// CHECK: #[[BLOCKED:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_sg_2d_block"} {
  tt.func public @matmul_kernel_with_block_pointers(%arg0: !tt.ptr<f16>) {
    %c8_i32 = arith.constant 8 : i32
    %c64_i64 = arith.constant 64 : i64
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<64x256xf16, #blocked>
    // CHECK-NOT: ttg.convert_layout
    %25 = ttg.convert_layout %cst : tensor<64x256xf16, #blocked> -> tensor<64x256xf16, #blocked1>
    // CHECK: tt.make_tensor_ptr {{.*}}, {{\[}}{{.*}}, {{.*}}], {{\[}}{{.*}}, {{.*}}], {{\[}}{{.*}}, {{.*}}] {order = array<i32: 1, 0>} : <tensor<64x256xf16, #[[BLOCKED]]>>
    %27 = tt.make_tensor_ptr %arg0, [%c256_i64, %c256_i64], [%c64_i64, %c1_i64], [%c8_i32, %c8_i32] {order = array<i32: 1, 0>} : <tensor<64x256xf16, #blocked1>>
    // CHECK: tt.store {{.*}}, {{.*}} {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x256xf16, #[[BLOCKED]]>>
    tt.store %27, %25 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x256xf16, #blocked1>>
    tt.return
  }
}
