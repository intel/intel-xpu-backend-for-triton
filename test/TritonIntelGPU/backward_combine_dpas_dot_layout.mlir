// RUN: triton-opt %s -split-input-file -tritonintelgpu-remove-layout-conversions='max-backward-remat-iterations=10' 2>&1 | FileCheck %s

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
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i64, %arg7: i32, %arg8: i64) {
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
    %19 = arith.muli %13, %c256_i32 : i32
    // CHECK: tt.make_tensor_descriptor
    // CHECK: tt.make_tensor_descriptor
    %desc_a = tt.make_tensor_descriptor %arg0, [%arg3, %arg5], [%arg6, %c1_i64] : <f16>, <tensor<64x32xf16>>
    %desc_b = tt.make_tensor_descriptor %arg1, [%arg5, %arg4], [%arg8, %c1_i64] : <f16>, <tensor<32x256xf16>>
    // CHECK: scf.for {{.*}} -> (tensor<64x256xf32, #[[DPAS]]>)
    // CHECK: tt.descriptor_load {{.*}} -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #[[DPAS]], kWidth = 1}>>
    // CHECK: tt.descriptor_load {{.*}} -> tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #[[DPAS]], kWidth = 2}>>
    // CHECK-NOT: ttg.convert_layout
    // CHECK: tt.dot
    // CHECK: scf.yield
    %23 = scf.for %arg9 = %c0_i32 to %arg5 step %c32_i32 iter_args(%arg10 = %cst) -> (tensor<64x256xf32, #dpas>) : i32 {
      %28 = tt.descriptor_load %desc_a[%14, %arg9] {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<64x32xf16>> -> tensor<64x32xf16, #blocked>
      %29 = tt.descriptor_load %desc_b[%arg9, %19] {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<32x256xf16>> -> tensor<32x256xf16, #blocked1>
      %30 = ttg.convert_layout %28 : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #dot0>
      %31 = ttg.convert_layout %29 : tensor<32x256xf16, #blocked1> -> tensor<32x256xf16, #dot1>
      %32 = tt.dot %30, %31, %arg10, inputPrecision = tf32 : tensor<64x32xf16, #dot0> * tensor<32x256xf16, #dot1> -> tensor<64x256xf32, #dpas>
      scf.yield %32 : tensor<64x256xf32, #dpas>
    }
    // CHECK: arith.truncf {{.*}} : tensor<64x256xf32, #[[DPAS]]> to tensor<64x256xf16, #[[DPAS]]>
    // CHECK-NOT: ttg.convert_layout
    // CHECK: tt.descriptor_store
    %24 = arith.truncf %23 : tensor<64x256xf32, #dpas> to tensor<64x256xf16, #dpas>
    %25 = ttg.convert_layout %24 : tensor<64x256xf16, #dpas> -> tensor<64x256xf16, #blocked1>
    %desc_c = tt.make_tensor_descriptor %arg2, [%arg3, %arg4], [%arg8, %c1_i64] : <f16>, <tensor<64x256xf16>>
    tt.descriptor_store %desc_c[%14, %19], %25 : !tt.tensordesc<tensor<64x256xf16>>, tensor<64x256xf16, #blocked1>
    tt.return
  }
}


// -----

// COM: Case 2: Similar to Case 1 but the loads do not have the blockIO "row_major" attribute.
// COM: Checks that DPAS encoding has been forwarded from the dot op to the store op via the loop return values
// COM: and that the ttg.convert_layout operation has been removed.
// CHECK: #[[DPAS:.+]] = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  tt.func public @matmul_kernel_no_block_io(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i64, %arg7: i32, %arg8: i64) {
    %c64_i32 = arith.constant 64 : i32
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x256xf32, #dpas>
    %desc_a = tt.make_tensor_descriptor %arg0, [%arg3, %arg5], [%arg6, %c1_i64] : <f16>, <tensor<64x32xf16>>
    %desc_b = tt.make_tensor_descriptor %arg1, [%arg5, %arg4], [%arg8, %c1_i64] : <f16>, <tensor<32x256xf16>>
    %23 = scf.for %arg9 = %c0_i32 to %arg5 step %c32_i32 iter_args(%arg10 = %cst) -> (tensor<64x256xf32, #dpas>) : i32 {
      %28 = tt.descriptor_load %desc_a[%c0_i32, %arg9] : !tt.tensordesc<tensor<64x32xf16>> -> tensor<64x32xf16, #blocked>
      %29 = tt.descriptor_load %desc_b[%arg9, %c0_i32] : !tt.tensordesc<tensor<32x256xf16>> -> tensor<32x256xf16, #blocked1>
      %30 = ttg.convert_layout %28 : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #dot0>
      %31 = ttg.convert_layout %29 : tensor<32x256xf16, #blocked1> -> tensor<32x256xf16, #dot1>
      %32 = tt.dot %30, %31, %arg10, inputPrecision = tf32 : tensor<64x32xf16, #dot0> * tensor<32x256xf16, #dot1> -> tensor<64x256xf32, #dpas>
      scf.yield %32 : tensor<64x256xf32, #dpas>
    }
    // CHECK: arith.truncf {{.*}} : tensor<64x256xf32, #[[DPAS]]> to tensor<64x256xf16, #[[DPAS]]>
    // CHECK-NOT: ttg.convert_layout
    // CHECK: tt.descriptor_store {{.*}} : !tt.tensordesc<tensor<64x256xf16>>, tensor<64x256xf16, #[[DPAS]]>
    %24 = arith.truncf %23 : tensor<64x256xf32, #dpas> to tensor<64x256xf16, #dpas>
    %25 = ttg.convert_layout %24 : tensor<64x256xf16, #dpas> -> tensor<64x256xf16, #blocked1>
    %desc_c = tt.make_tensor_descriptor %arg2, [%arg3, %arg4], [%arg8, %c1_i64] : <f16>, <tensor<64x256xf16>>
    tt.descriptor_store %desc_c[%c0_i32, %c0_i32], %25 : !tt.tensordesc<tensor<64x256xf16>>, tensor<64x256xf16, #blocked1>
    tt.return
  }
}

// -----

// COM: Case 3: Similar to Case 1 but with an additional store after the loop.
// COM: Checks that DPAS encoding has been forwarded from the dot op to the store op via the loop return values.
// CHECK: #[[BLOCKED:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
// CHECK: #[[DPAS:.+]] = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  tt.func public @matmul_kernel_with_extra_store(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i64, %arg7: i64, %arg8: i64, %arg13: !tt.ptr<f16>, %arg14: !tt.ptr<f32>) {
    %c64_i32 = arith.constant 64 : i32
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x256xf32, #dpas>
    %desc_a = tt.make_tensor_descriptor %arg0, [%arg3, %arg5], [%arg6, %c1_i64] : <f16>, <tensor<64x32xf16>>
    %desc_b = tt.make_tensor_descriptor %arg1, [%arg5, %arg4], [%arg7, %c1_i64] : <f16>, <tensor<32x256xf16>>
    %23 = scf.for %arg9 = %c0_i32 to %arg5 step %c32_i32 iter_args(%arg10 = %cst) -> (tensor<64x256xf32, #dpas>) : i32 {
      // COM: Layout conversions in the loop should be removed.
      // CHECK: scf.for
      // CHECK-NOT: ttg.convert_layout
      // CHECK: scf.yield
      %28 = tt.descriptor_load %desc_a[%c0_i32, %arg9] {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<64x32xf16>> -> tensor<64x32xf16, #blocked>
      %29 = tt.descriptor_load %desc_b[%arg9, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<32x256xf16>> -> tensor<32x256xf16, #blocked1>
      %30 = ttg.convert_layout %28 : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #dot0>
      %31 = ttg.convert_layout %29 : tensor<32x256xf16, #blocked1> -> tensor<32x256xf16, #dot1>
      %32 = tt.dot %30, %31, %arg10, inputPrecision = tf32 : tensor<64x32xf16, #dot0> * tensor<32x256xf16, #dot1> -> tensor<64x256xf32, #dpas>
      scf.yield %32 : tensor<64x256xf32, #dpas>
    }
    // CHECK: arith.truncf
    // CHECK-NOT: ttg.convert_layout
    // CHECK: tt.descriptor_store {{.*}} : !tt.tensordesc<tensor<64x256xf16>>, tensor<64x256xf16, #[[DPAS]]>
    // CHECK: tt.descriptor_load {{.*}} -> tensor<64x256xf16, #[[BLOCKED]]>
    %24 = arith.truncf %23 : tensor<64x256xf32, #dpas> to tensor<64x256xf16, #dpas>
    %25 = ttg.convert_layout %24 : tensor<64x256xf16, #dpas> -> tensor<64x256xf16, #blocked1>
    %desc_c = tt.make_tensor_descriptor %arg2, [%arg3, %arg4], [%arg8, %c1_i64] : <f16>, <tensor<64x256xf16>>
    tt.descriptor_store %desc_c[%c0_i32, %c0_i32], %25 : !tt.tensordesc<tensor<64x256xf16>>, tensor<64x256xf16, #blocked1>
    %35 = tt.descriptor_load %desc_c[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<64x256xf16>> -> tensor<64x256xf16, #blocked1>
    %desc_d = tt.make_tensor_descriptor %arg13, [%arg3, %arg5], [%arg6, %c1_i64] : <f16>, <tensor<64x64xf16>>
    %37 = tt.descriptor_load %desc_d[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<64x64xf16>> -> tensor<64x64xf16, #blocked>
    %38 = ttg.convert_layout %37 : tensor<64x64xf16, #blocked> -> tensor<64x64xf16, #dot0>
    %39 = ttg.convert_layout %35 : tensor<64x256xf16, #blocked1> -> tensor<64x256xf16, #dot1>
    %40 = tt.dot %38, %39, %cst, inputPrecision = tf32 : tensor<64x64xf16, #dot0> * tensor<64x256xf16, #dot1> -> tensor<64x256xf32, #dpas>
    // CHECK: tt.dot
    // CHECK-NOT: ttg.convert_layout
    // CHECK: tt.descriptor_store {{.*}} : !tt.tensordesc<tensor<64x256xf32>>, tensor<64x256xf32, #[[DPAS]]>
    %41 = ttg.convert_layout %40 : tensor<64x256xf32, #dpas> -> tensor<64x256xf32, #blocked1>
    %desc_e = tt.make_tensor_descriptor %arg14, [%arg3, %arg4], [%arg8, %c1_i64] : <f32>, <tensor<64x256xf32>>
    tt.descriptor_store %desc_e[%c0_i32, %c0_i32], %41 : !tt.tensordesc<tensor<64x256xf32>>, tensor<64x256xf32, #blocked1>
    tt.return
  }
}

// -----

// COM: Case 4: Similar to Case 1 but with a convert layout on the dot op return value in the loop.
// COM: Checks that DPAS encoding has been forwarded from the dot op to the store op through the loop results
// COM: and the ttg.convert_layout operations in the loop have been removed.
// CHECK: #[[DPAS:.+]] = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  tt.func public @matmul_kernel_convert_on_dot_result(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg5: i32) {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x256xf32, #blocked1>
    %desc_a = tt.make_tensor_descriptor %arg0, [%arg5, %arg5], [%c0_i64, %c1_i64] : <f16>, <tensor<64x32xf16>>
    %desc_b = tt.make_tensor_descriptor %arg1, [%arg5, %arg5], [%c0_i64, %c1_i64] : <f16>, <tensor<32x256xf16>>
    %23 = scf.for %arg9 = %c0_i32 to %arg5 step %c32_i32 iter_args(%arg10 = %cst) -> (tensor<64x256xf32, #blocked1>) : i32 {
      // CHECK: scf.for
      // CHECK-NOT: ttg.convert_layout
      // CHECK: scf.yield
      %28 = tt.descriptor_load %desc_a[%c0_i32, %arg9] {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<64x32xf16>> -> tensor<64x32xf16, #blocked>
      %29 = tt.descriptor_load %desc_b[%arg9, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<32x256xf16>> -> tensor<32x256xf16, #blocked1>
      %36 = ttg.convert_layout %arg10 : tensor<64x256xf32, #blocked1> -> tensor<64x256xf32, #dpas>
      %30 = ttg.convert_layout %28 : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #dot0>
      %31 = ttg.convert_layout %29 : tensor<32x256xf16, #blocked1> -> tensor<32x256xf16, #dot1>
      %32 = tt.dot %30, %31, %36, inputPrecision = tf32 : tensor<64x32xf16, #dot0> * tensor<32x256xf16, #dot1> -> tensor<64x256xf32, #dpas>
      %35 = ttg.convert_layout %32 : tensor<64x256xf32, #dpas> -> tensor<64x256xf32, #blocked1>
      scf.yield %35 : tensor<64x256xf32, #blocked1>
    }
    // CHECK: arith.truncf
    // CHECK-NOT: ttg.convert_layout
    // CHECK: tt.descriptor_store {{.*}} : !tt.tensordesc<tensor<64x256xf16>>, tensor<64x256xf16, #[[DPAS]]>
    %24 = arith.truncf %23 : tensor<64x256xf32, #blocked1> to tensor<64x256xf16, #blocked1>
    %desc_c = tt.make_tensor_descriptor %arg2, [%arg5, %arg5], [%c0_i64, %c1_i64] : <f16>, <tensor<64x256xf16>>
    tt.descriptor_store %desc_c[%c0_i32, %c0_i32], %24 : !tt.tensordesc<tensor<64x256xf16>>, tensor<64x256xf16, #blocked1>
    tt.return
  }
}

// -----

// COM: Case 5:
// COM: Checks that the convert_layout is eliminated and the descriptor store
// COM: receives the data directly. The store anchors #blocked1; the constant
// COM: adopts the store's encoding, removing the convert.
// CHECK: #[[$BLOCKED:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: @store_encoding_forwarding
  tt.func public @store_encoding_forwarding(%arg0: !tt.ptr<f16>) {
    %c8_i32 = arith.constant 8 : i32
    %c64_i64 = arith.constant 64 : i64
    %c1_i64 = arith.constant 1 : i64
    %c256_i32 = arith.constant 256 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x256xf16, #blocked>
    // CHECK-NOT: ttg.convert_layout
    %25 = ttg.convert_layout %cst : tensor<64x256xf16, #blocked> -> tensor<64x256xf16, #blocked1>
    // CHECK: tt.descriptor_store {{.*}} : !tt.tensordesc<tensor<64x256xf16>>, tensor<64x256xf16, #[[$BLOCKED]]>
    %desc = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c64_i64, %c1_i64] : <f16>, <tensor<64x256xf16>>
    tt.descriptor_store %desc[%c8_i32, %c8_i32], %25 : !tt.tensordesc<tensor<64x256xf16>>, tensor<64x256xf16, #blocked1>
    tt.return
  }
}

// -----

// COM: Fix for issue #4866

// CHECK: #[[BLOCKED:.+]] = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK: #[[BLOCKED1:.+]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @test_4866(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f32>, %arg2: i64) {
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #blocked>
    %cst_0 = arith.constant dense<5.000000e-01> : tensor<16x32xf32, #blocked1>
    %c64_i64 = arith.constant 64 : i64
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c16_i32 = arith.constant 16 : i32
    %desc_in = tt.make_tensor_descriptor %arg0, [%c16_i32, %c32_i32], [%c64_i64, %c1_i64] : <f16>, <tensor<16x32xf16>>
    %desc_out = tt.make_tensor_descriptor %arg1, [%c16_i32, %c32_i32], [%c64_i64, %c1_i64] : <f32>, <tensor<16x32xf32>>
    %2 = scf.for %arg3 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%acc_unused = %c0_i32) -> (i32) : i32 {
      // CHECK: scf.for {{.*}}
      // CHECK: tt.descriptor_load {{.*}} -> tensor<16x32xf16, #[[BLOCKED1]]>
      // CHECK: [[CONV1:%.*]] = ttg.convert_layout {{.*}} : tensor<16x32xf16, #[[BLOCKED1]]> -> tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #[[BLOCKED]]}>>
      // CHECK: [[DOT_RES:%.*]] = tt.dot {{.*}}, [[CONV1]], {{.*}} : tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #[[BLOCKED]]}>> * tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #[[BLOCKED]]}>> -> tensor<16x32xf32, #[[BLOCKED]]>
      // CHECK-NEXT: tt.descriptor_store {{.*}}, [[DOT_RES]] : !tt.tensordesc<tensor<16x32xf32>>, tensor<16x32xf32, #[[BLOCKED]]>
      %3 = tt.descriptor_load %desc_in[%c0_i32, %c32_i32] : !tt.tensordesc<tensor<16x32xf16>> -> tensor<16x32xf16, #blocked2>
      %4 = ttg.convert_layout %3 : tensor<16x32xf16, #blocked2> -> tensor<16x32xf16, #blocked1>
      %5 = ttg.convert_layout %cst : tensor<16x16xf16, #blocked> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked3}>>
      %6 = ttg.convert_layout %4 : tensor<16x32xf16, #blocked1> -> tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked3}>>
      %7 = ttg.convert_layout %cst_0 : tensor<16x32xf32, #blocked1> -> tensor<16x32xf32, #blocked3>
      %8 = tt.dot %5, %6, %7 : tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked3}>> * tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked3}>> -> tensor<16x32xf32, #blocked3>
      %9 = ttg.convert_layout %8 : tensor<16x32xf32, #blocked3> -> tensor<16x32xf32, #blocked1>
      %10 = ttg.convert_layout %9 : tensor<16x32xf32, #blocked1> -> tensor<16x32xf32, #blocked2>
      tt.descriptor_store %desc_out[%c0_i32, %c32_i32], %10 : !tt.tensordesc<tensor<16x32xf32>>, tensor<16x32xf32, #blocked2>
      scf.yield %acc_unused : i32
    }
    tt.return
  }
}

// -----

// CHECK: #[[$BLOCKED:.+]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [2, 1], order = [1, 0]}>
// CHECK-NOT: #ttg.blocked
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [2, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 2 : i32, ttig.support_2d_block_io} {
// CHECK-LABEL: @test_tdesc_layout_backward
  tt.func public @test_tdesc_layout_backward(%ptr: !tt.ptr<f16>) -> tensor<4x128xf16, #blocked1> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i64
    %c128 = arith.constant 128 : i64
    %c128_i32 = arith.constant 128 : i32
    %xoffset = tt.get_program_id x : i32
    %desc = tt.make_tensor_descriptor %ptr, [%c128_i32, %c128_i32], [%c128, %c1] : <f16>, <tensor<4x128xf16>>
    %val = tt.descriptor_load %desc[%xoffset, %c0] {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<4x128xf16>> -> tensor<4x128xf16, #blocked>
    // CHECK: tt.descriptor_load {{.*}} -> tensor<4x128xf16, #[[$BLOCKED]]>
    // CHECK-NOT: ttg.convert_layout
    // CHECK: tt.return
    %0 = ttg.convert_layout %val : tensor<4x128xf16, #blocked> -> tensor<4x128xf16, #blocked1>
    tt.return %0 : tensor<4x128xf16, #blocked1>
  }
}
