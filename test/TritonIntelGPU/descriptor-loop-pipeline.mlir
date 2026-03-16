// RUN: triton-opt %s -split-input-file -tritonintelgpu-pipeline="num-stages=3" | FileCheck %s

// COM: Test that descriptor loads for a tt.dot operation produce ttig.descriptor_prefetch ops
// COM: when pipelined with 3 stages.
// CHECK: #[[$DPAS:.+]] = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>

module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  tt.func public @matmul_kernel_descriptor(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i64, %arg7: i64) {
    // CHECK-LABEL:   tt.func public @matmul_kernel_descriptor
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #dpas>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64

    // COM: Create tensor descriptors for A [M, K] and B [K, N].
    %descA = tt.make_tensor_descriptor %arg0, [%arg3, %arg5], [%arg6, %c1_i64] : <f16>, <tensor<128x64xf16>>
    %descB = tt.make_tensor_descriptor %arg1, [%arg5, %arg4], [%arg7, %c1_i64] : <f16>, <tensor<64x256xf16>>

    // COM: 3-stage pipeline: 2 prefetching stages before the loop, 1 inside.
    // CHECK:      ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<tensor<128x64xf16>>
    // CHECK:      ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<tensor<64x256xf16>>
    // CHECK:      ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<tensor<128x64xf16>>
    // CHECK:      ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<tensor<64x256xf16>>
    // CHECK:      scf.for %[[IV:.*]] = {{.*}} to {{.*}} step {{.*}}
    // CHECK:        ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<tensor<128x64xf16>>
    // CHECK:        ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<tensor<64x256xf16>>
    // CHECK:        tt.dot {{.*}} : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>> * tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$DPAS]], kWidth = 2}>> -> tensor<128x256xf32, #[[$DPAS]]>
    // CHECK:        scf.yield
    %result:2 = scf.for %k = %c0_i32 to %arg5 step %c64_i32 iter_args(%acc = %cst, %koff = %c0_i32) -> (tensor<128x256xf32, #dpas>, i32) : i32 {
      %a = tt.descriptor_load %descA[%c0_i32, %koff] {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<128x64xf16>> -> tensor<128x64xf16, #dot0>
      %b = tt.descriptor_load %descB[%koff, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<64x256xf16>> -> tensor<64x256xf16, #dot1>
      %d = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<128x64xf16, #dot0> * tensor<64x256xf16, #dot1> -> tensor<128x256xf32, #dpas>
      %next_koff = arith.addi %koff, %c64_i32 : i32
      scf.yield %d, %next_koff : tensor<128x256xf32, #dpas>, i32
    }
    tt.return
  }
}
