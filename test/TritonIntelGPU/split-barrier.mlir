// RUN: triton-opt %s -split-input-file -tritonintelgpu-pipeline="num-stages=3 use-barrier" | FileCheck %s

// COM: Descriptor equivalent of the removed block-pointer split-barrier test.
// COM: Checks that descriptor prefetches are emitted around a pipelined dot loop
// COM: and that the barrier-enabled pipeline inserts split barrier arrive/wait.

// CHECK: #[[$DPAS:.+]] = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>

module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: tt.func public @matmul_kernel_descriptor_split_barrier
  // CHECK: ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<128x64xf16>
  // CHECK: ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<64x256xf16>
  // CHECK: scf.for {{.*}} -> (tensor<128x256xf32, #[[$DPAS]]>, i32, i32)
  // CHECK: %[[BDATA:.*]] = triton_gen.split_barrier_arrive
  // CHECK: ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<128x64xf16>
  // CHECK: ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<64x256xf16>
  // CHECK: tt.dot {{.*}} : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>> * tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$DPAS]], kWidth = 2}>> -> tensor<128x256xf32, #[[$DPAS]]>
  // CHECK: triton_gen.split_barrier_wait %[[BDATA]]
  tt.func public @matmul_kernel_descriptor_split_barrier(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i64, %arg7: i64) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #dpas>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %descA = tt.make_tensor_descriptor %arg0, [%arg3, %arg5], [%arg6, %c1_i64] : <f16>, <128x64xf16>
    %descB = tt.make_tensor_descriptor %arg1, [%arg5, %arg4], [%arg7, %c1_i64] : <f16>, <64x256xf16>
    %result:2 = scf.for %k = %c0_i32 to %arg5 step %c64_i32 iter_args(%acc = %cst, %koff = %c0_i32) -> (tensor<128x256xf32, #dpas>, i32) : i32 {
      %a = tt.descriptor_load %descA[%c0_i32, %koff] {ttig.block_io = "row_major"} : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #dot0>
      %b = tt.descriptor_load %descB[%koff, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x256xf16> -> tensor<64x256xf16, #dot1>
      %d = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<128x64xf16, #dot0> * tensor<64x256xf16, #dot1> -> tensor<128x256xf32, #dpas>
      %next_koff = arith.addi %koff, %c64_i32 : i32
      scf.yield %d, %next_koff : tensor<128x256xf32, #dpas>, i32
    }
    tt.return
  }
}
