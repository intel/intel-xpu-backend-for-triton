// RUN: triton-opt %s -split-input-file -tritonintelgpu-remove-layout-conversions | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>

// COM: ============================================================
// COM: Test 7: A descriptor-backed matmul loop carries a blocked
// COM: accumulator only in the input IR, but RemoveLayoutConversions
// COM: forwards the DPAS encoding through the loop results and drops
// COM: the loop-local and post-loop convert_layout ops before the
// COM: final descriptor_store.
// COM: ============================================================

// CHECK: #[[$DPAS:.+]] = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
// CHECK-LABEL: @descriptor_store_dpas_forwarded_through_loop
// CHECK: %[[LOOP:.*]]:2 = scf.for {{.*}} -> (tensor<64x256xf32, #[[$DPAS]]>, i32)
// CHECK-NOT: ttg.convert_layout
// CHECK: arith.truncf %[[LOOP]]#0 : tensor<64x256xf32, #[[$DPAS]]> to tensor<64x256xf16, #[[$DPAS]]>
// CHECK: tt.descriptor_store {{.*}}, {{.*}} : !tt.tensordesc<tensor<64x256xf16>>, tensor<64x256xf16, #[[$DPAS]]>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  tt.func @descriptor_store_dpas_forwarded_through_loop(%descA: !tt.tensordesc<tensor<64x32xf16>>, %descB: !tt.tensordesc<tensor<32x256xf16>>, %descC: !tt.tensordesc<tensor<64x256xf16>>, %kLimit: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x256xf32, #blocked1>
    %result:2 = scf.for %k = %c0_i32 to %kLimit step %c32_i32 iter_args(%acc = %cst, %koff = %c0_i32) -> (tensor<64x256xf32, #blocked1>, i32) : i32 {
      %a = tt.descriptor_load %descA[%c0_i32, %koff] {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<64x32xf16>> -> tensor<64x32xf16, #blocked>
      %b = tt.descriptor_load %descB[%koff, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<32x256xf16>> -> tensor<32x256xf16, #blocked1>
      %acc_dpas = ttg.convert_layout %acc : tensor<64x256xf32, #blocked1> -> tensor<64x256xf32, #dpas>
      %a_dpas = ttg.convert_layout %a : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #dot0>
      %b_dpas = ttg.convert_layout %b : tensor<32x256xf16, #blocked1> -> tensor<32x256xf16, #dot1>
      %d = tt.dot %a_dpas, %b_dpas, %acc_dpas, inputPrecision = tf32 : tensor<64x32xf16, #dot0> * tensor<32x256xf16, #dot1> -> tensor<64x256xf32, #dpas>
      %next_acc = ttg.convert_layout %d : tensor<64x256xf32, #dpas> -> tensor<64x256xf32, #blocked1>
      %next_koff = arith.addi %koff, %c32_i32 : i32
      scf.yield %next_acc, %next_koff : tensor<64x256xf32, #blocked1>, i32
    }
    %trunc = arith.truncf %result#0 : tensor<64x256xf32, #blocked1> to tensor<64x256xf16, #blocked1>
    tt.descriptor_store %descC[%c0_i32, %c0_i32], %trunc : !tt.tensordesc<tensor<64x256xf16>>, tensor<64x256xf16, #blocked1>
    tt.return
  }
}
