// RUN: triton-opt %s -split-input-file -tritonintelgpu-pipeline="num-stages=3" | FileCheck %s --implicit-check-not=ttig.prefetch

// COM: A K loop whose only matmul is a tt.dot_scaled is pipelined like a tt.dot loop: the 2D block
// COM: loads of the A and B operands are prefetched num-stages - 1 iterations ahead (issue #8931).
// COM: The scale operands are not prefetched as dot operands, not even one loaded in the loop with a
// COM: `ttig.block_io` load, as the B scale is here: a scale has a linear layout, not a dot-operand one,
// COM: so its load does not feed the dot as an operand tile (`ttig.prefetch` is the implicit CHECK-NOT
// COM: of the RUN line). This module has no `ttig.support_prefetch_256b`: with it (Xe3P and later),
// COM: the B scale load would be prefetched as an elementwise load instead.

#linear = #ttg.linear<{register = [[8, 0], [16, 0], [0, 1]], lane = [[1, 0], [2, 0], [4, 0], [0, 0]], warp = [[0, 0], [0, 0], [32, 0], [64, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[16, 0], [0, 1]], lane = [[1, 0], [2, 0], [4, 0], [8, 0]], warp = [[32, 0], [64, 0], [0, 0], [0, 0]], block = []}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 4, threadsPerWarp = 16, warpsPerCTA = [4, 4], repCluster = [4, 2], A = [32, 32], B = [32, 32], C = [32, 32]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_subgroup_matrix_multiply_accumulate_bf8, ttig.support_subgroup_scaled_matrix_multiply_accumulate} {
  // CHECK-LABEL: tt.func public @dot_scaled_only(
  // CHECK-SAME:      %[[DESC_A:.*]]: !tt.tensordesc<128x64xf8E4M3FN>, %[[DESC_B:.*]]: !tt.tensordesc<128x64xf8E4M3FN>, %[[OFF_M:.*]]: i32, %[[OFF_N:.*]]: i32, %{{.*}}: i32, %[[SCALE_A:.*]]: tensor<128x2xi8, #linear>, %[[SCALE_B_PTR:.*]]: tensor<128x2x!tt.ptr<i8>, #linear1>)
  // COM: Two iterations ahead for each operand before the loop:
  // CHECK:         ttig.descriptor_prefetch %[[DESC_A]][%[[OFF_M]], {{.*}}] {ttig.block_io = "row_major"} : !tt.tensordesc<128x64xf8E4M3FN>
  // CHECK:         ttig.descriptor_prefetch %[[DESC_B]][%[[OFF_N]], {{.*}}] {ttig.block_io = "column_major"} : !tt.tensordesc<128x64xf8E4M3FN>
  // CHECK:         ttig.descriptor_prefetch %[[DESC_A]][%[[OFF_M]], {{.*}}] {ttig.block_io = "row_major"} : !tt.tensordesc<128x64xf8E4M3FN>
  // CHECK:         ttig.descriptor_prefetch %[[DESC_B]][%[[OFF_N]], {{.*}}] {ttig.block_io = "column_major"} : !tt.tensordesc<128x64xf8E4M3FN>
  // CHECK:         scf.for
  // COM: ... and one of each in every iteration, next to the loads feeding the dot_scaled:
  // CHECK:           ttig.descriptor_prefetch %[[DESC_A]][%[[OFF_M]], {{.*}}] {ttig.block_io = "row_major"} : !tt.tensordesc<128x64xf8E4M3FN>
  // CHECK:           ttig.descriptor_prefetch %[[DESC_B]][%[[OFF_N]], {{.*}}] {ttig.block_io = "column_major"} : !tt.tensordesc<128x64xf8E4M3FN>
  // CHECK-DAG:       %[[A:.*]] = tt.descriptor_load %[[DESC_A]]
  // CHECK-DAG:       %[[B:.*]] = tt.descriptor_load %[[DESC_B]]
  // CHECK-DAG:       %[[SCALE_B:.*]] = tt.load %[[SCALE_B_PTR]] {ttig.block_io = "row_major"}
  // CHECK:           tt.dot_scaled %[[A]] scale %[[SCALE_A]], %[[B]] scale %[[SCALE_B]]
  // CHECK-NOT:       ttig.descriptor_prefetch
  tt.func public @dot_scaled_only(%descA: !tt.tensordesc<128x64xf8E4M3FN>, %descB: !tt.tensordesc<128x64xf8E4M3FN>, %offM: i32, %offN: i32, %K: i32, %scaleA: tensor<128x2xi8, #linear>, %scaleBPtr: tensor<128x2x!tt.ptr<i8>, #linear1>) -> tensor<128x128xf32, #mma> {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %acc = scf.for %k = %c0_i32 to %K step %c64_i32 iter_args(%acc_in = %cst) -> (tensor<128x128xf32, #mma>) : i32 {
      %a = tt.descriptor_load %descA[%offM, %k] {ttig.block_io = "row_major"} : !tt.tensordesc<128x64xf8E4M3FN> -> tensor<128x64xf8E4M3FN, #dot0>
      %b = tt.descriptor_load %descB[%offN, %k] {ttig.block_io = "column_major"} : !tt.tensordesc<128x64xf8E4M3FN> -> tensor<64x128xf8E4M3FN, #dot1>
      %scaleB = tt.load %scaleBPtr {ttig.block_io = "row_major"} : tensor<128x2x!tt.ptr<i8>, #linear1>
      %d = tt.dot_scaled %a scale %scaleA, %b scale %scaleB, %acc_in lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<128x64xf8E4M3FN, #dot0>, tensor<128x2xi8, #linear> * tensor<64x128xf8E4M3FN, #dot1>, tensor<128x2xi8, #linear1> -> tensor<128x128xf32, #mma>
      scf.yield %d : tensor<128x128xf32, #mma>
    }
    tt.return %acc : tensor<128x128xf32, #mma>
  }
}
