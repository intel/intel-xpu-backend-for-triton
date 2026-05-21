// RUN: triton-opt %s -split-input-file --tritonintelgpu-accelerate-matmul | FileCheck %s

// COM: Regression test for the broadcastScale extent in DecomposeScaledBlocked.
// COM:
// COM: broadcastScale used to hardcode 32 as the broadcast extent. After the port
// COM: from upstream, it derives the extent from DotScaledOp::deduceScaleFactor()
// COM: (defined in lib/Dialect/Triton/IR/Ops.cpp), which returns kdim / scaleK.
// COM: deduceScaleFactor() validates the result is 16 or 32 — the only two scale
// COM: factors that are dialect-legal today (MXFP4 = 32, NVFP4 = 16).
// COM:
// COM: This test pins down the contract by feeding a tt.dot_scaled with shapes that
// COM: yield scale factor 16 instead of 32 and asserting the resulting broadcast
// COM: extent matches. Without the port, the broadcast would emit `x32` and the
// COM: subsequent reshape would fail shape inference.
// COM:
// COM: Shape arithmetic for LHS (E2M1, lhs_k_pack = true):
// COM:   operand tensor<128x16xi8> -> packed K = 16, unpacked kdim = 16 * 2 = 32.
// COM:   scale   tensor<128x2xi8>  -> scaleK = 2.
// COM:   factor  = kdim / scaleK   = 32 / 2 = 16.
// COM:
// COM: The module deliberately omits ttig.support_subgroup_scaled_matrix_multiply_accumulate
// COM: so that AccelerateMatmul's BDPAS-aware patterns are gated off and
// COM: DecomposeScaledBlocked owns the lowering of tt.dot_scaled.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 8], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 4], warpsPerCTA = [16, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 4], order = [1, 0]}>

module attributes {ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func public @kernel_scale_factor_16
  // CHECK-NOT: tt.dot_scaled
  // CHECK-DAG: tt.broadcast {{.*}} -> tensor<128x2x16x{{.*}}
  // CHECK-DAG: tt.broadcast {{.*}} -> tensor<2x128x16x{{.*}}
  // CHECK: tt.dot
  tt.func public @kernel_scale_factor_16(%a: tensor<128x16xi8, #blocked2>, %scale_a: tensor<128x2xi8, #blocked1>, %b: tensor<16x128xi8, #blocked>, %scale_b: tensor<128x2xi8, #blocked1>) -> tensor<128x128xf32, #blocked> {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %0 = tt.dot_scaled %a scale %scale_a, %b scale %scale_b, %cst lhs = e2m1 rhs = e2m1 {fastMath = true, lhs_k_pack = true, rhs_k_pack = true} : tensor<128x16xi8, #blocked2>, tensor<128x2xi8, #blocked1> * tensor<16x128xi8, #blocked>, tensor<128x2xi8, #blocked1> -> tensor<128x128xf32, #blocked>
    tt.return %0 : tensor<128x128xf32, #blocked>
  }
}
