// RUN: triton-opt %s -split-input-file --tritonintelgpu-accelerate-matmul | FileCheck %s

// COM: Verifies the exponent-add fast path in DecomposeScaledBlocked:
// COM: when the compute type is bf16 and the scale element type is integer
// COM: (E8M0-in-uint8, per MXFP spec), the pass emits an integer add on the
// COM: bf16 raw bits instead of an FP multiply.  This avoids the bf16 mulf
// COM: that `arith_emulate_unsupported_floats` widens into a full-size f32
// COM: intermediate on BMG (no native bf16 arithmetic).
// COM:
// COM: The module deliberately omits ttig.support_subgroup_scaled_matrix_multiply_accumulate
// COM: so that AccelerateMatmul's BDPAS-aware patterns are gated off and
// COM: DecomposeScaledBlocked owns the lowering of tt.dot_scaled.

#blocked  = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 8], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 4], warpsPerCTA = [16, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 8], order = [1, 0]}>

module attributes {ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func public @kernel_exponent_add
  // CHECK-NOT:   tt.dot_scaled
  // COM: The scale-application step must be an integer add on i16 tensors,
  // COM: not an FP multiply on bf16/f32.  arith.mulf must not appear anywhere
  // COM: in the decomposed body — the only FP arithmetic left is tt.dot.
  // CHECK-DAG:   arith.addi {{.*}} : tensor<{{.*}}xi16
  // CHECK-NOT:   arith.mulf
  // CHECK:       tt.dot
  tt.func public @kernel_exponent_add(
      %a: tensor<128x64xf8E5M2, #blocked2>,
      %scale_a: tensor<128x2xi8, #blocked1>,
      %b: tensor<32x128xi8, #blocked>,
      %scale_b: tensor<128x2xi8, #blocked1>) -> tensor<128x128xf32, #blocked> {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %0 = tt.dot_scaled %a scale %scale_a, %b scale %scale_b, %cst lhs = e5m2 rhs = e2m1
         {fastMath = true, lhs_k_pack = true, rhs_k_pack = true}
         : tensor<128x64xf8E5M2, #blocked2>, tensor<128x2xi8, #blocked1>
         * tensor<32x128xi8, #blocked>, tensor<128x2xi8, #blocked1>
         -> tensor<128x128xf32, #blocked>
    tt.return %0 : tensor<128x128xf32, #blocked>
  }
}
