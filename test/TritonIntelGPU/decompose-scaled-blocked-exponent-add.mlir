// RUN: triton-opt %s -split-input-file --tritonintelgpu-accelerate-matmul | FileCheck %s

// COM: Verify that DecomposeScaledBlocked replaces the bf16 scale-multiply
// COM: with a sign-safe i16 exponent-add, eliminating f32 intermediates.
// COM:
// COM: On platforms without native bf16 arithmetic (e.g. BMG/Xe2),
// COM: arith_emulate_unsupported_floats widens MulFOp(bf16) to three large
// COM: f32 tensors (extf + mulf + truncf). The exponent-add avoids this by
// COM: computing the scale multiplication entirely in i16 arithmetic.
// COM:
// COM: This test has no ttig.support_subgroup_scaled_matrix_multiply_accumulate,
// COM: so AccelerateMatmul gates off the BDPAS-native path and
// COM: DecomposeScaledBlocked owns the tt.dot_scaled lowering.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 8], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 4], warpsPerCTA = [16, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 4], order = [1, 0]}>

module attributes {ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, "ttg.threads-per-warp" = 16 : i32} {

  // CHECK-LABEL: tt.func public @mxfp4_exponent_add
  // COM: tt.dot_scaled is decomposed
  // CHECK-NOT: tt.dot_scaled
  // COM: i16 exponent-add pattern: sign extraction, delta subtraction, addi, clamp
  // CHECK: arith.andi {{.*}} : tensor<{{.*}}xi16
  // CHECK: arith.subi {{.*}} : tensor<{{.*}}xi16
  // CHECK: arith.addi {{.*}} : tensor<{{.*}}xi16
  // COM: bitcast back to bf16 (result of exponent-add)
  // CHECK: arith.bitcast {{.*}} : tensor<{{.*}}xi16{{.*}}> to tensor<{{.*}}xbf16
  // COM: no bf16 float multiplication (would be widened to f32)
  // CHECK-NOT: arith.mulf {{.*}} : tensor<{{.*}}xbf16
  // COM: NaN mask still present (scale==0xFF → NaN): cmpi eq against 0xff
  // CHECK: arith.cmpi eq
  // CHECK: tt.dot
  tt.func public @mxfp4_exponent_add(
      %a: tensor<128x16xi8, #blocked2>,
      %scale_a: tensor<128x2xi8, #blocked1>,
      %b: tensor<16x128xi8, #blocked>,
      %scale_b: tensor<128x2xi8, #blocked1>,
      %c: tensor<128x128xf32, #blocked>) -> tensor<128x128xf32, #blocked> {
    %0 = tt.dot_scaled %a scale %scale_a, %b scale %scale_b, %c lhs = e2m1 rhs = e2m1 {fastMath = false, lhs_k_pack = true, rhs_k_pack = true} : tensor<128x16xi8, #blocked2>, tensor<128x2xi8, #blocked1> * tensor<16x128xi8, #blocked>, tensor<128x2xi8, #blocked1> -> tensor<128x128xf32, #blocked>
    tt.return %0 : tensor<128x128xf32, #blocked>
  }
}

// -----

// Verify fastMath=true: exponent-add applies but NaN mask is skipped.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 8], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 4], warpsPerCTA = [16, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 4], order = [1, 0]}>

module attributes {ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, "ttg.threads-per-warp" = 16 : i32} {

  // CHECK-LABEL: tt.func public @mxfp4_exponent_add_fastmath
  // CHECK-NOT: tt.dot_scaled
  // CHECK: arith.addi {{.*}} : tensor<{{.*}}xi16
  // CHECK: arith.bitcast {{.*}} : tensor<{{.*}}xi16{{.*}}> to tensor<{{.*}}xbf16
  // CHECK-NOT: arith.mulf {{.*}} : tensor<{{.*}}xbf16
  // COM: no NaN mask with fastMath=true (NaN mask uses cmpi eq against 0xff)
  // CHECK-NOT: arith.cmpi eq
  // CHECK: tt.dot
  tt.func public @mxfp4_exponent_add_fastmath(
      %a: tensor<128x16xi8, #blocked2>,
      %scale_a: tensor<128x2xi8, #blocked1>,
      %b: tensor<16x128xi8, #blocked>,
      %scale_b: tensor<128x2xi8, #blocked1>,
      %c: tensor<128x128xf32, #blocked>) -> tensor<128x128xf32, #blocked> {
    %0 = tt.dot_scaled %a scale %scale_a, %b scale %scale_b, %c lhs = e2m1 rhs = e2m1 {fastMath = true, lhs_k_pack = true, rhs_k_pack = true} : tensor<128x16xi8, #blocked2>, tensor<128x2xi8, #blocked1> * tensor<16x128xi8, #blocked>, tensor<128x2xi8, #blocked1> -> tensor<128x128xf32, #blocked>
    tt.return %0 : tensor<128x128xf32, #blocked>
  }
}
