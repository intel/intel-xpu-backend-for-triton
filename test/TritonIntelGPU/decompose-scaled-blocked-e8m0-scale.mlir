// RUN: triton-opt %s -split-input-file --tritonintelgpu-accelerate-matmul | FileCheck %s

// COM: Regression test for native f8E8M0FNU scales in DecomposeScaledBlocked.
// COM:
// COM: Upstream triton-lang/triton#12315 made f8E8M0FNU a legal tt.dot_scaled
// COM: scale type. Float8E8M0FNUType is a FloatType, so scaleTo16 and maskNan
// COM: used to send it down their float branches and emit
// COM: `tt.fp_to_fp f8E8M0FNU -> bf16`, which the Intel FpToFp lowering cannot
// COM: handle. Both helpers must instead bitcast the scale to i8 and reuse the
// COM: integer path: shift the byte into the exponent field, and compare the raw
// COM: byte against 0xFF for the NaN mask.
// COM:
// COM: The module deliberately omits ttig.support_subgroup_scaled_matrix_multiply_accumulate
// COM: so that AccelerateMatmul's BDPAS-aware patterns are gated off and
// COM: DecomposeScaledBlocked owns the lowering of tt.dot_scaled.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 8], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 4], warpsPerCTA = [16, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 4], order = [1, 0]}>

module attributes {ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func public @kernel_e8m0_scale
  // COM: Constants are hoisted to the function entry; 0xFF prints as -1 for i8.
  // CHECK: %[[FF:.*]] = arith.constant dense<-1> : tensor<128x4xi8, #{{.*}}>
  // CHECK-NOT: tt.fp_to_fp {{.*}}f8E8M0FNU
  // CHECK-NOT: arith.cmpf
  // COM: scaleTo16: bitcast to i8, then the integer decode path.
  // CHECK: %[[S:.*]] = tt.bitcast %[[SCALE:.*]] : tensor<128x4xf8E8M0FNU, #{{.*}}> -> tensor<128x4xi8, #{{.*}}>
  // CHECK: arith.extui %[[S]] : tensor<128x4xi8, #{{.*}}> to tensor<128x4xi16, #{{.*}}>
  // CHECK-NOT: tt.fp_to_fp {{.*}}f8E8M0FNU
  // CHECK-NOT: arith.cmpf
  // COM: maskNan: bitcast to i8, then compare the raw byte against 0xFF.
  // CHECK: %[[M:.*]] = tt.bitcast %[[SCALE]] : tensor<128x4xf8E8M0FNU, #{{.*}}> -> tensor<128x4xi8, #{{.*}}>
  // CHECK: arith.cmpi eq, %[[M]], %[[FF]] : tensor<128x4xi8, #{{.*}}>
  // CHECK-NOT: tt.fp_to_fp {{.*}}f8E8M0FNU
  // CHECK-NOT: arith.cmpf
  // CHECK: tt.dot
  // CHECK-NOT: tt.fp_to_fp {{.*}}f8E8M0FNU
  // CHECK-NOT: arith.cmpf
  // CHECK: tt.return
  tt.func public @kernel_e8m0_scale(%a: tensor<128x128xf8E4M3FN, #blocked2>, %scale_a: tensor<128x4xf8E8M0FNU, #blocked1>, %b: tensor<128x128xbf16, #blocked>) -> tensor<128x128xf32, #blocked> {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %0 = tt.dot_scaled %a scale %scale_a, %b, %cst lhs = e4m3 rhs = bf16 {fastMath = false} : tensor<128x128xf8E4M3FN, #blocked2>, tensor<128x4xf8E8M0FNU, #blocked1> * tensor<128x128xbf16, #blocked> -> tensor<128x128xf32, #blocked>
    tt.return %0 : tensor<128x128xf32, #blocked>
  }
}
