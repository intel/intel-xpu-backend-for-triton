// RUN: triton-opt %s -split-input-file --tritonintelgpu-accelerate-matmul | FileCheck %s --implicit-check-not='tt.fp_to_fp {{.*}}f8E8M0FNU' --implicit-check-not='arith.select'

// COM: Regression test for native f8E8M0FNU scales (#8556).
// COM:
// COM: Upstream triton-lang/triton#12315 made f8E8M0FNU a legal tt.dot_scaled
// COM: scale type. The Intel backend used to carry its own copy of the
// COM: decomposition, which sent the FloatType scale down a float branch and
// COM: emitted `tt.fp_to_fp f8E8M0FNU -> bf16`, a conversion the Intel FpToFp
// COM: lowering has no entry for. The Intel pattern now subclasses upstream's,
// COM: whose decode bitcasts the scale to i8, shifts the byte into the exponent
// COM: field, and with fastMath off applies fma(s, 0, s), which keeps finite
// COM: scales and turns the +inf that byte 0xFF decodes to into NaN. There is
// COM: no compare-and-select mask any more (hence the implicit check).
// COM:
// COM: The module deliberately omits ttig.support_subgroup_scaled_matrix_multiply_accumulate
// COM: so that AccelerateMatmul's BDPAS-aware patterns are gated off and
// COM: DecomposeScaledBlocked owns the lowering of tt.dot_scaled.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 8], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 4], warpsPerCTA = [16, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 4], order = [1, 0]}>

module attributes {ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func public @kernel_e8m0_scale
  // CHECK: %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : tensor<128x4xbf16, #{{.*}}>
  // COM: The native E8M0 scale is reinterpreted as its i8 bits and decoded on the integer path.
  // CHECK: %[[S:.*]] = tt.bitcast %{{.*}} : tensor<128x4xf8E8M0FNU, #{{.*}}> -> tensor<128x4xi8, #{{.*}}>
  // CHECK: arith.extui %[[S]] : tensor<128x4xi8, #{{.*}}> to tensor<128x4xi16, #{{.*}}>
  // CHECK: %[[S16:.*]] = tt.bitcast %{{.*}} : tensor<128x4xi16, #{{.*}}> -> tensor<128x4xbf16, #{{.*}}>
  // CHECK: math.fma %[[S16]], %[[ZERO]], %[[S16]] : tensor<128x4xbf16, #{{.*}}>
  // CHECK: tt.dot
  // CHECK: tt.return
  tt.func public @kernel_e8m0_scale(%a: tensor<128x128xf8E4M3FN, #blocked2>, %scale_a: tensor<128x4xf8E8M0FNU, #blocked1>, %b: tensor<128x128xbf16, #blocked>) -> tensor<128x128xf32, #blocked> {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %0 = tt.dot_scaled %a scale %scale_a, %b, %cst lhs = e4m3 rhs = bf16 {fastMath = false} : tensor<128x128xf8E4M3FN, #blocked2>, tensor<128x4xf8E8M0FNU, #blocked1> * tensor<128x128xbf16, #blocked> -> tensor<128x128xf32, #blocked>
    tt.return %0 : tensor<128x128xf32, #blocked>
  }
}
