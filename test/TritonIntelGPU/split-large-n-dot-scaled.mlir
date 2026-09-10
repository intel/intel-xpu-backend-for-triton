// RUN: triton-opt %s -split-input-file \
// RUN:   --tritonintelgpu-split-large-n-dot-scaled='num-warps=32' \
// RUN:   | FileCheck %s

// COM: Tests for the tritonintelgpu-split-large-n-dot-scaled pass.
// COM:
// COM: The pass splits tt.dot_scaled with fp4 B operand along N when the static
// COM: GRF estimate would overflow the 256-GRF budget:
// COM:   (bF32PerThread + accumF32PerThread) * 2 >= 256
// COM: where bF32PerThread = 2 * B.numElements / (numWarps * threadsPerWarp)
// COM:       accumF32PerThread = C.numElements / (numWarps * threadsPerWarp)
// COM:
// COM: The pass fires only when:
// COM:   - B elem type is E2M1 (fp4)
// COM:   - ttig.support_block_scale_dpas is absent (BMG, no hardware BDPAS)
// COM:   - GRF estimate >= 256
// COM:
// COM: Shapes used here with num-warps=32, threads-per-warp=16 (512 threads):
// COM:
// COM:   POSITIVE — 128x256x128 (BM=128, BN=256, BK=128):
// COM:     B = [64, 256] i8 packed fp4; bF32 = 2*64*256/512 = 64
// COM:     C = [128, 256] f32;          accum = 128*256/512 = 64
// COM:     (64+64)*2 = 256 >= 256 → SPLIT
// COM:
// COM:   NEGATIVE #1 — small N (128x128x128):
// COM:     B = [64, 128] i8 packed fp4; bF32 = 2*64*128/512 = 32
// COM:     C = [128, 128] f32;          accum = 128*128/512 = 32
// COM:     (32+32)*2 = 128 < 256 → NO SPLIT
// COM:
// COM:   NEGATIVE #2 — has ttig.support_block_scale_dpas (hardware BDPAS present):
// COM:     Same shapes as positive test but BDPAS attr set → NO SPLIT
// COM:
// COM:   NEGATIVE #3 — B is fp8 (E5M2), not fp4:
// COM:     bF32 is large but B is not E2M1 → NO SPLIT

// ─────────────────────────────────────────────────────────────────────────────
// POSITIVE: 128×256×128 fp8×fp4 — GRF overflows, split fires
// ─────────────────────────────────────────────────────────────────────────────

// CHECK-LABEL: tt.func public @matmul_128x256x128_split
// COM: Original single dot_scaled with [64,256] B must be gone.
// CHECK-NOT: tensor<64x256xi8>
// COM: Two half-N dot_scaled ops with [64,128] B must appear.
// CHECK-COUNT-2: tt.dot_scaled {{.*}}tensor<64x128xi8>
// COM: Reassembly operations must be present.
// CHECK: tt.join
// CHECK: tt.trans
// CHECK: tt.reshape
module attributes {"ttg.threads-per-warp" = 16 : i32,
                   ttig.min_sg_size = 16 : i32,
                   ttig.support_2d_block_io,
                   ttig.support_bfloat16_arithmetic,
                   ttig.support_bfloat16_conversion} {
  tt.func public @matmul_128x256x128_split(
      %a: tensor<128x128xf8E5M2>,
      %b: tensor<64x256xi8>,
      %scale_a: tensor<128x4xi8>,
      %scale_b: tensor<256x4xi8>,
      %c: tensor<128x256xf32>) -> tensor<128x256xf32> {
    %d = tt.dot_scaled %a scale %scale_a, %b scale %scale_b, %c
         lhs = e5m2 rhs = e2m1
         {fastMath = false, lhs_k_pack = true, rhs_k_pack = true}
         : tensor<128x128xf8E5M2>, tensor<128x4xi8>
         * tensor<64x256xi8>, tensor<256x4xi8>
         -> tensor<128x256xf32>
    tt.return %d : tensor<128x256xf32>
  }
}

// -----

// ─────────────────────────────────────────────────────────────────────────────
// NEGATIVE #1: 128×128×128 — GRF fits, no split
// ─────────────────────────────────────────────────────────────────────────────

// CHECK-LABEL: tt.func public @matmul_128x128x128_no_split
// COM: The original dot_scaled must be unchanged.
// CHECK: tt.dot_scaled {{.*}}tensor<64x128xi8>
// COM: No join/reshape reassembly should appear.
// CHECK-NOT: tt.join
module attributes {"ttg.threads-per-warp" = 16 : i32,
                   ttig.min_sg_size = 16 : i32,
                   ttig.support_2d_block_io,
                   ttig.support_bfloat16_arithmetic,
                   ttig.support_bfloat16_conversion} {
  tt.func public @matmul_128x128x128_no_split(
      %a: tensor<128x128xf8E5M2>,
      %b: tensor<64x128xi8>,
      %scale_a: tensor<128x4xi8>,
      %scale_b: tensor<128x4xi8>,
      %c: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %d = tt.dot_scaled %a scale %scale_a, %b scale %scale_b, %c
         lhs = e5m2 rhs = e2m1
         {fastMath = false, lhs_k_pack = true, rhs_k_pack = true}
         : tensor<128x128xf8E5M2>, tensor<128x4xi8>
         * tensor<64x128xi8>, tensor<128x4xi8>
         -> tensor<128x128xf32>
    tt.return %d : tensor<128x128xf32>
  }
}

// -----

// ─────────────────────────────────────────────────────────────────────────────
// NEGATIVE #2: hardware BDPAS present — no split even with large N
// ─────────────────────────────────────────────────────────────────────────────

// CHECK-LABEL: tt.func public @matmul_bdpas_no_split
// COM: ttig.support_block_scale_dpas present → pass is gated off.
// CHECK: tt.dot_scaled {{.*}}tensor<64x256xi8>
// CHECK-NOT: tt.join
module attributes {"ttg.threads-per-warp" = 16 : i32,
                   ttig.min_sg_size = 16 : i32,
                   ttig.support_subgroup_scaled_matrix_multiply_accumulate,
                   ttig.support_2d_block_io} {
  tt.func public @matmul_bdpas_no_split(
      %a: tensor<128x128xf8E5M2>,
      %b: tensor<64x256xi8>,
      %scale_a: tensor<128x4xi8>,
      %scale_b: tensor<256x4xi8>,
      %c: tensor<128x256xf32>) -> tensor<128x256xf32> {
    %d = tt.dot_scaled %a scale %scale_a, %b scale %scale_b, %c
         lhs = e5m2 rhs = e2m1
         {fastMath = false, lhs_k_pack = true, rhs_k_pack = true}
         : tensor<128x128xf8E5M2>, tensor<128x4xi8>
         * tensor<64x256xi8>, tensor<256x4xi8>
         -> tensor<128x256xf32>
    tt.return %d : tensor<128x256xf32>
  }
}

// -----

// ─────────────────────────────────────────────────────────────────────────────
// NEGATIVE #3: B is fp8 (E5M2), not fp4 (E2M1) — no split
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
// POSITIVE #2: 256×128×128 with A=fp4, B=fp8 — M-dimension GRF overflow
// ─────────────────────────────────────────────────────────────────────────────
// COM: A=fp4 [256,64] i8 (M=256, K=128, lhs_k_pack=True):
// COM:   aF32 = 2*256*64/512 = 64; accum = 256*128/512 = 64
// COM:   (64+64)*2 = 256 >= 256 → SPLIT along M

// CHECK-LABEL: tt.func public @matmul_256x128x128_afp4_split
// COM: Original single dot_scaled with [256,64] A must be gone.
// CHECK-NOT: tensor<256x64xi8>
// COM: Two half-M dot_scaled ops with [128,64] A must appear.
// CHECK-COUNT-2: tt.dot_scaled {{.*}}tensor<128x64xi8>
// COM: Reassembly operations must be present.
// CHECK: tt.join
// CHECK: tt.trans
// CHECK: tt.reshape
module attributes {"ttg.threads-per-warp" = 16 : i32,
                   ttig.min_sg_size = 16 : i32,
                   ttig.support_2d_block_io,
                   ttig.support_bfloat16_arithmetic,
                   ttig.support_bfloat16_conversion} {
  // A=fp4 packed: [M, K/2] = [256, 64] i8 with lhs_k_pack=true
  // B=fp8: [K, N] = [128, 128] f8E5M2 with rhs_k_pack=false
  tt.func public @matmul_256x128x128_afp4_split(
      %a: tensor<256x64xi8>,
      %b: tensor<128x128xf8E5M2>,
      %scale_a: tensor<256x4xi8>,
      %scale_b: tensor<128x4xi8>,
      %c: tensor<256x128xf32>) -> tensor<256x128xf32> {
    %d = tt.dot_scaled %a scale %scale_a, %b scale %scale_b, %c
         lhs = e2m1 rhs = e5m2
         {fastMath = false, lhs_k_pack = true, rhs_k_pack = false}
         : tensor<256x64xi8>, tensor<256x4xi8>
         * tensor<128x128xf8E5M2>, tensor<128x4xi8>
         -> tensor<256x128xf32>
    tt.return %d : tensor<256x128xf32>
  }
}

// -----

// ─────────────────────────────────────────────────────────────────────────────
// NEGATIVE #4: A=fp4 but small M (128×128×128) — GRF fits, no split
// ─────────────────────────────────────────────────────────────────────────────

// CHECK-LABEL: tt.func public @matmul_128x128x128_afp4_no_split
// COM: A=fp4 [128,64], B=fp8 [128,128]: aF32=32, accum=32 → (32+32)*2=128 < 256
// CHECK: tt.dot_scaled
// CHECK-NOT: tt.join
module attributes {"ttg.threads-per-warp" = 16 : i32,
                   ttig.min_sg_size = 16 : i32,
                   ttig.support_2d_block_io} {
  tt.func public @matmul_128x128x128_afp4_no_split(
      %a: tensor<128x64xi8>,
      %b: tensor<128x128xf8E5M2>,
      %scale_a: tensor<128x4xi8>,
      %scale_b: tensor<128x4xi8>,
      %c: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %d = tt.dot_scaled %a scale %scale_a, %b scale %scale_b, %c
         lhs = e2m1 rhs = e5m2
         {fastMath = false, lhs_k_pack = true, rhs_k_pack = false}
         : tensor<128x64xi8>, tensor<128x4xi8>
         * tensor<128x128xf8E5M2>, tensor<128x4xi8>
         -> tensor<128x128xf32>
    tt.return %d : tensor<128x128xf32>
  }
}

// -----

// CHECK-LABEL: tt.func public @matmul_fp8_b_no_split
// COM: B elem type is E5M2, not E2M1 → pass does not fire.
// CHECK: tt.dot_scaled
// CHECK-NOT: tt.join
module attributes {"ttg.threads-per-warp" = 16 : i32,
                   ttig.min_sg_size = 16 : i32,
                   ttig.support_2d_block_io} {
  // A=[128,128] fp8, B=[128,256] fp8 (not packed, not fp4), C=[128,256]
  tt.func public @matmul_fp8_b_no_split(
      %a: tensor<128x128xf8E5M2>,
      %b: tensor<128x256xf8E5M2>,
      %scale_a: tensor<128x4xi8>,
      %scale_b: tensor<256x4xi8>,
      %c: tensor<128x256xf32>) -> tensor<128x256xf32> {
    %d = tt.dot_scaled %a scale %scale_a, %b scale %scale_b, %c
         lhs = e5m2 rhs = e5m2
         {fastMath = false, lhs_k_pack = false, rhs_k_pack = false}
         : tensor<128x128xf8E5M2>, tensor<128x4xi8>
         * tensor<128x256xf8E5M2>, tensor<256x4xi8>
         -> tensor<128x256xf32>
    tt.return %d : tensor<128x256xf32>
  }
}
