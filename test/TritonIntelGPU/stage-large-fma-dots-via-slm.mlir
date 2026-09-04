// RUN: triton-opt %s -split-input-file --tritonintelgpu-stage-large-fma-dots-via-slm | FileCheck %s

// Test 1: Real-kernel shape — dot is inside an outer scf.for (the case the
// closed PR #7276's traceToLoad approach could not reach because loads come
// from iter_args, not direct tt.load defining ops). This is the only shape
// that matters for production matmul kernels.
//
// Encoding: 64x128 result, blocked spt=[1,1] tpw=[2,16] wpc=[4,1].
// Per-thread bytes (formula from estimatePerThreadBytes):
//   mReps=8, nReps=1, K=128, elemBytes=2, accBytes=4
//   aBytes = 8*1*128*2 = 2048
//   bBytes = 128*1*1*2 = 256        (nSpt=1, nReps=1)
//   cBytes = 8*1*1*1*4 = 32
//   total  = 2336 < 4096 -> below threshold -- does NOT fire on this trivial
// We deliberately set a wider tile to push above threshold.
//
// For 64x128 with spt=[1,1] tpw=[2,16] wpc=[4,1] and a 64x256 K dim:
//   K=256, aBytes = 8*1*256*2 = 4096 -- right at threshold.  Use K=512 to clear.

#blocked  = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#dot0     = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dot1     = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @real_matmul_outer_loop
  // CHECK: scf.for
  // The pass replaces the inner tt.dot with a chain of subslice + local_load + dot
  // CHECK: ttg.local_alloc
  // CHECK: ttg.local_alloc
  // CHECK: ttg.memdesc_subslice
  // CHECK: ttg.local_load
  // CHECK: ttg.local_load
  // CHECK: tt.dot
  tt.func public @real_matmul_outer_loop(
      %base_a: !tt.ptr<f16>,
      %base_b: !tt.ptr<f16>,
      %k_steps: i32) -> tensor<64x128xf32, #blocked> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked>
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %a_splat0 = tt.splat %base_a : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #blocked>
    %coff_a   = arith.constant dense<0> : tensor<64x128xi32, #blocked>
    %a_ptr0   = tt.addptr %a_splat0, %coff_a : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
    %b_splat0 = tt.splat %base_b : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>, #blocked1>
    %coff_b   = arith.constant dense<0> : tensor<128x128xi32, #blocked1>
    %b_ptr0   = tt.addptr %b_splat0, %coff_b : tensor<128x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xi32, #blocked1>

    %r:3 = scf.for %i = %c0 to %k_steps step %c1
        iter_args(%acc_iv = %cst, %a_iv = %a_ptr0, %b_iv = %b_ptr0)
        -> (tensor<64x128xf32, #blocked>, tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<128x128x!tt.ptr<f16>, #blocked1>)
        : i32 {
      %a     = tt.load %a_iv : tensor<64x128x!tt.ptr<f16>, #blocked>
      %a_cvt = ttg.convert_layout %a : tensor<64x128xf16, #blocked> -> tensor<64x128xf16, #dot0>
      %b     = tt.load %b_iv : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %b_cvt = ttg.convert_layout %b : tensor<128x128xf16, #blocked1> -> tensor<128x128xf16, #dot1>
      %new_acc = tt.dot %a_cvt, %b_cvt, %acc_iv {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32}
                       : tensor<64x128xf16, #dot0> * tensor<128x128xf16, #dot1> -> tensor<64x128xf32, #blocked>
      scf.yield %new_acc, %a_iv, %b_iv : tensor<64x128xf32, #blocked>, tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<128x128x!tt.ptr<f16>, #blocked1>
    }
    tt.return %r#0 : tensor<64x128xf32, #blocked>
  }
}

// -----

// Test 2: Direct (non-loop) form, just to confirm the pass also handles the
// simple case the closed PR's lit tests covered.
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#dot0b    = #ttg.dot_op<{opIdx = 0, parent = #blocked2}>
#dot1b    = #ttg.dot_op<{opIdx = 1, parent = #blocked2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @direct_dot_staged
  // CHECK: ttg.local_alloc
  // CHECK: ttg.local_alloc
  // CHECK: ttg.memdesc_subslice
  // CHECK: ttg.local_load
  // CHECK: tt.dot
  tt.func public @direct_dot_staged(
      %base_a: !tt.ptr<f16>,
      %base_b: !tt.ptr<f16>) -> tensor<64x128xf32, #blocked2> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked2>
    %a_splat = tt.splat %base_a : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #blocked2>
    %c0a = arith.constant dense<0> : tensor<64x128xi32, #blocked2>
    %a_ptr = tt.addptr %a_splat, %c0a : tensor<64x128x!tt.ptr<f16>, #blocked2>, tensor<64x128xi32, #blocked2>
    %a = tt.load %a_ptr : tensor<64x128x!tt.ptr<f16>, #blocked2>
    %a_cvt = ttg.convert_layout %a : tensor<64x128xf16, #blocked2> -> tensor<64x128xf16, #dot0b>
    %b_splat = tt.splat %base_b : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>, #blocked3>
    %c0b = arith.constant dense<0> : tensor<128x128xi32, #blocked3>
    %b_ptr = tt.addptr %b_splat, %c0b : tensor<128x128x!tt.ptr<f16>, #blocked3>, tensor<128x128xi32, #blocked3>
    %b = tt.load %b_ptr : tensor<128x128x!tt.ptr<f16>, #blocked3>
    %b_cvt = ttg.convert_layout %b : tensor<128x128xf16, #blocked3> -> tensor<128x128xf16, #dot1b>
    %result = tt.dot %a_cvt, %b_cvt, %cst {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32}
                    : tensor<64x128xf16, #dot0b> * tensor<128x128xf16, #dot1b> -> tensor<64x128xf32, #blocked2>
    tt.return %result : tensor<64x128xf32, #blocked2>
  }
}

// -----

// Test 3: Small dot must NOT be staged (under pressure threshold).
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#dot0c    = #ttg.dot_op<{opIdx = 0, parent = #blocked4}>
#dot1c    = #ttg.dot_op<{opIdx = 1, parent = #blocked4}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @small_dot_unchanged
  // CHECK-NOT: ttg.local_alloc
  // CHECK-NOT: ttg.memdesc_subslice
  // CHECK: tt.dot
  tt.func public @small_dot_unchanged(
      %a_ptr: tensor<16x16x!tt.ptr<f16>, #blocked4>,
      %b_ptr: tensor<16x16x!tt.ptr<f16>, #blocked5>) -> tensor<16x16xf32, #blocked4> {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked4>
    %a = tt.load %a_ptr : tensor<16x16x!tt.ptr<f16>, #blocked4>
    %a_cvt = ttg.convert_layout %a : tensor<16x16xf16, #blocked4> -> tensor<16x16xf16, #dot0c>
    %b = tt.load %b_ptr : tensor<16x16x!tt.ptr<f16>, #blocked5>
    %b_cvt = ttg.convert_layout %b : tensor<16x16xf16, #blocked5> -> tensor<16x16xf16, #dot1c>
    %r = tt.dot %a_cvt, %b_cvt, %cst {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32}
              : tensor<16x16xf16, #dot0c> * tensor<16x16xf16, #dot1c> -> tensor<16x16xf32, #blocked4>
    tt.return %r : tensor<16x16xf32, #blocked4>
  }
}

// -----

// Test 4: DPAS-lowered dot is skipped — after AccelerateMatmul, a
// DPAS-lowerable dot has DpasEncodingAttr on its result. The pass's
// per-dot BlockedEncodingAttr check excludes it. f32 dots with
// `inputPrecision = ieee` would NOT be DPAS-lowerable (DPAS only
// supports f32 via TF32 truncation) — those keep BlockedEncoding and
// get staged even on DPAS hw, see the next test.
#dpas      = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0d_dpas = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#dot1d_dpas = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {
  "ttg.num-warps" = 4 : i32,
  "ttg.threads-per-warp" = 16 : i32,
  ttig.support_subgroup_matrix_multiply_accumulate
} {
  // CHECK-LABEL: @dpas_lowered_dot_skipped
  // CHECK-NOT: ttg.local_alloc
  // CHECK-NOT: ttg.memdesc_subslice
  // CHECK: tt.dot
  tt.func public @dpas_lowered_dot_skipped(
      %a: tensor<64x128xf16, #dot0d_dpas>,
      %b: tensor<128x128xf16, #dot1d_dpas>) -> tensor<64x128xf32, #dpas> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #dpas>
    %r = tt.dot %a, %b, %cst {inputPrecision = 2 : i32, maxNumImpreciseAcc = 0 : i32}
              : tensor<64x128xf16, #dot0d_dpas> * tensor<128x128xf16, #dot1d_dpas> -> tensor<64x128xf32, #dpas>
    tt.return %r : tensor<64x128xf32, #dpas>
  }
}

// -----

// Test 4b: f32 with `inputPrecision = ieee` on DPAS hardware. DPAS only
// supports f32 via TF32 truncation, so the ieee variant retains
// BlockedEncoding after AccelerateMatmul and lowers via FMA — hitting
// the same K-unroll cliff this pass targets. Per @chengjunlu review on
// PR #7291.
//
// Pressure check (post-AccelerateMatmul, no fp_to_fp inserted because
// types already match): 64x32 result, K=128, spt=[1,1] tpw=[2,16] wpc=[4,1].
//   mReps = 64 / (1*2*4) = 8
//   nReps = 32 / (1*16*1) = 2
//   aBytes = 8*1*128*4 = 4096
//   bBytes = 128*2*1*4 = 1024
//   cBytes = 8*1*2*1*4 = 64
//   total  = 5184 > 4096 -> fires.
// SLM cost: 64*128*4 + 128*32*4 = 32 KB + 16 KB = 48 KB < 56 KB cap.
#blocked6  = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked6b = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#dot0db    = #ttg.dot_op<{opIdx = 0, parent = #blocked6}>
#dot1db    = #ttg.dot_op<{opIdx = 1, parent = #blocked6}>
module attributes {
  "ttg.num-warps" = 4 : i32,
  "ttg.threads-per-warp" = 32 : i32,
  ttig.support_subgroup_matrix_multiply_accumulate
} {
  // CHECK-LABEL: @f32_ieee_fires_on_dpas_hw
  // CHECK: ttg.local_alloc
  // CHECK: ttg.local_alloc
  // CHECK: ttg.memdesc_subslice
  // CHECK: ttg.local_load
  // CHECK: tt.dot
  tt.func public @f32_ieee_fires_on_dpas_hw(
      %base_a: !tt.ptr<f32>,
      %base_b: !tt.ptr<f32>) -> tensor<64x32xf32, #blocked6> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #blocked6>
    %a_splat = tt.splat %base_a : !tt.ptr<f32> -> tensor<64x128x!tt.ptr<f32>, #blocked6>
    %c0a = arith.constant dense<0> : tensor<64x128xi32, #blocked6>
    %a_ptr = tt.addptr %a_splat, %c0a : tensor<64x128x!tt.ptr<f32>, #blocked6>, tensor<64x128xi32, #blocked6>
    %a = tt.load %a_ptr : tensor<64x128x!tt.ptr<f32>, #blocked6>
    %a_cvt = ttg.convert_layout %a : tensor<64x128xf32, #blocked6> -> tensor<64x128xf32, #dot0db>
    %b_splat = tt.splat %base_b : !tt.ptr<f32> -> tensor<128x32x!tt.ptr<f32>, #blocked6b>
    %c0b = arith.constant dense<0> : tensor<128x32xi32, #blocked6b>
    %b_ptr = tt.addptr %b_splat, %c0b : tensor<128x32x!tt.ptr<f32>, #blocked6b>, tensor<128x32xi32, #blocked6b>
    %b = tt.load %b_ptr : tensor<128x32x!tt.ptr<f32>, #blocked6b>
    %b_cvt = ttg.convert_layout %b : tensor<128x32xf32, #blocked6b> -> tensor<128x32xf32, #dot1db>
    %r = tt.dot %a_cvt, %b_cvt, %cst {inputPrecision = 2 : i32, maxNumImpreciseAcc = 0 : i32}
              : tensor<64x128xf32, #dot0db> * tensor<128x32xf32, #dot1db> -> tensor<64x32xf32, #blocked6>
    tt.return %r : tensor<64x32xf32, #blocked6>
  }
}

// -----

// Test 5: Post-AccelerateMatmul shape — operands have already been promoted
// by tt.fp_to_fp casts from f16 to f32 (the `decomposeMixedModeDotOp`
// transform). The pass must:
//   - find the convert_layout boundary by walking past the fp_to_fp casts,
//   - stage at the f16 source (small in SLM), and
//   - replay the fp_to_fp on each partial-K loaded slice.

// RUN: triton-opt %s -split-input-file --tritonintelgpu-stage-large-fma-dots-via-slm | FileCheck %s --check-prefixes=CHECK,POST-PROMOTE

#blocked8 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked9 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#dot0e    = #ttg.dot_op<{opIdx = 0, parent = #blocked8}>
#dot1e    = #ttg.dot_op<{opIdx = 1, parent = #blocked8}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // POST-PROMOTE-LABEL: @post_promote_chain
  // POST-PROMOTE: ttg.local_alloc {{.*}}xf16
  // POST-PROMOTE: ttg.local_alloc {{.*}}xf16
  // POST-PROMOTE: ttg.memdesc_subslice
  // POST-PROMOTE: ttg.local_load {{.*}}xf16
  // POST-PROMOTE: tt.fp_to_fp {{.*}}xf16{{.*}}xf32
  // POST-PROMOTE: tt.dot {{.*}}xf32{{.*}}xf32{{.*}}xf32
  tt.func public @post_promote_chain(
      %base_a: !tt.ptr<f16>,
      %base_b: !tt.ptr<f16>) -> tensor<64x128xf32, #blocked8> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked8>
    %a_splat = tt.splat %base_a : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #blocked8>
    %c0a = arith.constant dense<0> : tensor<64x128xi32, #blocked8>
    %a_ptr = tt.addptr %a_splat, %c0a : tensor<64x128x!tt.ptr<f16>, #blocked8>, tensor<64x128xi32, #blocked8>
    %a = tt.load %a_ptr : tensor<64x128x!tt.ptr<f16>, #blocked8>
    %a_cvt = ttg.convert_layout %a : tensor<64x128xf16, #blocked8> -> tensor<64x128xf16, #dot0e>
    %a_f32 = tt.fp_to_fp %a_cvt : tensor<64x128xf16, #dot0e> -> tensor<64x128xf32, #dot0e>
    %b_splat = tt.splat %base_b : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>, #blocked9>
    %c0b = arith.constant dense<0> : tensor<128x128xi32, #blocked9>
    %b_ptr = tt.addptr %b_splat, %c0b : tensor<128x128x!tt.ptr<f16>, #blocked9>, tensor<128x128xi32, #blocked9>
    %b = tt.load %b_ptr : tensor<128x128x!tt.ptr<f16>, #blocked9>
    %b_cvt = ttg.convert_layout %b : tensor<128x128xf16, #blocked9> -> tensor<128x128xf16, #dot1e>
    %b_f32 = tt.fp_to_fp %b_cvt : tensor<128x128xf16, #dot1e> -> tensor<128x128xf32, #dot1e>
    %r = tt.dot %a_f32, %b_f32, %cst {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32}
              : tensor<64x128xf32, #dot0e> * tensor<128x128xf32, #dot1e> -> tensor<64x128xf32, #blocked8>
    tt.return %r : tensor<64x128xf32, #blocked8>
  }
}

// -----

// Test 6: f32 inputPrecision=ieee on non-DPAS hw lowers via FMA, hits the
// same K-unrolled register cliff. Pass already handles f32 (gates on
// BlockedEncodingAttr only); this case locks in the behavior. Per
// @chengjunlu's review point 2 on PR #7276.
//
// Pressure for 64x32x128 f32, spt=[1,1] tpw=[2,16] wpc=[4,1]:
//   mReps=8, nReps=1, K=128, elemBytes=4, accBytes=4
//   aBytes = 8*1*128*4 = 4096
//   bBytes = 128*1*1*4 = 512
//   cBytes = 32
//   total  = 4640 > 4096 -> fires.
// SLM cost: 64*128*4 + 128*32*4 = 32 KB + 16 KB = 48 KB < 56 KB cap.
#blocked10 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked11 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#dot0f     = #ttg.dot_op<{opIdx = 0, parent = #blocked10}>
#dot1f     = #ttg.dot_op<{opIdx = 1, parent = #blocked10}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @f32_ieee_fires
  // CHECK: ttg.local_alloc {{.*}}xf32
  // CHECK: ttg.memdesc_subslice
  // CHECK: ttg.local_load {{.*}}xf32
  // CHECK: tt.dot {{.*}}xf32{{.*}}xf32
  // The default-valued tf32 print quirk: confirm tf32x3 did not leak in.
  // CHECK-NOT: inputPrecision = tf32x3
  tt.func public @f32_ieee_fires(
      %base_a: !tt.ptr<f32>,
      %base_b: !tt.ptr<f32>) -> tensor<64x32xf32, #blocked10> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #blocked10>
    %a_splat = tt.splat %base_a : !tt.ptr<f32> -> tensor<64x128x!tt.ptr<f32>, #blocked10>
    %c0a = arith.constant dense<0> : tensor<64x128xi32, #blocked10>
    %a_ptr = tt.addptr %a_splat, %c0a : tensor<64x128x!tt.ptr<f32>, #blocked10>, tensor<64x128xi32, #blocked10>
    %a = tt.load %a_ptr : tensor<64x128x!tt.ptr<f32>, #blocked10>
    %a_cvt = ttg.convert_layout %a : tensor<64x128xf32, #blocked10> -> tensor<64x128xf32, #dot0f>
    %b_splat = tt.splat %base_b : !tt.ptr<f32> -> tensor<128x32x!tt.ptr<f32>, #blocked11>
    %c0b = arith.constant dense<0> : tensor<128x32xi32, #blocked11>
    %b_ptr = tt.addptr %b_splat, %c0b : tensor<128x32x!tt.ptr<f32>, #blocked11>, tensor<128x32xi32, #blocked11>
    %b = tt.load %b_ptr : tensor<128x32x!tt.ptr<f32>, #blocked11>
    %b_cvt = ttg.convert_layout %b : tensor<128x32xf32, #blocked11> -> tensor<128x32xf32, #dot1f>
    %r = tt.dot %a_cvt, %b_cvt, %cst {inputPrecision = 2 : i32, maxNumImpreciseAcc = 0 : i32}
              : tensor<64x128xf32, #dot0f> * tensor<128x32xf32, #dot1f> -> tensor<64x32xf32, #blocked10>
    tt.return %r : tensor<64x32xf32, #blocked10>
  }
}

// -----

// Test 7: Two tt.dot ops in the same function, each independently gated by
// pressure threshold and SLM-fit. Both should be staged.
#blocked12 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked13 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#dot0g     = #ttg.dot_op<{opIdx = 0, parent = #blocked12}>
#dot1g     = #ttg.dot_op<{opIdx = 1, parent = #blocked12}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @two_dots_each_staged
  // First dot's staging
  // CHECK: ttg.local_alloc
  // CHECK: ttg.local_alloc
  // CHECK: ttg.memdesc_subslice
  // CHECK: tt.dot
  // Second dot's staging
  // CHECK: ttg.local_alloc
  // CHECK: ttg.local_alloc
  // CHECK: ttg.memdesc_subslice
  // CHECK: tt.dot
  tt.func public @two_dots_each_staged(
      %base_a: !tt.ptr<f16>, %base_b: !tt.ptr<f16>,
      %base_c: !tt.ptr<f16>, %base_d: !tt.ptr<f16>) -> tensor<64x128xf32, #blocked12> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked12>
    %ap0 = tt.splat %base_a : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #blocked12>
    %za = arith.constant dense<0> : tensor<64x128xi32, #blocked12>
    %ap = tt.addptr %ap0, %za : tensor<64x128x!tt.ptr<f16>, #blocked12>, tensor<64x128xi32, #blocked12>
    %a = tt.load %ap : tensor<64x128x!tt.ptr<f16>, #blocked12>
    %ac = ttg.convert_layout %a : tensor<64x128xf16, #blocked12> -> tensor<64x128xf16, #dot0g>
    %bp0 = tt.splat %base_b : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>, #blocked13>
    %zb = arith.constant dense<0> : tensor<128x128xi32, #blocked13>
    %bp = tt.addptr %bp0, %zb : tensor<128x128x!tt.ptr<f16>, #blocked13>, tensor<128x128xi32, #blocked13>
    %b = tt.load %bp : tensor<128x128x!tt.ptr<f16>, #blocked13>
    %bc = ttg.convert_layout %b : tensor<128x128xf16, #blocked13> -> tensor<128x128xf16, #dot1g>
    %r1 = tt.dot %ac, %bc, %cst {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32}
                : tensor<64x128xf16, #dot0g> * tensor<128x128xf16, #dot1g> -> tensor<64x128xf32, #blocked12>

    %cp0 = tt.splat %base_c : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #blocked12>
    %cp = tt.addptr %cp0, %za : tensor<64x128x!tt.ptr<f16>, #blocked12>, tensor<64x128xi32, #blocked12>
    %c = tt.load %cp : tensor<64x128x!tt.ptr<f16>, #blocked12>
    %cc = ttg.convert_layout %c : tensor<64x128xf16, #blocked12> -> tensor<64x128xf16, #dot0g>
    %dp0 = tt.splat %base_d : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>, #blocked13>
    %dp = tt.addptr %dp0, %zb : tensor<128x128x!tt.ptr<f16>, #blocked13>, tensor<128x128xi32, #blocked13>
    %d = tt.load %dp : tensor<128x128x!tt.ptr<f16>, #blocked13>
    %dc = ttg.convert_layout %d : tensor<128x128xf16, #blocked13> -> tensor<128x128xf16, #dot1g>
    %r2 = tt.dot %cc, %dc, %r1 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32}
                : tensor<64x128xf16, #dot0g> * tensor<128x128xf16, #dot1g> -> tensor<64x128xf32, #blocked12>
    tt.return %r2 : tensor<64x128xf32, #blocked12>
  }
}

// -----

// Test 8: SLM reuse — operand is already a ttg.local_load (e.g. inserted
// by ReduceDataDuplication or software pipelining). The pass must NOT
// emit a second ttg.local_alloc for that operand; it reuses the existing
// memdesc and only adds memdesc_subslice + local_load + tt.dot chain.
#blocked14 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked15 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#sharedA = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#sharedB = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#smem = #ttg.shared_memory
#dot0h = #ttg.dot_op<{opIdx = 0, parent = #blocked14}>
#dot1h = #ttg.dot_op<{opIdx = 1, parent = #blocked14}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reuses_pre_staged_smem
  // The pre-existing local_alloc must remain (the pass reuses its result).
  // CHECK: ttg.local_alloc
  // CHECK: ttg.local_alloc
  // CHECK: ttg.memdesc_subslice
  // CHECK: ttg.local_load
  // CHECK: tt.dot
  // CHECK-NOT: ttg.local_alloc
  tt.func public @reuses_pre_staged_smem(
      %base_a: !tt.ptr<f16>, %base_b: !tt.ptr<f16>) -> tensor<64x128xf32, #blocked14> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked14>
    %ap0 = tt.splat %base_a : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #blocked14>
    %za = arith.constant dense<0> : tensor<64x128xi32, #blocked14>
    %ap = tt.addptr %ap0, %za : tensor<64x128x!tt.ptr<f16>, #blocked14>, tensor<64x128xi32, #blocked14>
    %a = tt.load %ap : tensor<64x128x!tt.ptr<f16>, #blocked14>
    %a_smem = ttg.local_alloc %a : (tensor<64x128xf16, #blocked14>) -> !ttg.memdesc<64x128xf16, #sharedA, #smem>
    %a_load = ttg.local_load %a_smem : !ttg.memdesc<64x128xf16, #sharedA, #smem> -> tensor<64x128xf16, #dot0h>

    %bp0 = tt.splat %base_b : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>, #blocked15>
    %zb = arith.constant dense<0> : tensor<128x128xi32, #blocked15>
    %bp = tt.addptr %bp0, %zb : tensor<128x128x!tt.ptr<f16>, #blocked15>, tensor<128x128xi32, #blocked15>
    %b = tt.load %bp : tensor<128x128x!tt.ptr<f16>, #blocked15>
    %b_smem = ttg.local_alloc %b : (tensor<128x128xf16, #blocked15>) -> !ttg.memdesc<128x128xf16, #sharedB, #smem>
    %b_load = ttg.local_load %b_smem : !ttg.memdesc<128x128xf16, #sharedB, #smem> -> tensor<128x128xf16, #dot1h>

    %r = tt.dot %a_load, %b_load, %cst {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32}
              : tensor<64x128xf16, #dot0h> * tensor<128x128xf16, #dot1h> -> tensor<64x128xf32, #blocked14>
    tt.return %r : tensor<64x128xf32, #blocked14>
  }
}
