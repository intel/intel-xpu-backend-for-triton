// RUN: triton-opt %s -split-input-file --tritonintelgpu-cooperative-fma-dots | FileCheck %s

// =============================================================================
// TEST MATRIX: Evaluate cooperative shuffle effectiveness across configurations
// =============================================================================

// -----
// Config 1: ARL-S real crashing config (64x128, K=128, f16)
// Encoding: spt=[1,1], tpw=[2,16], wpc=[4,1]
// sharingGroupA = tpw[N] = 16, sharingGroupB = tpw[M] = 2
// mReps = 64/(1*2*4) = 8, nReps = 128/(1*16*1) = 8
// origPressure = 8*1*128*2 + 128*8*1*2 + 8*8*4 = 4352
// shuffleRatio = (8+8)/(8*8*128) = 0.00195
// With kChunk=32: sharedPressure = 8*1*32*2 + 32*8*1*2 + 256 = 1280
// pressureReduction = 4352/1280 = 3.4x
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @arls_crash_config
  // CHECK: tt.dot {{.*}} {ttig.cooperative_sharing = {
  // CHECK-SAME: kChunk = 32 : i32
  // CHECK-SAME: sharingGroupA = 16 : i32
  // CHECK-SAME: sharingGroupB = 2 : i32
  tt.func public @arls_crash_config(
      %a: tensor<64x128xf16, #dot0>,
      %b: tensor<128x128xf16, #dot1>) -> tensor<64x128xf32, #blocked> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked>
    %result = tt.dot %a, %b, %cst {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<64x128xf16, #dot0> * tensor<128x128xf16, #dot1> -> tensor<64x128xf32, #blocked>
    tt.return %result : tensor<64x128xf32, #blocked>
  }
}

// -----
// Config 2: Wide N (64x256, K=128, f16) — more N-reps, great for A sharing
// mReps = 64/(1*2*4) = 8, nReps = 256/(1*16*1) = 16
// origPressure = 8*128*2 + 128*16*2 + 8*16*4 = 6656
// shuffleRatio = (8+16)/(8*16*128) = 0.00146
// With kChunk=32: sharedP = 8*32*2 + 32*16*2 + 512 = 2048
// pressureReduction = 6656/2048 = 3.25x
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dot0b = #ttg.dot_op<{opIdx = 0, parent = #blocked2}>
#dot1b = #ttg.dot_op<{opIdx = 1, parent = #blocked2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @wide_n_config
  // CHECK: tt.dot {{.*}} {ttig.cooperative_sharing = {
  // CHECK-SAME: sharingGroupA = 16 : i32
  tt.func public @wide_n_config(
      %a: tensor<64x128xf16, #dot0b>,
      %b: tensor<128x256xf16, #dot1b>) -> tensor<64x256xf32, #blocked2> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x256xf32, #blocked2>
    %result = tt.dot %a, %b, %cst {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<64x128xf16, #dot0b> * tensor<128x256xf16, #dot1b> -> tensor<64x256xf32, #blocked2>
    tt.return %result : tensor<64x256xf32, #blocked2>
  }
}

// -----
// Config 3: Deep K (64x64, K=256, f16) — maximum K pressure
// mReps = 64/(1*2*4) = 8, nReps = 64/(1*16*1) = 4
// origPressure = 8*256*2 + 256*4*2 + 8*4*4 = 6272
// shuffleRatio = (8+4)/(8*4*256) = 0.00146
// With kChunk=32: sharedP = 8*32*2 + 32*4*2 + 128 = 896
// pressureReduction = 6272/896 = 7.0x
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dot0c = #ttg.dot_op<{opIdx = 0, parent = #blocked3}>
#dot1c = #ttg.dot_op<{opIdx = 1, parent = #blocked3}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @deep_k_config
  // CHECK: tt.dot {{.*}} {ttig.cooperative_sharing = {
  // CHECK-SAME: kChunk = 32 : i32
  tt.func public @deep_k_config(
      %a: tensor<64x256xf16, #dot0c>,
      %b: tensor<256x64xf16, #dot1c>) -> tensor<64x64xf32, #blocked3> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked3>
    %result = tt.dot %a, %b, %cst {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<64x256xf16, #dot0c> * tensor<256x64xf16, #dot1c> -> tensor<64x64xf32, #blocked3>
    tt.return %result : tensor<64x64xf32, #blocked3>
  }
}

// -----
// Config 4: Max sharing (64x128, K=128, f16, tpw=[1,32])
// sharingGroupA = 32 (maximum possible!)
// mReps = 64/(1*1*4) = 16, nReps = 128/(1*32*1) = 4
// origPressure = 16*128*2 + 128*4*2 + 16*4*4 = 5376
// shuffleRatio = (16+4)/(16*4*128) = 0.00244
// With kChunk=32: sharedP = 16*32*2 + 32*4*2 + 256 = 1536
// pressureReduction = 5376/1536 = 3.5x
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#dot0d = #ttg.dot_op<{opIdx = 0, parent = #blocked4}>
#dot1d = #ttg.dot_op<{opIdx = 1, parent = #blocked4}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @max_sharing_config
  // CHECK: tt.dot {{.*}} {ttig.cooperative_sharing = {
  // CHECK-SAME: sharingGroupA = 32 : i32
  tt.func public @max_sharing_config(
      %a: tensor<64x128xf16, #dot0d>,
      %b: tensor<128x128xf16, #dot1d>) -> tensor<64x128xf32, #blocked4> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked4>
    %result = tt.dot %a, %b, %cst {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<64x128xf16, #dot0d> * tensor<128x128xf16, #dot1d> -> tensor<64x128xf32, #blocked4>
    tt.return %result : tensor<64x128xf32, #blocked4>
  }
}

// -----
// Config 5: Balanced encoding (64x128, K=128, f16, tpw=[4,8])
// sharingGroupA = 8, sharingGroupB = 4
// mReps = 64/(1*4*4) = 4, nReps = 128/(1*8*1) = 16
// origPressure = 4*128*2 + 128*16*2 + 4*16*4 = 5376
// shuffleRatio = (4+16)/(4*16*128) = 0.00244
// With kChunk=32: sharedP = 4*32*2 + 32*16*2 + 256 = 1536
// pressureReduction = 5376/1536 = 3.5x
#blocked5 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#dot0e = #ttg.dot_op<{opIdx = 0, parent = #blocked5}>
#dot1e = #ttg.dot_op<{opIdx = 1, parent = #blocked5}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @balanced_encoding
  // CHECK: tt.dot {{.*}} {ttig.cooperative_sharing = {
  // CHECK-SAME: sharingGroupA = 8 : i32
  // CHECK-SAME: sharingGroupB = 4 : i32
  tt.func public @balanced_encoding(
      %a: tensor<64x128xf16, #dot0e>,
      %b: tensor<128x128xf16, #dot1e>) -> tensor<64x128xf32, #blocked5> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked5>
    %result = tt.dot %a, %b, %cst {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<64x128xf16, #dot0e> * tensor<128x128xf16, #dot1e> -> tensor<64x128xf32, #blocked5>
    tt.return %result : tensor<64x128xf32, #blocked5>
  }
}

// -----
// Config 6: f32 operands (64x128, K=128) — 2x element size
// origPressure = 8*128*4 + 128*8*4 + 8*8*4 = 8448
// With kChunk=32: sharedP = 8*32*4 + 32*8*4 + 256 = 2304
// pressureReduction = 8448/2304 = 3.67x
#blocked6 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dot0f = #ttg.dot_op<{opIdx = 0, parent = #blocked6}>
#dot1f = #ttg.dot_op<{opIdx = 1, parent = #blocked6}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @f32_operands
  // CHECK: tt.dot {{.*}} {ttig.cooperative_sharing = {
  // CHECK-SAME: sharingGroupA = 16 : i32
  tt.func public @f32_operands(
      %a: tensor<64x128xf32, #dot0f>,
      %b: tensor<128x128xf32, #dot1f>) -> tensor<64x128xf32, #blocked6> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked6>
    %result = tt.dot %a, %b, %cst {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<64x128xf32, #dot0f> * tensor<128x128xf32, #dot1f> -> tensor<64x128xf32, #blocked6>
    tt.return %result : tensor<64x128xf32, #blocked6>
  }
}

// -----
// NEGATIVE: Small dot (16x16, K=16) — pressure too low for benefit
// mReps = 16/(1*2*4) = 2, nReps = 16/(1*16*1) = 1
// origPressure = 2*16*2 + 16*1*2 + 2*1*4 = 104 bytes
// pressureReduction would be < 2x with such small values
#blocked7 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dot0g = #ttg.dot_op<{opIdx = 0, parent = #blocked7}>
#dot1g = #ttg.dot_op<{opIdx = 1, parent = #blocked7}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @small_dot_no_sharing
  // CHECK: tt.dot
  // CHECK-NOT: ttig.cooperative_sharing
  tt.func public @small_dot_no_sharing(
      %a: tensor<16x16xf16, #dot0g>,
      %b: tensor<16x16xf16, #dot1g>) -> tensor<16x16xf32, #blocked7> {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked7>
    %result = tt.dot %a, %b, %cst {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<16x16xf16, #dot0g> * tensor<16x16xf16, #dot1g> -> tensor<16x16xf32, #blocked7>
    tt.return %result : tensor<16x16xf32, #blocked7>
  }
}

// -----
// NEGATIVE: DPAS hardware — pass should not run at all
#blocked8 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dot0h = #ttg.dot_op<{opIdx = 0, parent = #blocked8}>
#dot1h = #ttg.dot_op<{opIdx = 1, parent = #blocked8}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttig.support_subgroup_matrix_multiply_accumulate"} {
  // CHECK-LABEL: @dpas_hardware_no_sharing
  // CHECK: tt.dot
  // CHECK-NOT: ttig.cooperative_sharing
  tt.func public @dpas_hardware_no_sharing(
      %a: tensor<64x128xf16, #dot0h>,
      %b: tensor<128x128xf16, #dot1h>) -> tensor<64x128xf32, #blocked8> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked8>
    %result = tt.dot %a, %b, %cst {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<64x128xf16, #dot0h> * tensor<128x128xf16, #dot1h> -> tensor<64x128xf32, #blocked8>
    tt.return %result : tensor<64x128xf32, #blocked8>
  }
}

// -----
// NEGATIVE: M-heavy encoding (tpw=[16,2]) — sharingGroupA=2, sharingGroupB=16
// But sharingGroupA < 4 AND B sharing doesn't help much with spt=[1,1]
// Actually sharingGroupB=16 >= 4, so it might still fire...
// Let's check: mReps = 64/(1*16*1) = 4, nReps = 128/(1*2*4) = 16
// origPressure = 4*128*2 + 128*16*2 + 4*16*4 = 5376
// sharingGroupA=2 (nTpw), sharingGroupB=16 (mTpw)
// Since sharingGroupB >= 4, pass can still fire
#blocked9 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [1, 0]}>
#dot0i = #ttg.dot_op<{opIdx = 0, parent = #blocked9}>
#dot1i = #ttg.dot_op<{opIdx = 1, parent = #blocked9}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @m_heavy_encoding
  // CHECK: tt.dot {{.*}} {ttig.cooperative_sharing = {
  // CHECK-SAME: sharingGroupA = 2 : i32
  // CHECK-SAME: sharingGroupB = 16 : i32
  tt.func public @m_heavy_encoding(
      %a: tensor<64x128xf16, #dot0i>,
      %b: tensor<128x128xf16, #dot1i>) -> tensor<64x128xf32, #blocked9> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked9>
    %result = tt.dot %a, %b, %cst {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<64x128xf16, #dot0i> * tensor<128x128xf16, #dot1i> -> tensor<64x128xf32, #blocked9>
    tt.return %result : tensor<64x128xf32, #blocked9>
  }
}
