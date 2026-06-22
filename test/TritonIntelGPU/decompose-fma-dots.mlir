// RUN: triton-opt %s -split-input-file --tritonintelgpu-decompose-fma-dots | FileCheck %s

// Test 1: Real crashing config on ARL-S (64x128x128, f16, 4 warps, warp_size=32)
// This exactly matches the config that caused 587KB PTSS overflow.
// Encoding: spt=[1,1], tpw=[2,16], wpc=[4,1] for result shape 64x128
// perThreadBytes: aBytes=8*1*128*2=2048 + bBytes=128*8*1*2=2048 + cBytes=8*8*4=256 = 4352 > 4096
// perCTAK = 1*16*1 = 16, selectKTile(128, 16) -> 32
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @arl_s_crashing_dot_decomposed
  // CHECK: scf.for
  // CHECK:   tt.dot
  // CHECK:   scf.yield
  tt.func public @arl_s_crashing_dot_decomposed(
      %base_a: !tt.ptr<f16>,
      %base_b: !tt.ptr<f16>) -> tensor<64x128xf32, #blocked> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked>
    %a_splat = tt.splat %base_a : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #blocked>
    %c0a = arith.constant dense<0> : tensor<64x128xi32, #blocked>
    %a_ptr = tt.addptr %a_splat, %c0a : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
    %a = tt.load %a_ptr : tensor<64x128x!tt.ptr<f16>, #blocked>
    %a_cvt = ttg.convert_layout %a : tensor<64x128xf16, #blocked> -> tensor<64x128xf16, #dot0>
    %b_splat = tt.splat %base_b : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>, #blocked1>
    %c0b = arith.constant dense<0> : tensor<128x128xi32, #blocked1>
    %b_ptr = tt.addptr %b_splat, %c0b : tensor<128x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xi32, #blocked1>
    %b = tt.load %b_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    %b_cvt = ttg.convert_layout %b : tensor<128x128xf16, #blocked1> -> tensor<128x128xf16, #dot1>
    %result = tt.dot %a_cvt, %b_cvt, %cst {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<64x128xf16, #dot0> * tensor<128x128xf16, #dot1> -> tensor<64x128xf32, #blocked>
    tt.return %result : tensor<64x128xf32, #blocked>
  }
}

// -----

// Test 2: Small K dot should NOT be decomposed (below pressure threshold)
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#dot0b = #ttg.dot_op<{opIdx = 0, parent = #blocked2}>
#dot1b = #ttg.dot_op<{opIdx = 1, parent = #blocked2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @small_k_dot_unchanged
  // CHECK-NOT: scf.for
  // CHECK: tt.dot
  tt.func public @small_k_dot_unchanged(
      %a_ptr: tensor<16x16x!tt.ptr<f16>, #blocked2>,
      %b_ptr: tensor<16x16x!tt.ptr<f16>, #blocked3>) -> tensor<16x16xf32, #blocked2> {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked2>
    %a = tt.load %a_ptr : tensor<16x16x!tt.ptr<f16>, #blocked2>
    %a_cvt = ttg.convert_layout %a : tensor<16x16xf16, #blocked2> -> tensor<16x16xf16, #dot0b>
    %b = tt.load %b_ptr : tensor<16x16x!tt.ptr<f16>, #blocked3>
    %b_cvt = ttg.convert_layout %b : tensor<16x16xf16, #blocked3> -> tensor<16x16xf16, #dot1b>
    %result = tt.dot %a_cvt, %b_cvt, %cst {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<16x16xf16, #dot0b> * tensor<16x16xf16, #dot1b> -> tensor<16x16xf32, #blocked2>
    tt.return %result : tensor<16x16xf32, #blocked2>
  }
}

// -----

// Test 3: Dot on DPAS-capable hardware should NOT be decomposed
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#dot0c = #ttg.dot_op<{opIdx = 0, parent = #blocked4}>
#dot1c = #ttg.dot_op<{opIdx = 1, parent = #blocked4}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttig.support_subgroup_matrix_multiply_accumulate"} {
  // CHECK-LABEL: @dpas_hardware_unchanged
  // CHECK-NOT: scf.for
  // CHECK: tt.dot
  tt.func public @dpas_hardware_unchanged(
      %a_ptr: tensor<64x128x!tt.ptr<f16>, #blocked4>,
      %b_ptr: tensor<128x128x!tt.ptr<f16>, #blocked5>) -> tensor<64x128xf32, #blocked4> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked4>
    %a = tt.load %a_ptr : tensor<64x128x!tt.ptr<f16>, #blocked4>
    %a_cvt = ttg.convert_layout %a : tensor<64x128xf16, #blocked4> -> tensor<64x128xf16, #dot0c>
    %b = tt.load %b_ptr : tensor<128x128x!tt.ptr<f16>, #blocked5>
    %b_cvt = ttg.convert_layout %b : tensor<128x128xf16, #blocked5> -> tensor<128x128xf16, #dot1c>
    %result = tt.dot %a_cvt, %b_cvt, %cst {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<64x128xf16, #dot0c> * tensor<128x128xf16, #dot1c> -> tensor<64x128xf32, #blocked4>
    tt.return %result : tensor<64x128xf32, #blocked4>
  }
}
