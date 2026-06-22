// RUN: triton-opt %s -split-input-file --tritonintelgpu-decompose-fma-dots | FileCheck %s

// Test: Large K dot on non-DPAS hardware should be decomposed into scf.for.
// Encoding: sizePerThread=[1,4], K=512, f16 → perThreadBytes = (1*512 + 512*4)*2 = 5120 > 4096
// perCTAK = 4*4*1 = 16, selectKTile(512, 16) → 32 (512%32==0, 32%16==0)
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [4, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @large_k_dot_decomposed
  // CHECK: scf.for
  // CHECK:   tt.dot
  // CHECK:   scf.yield
  tt.func public @large_k_dot_decomposed(
      %base_a: !tt.ptr<f16>,
      %base_b: !tt.ptr<f16>) -> tensor<64x64xf32, #blocked> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    // Pointer for A [64, 512] via splat + offset
    %a_splat = tt.splat %base_a : !tt.ptr<f16> -> tensor<64x512x!tt.ptr<f16>, #blocked>
    %c0_i32 = arith.constant dense<0> : tensor<64x512xi32, #blocked>
    %a_ptr = tt.addptr %a_splat, %c0_i32 : tensor<64x512x!tt.ptr<f16>, #blocked>, tensor<64x512xi32, #blocked>
    %a = tt.load %a_ptr : tensor<64x512x!tt.ptr<f16>, #blocked>
    %a_cvt = ttg.convert_layout %a : tensor<64x512xf16, #blocked> -> tensor<64x512xf16, #dot0>
    // Pointer for B [512, 64] via splat + offset
    %b_splat = tt.splat %base_b : !tt.ptr<f16> -> tensor<512x64x!tt.ptr<f16>, #blocked1>
    %c0b_i32 = arith.constant dense<0> : tensor<512x64xi32, #blocked1>
    %b_ptr = tt.addptr %b_splat, %c0b_i32 : tensor<512x64x!tt.ptr<f16>, #blocked1>, tensor<512x64xi32, #blocked1>
    %b = tt.load %b_ptr : tensor<512x64x!tt.ptr<f16>, #blocked1>
    %b_cvt = ttg.convert_layout %b : tensor<512x64xf16, #blocked1> -> tensor<512x64xf16, #dot1>
    // Dot
    %result = tt.dot %a_cvt, %b_cvt, %cst {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<64x512xf16, #dot0> * tensor<512x64xf16, #dot1> -> tensor<64x64xf32, #blocked>
    tt.return %result : tensor<64x64xf32, #blocked>
  }
}

// -----

// Test: Small K dot should NOT be decomposed (below pressure threshold)
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [4, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#dot0b = #ttg.dot_op<{opIdx = 0, parent = #blocked2}>
#dot1b = #ttg.dot_op<{opIdx = 1, parent = #blocked2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
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

// Test: Dot on DPAS-capable hardware should NOT be decomposed
#blocked4 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [4, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#dot0c = #ttg.dot_op<{opIdx = 0, parent = #blocked4}>
#dot1c = #ttg.dot_op<{opIdx = 1, parent = #blocked4}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_subgroup_matrix_multiply_accumulate"} {
  // CHECK-LABEL: @dpas_hardware_unchanged
  // CHECK-NOT: scf.for
  // CHECK: tt.dot
  tt.func public @dpas_hardware_unchanged(
      %a_ptr: tensor<64x128x!tt.ptr<f16>, #blocked4>,
      %b_ptr: tensor<128x64x!tt.ptr<f16>, #blocked5>) -> tensor<64x64xf32, #blocked4> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked4>
    %a = tt.load %a_ptr : tensor<64x128x!tt.ptr<f16>, #blocked4>
    %a_cvt = ttg.convert_layout %a : tensor<64x128xf16, #blocked4> -> tensor<64x128xf16, #dot0c>
    %b = tt.load %b_ptr : tensor<128x64x!tt.ptr<f16>, #blocked5>
    %b_cvt = ttg.convert_layout %b : tensor<128x64xf16, #blocked5> -> tensor<128x64xf16, #dot1c>
    %result = tt.dot %a_cvt, %b_cvt, %cst {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<64x128xf16, #dot0c> * tensor<128x64xf16, #dot1c> -> tensor<64x64xf32, #blocked4>
    tt.return %result : tensor<64x64xf32, #blocked4>
  }
}
