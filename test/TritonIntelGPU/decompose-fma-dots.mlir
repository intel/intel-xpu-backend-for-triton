// RUN: triton-opt %s -split-input-file --tritonintelgpu-decompose-fma-dots | FileCheck %s

// Test: Large K dot on non-DPAS hardware should be decomposed into scf.for.
// The dot operands come from loads with pointer arithmetic that can be tiled.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [4, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @large_k_dot_decomposed
  // CHECK: scf.for
  // CHECK:   tt.dot
  // CHECK:   scf.yield
  tt.func public @large_k_dot_decomposed(
      %base_a: !tt.ptr<f16>,
      %base_b: !tt.ptr<f16>) -> tensor<64x64xf32, #blocked> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    // Build pointer for A [64, 128]
    %a_base = tt.splat %base_a : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #blocked>
    %a_range_m = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
    %a_range_k = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %c128_i32 = arith.constant dense<128> : tensor<64xi32>
    %a_m_offset = arith.muli %a_range_m, %c128_i32 : tensor<64xi32>
    %a_m_exp = tt.expand_dims %a_m_offset {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %a_m_bcast = tt.broadcast %a_m_exp : tensor<64x1xi32> -> tensor<64x128xi32>
    %a_k_exp = tt.expand_dims %a_range_k {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %a_k_bcast = tt.broadcast %a_k_exp : tensor<1x128xi32> -> tensor<64x128xi32>
    %a_offset = arith.addi %a_m_bcast, %a_k_bcast : tensor<64x128xi32>
    %a_ptr = tt.addptr %a_base, %a_offset : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32>
    %a = tt.load %a_ptr : tensor<64x128x!tt.ptr<f16>, #blocked>
    // Build pointer for B [128, 64]
    %b_base = tt.splat %base_b : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked1>
    %b_range_k = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %b_range_n = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
    %c64_i32 = arith.constant dense<64> : tensor<128xi32>
    %b_k_offset = arith.muli %b_range_k, %c64_i32 : tensor<128xi32>
    %b_k_exp = tt.expand_dims %b_k_offset {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    %b_k_bcast = tt.broadcast %b_k_exp : tensor<128x1xi32> -> tensor<128x64xi32>
    %b_n_exp = tt.expand_dims %b_range_n {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %b_n_bcast = tt.broadcast %b_n_exp : tensor<1x64xi32> -> tensor<128x64xi32>
    %b_offset = arith.addi %b_k_bcast, %b_n_bcast : tensor<128x64xi32>
    %b_ptr = tt.addptr %b_base, %b_offset : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32>
    %b = tt.load %b_ptr : tensor<128x64x!tt.ptr<f16>, #blocked1>
    // Dot
    %result = tt.dot %a, %b, %cst {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<64x128xf16, #blocked> * tensor<128x64xf16, #blocked1> -> tensor<64x64xf32, #blocked>
    tt.return %result : tensor<64x64xf32, #blocked>
  }
}

// -----

// Test: Small K dot should NOT be decomposed (below pressure threshold)
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [4, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @small_k_dot_unchanged
  // CHECK-NOT: scf.for
  // CHECK: tt.dot
  tt.func public @small_k_dot_unchanged(
      %a_ptr: tensor<16x16x!tt.ptr<f16>, #blocked2>,
      %b_ptr: tensor<16x16x!tt.ptr<f16>, #blocked3>) -> tensor<16x16xf32, #blocked2> {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked2>
    %a = tt.load %a_ptr : tensor<16x16x!tt.ptr<f16>, #blocked2>
    %b = tt.load %b_ptr : tensor<16x16x!tt.ptr<f16>, #blocked3>
    %result = tt.dot %a, %b, %cst {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<16x16xf16, #blocked2> * tensor<16x16xf16, #blocked3> -> tensor<16x16xf32, #blocked2>
    tt.return %result : tensor<16x16xf32, #blocked2>
  }
}

// -----

// Test: Dot on DPAS-capable hardware should NOT be decomposed
#blocked4 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [4, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_subgroup_matrix_multiply_accumulate"} {
  // CHECK-LABEL: @dpas_hardware_unchanged
  // CHECK-NOT: scf.for
  // CHECK: tt.dot
  tt.func public @dpas_hardware_unchanged(
      %a_ptr: tensor<64x128x!tt.ptr<f16>, #blocked4>,
      %b_ptr: tensor<128x64x!tt.ptr<f16>, #blocked5>) -> tensor<64x64xf32, #blocked4> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked4>
    %a = tt.load %a_ptr : tensor<64x128x!tt.ptr<f16>, #blocked4>
    %b = tt.load %b_ptr : tensor<128x64x!tt.ptr<f16>, #blocked5>
    %result = tt.dot %a, %b, %cst {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<64x128xf16, #blocked4> * tensor<128x64xf16, #blocked5> -> tensor<64x64xf32, #blocked4>
    tt.return %result : tensor<64x64xf32, #blocked4>
  }
}
