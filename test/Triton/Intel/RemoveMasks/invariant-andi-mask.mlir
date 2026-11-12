// RUN: triton-opt %s -split-input-file -triton-intel-remove-masks | FileCheck %s

module {
  // COM: Test that InvariantMaskValidator handles compound andi masks where
  // COM: both operands are valid invariant masks (existing patterns).
  tt.func public @test_invariant_andi_mask(
      %ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %arg1: i32, %arg2: i32, %bound_m: i32, %bound_n: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
    %0 = tt.get_program_id x : i32
    %1 = tt.get_num_programs x : i32

    // Two invariant masks in existing supported form: make_range < splat(N)
    %range_m = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %splat_m = tt.splat %bound_m : i32 -> tensor<64xi32>
    %mask_m_1d = arith.cmpi slt, %range_m, %splat_m : tensor<64xi32>
    %mask_m = tt.expand_dims %mask_m_1d {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
    %mask_m_bc = tt.broadcast %mask_m : tensor<64x1xi1> -> tensor<64x64xi1>

    %range_n = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %splat_n = tt.splat %bound_n : i32 -> tensor<64xi32>
    %mask_n_1d = arith.cmpi slt, %range_n, %splat_n : tensor<64xi32>
    %mask_n = tt.expand_dims %mask_n_1d {axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
    %mask_n_bc = tt.broadcast %mask_n : tensor<1x64xi1> -> tensor<64x64xi1>

    // Compound mask: andi(M_mask, N_mask)
    %mask = arith.andi %mask_m_bc, %mask_n_bc : tensor<64x64xi1>

    %base = tt.splat %ptr : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>>
    scf.for %iv = %0 to %arg2 step %1 : i32 {
      %loaded = tt.load %base, %mask, %cst : tensor<64x64x!tt.ptr<f32>>
    }
    tt.return
  }

  // CHECK: tt.func public @test_invariant_andi_mask
  // COM: Both sub-masks are valid invariant patterns. The compound andi triggers
  // COM: loop versioning with AND of both versioning conditions.
  // CHECK: scf.if
  // CHECK:   scf.for
  // CHECK:     tt.load {{%[0-9]+}} : tensor<64x64x!tt.ptr<f32>>
  // CHECK:   }
  // CHECK: } else {
  // CHECK:   scf.for
  // CHECK:     tt.load {{%[a-z0-9_]+}}, {{%[a-z0-9_]+}}, {{%[a-z0-9_]+}} : tensor<64x64x!tt.ptr<f32>>
  // CHECK:   }
  // CHECK: }
}
