// RUN: triton-opt %s -split-input-file -triton-intel-remove-masks | FileCheck %s

module {
  // COM: Test that boundary-check masks from RewriteTensorDescriptorToPointer
  // COM: are simplified when the K-dimension mask is always true (loop-dependent)
  // COM: and the M-dimension mask is invariant (loop-invariant boundary check).
  // COM: The K andi (sge >= 0 AND slt < K) should be stripped as always-true,
  // COM: leaving only the M dimension mask which triggers loop versioning.
  tt.func public @test_boundary_check_mask_simplification(
      %ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c4096_i32 = arith.constant 4096 : i32
    %c0_i64 = arith.constant 0 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<256x32xf16>

    // M dimension offset (loop-invariant, depends on program_id).
    %pid = tt.get_program_id x : i32
    %pid_offset = arith.muli %pid, %c32_i32 : i32
    %pid_offset_i64 = arith.extsi %pid_offset : i32 to i64
    %m_splat = tt.splat %pid_offset_i64 : i64 -> tensor<256xi64>
    %m_range = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %m_range_i64 = arith.extsi %m_range : tensor<256xi32> to tensor<256xi64>
    %m_offset = arith.addi %m_splat, %m_range_i64 : tensor<256xi64>
    %m_offset_2d = tt.expand_dims %m_offset {axis = 1 : i32} : tensor<256xi64> -> tensor<256x1xi64>

    // M dimension boundary check (invariant).
    %m_lb = tt.splat %c0_i64 : i64 -> tensor<256x1xi64>
    %m_ub = tt.splat %c4096_i64 : i64 -> tensor<256x1xi64>
    %m_sge = arith.cmpi sge, %m_offset_2d, %m_lb : tensor<256x1xi64>
    %m_slt = arith.cmpi slt, %m_offset_2d, %m_ub : tensor<256x1xi64>
    %m_mask = arith.andi %m_sge, %m_slt : tensor<256x1xi1>
    %m_mask_bc = tt.broadcast %m_mask : tensor<256x1xi1> -> tensor<256x32xi1>

    // K dimension range.
    %k_range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %k_range_i64 = arith.extsi %k_range : tensor<32xi32> to tensor<32xi64>

    // Pointer setup.
    %base_splat = tt.splat %ptr : !tt.ptr<f16> -> tensor<256x32x!tt.ptr<f16>>

    // Loop with K-dimension mask that is always true.
    %result = scf.for %iv = %c0_i32 to %c4096_i32 step %c32_i32
        iter_args(%acc = %cst) -> (tensor<256x32xf16>) : i32 {
      // K offset (loop-dependent iter_arg equivalent to IV).
      %k_iv_i64 = arith.extsi %iv : i32 to i64
      %k_splat = tt.splat %k_iv_i64 : i64 -> tensor<32xi64>
      %k_offset = arith.addi %k_splat, %k_range_i64 : tensor<32xi64>
      %k_offset_2d = tt.expand_dims %k_offset {axis = 0 : i32} : tensor<32xi64> -> tensor<1x32xi64>

      // K dimension boundary check (always true: iv in [0, 4064], range [0,31], max = 4095 < 4096).
      %k_lb = tt.splat %c0_i64 : i64 -> tensor<1x32xi64>
      %k_ub = tt.splat %c4096_i64 : i64 -> tensor<1x32xi64>
      %k_sge = arith.cmpi sge, %k_offset_2d, %k_lb : tensor<1x32xi64>
      %k_slt = arith.cmpi slt, %k_offset_2d, %k_ub : tensor<1x32xi64>
      %k_mask = arith.andi %k_sge, %k_slt : tensor<1x32xi1>
      %k_mask_bc = tt.broadcast %k_mask : tensor<1x32xi1> -> tensor<256x32xi1>

      // Combined mask: M (invariant) AND K (always true).
      %combined_mask = arith.andi %m_mask_bc, %k_mask_bc : tensor<256x32xi1>

      // Masked load.
      %loaded = tt.load %base_splat, %combined_mask, %cst : tensor<256x32x!tt.ptr<f16>>
      scf.yield %loaded : tensor<256x32xf16>
    }
    tt.return
  }

  // CHECK: tt.func public @test_boundary_check_mask_simplification
  // COM: After simplification:
  // COM: - K mask (always true) is stripped from the compound andi
  // COM: - M mask (invariant boundary check) triggers loop versioning
  // COM: - The "then" branch has loads without masks
  // CHECK: scf.if
  // CHECK:   scf.for
  // CHECK:     tt.load {{%[0-9]+}} : tensor<256x32x!tt.ptr<f16>>
  // CHECK:   }
  // CHECK: } else {
  // CHECK:   scf.for
  // CHECK:     tt.load {{%[0-9]+}}, {{%[0-9]+}}, {{%[0-9]+}} : tensor<256x32x!tt.ptr<f16>>
  // CHECK:   }
  // CHECK: }
}

// -----

module {
  // COM: Test that a K-dimension mask using an iter_arg (not the canonical IV)
  // COM: is correctly classified as always-true and stripped.
  tt.func public @test_iter_arg_k_mask(
      %ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c128_i32 = arith.constant 128 : i32
    %c0_i64 = arith.constant 0 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<32xf16>

    %k_range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %k_range_i64 = arith.extsi %k_range : tensor<32xi32> to tensor<32xi64>
    %base_splat = tt.splat %ptr : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>>

    // Loop with off_k as iter_arg (same as IV: init=0, step=32).
    %result = scf.for %iv = %c0_i32 to %c128_i32 step %c32_i32
        iter_args(%off_k = %c0_i32) -> (i32) : i32 {
      %off_k_i64 = arith.extsi %off_k : i32 to i64
      %k_splat = tt.splat %off_k_i64 : i64 -> tensor<32xi64>
      %k_offset = arith.addi %k_splat, %k_range_i64 : tensor<32xi64>

      // K boundary check: (off_k + range(0,32)) >= 0 AND < 4096.
      %k_lb = tt.splat %c0_i64 : i64 -> tensor<32xi64>
      %k_ub = tt.splat %c4096_i64 : i64 -> tensor<32xi64>
      %k_sge = arith.cmpi sge, %k_offset, %k_lb : tensor<32xi64>
      %k_slt = arith.cmpi slt, %k_offset, %k_ub : tensor<32xi64>
      %k_mask = arith.andi %k_sge, %k_slt : tensor<32xi1>

      %loaded = tt.load %base_splat, %k_mask, %cst : tensor<32x!tt.ptr<f16>>

      %next_off_k = arith.addi %off_k, %c32_i32 : i32
      scf.yield %next_off_k : i32
    }
    tt.return
  }

  // CHECK: tt.func public @test_iter_arg_k_mask
  // COM: The K mask (off_k + range(0,32) >= 0 AND < 4096) is always true because
  // COM: the iter_arg off_k has range [0, 96] and max element = 96+31 = 127 < 4096.
  // COM: The andi is replaced with constant true, enabling mask removal.
  // CHECK: scf.for
  // CHECK:   tt.load {{%[0-9]+}} : tensor<32x!tt.ptr<f16>>
  // CHECK: }
}
