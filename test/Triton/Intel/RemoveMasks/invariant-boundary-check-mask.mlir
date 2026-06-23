// RUN: triton-opt %s -split-input-file -triton-intel-remove-masks | FileCheck %s

module {
  // COM: Test that InvariantMaskValidator handles boundary-check patterns
  // COM: from RewriteTensorDescriptorToPointer:
  // COM:   (splat(offset) + ext(make_range(0, N))) cmp splat(constant)
  // COM: These are loop-invariant (depend on program_id, not loop IV) and
  // COM: trigger loop versioning.
  tt.func public @test_invariant_boundary_check(
      %ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %offset: i64) {
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c128_i32 = arith.constant 128 : i32
    %c0_i64 = arith.constant 0 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<256xf16>

    // Invariant boundary-check mask: (splat(offset) + ext(range)) >= 0 AND < 4096
    %range = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %range_i64 = arith.extsi %range : tensor<256xi32> to tensor<256xi64>
    %off_splat = tt.splat %offset : i64 -> tensor<256xi64>
    %combined = arith.addi %off_splat, %range_i64 : tensor<256xi64>

    %lb = tt.splat %c0_i64 : i64 -> tensor<256xi64>
    %ub = tt.splat %c4096_i64 : i64 -> tensor<256xi64>
    %mask_sge = arith.cmpi sge, %combined, %lb : tensor<256xi64>
    %mask_slt = arith.cmpi slt, %combined, %ub : tensor<256xi64>
    %mask = arith.andi %mask_sge, %mask_slt : tensor<256xi1>

    %base = tt.splat %ptr : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>>

    scf.for %iv = %c0_i32 to %c128_i32 step %c32_i32 : i32 {
      %loaded = tt.load %base, %mask, %cst : tensor<256x!tt.ptr<f16>>
    }
    tt.return
  }

  // CHECK-LABEL: tt.func public @test_invariant_boundary_check
  // COM: The invariant boundary-check mask triggers loop versioning.
  // COM: The "then" branch (mask is true) has unmasked loads.
  // CHECK: scf.if
  // CHECK:   scf.for
  // CHECK:     tt.load {{%[0-9]+}} : tensor<256x!tt.ptr<f16>>
  // CHECK:   }
  // CHECK: } else {
  // CHECK:   scf.for
  // CHECK:     tt.load {{%.+}}, {{%.+}}, {{%.+}} : tensor<256x!tt.ptr<f16>>
  // CHECK:   }
  // CHECK: }
}
