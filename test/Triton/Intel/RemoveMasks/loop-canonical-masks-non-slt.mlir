// RUN: triton-opt %s -triton-intel-remove-masks | FileCheck %s

// COM: CanonicalMaskValidator must only accept the strict signed `<` predicate.
// COM: The canonical-form recognition and the generated versioning condition
// COM: (RemSIOp + sgt against a DivSIOp-folded upper bound) are derived assuming
// COM: strict signed semantics. For sle/ult/ule masks that otherwise look
// COM: canonical, the validator must bail out so the loop is left unchanged.

module {
  // COM: `sle` mask with a canonical-shape cdiv upper bound. The validator
  // COM: must refuse to rewrite this loop.
  tt.func public @test_canonical_mask_sle_not_versioned(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x256xf16>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x64xf16>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_4 = arith.constant dense<64> : tensor<128x64xi32>
    %c64_i32 = arith.constant 64 : i32
    %cst_5 = arith.constant dense<512> : tensor<64x1xi32>
    %cst_6 = arith.constant dense<512> : tensor<128x1xi32>
    %c256_i32 = arith.constant 256 : i32
    %c512_i32 = arith.constant 512 : i32
    %c128_i32 = arith.constant 128 : i32
    %cst_3 = arith.constant dense<32768> : tensor<64x256xi32>
    %c8_i32 = arith.constant 8 : i32
    %10 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %15 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %19 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %20 = tt.expand_dims %10 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    %21 = arith.muli %20, %cst_6 : tensor<128x1xi32>
    %22 = tt.expand_dims %19 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %23 = tt.broadcast %21 : tensor<128x1xi32> -> tensor<128x64xi32>
    %24 = tt.broadcast %22 : tensor<1x64xi32> -> tensor<128x64xi32>
    %25 = arith.addi %23, %24 : tensor<128x64xi32>
    %26 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>>
    %27 = tt.addptr %26, %25 : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
    %28 = tt.expand_dims %19 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %29 = arith.muli %28, %cst_5 : tensor<64x1xi32>
    %30 = tt.expand_dims %15 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    %31 = tt.broadcast %29 : tensor<64x1xi32> -> tensor<64x256xi32>
    %32 = tt.broadcast %30 : tensor<1x256xi32> -> tensor<64x256xi32>
    %33 = arith.addi %31, %32 : tensor<64x256xi32>
    %34 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x256x!tt.ptr<f16>>
    %35 = tt.addptr %34, %33 : tensor<64x256x!tt.ptr<f16>>, tensor<64x256xi32>
    %36:3 = scf.for %arg3 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg4 = %cst, %arg5 = %27, %arg6 = %35) -> (tensor<128x256xf32>, tensor<128x64x!tt.ptr<f16>>, tensor<64x256x!tt.ptr<f16>>)  : i32 {
      %51 = arith.muli %arg3, %c64_i32 : i32
      %52 = arith.subi %c512_i32, %51 : i32
      %53 = tt.splat %52 : i32 -> tensor<1x64xi32>
      // Use `sle` instead of `slt`: validator must NOT recognize this as canonical.
      %54 = arith.cmpi sle, %22, %53 : tensor<1x64xi32>
      %55 = tt.broadcast %54 : tensor<1x64xi1> -> tensor<128x64xi1>
      %56 = tt.load %arg5, %55, %cst_1 : tensor<128x64x!tt.ptr<f16>>
      %57 = tt.splat %52 : i32 -> tensor<64x1xi32>
      %58 = arith.cmpi sle, %28, %57 : tensor<64x1xi32>
      %59 = tt.broadcast %58 : tensor<64x1xi1> -> tensor<64x256xi1>
      %60 = tt.load %arg6, %59, %cst_0 : tensor<64x256x!tt.ptr<f16>>
      %61 = tt.dot %56, %60, %arg4, inputPrecision = tf32 : tensor<128x64xf16> * tensor<64x256xf16> -> tensor<128x256xf32>
      %62 = tt.addptr %arg5, %cst_4 : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
      %63 = tt.addptr %arg6, %cst_3 : tensor<64x256x!tt.ptr<f16>>, tensor<64x256xi32>
      scf.yield %61, %62, %63 : tensor<128x256xf32>, tensor<128x64x!tt.ptr<f16>>, tensor<64x256x!tt.ptr<f16>>
    }
    tt.return
  }

  // CHECK-LABEL: tt.func public @test_canonical_mask_sle_not_versioned
  // COM: No versioning `scf.if` is inserted: the loop is left as-is, with the
  // COM: masks kept on the loads.
  // CHECK-NOT:     scf.if
  // CHECK:         scf.for
  // CHECK:           [[M1:%.+]] = arith.cmpi sle,
  // CHECK:           tt.load {{.*}}, {{.*}}, {{.*}} : tensor<128x64x!tt.ptr<f16>>
  // CHECK:           [[M2:%.+]] = arith.cmpi sle,
  // CHECK:           tt.load {{.*}}, {{.*}}, {{.*}} : tensor<64x256x!tt.ptr<f16>>
  // CHECK:         }
  // CHECK-NOT:     scf.if
  // CHECK:         tt.return
}

// -----

module {
  // COM: Same structural canonical pattern but with `ult`. Must also be refused.
  tt.func public @test_canonical_mask_ult_not_versioned(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x256xf16>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x64xf16>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_4 = arith.constant dense<64> : tensor<128x64xi32>
    %c64_i32 = arith.constant 64 : i32
    %cst_5 = arith.constant dense<512> : tensor<64x1xi32>
    %cst_6 = arith.constant dense<512> : tensor<128x1xi32>
    %c256_i32 = arith.constant 256 : i32
    %c512_i32 = arith.constant 512 : i32
    %c128_i32 = arith.constant 128 : i32
    %cst_3 = arith.constant dense<32768> : tensor<64x256xi32>
    %c8_i32 = arith.constant 8 : i32
    %10 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %15 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %19 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %20 = tt.expand_dims %10 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    %21 = arith.muli %20, %cst_6 : tensor<128x1xi32>
    %22 = tt.expand_dims %19 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %23 = tt.broadcast %21 : tensor<128x1xi32> -> tensor<128x64xi32>
    %24 = tt.broadcast %22 : tensor<1x64xi32> -> tensor<128x64xi32>
    %25 = arith.addi %23, %24 : tensor<128x64xi32>
    %26 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>>
    %27 = tt.addptr %26, %25 : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
    %28 = tt.expand_dims %19 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %29 = arith.muli %28, %cst_5 : tensor<64x1xi32>
    %30 = tt.expand_dims %15 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    %31 = tt.broadcast %29 : tensor<64x1xi32> -> tensor<64x256xi32>
    %32 = tt.broadcast %30 : tensor<1x256xi32> -> tensor<64x256xi32>
    %33 = arith.addi %31, %32 : tensor<64x256xi32>
    %34 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x256x!tt.ptr<f16>>
    %35 = tt.addptr %34, %33 : tensor<64x256x!tt.ptr<f16>>, tensor<64x256xi32>
    %36:3 = scf.for %arg3 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg4 = %cst, %arg5 = %27, %arg6 = %35) -> (tensor<128x256xf32>, tensor<128x64x!tt.ptr<f16>>, tensor<64x256x!tt.ptr<f16>>)  : i32 {
      %51 = arith.muli %arg3, %c64_i32 : i32
      %52 = arith.subi %c512_i32, %51 : i32
      %53 = tt.splat %52 : i32 -> tensor<1x64xi32>
      %54 = arith.cmpi ult, %22, %53 : tensor<1x64xi32>
      %55 = tt.broadcast %54 : tensor<1x64xi1> -> tensor<128x64xi1>
      %56 = tt.load %arg5, %55, %cst_1 : tensor<128x64x!tt.ptr<f16>>
      %57 = tt.splat %52 : i32 -> tensor<64x1xi32>
      %58 = arith.cmpi ult, %28, %57 : tensor<64x1xi32>
      %59 = tt.broadcast %58 : tensor<64x1xi1> -> tensor<64x256xi1>
      %60 = tt.load %arg6, %59, %cst_0 : tensor<64x256x!tt.ptr<f16>>
      %61 = tt.dot %56, %60, %arg4, inputPrecision = tf32 : tensor<128x64xf16> * tensor<64x256xf16> -> tensor<128x256xf32>
      %62 = tt.addptr %arg5, %cst_4 : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
      %63 = tt.addptr %arg6, %cst_3 : tensor<64x256x!tt.ptr<f16>>, tensor<64x256xi32>
      scf.yield %61, %62, %63 : tensor<128x256xf32>, tensor<128x64x!tt.ptr<f16>>, tensor<64x256x!tt.ptr<f16>>
    }
    tt.return
  }

  // CHECK-LABEL: tt.func public @test_canonical_mask_ult_not_versioned
  // CHECK-NOT:     scf.if
  // CHECK:         scf.for
  // CHECK:           [[M1:%.+]] = arith.cmpi ult,
  // CHECK:           tt.load {{.*}}, {{.*}}, {{.*}} : tensor<128x64x!tt.ptr<f16>>
  // CHECK:           [[M2:%.+]] = arith.cmpi ult,
  // CHECK:           tt.load {{.*}}, {{.*}}, {{.*}} : tensor<64x256x!tt.ptr<f16>>
  // CHECK:         }
  // CHECK-NOT:     scf.if
  // CHECK:         tt.return
}
