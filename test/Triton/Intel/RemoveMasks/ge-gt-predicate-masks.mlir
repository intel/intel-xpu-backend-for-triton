// RUN: triton-opt %s -split-input-file -triton-intel-remove-masks | FileCheck %s

// COM: Test that masks with sge (>=) predicates are correctly classified.
// COM: The mask (IV + range(0,32)) >= 0 is always true since IV starts at 0
// COM: and make_range produces non-negative values.
tt.func public @test_sge_mask_always_true(
    %ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
  %c0_i32 = arith.constant 0 : i32
  %c32_i32 = arith.constant 32 : i32
  %c512_i32 = arith.constant 512 : i32
  %cst_zero = arith.constant dense<0> : tensor<1x32xi32>
  %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf16>

  %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %range_2d = tt.expand_dims %range {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
  %base = tt.splat %ptr : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>>

  %result = scf.for %iv = %c0_i32 to %c512_i32 step %c32_i32
      iter_args(%acc = %cst) -> (tensor<32x32xf16>) : i32 {
    %iv_splat = tt.splat %iv : i32 -> tensor<1x32xi32>
    %offset = arith.addi %iv_splat, %range_2d : tensor<1x32xi32>
    // sge mask: (IV + range(0,32)) >= 0 — always true.
    %mask_1d = arith.cmpi sge, %offset, %cst_zero : tensor<1x32xi32>
    %mask = tt.broadcast %mask_1d : tensor<1x32xi1> -> tensor<32x32xi1>

    %loaded = tt.load %base, %mask, %cst : tensor<32x32x!tt.ptr<f16>>
    scf.yield %loaded : tensor<32x32xf16>
  }
  tt.return
}

// CHECK-LABEL: @test_sge_mask_always_true
// COM: The sge mask is always true (min element = 0 + 0 = 0 >= 0).
// CHECK: scf.for
// CHECK-NOT: tt.load {{.*}}, {{.*}},
// CHECK:   tt.load {{%[0-9]+}} : tensor<32x32x!tt.ptr<f16>>
// CHECK: }

// -----

// COM: Test sgt (>) predicate: (IV + range(0,32)) > -1 is always true since
// COM: min element = 0 + 0 = 0 > -1.
tt.func public @test_sgt_mask_always_true(
    %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> tensor<32xf32> {
  %cst = arith.constant dense<0.000000e+00> : tensor<32xf32>
  %c0_i32 = arith.constant 0 : i32
  %c32_i32 = arith.constant 32 : i32
  %c512_i32 = arith.constant 512 : i32
  %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %lower = arith.constant dense<-1> : tensor<32xi32>
  %base_ptrs = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>

  %r = scf.for %iv = %c0_i32 to %c512_i32 step %c32_i32 iter_args(%acc = %cst) -> (tensor<32xf32>) : i32 {
    %iv_splat = tt.splat %iv : i32 -> tensor<32xi32>
    %offsets = arith.addi %iv_splat, %range : tensor<32xi32>
    %mask = arith.cmpi sgt, %offsets, %lower : tensor<32xi32>
    %ptrs = tt.addptr %base_ptrs, %offsets : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %load = tt.load %ptrs, %mask, %cst : tensor<32x!tt.ptr<f32>>
    %new = arith.addf %acc, %load : tensor<32xf32>
    scf.yield %new : tensor<32xf32>
  }
  tt.return %r : tensor<32xf32>
}

// CHECK-LABEL: @test_sgt_mask_always_true
// CHECK: scf.for
// CHECK-NOT: tt.load {{.*}}, {{.*}},
// CHECK:   tt.load {{%[0-9]+}} : tensor<32x!tt.ptr<f32>>

// -----

// COM: Test uge (>=u) predicate: (IV + range(0,32)) >=u 0 is always true
// COM: since any unsigned value >= 0.
tt.func public @test_uge_mask_always_true(
    %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> tensor<32xf32> {
  %cst = arith.constant dense<0.000000e+00> : tensor<32xf32>
  %c0_i32 = arith.constant 0 : i32
  %c32_i32 = arith.constant 32 : i32
  %c512_i32 = arith.constant 512 : i32
  %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %lower = arith.constant dense<0> : tensor<32xi32>
  %base_ptrs = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>

  %r = scf.for %iv = %c0_i32 to %c512_i32 step %c32_i32 iter_args(%acc = %cst) -> (tensor<32xf32>) : i32 {
    %iv_splat = tt.splat %iv : i32 -> tensor<32xi32>
    %offsets = arith.addi %iv_splat, %range : tensor<32xi32>
    %mask = arith.cmpi uge, %offsets, %lower : tensor<32xi32>
    %ptrs = tt.addptr %base_ptrs, %offsets : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %load = tt.load %ptrs, %mask, %cst : tensor<32x!tt.ptr<f32>>
    %new = arith.addf %acc, %load : tensor<32xf32>
    scf.yield %new : tensor<32xf32>
  }
  tt.return %r : tensor<32xf32>
}

// CHECK-LABEL: @test_uge_mask_always_true
// CHECK: scf.for
// CHECK-NOT: tt.load {{.*}}, {{.*}},
// CHECK:   tt.load {{%[0-9]+}} : tensor<32x!tt.ptr<f32>>

// -----

// COM: Test ugt (>u) predicate: (IV + range(1,33)) >u 0 is always true since
// COM: min element = 0 + 1 = 1 > 0.
tt.func public @test_ugt_mask_always_true(
    %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> tensor<32xf32> {
  %cst = arith.constant dense<0.000000e+00> : tensor<32xf32>
  %c0_i32 = arith.constant 0 : i32
  %c32_i32 = arith.constant 32 : i32
  %c512_i32 = arith.constant 512 : i32
  %range = tt.make_range {end = 33 : i32, start = 1 : i32} : tensor<32xi32>
  %lower = arith.constant dense<0> : tensor<32xi32>
  %base_ptrs = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>

  %r = scf.for %iv = %c0_i32 to %c512_i32 step %c32_i32 iter_args(%acc = %cst) -> (tensor<32xf32>) : i32 {
    %iv_splat = tt.splat %iv : i32 -> tensor<32xi32>
    %offsets = arith.addi %iv_splat, %range : tensor<32xi32>
    %mask = arith.cmpi ugt, %offsets, %lower : tensor<32xi32>
    %ptrs = tt.addptr %base_ptrs, %offsets : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %load = tt.load %ptrs, %mask, %cst : tensor<32x!tt.ptr<f32>>
    %new = arith.addf %acc, %load : tensor<32xf32>
    scf.yield %new : tensor<32xf32>
  }
  tt.return %r : tensor<32xf32>
}

// CHECK-LABEL: @test_ugt_mask_always_true
// CHECK: scf.for
// CHECK-NOT: tt.load {{.*}}, {{.*}},
// CHECK:   tt.load {{%[0-9]+}} : tensor<32x!tt.ptr<f32>>

// -----

// COM: Negative test -- sge AlwaysFalse: (IV + range(0,32)) >= 1000 is always
// COM: false since max element = 480 + 31 = 511 < 1000.
tt.func public @test_sge_mask_always_false(
    %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> tensor<32xf32> {
  %cst = arith.constant dense<0.000000e+00> : tensor<32xf32>
  %c0_i32 = arith.constant 0 : i32
  %c32_i32 = arith.constant 32 : i32
  %c512_i32 = arith.constant 512 : i32
  %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %bound = arith.constant dense<1000> : tensor<32xi32>
  %base_ptrs = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>

  %r = scf.for %iv = %c0_i32 to %c512_i32 step %c32_i32 iter_args(%acc = %cst) -> (tensor<32xf32>) : i32 {
    %iv_splat = tt.splat %iv : i32 -> tensor<32xi32>
    %offsets = arith.addi %iv_splat, %range : tensor<32xi32>
    %mask = arith.cmpi sge, %offsets, %bound : tensor<32xi32>
    %ptrs = tt.addptr %base_ptrs, %offsets : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %load = tt.load %ptrs, %mask, %cst : tensor<32x!tt.ptr<f32>>
    %new = arith.addf %acc, %load : tensor<32xf32>
    scf.yield %new : tensor<32xf32>
  }
  tt.return %r : tensor<32xf32>
}

// CHECK-LABEL: @test_sge_mask_always_false
// COM: AlwaysFalse: uses of the load are replaced by the other/padding value.
// COM: The dead masked load may remain but its result is unused.
// CHECK: scf.for
// CHECK:   arith.addf {{.*}}, %cst : tensor<32xf32>
// CHECK: }

// -----

// COM: Negative test -- sgt AlwaysFalse: (IV + range(0,32)) > 999 is always
// COM: false since max element = 480 + 31 = 511 <= 999.
tt.func public @test_sgt_mask_always_false(
    %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> tensor<32xf32> {
  %cst = arith.constant dense<0.000000e+00> : tensor<32xf32>
  %c0_i32 = arith.constant 0 : i32
  %c32_i32 = arith.constant 32 : i32
  %c512_i32 = arith.constant 512 : i32
  %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %bound = arith.constant dense<999> : tensor<32xi32>
  %base_ptrs = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>

  %r = scf.for %iv = %c0_i32 to %c512_i32 step %c32_i32 iter_args(%acc = %cst) -> (tensor<32xf32>) : i32 {
    %iv_splat = tt.splat %iv : i32 -> tensor<32xi32>
    %offsets = arith.addi %iv_splat, %range : tensor<32xi32>
    %mask = arith.cmpi sgt, %offsets, %bound : tensor<32xi32>
    %ptrs = tt.addptr %base_ptrs, %offsets : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %load = tt.load %ptrs, %mask, %cst : tensor<32x!tt.ptr<f32>>
    %new = arith.addf %acc, %load : tensor<32xf32>
    scf.yield %new : tensor<32xf32>
  }
  tt.return %r : tensor<32xf32>
}

// CHECK-LABEL: @test_sgt_mask_always_false
// COM: AlwaysFalse: uses of the load are replaced by the other/padding value.
// CHECK: scf.for
// CHECK:   arith.addf {{.*}}, %cst : tensor<32xf32>
// CHECK: }
