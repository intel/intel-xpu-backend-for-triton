// RUN: triton-opt %s -triton-intel-remove-masks | FileCheck %s

module {
  // COM: Test unnecessary mask removal with slt predicate
  // COM: Mask [IV+[0..31]] < splat(64) is always true for IV in [0, 32) step 32
  tt.func public @test_unnecessary_mask_slt(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> tensor<32xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<32xf32>
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %upper = arith.constant dense<64> : tensor<32xi32>
    %base_ptrs = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>

    %r = scf.for %iv = %c0_i32 to %c32_i32 step %c32_i32 iter_args(%acc = %cst) -> (tensor<32xf32>) : i32 {
      %iv_splat = tt.splat %iv : i32 -> tensor<32xi32>
      %offsets = arith.addi %iv_splat, %range : tensor<32xi32>
      %mask = arith.cmpi slt, %offsets, %upper : tensor<32xi32>
      %ptrs = tt.addptr %base_ptrs, %offsets : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
      %load = tt.load %ptrs, %mask, %cst : tensor<32x!tt.ptr<f32>>
      %new = arith.addf %acc, %load : tensor<32xf32>
      scf.yield %new : tensor<32xf32>
    }
    tt.return %r : tensor<32xf32>
  }

  // CHECK-LABEL: tt.func public @test_unnecessary_mask_slt
  // CHECK:         scf.for
  // CHECK:           [[OFFS:%.+]] = arith.addi {{%.+}}, {{%.+}} : tensor<32xi32>
  // CHECK:           [[PTR:%.+]] = tt.addptr {{%.+}}, [[OFFS]] : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
  // CHECK:           [[LOAD:%.+]] = tt.load [[PTR]] : tensor<32x!tt.ptr<f32>>
}

// -----

module {
  // COM: Test unnecessary mask removal with sle predicate
  // COM: Mask [IV+[0..31]] <= splat(63) is always true for IV in [0, 32) step 32
  tt.func public @test_unnecessary_mask_sle(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> tensor<32xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<32xf32>
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %upper = arith.constant dense<63> : tensor<32xi32>
    %base_ptrs = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>

    %r = scf.for %iv = %c0_i32 to %c32_i32 step %c32_i32 iter_args(%acc = %cst) -> (tensor<32xf32>) : i32 {
      %iv_splat = tt.splat %iv : i32 -> tensor<32xi32>
      %offsets = arith.addi %iv_splat, %range : tensor<32xi32>
      %mask = arith.cmpi sle, %offsets, %upper : tensor<32xi32>
      %ptrs = tt.addptr %base_ptrs, %offsets : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
      %load = tt.load %ptrs, %mask, %cst : tensor<32x!tt.ptr<f32>>
      %new = arith.addf %acc, %load : tensor<32xf32>
      scf.yield %new : tensor<32xf32>
    }
    tt.return %r : tensor<32xf32>
  }

  // CHECK-LABEL: tt.func public @test_unnecessary_mask_sle
  // CHECK:         scf.for
  // CHECK:           [[OFFS:%.+]] = arith.addi {{%.+}}, {{%.+}} : tensor<32xi32>
  // CHECK:           [[PTR:%.+]] = tt.addptr {{%.+}}, [[OFFS]] : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
  // CHECK:           [[LOAD:%.+]] = tt.load [[PTR]] : tensor<32x!tt.ptr<f32>>
}

// -----

module {
  // COM: Test unnecessary mask removal with ult predicate
  // COM: Mask [IV+[0..31]] <u splat(64) is always true for IV in [0, 32) step 32 (unsigned)
  tt.func public @test_unnecessary_mask_ult(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> tensor<32xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<32xf32>
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %upper = arith.constant dense<64> : tensor<32xi32>
    %base_ptrs = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>

    %r = scf.for %iv = %c0_i32 to %c32_i32 step %c32_i32 iter_args(%acc = %cst) -> (tensor<32xf32>) : i32 {
      %iv_splat = tt.splat %iv : i32 -> tensor<32xi32>
      %offsets = arith.addi %iv_splat, %range : tensor<32xi32>
      %mask = arith.cmpi ult, %offsets, %upper : tensor<32xi32>
      %ptrs = tt.addptr %base_ptrs, %offsets : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
      %load = tt.load %ptrs, %mask, %cst : tensor<32x!tt.ptr<f32>>
      %new = arith.addf %acc, %load : tensor<32xf32>
      scf.yield %new : tensor<32xf32>
    }
    tt.return %r : tensor<32xf32>
  }

  // CHECK-LABEL: tt.func public @test_unnecessary_mask_ult
  // CHECK:         scf.for
  // CHECK:           [[OFFS:%.+]] = arith.addi {{%.+}}, {{%.+}} : tensor<32xi32>
  // CHECK:           [[PTR:%.+]] = tt.addptr {{%.+}}, [[OFFS]] : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
  // CHECK:           [[LOAD:%.+]] = tt.load [[PTR]] : tensor<32x!tt.ptr<f32>>
}

// -----

module {
  // COM: Test unnecessary mask removal with ule predicate
  // COM: Mask [IV+[0..31]] <=u splat(63) is always true for IV in [0, 32) step 32 (unsigned)
  tt.func public @test_unnecessary_mask_ule(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> tensor<32xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<32xf32>
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %upper = arith.constant dense<63> : tensor<32xi32>
    %base_ptrs = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>

    %r = scf.for %iv = %c0_i32 to %c32_i32 step %c32_i32 iter_args(%acc = %cst) -> (tensor<32xf32>) : i32 {
      %iv_splat = tt.splat %iv : i32 -> tensor<32xi32>
      %offsets = arith.addi %iv_splat, %range : tensor<32xi32>
      %mask = arith.cmpi ule, %offsets, %upper : tensor<32xi32>
      %ptrs = tt.addptr %base_ptrs, %offsets : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
      %load = tt.load %ptrs, %mask, %cst : tensor<32x!tt.ptr<f32>>
      %new = arith.addf %acc, %load : tensor<32xf32>
      scf.yield %new : tensor<32xf32>
    }
    tt.return %r : tensor<32xf32>
  }

  // CHECK-LABEL: tt.func public @test_unnecessary_mask_ule
  // CHECK:         scf.for
  // CHECK:           [[OFFS:%.+]] = arith.addi {{%.+}}, {{%.+}} : tensor<32xi32>
  // CHECK:           [[PTR:%.+]] = tt.addptr {{%.+}}, [[OFFS]] : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
  // CHECK:           [[LOAD:%.+]] = tt.load [[PTR]] : tensor<32x!tt.ptr<f32>>
}

// -----

module {
  // COM: Negative test - unsupported predicates should preserve masks
  // COM: sgt predicate is not supported for mask removal
  tt.func public @test_unsupported_sgt(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> tensor<32xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<32xf32>
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %lower = arith.constant dense<-1> : tensor<32xi32>
    %base_ptrs = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>

    %r = scf.for %iv = %c0_i32 to %c32_i32 step %c32_i32 iter_args(%acc = %cst) -> (tensor<32xf32>) : i32 {
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

  // CHECK-LABEL: tt.func public @test_unsupported_sgt
  // CHECK:         scf.for
  // CHECK:           [[MASK:%.+]] = arith.cmpi sgt, {{%.+}}, {{%.+}} : tensor<32xi32>
  // CHECK:           [[PTR:%.+]] = tt.addptr {{%.+}}, {{%.+}} : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
  // CHECK:           [[LOAD:%.+]] = tt.load [[PTR]], [[MASK]], {{%.+}} : tensor<32x!tt.ptr<f32>>
}

// -----

module {
  // COM: Negative test - eq predicate should preserve masks
  tt.func public @test_unsupported_eq(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> tensor<32xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<32xf32>
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %target = arith.constant dense<0> : tensor<32xi32>
    %base_ptrs = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>

    %r = scf.for %iv = %c0_i32 to %c32_i32 step %c32_i32 iter_args(%acc = %cst) -> (tensor<32xf32>) : i32 {
      %iv_splat = tt.splat %iv : i32 -> tensor<32xi32>
      %offsets = arith.addi %iv_splat, %range : tensor<32xi32>
      %mask = arith.cmpi eq, %offsets, %target : tensor<32xi32>
      %ptrs = tt.addptr %base_ptrs, %offsets : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
      %load = tt.load %ptrs, %mask, %cst : tensor<32x!tt.ptr<f32>>
      %new = arith.addf %acc, %load : tensor<32xf32>
      scf.yield %new : tensor<32xf32>
    }
    tt.return %r : tensor<32xf32>
  }

  // CHECK-LABEL: tt.func public @test_unsupported_eq
  // CHECK:         scf.for
  // CHECK:           [[MASK:%.+]] = arith.cmpi eq, {{%.+}}, {{%.+}} : tensor<32xi32>
  // CHECK:           [[PTR:%.+]] = tt.addptr {{%.+}}, {{%.+}} : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
  // CHECK:           [[LOAD:%.+]] = tt.load [[PTR]], [[MASK]], {{%.+}} : tensor<32x!tt.ptr<f32>>
}

// -----

module {
  // COM: Negative test - ne predicate should preserve masks
  tt.func public @test_unsupported_ne(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> tensor<32xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<32xf32>
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %target = arith.constant dense<100> : tensor<32xi32>
    %base_ptrs = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>

    %r = scf.for %iv = %c0_i32 to %c32_i32 step %c32_i32 iter_args(%acc = %cst) -> (tensor<32xf32>) : i32 {
      %iv_splat = tt.splat %iv : i32 -> tensor<32xi32>
      %offsets = arith.addi %iv_splat, %range : tensor<32xi32>
      %mask = arith.cmpi ne, %offsets, %target : tensor<32xi32>
      %ptrs = tt.addptr %base_ptrs, %offsets : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
      %load = tt.load %ptrs, %mask, %cst : tensor<32x!tt.ptr<f32>>
      %new = arith.addf %acc, %load : tensor<32xf32>
      scf.yield %new : tensor<32xf32>
    }
    tt.return %r : tensor<32xf32>
  }

  // CHECK-LABEL: tt.func public @test_unsupported_ne
  // CHECK:         scf.for
  // CHECK:           [[MASK:%.+]] = arith.cmpi ne, {{%.+}}, {{%.+}} : tensor<32xi32>
  // CHECK:           [[PTR:%.+]] = tt.addptr {{%.+}}, {{%.+}} : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
  // CHECK:           [[LOAD:%.+]] = tt.load [[PTR]], [[MASK]], {{%.+}} : tensor<32x!tt.ptr<f32>>
}
