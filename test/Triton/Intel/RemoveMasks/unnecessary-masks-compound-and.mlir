// RUN: triton-opt %s -triton-intel-remove-masks | FileCheck %s

module {
  // COM: Test compound AND with both conjuncts always true.
  // COM: ([IV+[0..31]] < 64) AND ([IV+[0..31]] < 64) -- the load's mask is
  // COM: dropped; the arith.andi and its cmp operands survive as dead IR.
  tt.func public @test_compound_and_both_true(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> tensor<32xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<32xf32>
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %upper = arith.constant dense<64> : tensor<32xi32>
    %base_ptrs = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>

    %r = scf.for %iv = %c0_i32 to %c32_i32 step %c32_i32 iter_args(%acc = %cst) -> (tensor<32xf32>) : i32 {
      %iv_splat = tt.splat %iv : i32 -> tensor<32xi32>
      %offsets = arith.addi %iv_splat, %range : tensor<32xi32>
      %mask1 = arith.cmpi slt, %offsets, %upper : tensor<32xi32>
      %mask2 = arith.cmpi slt, %offsets, %upper : tensor<32xi32>
      %mask = arith.andi %mask1, %mask2 : tensor<32xi1>
      %ptrs = tt.addptr %base_ptrs, %offsets : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
      %load = tt.load %ptrs, %mask, %cst : tensor<32x!tt.ptr<f32>>
      %new = arith.addf %acc, %load : tensor<32xf32>
      scf.yield %new : tensor<32xf32>
    }
    tt.return %r : tensor<32xf32>
  }

  // CHECK-LABEL: tt.func public @test_compound_and_both_true
  // CHECK:         scf.for
  // CHECK:           [[OFFS:%.+]] = arith.addi {{%.+}}, {{%.+}} : tensor<32xi32>
  // CHECK:           [[PTR:%.+]] = tt.addptr {{%.+}}, [[OFFS]] : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
  // CHECK:           [[LOAD:%.+]] = tt.load [[PTR]] : tensor<32x!tt.ptr<f32>>
}

// -----

module {
  // COM: Test compound AND with one conjunct always true, one dynamic.
  // COM: ([IV+[0..31]] < 64) AND ([IV+[0..31]] < dynamic) -- the overall
  // COM: classification is Unknown, so the load keeps its full andi mask.
  tt.func public @test_compound_and_one_true(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32) -> tensor<32xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<32xf32>
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %upper = arith.constant dense<64> : tensor<32xi32>
    %dyn_upper = tt.splat %arg1 : i32 -> tensor<32xi32>
    %base_ptrs = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>

    %r = scf.for %iv = %c0_i32 to %c32_i32 step %c32_i32 iter_args(%acc = %cst) -> (tensor<32xf32>) : i32 {
      %iv_splat = tt.splat %iv : i32 -> tensor<32xi32>
      %offsets = arith.addi %iv_splat, %range : tensor<32xi32>
      %mask1 = arith.cmpi slt, %offsets, %upper : tensor<32xi32>
      %mask2 = arith.cmpi slt, %offsets, %dyn_upper : tensor<32xi32>
      %mask = arith.andi %mask1, %mask2 : tensor<32xi1>
      %ptrs = tt.addptr %base_ptrs, %offsets : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
      %load = tt.load %ptrs, %mask, %cst : tensor<32x!tt.ptr<f32>>
      %new = arith.addf %acc, %load : tensor<32xf32>
      scf.yield %new : tensor<32xf32>
    }
    tt.return %r : tensor<32xf32>
  }

  // CHECK-LABEL: tt.func public @test_compound_and_one_true
  // CHECK:         scf.for
  // CHECK:           [[OFFS:%.+]] = arith.addi {{%.+}}, {{%.+}} : tensor<32xi32>
  // CHECK-DAG:       [[M1:%.+]] = arith.cmpi slt, [[OFFS]], {{%.+}} : tensor<32xi32>
  // CHECK-DAG:       [[M2:%.+]] = arith.cmpi slt, [[OFFS]], {{%.+}} : tensor<32xi32>
  // CHECK:           [[MASK:%.+]] = arith.andi {{%.+}}, {{%.+}} : tensor<32xi1>
  // CHECK:           [[PTR:%.+]] = tt.addptr {{%.+}}, [[OFFS]] : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
  // CHECK:           [[LOAD:%.+]] = tt.load [[PTR]], [[MASK]], {{%.+}} : tensor<32x!tt.ptr<f32>>
}

// -----

module {
  // COM: Test nested compound AND with all conjuncts always true.
  // COM: (([IV+[0..31]] < 64) AND ([IV+[0..31]] < 64)) AND ([IV+[0..31]] < 64)
  // COM: -- the load's mask is dropped; the andi tree survives as dead IR.
  tt.func public @test_nested_and_all_true(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> tensor<32xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<32xf32>
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %upper = arith.constant dense<64> : tensor<32xi32>
    %base_ptrs = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>

    %r = scf.for %iv = %c0_i32 to %c32_i32 step %c32_i32 iter_args(%acc = %cst) -> (tensor<32xf32>) : i32 {
      %iv_splat = tt.splat %iv : i32 -> tensor<32xi32>
      %offsets = arith.addi %iv_splat, %range : tensor<32xi32>
      %mask1 = arith.cmpi slt, %offsets, %upper : tensor<32xi32>
      %mask2 = arith.cmpi slt, %offsets, %upper : tensor<32xi32>
      %mask3 = arith.andi %mask1, %mask2 : tensor<32xi1>
      %mask4 = arith.cmpi slt, %offsets, %upper : tensor<32xi32>
      %mask = arith.andi %mask3, %mask4 : tensor<32xi1>
      %ptrs = tt.addptr %base_ptrs, %offsets : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
      %load = tt.load %ptrs, %mask, %cst : tensor<32x!tt.ptr<f32>>
      %new = arith.addf %acc, %load : tensor<32xf32>
      scf.yield %new : tensor<32xf32>
    }
    tt.return %r : tensor<32xf32>
  }

  // CHECK-LABEL: tt.func public @test_nested_and_all_true
  // CHECK:         scf.for
  // CHECK:           [[OFFS:%.+]] = arith.addi {{%.+}}, {{%.+}} : tensor<32xi32>
  // CHECK:           [[PTR:%.+]] = tt.addptr {{%.+}}, [[OFFS]] : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
  // CHECK:           [[LOAD:%.+]] = tt.load [[PTR]] : tensor<32x!tt.ptr<f32>>
}

// -----

module {
  // COM: Test mixed predicates in compound AND.
  // COM: ([IV+[0..31]] < 64) AND ([IV+[0..31]] <= 63) -- both always true,
  // COM: the load's mask is dropped.
  tt.func public @test_mixed_predicates_and(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> tensor<32xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<32xf32>
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %upper1 = arith.constant dense<64> : tensor<32xi32>
    %upper2 = arith.constant dense<63> : tensor<32xi32>
    %base_ptrs = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>

    %r = scf.for %iv = %c0_i32 to %c32_i32 step %c32_i32 iter_args(%acc = %cst) -> (tensor<32xf32>) : i32 {
      %iv_splat = tt.splat %iv : i32 -> tensor<32xi32>
      %offsets = arith.addi %iv_splat, %range : tensor<32xi32>
      %mask1 = arith.cmpi slt, %offsets, %upper1 : tensor<32xi32>
      %mask2 = arith.cmpi sle, %offsets, %upper2 : tensor<32xi32>
      %mask = arith.andi %mask1, %mask2 : tensor<32xi1>
      %ptrs = tt.addptr %base_ptrs, %offsets : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
      %load = tt.load %ptrs, %mask, %cst : tensor<32x!tt.ptr<f32>>
      %new = arith.addf %acc, %load : tensor<32xf32>
      scf.yield %new : tensor<32xf32>
    }
    tt.return %r : tensor<32xf32>
  }

  // CHECK-LABEL: tt.func public @test_mixed_predicates_and
  // CHECK:         scf.for
  // CHECK:           [[OFFS:%.+]] = arith.addi {{%.+}}, {{%.+}} : tensor<32xi32>
  // CHECK:           [[PTR:%.+]] = tt.addptr {{%.+}}, [[OFFS]] : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
  // CHECK:           [[LOAD:%.+]] = tt.load [[PTR]] : tensor<32x!tt.ptr<f32>>
}

// -----

module {
  // COM: Negative test - OR of comparisons should preserve masks
  // COM: ([IV+[0..31]] < 64) OR ([IV+[0..31]] < dynamic) is not simplified
  tt.func public @test_unsupported_or(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32) -> tensor<32xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<32xf32>
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %upper = arith.constant dense<64> : tensor<32xi32>
    %dyn_upper = tt.splat %arg1 : i32 -> tensor<32xi32>
    %base_ptrs = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>

    %r = scf.for %iv = %c0_i32 to %c32_i32 step %c32_i32 iter_args(%acc = %cst) -> (tensor<32xf32>) : i32 {
      %iv_splat = tt.splat %iv : i32 -> tensor<32xi32>
      %offsets = arith.addi %iv_splat, %range : tensor<32xi32>
      %mask1 = arith.cmpi slt, %offsets, %upper : tensor<32xi32>
      %mask2 = arith.cmpi slt, %offsets, %dyn_upper : tensor<32xi32>
      %mask = arith.ori %mask1, %mask2 : tensor<32xi1>
      %ptrs = tt.addptr %base_ptrs, %offsets : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
      %load = tt.load %ptrs, %mask, %cst : tensor<32x!tt.ptr<f32>>
      %new = arith.addf %acc, %load : tensor<32xf32>
      scf.yield %new : tensor<32xf32>
    }
    tt.return %r : tensor<32xf32>
  }

  // CHECK-LABEL: tt.func public @test_unsupported_or
  // CHECK:         scf.for
  // CHECK:           [[MASK:%.+]] = arith.ori {{%.+}}, {{%.+}} : tensor<32xi1>
  // CHECK:           [[PTR:%.+]] = tt.addptr {{%.+}}, {{%.+}} : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
  // CHECK:           [[LOAD:%.+]] = tt.load [[PTR]], [[MASK]], {{%.+}} : tensor<32x!tt.ptr<f32>>
}

// -----

module {
  // COM: Test compound AND whose mask is also used by a select.
  // COM: The load's mask and the select's condition resolve to AlwaysTrue, so
  // COM: the load is replaced by an unmasked load and the select's trueValue
  // COM: (that unmasked load) is propagated to the downstream arith.addf.
  tt.func public @test_and_used_by_select(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> tensor<32xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<32xf32>
    %cst_1 = arith.constant dense<1.000000e+00> : tensor<32xf32>
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %upper = arith.constant dense<64> : tensor<32xi32>
    %base_ptrs = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>

    %r = scf.for %iv = %c0_i32 to %c32_i32 step %c32_i32 iter_args(%acc = %cst) -> (tensor<32xf32>) : i32 {
      %iv_splat = tt.splat %iv : i32 -> tensor<32xi32>
      %offsets = arith.addi %iv_splat, %range : tensor<32xi32>
      %mask1 = arith.cmpi slt, %offsets, %upper : tensor<32xi32>
      %mask2 = arith.cmpi slt, %offsets, %upper : tensor<32xi32>
      %mask = arith.andi %mask1, %mask2 : tensor<32xi1>
      %ptrs = tt.addptr %base_ptrs, %offsets : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
      %load = tt.load %ptrs, %mask, %cst : tensor<32x!tt.ptr<f32>>
      %selected = arith.select %mask, %load, %cst_1 : tensor<32xi1>, tensor<32xf32>
      %new = arith.addf %acc, %selected : tensor<32xf32>
      scf.yield %new : tensor<32xf32>
    }
    tt.return %r : tensor<32xf32>
  }

  // CHECK-LABEL: tt.func public @test_and_used_by_select
  // CHECK:         scf.for [[IV:%.+]] = %{{.+}} to %{{.+}} step %{{.+}} iter_args([[ACC:%.+]] = %{{.+}})
  // CHECK:           [[PTR:%.+]] = tt.addptr {{%.+}}, {{%.+}} : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
  // CHECK:           [[LOAD:%.+]] = tt.load [[PTR]] : tensor<32x!tt.ptr<f32>>
  // COM: The select's trueValue (the unmasked load) is propagated to the addf.
  // CHECK:           [[NEW:%.+]] = arith.addf [[ACC]], [[LOAD]] : tensor<32xf32>
  // CHECK:           scf.yield [[NEW]]
}
