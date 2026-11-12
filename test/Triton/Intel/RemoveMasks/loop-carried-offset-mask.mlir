// RUN: triton-opt %s -split-input-file -triton-intel-remove-masks | FileCheck %s

module {
  // COM: A bounds mask on a loop-carried offset that the loop *updates* must be
  // COM: kept. `getFinalValue` used to resolve the `%offs` iteration argument
  // COM: to its init `tt.make_range(0, 32)`, describing the first iteration
  // COM: only; the loop adds 4096 each time, so the offsets are 0, 4096, 8192,
  // COM: 12288 and the mask is false for every lane of the last three
  // COM: iterations. Judging it against the init value declared it always-true
  // COM: and `dropMask` deleted it (along with `other`), leaving the load
  // COM: reading out of bounds and returning what it found instead of `other`
  // COM: (#8060).
  tt.func public @loop_carried_offset_mask(
      %src: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %dst: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32xf16>
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c128_i32 = arith.constant 128 : i32
    %cst_0 = arith.constant dense<4096> : tensor<32xi32>

    %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %res = scf.for %iv = %c0_i32 to %c128_i32 step %c32_i32
        iter_args(%offs = %range) -> (tensor<32xi32>) : i32 {
      %iv_splat = tt.splat %iv : i32 -> tensor<32xi32>
      %idx = arith.addi %iv_splat, %offs : tensor<32xi32>
      %mask = arith.cmpi slt, %idx, %cst_0 : tensor<32xi32>

      %src_iv = tt.addptr %src, %iv : !tt.ptr<f16>, i32
      %src_splat = tt.splat %src_iv : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>>
      %src_ptr = tt.addptr %src_splat, %offs : tensor<32x!tt.ptr<f16>>, tensor<32xi32>
      %loaded = tt.load %src_ptr, %mask, %cst : tensor<32x!tt.ptr<f16>>

      %dst_iv = tt.addptr %dst, %iv : !tt.ptr<f16>, i32
      %dst_splat = tt.splat %dst_iv : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>>
      %dst_ptr = tt.addptr %dst_splat, %range : tensor<32x!tt.ptr<f16>>, tensor<32xi32>
      tt.store %dst_ptr, %loaded : tensor<32x!tt.ptr<f16>>

      %next_offs = arith.addi %offs, %cst_0 : tensor<32xi32>
      scf.yield %next_offs : tensor<32xi32>
    }
    tt.return
  }

  // CHECK-LABEL: @loop_carried_offset_mask
  // CHECK: scf.for
  // CHECK:   %[[MASK:.*]] = arith.cmpi slt
  // CHECK-NOT: tt.load
  // CHECK:   %[[LOAD:.*]] = tt.load {{.*}}, %[[MASK]], %cst : tensor<32x!tt.ptr<f16>>
  // CHECK:   tt.store {{.*}}, %[[LOAD]] :
  // CHECK: }
}

// -----

module {
  // COM: Same shape, but the compared-against *bound* is the loop-carried value
  // COM: the loop updates. Resolving it to its init `dense<4096>` proved the
  // COM: mask always-true for the first iteration only.
  tt.func public @loop_carried_bound_mask(
      %src: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %dst: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32xf16>
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c128_i32 = arith.constant 128 : i32
    %cst_ub = arith.constant dense<4096> : tensor<32xi32>
    %cst_dec = arith.constant dense<4096> : tensor<32xi32>

    %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %src_splat = tt.splat %src : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>>
    %res = scf.for %iv = %c0_i32 to %c128_i32 step %c32_i32
        iter_args(%bound = %cst_ub) -> (tensor<32xi32>) : i32 {
      %iv_splat = tt.splat %iv : i32 -> tensor<32xi32>
      %idx = arith.addi %iv_splat, %range : tensor<32xi32>
      %mask = arith.cmpi slt, %idx, %bound : tensor<32xi32>

      %loaded = tt.load %src_splat, %mask, %cst : tensor<32x!tt.ptr<f16>>

      %dst_iv = tt.addptr %dst, %iv : !tt.ptr<f16>, i32
      %dst_splat = tt.splat %dst_iv : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>>
      %dst_ptr = tt.addptr %dst_splat, %range : tensor<32x!tt.ptr<f16>>, tensor<32xi32>
      tt.store %dst_ptr, %loaded : tensor<32x!tt.ptr<f16>>

      %next_bound = arith.subi %bound, %cst_dec : tensor<32xi32>
      scf.yield %next_bound : tensor<32xi32>
    }
    tt.return
  }

  // CHECK-LABEL: @loop_carried_bound_mask
  // CHECK: scf.for
  // CHECK:   %[[MASK:.*]] = arith.cmpi slt
  // CHECK-NOT: tt.load
  // CHECK:   %[[LOAD:.*]] = tt.load {{.*}}, %[[MASK]], %cst : tensor<32x!tt.ptr<f16>>
  // CHECK:   tt.store {{.*}}, %[[LOAD]] :
  // CHECK: }
}

// -----

module {
  // COM: Two iteration arguments start at the lower bound: `%k` advances like
  // COM: the induction variable, `%j` by 4096. The mask is on `%j`, so it must
  // COM: not be admitted on the strength of `%k` being IV-equivalent: `%j` is
  // COM: 0, 4096, 8192, 12288 and the mask is false for every lane after the
  // COM: first iteration.
  tt.func public @sibling_iter_arg_mask(
      %src: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %dst: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32xf16>
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c128_i32 = arith.constant 128 : i32
    %c4096_i32 = arith.constant 4096 : i32
    %cst_ub = arith.constant dense<4096> : tensor<32xi32>

    %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %src_splat = tt.splat %src : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>>
    %res:2 = scf.for %iv = %c0_i32 to %c128_i32 step %c32_i32
        iter_args(%k = %c0_i32, %j = %c0_i32) -> (i32, i32) : i32 {
      %j_splat = tt.splat %j : i32 -> tensor<32xi32>
      %idx = arith.addi %j_splat, %range : tensor<32xi32>
      %mask = arith.cmpi slt, %idx, %cst_ub : tensor<32xi32>

      %src_ptr = tt.addptr %src_splat, %idx : tensor<32x!tt.ptr<f16>>, tensor<32xi32>
      %loaded = tt.load %src_ptr, %mask, %cst : tensor<32x!tt.ptr<f16>>

      %dst_k = tt.addptr %dst, %k : !tt.ptr<f16>, i32
      %dst_splat = tt.splat %dst_k : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>>
      %dst_ptr = tt.addptr %dst_splat, %range : tensor<32x!tt.ptr<f16>>, tensor<32xi32>
      tt.store %dst_ptr, %loaded : tensor<32x!tt.ptr<f16>>

      %next_k = arith.addi %k, %c32_i32 : i32
      %next_j = arith.addi %j, %c4096_i32 : i32
      scf.yield %next_k, %next_j : i32, i32
    }
    tt.return
  }

  // CHECK-LABEL: @sibling_iter_arg_mask
  // CHECK: scf.for
  // CHECK:   %[[MASK:.*]] = arith.cmpi slt
  // CHECK-NOT: tt.load
  // CHECK:   %[[LOAD:.*]] = tt.load {{.*}}, %[[MASK]], %cst : tensor<32x!tt.ptr<f16>>
  // CHECK:   tt.store {{.*}}, %[[LOAD]] :
  // CHECK: }
}

// -----

module {
  // COM: `%k` advances by the loop step but starts at 4064, not at the lower
  // COM: bound, so it is `%iv + 4064` and not IV-equivalent. The mask holds on
  // COM: the first iteration only: `%k` is then 4096, 4128, 4160 and every lane
  // COM: is false. Judged by the induction variable's range it would look
  // COM: always-true and be removed.
  tt.func public @iter_arg_init_not_lower_bound_mask(
      %src: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %dst: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32xf16>
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c128_i32 = arith.constant 128 : i32
    %c4064_i32 = arith.constant 4064 : i32
    %cst_ub = arith.constant dense<4096> : tensor<32xi32>

    %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %src_splat = tt.splat %src : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>>
    %res = scf.for %iv = %c0_i32 to %c128_i32 step %c32_i32
        iter_args(%k = %c4064_i32) -> (i32) : i32 {
      %k_splat = tt.splat %k : i32 -> tensor<32xi32>
      %idx = arith.addi %k_splat, %range : tensor<32xi32>
      %mask = arith.cmpi slt, %idx, %cst_ub : tensor<32xi32>

      %src_ptr = tt.addptr %src_splat, %idx : tensor<32x!tt.ptr<f16>>, tensor<32xi32>
      %loaded = tt.load %src_ptr, %mask, %cst : tensor<32x!tt.ptr<f16>>

      %dst_iv = tt.addptr %dst, %iv : !tt.ptr<f16>, i32
      %dst_splat = tt.splat %dst_iv : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>>
      %dst_ptr = tt.addptr %dst_splat, %range : tensor<32x!tt.ptr<f16>>, tensor<32xi32>
      tt.store %dst_ptr, %loaded : tensor<32x!tt.ptr<f16>>

      %next_k = arith.addi %k, %c32_i32 : i32
      scf.yield %next_k : i32
    }
    tt.return
  }

  // CHECK-LABEL: @iter_arg_init_not_lower_bound_mask
  // CHECK: scf.for
  // CHECK:   %[[MASK:.*]] = arith.cmpi slt
  // CHECK-NOT: tt.load
  // CHECK:   %[[LOAD:.*]] = tt.load {{.*}}, %[[MASK]], %cst : tensor<32x!tt.ptr<f16>>
  // CHECK:   tt.store {{.*}}, %[[LOAD]] :
  // CHECK: }
}

// -----

module {
  // COM: The mask itself is the loop-carried value: `%mask0` holds for the
  // COM: first iteration and the loop recomputes it from the advanced offsets.
  // COM: Resolving `%mask` to `%mask0` made it look loop invariant, so it was
  // COM: versioned on `%mask0`'s condition (`n > 31`) and the loop taking that
  // COM: branch loaded unmasked for every iteration.
  tt.func public @loop_carried_mask(
      %src: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %dst: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %n: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32xf16>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %cst_32 = arith.constant dense<32> : tensor<32xi32>

    %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %n_splat = tt.splat %n : i32 -> tensor<32xi32>
    %mask0 = arith.cmpi slt, %range, %n_splat : tensor<32xi32>
    %src_splat = tt.splat %src : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>>
    %dst_splat = tt.splat %dst : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>>
    %res:2 = scf.for %iv = %c0_i32 to %c4_i32 step %c1_i32
        iter_args(%offs = %range, %mask = %mask0) -> (tensor<32xi32>, tensor<32xi1>) : i32 {
      %src_ptr = tt.addptr %src_splat, %offs : tensor<32x!tt.ptr<f16>>, tensor<32xi32>
      %loaded = tt.load %src_ptr, %mask, %cst : tensor<32x!tt.ptr<f16>>
      %dst_ptr = tt.addptr %dst_splat, %offs : tensor<32x!tt.ptr<f16>>, tensor<32xi32>
      tt.store %dst_ptr, %loaded : tensor<32x!tt.ptr<f16>>

      %next_offs = arith.addi %offs, %cst_32 : tensor<32xi32>
      %next_mask = arith.cmpi slt, %next_offs, %n_splat : tensor<32xi32>
      scf.yield %next_offs, %next_mask : tensor<32xi32>, tensor<32xi1>
    }
    tt.return
  }

  // CHECK-LABEL: @loop_carried_mask
  // CHECK-NOT: scf.if
  // CHECK: scf.for {{.*}} iter_args({{.*}}, %[[MASK:arg[0-9]+]] = %{{[0-9]+}})
  // CHECK-NOT: tt.load
  // CHECK:   %[[LOAD:.*]] = tt.load {{.*}}, %[[MASK]], %cst : tensor<32x!tt.ptr<f16>>
  // CHECK:   tt.store {{.*}}, %[[LOAD]] :
  // CHECK: }
}

// -----

module {
  // COM: The mask is in the canonical form `END-1 < N-i*END` that
  // COM: `CanonicalMaskValidator` versions the loop on, but its `N` is the
  // COM: iteration argument `%m`, which the loop decrements. Resolving `%m` to
  // COM: its init `%n` matched the `%n` in the upper bound `(%n+31)/32`, so
  // COM: the mask looked implied by `%n % 32 == 0 && %n > 32` and the "then"
  // COM: loop dropped it. The real bound is `%n-2*i*32`: for `%n` = 64
  // COM: the condition holds, yet the second iteration's mask is false for
  // COM: every lane, which must suppress that load rather than return `%src`
  // COM: data in place of `other`.
  tt.func public @canonical_mask_loop_carried_n(
      %src: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %dst: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %n: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32xf16>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c31_i32 = arith.constant 31 : i32
    %c32_i32 = arith.constant 32 : i32

    %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %num = arith.addi %n, %c31_i32 : i32
    %ub = arith.divsi %num, %c32_i32 : i32
    %res = scf.for %iv = %c0_i32 to %ub step %c1_i32 iter_args(%m = %n) -> (i32) : i32 {
      %off = arith.muli %iv, %c32_i32 : i32
      %bound = arith.subi %m, %off : i32
      %bound_splat = tt.splat %bound : i32 -> tensor<32xi32>
      %mask = arith.cmpi slt, %range, %bound_splat : tensor<32xi32>

      %src_iv = tt.addptr %src, %off : !tt.ptr<f16>, i32
      %src_splat = tt.splat %src_iv : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>>
      %src_ptr = tt.addptr %src_splat, %range : tensor<32x!tt.ptr<f16>>, tensor<32xi32>
      %loaded = tt.load %src_ptr, %mask, %cst : tensor<32x!tt.ptr<f16>>

      %dst_iv = tt.addptr %dst, %off : !tt.ptr<f16>, i32
      %dst_splat = tt.splat %dst_iv : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>>
      %dst_ptr = tt.addptr %dst_splat, %range : tensor<32x!tt.ptr<f16>>, tensor<32xi32>
      tt.store %dst_ptr, %loaded : tensor<32x!tt.ptr<f16>>

      %next_m = arith.subi %m, %c32_i32 : i32
      scf.yield %next_m : i32
    }
    tt.return
  }

  // CHECK-LABEL: @canonical_mask_loop_carried_n
  // CHECK-NOT: scf.if
  // CHECK: scf.for
  // CHECK:   %[[MASK:.*]] = arith.cmpi slt
  // CHECK-NOT: tt.load
  // CHECK:   %[[LOAD:.*]] = tt.load {{.*}}, %[[MASK]], %cst : tensor<32x!tt.ptr<f16>>
  // CHECK:   tt.store {{.*}}, %[[LOAD]] :
  // CHECK: }
}
