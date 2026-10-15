// RUN: env TRITON_INTEL_SYMBOLIC_MASKS=1 triton-opt %s -split-input-file -triton-intel-remove-masks -canonicalize | FileCheck %s --check-prefixes=CHECK,SYM
// RUN: triton-opt %s -split-input-file -triton-intel-remove-masks -canonicalize | FileCheck %s --check-prefixes=CHECK,LEGACY

// COM: Inductor reduction shape: no legacy validator fires; the symbolic
// COM: validator versions the loop on rnumel % 64 == 0 and unmasks the then-copy.
// COM: -canonicalize removes the dead masked load dropMask leaves behind, so the
// COM: CHECK-NOT below is meaningful.

// CHECK-LABEL: tt.func @inductor_reduction
tt.func @inductor_reduction(%ptr: !tt.ptr<f32>, %rnumel: i32) {
  %c0 = arith.constant 0 : i32
  %c64 = arith.constant 64 : i32
  %lane = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %ns = tt.splat %rnumel : i32 -> tensor<64xi32>
  %ps = tt.splat %ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
  // SYM-DAG:  %[[C64:.*]] = arith.constant 64 : i64
  // SYM-DAG:  %[[C0:.*]] = arith.constant 0 : i64
  // SYM:      %[[REM:.*]] = arith.remsi %{{.*}}, %[[C64]] : i64
  // SYM:      %[[GUARD:.*]] = arith.cmpi eq, %[[REM]], %[[C0]] : i64
  // SYM:      scf.if %[[GUARD]] {
  // SYM:        scf.for
  // SYM-NOT:      tt.load %{{.*}}, %{{.*}} :
  // SYM:          tt.load %{{[^,]*}} : tensor<64x!tt.ptr<f32>>
  // SYM:      } else {
  // SYM:        scf.for
  // SYM:          tt.load %{{.*}}, %{{.*}} : tensor<64x!tt.ptr<f32>>
  // LEGACY-NOT: scf.if
  // LEGACY:     tt.load %{{.*}}, %{{.*}} : tensor<64x!tt.ptr<f32>>
  scf.for %r = %c0 to %rnumel step %c64 : i32 {
    %rs = tt.splat %r : i32 -> tensor<64xi32>
    %idx = arith.addi %rs, %lane : tensor<64xi32>
    %mask = arith.cmpi slt, %idx, %ns : tensor<64xi32>
    %p = tt.addptr %ps, %idx : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %v = tt.load %p, %mask : tensor<64x!tt.ptr<f32>>
    tt.store %p, %v, %mask : tensor<64x!tt.ptr<f32>>
    scf.yield
  }
  tt.return
}

// -----

// COM: Two loads, only one provable. The second mask compares against a value
// COM: loaded *inside* the loop body, which is loop-varying and has no range, so
// COM: it is Opaque and the query is Unknown. The loop is still versioned for the
// COM: first load, and the second keeps its mask in both copies. Loaded before
// COM: the loop the same value would be a guardable invariant.

// CHECK-LABEL: tt.func @two_loads_one_provable
tt.func @two_loads_one_provable(%ptr: !tt.ptr<f32>, %qtr: !tt.ptr<i32>, %rnumel: i32) {
  %c0 = arith.constant 0 : i32
  %c64 = arith.constant 64 : i32
  %lane = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %ns = tt.splat %rnumel : i32 -> tensor<64xi32>
  %ps = tt.splat %ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
  // COM: Checks follow program order: provable tensor load, scalar load, then
  // COM: the unprovable tensor load.
  // SYM-DAG:  %[[C64:.*]] = arith.constant 64 : i64
  // SYM-DAG:  %[[C0:.*]] = arith.constant 0 : i64
  // SYM:      %[[REM:.*]] = arith.remsi %{{.*}}, %[[C64]] : i64
  // SYM:      %[[GUARD:.*]] = arith.cmpi eq, %[[REM]], %[[C0]] : i64
  // SYM:      scf.if %[[GUARD]] {
  // SYM:        scf.for
  // SYM:          tt.load %{{[^,]*}} : tensor<64x!tt.ptr<f32>>
  // SYM:          tt.load %{{.*}} : !tt.ptr<i32>
  // SYM:          tt.load %{{.*}}, %{{.*}} : tensor<64x!tt.ptr<f32>>
  // SYM:      } else {
  // SYM:        scf.for
  // SYM:          tt.load %{{.*}}, %{{.*}} : tensor<64x!tt.ptr<f32>>
  // SYM:          tt.load %{{.*}} : !tt.ptr<i32>
  // SYM:          tt.load %{{.*}}, %{{.*}} : tensor<64x!tt.ptr<f32>>
  // LEGACY-NOT: scf.if
  scf.for %r = %c0 to %rnumel step %c64 : i32 {
    %rs = tt.splat %r : i32 -> tensor<64xi32>
    %idx = arith.addi %rs, %lane : tensor<64xi32>
    %mask = arith.cmpi slt, %idx, %ns : tensor<64xi32>
    %p = tt.addptr %ps, %idx : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %v = tt.load %p, %mask : tensor<64x!tt.ptr<f32>>
    %other = tt.load %qtr : !tt.ptr<i32>
    %os = tt.splat %other : i32 -> tensor<64xi32>
    %mask2 = arith.cmpi slt, %idx, %os : tensor<64xi32>
    %v2 = tt.load %p, %mask2 : tensor<64x!tt.ptr<f32>>
    tt.store %p, %v, %mask : tensor<64x!tt.ptr<f32>>
    tt.store %p, %v2, %mask2 : tensor<64x!tt.ptr<f32>>
    scf.yield
  }
  tt.return
}

// -----

// COM: An arith.select consuming the provable mask. The symbolic validator
// COM: collects selects as well as loads, and dropMask replaces the select with
// COM: its true value, so the then-copy stores %one directly.

// CHECK-LABEL: tt.func @select_mask
tt.func @select_mask(%ptr: !tt.ptr<f32>, %rnumel: i32) {
  %c0 = arith.constant 0 : i32
  %c64 = arith.constant 64 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<64xf32>
  %one = arith.constant dense<1.000000e+00> : tensor<64xf32>
  %lane = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %ns = tt.splat %rnumel : i32 -> tensor<64xi32>
  %ps = tt.splat %ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
  // SYM:      %[[ONE:.*]] = arith.constant dense<1.000000e+00> : tensor<64xf32>
  // SYM:      scf.if
  // SYM:        scf.for
  // SYM-NOT:      arith.select
  // SYM:          tt.store %{{.*}}, %[[ONE]] : tensor<64x!tt.ptr<f32>>
  // SYM:      } else {
  // SYM:        scf.for
  // SYM:          arith.select
  // LEGACY-NOT: scf.if
  scf.for %r = %c0 to %rnumel step %c64 : i32 {
    %rs = tt.splat %r : i32 -> tensor<64xi32>
    %idx = arith.addi %rs, %lane : tensor<64xi32>
    %mask = arith.cmpi slt, %idx, %ns : tensor<64xi32>
    %p = tt.addptr %ps, %idx : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %sel = arith.select %mask, %one, %cst : tensor<64xi1>, tensor<64xf32>
    tt.store %p, %sel : tensor<64x!tt.ptr<f32>>
    scf.yield
  }
  tt.return
}

// -----

// COM: Two independent invariant masks whose `tt.load`s share a loc name
// COM: ("offset"). Census ids and guard dedup key
// COM: on the SSA value, not the loc text, so sharing a name must not merge
// COM: the two loads' evidence. Neither load has a narrowed range, and N is a
// COM: kernel arg: the residual `N - offset - 63` has two unconstrained
// COM: invariant symbols and no substituted IV or quotient, so only the
// COM: residual-guard candidate reaches it. The two conditions it emits per
// COM: load (one per `tt.load`, pinned by captures rather than a count:
// COM: materialization also extends N) are what distinguishes the loads from
// COM: each other.

// CHECK-LABEL: tt.func @same_loc_two_guards
tt.func @same_loc_two_guards(%ptr1: !tt.ptr<f32>, %ptr2: !tt.ptr<f32>,
                             %offp1: !tt.ptr<i32>, %offp2: !tt.ptr<i32>,
                             %N: i32) {
  %c0 = arith.constant 0 : i32
  %c64 = arith.constant 64 : i32
  // SYM: %[[O1:.*]] = tt.load %{{.*}} : !tt.ptr<i32>
  // SYM: %[[O2:.*]] = tt.load %{{.*}} : !tt.ptr<i32>
  %o1 = tt.load %offp1 : !tt.ptr<i32> loc("offset")
  %o2 = tt.load %offp2 : !tt.ptr<i32> loc("offset")
  %lane = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %ns = tt.splat %N : i32 -> tensor<64xi32>
  %os1 = tt.splat %o1 : i32 -> tensor<64xi32>
  %os2 = tt.splat %o2 : i32 -> tensor<64xi32>
  %idx1 = arith.addi %lane, %os1 : tensor<64xi32>
  %mask1 = arith.cmpi slt, %idx1, %ns : tensor<64xi32>
  %idx2 = arith.addi %lane, %os2 : tensor<64xi32>
  %mask2 = arith.cmpi slt, %idx2, %ns : tensor<64xi32>
  %ps1 = tt.splat %ptr1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
  %ps2 = tt.splat %ptr2 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
  // SYM-DAG: arith.extsi %[[O1]]
  // SYM-DAG: arith.extsi %[[O2]]
  // SYM: scf.if
  // SYM:   scf.for
  // SYM-NOT:   tt.load %{{.*}}, %{{.*}} :
  // SYM:       tt.load %{{[^,]*}} : tensor<64x!tt.ptr<f32>>
  // SYM:       tt.load %{{[^,]*}} : tensor<64x!tt.ptr<f32>>
  // SYM: } else {
  // SYM:   scf.for
  // SYM:     tt.load %{{.*}}, %{{.*}} : tensor<64x!tt.ptr<f32>>
  // SYM:     tt.load %{{.*}}, %{{.*}} : tensor<64x!tt.ptr<f32>>
  scf.for %r = %c0 to %N step %c64 : i32 {
    %v1 = tt.load %ps1, %mask1 : tensor<64x!tt.ptr<f32>>
    %v2 = tt.load %ps2, %mask2 : tensor<64x!tt.ptr<f32>>
    tt.store %ps1, %v1, %mask1 : tensor<64x!tt.ptr<f32>>
    tt.store %ps2, %v2, %mask2 : tensor<64x!tt.ptr<f32>>
    scf.yield
  }
  tt.return
}

// -----

// COM: A loop-carried mask: an i1 iter_arg initialized `true` that turns false
// COM: after the second iteration (r < 64 fails at r = 64, so the iteration at
// COM: r = 128 runs with a false mask). The collector refuses a block-argument
// COM: mask, so the prover is never asked; unmasking on the `true` init would be
// COM: wrong. No versioning, mask kept.

// CHECK-LABEL: tt.func @loop_carried_mask
tt.func @loop_carried_mask(%ptr: !tt.ptr<f32>, %rnumel: i32) {
  %c0 = arith.constant 0 : i32
  %c64 = arith.constant 64 : i32
  %true = arith.constant true
  %lane = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %ps = tt.splat %ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
  // CHECK-NOT: scf.if
  // CHECK:     tt.load %{{.*}}, %{{.*}} : tensor<64x!tt.ptr<f32>>
  %res = scf.for %r = %c0 to %rnumel step %c64 iter_args(%m = %true) -> (i1) : i32 {
    %ms = tt.splat %m : i1 -> tensor<64xi1>
    %p = tt.addptr %ps, %lane : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %v = tt.load %p, %ms : tensor<64x!tt.ptr<f32>>
    tt.store %p, %v, %ms : tensor<64x!tt.ptr<f32>>
    %next = arith.cmpi slt, %r, %c64 : i32
    scf.yield %next : i1
  }
  tt.return
}

// -----

// COM: A Refuted mask on a load with no `other`. The largest index is 127+31,
// COM: well below 4096, so `idx >= 4096` is false in every element. A
// COM: Refuted-only loop gets no guard, so the driver drops the mask without
// COM: versioning: dropMask takes the getZeroAttr branch, replaces the uses with
// COM: a zero constant, and the driver then erases the now-unused load.
// COM: No LEGACY lines: whether the legacy RemovableMaskValidator classifies
// COM: this mask as false has not been established, and this section is
// COM: about the symbolic driver.

// CHECK-LABEL: tt.func @refuted_no_other
tt.func @refuted_no_other(%ptr: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : i32
  %c32 = arith.constant 32 : i32
  %c128 = arith.constant 128 : i32
  %c4096 = arith.constant dense<4096> : tensor<32xi32>
  %lane = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %ps = tt.splat %ptr : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
  // SYM-NOT: scf.if
  // SYM:     arith.constant dense<0.000000e+00>
  // SYM-NOT: tt.load
  scf.for %r = %c0 to %c128 step %c32 : i32 {
    %rs = tt.splat %r : i32 -> tensor<32xi32>
    %idx = arith.addi %rs, %lane : tensor<32xi32>
    %mask = arith.cmpi sge, %idx, %c4096 : tensor<32xi32>
    %p = tt.addptr %ps, %idx : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %v = tt.load %p, %mask : tensor<32x!tt.ptr<f32>>
    tt.store %p, %v : tensor<32x!tt.ptr<f32>>
    scf.yield
  }
  tt.return
}

// -----

// COM: A Satisfied mask on a volatile load. Exactly one load must remain in the
// COM: loop, unmasked and still volatile: only erasing the replaced load after
// COM: dropMask achieves that, since canonicalization keeps a volatile load.
// COM: No LEGACY lines: the legacy RemovableMaskValidator walk calls dropMask
// COM: without erasing and so keeps the original volatile load too, a
// COM: pre-existing behaviour this change does not alter.

// CHECK-LABEL: tt.func @volatile_unconditional
tt.func @volatile_unconditional(%ptr: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : i32
  %c32 = arith.constant 32 : i32
  %c128 = arith.constant 128 : i32
  %c4096 = arith.constant dense<4096> : tensor<32xi32>
  %cst = arith.constant dense<0.000000e+00> : tensor<32xf32>
  %lane = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %ps = tt.splat %ptr : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
  // SYM:     tt.load %{{[^ ,]+}} {isVolatile = true}
  // SYM-NOT: tt.load
  scf.for %r = %c0 to %c128 step %c32 : i32 {
    %rs = tt.splat %r : i32 -> tensor<32xi32>
    %idx = arith.addi %rs, %lane : tensor<32xi32>
    %mask = arith.cmpi slt, %idx, %c4096 : tensor<32xi32>
    %p = tt.addptr %ps, %idx : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %v = tt.load %p, %mask, %cst {isVolatile = true} : tensor<32x!tt.ptr<f32>>
    tt.store %p, %v : tensor<32x!tt.ptr<f32>>
    scf.yield
  }
  tt.return
}

// -----

// COM: zext(x) <= sext(x) is false at x = -1, yet both sides normalize to x. x
// COM: is an unconstrained scalar loaded in the loop, so the zext's
// COM: non-negativity obligation has no bound and the query must stay Unknown.
// COM: The explicit `other` makes a wrongly dropped mask visible.

// CHECK-LABEL: tt.func @unbounded_cancel_keeps_mask
tt.func @unbounded_cancel_keeps_mask(%ptr: !tt.ptr<f32>, %qtr: !tt.ptr<i32>, %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %other = arith.constant dense<7.000000e+00> : tensor<4xf32>
  %ps = tt.splat %ptr : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
  // CHECK-NOT: scf.if
  // CHECK:     tt.load %{{.*}}, %{{.*}}, %{{.*}} : tensor<4x!tt.ptr<f32>>
  scf.for %i = %c0 to %n step %c1 : i32 {
    %x = tt.load %qtr : !tt.ptr<i32>
    %u = arith.extui %x : i32 to i64
    %s = arith.extsi %x : i32 to i64
    %m = arith.cmpi sle, %u, %s : i64
    %ms = tt.splat %m : i1 -> tensor<4xi1>
    %v = tt.load %ps, %ms, %other : tensor<4x!tt.ptr<f32>>
    tt.store %ps, %v : tensor<4x!tt.ptr<f32>>
    scf.yield
  }
  tt.return
}

// -----

// COM: With n = 100 the i8 value y = n/2 - n/64 + 120 wraps (50 - 1 + 120 = 169),
// COM: so zext(y) <= sext(y) is false. A quotient is bounded by its dividend
// COM: divided by the divisor, not by the dividend, so the loop is versioned on
// COM: n/2 - n/64 <= 7 and the else-copy keeps the mask and `other`.

// CHECK-LABEL: tt.func @quotient_wrap_guard
tt.func @quotient_wrap_guard(%ptr: !tt.ptr<f32>, %n: i8, %cnt: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i8
  %c64 = arith.constant 64 : i8
  %c100 = arith.constant 100 : i8
  %c120 = arith.constant 120 : i8
  %eq = arith.cmpi eq, %n, %c100 : i8
  llvm.intr.assume %eq : i1
  %q2 = arith.divsi %n, %c2 : i8
  %q64 = arith.divsi %n, %c64 : i8
  %d = arith.subi %q2, %q64 : i8
  %y = arith.addi %d, %c120 : i8
  %u = arith.extui %y : i8 to i64
  %s = arith.extsi %y : i8 to i64
  %m = arith.cmpi sle, %u, %s : i64
  %ms = tt.splat %m : i1 -> tensor<4xi1>
  %other = arith.constant dense<7.000000e+00> : tensor<4xf32>
  %ps = tt.splat %ptr : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
  // SYM-DAG:  %[[C7:.*]] = arith.constant 7 : i64
  // SYM-DAG:  %[[C2:.*]] = arith.constant 2 : i64
  // SYM-DAG:  %[[C64:.*]] = arith.constant 64 : i64
  // SYM:      %[[Q2:.*]] = arith.divsi %{{.*}}, %[[C2]] : i64
  // SYM:      %[[Q64:.*]] = arith.divsi %{{.*}}, %[[C64]] : i64
  // SYM:      %[[DIFF:.*]] = arith.subi %[[Q2]], %[[Q64]] : i64
  // SYM:      %[[GUARD:.*]] = arith.cmpi sle, %[[DIFF]], %[[C7]] : i64
  // SYM:      scf.if %[[GUARD]] {
  // SYM:        scf.for
  // SYM-NOT:      tt.load %{{.*}}, %{{.*}}, %{{.*}} :
  // SYM:          tt.load %{{[^,]*}} : tensor<4x!tt.ptr<f32>>
  // SYM:      } else {
  // SYM:        scf.for
  // SYM:          tt.load %{{.*}}, %{{.*}}, %{{.*}} : tensor<4x!tt.ptr<f32>>
  // LEGACY-NOT: scf.if
  // LEGACY:     tt.load %{{.*}}, %{{.*}}, %{{.*}} : tensor<4x!tt.ptr<f32>>
  scf.for %i = %c0 to %cnt step %c1 : i32 {
    %v = tt.load %ps, %ms, %other : tensor<4x!tt.ptr<f32>>
    tt.store %ps, %v : tensor<4x!tt.ptr<f32>>
    scf.yield
  }
  tt.return
}

// -----

// COM: One loop with a Satisfied mask (lane < 64) and a ConditionallySatisfied
// COM: one. The driver drops the unconditional mask before versioning, so both
// COM: copies load %q unmasked; only the then-copy unmasks %p.

// CHECK-LABEL: tt.func @mixed_verdicts
tt.func @mixed_verdicts(%ptr: !tt.ptr<f32>, %qtr: !tt.ptr<f32>, %rnumel: i32) {
  %c0 = arith.constant 0 : i32
  %c64 = arith.constant 64 : i32
  %c64t = arith.constant dense<64> : tensor<64xi32>
  %lane = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %ns = tt.splat %rnumel : i32 -> tensor<64xi32>
  %ps = tt.splat %ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
  %qs = tt.splat %qtr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
  // SYM:      scf.if
  // SYM:        scf.for
  // SYM-NOT:      tt.load %{{.*}}, %{{.*}} :
  // SYM:          tt.load %{{[^,]*}} : tensor<64x!tt.ptr<f32>>
  // SYM-NOT:      tt.load
  // SYM:          tt.load %{{[^,]*}} : tensor<64x!tt.ptr<f32>>
  // SYM-NOT:      tt.load
  // SYM:      } else {
  // SYM:        scf.for
  // SYM:          tt.load %{{[^,]*}} : tensor<64x!tt.ptr<f32>>
  // SYM:          tt.load %{{.*}}, %{{.*}} : tensor<64x!tt.ptr<f32>>
  scf.for %r = %c0 to %rnumel step %c64 : i32 {
    %rs = tt.splat %r : i32 -> tensor<64xi32>
    %idx = arith.addi %rs, %lane : tensor<64xi32>
    %always = arith.cmpi slt, %lane, %c64t : tensor<64xi32>
    %mask = arith.cmpi slt, %idx, %ns : tensor<64xi32>
    %p = tt.addptr %ps, %idx : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %q = tt.addptr %qs, %lane : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %w = tt.load %q, %always : tensor<64x!tt.ptr<f32>>
    %v = tt.load %p, %mask : tensor<64x!tt.ptr<f32>>
    %s = arith.addf %v, %w : tensor<64xf32>
    tt.store %p, %s, %mask : tensor<64x!tt.ptr<f32>>
    scf.yield
  }
  tt.return
}

// -----

// COM: Two loads whose masks need the same condition: the loop's conditions
// COM: are deduplicated, so the guard is a single comparison.

// CHECK-LABEL: tt.func @shared_condition
tt.func @shared_condition(%ptr: !tt.ptr<f32>, %qtr: !tt.ptr<f32>, %rnumel: i32) {
  %c0 = arith.constant 0 : i32
  %c64 = arith.constant 64 : i32
  %lane = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %ns = tt.splat %rnumel : i32 -> tensor<64xi32>
  %ps = tt.splat %ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
  %qs = tt.splat %qtr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
  // SYM:      %[[REM:.*]] = arith.remsi
  // SYM-NOT:  arith.remsi
  // SYM-NOT:  arith.andi
  // SYM:      %[[GUARD:.*]] = arith.cmpi eq, %[[REM]]
  // SYM-NOT:  arith.cmpi
  // SYM-NOT:  arith.andi
  // SYM:      scf.if %[[GUARD]] {
  // SYM:        scf.for
  // SYM-NOT:      tt.load %{{.*}}, %{{.*}} :
  // SYM:          tt.load %{{[^,]*}} : tensor<64x!tt.ptr<f32>>
  // SYM:          tt.load %{{[^,]*}} : tensor<64x!tt.ptr<f32>>
  // SYM:      } else {
  // SYM:        scf.for
  // SYM:          tt.load %{{.*}}, %{{.*}} : tensor<64x!tt.ptr<f32>>
  // SYM:          tt.load %{{.*}}, %{{.*}} : tensor<64x!tt.ptr<f32>>
  scf.for %r = %c0 to %rnumel step %c64 : i32 {
    %rs = tt.splat %r : i32 -> tensor<64xi32>
    %idx = arith.addi %rs, %lane : tensor<64xi32>
    %mask1 = arith.cmpi slt, %idx, %ns : tensor<64xi32>
    %mask2 = arith.cmpi slt, %idx, %ns : tensor<64xi32>
    %p = tt.addptr %ps, %idx : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %q = tt.addptr %qs, %idx : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %v = tt.load %p, %mask1 : tensor<64x!tt.ptr<f32>>
    %w = tt.load %q, %mask2 : tensor<64x!tt.ptr<f32>>
    %s = arith.addf %v, %w : tensor<64xf32>
    tt.store %p, %s, %mask1 : tensor<64x!tt.ptr<f32>>
    scf.yield
  }
  tt.return
}

// -----

// COM: Two outermost loops, the second bounded by the first loop's result. The
// COM: second guard is built before the first loop is versioned, and must end
// COM: up reading the first scf.if's result.

// CHECK-LABEL: tt.func @dependent_loops
tt.func @dependent_loops(%ptr: !tt.ptr<f32>, %rnumel: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c64 = arith.constant 64 : i32
  %lane = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %ns = tt.splat %rnumel : i32 -> tensor<64xi32>
  %ps = tt.splat %ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
  // SYM:      %[[R:.*]] = scf.if %{{.*}} -> (i32) {
  // SYM:      } else {
  // SYM:      %[[EXT:.*]] = arith.extsi %[[R]] : i32 to i64
  // SYM:      %[[REM:.*]] = arith.remsi %[[EXT]], %{{.*}} : i64
  // SYM:      %[[GUARD:.*]] = arith.cmpi eq, %[[REM]], %{{.*}} : i64
  // SYM:      scf.if %[[GUARD]] {
  // SYM:        scf.for %{{.*}} = %{{.*}} to %[[R]] step
  // SYM-NOT:      tt.load %{{.*}}, %{{.*}} :
  // SYM:          tt.load %{{[^,]*}} : tensor<64x!tt.ptr<f32>>
  // SYM:      } else {
  %n2 = scf.for %r = %c0 to %rnumel step %c64 iter_args(%a = %rnumel) -> (i32) : i32 {
    %rs = tt.splat %r : i32 -> tensor<64xi32>
    %idx = arith.addi %rs, %lane : tensor<64xi32>
    %mask = arith.cmpi slt, %idx, %ns : tensor<64xi32>
    %p = tt.addptr %ps, %idx : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %v = tt.load %p, %mask : tensor<64x!tt.ptr<f32>>
    tt.store %p, %v, %mask : tensor<64x!tt.ptr<f32>>
    %a2 = arith.addi %a, %c1 : i32
    scf.yield %a2 : i32
  }
  %ns2 = tt.splat %n2 : i32 -> tensor<64xi32>
  scf.for %r = %c0 to %n2 step %c64 : i32 {
    %rs = tt.splat %r : i32 -> tensor<64xi32>
    %idx = arith.addi %rs, %lane : tensor<64xi32>
    %mask = arith.cmpi slt, %idx, %ns2 : tensor<64xi32>
    %p = tt.addptr %ps, %idx : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %v = tt.load %p, %mask : tensor<64x!tt.ptr<f32>>
    tt.store %p, %v, %mask : tensor<64x!tt.ptr<f32>>
    scf.yield
  }
  tt.return
}

// -----

// COM: Four loads shaped like @same_loc_two_guards: each proof needs one Fact
// COM: (N - offset >= 64) and one Guard (offset <= INT_MAX - 63), so the loop's
// COM: four Facts are exactly kMaxFactConditions and the loop is versioned.
// COM: @over_budget below adds a fifth load.

// CHECK-LABEL: tt.func @at_fact_budget
tt.func @at_fact_budget(%p0: !tt.ptr<f32>, %o0p: !tt.ptr<i32>, %p1: !tt.ptr<f32>, %o1p: !tt.ptr<i32>, %p2: !tt.ptr<f32>, %o2p: !tt.ptr<i32>, %p3: !tt.ptr<f32>, %o3p: !tt.ptr<i32>, %N: i32) {
  %c0 = arith.constant 0 : i32
  %c64 = arith.constant 64 : i32
  %lane = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %ns = tt.splat %N : i32 -> tensor<64xi32>
  %o0 = tt.load %o0p : !tt.ptr<i32>
  %os0 = tt.splat %o0 : i32 -> tensor<64xi32>
  %idx0 = arith.addi %lane, %os0 : tensor<64xi32>
  %mask0 = arith.cmpi slt, %idx0, %ns : tensor<64xi32>
  %ps0 = tt.splat %p0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
  %o1 = tt.load %o1p : !tt.ptr<i32>
  %os1 = tt.splat %o1 : i32 -> tensor<64xi32>
  %idx1 = arith.addi %lane, %os1 : tensor<64xi32>
  %mask1 = arith.cmpi slt, %idx1, %ns : tensor<64xi32>
  %ps1 = tt.splat %p1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
  %o2 = tt.load %o2p : !tt.ptr<i32>
  %os2 = tt.splat %o2 : i32 -> tensor<64xi32>
  %idx2 = arith.addi %lane, %os2 : tensor<64xi32>
  %mask2 = arith.cmpi slt, %idx2, %ns : tensor<64xi32>
  %ps2 = tt.splat %p2 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
  %o3 = tt.load %o3p : !tt.ptr<i32>
  %os3 = tt.splat %o3 : i32 -> tensor<64xi32>
  %idx3 = arith.addi %lane, %os3 : tensor<64xi32>
  %mask3 = arith.cmpi slt, %idx3, %ns : tensor<64xi32>
  %ps3 = tt.splat %p3 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
  // SYM:      scf.if
  // SYM:        scf.for
  // SYM-NOT:      tt.load %{{.*}}, %{{.*}} :
  // SYM:      } else {
  // SYM:        scf.for
  // SYM-COUNT-4:  tt.load %{{.*}}, %{{.*}} : tensor<64x!tt.ptr<f32>>
  scf.for %r = %c0 to %N step %c64 : i32 {
    %v0 = tt.load %ps0, %mask0 : tensor<64x!tt.ptr<f32>>
    tt.store %ps0, %v0, %mask0 : tensor<64x!tt.ptr<f32>>
    %v1 = tt.load %ps1, %mask1 : tensor<64x!tt.ptr<f32>>
    tt.store %ps1, %v1, %mask1 : tensor<64x!tt.ptr<f32>>
    %v2 = tt.load %ps2, %mask2 : tensor<64x!tt.ptr<f32>>
    tt.store %ps2, %v2, %mask2 : tensor<64x!tt.ptr<f32>>
    %v3 = tt.load %ps3, %mask3 : tensor<64x!tt.ptr<f32>>
    tt.store %ps3, %v3, %mask3 : tensor<64x!tt.ptr<f32>>
    scf.yield
  }
  tt.return
}

// -----

// COM: @at_fact_budget with a fifth load. Every proof is within budget, but
// COM: the loop's five Facts exceed kMaxFactConditions (4), so the loop is not
// COM: versioned and every mask is kept.

// CHECK-LABEL: tt.func @over_budget
tt.func @over_budget(%p0: !tt.ptr<f32>, %o0p: !tt.ptr<i32>, %p1: !tt.ptr<f32>, %o1p: !tt.ptr<i32>, %p2: !tt.ptr<f32>, %o2p: !tt.ptr<i32>, %p3: !tt.ptr<f32>, %o3p: !tt.ptr<i32>, %p4: !tt.ptr<f32>, %o4p: !tt.ptr<i32>, %N: i32) {
  %c0 = arith.constant 0 : i32
  %c64 = arith.constant 64 : i32
  %lane = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %ns = tt.splat %N : i32 -> tensor<64xi32>
  %o0 = tt.load %o0p : !tt.ptr<i32>
  %os0 = tt.splat %o0 : i32 -> tensor<64xi32>
  %idx0 = arith.addi %lane, %os0 : tensor<64xi32>
  %mask0 = arith.cmpi slt, %idx0, %ns : tensor<64xi32>
  %ps0 = tt.splat %p0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
  %o1 = tt.load %o1p : !tt.ptr<i32>
  %os1 = tt.splat %o1 : i32 -> tensor<64xi32>
  %idx1 = arith.addi %lane, %os1 : tensor<64xi32>
  %mask1 = arith.cmpi slt, %idx1, %ns : tensor<64xi32>
  %ps1 = tt.splat %p1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
  %o2 = tt.load %o2p : !tt.ptr<i32>
  %os2 = tt.splat %o2 : i32 -> tensor<64xi32>
  %idx2 = arith.addi %lane, %os2 : tensor<64xi32>
  %mask2 = arith.cmpi slt, %idx2, %ns : tensor<64xi32>
  %ps2 = tt.splat %p2 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
  %o3 = tt.load %o3p : !tt.ptr<i32>
  %os3 = tt.splat %o3 : i32 -> tensor<64xi32>
  %idx3 = arith.addi %lane, %os3 : tensor<64xi32>
  %mask3 = arith.cmpi slt, %idx3, %ns : tensor<64xi32>
  %ps3 = tt.splat %p3 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
  %o4 = tt.load %o4p : !tt.ptr<i32>
  %os4 = tt.splat %o4 : i32 -> tensor<64xi32>
  %idx4 = arith.addi %lane, %os4 : tensor<64xi32>
  %mask4 = arith.cmpi slt, %idx4, %ns : tensor<64xi32>
  %ps4 = tt.splat %p4 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
  // SYM-NOT:      scf.if
  // SYM:          scf.for
  // SYM-COUNT-5:    tt.load %{{.*}}, %{{.*}} : tensor<64x!tt.ptr<f32>>
  // SYM-NOT:      scf.if
  scf.for %r = %c0 to %N step %c64 : i32 {
    %v0 = tt.load %ps0, %mask0 : tensor<64x!tt.ptr<f32>>
    tt.store %ps0, %v0, %mask0 : tensor<64x!tt.ptr<f32>>
    %v1 = tt.load %ps1, %mask1 : tensor<64x!tt.ptr<f32>>
    tt.store %ps1, %v1, %mask1 : tensor<64x!tt.ptr<f32>>
    %v2 = tt.load %ps2, %mask2 : tensor<64x!tt.ptr<f32>>
    tt.store %ps2, %v2, %mask2 : tensor<64x!tt.ptr<f32>>
    %v3 = tt.load %ps3, %mask3 : tensor<64x!tt.ptr<f32>>
    tt.store %ps3, %v3, %mask3 : tensor<64x!tt.ptr<f32>>
    %v4 = tt.load %ps4, %mask4 : tensor<64x!tt.ptr<f32>>
    tt.store %ps4, %v4, %mask4 : tensor<64x!tt.ptr<f32>>
    scf.yield
  }
  tt.return
}

// -----

// COM: Eight loads shaped like @unbounded_cancel_keeps_mask, but with x loaded
// COM: before the loop: each proof needs one Guard (x >= 0) and no Fact, so the
// COM: loop's eight Guards are exactly kMaxGuards and the loop is versioned.
// COM: @over_guard_budget below adds a ninth load.

// CHECK-LABEL: tt.func @at_guard_budget
tt.func @at_guard_budget(%p0: !tt.ptr<f32>, %x0p: !tt.ptr<i32>, %p1: !tt.ptr<f32>, %x1p: !tt.ptr<i32>, %p2: !tt.ptr<f32>, %x2p: !tt.ptr<i32>, %p3: !tt.ptr<f32>, %x3p: !tt.ptr<i32>, %p4: !tt.ptr<f32>, %x4p: !tt.ptr<i32>, %p5: !tt.ptr<f32>, %x5p: !tt.ptr<i32>, %p6: !tt.ptr<f32>, %x6p: !tt.ptr<i32>, %p7: !tt.ptr<f32>, %x7p: !tt.ptr<i32>, %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %other = arith.constant dense<7.000000e+00> : tensor<4xf32>
  %x0 = tt.load %x0p : !tt.ptr<i32>
  %u0 = arith.extui %x0 : i32 to i64
  %s0 = arith.extsi %x0 : i32 to i64
  %m0 = arith.cmpi sle, %u0, %s0 : i64
  %ms0 = tt.splat %m0 : i1 -> tensor<4xi1>
  %ps0 = tt.splat %p0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
  %x1 = tt.load %x1p : !tt.ptr<i32>
  %u1 = arith.extui %x1 : i32 to i64
  %s1 = arith.extsi %x1 : i32 to i64
  %m1 = arith.cmpi sle, %u1, %s1 : i64
  %ms1 = tt.splat %m1 : i1 -> tensor<4xi1>
  %ps1 = tt.splat %p1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
  %x2 = tt.load %x2p : !tt.ptr<i32>
  %u2 = arith.extui %x2 : i32 to i64
  %s2 = arith.extsi %x2 : i32 to i64
  %m2 = arith.cmpi sle, %u2, %s2 : i64
  %ms2 = tt.splat %m2 : i1 -> tensor<4xi1>
  %ps2 = tt.splat %p2 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
  %x3 = tt.load %x3p : !tt.ptr<i32>
  %u3 = arith.extui %x3 : i32 to i64
  %s3 = arith.extsi %x3 : i32 to i64
  %m3 = arith.cmpi sle, %u3, %s3 : i64
  %ms3 = tt.splat %m3 : i1 -> tensor<4xi1>
  %ps3 = tt.splat %p3 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
  %x4 = tt.load %x4p : !tt.ptr<i32>
  %u4 = arith.extui %x4 : i32 to i64
  %s4 = arith.extsi %x4 : i32 to i64
  %m4 = arith.cmpi sle, %u4, %s4 : i64
  %ms4 = tt.splat %m4 : i1 -> tensor<4xi1>
  %ps4 = tt.splat %p4 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
  %x5 = tt.load %x5p : !tt.ptr<i32>
  %u5 = arith.extui %x5 : i32 to i64
  %s5 = arith.extsi %x5 : i32 to i64
  %m5 = arith.cmpi sle, %u5, %s5 : i64
  %ms5 = tt.splat %m5 : i1 -> tensor<4xi1>
  %ps5 = tt.splat %p5 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
  %x6 = tt.load %x6p : !tt.ptr<i32>
  %u6 = arith.extui %x6 : i32 to i64
  %s6 = arith.extsi %x6 : i32 to i64
  %m6 = arith.cmpi sle, %u6, %s6 : i64
  %ms6 = tt.splat %m6 : i1 -> tensor<4xi1>
  %ps6 = tt.splat %p6 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
  %x7 = tt.load %x7p : !tt.ptr<i32>
  %u7 = arith.extui %x7 : i32 to i64
  %s7 = arith.extsi %x7 : i32 to i64
  %m7 = arith.cmpi sle, %u7, %s7 : i64
  %ms7 = tt.splat %m7 : i1 -> tensor<4xi1>
  %ps7 = tt.splat %p7 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
  // SYM:      scf.if
  // SYM:        scf.for
  // SYM-NOT:      tt.load %{{.*}}, %{{.*}}, %{{.*}} :
  // SYM:      } else {
  // SYM:        scf.for
  // SYM-COUNT-8:  tt.load %{{.*}}, %{{.*}}, %{{.*}} : tensor<4x!tt.ptr<f32>>
  scf.for %i = %c0 to %n step %c1 : i32 {
    %v0 = tt.load %ps0, %ms0, %other : tensor<4x!tt.ptr<f32>>
    tt.store %ps0, %v0 : tensor<4x!tt.ptr<f32>>
    %v1 = tt.load %ps1, %ms1, %other : tensor<4x!tt.ptr<f32>>
    tt.store %ps1, %v1 : tensor<4x!tt.ptr<f32>>
    %v2 = tt.load %ps2, %ms2, %other : tensor<4x!tt.ptr<f32>>
    tt.store %ps2, %v2 : tensor<4x!tt.ptr<f32>>
    %v3 = tt.load %ps3, %ms3, %other : tensor<4x!tt.ptr<f32>>
    tt.store %ps3, %v3 : tensor<4x!tt.ptr<f32>>
    %v4 = tt.load %ps4, %ms4, %other : tensor<4x!tt.ptr<f32>>
    tt.store %ps4, %v4 : tensor<4x!tt.ptr<f32>>
    %v5 = tt.load %ps5, %ms5, %other : tensor<4x!tt.ptr<f32>>
    tt.store %ps5, %v5 : tensor<4x!tt.ptr<f32>>
    %v6 = tt.load %ps6, %ms6, %other : tensor<4x!tt.ptr<f32>>
    tt.store %ps6, %v6 : tensor<4x!tt.ptr<f32>>
    %v7 = tt.load %ps7, %ms7, %other : tensor<4x!tt.ptr<f32>>
    tt.store %ps7, %v7 : tensor<4x!tt.ptr<f32>>
    scf.yield
  }
  tt.return
}

// -----

// COM: @at_guard_budget with a ninth load: the loop's nine Guards exceed
// COM: kMaxGuards (8), so the loop is not versioned and every mask is kept.

// CHECK-LABEL: tt.func @over_guard_budget
tt.func @over_guard_budget(%p0: !tt.ptr<f32>, %x0p: !tt.ptr<i32>, %p1: !tt.ptr<f32>, %x1p: !tt.ptr<i32>, %p2: !tt.ptr<f32>, %x2p: !tt.ptr<i32>, %p3: !tt.ptr<f32>, %x3p: !tt.ptr<i32>, %p4: !tt.ptr<f32>, %x4p: !tt.ptr<i32>, %p5: !tt.ptr<f32>, %x5p: !tt.ptr<i32>, %p6: !tt.ptr<f32>, %x6p: !tt.ptr<i32>, %p7: !tt.ptr<f32>, %x7p: !tt.ptr<i32>, %p8: !tt.ptr<f32>, %x8p: !tt.ptr<i32>, %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %other = arith.constant dense<7.000000e+00> : tensor<4xf32>
  %x0 = tt.load %x0p : !tt.ptr<i32>
  %u0 = arith.extui %x0 : i32 to i64
  %s0 = arith.extsi %x0 : i32 to i64
  %m0 = arith.cmpi sle, %u0, %s0 : i64
  %ms0 = tt.splat %m0 : i1 -> tensor<4xi1>
  %ps0 = tt.splat %p0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
  %x1 = tt.load %x1p : !tt.ptr<i32>
  %u1 = arith.extui %x1 : i32 to i64
  %s1 = arith.extsi %x1 : i32 to i64
  %m1 = arith.cmpi sle, %u1, %s1 : i64
  %ms1 = tt.splat %m1 : i1 -> tensor<4xi1>
  %ps1 = tt.splat %p1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
  %x2 = tt.load %x2p : !tt.ptr<i32>
  %u2 = arith.extui %x2 : i32 to i64
  %s2 = arith.extsi %x2 : i32 to i64
  %m2 = arith.cmpi sle, %u2, %s2 : i64
  %ms2 = tt.splat %m2 : i1 -> tensor<4xi1>
  %ps2 = tt.splat %p2 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
  %x3 = tt.load %x3p : !tt.ptr<i32>
  %u3 = arith.extui %x3 : i32 to i64
  %s3 = arith.extsi %x3 : i32 to i64
  %m3 = arith.cmpi sle, %u3, %s3 : i64
  %ms3 = tt.splat %m3 : i1 -> tensor<4xi1>
  %ps3 = tt.splat %p3 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
  %x4 = tt.load %x4p : !tt.ptr<i32>
  %u4 = arith.extui %x4 : i32 to i64
  %s4 = arith.extsi %x4 : i32 to i64
  %m4 = arith.cmpi sle, %u4, %s4 : i64
  %ms4 = tt.splat %m4 : i1 -> tensor<4xi1>
  %ps4 = tt.splat %p4 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
  %x5 = tt.load %x5p : !tt.ptr<i32>
  %u5 = arith.extui %x5 : i32 to i64
  %s5 = arith.extsi %x5 : i32 to i64
  %m5 = arith.cmpi sle, %u5, %s5 : i64
  %ms5 = tt.splat %m5 : i1 -> tensor<4xi1>
  %ps5 = tt.splat %p5 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
  %x6 = tt.load %x6p : !tt.ptr<i32>
  %u6 = arith.extui %x6 : i32 to i64
  %s6 = arith.extsi %x6 : i32 to i64
  %m6 = arith.cmpi sle, %u6, %s6 : i64
  %ms6 = tt.splat %m6 : i1 -> tensor<4xi1>
  %ps6 = tt.splat %p6 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
  %x7 = tt.load %x7p : !tt.ptr<i32>
  %u7 = arith.extui %x7 : i32 to i64
  %s7 = arith.extsi %x7 : i32 to i64
  %m7 = arith.cmpi sle, %u7, %s7 : i64
  %ms7 = tt.splat %m7 : i1 -> tensor<4xi1>
  %ps7 = tt.splat %p7 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
  %x8 = tt.load %x8p : !tt.ptr<i32>
  %u8 = arith.extui %x8 : i32 to i64
  %s8 = arith.extsi %x8 : i32 to i64
  %m8 = arith.cmpi sle, %u8, %s8 : i64
  %ms8 = tt.splat %m8 : i1 -> tensor<4xi1>
  %ps8 = tt.splat %p8 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
  // SYM-NOT:      scf.if
  // SYM:          scf.for
  // SYM-COUNT-9:    tt.load %{{.*}}, %{{.*}}, %{{.*}} : tensor<4x!tt.ptr<f32>>
  // SYM-NOT:      scf.if
  scf.for %i = %c0 to %n step %c1 : i32 {
    %v0 = tt.load %ps0, %ms0, %other : tensor<4x!tt.ptr<f32>>
    tt.store %ps0, %v0 : tensor<4x!tt.ptr<f32>>
    %v1 = tt.load %ps1, %ms1, %other : tensor<4x!tt.ptr<f32>>
    tt.store %ps1, %v1 : tensor<4x!tt.ptr<f32>>
    %v2 = tt.load %ps2, %ms2, %other : tensor<4x!tt.ptr<f32>>
    tt.store %ps2, %v2 : tensor<4x!tt.ptr<f32>>
    %v3 = tt.load %ps3, %ms3, %other : tensor<4x!tt.ptr<f32>>
    tt.store %ps3, %v3 : tensor<4x!tt.ptr<f32>>
    %v4 = tt.load %ps4, %ms4, %other : tensor<4x!tt.ptr<f32>>
    tt.store %ps4, %v4 : tensor<4x!tt.ptr<f32>>
    %v5 = tt.load %ps5, %ms5, %other : tensor<4x!tt.ptr<f32>>
    tt.store %ps5, %v5 : tensor<4x!tt.ptr<f32>>
    %v6 = tt.load %ps6, %ms6, %other : tensor<4x!tt.ptr<f32>>
    tt.store %ps6, %v6 : tensor<4x!tt.ptr<f32>>
    %v7 = tt.load %ps7, %ms7, %other : tensor<4x!tt.ptr<f32>>
    tt.store %ps7, %v7 : tensor<4x!tt.ptr<f32>>
    %v8 = tt.load %ps8, %ms8, %other : tensor<4x!tt.ptr<f32>>
    tt.store %ps8, %v8 : tensor<4x!tt.ptr<f32>>
    scf.yield
  }
  tt.return
}
