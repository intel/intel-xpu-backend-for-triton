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
  // SYM:      %[[REM:.*]] = arith.remsi %{{.*}}, %{{.*}} : i64
  // SYM:      %[[GUARD:.*]] = arith.cmpi eq, %[[REM]], %{{.*}} : i64
  // SYM:      scf.if %[[GUARD]] {
  // SYM:        scf.for
  // SYM-NOT:      tt.load %{{.*}}, %{{.*}} :
  // SYM:          tt.load %{{.*}} : tensor<64x!tt.ptr<f32>>
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
  // SYM:      scf.if
  // SYM:        scf.for
  // SYM:          tt.load %{{.*}} : tensor<64x!tt.ptr<f32>>
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
// COM: its true value, so the then-copy contains no select at all.

// CHECK-LABEL: tt.func @select_mask
tt.func @select_mask(%ptr: !tt.ptr<f32>, %rnumel: i32) {
  %c0 = arith.constant 0 : i32
  %c64 = arith.constant 64 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<64xf32>
  %one = arith.constant dense<1.000000e+00> : tensor<64xf32>
  %lane = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %ns = tt.splat %rnumel : i32 -> tensor<64xi32>
  %ps = tt.splat %ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
  // SYM:      scf.if
  // SYM:        scf.for
  // SYM-NOT:      arith.select
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
  // SYM:       tt.load %{{.*}} : tensor<64x!tt.ptr<f32>>
  // SYM:       tt.load %{{.*}} : tensor<64x!tt.ptr<f32>>
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

// COM: A loop-carried mask: an i1 iter_arg initialized `true` and yielding a
// COM: computed value. proveTrue stops at the block argument rather than
// COM: substituting the init value, which would read as unconditionally true and
// COM: unmask a load the mask is guarding. No versioning, mask kept.

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
    %next = arith.cmpi slt, %r, %rnumel : i32
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
