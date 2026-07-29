// Selectivity of `--triton-rewrite-tensor-descriptor-to-pointer=loop-recreated-only=true`.
//
// In loop-recreated-only mode the pass demotes ONLY descriptors that are
// recreated inside a loop and cannot be hoisted (operands not loop-invariant).
// Everything else — out-of-loop descriptors, and loop descriptors whose operands
// are all loop-invariant (LICM would hoist them) — stays a `!tt.tensordesc` and
// keeps the TMA path.
//
// Default (static) mode rewrites ALL descriptors; that is covered by
// rewrite-tensor-descriptor-to-pointer.mlir. Here we only test the selective mode.
//
// NOTE: run on a build that INCLUDES the loop-recreated-only patch. On an
// unpatched Triton the option is unknown and triton-opt errors out.

// RUN: triton-opt %s -split-input-file \
// RUN:   --triton-rewrite-tensor-descriptor-to-pointer=loop-recreated-only=true \
// RUN:   | FileCheck %s

// -----
// Case 1: descriptor recreated in a loop with a loop-VARYING base (base depends
// on the induction variable) -> DEMOTED to a masked pointer load; no descriptor
// op survives inside the loop.
//
// The realistic paged-KV pattern derives the base from an in-loop
// `tt.load(block_table[j])`. We instead make the base depend directly on the
// induction variable so the loop body contains exactly ONE `tt.load` (the
// demoted one). That keeps the CHECK-NOT bracket [scf.for .. tt.load] tight: if
// demotion did NOT happen there would be no `tt.load` at all and the final
// CHECK would fail — i.e. the assertion is non-vacuous.

// CHECK-LABEL: @loop_varying_demoted
module {
  tt.func public @loop_varying_demoted(
      %kv: !tt.ptr<bf16>, %stride1: i64)
      -> tensor<16x128xbf16> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : i32
    %c128 = arith.constant 128 : i32
    %c1_i64 = arith.constant 1 : i64
    %c32 = arith.constant 32 : i32
    %cst = arith.constant dense<0.0> : tensor<16x128xbf16>
    // CHECK: scf.for
    %r = scf.for %j = %c0 to %c32 step %c1 iter_args(%acc = %cst) -> (tensor<16x128xbf16>) : i32 {
      // base depends on the induction variable -> loop-varying -> demoted, so
      // these two ops vanish and a masked tt.load takes their place.
      // CHECK-NOT: tt.make_tensor_descriptor
      // CHECK-NOT: tt.descriptor_load
      // CHECK: tt.load
      %base = tt.addptr %kv, %j : !tt.ptr<bf16>, i32
      %desc = tt.make_tensor_descriptor %base, [%c16, %c128], [%stride1, %c1_i64] {order = array<i32: 1, 0>} : <bf16>, <16x128xbf16>
      %tile = tt.descriptor_load %desc[%c0, %c0] : !tt.tensordesc<16x128xbf16> -> tensor<16x128xbf16>
      scf.yield %tile : tensor<16x128xbf16>
    }
    tt.return %r : tensor<16x128xbf16>
  }
}

// -----
// Case 2: descriptor built INSIDE a loop but from loop-INVARIANT operands only
// (function args) -> LICM could hoist it, so it is KEPT as a descriptor.

// CHECK-LABEL: @loop_invariant_kept
module {
  tt.func public @loop_invariant_kept(
      %base: !tt.ptr<bf16>, %s0: i32, %s1: i32, %st0: i64, %st1: i64)
      -> tensor<16x128xbf16> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c32 = arith.constant 32 : i32
    %cst = arith.constant dense<0.0> : tensor<16x128xbf16>
    %r = scf.for %j = %c0 to %c32 step %c1 iter_args(%acc = %cst) -> (tensor<16x128xbf16>) : i32 {
      // all operands are loop-invariant -> kept on the TMA path
      // CHECK: tt.make_tensor_descriptor
      %desc = tt.make_tensor_descriptor %base, [%s0, %s1], [%st0, %st1] {order = array<i32: 1, 0>} : <bf16>, <16x128xbf16>
      // CHECK: tt.descriptor_load
      %tile = tt.descriptor_load %desc[%c0, %c0] : !tt.tensordesc<16x128xbf16> -> tensor<16x128xbf16>
      scf.yield %tile : tensor<16x128xbf16>
    }
    tt.return %r : tensor<16x128xbf16>
  }
}

// -----
// Case 2b: descriptor built inside a loop from an in-loop TEMPORARY that is
// itself computed only from loop-invariant operands (`%t = arith.muli %st0,
// %st1`). A shallow "all operands defined outside the loop?" check would see
// `%t` defined in-loop and wrongly demote the descriptor — but LICM would hoist
// `%t` first and then the descriptor, so it must be KEPT. This is the
// transitive-invariance case the recursive check exists for.

// CHECK-LABEL: @loop_invariant_via_temporary_kept
module {
  tt.func public @loop_invariant_via_temporary_kept(
      %base: !tt.ptr<bf16>, %s0: i32, %s1: i32, %st0: i64, %st1: i64)
      -> tensor<16x128xbf16> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c32 = arith.constant 32 : i32
    %cst = arith.constant dense<0.0> : tensor<16x128xbf16>
    %r = scf.for %j = %c0 to %c32 step %c1 iter_args(%acc = %cst) -> (tensor<16x128xbf16>) : i32 {
      // %t depends only on loop-invariant args -> LICM hoists it, so the
      // descriptor is hoistable and stays on the TMA path.
      // CHECK: tt.make_tensor_descriptor
      %t = arith.muli %st0, %st1 : i64
      %desc = tt.make_tensor_descriptor %base, [%s0, %s1], [%t, %st1] {order = array<i32: 1, 0>} : <bf16>, <16x128xbf16>
      // CHECK: tt.descriptor_load
      %tile = tt.descriptor_load %desc[%c0, %c0] : !tt.tensordesc<16x128xbf16> -> tensor<16x128xbf16>
      scf.yield %tile : tensor<16x128xbf16>
    }
    tt.return %r : tensor<16x128xbf16>
  }
}

// -----
// Case 3: descriptor built OUTSIDE any loop -> KEPT (not our target).

// CHECK-LABEL: @out_of_loop_kept
module {
  tt.func public @out_of_loop_kept(
      %base: !tt.ptr<bf16>, %s0: i32, %s1: i32, %st0: i64, %st1: i64)
      -> tensor<16x128xbf16> {
    %c0 = arith.constant 0 : i32
    // CHECK: tt.make_tensor_descriptor
    %desc = tt.make_tensor_descriptor %base, [%s0, %s1], [%st0, %st1] {order = array<i32: 1, 0>} : <bf16>, <16x128xbf16>
    // CHECK: tt.descriptor_load
    %tile = tt.descriptor_load %desc[%c0, %c0] : !tt.tensordesc<16x128xbf16> -> tensor<16x128xbf16>
    tt.return %tile : tensor<16x128xbf16>
  }
}

// -----
// Case 4: MIXED — an out-of-loop descriptor (KEPT) and a loop-varying descriptor
// (DEMOTED) in one function. Per-descriptor selectivity: the out-of-loop
// descriptor + its load stay TMA; the in-loop pair becomes a masked tt.load and
// no descriptor op survives after the loop header.

// CHECK-LABEL: @mixed_only_varying_demoted
module {
  tt.func public @mixed_only_varying_demoted(
      %kv: !tt.ptr<bf16>, %base2: !tt.ptr<bf16>,
      %s0: i32, %s1: i32, %stride0: i64, %stride1: i64)
      -> (tensor<16x128xbf16>, tensor<16x128xbf16>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : i32
    %c128 = arith.constant 128 : i32
    %c1_i64 = arith.constant 1 : i64
    %c32 = arith.constant 32 : i32
    %cst = arith.constant dense<0.0> : tensor<16x128xbf16>
    // out-of-loop descriptor: kept, along with its descriptor_load
    // CHECK: tt.make_tensor_descriptor
    %descA = tt.make_tensor_descriptor %base2, [%s0, %s1], [%stride0, %stride1] {order = array<i32: 1, 0>} : <bf16>, <16x128xbf16>
    // CHECK: tt.descriptor_load
    %tileA = tt.descriptor_load %descA[%c0, %c0] : !tt.tensordesc<16x128xbf16> -> tensor<16x128xbf16>
    // CHECK: scf.for
    // Past this point no descriptor op survives: the in-loop one was demoted.
    // The loop body has exactly one tt.load (the demoted one), so this bracket
    // is tight — see the note on Case 1.
    // CHECK-NOT: tt.make_tensor_descriptor
    // CHECK-NOT: tt.descriptor_load
    // CHECK: tt.load
    %r = scf.for %j = %c0 to %c32 step %c1 iter_args(%acc = %cst) -> (tensor<16x128xbf16>) : i32 {
      %base = tt.addptr %kv, %j : !tt.ptr<bf16>, i32
      %descB = tt.make_tensor_descriptor %base, [%c16, %c128], [%stride1, %c1_i64] {order = array<i32: 1, 0>} : <bf16>, <16x128xbf16>
      %tileB = tt.descriptor_load %descB[%c0, %c0] : !tt.tensordesc<16x128xbf16> -> tensor<16x128xbf16>
      scf.yield %tileB : tensor<16x128xbf16>
    }
    tt.return %tileA, %r : tensor<16x128xbf16>, tensor<16x128xbf16>
  }
}

// -----
// Case 5: MIXED PROVENANCE through one value. An scf.if inside the loop merges a
// hoistable (loop-invariant) make and a loop-recreated make into a single
// !tt.tensordesc that feeds descriptor_load. Only the else-make is loop-recreated,
// so a per-make rule alone would demote just that one — leaving the scf.if and the
// descriptor_load holding a value whose type is half-rewritten, which the
// converter cannot express (`buildMaterializations = false`).
//
// Both makes appear on the same ops (the yields, the scf.if, the load), so they
// are entangled and decide together. Neither reaches a consumer this mode keeps
// legal, so the OPTIMISTIC direction wins and both are demoted. Result: neither
// make nor the load survives — one masked tt.load remains.

// CHECK-LABEL: @mixed_provenance_merged_demoted
module {
  tt.func public @mixed_provenance_merged_demoted(
      %kv: !tt.ptr<bf16>, %base_inv: !tt.ptr<bf16>,
      %s0: i32, %s1: i32, %st0: i64, %st1: i64, %cond: i1)
      -> tensor<16x128xbf16> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : i32
    %c128 = arith.constant 128 : i32
    %c1_i64 = arith.constant 1 : i64
    %c32 = arith.constant 32 : i32
    %cst = arith.constant dense<0.0> : tensor<16x128xbf16>
    // CHECK: scf.for
    // CHECK-NOT: tt.make_tensor_descriptor
    // CHECK-NOT: tt.descriptor_load
    // CHECK: tt.load
    %r = scf.for %j = %c0 to %c32 step %c1 iter_args(%acc = %cst) -> (tensor<16x128xbf16>) : i32 {
      %desc = scf.if %cond -> (!tt.tensordesc<16x128xbf16>) {
        %d_inv = tt.make_tensor_descriptor %base_inv, [%s0, %s1], [%st0, %st1] {order = array<i32: 1, 0>} : <bf16>, <16x128xbf16>
        scf.yield %d_inv : !tt.tensordesc<16x128xbf16>
      } else {
        %base_var = tt.addptr %kv, %j : !tt.ptr<bf16>, i32
        %d_var = tt.make_tensor_descriptor %base_var, [%c16, %c128], [%st1, %c1_i64] {order = array<i32: 1, 0>} : <bf16>, <16x128xbf16>
        scf.yield %d_var : !tt.tensordesc<16x128xbf16>
      }
      %tile = tt.descriptor_load %desc[%c0, %c0] : !tt.tensordesc<16x128xbf16> -> tensor<16x128xbf16>
      scf.yield %tile : tensor<16x128xbf16>
    }
    tt.return %r : tensor<16x128xbf16>
  }
}

// -----
// Case 5b: TWO DESCRIPTORS CARRIED BY ONE LOOP. The loop's iter_args carry a
// loop-recreated descriptor and a hoistable one. No single *value* merges them,
// so a value-wise merge rule would demote only the recreated one — but the unit
// of rewriting is the op: the type converter expands every !tt.tensordesc in the
// scf.for signature at once, so a half-converted loop is not expressible.
// Entangling per-op demotes both, and the loop keeps no descriptor op.

// CHECK-LABEL: @two_descs_one_loop_demoted
module {
  tt.func public @two_descs_one_loop_demoted(
      %kv: !tt.ptr<bf16>, %base_inv: !tt.ptr<bf16>,
      %s0: i32, %s1: i32, %st0: i64, %st1: i64)
      -> tensor<16x128xbf16> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : i32
    %c128 = arith.constant 128 : i32
    %c1_i64 = arith.constant 1 : i64
    %c32 = arith.constant 32 : i32
    %cst = arith.constant dense<0.0> : tensor<16x128xbf16>
    %base0 = tt.addptr %kv, %c0 : !tt.ptr<bf16>, i32
    // Neither make survives the loop header (both demoted)...
    // CHECK-NOT: tt.make_tensor_descriptor
    %d_var0 = tt.make_tensor_descriptor %base0, [%c16, %c128], [%st1, %c1_i64] {order = array<i32: 1, 0>} : <bf16>, <16x128xbf16>
    %d_inv = tt.make_tensor_descriptor %base_inv, [%s0, %s1], [%st0, %st1] {order = array<i32: 1, 0>} : <bf16>, <16x128xbf16>
    // CHECK: scf.for
    // ... and neither descriptor_load does: each becomes a masked tt.load, the
    // in-loop one first, then the post-loop one. Both brackets are tight — a
    // surviving descriptor_load would trip the CHECK-NOT, and a non-demoted
    // descriptor would produce no tt.load for the following CHECK to match.
    // CHECK-NOT: tt.descriptor_load
    // CHECK: tt.load
    // CHECK-NOT: tt.descriptor_load
    // CHECK: tt.load
    %r:2 = scf.for %j = %c0 to %c32 step %c1
        iter_args(%dv = %d_var0, %di = %d_inv)
        -> (!tt.tensordesc<16x128xbf16>, !tt.tensordesc<16x128xbf16>) : i32 {
      %tile = tt.descriptor_load %dv[%c0, %c0] : !tt.tensordesc<16x128xbf16> -> tensor<16x128xbf16>
      // recreated every iteration: base depends on the induction variable
      %base_var = tt.addptr %kv, %j : !tt.ptr<bf16>, i32
      %d_var = tt.make_tensor_descriptor %base_var, [%c16, %c128], [%st1, %c1_i64] {order = array<i32: 1, 0>} : <bf16>, <16x128xbf16>
      scf.yield %d_var, %di : !tt.tensordesc<16x128xbf16>, !tt.tensordesc<16x128xbf16>
    }
    %out = tt.descriptor_load %r#1[%c0, %c0] : !tt.tensordesc<16x128xbf16> -> tensor<16x128xbf16>
    tt.return %out : tensor<16x128xbf16>
  }
}

// -----
// Case 6: MERGE WITH AN UNTRACEABLE DESCRIPTOR. Same shape as Case 5, but the
// scf.if's then-branch yields a descriptor FUNCTION ARGUMENT instead of a make.
// Provenance is therefore only partially nameable: the merged value names the
// loop-recreated make, but a block argument of the func — which no
// `MakeTensorDescOp` backs — also flows into it.
//
// Forward propagation seeds the func argument as unnameable and carries that
// through the merge, so the merged value comes back `unknown` while still naming
// the make. That pins the group: demoting the make would require rewriting the
// func signature, which stays legal in this mode. Nothing is rewritten — the
// loop keeps both the descriptor and its load, and NO masked tt.load appears.

// CHECK-LABEL: @merge_with_untraceable_kept
module {
  tt.func public @merge_with_untraceable_kept(
      %kv: !tt.ptr<bf16>, %argdesc: !tt.tensordesc<16x128xbf16>,
      %st1: i64, %cond: i1)
      -> tensor<16x128xbf16> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : i32
    %c128 = arith.constant 128 : i32
    %c1_i64 = arith.constant 1 : i64
    %c32 = arith.constant 32 : i32
    %cst = arith.constant dense<0.0> : tensor<16x128xbf16>
    // CHECK: scf.for
    // Nothing is demoted: the descriptor and its load survive, and no masked
    // pointer load is generated in the loop.
    // CHECK: tt.make_tensor_descriptor
    // CHECK: tt.descriptor_load
    // CHECK-NOT: tt.load
    %r = scf.for %j = %c0 to %c32 step %c1 iter_args(%acc = %cst) -> (tensor<16x128xbf16>) : i32 {
      %desc = scf.if %cond -> (!tt.tensordesc<16x128xbf16>) {
        // untraceable: a descriptor func arg, backed by no make
        scf.yield %argdesc : !tt.tensordesc<16x128xbf16>
      } else {
        %base_var = tt.addptr %kv, %j : !tt.ptr<bf16>, i32
        %d_var = tt.make_tensor_descriptor %base_var, [%c16, %c128], [%st1, %c1_i64] {order = array<i32: 1, 0>} : <bf16>, <16x128xbf16>
        scf.yield %d_var : !tt.tensordesc<16x128xbf16>
      }
      %tile = tt.descriptor_load %desc[%c0, %c0] : !tt.tensordesc<16x128xbf16> -> tensor<16x128xbf16>
      scf.yield %tile : tensor<16x128xbf16>
    }
    tt.return %r : tensor<16x128xbf16>
  }
}

// -----
// Case 7: MERGE WITH A POISON DESCRIPTOR. Same shape as Case 6, but the
// unnameable side is `ub.poison` rather than a function argument — the shape the
// pipeliner produces for a loop-carried descriptor's initial value.
//
// `ub.poison` is not an op the provenance walk models, so its result is poisoned
// at seed time and the merged descriptor comes back unnameable. That pins the
// make: demoting it would delete a descriptor the still-legal `ub.poison`
// operand is typed by. Nothing is rewritten.

// CHECK-LABEL: @merge_with_poison_kept
module {
  tt.func public @merge_with_poison_kept(
      %kv: !tt.ptr<bf16>, %st1: i64, %cond: i1)
      -> tensor<16x128xbf16> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : i32
    %c128 = arith.constant 128 : i32
    %c1_i64 = arith.constant 1 : i64
    %c32 = arith.constant 32 : i32
    %cst = arith.constant dense<0.0> : tensor<16x128xbf16>
    // CHECK: scf.for
    // Nothing is demoted: the descriptor and its load survive, and no masked
    // pointer load is generated in the loop.
    // CHECK: tt.make_tensor_descriptor
    // CHECK: tt.descriptor_load
    // CHECK-NOT: tt.load
    %r = scf.for %j = %c0 to %c32 step %c1 iter_args(%acc = %cst) -> (tensor<16x128xbf16>) : i32 {
      %desc = scf.if %cond -> (!tt.tensordesc<16x128xbf16>) {
        // unnameable: poison placeholder, backed by no make
        %p = ub.poison : !tt.tensordesc<16x128xbf16>
        scf.yield %p : !tt.tensordesc<16x128xbf16>
      } else {
        %base_var = tt.addptr %kv, %j : !tt.ptr<bf16>, i32
        %d_var = tt.make_tensor_descriptor %base_var, [%c16, %c128], [%st1, %c1_i64] {order = array<i32: 1, 0>} : <bf16>, <16x128xbf16>
        scf.yield %d_var : !tt.tensordesc<16x128xbf16>
      }
      %tile = tt.descriptor_load %desc[%c0, %c0] : !tt.tensordesc<16x128xbf16> -> tensor<16x128xbf16>
      scf.yield %tile : tensor<16x128xbf16>
    }
    tt.return %r : tensor<16x128xbf16>
  }
}
