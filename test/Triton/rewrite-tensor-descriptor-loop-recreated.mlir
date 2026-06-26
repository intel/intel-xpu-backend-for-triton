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
