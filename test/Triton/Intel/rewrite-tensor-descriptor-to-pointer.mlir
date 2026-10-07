// RUN: triton-opt %s --triton-intel-rewrite-tensor-descriptor-to-pointer --canonicalize --cse --split-input-file | FileCheck %s --implicit-check-not=unrealized_conversion_cast
// RUN: triton-opt %s --triton-intel-rewrite-tensor-descriptor-to-pointer --split-input-file | FileCheck %s --check-prefix=RAW --implicit-check-not=unrealized_conversion_cast
// RUN: triton-opt %s --triton-intel-rewrite-tensor-descriptor-to-pointer --cse --canonicalize --cse --split-input-file > %t.cse
// RUN: FileCheck %s --implicit-check-not=unrealized_conversion_cast < %t.cse
// RUN: triton-opt %t.cse --triton-intel-rewrite-tensor-descriptor-to-pointer --split-input-file > %t.twice
// RUN: FileCheck %s --check-prefix=TWICE < %t.twice

// COM: RAW checks bound maker/access counts before cleanup can hide extra clones.
// COM: Immediate CSE must run after fallback conversion, not between splitting
// COM: and conversion. TWICE checks ensure a second pass does not duplicate the
// COM: native clone; unrelated call-attribute inference may still update.

module {
  tt.func public @load(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32, %arg2: i32) -> (tensor<128x128xf32>) {
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c1_i64, %c256_i64] {order = array<i32: 0>} : <f32>, <128x128xf32>
    %3 = tt.descriptor_load %0[%arg1, %arg2] : !tt.tensordesc<128x128xf32> -> tensor<128x128xf32>
    tt.return %3 : tensor<128x128xf32>
  }
}

// CHECK-LABEL: @load
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor
// CHECK: tt.descriptor_load [[DESC]]

// -----

module {
  tt.func public @store(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32, %arg2: i32, %arg3: tensor<128x128xf32>) {
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c1_i64, %c256_i64] {order = array<i32: 0>} : <f32>, <128x128xf32>
    tt.descriptor_store %0[%arg1, %arg2], %arg3 : !tt.tensordesc<128x128xf32>, tensor<128x128xf32>
    tt.return
  }
}

// CHECK-LABEL: @store
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor
// CHECK: tt.descriptor_store [[DESC]]

// -----

module {
  tt.func public @callee(%tensordesc: !tt.tensordesc<128x128xf32>) -> !tt.tensordesc<128x128xf32> {
    tt.return %tensordesc : !tt.tensordesc<128x128xf32>
  }

  tt.func public @caller(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %c1_i64 = arith.constant 1 : i64
    %c256_i32 = arith.constant 256 : i32
    %c256_i64 = arith.constant 256 : i64
    %0 = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c256_i64, %c1_i64] {order = array<i32: 0>} : <f32>, <128x128xf32>
    %1 = tt.call @callee(%0) : (!tt.tensordesc<128x128xf32>) -> !tt.tensordesc<128x128xf32>
    tt.return
  }
}

// CHECK-LABEL: @callee
// CHECK-SAME: %[[PTR:[^:]*]]
// CHECK-SAME: %[[SHAPE0:[^:]*]]
// CHECK-SAME: %[[SHAPE1:[^:]*]]
// CHECK-SAME: %[[STRIDE0:[^:]*]]
// CHECK-SAME: %[[STRIDE1:[^:]*]]
// CHECK-SAME: %[[PAD:[^:]*]]
// CHECK-SAME: %[[ROUND:[^:]*]]
// CHECK-NEXT: tt.return %[[PTR]], %[[SHAPE0]], %[[SHAPE1]], %[[STRIDE0]], %[[STRIDE1]], %[[PAD]], %[[ROUND]]

// CHECK-LABEL: @caller
// CHECK-SAME: %[[PTR:[^:]*]]
// CHECK-DAG: %[[c1:.*]] = arith.constant 1 : i64
// CHECK-DAG: %[[c256:.*]] = arith.constant 256 : i64
// CHECK: %{{.*}}:7 = tt.call @callee(%[[PTR]], %[[c256]], %[[c256]], %[[c256]], %[[c1]], %false, %false)
// CHECK-SAME -> (!tt.ptr<f32>, i64, i64, i64, i64, i1, i1)

// -----

module {
  tt.func public @arg_attr(%arg0: !tt.tensordesc<128x128xf32>, %arg1: i32 {tt.divisibility = 16 : i32}) {
    tt.return
  }
}

// CHECK-LABEL: @arg_attr
// CHECK-SAME: %arg7: i32 {tt.divisibility = 16 : i32}) {

// -----

module {
  tt.func public @gather(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> (tensor<32x128xf32>) {
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c1_i64, %c256_i64] {order = array<i32: 0>} : <f32>, <1x128xf32>
    %cst = arith.constant dense<1> : tensor<32xi32>
    %3 = tt.descriptor_gather %0[%cst, %c0_i32] : (!tt.tensordesc<1x128xf32>, tensor<32xi32>, i32) -> tensor<32x128xf32>
    tt.return %3 : tensor<32x128xf32>
  }
}

// CHECK-LABEL: @gather
// CHECK-SAME: %[[ARG0:[^:]*]]
// CHECK-DAG: %[[CST:.*]] = arith.constant dense<0> : tensor<1x128xi64>
// CHECK-DAG: %[[CST_0:.*]] = arith.constant dense<256> : tensor<1x128xi64>
// CHECK-DAG: %[[CST_1:.*]] = arith.constant dense<1> : tensor<32x128xi64>
// CHECK-DAG: %[[CST_2:.*]] = arith.constant dense<0.000000e+00> : tensor<32x128xf32>

// CHECK-DAG: %[[VAL0:.*]] = tt.make_range {end = 128 : i32, start = 0 : i32}
// CHECK-DAG: %[[VAL1:.*]] = arith.extsi %[[VAL0]] :
// CHECK-DAG: %[[VAL2:.*]] = tt.expand_dims %[[VAL1]] {axis = 0 : i32}
// CHECK-DAG: %[[VAL3:.*]] = tt.splat %[[ARG0]] :
// CHECK-DAG: %[[VAL4:.*]] = tt.addptr %[[VAL3]], %[[CST_1]] :
// CHECK-DAG: %[[VAL5:.*]] = arith.muli %[[VAL2]], %[[CST_0]] :
// CHECK-DAG: %[[VAL6:.*]] = tt.broadcast %[[VAL5]] : tensor<1x128xi64> -> tensor<32x128xi64>
// CHECK-DAG: %[[VAL7:.*]] = tt.addptr %[[VAL4]], %[[VAL6]] :

// CHECK-DAG: %[[VAL8:.*]] = arith.cmpi sge, %[[VAL2]], %[[CST]]
// CHECK-DAG: %[[VAL9:.*]] = arith.cmpi slt, %[[VAL2]], %[[CST_0]]
// CHECK-DAG: %[[VAL10:.*]] = arith.andi %[[VAL8]], %[[VAL9]]
// CHECK-DAG: %[[VAL11:.*]] = tt.broadcast %[[VAL10]] : tensor<1x128xi1> -> tensor<32x128xi1>

// CHECK-DAG: %[[VAL12:.*]] = tt.load %[[VAL7]], %[[VAL11]], %[[CST_2]]
// CHECK: tt.return %[[VAL12]] :

// -----

module {
  tt.func public @multi_users(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32, %arg2: i32, %arg3: tensor<1x128xf32>) -> (tensor<32x128xf32>) {
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c1_i64, %c256_i64] {order = array<i32: 0>} : <f32>, <1x128xf32>
    %cst = arith.constant dense<1> : tensor<32xi32>
    %1 = tt.descriptor_gather %0[%cst, %c0_i32] : (!tt.tensordesc<1x128xf32>, tensor<32xi32>, i32) -> tensor<32x128xf32>
    tt.descriptor_store %0[%arg1, %arg2], %arg3 : !tt.tensordesc<1x128xf32>, tensor<1x128xf32>
    tt.return %1 : tensor<32x128xf32>
  }
}

// COM: The non-contiguous gather still makes the original descriptor unhandled.
// COM: Splitting gives the direct store its own native maker, while the gather
// COM: remains on the pointer fallback path. No memory operation is duplicated.
// CHECK-LABEL: @multi_users
// CHECK-SAME: %[[ARG0:[^:]*]]: !tt.ptr<f32>
// CHECK-SAME: %[[ARG1:[^:]*]]: i32
// CHECK-SAME: %[[ARG2:[^:]*]]: i32
// CHECK-SAME: %[[ARG3:[^:]*]]: tensor<1x128xf32>
// CHECK: %[[DESC:.*]] = tt.make_tensor_descriptor %[[ARG0]],

// COM: Gather path: lowered to tt.load with pointer arithmetic.
// CHECK-NOT: tt.make_tensor_descriptor
// CHECK-NOT: tt.descriptor_gather
// CHECK: %[[GATHER:.*]] = tt.load {{.*}} : tensor<32x128x!tt.ptr<f32>>

// COM: Store path: the direct descriptor operand uses the retained maker.
// CHECK-NOT: tt.make_tensor_descriptor
// CHECK-NOT: tt.store
// CHECK: tt.descriptor_store %[[DESC]][%[[ARG1]], %[[ARG2]]], %[[ARG3]] : !tt.tensordesc<1x128xf32>, tensor<1x128xf32>
// CHECK-NEXT: tt.return %[[GATHER]] : tensor<32x128xf32>

// -----

// COM: Host-side tensor descriptor: descriptor is a function argument with
// COM: frontend shape/stride args following it. A synthetic MakeTensorDescOp
// COM: should be inserted, preserving the descriptor_load on the fast path.
module {
  tt.func public @host_descriptor_load(%desc: !tt.tensordesc<128x64xf16>, %sh0: i32, %sh1: i32, %st0: i64, %st1: i64, %offset_y: i32, %offset_x: i32) -> tensor<128x64xf16> {
    %0 = tt.descriptor_load %desc[%offset_y, %offset_x] : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16>
    tt.return %0 : tensor<128x64xf16>
  }
}

// CHECK-LABEL: @host_descriptor_load
// COM: The function signature should be expanded (no !tt.tensordesc in args).
// CHECK-NOT: !tt.tensordesc
// COM: A synthetic MakeTensorDescOp is inserted and descriptor_load preserved.
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor
// CHECK: tt.descriptor_load [[DESC]]

// -----

// COM: Host-side descriptor store: same pattern as load.
module {
  tt.func public @host_descriptor_store(%desc: !tt.tensordesc<128x64xf16>, %sh0: i32, %sh1: i32, %st0: i64, %st1: i64, %offset_y: i32, %offset_x: i32, %data: tensor<128x64xf16>) {
    tt.descriptor_store %desc[%offset_y, %offset_x], %data : !tt.tensordesc<128x64xf16>, tensor<128x64xf16>
    tt.return
  }
}

// CHECK-LABEL: @host_descriptor_store
// CHECK-NOT: !tt.tensordesc
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor
// CHECK: tt.descriptor_store [[DESC]]

// -----

// COM: Host-side descriptor that feeds a gather op should NOT be preserved
// COM: (falls back to pointer path).
module {
  tt.func public @host_descriptor_gather(%desc: !tt.tensordesc<1x128xf32>, %sh0: i32, %sh1: i32, %st0: i64, %st1: i64, %offset: i32) -> tensor<32x128xf32> {
    %cst = arith.constant dense<1> : tensor<32xi32>
    %0 = tt.descriptor_gather %desc[%cst, %offset] : (!tt.tensordesc<1x128xf32>, tensor<32xi32>, i32) -> tensor<32x128xf32>
    tt.return %0 : tensor<32x128xf32>
  }
}

// CHECK-LABEL: @host_descriptor_gather
// COM: Should be lowered to pointer path (tt.load), not preserved.
// CHECK-NOT: tt.make_tensor_descriptor
// CHECK-NOT: tt.descriptor_gather
// CHECK: tt.load

// -----

// COM: Negative twin of @host_descriptor_load: same descriptor entry-block
// COM: argument, but the function is private, so synthesizeDescriptorsFromFuncArgs
// COM: skips it and the trace stays empty. The legality predicate relies on
// COM: DescriptorDefinitions::allSatisfy being *false* for an empty trace -- an
// COM: untraceable descriptor is not a candidate -- which makes the load illegal
// COM: and sends it down the pointer path. Were allSatisfy vacuously true on an
// COM: empty trace, the load would be ruled legal and survive as
// COM: tt.descriptor_load on an operand whose type the signature conversion has
// COM: already rewritten.
module {
  tt.func private @private_descriptor_load(%desc: !tt.tensordesc<128x64xf16>, %sh0: i32, %sh1: i32, %st0: i64, %st1: i64, %offset_y: i32, %offset_x: i32) -> tensor<128x64xf16> {
    %0 = tt.descriptor_load %desc[%offset_y, %offset_x] : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16>
    tt.return %0 : tensor<128x64xf16>
  }
}

// CHECK-LABEL: @private_descriptor_load
// CHECK-NOT: tt.make_tensor_descriptor
// CHECK-NOT: tt.descriptor_load
// CHECK: tt.load

// -----

// COM: === Legality/robustness of the 1->N descriptor expansion (issue #8166) ===
// COM:
// COM: A !tt.tensordesc produced by an `scf.if` RESULT. The `scf.if` itself has
// COM: no descriptor-typed OPERAND, and the pass' dynamic legality predicate only
// COM: inspects operands, so the `scf.if` is ruled legal while its `scf.yield` is
// COM: converted 1->7 (ptr + 2 shapes + 2 strides + padding:i1 + roundF32ToTF32:i1).
// COM: The result is a region-branch arity mismatch. `tt.descriptor_gather` is the
// COM: forcing function here: a gather consumer never registers its descriptor as a
// COM: candidate, so expansion is mandatory and cannot be dodged.
// COM:
// COM: Fix witness: with legality decided on operands alone the `scf.if` stays
// COM: legal and the verifier rejects the region-branch arity -- 7 yielded operands
// COM: against 1 expected input.
// COM:
// COM: Post-fix, the widened 7-result `scf.if` is NOT observable in this output and
// COM: must not be asserted: `scf.if` with side-effect-free arms canonicalizes to
// COM: per-result `arith.select`, so the `--canonicalize` in the RUN line always
// COM: removes the region op. (Verified separately: even with distinct base pointers
// COM: AND distinct padding in the two arms, so that nothing can be CSE'd, the
// COM: output still has no `scf.if` -- just `arith.select` on the differing
// COM: components.) Here both arms are identical, so even the selects fold away.
// COM: What guards the arity is the RUN line's exit status, not a CHECK: the base
// COM: failure was a region-branch VERIFIER error, so a regression makes triton-opt
// COM: exit non-zero and lit fails before FileCheck is ever consulted.
module {
  tt.func public @if_gather(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %cond: i1) -> tensor<32x128xf32> {
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %cst = arith.constant dense<1> : tensor<32xi32>
    %desc = scf.if %cond -> (!tt.tensordesc<1x128xf32>) {
      %d0 = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c1_i64, %c256_i64] {order = array<i32: 0>} : <f32>, <1x128xf32>
      scf.yield %d0 : !tt.tensordesc<1x128xf32>
    } else {
      %d1 = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c1_i64, %c256_i64] {order = array<i32: 0>} : <f32>, <1x128xf32>
      scf.yield %d1 : !tt.tensordesc<1x128xf32>
    }
    %0 = tt.descriptor_gather %desc[%cst, %c0_i32] : (!tt.tensordesc<1x128xf32>, tensor<32xi32>, i32) -> tensor<32x128xf32>
    tt.return %0 : tensor<32x128xf32>
  }
}

// CHECK-LABEL: @if_gather
// COM: `--canonicalize` hoists every constant to the top of the function, so the
// COM: region between the label and the first constant anchor holds only the
// COM: function header and constants, and a CHECK-NOT there could never fire. The
// COM: CHECK-NOTs in this case and the ones below therefore sit after the constant
// COM: anchors, between two positive anchors, where the expanded (or surviving) ops
// COM: actually are.
// COM: The zero splat exists only because the gather was expanded: a
// COM: `tt.descriptor_gather` has no `other` operand, so a no-op regression cannot
// COM: satisfy this line. Both arms request the default PAD_ZERO, so the padding
// COM: flag folds to `false` and the fill is the bare zero splat, with no select.
// CHECK: %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : tensor<32x128xf32>
// CHECK-NOT: scf.if
// CHECK-NOT: tt.make_tensor_descriptor
// CHECK-NOT: tt.descriptor_gather
// CHECK: %[[MASK:.*]] = tt.broadcast %{{.*}} : tensor<1x128xi1> -> tensor<32x128xi1>
// CHECK: %[[VAL:.*]] = tt.load %{{.*}}, %[[MASK]], %[[ZERO]] : tensor<32x128x!tt.ptr<f32>>
// CHECK: tt.return %[[VAL]] :

// -----

// COM: `scf.for` twin of @if_gather, and the regression test for issue #8167.
// COM: The descriptor is an iter-arg and the loop RESULT feeds the gather. Unlike
// COM: `scf.if`, this shape does not merely mis-convert: the pass walks the region
// COM: while it is transiently empty.
// COM:
// COM: Fix witness: the provenance walk reaches the loop body while it is
// COM: transiently empty and aborts on `SingleBlock<scf::ForOp>::getBody`'s
// COM: "unexpected empty region". Being a process abort, it takes the whole
// COM: -split-input-file run down with it, so no other chunk in this file can be
// COM: observed while it is broken. @for_load below is the control that isolates
// COM: the trigger.
// COM:
// COM: As with @if_gather, the expanded loop is NOT observable post-fix and must not
// COM: be asserted: every one of the 7 components is loop-invariant here (the yield
// COM: just passes the iter-arg through), so `--canonicalize` hoists them all out
// COM: and deletes the loop. @for_divergent_padding below is the case where a
// COM: component genuinely varies across iterations, and there the loop DOES survive
// COM: with an observable iter-arg list -- that is where the loop shape is asserted.
// COM: Here, as above, the guard against a regression is the RUN line's exit status:
// COM: the base failure was a process abort.
module {
  tt.func public @for_gather(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> tensor<32x128xf32> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %c256_i32 = arith.constant 256 : i32
    %cst = arith.constant dense<1> : tensor<32xi32>
    %d0 = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c1_i64, %c256_i64] {order = array<i32: 0>} : <f32>, <1x128xf32>
    %desc = scf.for %i = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%arg = %d0) -> (!tt.tensordesc<1x128xf32>) : i32 {
      scf.yield %arg : !tt.tensordesc<1x128xf32>
    }
    %0 = tt.descriptor_gather %desc[%cst, %c0_i32] : (!tt.tensordesc<1x128xf32>, tensor<32xi32>, i32) -> tensor<32x128xf32>
    tt.return %0 : tensor<32x128xf32>
  }
}

// CHECK-LABEL: @for_gather
// CHECK: %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : tensor<32x128xf32>
// CHECK-NOT: scf.for
// CHECK-NOT: tt.make_tensor_descriptor
// CHECK-NOT: tt.descriptor_gather
// CHECK: %[[MASK:.*]] = tt.broadcast %{{.*}} : tensor<1x128xi1> -> tensor<32x128xi1>
// CHECK: %[[VAL:.*]] = tt.load %{{.*}}, %[[MASK]], %[[ZERO]] : tensor<32x128x!tt.ptr<f32>>
// CHECK: tt.return %[[VAL]] :

// -----

// COM: CONTROL for @for_gather: byte-for-byte the same loop shape, but the
// COM: consumer is `tt.descriptor_load`, so the descriptor stays a candidate, the
// COM: loop is never converted, and nothing crashes. This is what proves the
// COM: #8167 crash needs the ForOp to be *converted*, not merely to carry a
// COM: descriptor iter-arg. Regression guard: green before and after the fix.
// COM:
// COM: Note the `scf.for` is absent from the output: with the descriptor left
// COM: alone the loop becomes trivially loop-invariant and the `--canonicalize` in
// COM: the RUN line erases it. That happens strictly AFTER the pass under test,
// COM: which did see the loop. The assertion that matters is that
// COM: `tt.make_tensor_descriptor` and `tt.descriptor_load` both survive.
module {
  tt.func public @for_load(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> tensor<128x128xf32> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %c256_i32 = arith.constant 256 : i32
    %d0 = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c1_i64, %c256_i64] {order = array<i32: 0>} : <f32>, <128x128xf32>
    %desc = scf.for %i = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%arg = %d0) -> (!tt.tensordesc<128x128xf32>) : i32 {
      scf.yield %arg : !tt.tensordesc<128x128xf32>
    }
    %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<128x128xf32> -> tensor<128x128xf32>
    tt.return %0 : tensor<128x128xf32>
  }
}

// CHECK-LABEL: @for_load
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor
// CHECK: tt.descriptor_load [[DESC]]
// CHECK: tt.return

// -----

// COM: `tt.call` sibling of @if_gather (also issue #8166): a `noinline` `tt.func`
// COM: RETURNS a !tt.tensordesc and the caller feeds that result to a gather. The
// COM: `tt.call` has no descriptor-typed operand, so the operands-only predicate
// COM: rules it legal while the callee signature is rewritten underneath it.
// COM:
// COM: Fix witness: the call site keeps its 1-result signature while the callee is
// COM: rewritten to 7, and the verifier reports "incorrect number of results for
// COM: callee".
// COM:
// COM: This is the ONE case of the three where the widened arity really is
// COM: observable -- a `tt.call` has no region for `--canonicalize` to collapse, so
// COM: the 7-result call site survives verbatim and is asserted below, mirroring the
// COM: @callee/@caller pair earlier in this file.
// COM:
// COM: It is also the only case where the padding flag itself survives as an i1: it
// COM: arrives as a call result, so nothing can fold the `arith.select` onto another
// COM: condition the way the divergent cases below do. That makes this the reference
// COM: for the canonical select direction -- flag true selects NaN.
module {
  tt.func private @make(%arg0: !tt.ptr<f32>) -> !tt.tensordesc<1x128xf32> attributes {noinline = true} {
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %c256_i32 = arith.constant 256 : i32
    %d0 = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c1_i64, %c256_i64] {order = array<i32: 0>} : <f32>, <1x128xf32>
    tt.return %d0 : !tt.tensordesc<1x128xf32>
  }
  tt.func public @call_gather(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> tensor<32x128xf32> {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<1> : tensor<32xi32>
    %desc = tt.call @make(%arg0) : (!tt.ptr<f32>) -> !tt.tensordesc<1x128xf32>
    %0 = tt.descriptor_gather %desc[%cst, %c0_i32] : (!tt.tensordesc<1x128xf32>, tensor<32xi32>, i32) -> tensor<32x128xf32>
    tt.return %0 : tensor<32x128xf32>
  }
}

// COM: The callee's signature and its `tt.return` both widen 1 -> 7. The tuple order
// COM: is ptr, shape0, shape1, stride0, stride1, padding:i1, roundF32ToTF32:i1, which
// COM: is what makes `#5` the padding flag below.
// CHECK-LABEL: @make
// CHECK-SAME: -> (!tt.ptr<f32>, i64, i64, i64, i64, i1, i1)
// CHECK: tt.return %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !tt.ptr<f32>, i64, i64, i64, i64, i1, i1

// CHECK-LABEL: @call_gather
// CHECK-SAME: %[[ARG0:[^:]*]]
// CHECK-DAG: %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : tensor<32x128xf32>
// CHECK-DAG: %[[NAN:.*]] = arith.constant dense<0x7FC00000> : tensor<32x128xf32>
// CHECK: %[[DESC:.*]]:7 = tt.call @make(%[[ARG0]]) : (!tt.ptr<f32>) -> (!tt.ptr<f32>, i64, i64, i64, i64, i1, i1)
// CHECK-NOT: !tt.tensordesc
// CHECK-NOT: tt.descriptor_gather
// CHECK: %[[MASK:.*]] = arith.andi %{{.*}}, %{{.*}} : tensor<32x128xi1>
// COM: The padding flag is opaque across the call boundary, so the select survives
// COM: and pins the direction: true -> NaN, false -> zero.
// CHECK: %[[OTHER:.*]] = arith.select %[[DESC]]#5, %[[NAN]], %[[ZERO]] : tensor<32x128xf32>
// CHECK: tt.load %{{.*}}, %[[MASK]], %[[OTHER]] : tensor<32x128x!tt.ptr<f32>>
// COM: `%[[DESC]]#6` is the roundF32ToTF32 flag; the TF32-rounding chain it guards is
// COM: shared with the pre-existing cases above and is deliberately not re-asserted
// COM: here, where the subject is the call arity.
// CHECK: tt.return

// -----

// COM: === Divergent descriptor padding (issue #8102) ===
// COM:
// COM: The issue's literal shape: an `scf.if` yields a PAD_ZERO descriptor from one
// COM: arm and a PAD_NAN one from the other, and the merge feeds a
// COM: `tt.descriptor_load`. `DescriptorDefinitions::consistentPadding()` returns
// COM: nullopt, so every producer of the padding decision bails; the
// COM: descriptor-native LLVM lowering then reads "no info" as PAD_ZERO and the
// COM: branch that asked for a NaN fill silently gets zeros.
// COM:
// COM: The fix routes such a load to this pointer-expansion path, which already
// COM: models padding correctly: padding becomes a runtime i1 in the flattened
// COM: descriptor and the out-of-bounds fill becomes
// COM: `arith.select %padFlag, NaN_splat, zero_splat` feeding `tt.load`'s `other`.
// COM:
// COM: Fix witness: without the fix the pass is a no-op on this shape -- both
// COM: `tt.make_tensor_descriptor`s and the `tt.descriptor_load` survive, tripping
// COM: the CHECK-NOTs below, and the load stays on the descriptor-native route
// COM: where the silent PAD_ZERO degradation happens downstream.
// COM:
// COM: Post-fix the `scf.if` is gone (side-effect-free arms canonicalize to
// COM: per-result selects) and the fill really is a runtime select, but NOTE THE
// COM: OPERAND ORDER: `--canonicalize` folds the i1 padding flag into the original
// COM: branch condition, so instead of `select %padFlag, NaN, zero` the output reads
// COM: `select %cond, zero, NaN` -- the then-arm asked for PAD_ZERO, so a true
// COM: condition selects the ZERO splat. Asserting the operand order is the point:
// COM: swapping it would silently give every branch the other branch's fill.
// COM: @call_gather above keeps the un-folded `select %padFlag, NaN, zero` direction,
// COM: because there the flag is opaque across a call boundary.
module {
  tt.func public @if_divergent_padding(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %cond: i1) -> tensor<128x128xf32> {
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %desc = scf.if %cond -> (!tt.tensordesc<128x128xf32>) {
      %d0 = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c1_i64, %c256_i64] {order = array<i32: 0>, padding = 1 : i32} : <f32>, <128x128xf32>
      scf.yield %d0 : !tt.tensordesc<128x128xf32>
    } else {
      %d1 = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c1_i64, %c256_i64] {order = array<i32: 0>, padding = 2 : i32} : <f32>, <128x128xf32>
      scf.yield %d1 : !tt.tensordesc<128x128xf32>
    }
    %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<128x128xf32> -> tensor<128x128xf32>
    tt.return %0 : tensor<128x128xf32>
  }
}

// CHECK-LABEL: @if_divergent_padding
// CHECK-DAG: %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
// CHECK-DAG: %[[NAN:.*]] = arith.constant dense<0x7FC00000> : tensor<128x128xf32>
// COM: The select's condition is literally the i1 function argument, i.e. the
// COM: original `%cond`: the padding choice is still driven by the same runtime
// COM: predicate that chose the descriptor, which is the whole correctness claim.
// CHECK: %[[OTHER:.*]] = arith.select %arg1, %[[ZERO]], %[[NAN]] : tensor<128x128xf32>
// CHECK-NOT: tt.make_tensor_descriptor
// CHECK-NOT: tt.descriptor_load
// CHECK: %[[MASK:.*]] = arith.andi %{{.*}}, %{{.*}} : tensor<128x128xi1>
// CHECK: %[[VAL:.*]] = tt.load %{{.*}}, %[[MASK]], %[[OTHER]] : tensor<128x128x!tt.ptr<f32>>
// CHECK: tt.return %[[VAL]] :

// -----

// COM: The same divergence with NO region op at all: `arith.select` on two
// COM: !tt.tensordesc values, which `findAllMakeTensorDescOps` traces through
// COM: (Utils/Utility.cpp, the arith::SelectOp case). This matters because it
// COM: decouples #8102 from #8166/#8167: a region-free shape cannot be perturbed
// COM: by the transient-empty-region window, so if this case regresses the cause
// COM: is the padding decision itself and nothing else.
// COM:
// COM: Fix witness: as above, the pass is a no-op on this shape without the fix, so
// COM: the surviving descriptor ops trip the CHECK-NOTs below.
// COM:
// COM: Post-fix output is identical to @if_divergent_padding's, down to the operand
// COM: order of the fill select -- which is itself the useful signal: a region op and
// COM: a plain `arith.select` on descriptors converge on the same expansion, so the
// COM: padding handling is not smuggled in by the region-branch machinery.
module {
  tt.func public @select_divergent_padding(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %cond: i1) -> tensor<128x128xf32> {
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %d0 = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c1_i64, %c256_i64] {order = array<i32: 0>, padding = 1 : i32} : <f32>, <128x128xf32>
    %d1 = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c1_i64, %c256_i64] {order = array<i32: 0>, padding = 2 : i32} : <f32>, <128x128xf32>
    %desc = arith.select %cond, %d0, %d1 : !tt.tensordesc<128x128xf32>
    %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<128x128xf32> -> tensor<128x128xf32>
    tt.return %0 : tensor<128x128xf32>
  }
}

// CHECK-LABEL: @select_divergent_padding
// CHECK-DAG: %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
// CHECK-DAG: %[[NAN:.*]] = arith.constant dense<0x7FC00000> : tensor<128x128xf32>
// CHECK: %[[OTHER:.*]] = arith.select %arg1, %[[ZERO]], %[[NAN]] : tensor<128x128xf32>
// CHECK-NOT: tt.make_tensor_descriptor
// CHECK-NOT: tt.descriptor_load
// CHECK: %[[MASK:.*]] = arith.andi %{{.*}}, %{{.*}} : tensor<128x128xi1>
// CHECK: %[[VAL:.*]] = tt.load %{{.*}}, %[[MASK]], %[[OTHER]] : tensor<128x128x!tt.ptr<f32>>
// CHECK: tt.return %[[VAL]] :

// -----

// COM: NEGATIVE TWIN of @if_divergent_padding, and the case that stops the fix
// COM: from over-evicting. Both `scf.if` arms ask for PAD_NAN, so
// COM: consistentPadding() yields PAD_NAN and the descriptor-native route is still
// COM: correct and still faster. The descriptor MUST be kept.
// COM:
// COM: The trace sees two candidates here regardless of the base pointers: the
// COM: two `tt.make_tensor_descriptor` ops live in sibling `scf.if` regions, and
// COM: the `--cse` in the RUN line runs only after the pass under test and does not
// COM: merge ops across sibling regions anyway. Using `%arg0` for both arms also
// COM: passes. The distinct base pointers (%arg0 vs %arg1) only make the two
// COM: producers visibly independent.
// COM:
// COM: Regression guard: green before and after the fix.
module {
  tt.func public @if_consistent_padding(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %cond: i1) -> tensor<128x128xf32> {
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %desc = scf.if %cond -> (!tt.tensordesc<128x128xf32>) {
      %d0 = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c1_i64, %c256_i64] {order = array<i32: 0>, padding = 2 : i32} : <f32>, <128x128xf32>
      scf.yield %d0 : !tt.tensordesc<128x128xf32>
    } else {
      %d1 = tt.make_tensor_descriptor %arg1, [%c256_i32, %c256_i32], [%c1_i64, %c256_i64] {order = array<i32: 0>, padding = 2 : i32} : <f32>, <128x128xf32>
      scf.yield %d1 : !tt.tensordesc<128x128xf32>
    }
    %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<128x128xf32> -> tensor<128x128xf32>
    tt.return %0 : tensor<128x128xf32>
  }
}

// CHECK-LABEL: @if_consistent_padding
// CHECK: [[DESC:%.*]] = scf.if
// COM: Both producers keep `padding = 2 : i32`, i.e. the PAD_NAN request survived
// COM: the trace. (PAD_ZERO is the default and is elided by the printer, so
// COM: matching the literal `padding = 2` is what discriminates the two.)
// CHECK: tt.make_tensor_descriptor {{.*}}padding = 2 : i32
// CHECK: tt.make_tensor_descriptor {{.*}}padding = 2 : i32
// CHECK: tt.descriptor_load [[DESC]]
// CHECK: tt.return

// -----

// COM: Divergent padding carried through an `scf.for` iter-arg: the init arg is
// COM: the PAD_ZERO descriptor and the yielded value is the PAD_NAN one, so the
// COM: block argument traces to both candidates. The consumer is an in-loop
// COM: `tt.descriptor_load` on that block argument.
// COM:
// COM: This is the only #8102 case that exercises the loop path, and this suite had
// COM: zero `scf.for` coverage before -- which is why both #8102 and #8167 survived
// COM: here undetected.
// COM:
// COM: Fix witness: without the fix the pass is a no-op here, so both
// COM: `tt.make_tensor_descriptor`s and the in-loop `tt.descriptor_load` survive.
// COM:
// COM: This is the one case in the file where the expanded loop survives
// COM: `--canonicalize`, and the surviving shape is much more interesting than the
// COM: 7-iter-arg form one might expect. Six of the seven components (ptr, both
// COM: shapes, both strides, roundF32ToTF32) are the same value on entry and on the
// COM: back edge, so they are hoisted out; the padding flag is NOT -- it enters as
// COM: `false` (PAD_ZERO, from the init arg %d0) and `true` is yielded (PAD_NAN, from
// COM: %d1). The loop therefore carries exactly `(i1, tensor<128x128xf32>)`: the
// COM: padding flag and the accumulator.
// COM:
// COM: That single loop-carried i1 IS the bug fix, made observable. On the
// COM: descriptor-native route there is nowhere to put it -- padding must be a
// COM: compile-time constant there -- which is precisely why a divergent descriptor
// COM: has to leave that route. If this loop ever stops carrying the flag, the first
// COM: iteration's zero fill has silently become every iteration's fill.
module {
  tt.func public @for_divergent_padding(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> tensor<128x128xf32> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %c256_i32 = arith.constant 256 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %d0 = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c1_i64, %c256_i64] {order = array<i32: 0>, padding = 1 : i32} : <f32>, <128x128xf32>
    %d1 = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c1_i64, %c256_i64] {order = array<i32: 0>, padding = 2 : i32} : <f32>, <128x128xf32>
    %res:2 = scf.for %i = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%accd = %d0, %acc = %cst) -> (!tt.tensordesc<128x128xf32>, tensor<128x128xf32>) : i32 {
      %v = tt.descriptor_load %accd[%c0_i32, %c0_i32] : !tt.tensordesc<128x128xf32> -> tensor<128x128xf32>
      scf.yield %d1, %v : !tt.tensordesc<128x128xf32>, tensor<128x128xf32>
    }
    tt.return %res#1 : tensor<128x128xf32>
  }
}

// CHECK-LABEL: @for_divergent_padding
// CHECK-DAG: %[[NAN:.*]] = arith.constant dense<0x7FC00000> : tensor<128x128xf32>
// CHECK-DAG: %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
// CHECK-DAG: %[[FALSE:.*]] = arith.constant false
// CHECK-DAG: %[[TRUE:.*]] = arith.constant true
// COM: Exactly two iter-args survive, and the first is the i1 padding flag entering
// COM: as PAD_ZERO. The second, the accumulator, is initialised with the same zero
// COM: splat the fill select uses -- that is an incidental CSE, not a claim about
// COM: padding.
// CHECK: %[[LOOP:.*]]:2 = scf.for {{.*}} iter_args(%[[PAD:[^ ]+]] = %[[FALSE]], %{{[^ ]+}} = %[[ZERO]]) -> (i1, tensor<128x128xf32>)
// COM: Inside the loop the fill is selected off the loop-carried flag, so it differs
// COM: between the first iteration (zero) and the rest (NaN).
// CHECK: %[[OTHER:.*]] = arith.select %[[PAD]], %[[NAN]], %[[ZERO]] : tensor<128x128xf32>
// CHECK-NOT: tt.make_tensor_descriptor
// CHECK-NOT: tt.descriptor_load
// CHECK: %[[MASK:.*]] = arith.andi %{{.*}}, %{{.*}} : tensor<128x128xi1>
// CHECK: %[[VAL:.*]] = tt.load %{{.*}}, %[[MASK]], %[[OTHER]] : tensor<128x128x!tt.ptr<f32>>
// COM: PAD_NAN is what the back edge yields, mirroring the source loop yielding the
// COM: PAD_NAN descriptor %d1.
// CHECK: scf.yield %[[TRUE]], %[[VAL]] : i1, tensor<128x128xf32>
// CHECK: tt.return %[[LOOP]]#1 :

// -----

// COM: The shared PAD_ZERO maker feeds a divergent merge and a direct load.
// COM: Selective splitting deliberately changes the former all-fallback policy:
// COM: the direct load keeps a native clone with its own PAD_ZERO padding, while
// COM: the merge still expands with the runtime-selected fill. The original and
// COM: the else-arm maker remain fallback-only; no descriptor materialization is
// COM: needed, and the return order must not change.
module {
  tt.func public @if_divergent_padding_shared_producer(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %cond: i1) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %d0 = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c1_i64, %c256_i64] {order = array<i32: 0>, padding = 1 : i32} : <f32>, <128x128xf32>
    %desc = scf.if %cond -> (!tt.tensordesc<128x128xf32>) {
      scf.yield %d0 : !tt.tensordesc<128x128xf32>
    } else {
      %d1 = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c1_i64, %c256_i64] {order = array<i32: 0>, padding = 2 : i32} : <f32>, <128x128xf32>
      scf.yield %d1 : !tt.tensordesc<128x128xf32>
    }
    %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<128x128xf32> -> tensor<128x128xf32>
    %1 = tt.descriptor_load %d0[%c0_i32, %c0_i32] : !tt.tensordesc<128x128xf32> -> tensor<128x128xf32>
    tt.return %0, %1 : tensor<128x128xf32>, tensor<128x128xf32>
  }
}

// CHECK-LABEL: @if_divergent_padding_shared_producer
// CHECK-DAG: %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
// CHECK-DAG: %[[NAN:.*]] = arith.constant dense<0x7FC00000> : tensor<128x128xf32>
// CHECK-DAG: %[[C256:.*]] = arith.constant 256 : i32
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i32
// COM: PAD_ZERO is the default and is elided; the complete dictionary excludes
// COM: PAD_NAN on this retained maker.
// CHECK: %[[DESC:.*]] = tt.make_tensor_descriptor %arg0, [%[[C256]], %[[C256]]], {{.*}} {order = array<i32: 0>} :
// CHECK-NOT: tt.make_tensor_descriptor
// CHECK: %[[OTHER:.*]] = arith.select %arg1, %[[ZERO]], %[[NAN]] : tensor<128x128xf32>
// CHECK-NOT: tt.make_tensor_descriptor
// CHECK-NOT: tt.descriptor_load
// CHECK: %[[MASK:.*]] = arith.andi %{{.*}}, %{{.*}} : tensor<128x128xi1>
// COM: Only the divergent access is pointer-based. The direct load uses exactly
// COM: the retained PAD_ZERO maker, not the divergent descriptor or fill.
// CHECK: %[[V0:.*]] = tt.load %{{.*}}, %[[MASK]], %[[OTHER]] : tensor<128x128x!tt.ptr<f32>>
// CHECK-NEXT: %[[V1:.*]] = tt.descriptor_load %[[DESC]][%[[C0]], %[[C0]]] : !tt.tensordesc<128x128xf32> -> tensor<128x128xf32>
// CHECK-NEXT: tt.return %[[V0]], %[[V1]] :

// -----

// COM: Eviction must be closed over every op that shares descriptors (82d7c7c16).
// COM: Two selects share the PAD_ZERO producer %dA: %s1 merges it with the PAD_NAN
// COM: %dB (divergent, so %s1's load is evicted), %s2 merges it with the PAD_ZERO
// COM: %dC (consistent on its own). Evicting only %s1's trace leaves %s2 mixing an
// COM: evicted %dA with a kept %dC, and with `buildMaterializations = false` that
// COM: leaks a `builtin.unrealized_conversion_cast`. Closing the eviction over %s2
// COM: evicts %dC too, so both loads are expanded.
// COM:
// COM: Fix witness: without the closure all three `tt.make_tensor_descriptor`s, both
// COM: selects and both `tt.descriptor_load`s survive. The leaked cast is the second
// COM: symptom, reachable only once the eviction is partial, which is why the
// COM: CHECK-NOTs below pin both.
module {
  tt.func public @select_shared_producer_chain(%a: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %b: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %c: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %p: i1, %q: i1, %o: i32) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %c256_i32 = arith.constant 256 : i32
    %dA = tt.make_tensor_descriptor %a, [%c256_i32, %c256_i32], [%c256_i64, %c1_i64] {padding = 1 : i32} : <f32>, <128x128xf32>
    %dB = tt.make_tensor_descriptor %b, [%c256_i32, %c256_i32], [%c256_i64, %c1_i64] {padding = 2 : i32} : <f32>, <128x128xf32>
    %dC = tt.make_tensor_descriptor %c, [%c256_i32, %c256_i32], [%c256_i64, %c1_i64] {padding = 1 : i32} : <f32>, <128x128xf32>
    %s1 = arith.select %p, %dA, %dB : !tt.tensordesc<128x128xf32>
    %s2 = arith.select %q, %dA, %dC : !tt.tensordesc<128x128xf32>
    %l1 = tt.descriptor_load %s1[%o, %o] : !tt.tensordesc<128x128xf32> -> tensor<128x128xf32>
    %l2 = tt.descriptor_load %s2[%o, %o] : !tt.tensordesc<128x128xf32> -> tensor<128x128xf32>
    tt.return %l1, %l2 : tensor<128x128xf32>, tensor<128x128xf32>
  }
}

// CHECK-LABEL: @select_shared_producer_chain
// CHECK-DAG: %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
// CHECK-DAG: %[[NAN:.*]] = arith.constant dense<0x7FC00000> : tensor<128x128xf32>
// COM: Both descriptor selects became base-pointer selects.
// CHECK: arith.select %arg3, %arg0, %arg1 : !tt.ptr<f32>
// CHECK-NOT: unrealized_conversion_cast
// CHECK: arith.select %arg4, %arg0, %arg2 : !tt.ptr<f32>
// CHECK-NOT: unrealized_conversion_cast
// COM: Only %s1 is divergent, so only its fill is a runtime select.
// CHECK: %[[OTHER:.*]] = arith.select %arg3, %[[ZERO]], %[[NAN]] : tensor<128x128xf32>
// CHECK-NOT: unrealized_conversion_cast
// CHECK-NOT: tt.make_tensor_descriptor
// CHECK-NOT: tt.descriptor_load
// CHECK: %[[V0:.*]] = tt.load %{{.*}}, %[[MASK:.*]], %[[OTHER]] : tensor<128x128x!tt.ptr<f32>>
// CHECK-NOT: unrealized_conversion_cast
// CHECK-NOT: tt.descriptor_load
// CHECK: %[[V1:.*]] = tt.load %{{.*}}, %[[MASK]], %[[ZERO]] : tensor<128x128x!tt.ptr<f32>>
// CHECK: tt.return %[[V0]], %[[V1]] :

// -----

// COM: Eviction closure over a loop carrying two descriptors (82d7c7c16). The
// COM: gathered descriptor %dG must be expanded (a gather is never a candidate), and
// COM: the `scf.for` also carries the loaded descriptor %dL. One op cannot mix a
// COM: legal and an illegal descriptor, so %dL is evicted with it and both
// COM: accesses become `tt.load`s. Neither descriptor asks for PAD_NAN, so both
// COM: fills are plain zero splats.
// COM:
// COM: Fix witness: without the region-less guard the provenance walk aborts on
// COM: `SingleBlock<scf::ForOp>::getBody`'s "unexpected empty region".
module {
  tt.func public @for_mixed_gather_load(%a: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %b: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %n: i32) -> (tensor<32x128xf32>, tensor<128x128xf32>) {
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c256_i32 = arith.constant 256 : i32
    %cst = arith.constant dense<1> : tensor<32xi32>
    %dG = tt.make_tensor_descriptor %a, [%c256_i32, %c256_i32], [%c256_i64, %c1_i64] : <f32>, <1x128xf32>
    %dL = tt.make_tensor_descriptor %b, [%c256_i32, %c256_i32], [%c256_i64, %c1_i64] : <f32>, <128x128xf32>
    %r:2 = scf.for %i = %c0_i32 to %n step %c1_i32 iter_args(%g = %dG, %l = %dL) -> (!tt.tensordesc<1x128xf32>, !tt.tensordesc<128x128xf32>) : i32 {
      scf.yield %g, %l : !tt.tensordesc<1x128xf32>, !tt.tensordesc<128x128xf32>
    }
    %0 = tt.descriptor_gather %r#0[%cst, %c0_i32] : (!tt.tensordesc<1x128xf32>, tensor<32xi32>, i32) -> tensor<32x128xf32>
    %1 = tt.descriptor_load %r#1[%c0_i32, %c0_i32] : !tt.tensordesc<128x128xf32> -> tensor<128x128xf32>
    tt.return %0, %1 : tensor<32x128xf32>, tensor<128x128xf32>
  }
}

// CHECK-LABEL: @for_mixed_gather_load
// CHECK-DAG: %[[ZERO_L:.*]] = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
// CHECK-DAG: %[[ZERO_G:.*]] = arith.constant dense<0.000000e+00> : tensor<32x128xf32>
// CHECK: tt.make_range
// CHECK-NOT: unrealized_conversion_cast
// CHECK-NOT: tt.make_tensor_descriptor
// CHECK-NOT: tt.descriptor_gather
// CHECK-NOT: tt.descriptor_load
// CHECK: tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x128x!tt.ptr<f32>>
// CHECK-NOT: unrealized_conversion_cast
// CHECK-NOT: tt.descriptor_gather
// CHECK: %[[V0:.*]] = tt.load %{{.*}}, %{{.*}}, %[[ZERO_G]] : tensor<32x128x!tt.ptr<f32>>
// CHECK-NOT: unrealized_conversion_cast
// CHECK-NOT: tt.descriptor_load
// CHECK: tt.splat %arg1 : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>>
// CHECK: %[[V1:.*]] = tt.load %{{.*}}, %{{.*}}, %[[ZERO_L]] : tensor<128x128x!tt.ptr<f32>>
// CHECK: tt.return %[[V0]], %[[V1]] :

// -----

// COM: A loop result is its tied init when the loop runs zero times (6bd0d53ae).
// COM: %r is %dZ (PAD_ZERO) on a zero-trip loop and %dN (PAD_NAN) otherwise, so
// COM: its provenance is divergent and the load is expanded. The padding flag is
// COM: carried through the loop, entering as false (PAD_ZERO) and yielded as true.
// COM:
// COM: Fix witness: the same "unexpected empty region" abort as @for_mixed_gather_load
// COM: without the region-less guard. See @for_zero_trip_divergent_padding in
// COM: test/TritonIntelGPU/find-defining-op-loops.mlir for the TTGIR side.
module {
  tt.func @for_zero_trip_divergent_padding(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %pitch: i64 {tt.divisibility = 16 : i32}, %n: i32) -> tensor<64x32xf16> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %dZ = tt.make_tensor_descriptor %arg0, [%c64_i32, %c32_i32], [%pitch, %c1_i64] {padding = 1 : i32} : !tt.ptr<f16>, !tt.tensordesc<64x32xf16>
    %r = scf.for %i = %c0_i32 to %n step %c1_i32 iter_args(%x = %dZ) -> (!tt.tensordesc<64x32xf16>) : i32 {
      %dN = tt.make_tensor_descriptor %arg1, [%c64_i32, %c32_i32], [%pitch, %c1_i64] {padding = 2 : i32} : !tt.ptr<f16>, !tt.tensordesc<64x32xf16>
      scf.yield %dN : !tt.tensordesc<64x32xf16>
    }
    %ld = tt.descriptor_load %r[%c0_i32, %c0_i32] : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16>
    tt.return %ld : tensor<64x32xf16>
  }
}

// CHECK-LABEL: @for_zero_trip_divergent_padding
// CHECK-DAG: %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : tensor<64x32xf16>
// CHECK-DAG: %[[NAN:.*]] = arith.constant dense<0x7E00> : tensor<64x32xf16>
// CHECK-DAG: %[[TRUE:.*]] = arith.constant true
// CHECK-DAG: %[[FALSE:.*]] = arith.constant false
// CHECK: %[[R:.*]]:2 = scf.for {{.*}} iter_args(%{{[^ ]+}} = %arg0, %{{[^ ]+}} = %[[FALSE]]) -> (!tt.ptr<f16>, i1)
// CHECK-NOT: unrealized_conversion_cast
// CHECK-NOT: tt.make_tensor_descriptor
// CHECK: scf.yield %arg1, %[[TRUE]] : !tt.ptr<f16>, i1
// CHECK: %[[OTHER:.*]] = arith.select %[[R]]#1, %[[NAN]], %[[ZERO]] : tensor<64x32xf16>
// CHECK-NOT: unrealized_conversion_cast
// CHECK-NOT: tt.make_tensor_descriptor
// CHECK-NOT: tt.descriptor_load
// CHECK: tt.splat %[[R]]#0 : !tt.ptr<f16> -> tensor<64x32x!tt.ptr<f16>>
// CHECK-NOT: tt.descriptor_load
// CHECK: %[[V:.*]] = tt.load %{{.*}}, %{{.*}}, %[[OTHER]] : tensor<64x32x!tt.ptr<f16>>
// CHECK: tt.return %[[V]] :

// -----

// COM: An scf.while result is the matching scf.condition operand, not the
// COM: after-region yield (0fa440764). %r is always %dN (PAD_NAN); %dZ only
// COM: re-enters the before region.
// COM:
// COM: The load is expanded to pointers, and that is intended: the while's init
// COM: %dZ is not a candidate, so the closure of 82d7c7c16 evicts the whole group
// COM: that the scf.while ties together, %dN included. The fill is the bare NaN
// COM: splat (no select), because the load's own provenance is exactly %dN, and
// COM: the pointer is %arg1, %dN's base.
// COM:
// COM: Fix witness: the verifier rejects the scf.while region-branch arity -- 7
// COM: `scf.condition` operands against 1 expected input. See @while_condition_padding
// COM: in test/TritonIntelGPU/find-defining-op-loops.mlir for the TTGIR side.
module {
  tt.func @while_condition_padding(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %pitch: i64 {tt.divisibility = 16 : i32}, %c: i1) -> tensor<64x32xf16> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %dZ = tt.make_tensor_descriptor %arg0, [%c64_i32, %c32_i32], [%pitch, %c1_i64] {padding = 1 : i32} : !tt.ptr<f16>, !tt.tensordesc<64x32xf16>
    %r = scf.while (%x = %dZ) : (!tt.tensordesc<64x32xf16>) -> !tt.tensordesc<64x32xf16> {
      %dN = tt.make_tensor_descriptor %arg1, [%c64_i32, %c32_i32], [%pitch, %c1_i64] {padding = 2 : i32} : !tt.ptr<f16>, !tt.tensordesc<64x32xf16>
      scf.condition(%c) %dN : !tt.tensordesc<64x32xf16>
    } do {
    ^bb0(%y: !tt.tensordesc<64x32xf16>):
      scf.yield %dZ : !tt.tensordesc<64x32xf16>
    }
    %ld = tt.descriptor_load %r[%c0_i32, %c0_i32] : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16>
    tt.return %ld : tensor<64x32xf16>
  }
}

// CHECK-LABEL: @while_condition_padding
// CHECK: %[[NAN:.*]] = arith.constant dense<0x7E00> : tensor<64x32xf16>
// CHECK: tt.make_range
// CHECK-NOT: unrealized_conversion_cast
// CHECK-NOT: tt.make_tensor_descriptor
// CHECK-NOT: tt.descriptor_load
// CHECK: tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x32x!tt.ptr<f16>>
// CHECK-NOT: unrealized_conversion_cast
// CHECK-NOT: tt.descriptor_load
// CHECK: %[[V:.*]] = tt.load %{{.*}}, %{{.*}}, %[[NAN]] : tensor<64x32x!tt.ptr<f16>>
// CHECK: tt.return %[[V]] :

// -----

// COM: The untraceable function argument still evicts the descriptor-return
// COM: group (#8170), and both descriptor results must expand 1 -> 7. Splitting
// COM: now isolates the direct load on a native clone rather than expanding it
// COM: with the return operands. Pin every returned component to catch mixed
// COM: representations, swapped results, or a leaked materialization.
module {
  tt.func private @f(%arg: !tt.tensordesc<128x128xf32>, %c: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %o: i32) -> (tensor<128x128xf32>, !tt.tensordesc<128x128xf32>, !tt.tensordesc<128x128xf32>) attributes {noinline = true} {
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %c256_i32 = arith.constant 256 : i32
    %dC = tt.make_tensor_descriptor %c, [%c256_i32, %c256_i32], [%c256_i64, %c1_i64] {padding = 1 : i32} : <f32>, <128x128xf32>
    %l = tt.descriptor_load %dC[%o, %o] : !tt.tensordesc<128x128xf32> -> tensor<128x128xf32>
    tt.return %l, %dC, %arg : tensor<128x128xf32>, !tt.tensordesc<128x128xf32>, !tt.tensordesc<128x128xf32>
  }
}

// CHECK-LABEL: @f
// CHECK-SAME: %[[PTR:[^:]*]]: !tt.ptr<f32>, %[[SHAPE0:[^:]*]]: i64, %[[SHAPE1:[^:]*]]: i64, %[[STRIDE0:[^:]*]]: i64, %[[STRIDE1:[^:]*]]: i64, %[[PAD:[^:]*]]: i1, %[[ROUND:[^:]*]]: i1,
// CHECK-SAME: %[[LOCAL_PTR:[^:]*]]: !tt.ptr<f32>
// CHECK-SAME: %[[OFFSET:[^:]*]]: i32) -> (tensor<128x128xf32>, !tt.ptr<f32>, i64, i64, i64, i64, i1, i1, !tt.ptr<f32>, i64, i64, i64, i64, i1, i1)
// CHECK-DAG: %[[C256_I32:.*]] = arith.constant 256 : i32
// CHECK-DAG: %[[C256_I64:.*]] = arith.constant 256 : i64
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i64
// CHECK-DAG: %[[FALSE:.*]] = arith.constant false
// CHECK-NOT: tt.make_tensor_descriptor
// CHECK-NOT: tt.load
// CHECK: %[[DESC:.*]] = tt.make_tensor_descriptor %[[LOCAL_PTR]], [%[[C256_I32]], %[[C256_I32]]], [%[[C256_I64]], %[[C1]]]
// CHECK-NEXT: %[[V:.*]] = tt.descriptor_load %[[DESC]][%[[OFFSET]], %[[OFFSET]]] : !tt.tensordesc<128x128xf32> -> tensor<128x128xf32>
// CHECK-NEXT: tt.return %[[V]], %[[LOCAL_PTR]], %[[C256_I64]], %[[C256_I64]], %[[C256_I64]], %[[C1]], %[[FALSE]], %[[FALSE]], %[[PTR]], %[[SHAPE0]], %[[SHAPE1]], %[[STRIDE0]], %[[STRIDE1]], %[[PAD]], %[[ROUND]] : tensor<128x128xf32>, !tt.ptr<f32>, i64, i64, i64, i64, i1, i1, !tt.ptr<f32>, i64, i64, i64, i64, i1, i1

// -----

// COM: A select joining a local descriptor and an untraceable function argument
// COM: must not crash AxisInfo analysis (#8170). The first, direct load now keeps
// COM: a native maker, while the selected access still expands. Its incoming
// COM: descriptor may request TF32 rounding, so the second returned tensor is the
// COM: post-load conditional result, not necessarily the raw tt.load result.
module {
  tt.func public @desc_select_untraceable(%arg: !tt.tensordesc<128x128xf32>, %c: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %p: i1, %o: i32) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %c256_i32 = arith.constant 256 : i32
    %d = tt.make_tensor_descriptor %c, [%c256_i32, %c256_i32], [%c256_i64, %c1_i64] {padding = 1 : i32} : <f32>, <128x128xf32>
    %l = tt.descriptor_load %d[%o, %o] : !tt.tensordesc<128x128xf32> -> tensor<128x128xf32>
    %s = arith.select %p, %d, %arg : !tt.tensordesc<128x128xf32>
    %l2 = tt.descriptor_load %s[%o, %o] : !tt.tensordesc<128x128xf32> -> tensor<128x128xf32>
    tt.return %l, %l2 : tensor<128x128xf32>, tensor<128x128xf32>
  }
}

// CHECK-LABEL: @desc_select_untraceable
// CHECK-SAME: %[[PTR:[^:]*]]: !tt.ptr<f32>
// CHECK-SAME: %[[PAD:[^:]*]]: i1, %[[ROUND:[^:]*]]: i1, %[[LOCAL_PTR:[^:]*]]: !tt.ptr<f32>
// CHECK-SAME: %[[COND:[^:]*]]: i1, %[[OFFSET:[^:]*]]: i32)
// CHECK: %[[DESC:.*]] = tt.make_tensor_descriptor %[[LOCAL_PTR]],
// CHECK-NEXT: %[[LOCAL:.*]] = tt.descriptor_load %[[DESC]][%[[OFFSET]], %[[OFFSET]]] : !tt.tensordesc<128x128xf32> -> tensor<128x128xf32>
// CHECK-NOT: tt.make_tensor_descriptor
// CHECK-NOT: tt.descriptor_load
// CHECK: %[[SELECTED_PTR:.*]] = arith.select %[[COND]], %[[LOCAL_PTR]], %[[PTR]] : !tt.ptr<f32>
// CHECK: tt.splat %[[SELECTED_PTR]] : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>>
// CHECK: %[[LOADED:.*]] = tt.load {{.*}} : tensor<128x128x!tt.ptr<f32>>
// COM: Track both branches of the existing TF32 post-processing to prove the
// COM: second return still consumes the selected load, and not the native one.
// CHECK: %[[SELECTED:.*]] = scf.if {{.*}} -> (tensor<128x128xf32>)
// CHECK: %[[BITS:.*]] = tt.bitcast %[[LOADED]] : tensor<128x128xf32> -> tensor<128x128xi32>
// CHECK: %[[ROUNDED:.*]] = tt.bitcast %{{.*}} : tensor<128x128xi32> -> tensor<128x128xf32>
// CHECK: scf.yield %[[ROUNDED]] : tensor<128x128xf32>
// CHECK: } else {
// CHECK-NEXT: scf.yield %[[LOADED]] : tensor<128x128xf32>
// CHECK: tt.return %[[LOCAL]], %[[SELECTED]] : tensor<128x128xf32>, tensor<128x128xf32>

// -----

// Multi-dimensional descriptor AxisInfo is not forwarded as scalar hints to a
// callee.
module {
  tt.func public @desc_call_kernel(%arg: !tt.tensordesc<32x32xf32> {tt.divisibility = 16 : i32}, %out: !tt.ptr<f32>) {
    %v = tt.call @desc_call_callee(%arg) : (!tt.tensordesc<32x32xf32>) -> tensor<32x32xf32>
    %p = tt.splat %out : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>>
    tt.store %p, %v : tensor<32x32x!tt.ptr<f32>>
    tt.return
  }
  tt.func private @desc_call_callee(%arg: !tt.tensordesc<32x32xf32>) -> tensor<32x32xf32> attributes {noinline = true} {
    %c0 = arith.constant 0 : i32
    %v = tt.descriptor_load %arg[%c0, %c0] : !tt.tensordesc<32x32xf32> -> tensor<32x32xf32>
    tt.return %v : tensor<32x32xf32>
  }
}

// CHECK-LABEL: @desc_call_kernel
// CHECK: tt.call @desc_call_callee
// CHECK: tt.store
// CHECK-LABEL: tt.func private @desc_call_callee
// CHECK-NOT: tt.contiguity
// CHECK-NOT: tt.divisibility
// CHECK-NOT: tt.constancy
// CHECK: arith.constant
// CHECK: tt.load
// CHECK: tt.return

// -----

// COM: Wholly native sharing is not a split request. The two loads and the
// COM: intervening store must share one maker, even before CSE. Different load
// COM: offsets and the store prevent cleanup from hiding duplicated accesses.
module {
  tt.func public @native_only_shared(%base: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %height: i32, %width: i32, %pitch: i64, %row: i32, %col0: i32, %col1: i32) -> (tensor<8x16xf16>, tensor<8x16xf16>) {
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %base, [%height, %width], [%pitch, %c1_i64] : <f16>, <8x16xf16>
    %v0 = tt.descriptor_load %desc[%row, %col0] : !tt.tensordesc<8x16xf16> -> tensor<8x16xf16>
    tt.descriptor_store %desc[%row, %col0], %v0 : !tt.tensordesc<8x16xf16>, tensor<8x16xf16>
    %v1 = tt.descriptor_load %desc[%row, %col1] : !tt.tensordesc<8x16xf16> -> tensor<8x16xf16>
    tt.return %v0, %v1 : tensor<8x16xf16>, tensor<8x16xf16>
  }
}

// CHECK-LABEL: @native_only_shared
// CHECK: %[[DESC:.*]] = tt.make_tensor_descriptor
// CHECK-NEXT: %[[V0:.*]] = tt.descriptor_load %[[DESC]]
// CHECK-NEXT: tt.descriptor_store %[[DESC]]{{.*}}, %[[V0]] :
// CHECK-NEXT: %[[V1:.*]] = tt.descriptor_load %[[DESC]]
// CHECK-NEXT: tt.return %[[V0]], %[[V1]] :

// RAW-LABEL: @native_only_shared
// RAW-SAME: %[[BASE:[^:]*]]: !tt.ptr<f16>
// RAW-SAME: %[[HEIGHT:[^:]*]]: i32, %[[WIDTH:[^:]*]]: i32, %[[PITCH:[^:]*]]: i64, %[[ROW:[^:]*]]: i32, %[[COL0:[^:]*]]: i32, %[[COL1:[^:]*]]: i32)
// RAW-NOT: tt.make_tensor_descriptor
// RAW: %[[ONE:.*]] = arith.constant 1 : i64
// RAW-NOT: tt.make_tensor_descriptor
// RAW: %[[DESC:.*]] = tt.make_tensor_descriptor %[[BASE]], [%[[HEIGHT]], %[[WIDTH]]], [%[[PITCH]], %[[ONE]]] :
// RAW-NEXT: %[[V0:.*]] = tt.descriptor_load %[[DESC]][%[[ROW]], %[[COL0]]] :
// RAW-NEXT: tt.descriptor_store %[[DESC]][%[[ROW]], %[[COL0]]], %[[V0]] :
// RAW-NEXT: %[[V1:.*]] = tt.descriptor_load %[[DESC]][%[[ROW]], %[[COL1]]] :
// RAW-NEXT: tt.return %[[V0]], %[[V1]] :
// RAW-NOT: tt.make_tensor_descriptor
// RAW-NOT: tt.descriptor_load
// RAW-NOT: tt.descriptor_store

// -----

// COM: Wholly fallback sharing has no direct load/store descriptor operand to
// COM: rescue. Repeated row indices are neither contiguous nor at most four
// COM: contiguous sub-ranges, so the greedy gather pre-rewrites do not apply.
module {
  tt.func public @fallback_only_shared(%base: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %height: i32, %width: i32, %pitch: i64, %col0: i32, %col1: i32) -> (tensor<32x16xf16>, tensor<32x16xf16>) {
    %c1_i64 = arith.constant 1 : i64
    %rows = arith.constant dense<1> : tensor<32xi32>
    %desc = tt.make_tensor_descriptor %base, [%height, %width], [%pitch, %c1_i64] : <f16>, <1x16xf16>
    %v0 = tt.descriptor_gather %desc[%rows, %col0] : (!tt.tensordesc<1x16xf16>, tensor<32xi32>, i32) -> tensor<32x16xf16>
    %v1 = tt.descriptor_gather %desc[%rows, %col1] : (!tt.tensordesc<1x16xf16>, tensor<32xi32>, i32) -> tensor<32x16xf16>
    tt.return %v0, %v1 : tensor<32x16xf16>, tensor<32x16xf16>
  }
}

// CHECK-LABEL: @fallback_only_shared
// CHECK-NOT: tt.{{make_tensor_descriptor|descriptor_}}
// CHECK: %[[V0:.*]] = tt.load {{.*}} : tensor<32x16x!tt.ptr<f16>>
// CHECK-NOT: tt.{{make_tensor_descriptor|descriptor_}}
// CHECK: %[[V1:.*]] = tt.load {{.*}} : tensor<32x16x!tt.ptr<f16>>
// CHECK-NOT: tt.{{make_tensor_descriptor|descriptor_}}
// CHECK: tt.return %[[V0]], %[[V1]] :

// RAW-LABEL: @fallback_only_shared
// RAW-NOT: tt.{{make_tensor_descriptor|descriptor_}}
// RAW: %[[V0:.*]] = tt.load {{.*}} : tensor<32x16x!tt.ptr<f16>>
// RAW-NOT: tt.{{make_tensor_descriptor|descriptor_}}
// RAW-NOT: tt.load
// RAW: %[[V1:.*]] = tt.load {{.*}} : tensor<32x16x!tt.ptr<f16>>
// RAW-NOT: tt.{{make_tensor_descriptor|descriptor_}}
// RAW-NOT: tt.load
// RAW: tt.return %[[V0]], %[[V1]] :
// RAW-NOT: tt.{{make_tensor_descriptor|descriptor_}}

// -----

// COM: One evicted maker has several direct users and one select-connected
// COM: fallback access. All direct descriptor operands must share ONE complete
// COM: native clone, including PAD_NAN, dynamic shape/stride operands, and the
// COM: discardable order attribute. No load or store may be cloned. f16 isolates
// COM: splitting/padding from the focal f32 case's runtime TF32 processing.
module {
  tt.func public @split_multiple_direct_users_nan(%incoming: !tt.tensordesc<8x16xf16>, %base: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %height: i32, %width: i32, %pitch: i64, %choose_local: i1, %row: i32, %col0: i32, %col1: i32) -> (tensor<8x16xf16>, tensor<8x16xf16>, tensor<8x16xf16>) {
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %base, [%height, %width], [%pitch, %c1_i64] {order = array<i32: 1, 0>, padding = 2 : i32} : <f16>, <8x16xf16>
    %v0 = tt.descriptor_load %desc[%row, %col0] : !tt.tensordesc<8x16xf16> -> tensor<8x16xf16>
    tt.descriptor_store %desc[%row, %col0], %v0 : !tt.tensordesc<8x16xf16>, tensor<8x16xf16>
    %v1 = tt.descriptor_load %desc[%row, %col1] : !tt.tensordesc<8x16xf16> -> tensor<8x16xf16>
    %selected = arith.select %choose_local, %desc, %incoming : !tt.tensordesc<8x16xf16>
    %v2 = tt.descriptor_load %selected[%row, %col1] : !tt.tensordesc<8x16xf16> -> tensor<8x16xf16>
    tt.return %v0, %v1, %v2 : tensor<8x16xf16>, tensor<8x16xf16>, tensor<8x16xf16>
  }
}

// CHECK-LABEL: @split_multiple_direct_users_nan
// CHECK: %[[DESC:.*]] = tt.make_tensor_descriptor {{.*}} {order = array<i32: 1, 0>, padding = 2 : i32} :
// CHECK-NEXT: %[[V0:.*]] = tt.descriptor_load %[[DESC]]
// CHECK-NEXT: tt.descriptor_store %[[DESC]]{{.*}}, %[[V0]] :
// CHECK-NEXT: %[[V1:.*]] = tt.descriptor_load %[[DESC]]
// CHECK-NOT: tt.make_tensor_descriptor
// CHECK-NOT: tt.descriptor_
// CHECK: %[[V2:.*]] = tt.load {{.*}} : tensor<8x16x!tt.ptr<f16>>
// CHECK-NEXT: tt.return %[[V0]], %[[V1]], %[[V2]] :

// RAW-LABEL: @split_multiple_direct_users_nan
// RAW-SAME: %[[INCOMING_PTR:[^:]*]]: !tt.ptr<f16>
// RAW-SAME: %[[PAD:[^:]*]]: i1, %[[ROUND:[^:]*]]: i1, %[[BASE:[^:]*]]: !tt.ptr<f16>
// RAW-SAME: %[[HEIGHT:[^:]*]]: i32, %[[WIDTH:[^:]*]]: i32, %[[PITCH:[^:]*]]: i64, %[[COND:[^:]*]]: i1, %[[ROW:[^:]*]]: i32, %[[COL0:[^:]*]]: i32, %[[COL1:[^:]*]]: i32)
// RAW-NOT: tt.make_tensor_descriptor
// RAW: %[[ONE:.*]] = arith.constant 1 : i64
// RAW-NOT: tt.make_tensor_descriptor
// RAW: %[[DESC:.*]] = tt.make_tensor_descriptor %[[BASE]], [%[[HEIGHT]], %[[WIDTH]]], [%[[PITCH]], %[[ONE]]] {order = array<i32: 1, 0>, padding = 2 : i32} :
// RAW-NEXT: %[[V0:.*]] = tt.descriptor_load %[[DESC]][%[[ROW]], %[[COL0]]] :
// RAW-NEXT: tt.descriptor_store %[[DESC]][%[[ROW]], %[[COL0]]], %[[V0]] :
// RAW-NEXT: %[[V1:.*]] = tt.descriptor_load %[[DESC]][%[[ROW]], %[[COL1]]] :
// RAW-NOT: tt.make_tensor_descriptor
// RAW-NOT: tt.descriptor_
// RAW: %[[SELECTED:.*]]:7 = scf.if %[[COND]] -> (!tt.ptr<f16>, i64, i64, i64, i64, i1, i1) {
// RAW: scf.yield %[[BASE]], {{.*}} : !tt.ptr<f16>, i64, i64, i64, i64, i1, i1
// RAW: } else {
// RAW: scf.yield %[[INCOMING_PTR]], {{.*}} : !tt.ptr<f16>, i64, i64, i64, i64, i1, i1
// RAW: }
// RAW-NOT: tt.make_tensor_descriptor
// RAW-NOT: tt.descriptor_
// RAW: tt.splat %[[SELECTED]]#0 : !tt.ptr<f16> -> tensor<8x16x!tt.ptr<f16>>
// RAW-NOT: tt.make_tensor_descriptor
// RAW-NOT: tt.descriptor_
// RAW: %[[V2:.*]] = tt.load {{.*}} : tensor<8x16x!tt.ptr<f16>>
// RAW-NEXT: tt.return %[[V0]], %[[V1]], %[[V2]] :
// RAW-NOT: tt.make_tensor_descriptor
// RAW-NOT: tt.descriptor_
// RAW-NOT: tt.load

// TWICE-LABEL: @split_multiple_direct_users_nan
// TWICE: tt.make_tensor_descriptor
// TWICE-NEXT: tt.descriptor_load
// TWICE-NEXT: tt.descriptor_store
// TWICE-NEXT: tt.descriptor_load
// TWICE-NOT: tt.make_tensor_descriptor
// TWICE-NOT: tt.descriptor_load
// TWICE: tt.load
// TWICE-NOT: tt.make_tensor_descriptor
// TWICE-NOT: tt.descriptor_load

// -----

// COM: Loop-carried sharing: the maker seeds an scf.for iter_arg whose yield is
// COM: an untraceable host descriptor, and is also loaded directly in the loop
// COM: body. The direct load must use a native clone defined before the loop;
// COM: the carried descriptor must become pointer sidecars with a tt.load.
module {
  tt.func public @split_loop_carried(%incoming: !tt.tensordesc<8x16xf16>, %base: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %height: i32, %width: i32, %pitch: i64, %n: i32) -> tensor<8x16xf16> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<8x16xf16>
    %desc = tt.make_tensor_descriptor %base, [%height, %width], [%pitch, %c1_i64] : <f16>, <8x16xf16>
    %res:2 = scf.for %iv = %c0_i32 to %n step %c1_i32 iter_args(%acc = %cst, %cur = %desc) -> (tensor<8x16xf16>, !tt.tensordesc<8x16xf16>) : i32 {
      %direct = tt.descriptor_load %desc[%iv, %c0_i32] : !tt.tensordesc<8x16xf16> -> tensor<8x16xf16>
      %carried = tt.descriptor_load %cur[%iv, %c0_i32] : !tt.tensordesc<8x16xf16> -> tensor<8x16xf16>
      %sum = arith.addf %direct, %carried : tensor<8x16xf16>
      %next = arith.addf %acc, %sum : tensor<8x16xf16>
      scf.yield %next, %incoming : tensor<8x16xf16>, !tt.tensordesc<8x16xf16>
    }
    tt.return %res#0 : tensor<8x16xf16>
  }
}

// CHECK-LABEL: @split_loop_carried
// CHECK-SAME: (%[[INCOMING_PTR:[^:]+]]: !tt.ptr<f16>, {{.*}}%[[BASE:[^:]+]]: !tt.ptr<f16> {tt.divisibility = 16 : i32}
// CHECK: %[[NATIVE:.*]] = tt.make_tensor_descriptor %[[BASE]], {{.*}} : <f16>, <8x16xf16>
// CHECK-NOT: tt.make_tensor_descriptor
// CHECK: scf.for {{.*}} iter_args({{.*}}, %{{.*}} = %[[BASE]], {{.*}}) -> (tensor<8x16xf16>, !tt.ptr<f16>, i64, i64, i64, i64, i1)
// CHECK: tt.descriptor_load %[[NATIVE]][
// CHECK-NOT: tt.descriptor_load
// CHECK: tt.load {{.*}} : tensor<8x16x!tt.ptr<f16>>
// CHECK-NOT: tt.descriptor_load
// CHECK: scf.yield {{.*}}, %[[INCOMING_PTR]], {{.*}} : tensor<8x16xf16>, !tt.ptr<f16>, i64, i64, i64, i64, i1
// CHECK-NOT: tt.make_tensor_descriptor
// CHECK-NOT: tt.descriptor_

// RAW-LABEL: @split_loop_carried
// RAW-COUNT-1: tt.make_tensor_descriptor
// RAW-NOT: tt.make_tensor_descriptor
// RAW: scf.for
// RAW: tt.descriptor_load
// RAW-NOT: tt.descriptor_load
// RAW: tt.load {{.*}} : tensor<8x16x!tt.ptr<f16>>
// RAW-NOT: tt.load
// RAW-NOT: tt.{{make_tensor_descriptor|descriptor_}}

// TWICE-LABEL: @split_loop_carried
// TWICE: tt.make_tensor_descriptor
// TWICE-NOT: tt.make_tensor_descriptor
// TWICE: scf.for
// TWICE: tt.descriptor_load
// TWICE-NOT: tt.descriptor_load
// TWICE: tt.load
// TWICE-NOT: tt.{{make_tensor_descriptor|descriptor_}}

// -----

// COM: Region-terminator merges (#8256 review). A region op is converted as a
// COM: unit with its terminators, but its own result trace collapses to empty
// COM: as soon as one incoming edge is untraceable -- so the local maker was
// COM: only ever seen at the yield, stayed native, and left the yield at the
// COM: pre-expansion arity. In all four cases below %incoming sits immediately
// COM: before a !tt.ptr argument, which fails the shape/stride layout check in
// COM: synthesizeDescriptorsFromFuncArgs and keeps it untraceable; giving it an
// COM: i32 successor would synthesize a maker and dissolve the case under test.

// COM: `scf.if` merging the host descriptor with a directly-loaded local maker.
module {
  tt.func public @split_if_yield_local(%incoming: !tt.tensordesc<8x16xf16>, %base: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %height: i32, %width: i32, %pitch: i64, %choose_incoming: i1, %row: i32, %col: i32) -> (tensor<8x16xf16>, tensor<8x16xf16>) {
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %base, [%height, %width], [%pitch, %c1_i64] : <f16>, <8x16xf16>
    %direct = tt.descriptor_load %desc[%row, %col] : !tt.tensordesc<8x16xf16> -> tensor<8x16xf16>
    %merged = scf.if %choose_incoming -> (!tt.tensordesc<8x16xf16>) {
      scf.yield %incoming : !tt.tensordesc<8x16xf16>
    } else {
      scf.yield %desc : !tt.tensordesc<8x16xf16>
    }
    %v = tt.descriptor_load %merged[%row, %col] : !tt.tensordesc<8x16xf16> -> tensor<8x16xf16>
    tt.return %direct, %v : tensor<8x16xf16>, tensor<8x16xf16>
  }
}

// CHECK-LABEL: @split_if_yield_local
// CHECK-SAME: %[[INCOMING_PTR:[^:]*]]: !tt.ptr<f16>, %{{[^:]*}}: i64, %{{[^:]*}}: i64, %{{[^:]*}}: i64, %{{[^:]*}}: i64, %{{[^:]*}}: i1, %{{[^:]*}}: i1,
// CHECK-SAME: %[[BASE:[^:]*]]: !tt.ptr<f16> {tt.divisibility = 16 : i32}
// CHECK-SAME: %[[COND:[^:]*]]: i1, %[[ROW:[^:]*]]: i32, %[[COL:[^:]*]]: i32)
// CHECK: %[[DESC:.*]] = tt.make_tensor_descriptor %[[BASE]],
// CHECK-NEXT: %[[DIRECT:.*]] = tt.descriptor_load %[[DESC]][%[[ROW]], %[[COL]]] : !tt.tensordesc<8x16xf16> -> tensor<8x16xf16>
// CHECK-NOT: tt.{{make_tensor_descriptor|descriptor_}}
// COM: The widened 7-result scf.if canonicalizes away; the merge survives as a
// COM: per-component select, so the pointer the fallback load uses is chosen
// COM: between the host descriptor's and the evicted maker's own base.
// CHECK: arith.select %[[COND]], %[[INCOMING_PTR]], %[[BASE]] : !tt.ptr<f16>
// CHECK: %[[V:.*]] = tt.load
// CHECK-NEXT: tt.return %[[DIRECT]], %[[V]] : tensor<8x16xf16>, tensor<8x16xf16>
// CHECK-NOT: tt.{{make_tensor_descriptor|descriptor_}}

// RAW-LABEL: @split_if_yield_local
// RAW-SAME: %[[INCOMING_PTR:[^:]*]]: !tt.ptr<f16>
// RAW-SAME: %[[BASE:[^:]*]]: !tt.ptr<f16> {tt.divisibility = 16 : i32}
// RAW: %[[DESC:.*]] = tt.make_tensor_descriptor %[[BASE]],
// RAW-NEXT: %[[DIRECT:.*]] = tt.descriptor_load %[[DESC]][
// RAW-NOT: tt.{{make_tensor_descriptor|descriptor_}}
// RAW: %[[SEL:.*]]:7 = scf.if %{{.*}} -> (!tt.ptr<f16>, i64, i64, i64, i64, i1, i1) {
// RAW-NEXT: scf.yield %[[INCOMING_PTR]], {{.*}} : !tt.ptr<f16>, i64, i64, i64, i64, i1, i1
// RAW-NEXT: } else {
// RAW-NEXT: scf.yield %[[BASE]], {{.*}} : !tt.ptr<f16>, i64, i64, i64, i64, i1, i1
// RAW: tt.splat %[[SEL]]#0 : !tt.ptr<f16> -> tensor<8x16x!tt.ptr<f16>>
// RAW: tt.load
// RAW-NOT: tt.{{make_tensor_descriptor|descriptor_|load }}

// TWICE-LABEL: @split_if_yield_local
// TWICE: tt.make_tensor_descriptor
// TWICE-NEXT: tt.descriptor_load
// TWICE-NOT: tt.{{make_tensor_descriptor|descriptor_}}
// TWICE: tt.load
// TWICE-NOT: tt.{{make_tensor_descriptor|descriptor_}}

// -----

// COM: `scf.for` twin: the host descriptor seeds the iter_arg and the local
// COM: maker is yielded. The loop result is loaded, so the loop stays live
// COM: through the greedy pre-rewrite that runs before candidate collection.
module {
  tt.func public @split_for_yield_local(%incoming: !tt.tensordesc<8x16xf16>, %base: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %height: i32, %width: i32, %pitch: i64, %n: i32, %row: i32, %col: i32) -> (tensor<8x16xf16>, tensor<8x16xf16>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %base, [%height, %width], [%pitch, %c1_i64] : <f16>, <8x16xf16>
    %direct = tt.descriptor_load %desc[%row, %col] : !tt.tensordesc<8x16xf16> -> tensor<8x16xf16>
    %carried = scf.for %iv = %c0_i32 to %n step %c1_i32 iter_args(%cur = %incoming) -> (!tt.tensordesc<8x16xf16>) : i32 {
      scf.yield %desc : !tt.tensordesc<8x16xf16>
    }
    %v = tt.descriptor_load %carried[%row, %col] : !tt.tensordesc<8x16xf16> -> tensor<8x16xf16>
    tt.return %direct, %v : tensor<8x16xf16>, tensor<8x16xf16>
  }
}

// CHECK-LABEL: @split_for_yield_local
// CHECK-SAME: %[[INCOMING_PTR:[^:]*]]: !tt.ptr<f16>
// CHECK-SAME: %[[BASE:[^:]*]]: !tt.ptr<f16> {tt.divisibility = 16 : i32}
// CHECK: %[[DESC:.*]] = tt.make_tensor_descriptor %[[BASE]],
// CHECK-NEXT: %[[DIRECT:.*]] = tt.descriptor_load %[[DESC]][
// CHECK-NOT: tt.{{make_tensor_descriptor|descriptor_}}
// COM: Canonicalize drops the loop-carried components the body never reads, so
// COM: only RAW pins the full 1 -> 7 expansion. What matters here is that the
// COM: iter_arg is seeded from the host pointer and the yield hands back the
// COM: evicted maker's components -- both sides on the pointer path.
// CHECK: scf.for {{.*}} iter_args(%{{[^ ]+}} = %[[INCOMING_PTR]],
// CHECK: scf.yield %[[BASE]],
// CHECK: %[[V:.*]] = tt.load
// CHECK-NEXT: tt.return %[[DIRECT]], %[[V]] : tensor<8x16xf16>, tensor<8x16xf16>
// CHECK-NOT: tt.{{make_tensor_descriptor|descriptor_}}

// RAW-LABEL: @split_for_yield_local
// RAW-SAME: %[[INCOMING_PTR:[^:]*]]: !tt.ptr<f16>
// RAW-SAME: %[[BASE:[^:]*]]: !tt.ptr<f16> {tt.divisibility = 16 : i32}
// RAW: %[[DESC:.*]] = tt.make_tensor_descriptor %[[BASE]],
// RAW-NEXT: %[[DIRECT:.*]] = tt.descriptor_load %[[DESC]][
// RAW-NOT: tt.{{make_tensor_descriptor|descriptor_}}
// RAW: scf.for {{.*}} iter_args(%{{[^ ]+}} = %[[INCOMING_PTR]], {{.*}}) -> (!tt.ptr<f16>, i64, i64, i64, i64, i1, i1)
// RAW-NEXT: scf.yield %[[BASE]], {{.*}} : !tt.ptr<f16>, i64, i64, i64, i64, i1, i1
// RAW: tt.load
// RAW-NOT: tt.{{make_tensor_descriptor|descriptor_|load }}

// TWICE-LABEL: @split_for_yield_local
// TWICE: tt.make_tensor_descriptor
// TWICE-NEXT: tt.descriptor_load
// TWICE-NOT: tt.{{make_tensor_descriptor|descriptor_}}
// TWICE: tt.load
// TWICE-NOT: tt.{{make_tensor_descriptor|descriptor_}}

// -----

// COM: `scf.while` twin. `scf.condition` must forward the before-region
// COM: argument: a condition yielding %desc instead would place it in the
// COM: while's own operand/result group beside the untraceable init and evict
// COM: it even before the fix, so the case would prove nothing.
module {
  tt.func public @split_while_yield_local(%incoming: !tt.tensordesc<8x16xf16>, %base: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %height: i32, %width: i32, %pitch: i64, %keep_going: i1, %row: i32, %col: i32) -> (tensor<8x16xf16>, tensor<8x16xf16>) {
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %base, [%height, %width], [%pitch, %c1_i64] : <f16>, <8x16xf16>
    %direct = tt.descriptor_load %desc[%row, %col] : !tt.tensordesc<8x16xf16> -> tensor<8x16xf16>
    %r = scf.while (%x = %incoming) : (!tt.tensordesc<8x16xf16>) -> !tt.tensordesc<8x16xf16> {
      scf.condition(%keep_going) %x : !tt.tensordesc<8x16xf16>
    } do {
    ^bb0(%y: !tt.tensordesc<8x16xf16>):
      scf.yield %desc : !tt.tensordesc<8x16xf16>
    }
    %v = tt.descriptor_load %r[%row, %col] : !tt.tensordesc<8x16xf16> -> tensor<8x16xf16>
    tt.return %direct, %v : tensor<8x16xf16>, tensor<8x16xf16>
  }
}

// CHECK-LABEL: @split_while_yield_local
// CHECK-SAME: %[[INCOMING_PTR:[^:]*]]: !tt.ptr<f16>
// CHECK-SAME: %[[BASE:[^:]*]]: !tt.ptr<f16> {tt.divisibility = 16 : i32}
// CHECK: %[[DESC:.*]] = tt.make_tensor_descriptor %[[BASE]],
// CHECK-NEXT: %[[DIRECT:.*]] = tt.descriptor_load %[[DESC]][
// CHECK-NOT: tt.{{make_tensor_descriptor|descriptor_}}
// CHECK: scf.while (%{{[^ ]+}} = %[[INCOMING_PTR]],
// CHECK: scf.yield %[[BASE]],
// CHECK: %[[V:.*]] = tt.load
// CHECK-NEXT: tt.return %[[DIRECT]], %[[V]] : tensor<8x16xf16>, tensor<8x16xf16>
// CHECK-NOT: tt.{{make_tensor_descriptor|descriptor_}}

// RAW-LABEL: @split_while_yield_local
// RAW-SAME: %[[INCOMING_PTR:[^:]*]]: !tt.ptr<f16>
// RAW-SAME: %[[BASE:[^:]*]]: !tt.ptr<f16> {tt.divisibility = 16 : i32}
// RAW: %[[DESC:.*]] = tt.make_tensor_descriptor %[[BASE]],
// RAW-NEXT: %[[DIRECT:.*]] = tt.descriptor_load %[[DESC]][
// RAW-NOT: tt.{{make_tensor_descriptor|descriptor_}}
// COM: Both of the while's regions are reached: the condition forwards the
// COM: before-region arguments and the after-yield hands back the evicted
// COM: maker, each at the full 7-component width.
// RAW: scf.while (%{{[^ ]+}} = %[[INCOMING_PTR]], {{.*}}) : (!tt.ptr<f16>, i64, i64, i64, i64, i1, i1) -> (!tt.ptr<f16>, i64, i64, i64, i64, i1, i1)
// RAW-NEXT: scf.condition(%{{[^ ]+}}) %{{.*}} : !tt.ptr<f16>, i64, i64, i64, i64, i1, i1
// RAW: scf.yield %[[BASE]], {{.*}} : !tt.ptr<f16>, i64, i64, i64, i64, i1, i1
// RAW: tt.load
// RAW-NOT: tt.{{make_tensor_descriptor|descriptor_|load }}

// TWICE-LABEL: @split_while_yield_local
// TWICE: tt.make_tensor_descriptor
// TWICE-NEXT: tt.descriptor_load
// TWICE-NOT: tt.{{make_tensor_descriptor|descriptor_}}
// TWICE: tt.load
// TWICE-NOT: tt.{{make_tensor_descriptor|descriptor_}}

// -----

// COM: The opposite direction: here the `scf.while` itself stays legal -- its
// COM: init traces to %d0 and its result traces through `scf.condition` to %d1,
// COM: both candidates -- while the after-yield's untraceable operand makes
// COM: only the yield illegal, so the yield expands against an unexpanded
// COM: before block. Loading the while result is what makes %d1 a candidate;
// COM: without that load the existing closure already evicts %d0.
module {
  tt.func public @while_after_yield_untraceable(%incoming: !tt.tensordesc<8x16xf16>, %base: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %other: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %height: i32, %width: i32, %pitch: i64, %keep_going: i1, %row: i32, %col: i32) -> (tensor<8x16xf16>, tensor<8x16xf16>) {
    %c1_i64 = arith.constant 1 : i64
    %d0 = tt.make_tensor_descriptor %base, [%height, %width], [%pitch, %c1_i64] : <f16>, <8x16xf16>
    %direct = tt.descriptor_load %d0[%row, %col] : !tt.tensordesc<8x16xf16> -> tensor<8x16xf16>
    %r = scf.while (%x = %d0) : (!tt.tensordesc<8x16xf16>) -> !tt.tensordesc<8x16xf16> {
      %d1 = tt.make_tensor_descriptor %other, [%height, %width], [%pitch, %c1_i64] : <f16>, <8x16xf16>
      scf.condition(%keep_going) %d1 : !tt.tensordesc<8x16xf16>
    } do {
    ^bb0(%y: !tt.tensordesc<8x16xf16>):
      scf.yield %incoming : !tt.tensordesc<8x16xf16>
    }
    %v = tt.descriptor_load %r[%row, %col] : !tt.tensordesc<8x16xf16> -> tensor<8x16xf16>
    tt.return %direct, %v : tensor<8x16xf16>, tensor<8x16xf16>
  }
}

// CHECK-LABEL: @while_after_yield_untraceable
// CHECK-SAME: %[[BASE:[^:]*]]: !tt.ptr<f16> {tt.divisibility = 16 : i32}
// COM: %d0's clone keeps the direct load native. %d1 lived only in the
// COM: condition, so it has no direct use to split and leaves entirely.
// CHECK: %[[DESC:.*]] = tt.make_tensor_descriptor %[[BASE]],
// CHECK-NEXT: %[[DIRECT:.*]] = tt.descriptor_load %[[DESC]][
// CHECK-NOT: tt.{{make_tensor_descriptor|descriptor_}}
// COM: The while itself is not asserted here: once both makers' components are
// COM: CSEd the loop is dead, so it survives `--canonicalize --cse` but not
// COM: `--cse --canonicalize --cse`. RAW pins its expansion instead.
// CHECK: %[[V:.*]] = tt.load
// CHECK-NEXT: tt.return %[[DIRECT]], %[[V]] : tensor<8x16xf16>, tensor<8x16xf16>
// CHECK-NOT: tt.{{make_tensor_descriptor|descriptor_}}

// RAW-LABEL: @while_after_yield_untraceable
// RAW-SAME: %[[INCOMING_PTR:[^:]*]]: !tt.ptr<f16>
// RAW-SAME: %[[BASE:[^:]*]]: !tt.ptr<f16> {tt.divisibility = 16 : i32}
// RAW-SAME: %[[OTHER:[^:]*]]: !tt.ptr<f16> {tt.divisibility = 16 : i32}
// RAW: %[[DESC:.*]] = tt.make_tensor_descriptor %[[BASE]],
// RAW-NEXT: %[[DIRECT:.*]] = tt.descriptor_load %[[DESC]][
// RAW-NOT: tt.{{make_tensor_descriptor|descriptor_}}
// COM: The whole while is now expanded: init from %d0's components, the
// COM: condition from %d1's, and the after-yield from the host's. Before the
// COM: fix only the after-yield was converted, against a 1-input before block.
// RAW: scf.while (%{{[^ ]+}} = %[[BASE]], {{.*}}) : (!tt.ptr<f16>, i64, i64, i64, i64, i1, i1) -> (!tt.ptr<f16>, i64, i64, i64, i64, i1, i1)
// RAW: scf.condition(%{{[^ ]+}}) %[[OTHER]], {{.*}} : !tt.ptr<f16>, i64, i64, i64, i64, i1, i1
// RAW: scf.yield %[[INCOMING_PTR]], {{.*}} : !tt.ptr<f16>, i64, i64, i64, i64, i1, i1
// RAW: tt.load
// RAW-NOT: tt.{{make_tensor_descriptor|descriptor_|load }}

// TWICE-LABEL: @while_after_yield_untraceable
// TWICE: tt.make_tensor_descriptor
// TWICE-NEXT: tt.descriptor_load
// TWICE-NOT: tt.{{make_tensor_descriptor|descriptor_}}
// TWICE: tt.load
// TWICE-NOT: tt.{{make_tensor_descriptor|descriptor_}}

// -----

// COM: Function-boundary merges (#8256 review). A FuncOp whose signature holds
// COM: a descriptor is unconditionally illegal, so its arguments and results
// COM: always expand 1 -> 7 -- but a `tt.call` operand or `tt.return` operand
// COM: tracing to a kept maker stayed legal, so the call/return was left at the
// COM: old arity. Evict such makers like gather/scatter; the split then keeps
// COM: the direct access on a native clone.

// COM: Descriptor crossing into a callee. The callee is private, so the
// COM: host-descriptor synthesis pre-pass (public only) never sees it.
module {
  tt.func private @desc_operand_callee(%d: !tt.tensordesc<8x16xf16>, %row: i32, %col: i32) -> tensor<8x16xf16> attributes {noinline = true} {
    %v = tt.descriptor_load %d[%row, %col] : !tt.tensordesc<8x16xf16> -> tensor<8x16xf16>
    tt.return %v : tensor<8x16xf16>
  }

  tt.func public @split_call_operand(%base: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %height: i32, %width: i32, %pitch: i64, %row: i32, %col: i32) -> (tensor<8x16xf16>, tensor<8x16xf16>) {
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %base, [%height, %width], [%pitch, %c1_i64] : <f16>, <8x16xf16>
    %direct = tt.descriptor_load %desc[%row, %col] : !tt.tensordesc<8x16xf16> -> tensor<8x16xf16>
    %called = tt.call @desc_operand_callee(%desc, %row, %col) : (!tt.tensordesc<8x16xf16>, i32, i32) -> tensor<8x16xf16>
    tt.return %direct, %called : tensor<8x16xf16>, tensor<8x16xf16>
  }
}

// CHECK-LABEL: tt.func private @desc_operand_callee
// CHECK-SAME: %{{[^:]*}}: !tt.ptr<f16>, %{{[^:]*}}: i64, %{{[^:]*}}: i64, %{{[^:]*}}: i64, %{{[^:]*}}: i64, %{{[^:]*}}: i1, %{{[^:]*}}: i1,
// CHECK-NOT: tt.descriptor_load
// CHECK: tt.load
// CHECK-LABEL: @split_call_operand
// CHECK-SAME: %[[BASE:[^:]*]]: !tt.ptr<f16> {tt.divisibility = 16 : i32}
// CHECK-SAME: %[[ROW:[^:]*]]: i32, %[[COL:[^:]*]]: i32)
// CHECK: %[[DESC:.*]] = tt.make_tensor_descriptor %[[BASE]],
// CHECK-NEXT: %[[DIRECT:.*]] = tt.descriptor_load %[[DESC]][%[[ROW]], %[[COL]]] : !tt.tensordesc<8x16xf16> -> tensor<8x16xf16>
// CHECK-NOT: tt.{{make_tensor_descriptor|descriptor_}}
// CHECK: %[[CALLED:.*]] = tt.call @desc_operand_callee(%[[BASE]], {{.*}}) : (!tt.ptr<f16>, i64, i64, i64, i64, i1, i1, i32, i32) -> tensor<8x16xf16>
// CHECK-NEXT: tt.return %[[DIRECT]], %[[CALLED]] : tensor<8x16xf16>, tensor<8x16xf16>
// CHECK-NOT: tt.{{make_tensor_descriptor|descriptor_}}

// COM: RAW bounds the clone count before cleanup: exactly one maker and one
// COM: native load in the caller, and none in the callee.
// RAW-LABEL: tt.func private @desc_operand_callee
// RAW-NOT: tt.{{make_tensor_descriptor|descriptor_}}
// RAW-LABEL: @split_call_operand
// RAW-SAME: %[[BASE:[^:]*]]: !tt.ptr<f16> {tt.divisibility = 16 : i32}
// RAW: %[[DESC:.*]] = tt.make_tensor_descriptor %[[BASE]],
// RAW-NEXT: %[[DIRECT:.*]] = tt.descriptor_load %[[DESC]][
// RAW-NOT: tt.{{make_tensor_descriptor|descriptor_}}
// RAW: tt.call @desc_operand_callee(%[[BASE]], {{.*}}) : (!tt.ptr<f16>, i64, i64, i64, i64, i1, i1, i32, i32) -> tensor<8x16xf16>
// RAW-NOT: tt.{{make_tensor_descriptor|descriptor_}}

// TWICE-LABEL: tt.func private @desc_operand_callee
// TWICE-NOT: tt.{{make_tensor_descriptor|descriptor_}}
// TWICE-LABEL: @split_call_operand
// TWICE: tt.make_tensor_descriptor
// TWICE-NEXT: tt.descriptor_load
// TWICE-NOT: tt.{{make_tensor_descriptor|descriptor_}}
// TWICE: tt.call @desc_operand_callee
// TWICE-NOT: tt.{{make_tensor_descriptor|descriptor_}}

// -----

// COM: Descriptor returned out of a helper -- @f without its untraceable
// COM: argument, so nothing but the return operand can drive eviction.
module {
  tt.func private @return_desc_helper(%base: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %height: i32, %width: i32, %pitch: i64, %row: i32, %col: i32) -> (tensor<8x16xf16>, !tt.tensordesc<8x16xf16>) attributes {noinline = true} {
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %base, [%height, %width], [%pitch, %c1_i64] : <f16>, <8x16xf16>
    %direct = tt.descriptor_load %desc[%row, %col] : !tt.tensordesc<8x16xf16> -> tensor<8x16xf16>
    tt.return %direct, %desc : tensor<8x16xf16>, !tt.tensordesc<8x16xf16>
  }

  tt.func public @split_return_operand(%base: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %height: i32, %width: i32, %pitch: i64, %row: i32, %col: i32) -> (tensor<8x16xf16>, tensor<8x16xf16>) {
    %r:2 = tt.call @return_desc_helper(%base, %height, %width, %pitch, %row, %col) : (!tt.ptr<f16>, i32, i32, i64, i32, i32) -> (tensor<8x16xf16>, !tt.tensordesc<8x16xf16>)
    %v = tt.descriptor_load %r#1[%row, %col] : !tt.tensordesc<8x16xf16> -> tensor<8x16xf16>
    tt.return %r#0, %v : tensor<8x16xf16>, tensor<8x16xf16>
  }
}

// CHECK-LABEL: tt.func private @return_desc_helper
// CHECK-SAME: -> (tensor<8x16xf16>, !tt.ptr<f16>, i64, i64, i64, i64, i1, i1)
// CHECK: %[[DESC:.*]] = tt.make_tensor_descriptor %[[BASE:[^,]*]],
// CHECK-NEXT: %[[DIRECT:.*]] = tt.descriptor_load %[[DESC]][
// COM: The maker is evicted for the return, but its direct load rides the
// COM: native clone -- so the helper returns the components while still
// COM: performing a descriptor load.
// CHECK-NEXT: tt.return %[[DIRECT]], %[[BASE]], {{.*}} : tensor<8x16xf16>, !tt.ptr<f16>, i64, i64, i64, i64, i1, i1
// CHECK-NOT: tt.{{make_tensor_descriptor|descriptor_}}
// CHECK-LABEL: @split_return_operand
// CHECK: %[[R:.*]]:8 = tt.call @return_desc_helper({{.*}}) : (!tt.ptr<f16>, i32, i32, i64, i32, i32) -> (tensor<8x16xf16>, !tt.ptr<f16>, i64, i64, i64, i64, i1, i1)
// CHECK-NOT: tt.{{make_tensor_descriptor|descriptor_}}
// CHECK: tt.splat %[[R]]#1 : !tt.ptr<f16> -> tensor<8x16x!tt.ptr<f16>>
// CHECK: %[[V:.*]] = tt.load
// CHECK-NEXT: tt.return %[[R]]#0, %[[V]] : tensor<8x16xf16>, tensor<8x16xf16>
// CHECK-NOT: tt.{{make_tensor_descriptor|descriptor_}}

// RAW-LABEL: tt.func private @return_desc_helper
// RAW: %[[DESC:.*]] = tt.make_tensor_descriptor %[[BASE:[^,]*]],
// RAW-NEXT: %[[DIRECT:.*]] = tt.descriptor_load %[[DESC]][
// RAW-NEXT: tt.return %[[DIRECT]], %[[BASE]], {{.*}} : tensor<8x16xf16>, !tt.ptr<f16>, i64, i64, i64, i64, i1, i1
// RAW-NOT: tt.{{make_tensor_descriptor|descriptor_}}
// RAW-LABEL: @split_return_operand
// RAW: %[[R:.*]]:8 = tt.call @return_desc_helper(
// RAW-NOT: tt.{{make_tensor_descriptor|descriptor_}}
// RAW: tt.splat %[[R]]#1 : !tt.ptr<f16> -> tensor<8x16x!tt.ptr<f16>>
// RAW-NOT: tt.{{make_tensor_descriptor|descriptor_}}

// TWICE-LABEL: tt.func private @return_desc_helper
// TWICE: tt.make_tensor_descriptor
// TWICE-NEXT: tt.descriptor_load
// TWICE-NOT: tt.{{make_tensor_descriptor|descriptor_}}
// TWICE-LABEL: @split_return_operand
// TWICE-NOT: tt.{{make_tensor_descriptor|descriptor_}}
// TWICE: tt.load
