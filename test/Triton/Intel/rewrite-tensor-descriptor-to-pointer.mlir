// RUN: triton-opt %s --triton-intel-rewrite-tensor-descriptor-to-pointer --canonicalize --cse --split-input-file | FileCheck %s

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

// COM: Descriptor has two users: gather (-> tt.load) and store (-> tt.store).
// COM: Both are lowered to the pointer fallback path since the gather makes
// COM: the descriptor "unhandled".
// CHECK-LABEL: @multi_users
// CHECK-SAME: %[[ARG0:[^:]*]]: !tt.ptr<f32>
// CHECK-SAME: %[[ARG1:[^:]*]]: i32
// CHECK-SAME: %[[ARG2:[^:]*]]: i32
// CHECK-SAME: %[[ARG3:[^:]*]]: tensor<1x128xf32>

// COM: Gather path: lowered to tt.load with pointer arithmetic.
// CHECK: tt.load {{.*}} : tensor<32x128x!tt.ptr<f32>>

// COM: Store path: lowered to tt.store with pointer arithmetic.
// CHECK: tt.store {{.*}}, %[[ARG3]], {{.*}} : tensor<1x128x!tt.ptr<f32>>

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
// COM: Measured at base commit 495054198: hard error
// COM:   'scf.if' op along control flow edge from Operation scf.yield to Operation
// COM:   scf.if: region branch point has 7 operands, but region successor needs 1
// COM:   inputs
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
// COM: Measured at base commit 495054198: ASSERTION CRASH, exit 141
// COM:   triton-opt: mlir/include/mlir/IR/OpDefinition.h:920:
// COM:   mlir::Block *mlir::OpTrait::SingleBlock<mlir::scf::ForOp>::getBody(unsigned)
// COM:   Assertion `!region.empty() && "unexpected empty region"' failed.
// COM: Because the process aborts, this chunk kills the whole -split-input-file
// COM: run at base -- other chunks in this file cannot be observed until it is
// COM: fixed. @for_load below is the control that isolates the trigger.
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
// COM: descriptor iter-arg. It is green at base and must stay green.
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
// COM: Measured at base commit 495054198: hard error
// COM:   'tt.call' op incorrect number of results for callee
// COM:
// COM: This is the ONE case of the three where the widened arity really is
// COM: observable -- a `tt.call` has no region for `--canonicalize` to collapse, so
// COM: the 7-result call site survives verbatim and is asserted below, mirroring the
// COM: @callee/@caller pair earlier in this file.
// COM:
// COM: It is also the only case where the padding flag stays a genuine runtime value:
// COM: it arrives as a call result, so nothing can fold the `arith.select` that
// COM: chooses between the NaN and zero fills. That makes this the reference for the
// COM: canonical select direction -- flag true selects NaN.
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
// COM: Measured at base commit 495054198: WRONG-BUT-GREEN. The pass is a no-op --
// COM: `tt.make_tensor_descriptor` (x2) and `tt.descriptor_load` all survive, so
// COM: the load stays on the descriptor-native route and the silent PAD_ZERO
// COM: degradation happens downstream.
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
// COM: Measured at base commit 495054198: WRONG-BUT-GREEN -- pass is a no-op, both
// COM: `tt.make_tensor_descriptor` and the `tt.descriptor_load` survive.
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
// COM: Measured at base commit 495054198: GREEN, and must stay green.
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
// COM: Measured at base commit 495054198: WRONG-BUT-GREEN -- pass is a no-op, both
// COM: `tt.make_tensor_descriptor` and the in-loop `tt.descriptor_load` survive.
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

// COM: THIS TEST EXISTS TO DOCUMENT A COST, not to celebrate a behaviour.
// COM:
// COM: One `tt.make_tensor_descriptor` (%d0, PAD_ZERO) feeds two things: the
// COM: divergent `scf.if` merge whose other arm is PAD_NAN, and -- directly -- a
// COM: second `tt.descriptor_load` whose provenance is the single candidate %d0 and
// COM: is therefore perfectly consistent. That second load would be safe on the
// COM: descriptor-native route, yet BOTH loads are expected to become `tt.load`.
// COM:
// COM: The reason is structural, not a judgement call: this pass runs the type
// COM: conversion with `buildMaterializations = false`, so there is no
// COM: descriptor<->{ptr,shape,stride,pad,round} materialization available to
// COM: bridge the two worlds. A single `tt.make_tensor_descriptor` is therefore
// COM: either expanded for all of its users or for none of them -- legality is
// COM: per-producer and cannot be mixed within one producer's use set. Evicting the
// COM: shared producer costs the consistent load its 2D block I/O fast path.
// COM:
// COM: IF SOMEONE LATER NARROWS THE EVICTION (e.g. by cloning the shared producer
// COM: so the consistent load keeps its own candidate, or by enabling
// COM: materializations), THIS IS THE TEST THAT SHOULD CHANGE: the second load
// COM: would then be expected to stay a `tt.descriptor_load`. Do not "fix" it by
// COM: relaxing the CHECKs; change it deliberately and say why.
// COM:
// COM: Measured at base commit 495054198: WRONG-BUT-GREEN -- pass is a no-op, the
// COM: shared producer and both `tt.descriptor_load`s survive.
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
// CHECK: %[[OTHER:.*]] = arith.select %arg1, %[[ZERO]], %[[NAN]] : tensor<128x128xf32>
// CHECK-NOT: tt.make_tensor_descriptor
// CHECK-NOT: tt.descriptor_load
// CHECK: %[[MASK:.*]] = arith.andi %{{.*}}, %{{.*}} : tensor<128x128xi1>
// COM: The two loads are distinguished by their `other` operand, and that is the
// COM: assertion that matters: the divergent load gets the runtime select, while the
// COM: load whose provenance is only the shared PAD_ZERO producer gets the bare zero
// COM: splat. So the eviction costs the second load its fast path (the documented
// COM: cost above) WITHOUT corrupting its fill -- it is still exactly PAD_ZERO, not
// COM: the divergent select.
// CHECK: %[[V0:.*]] = tt.load %{{.*}}, %[[MASK]], %[[OTHER]] : tensor<128x128x!tt.ptr<f32>>
// CHECK: %[[V1:.*]] = tt.load %{{.*}}, %[[MASK]], %[[ZERO]] : tensor<128x128x!tt.ptr<f32>>
// CHECK: tt.return %[[V0]], %[[V1]] :

// -----

// COM: Eviction must be closed over every op that shares descriptors (82d7c7c16).
// COM: Two selects share the PAD_ZERO producer %dA: %s1 merges it with the PAD_NAN
// COM: %dB (divergent, so %s1's load is evicted), %s2 merges it with the PAD_ZERO
// COM: %dC (consistent on its own). Evicting only %s1's trace leaves %s2 mixing an
// COM: evicted %dA with a kept %dC, and with `buildMaterializations = false` that
// COM: leaks a `builtin.unrealized_conversion_cast`. Closing the eviction over %s2
// COM: evicts %dC too, so both loads are expanded.
// COM:
// COM: Measured on a pre-branch binary (no #8102 work at all): the old #8102
// COM: behaviour -- all three `tt.make_tensor_descriptor`s, both selects and both
// COM: `tt.descriptor_load`s survive. The cast leak itself comes from d7a857133 on
// COM: this branch and was not measured separately.
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
// COM: Measured on a pre-branch binary: triton-opt aborts (exit 134) on
// COM: `SingleBlock<scf::ForOp>::getBody` "unexpected empty region".
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
// COM: Measured on a pre-branch binary: triton-opt aborts (exit 134) on
// COM: `SingleBlock<scf::ForOp>::getBody` "unexpected empty region". See
// COM: @for_zero_trip_divergent_padding in
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
// COM: Measured on a pre-branch binary: the verifier fails with "'scf.while' op
// COM: along control flow edge from Operation scf.condition to Operation
// COM: scf.while: region branch point has 7 operands, but region successor needs
// COM: 1 inputs". See @while_condition_padding in
// COM: test/TritonIntelGPU/find-defining-op-loops.mlir for the TTGIR side.
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
