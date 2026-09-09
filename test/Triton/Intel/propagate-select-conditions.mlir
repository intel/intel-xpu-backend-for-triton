// RUN: triton-opt %s -split-input-file -triton-intel-propagate-select-conditions | FileCheck %s

// COM: The core rewrite. In the true arm of `select %c`, the mask `%c & %w` is
// COM: `%w`, so the narrow load coincides with the wide one already present.
// COM: `%v0` has no `other` and `%v1`'s is zero, which denote the same masked-off
// COM: lanes, so the two are interchangeable and nothing has to be mutated.
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @reuse_wider_load(%ptr: tensor<512x!tt.ptr<f16>>, %c: tensor<512xi1>, %w: tensor<512xi1>) -> tensor<512xf16> {
    %cst = arith.constant dense<0.000000e+00> : tensor<512xf16>
    %m = arith.andi %c, %w : tensor<512xi1>
    %v0 = tt.load %ptr, %w : tensor<512x!tt.ptr<f16>>
    %v1 = tt.load %ptr, %m, %cst : tensor<512x!tt.ptr<f16>>
    %r = arith.select %c, %v1, %v0 : tensor<512xi1>, tensor<512xf16>
    tt.return %r : tensor<512xf16>
  }
}
// CHECK-LABEL:   tt.func @reuse_wider_load(
// CHECK:           %[[V0:.*]] = tt.load %arg0, %arg2 :
// CHECK-NOT:       tt.load
// CHECK:           tt.return %[[V0]]

// -----

// COM: The assumption has to reach through nested selects on other conditions,
// COM: which is what an Inductor `tl.where` tree looks like. `%d` is unknown, so
// COM: both of its arms are visited with the outer facts still in hand.
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @through_nested_selects(%ptr: tensor<512x!tt.ptr<f16>>, %c: tensor<512xi1>, %w: tensor<512xi1>, %d: tensor<512xi1>, %z: tensor<512xf16>) -> tensor<512xf16> {
    %cst = arith.constant dense<0.000000e+00> : tensor<512xf16>
    %m = arith.andi %c, %w : tensor<512xi1>
    %v0 = tt.load %ptr, %w, %cst : tensor<512x!tt.ptr<f16>>
    %v1 = tt.load %ptr, %m, %cst : tensor<512x!tt.ptr<f16>>
    %inner = arith.select %d, %z, %v1 : tensor<512xi1>, tensor<512xf16>
    %r = arith.select %c, %inner, %v0 : tensor<512xi1>, tensor<512xf16>
    tt.return %r : tensor<512xf16>
  }
}
// CHECK-LABEL:   tt.func @through_nested_selects(
// CHECK:           %[[V0:.*]] = tt.load %{{.*}}, %arg2, %{{.*}} :
// CHECK-NOT:       tt.load
// CHECK:           %[[NEW:.*]] = arith.select %arg3, %arg4, %[[V0]]
// CHECK:           arith.select %arg1, %[[NEW]], %[[V0]]

// -----

// COM: In the *false* arm of `select %c` the mask `%c & %w` is false, so the load
// COM: contributes nothing but its `other`.
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @false_arm_yields_other(%ptr: tensor<512x!tt.ptr<f16>>, %c: tensor<512xi1>, %w: tensor<512xi1>, %z: tensor<512xf16>) -> tensor<512xf16> {
    %cst = arith.constant dense<0.000000e+00> : tensor<512xf16>
    %m = arith.andi %c, %w : tensor<512xi1>
    %v1 = tt.load %ptr, %m, %cst : tensor<512x!tt.ptr<f16>>
    %r = arith.select %c, %z, %v1 : tensor<512xi1>, tensor<512xf16>
    tt.return %r : tensor<512xf16>
  }
}
// CHECK-LABEL:   tt.func @false_arm_yields_other(
// CHECK-NOT:       tt.load
// CHECK:           %[[CST:.*]] = arith.constant dense<0.000000e+00>
// CHECK:           arith.select %arg1, %arg3, %[[CST]]

// -----

// COM: A mask that weakens all the way to true can reuse an *unmasked* load. That
// COM: load covers every lane, so there is nothing to reconcile -- and pinning an
// COM: `other` onto it would not even verify, since `other` requires a mask.
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @reuse_unmasked_load(%ptr: tensor<512x!tt.ptr<f16>>, %c: tensor<512xi1>) -> tensor<512xf16> {
    %cst = arith.constant dense<0.000000e+00> : tensor<512xf16>
    %true = arith.constant dense<true> : tensor<512xi1>
    %m = arith.andi %c, %true : tensor<512xi1>
    %v0 = tt.load %ptr : tensor<512x!tt.ptr<f16>>
    %v1 = tt.load %ptr, %m, %cst : tensor<512x!tt.ptr<f16>>
    %r = arith.select %c, %v1, %v0 : tensor<512xi1>, tensor<512xf16>
    tt.return %r : tensor<512xf16>
  }
}
// CHECK-LABEL:   tt.func @reuse_unmasked_load(
// CHECK:           %[[V0:.*]] = tt.load %arg0 :
// CHECK-NOT:       tt.load
// CHECK:           tt.return %[[V0]]

// -----

// COM: A load with no `other` yields zero on its masked-off lanes, not something
// COM: unspecified, so it cannot stand in for a load whose `other` is non-zero.
// COM: The pass must not "refine" the candidate by pinning an `other` onto it --
// COM: that would change what the candidate's own uses observe.
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @nonzero_other_blocks_reuse(%ptr: tensor<512x!tt.ptr<f16>>, %c: tensor<512xi1>, %w: tensor<512xi1>) -> tensor<512xf16> {
    %one = arith.constant dense<1.000000e+00> : tensor<512xf16>
    %m = arith.andi %c, %w : tensor<512xi1>
    %v0 = tt.load %ptr, %w : tensor<512x!tt.ptr<f16>>
    %v1 = tt.load %ptr, %m, %one : tensor<512x!tt.ptr<f16>>
    %r = arith.select %c, %v1, %v0 : tensor<512xi1>, tensor<512xf16>
    tt.return %r : tensor<512xf16>
  }
}
// COM: Both loads survive. `%v0` is then left observed only under the false arm,
// COM: which is exactly what the narrowing direction acts on, so its mask comes
// COM: out strengthened to `%w & !%c`.
// CHECK-LABEL:   tt.func @nonzero_other_blocks_reuse(
// CHECK:           %[[ONE:.*]] = arith.constant dense<1.000000e+00>
// CHECK:           %[[M:.*]] = arith.andi %arg1, %arg2
// CHECK:           %[[NC:.*]] = arith.xori %arg1, %{{.*}}
// CHECK:           %[[MV0:.*]] = arith.andi %arg2, %[[NC]]
// CHECK:           %[[V0:.*]] = tt.load %arg0, %[[MV0]] :
// CHECK:           %[[V1:.*]] = tt.load %arg0, %[[M]], %[[ONE]] :
// CHECK:           arith.select %arg1, %[[V1]], %[[V0]]

// -----

// COM: A weakened mask is never issued as a new load: there is no `tt.load %ptr,
// COM: %w` to reuse here, and reading under `%w` would touch addresses the
// COM: program never touched.
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @never_widens_a_load(%ptr: tensor<512x!tt.ptr<f16>>, %c: tensor<512xi1>, %w: tensor<512xi1>, %z: tensor<512xf16>) -> tensor<512xf16> {
    %cst = arith.constant dense<0.000000e+00> : tensor<512xf16>
    %m = arith.andi %c, %w : tensor<512xi1>
    %v1 = tt.load %ptr, %m, %cst : tensor<512x!tt.ptr<f16>>
    %r = arith.select %c, %v1, %z : tensor<512xi1>, tensor<512xf16>
    tt.return %r : tensor<512xf16>
  }
}
// CHECK-LABEL:   tt.func @never_widens_a_load(
// CHECK:           %[[M:.*]] = arith.andi %arg1, %arg2
// CHECK:           %[[V1:.*]] = tt.load %arg0, %[[M]], %{{.*}} :
// CHECK:           arith.select %arg1, %[[V1]], %arg3

// -----

// COM: A write between the two loads means they need not observe the same
// COM: memory, so the wider one cannot be reused.
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @write_between_blocks_reuse(%ptr: tensor<512x!tt.ptr<f16>>, %c: tensor<512xi1>, %w: tensor<512xi1>, %val: tensor<512xf16>) -> tensor<512xf16> {
    %cst = arith.constant dense<0.000000e+00> : tensor<512xf16>
    %m = arith.andi %c, %w : tensor<512xi1>
    %v0 = tt.load %ptr, %w, %cst : tensor<512x!tt.ptr<f16>>
    tt.store %ptr, %val, %w : tensor<512x!tt.ptr<f16>>
    %v1 = tt.load %ptr, %m, %cst : tensor<512x!tt.ptr<f16>>
    %r = arith.select %c, %v1, %v0 : tensor<512xi1>, tensor<512xf16>
    tt.return %r : tensor<512xf16>
  }
}
// CHECK-LABEL:   tt.func @write_between_blocks_reuse(
// CHECK:           %[[V0:.*]] = tt.load
// CHECK:           tt.store
// CHECK:           %[[V1:.*]] = tt.load
// CHECK:           arith.select %arg1, %[[V1]], %[[V0]]

// -----

// COM: Two different defined `other` values are not interchangeable on the lanes
// COM: the weakened mask excludes.
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @conflicting_other_blocks_reuse(%ptr: tensor<512x!tt.ptr<f16>>, %c: tensor<512xi1>, %w: tensor<512xi1>) -> tensor<512xf16> {
    %zero = arith.constant dense<0.000000e+00> : tensor<512xf16>
    %one = arith.constant dense<1.000000e+00> : tensor<512xf16>
    %m = arith.andi %c, %w : tensor<512xi1>
    %v0 = tt.load %ptr, %w, %one : tensor<512x!tt.ptr<f16>>
    %v1 = tt.load %ptr, %m, %zero : tensor<512x!tt.ptr<f16>>
    %r = arith.select %c, %v1, %v0 : tensor<512xi1>, tensor<512xf16>
    tt.return %r : tensor<512xf16>
  }
}
// CHECK-LABEL:   tt.func @conflicting_other_blocks_reuse(
// CHECK:           %[[M:.*]] = arith.andi %arg1, %arg2
// CHECK:           %[[NC:.*]] = arith.xori %arg1, %{{.*}}
// CHECK:           %[[MV0:.*]] = arith.andi %arg2, %[[NC]]
// CHECK:           %[[V0:.*]] = tt.load %arg0, %[[MV0]], %{{.*}} :
// CHECK:           %[[V1:.*]] = tt.load %arg0, %[[M]], %{{.*}} :
// CHECK:           arith.select %arg1, %[[V1]], %[[V0]]

// -----

// COM: A lane-mixing operation between the select and the load blocks the
// COM: propagation: `%c` holds only on the lanes that took the arm, and a
// COM: reduction reads all of them.
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @reduction_blocks_propagation(%ptr: tensor<512x!tt.ptr<f16>>, %c: tensor<512xi1>, %w: tensor<512xi1>) -> tensor<512xf16> {
    %cst = arith.constant dense<0.000000e+00> : tensor<512xf16>
    %m = arith.andi %c, %w : tensor<512xi1>
    %v0 = tt.load %ptr, %w, %cst : tensor<512x!tt.ptr<f16>>
    %v1 = tt.load %ptr, %m, %cst : tensor<512x!tt.ptr<f16>>
    %sum = "tt.reduce"(%v1) <{axis = 0 : i32}> ({
    ^bb0(%a: f16, %b: f16):
      %add = arith.addf %a, %b : f16
      tt.reduce.return %add : f16
    }) : (tensor<512xf16>) -> f16
    %bcast = tt.splat %sum : f16 -> tensor<512xf16>
    %r = arith.select %c, %bcast, %v0 : tensor<512xi1>, tensor<512xf16>
    tt.return %r : tensor<512xf16>
  }
}
// CHECK-LABEL:   tt.func @reduction_blocks_propagation(
// CHECK:           %[[M:.*]] = arith.andi %arg1, %arg2
// CHECK:           %[[V1:.*]] = tt.load %arg0, %[[M]], %{{.*}} :
// CHECK:           "tt.reduce"(%[[V1]])

// -----

// COM: `%c & %w -> %w` is a fact about *masks*. A wider `arith.andi` is a bitwise
// COM: AND, where a constant operand of 1 does not make the operation an
// COM: identity, so the same rewrite must not be applied to it. Here the load
// COM: does fold, so the arm is committed and any i32 damage would ship with it.
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @wide_andi_is_not_a_mask(%p: !tt.ptr<f32>, %c: i1, %w: i1, %x: i32) -> f32 {
    %c1 = arith.constant 1 : i32
    %c5 = arith.constant 5 : i32
    %m = arith.andi %c, %w : i1
    %v0 = tt.load %p, %w : !tt.ptr<f32>
    %v1 = tt.load %p, %m : !tt.ptr<f32>
    %k = arith.select %c, %c1, %c5 : i32
    %a = arith.andi %k, %x : i32
    %af = arith.sitofp %a : i32 to f32
    %s = arith.addf %v1, %af : f32
    %r = arith.select %c, %s, %v0 : f32
    tt.return %r : f32
  }
}
// COM: The load still folds onto `%v0`, but the `andi` survives with both of its
// COM: operands: the constant 1 is kept, not dropped.
// CHECK-LABEL:   tt.func @wide_andi_is_not_a_mask(
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : i32
// CHECK:           %[[V0:.*]] = tt.load %arg0, %arg2 :
// CHECK:           %[[A:.*]] = arith.andi %arg3, %[[C1]] : i32
// CHECK:           %[[AF:.*]] = arith.sitofp %[[A]] : i32 to f32
// CHECK:           %[[S:.*]] = arith.addf %[[V0]], %[[AF]] : f32
// CHECK:           arith.select %arg1, %[[S]], %[[V0]]

// -----

// COM: A conjunction nested more than one level deep. Under the two assumptions
// COM: `%a` and `%b`, `(%a & %b) & %w` is `%w` -- but only if the inner `%a & %b`
// COM: itself collapses first, which needs `%b` substituted by its assumed
// COM: constant. This is the shape a two-level `tl.where` over guarded loads
// COM: produces, so the deepest load has to fold too, not just the shallow one.
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @nested_conjunction(%ptr: tensor<512x!tt.ptr<f32>>, %a: tensor<512xi1>, %b: tensor<512xi1>, %w: tensor<512xi1>) -> tensor<512xf32> {
    %mab = arith.andi %a, %b : tensor<512xi1>
    %m3 = arith.andi %mab, %w : tensor<512xi1>
    %v1 = tt.load %ptr, %w : tensor<512x!tt.ptr<f32>>
    %v3 = tt.load %ptr, %m3 : tensor<512x!tt.ptr<f32>>
    %inner = arith.select %b, %v3, %v1 : tensor<512xi1>, tensor<512xf32>
    %outer = arith.select %a, %inner, %v1 : tensor<512xi1>, tensor<512xf32>
    tt.return %outer : tensor<512xf32>
  }
}
// COM: The rewritten arm is a fresh select, leaving the old one and the load it
// COM: kept alive dead. The pass's own clean-up retires both, so nothing of the
// COM: two-level tree survives.
// CHECK-LABEL:   tt.func @nested_conjunction(
// CHECK:           %[[V1:.*]] = tt.load %arg0, %arg3 :
// CHECK-NOT:       tt.load
// CHECK:           tt.return %[[V1]]

// -----

// COM: The same rule reaching all the way down: the mask *is* the condition, and
// COM: an unmasked load of the same pointers is available, so the masked load is
// COM: redundant in the true arm.
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @mask_is_the_condition(%ptr: tensor<512x!tt.ptr<f32>>, %c: tensor<512xi1>, %z: tensor<512xf32>) -> tensor<512xf32> {
    %v0 = tt.load %ptr : tensor<512x!tt.ptr<f32>>
    %v1 = tt.load %ptr, %c : tensor<512x!tt.ptr<f32>>
    %r = arith.select %c, %v1, %z : tensor<512xi1>, tensor<512xf32>
    tt.return %r : tensor<512xf32>
  }
}
// CHECK-LABEL:   tt.func @mask_is_the_condition(
// CHECK:           %[[V0:.*]] = tt.load %arg0 :
// CHECK-NOT:       tt.load
// CHECK:           arith.select %arg1, %[[V0]], %arg2

// -----

// COM: Nothing to fold: the pass must not clone operations when no load becomes
// COM: redundant.
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @no_load_no_rewrite(%c: tensor<512xi1>, %w: tensor<512xi1>, %x: tensor<512xf16>, %y: tensor<512xf16>) -> tensor<512xf16> {
    %m = arith.andi %c, %w : tensor<512xi1>
    %inner = arith.select %m, %x, %y : tensor<512xi1>, tensor<512xf16>
    %r = arith.select %c, %inner, %y : tensor<512xi1>, tensor<512xf16>
    tt.return %r : tensor<512xf16>
  }
}
// CHECK-LABEL:   tt.func @no_load_no_rewrite(
// CHECK:           %[[M:.*]] = arith.andi %arg0, %arg1
// CHECK:           %[[INNER:.*]] = arith.select %[[M]], %arg2, %arg3
// CHECK:           arith.select %arg0, %[[INNER]], %arg3
// CHECK-NOT:       arith.select

// -----

// COM: The core rewrite. `%v` is read only in the true arm of `select %c`, so it
// COM: need not read the lanes where `%c` is false.
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @narrow_by_true_arm(%ptr: tensor<512x!tt.ptr<f16>>, %c: tensor<512xi1>, %w: tensor<512xi1>, %z: tensor<512xf16>) -> tensor<512xf16> {
    %cst = arith.constant dense<0.000000e+00> : tensor<512xf16>
    %v = tt.load %ptr, %w, %cst : tensor<512x!tt.ptr<f16>>
    %r = arith.select %c, %v, %z : tensor<512xi1>, tensor<512xf16>
    tt.return %r : tensor<512xf16>
  }
}
// CHECK-LABEL:   tt.func @narrow_by_true_arm(
// CHECK:           %[[M:.*]] = arith.andi %arg2, %arg1 : tensor<512xi1>
// CHECK:           %[[V:.*]] = tt.load %arg0, %[[M]], %{{.*}} :
// CHECK:           arith.select %arg1, %[[V]], %arg3

// -----

// COM: A condition the mask already requires adds nothing, and must not be
// COM: re-anded just because the mask spells it out as its parts: the mask carries
// COM: a conjunction as its own conjuncts rather than as the one value the select
// COM: tests, so the two have to be compared flattened. `(%a & %b) & %w` already
// COM: implies `%a & %b`. @never_widens_a_load above covers the unflattened form.
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @implied_condition_adds_nothing(%ptr: tensor<512x!tt.ptr<f16>>, %a: tensor<512xi1>, %b: tensor<512xi1>, %w: tensor<512xi1>, %z: tensor<512xf16>) -> tensor<512xf16> {
    %cst = arith.constant dense<0.000000e+00> : tensor<512xf16>
    %ab = arith.andi %a, %b : tensor<512xi1>
    %m = arith.andi %ab, %w : tensor<512xi1>
    %v = tt.load %ptr, %m, %cst : tensor<512x!tt.ptr<f16>>
    %r = arith.select %ab, %v, %z : tensor<512xi1>, tensor<512xf16>
    tt.return %r : tensor<512xf16>
  }
}
// CHECK-LABEL:   tt.func @implied_condition_adds_nothing(
// CHECK:           %[[AB:.*]] = arith.andi %arg1, %arg2
// CHECK:           %[[M:.*]] = arith.andi %[[AB]], %arg3
// CHECK:           %[[V:.*]] = tt.load %arg0, %[[M]], %{{.*}} :
// CHECK-NOT:       arith.andi
// CHECK:           arith.select %[[AB]], %[[V]], %arg4

// -----

// COM: The condition is routinely computed *after* the load it narrows, because
// COM: the two are unrelated in the source. Its own inputs are already available,
// COM: so only its arithmetic has to move above the load.
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @hoists_the_condition(%ptr: tensor<512x!tt.ptr<f16>>, %x: tensor<512xi32>, %w: tensor<512xi1>, %z: tensor<512xf16>) -> tensor<512xf16> {
    %cst = arith.constant dense<0.000000e+00> : tensor<512xf16>
    %lim = arith.constant dense<7> : tensor<512xi32>
    %v = tt.load %ptr, %w, %cst : tensor<512x!tt.ptr<f16>>
    %c = arith.cmpi slt, %x, %lim : tensor<512xi32>
    %r = arith.select %c, %v, %z : tensor<512xi1>, tensor<512xf16>
    tt.return %r : tensor<512xf16>
  }
}
// COM: The `cmpi` now precedes the load it feeds the mask of.
// CHECK-LABEL:   tt.func @hoists_the_condition(
// CHECK:           %[[C:.*]] = arith.cmpi slt, %arg1
// CHECK:           %[[M:.*]] = arith.andi %arg2, %[[C]]
// CHECK:           %[[V:.*]] = tt.load %arg0, %[[M]], %{{.*}} :
// CHECK:           arith.select %[[C]], %[[V]], %arg3

// -----

// COM: A condition that cannot be made available at the load leaves the mask
// COM: alone. The `tt.load` feeding it may not be moved above a store it is
// COM: ordered after.
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @unhoistable_condition_blocks_narrowing(%ptr: tensor<512x!tt.ptr<f16>>, %cptr: tensor<512x!tt.ptr<i32>>, %w: tensor<512xi1>, %z: tensor<512xf16>, %val: tensor<512xi32>) -> tensor<512xf16> {
    %cst = arith.constant dense<0.000000e+00> : tensor<512xf16>
    %lim = arith.constant dense<7> : tensor<512xi32>
    %v = tt.load %ptr, %w, %cst : tensor<512x!tt.ptr<f16>>
    tt.store %cptr, %val, %w : tensor<512x!tt.ptr<i32>>
    %x = tt.load %cptr, %w : tensor<512x!tt.ptr<i32>>
    %c = arith.cmpi slt, %x, %lim : tensor<512xi32>
    %r = arith.select %c, %v, %z : tensor<512xi1>, tensor<512xf16>
    tt.return %r : tensor<512xf16>
  }
}
// CHECK-LABEL:   tt.func @unhoistable_condition_blocks_narrowing(
// CHECK:           %[[V:.*]] = tt.load %arg0, %arg2, %{{.*}} :
// CHECK:           tt.store
// CHECK:           arith.select %{{.*}}, %[[V]], %arg3

// -----

// COM: Two consumers observe the value under the *disjunction* of their
// COM: conditions, which is not what narrowing by either one would produce.
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @two_consumers_block_narrowing(%ptr: tensor<512x!tt.ptr<f16>>, %c: tensor<512xi1>, %d: tensor<512xi1>, %w: tensor<512xi1>, %z: tensor<512xf16>) -> tensor<512xf16> {
    %cst = arith.constant dense<0.000000e+00> : tensor<512xf16>
    %v = tt.load %ptr, %w, %cst : tensor<512x!tt.ptr<f16>>
    %r0 = arith.select %c, %v, %z : tensor<512xi1>, tensor<512xf16>
    %r1 = arith.select %d, %v, %z : tensor<512xi1>, tensor<512xf16>
    %r = arith.addf %r0, %r1 : tensor<512xf16>
    tt.return %r : tensor<512xf16>
  }
}
// CHECK-LABEL:   tt.func @two_consumers_block_narrowing(
// CHECK:           %[[V:.*]] = tt.load %arg0, %arg3, %{{.*}} :
// CHECK-NOT:       arith.andi
// CHECK:           arith.select %arg1, %[[V]], %arg4

// -----

// COM: A load whose result is used as a select's *condition* is not a datum the
// COM: select picks lanes of, so it carries no condition to narrow by.
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @load_as_condition_blocks_narrowing(%ptr: tensor<512x!tt.ptr<i1>>, %w: tensor<512xi1>, %x: tensor<512xf16>, %y: tensor<512xf16>) -> tensor<512xf16> {
    %cst = arith.constant dense<false> : tensor<512xi1>
    %v = tt.load %ptr, %w, %cst : tensor<512x!tt.ptr<i1>>
    %r = arith.select %v, %x, %y : tensor<512xi1>, tensor<512xf16>
    tt.return %r : tensor<512xf16>
  }
}
// CHECK-LABEL:   tt.func @load_as_condition_blocks_narrowing(
// CHECK:           %[[V:.*]] = tt.load %arg0, %arg1, %{{.*}} :
// CHECK-NOT:       arith.andi
// CHECK:           arith.select %[[V]], %arg2, %arg3

// -----

// COM: A volatile load has to be issued exactly as written.
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @volatile_load_is_left_alone(%ptr: tensor<512x!tt.ptr<f16>>, %c: tensor<512xi1>, %w: tensor<512xi1>, %z: tensor<512xf16>) -> tensor<512xf16> {
    %cst = arith.constant dense<0.000000e+00> : tensor<512xf16>
    %v = tt.load %ptr, %w, %cst {isVolatile = true} : tensor<512x!tt.ptr<f16>>
    %r = arith.select %c, %v, %z : tensor<512xi1>, tensor<512xf16>
    tt.return %r : tensor<512xf16>
  }
}
// CHECK-LABEL:   tt.func @volatile_load_is_left_alone(
// CHECK:           %[[V:.*]] = tt.load %arg0, %arg2, %{{.*}} {isVolatile = true}
// CHECK-NOT:       arith.andi
// CHECK:           arith.select %arg1, %[[V]], %arg3

// -----

// COM: An unmasked load has no mask operand to strengthen. Giving it one would
// COM: be a narrowing too, but it would also change what the lanes it stops
// COM: reading yield from the loaded value to a fabricated `other`.
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @unmasked_load_is_left_alone(%ptr: tensor<512x!tt.ptr<f16>>, %c: tensor<512xi1>, %z: tensor<512xf16>) -> tensor<512xf16> {
    %v = tt.load %ptr : tensor<512x!tt.ptr<f16>>
    %r = arith.select %c, %v, %z : tensor<512xi1>, tensor<512xf16>
    tt.return %r : tensor<512xf16>
  }
}
// CHECK-LABEL:   tt.func @unmasked_load_is_left_alone(
// CHECK:           %[[V:.*]] = tt.load %arg0 :
// CHECK-NOT:       arith.andi
// CHECK:           arith.select %arg1, %[[V]], %arg2

// -----

// COM: The narrowing reaches through the lane-wise arithmetic Inductor puts
// COM: between a load and the `tl.where` that consumes it.
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @reaches_through_lanewise_ops(%ptr: tensor<512x!tt.ptr<f16>>, %c: tensor<512xi1>, %w: tensor<512xi1>, %z: tensor<512xf32>) -> tensor<512xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<512xf16>
    %v = tt.load %ptr, %w, %cst : tensor<512x!tt.ptr<f16>>
    %e = arith.extf %v : tensor<512xf16> to tensor<512xf32>
    %n = arith.negf %e : tensor<512xf32>
    %r = arith.select %c, %n, %z : tensor<512xi1>, tensor<512xf32>
    tt.return %r : tensor<512xf32>
  }
}
// CHECK-LABEL:   tt.func @reaches_through_lanewise_ops(
// CHECK:           %[[M:.*]] = arith.andi %arg2, %arg1
// CHECK:           %[[V:.*]] = tt.load %arg0, %[[M]], %{{.*}} :

// -----

// COM: Nested selects compose: the value is observed only where both conditions
// COM: agree, so both narrow the mask.
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @nested_selects_compose(%ptr: tensor<512x!tt.ptr<f16>>, %c: tensor<512xi1>, %d: tensor<512xi1>, %w: tensor<512xi1>, %z: tensor<512xf16>) -> tensor<512xf16> {
    %cst = arith.constant dense<0.000000e+00> : tensor<512xf16>
    %v = tt.load %ptr, %w, %cst : tensor<512x!tt.ptr<f16>>
    %inner = arith.select %c, %v, %z : tensor<512xi1>, tensor<512xf16>
    %outer = arith.select %d, %inner, %z : tensor<512xi1>, tensor<512xf16>
    tt.return %outer : tensor<512xf16>
  }
}
// CHECK-LABEL:   tt.func @nested_selects_compose(
// CHECK:           %[[M0:.*]] = arith.andi %arg3, %arg1
// CHECK:           %[[M1:.*]] = arith.andi %[[M0]], %arg2
// CHECK:           %[[V:.*]] = tt.load %arg0, %[[M1]], %{{.*}} :

// -----

// COM: `tt.elementwise_inline_asm` carries the `Elementwise` trait, but hands the
// COM: asm `packed_element` lanes at a time and leaves the grouping to it, so a
// COM: lane of its result may come from a neighbouring lane's input. Narrowing
// COM: through it would feed those lanes `other` instead of loaded data, so it
// COM: ends the walk like any other lane-mixing consumer.
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @inline_asm_blocks_narrowing(%ptr: tensor<512x!tt.ptr<f16>>, %c: tensor<512xi1>, %w: tensor<512xi1>, %z: tensor<512xf16>) -> tensor<512xf16> {
    %cst = arith.constant dense<0.000000e+00> : tensor<512xf16>
    %v = tt.load %ptr, %w, %cst : tensor<512x!tt.ptr<f16>>
    %a = tt.elementwise_inline_asm "nop" {constraints = "=r,r", packed_element = 2 : i32, pure = true} %v : tensor<512xf16> -> tensor<512xf16>
    %r = arith.select %c, %a, %z : tensor<512xi1>, tensor<512xf16>
    tt.return %r : tensor<512xf16>
  }
}
// CHECK-LABEL:   tt.func @inline_asm_blocks_narrowing(
// CHECK:           %[[V:.*]] = tt.load %arg0, %arg2, %{{.*}} :
// CHECK-NOT:       arith.andi
// CHECK:           tt.elementwise_inline_asm

// -----

// COM: A rank-2 load is a candidate for the 2D block I/O path, which
// COM: `MaterializeBlockPointer` withholds from a mask whose per-dimension
// COM: constancy is not a power of two of at least 2. The conditions narrowing
// COM: would AND in are routinely data-dependent, whose constancy is 1, and
// COM: losing the block tile costs far more than the lanes narrowing saves.
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @rank2_load_is_left_alone(%ptr: tensor<64x64x!tt.ptr<f16>>, %c: tensor<64x64xi1>, %w: tensor<64x64xi1>, %z: tensor<64x64xf16>) -> tensor<64x64xf16> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf16>
    %v = tt.load %ptr, %w, %cst : tensor<64x64x!tt.ptr<f16>>
    %r = arith.select %c, %v, %z : tensor<64x64xi1>, tensor<64x64xf16>
    tt.return %r : tensor<64x64xf16>
  }
}
// CHECK-LABEL:   tt.func @rank2_load_is_left_alone(
// CHECK:           %[[V:.*]] = tt.load %arg0, %arg2, %{{.*}} :
// CHECK-NOT:       arith.andi
// CHECK:           arith.select %arg1, %[[V]], %arg3
