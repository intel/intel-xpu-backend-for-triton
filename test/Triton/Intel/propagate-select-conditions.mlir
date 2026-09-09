// RUN: triton-opt %s -split-input-file -triton-intel-propagate-select-conditions | FileCheck %s
// RUN: triton-opt %s -split-input-file -triton-intel-propagate-select-conditions -cse -canonicalize | FileCheck %s --check-prefix=FOLDED

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
// CHECK-NOT:       %[[V0]] = tt.load
// CHECK:           arith.select %arg1, %[[V0]], %[[V0]]
// FOLDED-LABEL:   tt.func @reuse_wider_load(
// FOLDED:           %[[V0:.*]] = tt.load
// FOLDED-NOT:       tt.load
// FOLDED:           tt.return %[[V0]]

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
// CHECK:           %[[NEW:.*]] = arith.select %arg3, %arg4, %[[V0]]
// CHECK:           arith.select %arg1, %[[NEW]], %[[V0]]
// FOLDED-LABEL:   tt.func @through_nested_selects(
// FOLDED:           %[[V0:.*]] = tt.load
// FOLDED-NOT:       tt.load

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
// CHECK:           %[[CST:.*]] = arith.constant dense<0.000000e+00>
// CHECK:           arith.select %arg1, %arg3, %[[CST]]
// FOLDED-LABEL:   tt.func @false_arm_yields_other(
// FOLDED-NOT:       tt.load

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
// CHECK-NOT:       %[[V0]], %{{.*}} :
// CHECK:           arith.select %arg1, %[[V0]], %[[V0]]
// FOLDED-LABEL:   tt.func @reuse_unmasked_load(
// FOLDED:           %[[V0:.*]] = tt.load %arg0 :
// FOLDED-NOT:       tt.load
// FOLDED:           tt.return %[[V0]]

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
// CHECK-LABEL:   tt.func @nonzero_other_blocks_reuse(
// CHECK:           %[[ONE:.*]] = arith.constant dense<1.000000e+00>
// CHECK:           %[[M:.*]] = arith.andi %arg1, %arg2
// CHECK:           %[[V0:.*]] = tt.load %arg0, %arg2 :
// CHECK:           %[[V1:.*]] = tt.load %arg0, %[[M]], %[[ONE]] :
// CHECK:           arith.select %arg1, %[[V1]], %[[V0]]
// FOLDED-LABEL:   tt.func @nonzero_other_blocks_reuse(
// FOLDED-COUNT-2:   tt.load

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
// FOLDED-LABEL:   tt.func @never_widens_a_load(
// FOLDED-COUNT-1:   tt.load

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
// FOLDED-LABEL:   tt.func @write_between_blocks_reuse(
// FOLDED-COUNT-2:   tt.load

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
// CHECK:           %[[V0:.*]] = tt.load %arg0, %arg2, %{{.*}} :
// CHECK:           %[[V1:.*]] = tt.load %arg0, %[[M]], %{{.*}} :
// CHECK:           arith.select %arg1, %[[V1]], %[[V0]]
// FOLDED-LABEL:   tt.func @conflicting_other_blocks_reuse(
// FOLDED-COUNT-2:   tt.load

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
// FOLDED-LABEL:   tt.func @reduction_blocks_propagation(
// FOLDED-COUNT-2:   tt.load

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
// CHECK:           %[[A:.*]] = arith.andi %[[C1]], %arg3 : i32
// CHECK:           %[[AF:.*]] = arith.sitofp %[[A]] : i32 to f32
// CHECK:           %[[S:.*]] = arith.addf %[[V0]], %[[AF]] : f32
// CHECK:           arith.select %arg1, %[[S]], %[[V0]]
// FOLDED-LABEL:   tt.func @wide_andi_is_not_a_mask(
// FOLDED:           %[[FC1:.*]] = arith.constant 1 : i32
// FOLDED:           arith.andi %arg3, %[[FC1]] : i32

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
// COM: The pass itself does not clean up: the rewritten arm is a fresh select and
// COM: the old one, with the load it kept alive, is left dead for the canonicalizer.
// CHECK-LABEL:   tt.func @nested_conjunction(
// CHECK:           %[[V1:.*]] = tt.load %arg0, %arg3 :
// CHECK:           %[[INNER:.*]] = arith.select %arg2, %[[V1]], %[[V1]]
// CHECK:           arith.select %arg1, %[[INNER]], %[[V1]]
// FOLDED-LABEL:   tt.func @nested_conjunction(
// FOLDED:           %[[V1:.*]] = tt.load
// FOLDED-NOT:       tt.load
// FOLDED:           tt.return %[[V1]]

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
// CHECK:           arith.select %arg1, %[[V0]], %arg2
// FOLDED-LABEL:   tt.func @mask_is_the_condition(
// FOLDED:           %[[V0:.*]] = tt.load %arg0 :
// FOLDED-NOT:       tt.load

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
// FOLDED-LABEL:   tt.func @no_load_no_rewrite(
