// RUN: triton-opt %s -split-input-file -tritonintelgpu-code-sinking | FileCheck %s

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @sink_load_used_after_loop(%ptr_a: !tt.ptr<f32>, %ptr_b: !tt.ptr<f32>, %ptr_out: !tt.ptr<f32>, %init: f32) {
    // COM: This test verifies that a tt.load whose result is not used inside
    // COM: the loop is sunk to just before its first use after the loop.
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c10_i32 = arith.constant 10 : i32

    // CHECK-LABEL: tt.func @sink_load_used_after_loop
    // CHECK: %[[PTR_B:.*]] = tt.splat {{.*}} : !tt.ptr<f32>
    // CHECK-NOT: tt.load %[[PTR_B]]
    %splat_b = tt.splat %ptr_b : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %b = tt.load %splat_b : tensor<256x!tt.ptr<f32>>

    %splat_a = tt.splat %ptr_a : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %init_splat = tt.splat %init : f32 -> tensor<256xf32>

    // CHECK: scf.for
    %result = scf.for %iv = %c0_i32 to %c10_i32 step %c1_i32 iter_args(%acc = %init_splat) -> (tensor<256xf32>) : i32 {
      %a = tt.load %splat_a : tensor<256x!tt.ptr<f32>>
      %sum = arith.addf %acc, %a : tensor<256xf32>
      scf.yield %sum : tensor<256xf32>
    }

    // CHECK: %[[B:.*]] = tt.load %[[PTR_B]]
    // CHECK: arith.addf {{.*}}, %[[B]]
    %final = arith.addf %result, %b : tensor<256xf32>

    %splat_out = tt.splat %ptr_out : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    tt.store %splat_out, %final : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @used_in_loop(%ptr_a: !tt.ptr<f32>, %ptr_b: !tt.ptr<f32>, %ptr_out: !tt.ptr<f32>, %init: f32) {
    // COM: This test verifies that a tt.load whose result IS used inside
    // COM: the loop is NOT moved.
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c10_i32 = arith.constant 10 : i32

    // CHECK-LABEL: tt.func @used_in_loop
    // CHECK: %[[B:.*]] = tt.load
    // CHECK: scf.for
    %splat_b = tt.splat %ptr_b : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %b = tt.load %splat_b : tensor<256x!tt.ptr<f32>>

    %splat_a = tt.splat %ptr_a : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %init_splat = tt.splat %init : f32 -> tensor<256xf32>

    %result = scf.for %iv = %c0_i32 to %c10_i32 step %c1_i32 iter_args(%acc = %init_splat) -> (tensor<256xf32>) : i32 {
      %a = tt.load %splat_a : tensor<256x!tt.ptr<f32>>
      // CHECK: arith.addf {{.*}}, %[[B]]
      %with_b = arith.addf %a, %b : tensor<256xf32>
      %sum = arith.addf %acc, %with_b : tensor<256xf32>
      scf.yield %sum : tensor<256xf32>
    }

    %splat_out = tt.splat %ptr_out : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    tt.store %splat_out, %result : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @store_in_loop_body(%ptr_a: !tt.ptr<f32>, %ptr_b: !tt.ptr<f32>, %ptr_out: !tt.ptr<f32>, %ptr_tmp: !tt.ptr<f32>, %init: f32) {
    // COM: This test verifies that a tt.load is NOT moved if the loop body
    // COM: contains a tt.store (which could alias the load).
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c10_i32 = arith.constant 10 : i32

    // CHECK-LABEL: tt.func @store_in_loop_body
    // CHECK: %[[B:.*]] = tt.load
    // CHECK: scf.for
    %splat_b = tt.splat %ptr_b : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %b = tt.load %splat_b : tensor<256x!tt.ptr<f32>>

    %splat_a = tt.splat %ptr_a : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %splat_tmp = tt.splat %ptr_tmp : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %init_splat = tt.splat %init : f32 -> tensor<256xf32>

    %result = scf.for %iv = %c0_i32 to %c10_i32 step %c1_i32 iter_args(%acc = %init_splat) -> (tensor<256xf32>) : i32 {
      %a = tt.load %splat_a : tensor<256x!tt.ptr<f32>>
      // CHECK: tt.store
      tt.store %splat_tmp, %a : tensor<256x!tt.ptr<f32>>
      %sum = arith.addf %acc, %a : tensor<256xf32>
      scf.yield %sum : tensor<256xf32>
    }

    %final = arith.addf %result, %b : tensor<256xf32>
    %splat_out = tt.splat %ptr_out : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    tt.store %splat_out, %final : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @used_before_loop(%ptr_a: !tt.ptr<f32>, %ptr_b: !tt.ptr<f32>, %ptr_out: !tt.ptr<f32>, %init: f32) {
    // COM: This test verifies that a tt.load is NOT moved if it has
    // COM: a use before the loop.
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c10_i32 = arith.constant 10 : i32

    // CHECK-LABEL: tt.func @used_before_loop
    // CHECK: %[[B:.*]] = tt.load
    %splat_b = tt.splat %ptr_b : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %b = tt.load %splat_b : tensor<256x!tt.ptr<f32>>

    %splat_a = tt.splat %ptr_a : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %init_splat = tt.splat %init : f32 -> tensor<256xf32>

    // Use %b before the loop
    // CHECK: arith.addf {{.*}}, %[[B]]
    %pre_use = arith.addf %init_splat, %b : tensor<256xf32>

    // CHECK: scf.for
    %result = scf.for %iv = %c0_i32 to %c10_i32 step %c1_i32 iter_args(%acc = %pre_use) -> (tensor<256xf32>) : i32 {
      %a = tt.load %splat_a : tensor<256x!tt.ptr<f32>>
      %sum = arith.addf %acc, %a : tensor<256xf32>
      scf.yield %sum : tensor<256xf32>
    }

    %final = arith.addf %result, %b : tensor<256xf32>
    %splat_out = tt.splat %ptr_out : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    tt.store %splat_out, %final : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @passed_through_iter_args(%ptr_a: !tt.ptr<f32>, %ptr_b: !tt.ptr<f32>, %ptr_out: !tt.ptr<f32>, %init: f32) {
    // COM: This test verifies that a tt.load is NOT moved if it is passed
    // COM: through the loop via iter_args (making it live through the loop).
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c10_i32 = arith.constant 10 : i32

    // CHECK-LABEL: tt.func @passed_through_iter_args
    // CHECK: %[[B:.*]] = tt.load
    %splat_b = tt.splat %ptr_b : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %b = tt.load %splat_b : tensor<256x!tt.ptr<f32>>

    %splat_a = tt.splat %ptr_a : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %init_splat = tt.splat %init : f32 -> tensor<256xf32>

    // CHECK: scf.for
    // Pass %b as an iter_arg
    %result:2 = scf.for %iv = %c0_i32 to %c10_i32 step %c1_i32 iter_args(%acc = %init_splat, %b_through = %b) -> (tensor<256xf32>, tensor<256xf32>) : i32 {
      %a = tt.load %splat_a : tensor<256x!tt.ptr<f32>>
      %sum = arith.addf %acc, %a : tensor<256xf32>
      scf.yield %sum, %b_through : tensor<256xf32>, tensor<256xf32>
    }

    %final = arith.addf %result#0, %result#1 : tensor<256xf32>
    %splat_out = tt.splat %ptr_out : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    tt.store %splat_out, %final : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @two_loads_used_after_loop(%ptr_a: !tt.ptr<f32>, %ptr_g: !tt.ptr<f32>, %ptr_fc: !tt.ptr<f32>, %ptr_out: !tt.ptr<f32>, %init: f32) {
    // COM: The real SwiGLU scenario: two bias loads (b_g, b_fc) before the
    // COM: K-loop, each first used after it. Both should be sunk below the loop.
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c10_i32 = arith.constant 10 : i32

    // CHECK-LABEL: tt.func @two_loads_used_after_loop
    // COM: Neither bias load may appear before the loop (both are loop-dead).
    // CHECK-NOT: tt.load
    %splat_g = tt.splat %ptr_g : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %splat_fc = tt.splat %ptr_fc : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %b_g = tt.load %splat_g : tensor<256x!tt.ptr<f32>>
    %b_fc = tt.load %splat_fc : tensor<256x!tt.ptr<f32>>

    %splat_a = tt.splat %ptr_a : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %init_splat = tt.splat %init : f32 -> tensor<256xf32>

    // CHECK: scf.for
    %result = scf.for %iv = %c0_i32 to %c10_i32 step %c1_i32 iter_args(%acc = %init_splat) -> (tensor<256xf32>) : i32 {
      %a = tt.load %splat_a : tensor<256x!tt.ptr<f32>>
      %sum = arith.addf %acc, %a : tensor<256xf32>
      scf.yield %sum : tensor<256xf32>
    }

    // COM: Both loads (and their pure splat operands) are sunk below the loop,
    // COM: each load just before its use. The fixpoint also sinks the splats.
    // CHECK: tt.load
    // CHECK: arith.addf
    // CHECK: tt.load
    // CHECK: arith.addf
    %acc_g = arith.addf %result, %b_g : tensor<256xf32>
    %acc_fc = arith.addf %result, %b_fc : tensor<256xf32>
    %final = arith.addf %acc_g, %acc_fc : tensor<256xf32>

    %splat_out = tt.splat %ptr_out : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    tt.store %splat_out, %final : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @store_after_loop_before_use(%ptr_a: !tt.ptr<f32>, %ptr_b: !tt.ptr<f32>, %ptr_out: !tt.ptr<f32>, %ptr_tmp: !tt.ptr<f32>, %init: f32) {
    // COM: A store sits between the loop and the load's first use. The store
    // COM: could alias the load, so the load must NOT be sunk past it.
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c10_i32 = arith.constant 10 : i32

    // CHECK-LABEL: tt.func @store_after_loop_before_use
    // CHECK: %[[B:.*]] = tt.load
    // CHECK: scf.for
    %splat_b = tt.splat %ptr_b : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %b = tt.load %splat_b : tensor<256x!tt.ptr<f32>>

    %splat_a = tt.splat %ptr_a : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %splat_tmp = tt.splat %ptr_tmp : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %init_splat = tt.splat %init : f32 -> tensor<256xf32>

    %result = scf.for %iv = %c0_i32 to %c10_i32 step %c1_i32 iter_args(%acc = %init_splat) -> (tensor<256xf32>) : i32 {
      %a = tt.load %splat_a : tensor<256x!tt.ptr<f32>>
      %sum = arith.addf %acc, %a : tensor<256xf32>
      scf.yield %sum : tensor<256xf32>
    }

    // COM: Aliasing write after the loop, before the use -> blocks the sink.
    // CHECK: tt.store
    tt.store %splat_tmp, %result : tensor<256x!tt.ptr<f32>>
    %final = arith.addf %result, %b : tensor<256xf32>

    %splat_out = tt.splat %ptr_out : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    tt.store %splat_out, %final : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @sink_pure_non_load(%ptr_a: !tt.ptr<f32>, %x: f32, %y: f32, %ptr_out: !tt.ptr<f32>, %init: f32) {
    // COM: The pass is not load-specific: any pure op unused in the loop and
    // COM: used after it is sunk. Here an elementwise arith.mulf is sunk.
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c10_i32 = arith.constant 10 : i32

    // CHECK-LABEL: tt.func @sink_pure_non_load
    %sx = tt.splat %x : f32 -> tensor<256xf32>
    %sy = tt.splat %y : f32 -> tensor<256xf32>
    // COM: The pure product must not appear before the loop.
    // CHECK-NOT: arith.mulf
    %prod = arith.mulf %sx, %sy : tensor<256xf32>

    %splat_a = tt.splat %ptr_a : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %init_splat = tt.splat %init : f32 -> tensor<256xf32>

    // CHECK: scf.for
    %result = scf.for %iv = %c0_i32 to %c10_i32 step %c1_i32 iter_args(%acc = %init_splat) -> (tensor<256xf32>) : i32 {
      %a = tt.load %splat_a : tensor<256x!tt.ptr<f32>>
      %sum = arith.addf %acc, %a : tensor<256xf32>
      scf.yield %sum : tensor<256xf32>
    }

    // COM: The product is sunk below the loop, before its first use.
    // CHECK: arith.mulf
    // CHECK: arith.addf
    %final = arith.addf %result, %prod : tensor<256xf32>

    %splat_out = tt.splat %ptr_out : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    tt.store %splat_out, %final : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @sink_load_past_while(%ptr_a: !tt.ptr<f32>, %ptr_b: !tt.ptr<f32>, %ptr_out: !tt.ptr<f32>, %init_cond: i1) {
    // COM: This test verifies that a tt.load whose result is not used inside
    // COM: an scf.while loop is sunk to just before its first use after the loop.
    %c0_i32 = arith.constant 0 : i32
    %c10_i32 = arith.constant 10 : i32

    // CHECK-LABEL: tt.func @sink_load_past_while
    // COM: The bias load (and its pure splat operand) are loop-dead -> must not
    // COM: appear before the while loop.
    // CHECK-NOT: tt.load
    %splat_b = tt.splat %ptr_b : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %b = tt.load %splat_b : tensor<256x!tt.ptr<f32>>

    %splat_a = tt.splat %ptr_a : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>

    // CHECK: scf.while
    %result = scf.while (%arg_ctr = %c0_i32, %arg_cond = %init_cond) : (i32, i1) -> (i32) {
      scf.condition(%arg_cond) %arg_ctr : i32
    } do {
    ^bb0(%ctr: i32):
      %incremented = arith.addi %ctr, %c0_i32 : i32
      %a = tt.load %splat_a : tensor<256x!tt.ptr<f32>>
      %cmp = arith.cmpi slt, %incremented, %c10_i32 : i32
      scf.yield %incremented, %cmp : i32, i1
    }

    // COM: The load is sunk below the while, just before its first use. Anchor
    // COM: on the post-loop sitofp so we skip the in-loop load of %splat_a.
    // CHECK: arith.sitofp
    // CHECK: %[[B:.*]] = tt.load
    // CHECK: arith.addf {{.*}}, %[[B]]
    %result_tensor = tt.splat %result : i32 -> tensor<256xi32>
    %result_float = arith.sitofp %result_tensor : tensor<256xi32> to tensor<256xf32>
    %final = arith.addf %result_float, %b : tensor<256xf32>

    %splat_out = tt.splat %ptr_out : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    tt.store %splat_out, %final : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @sink_load_past_if(%ptr_a: !tt.ptr<f32>, %ptr_b: !tt.ptr<f32>, %ptr_out: !tt.ptr<f32>, %cond: i1) {
    // COM: This test verifies that a tt.load whose result is not used inside
    // COM: an scf.if is sunk to just before its first use after the scf.if.
    %c1_f32 = arith.constant 1.0 : f32
    %c2_f32 = arith.constant 2.0 : f32

    // CHECK-LABEL: tt.func @sink_load_past_if
    // COM: The bias load (and its pure splat operand) are dead across the
    // COM: scf.if -> must not appear before it. Note %a IS used inside the if,
    // COM: so %a's load must stay above the if.
    // CHECK: tt.load
    // CHECK: scf.if
    %splat_b = tt.splat %ptr_b : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %b = tt.load %splat_b : tensor<256x!tt.ptr<f32>>

    %splat_a = tt.splat %ptr_a : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %a = tt.load %splat_a : tensor<256x!tt.ptr<f32>>

    %result = scf.if %cond -> (tensor<256xf32>) {
      %splat_1 = tt.splat %c1_f32 : f32 -> tensor<256xf32>
      %then_val = arith.addf %a, %splat_1 : tensor<256xf32>
      scf.yield %then_val : tensor<256xf32>
    } else {
      %splat_2 = tt.splat %c2_f32 : f32 -> tensor<256xf32>
      %else_val = arith.mulf %a, %splat_2 : tensor<256xf32>
      scf.yield %else_val : tensor<256xf32>
    }

    // COM: %b's load is sunk below the if, just before its use.
    // CHECK: %[[B:.*]] = tt.load
    // CHECK: arith.addf {{.*}}, %[[B]]
    %final = arith.addf %result, %b : tensor<256xf32>

    %splat_out = tt.splat %ptr_out : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    tt.store %splat_out, %final : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @used_inside_if_branch(%ptr_a: !tt.ptr<f32>, %ptr_b: !tt.ptr<f32>, %ptr_out: !tt.ptr<f32>, %cond: i1) {
    // COM: This test verifies that a tt.load is NOT moved if the loaded value
    // COM: is referenced inside one of the scf.if branches.
    %c1_f32 = arith.constant 1.0 : f32

    // CHECK-LABEL: tt.func @used_inside_if_branch
    // CHECK: %[[B:.*]] = tt.load
    %splat_b = tt.splat %ptr_b : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %b = tt.load %splat_b : tensor<256x!tt.ptr<f32>>

    %splat_a = tt.splat %ptr_a : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %a = tt.load %splat_a : tensor<256x!tt.ptr<f32>>

    // CHECK: scf.if
    // CHECK: arith.addf {{.*}}, %[[B]]
    %result = scf.if %cond -> (tensor<256xf32>) {
      // Use %b inside the then-branch
      %then_val = arith.addf %a, %b : tensor<256xf32>
      scf.yield %then_val : tensor<256xf32>
    } else {
      %splat_1 = tt.splat %c1_f32 : f32 -> tensor<256xf32>
      %else_val = arith.addf %a, %splat_1 : tensor<256xf32>
      scf.yield %else_val : tensor<256xf32>
    }

    %final = arith.addf %result, %b : tensor<256xf32>

    %splat_out = tt.splat %ptr_out : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    tt.store %splat_out, %final : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @used_as_barrier_operand(%ptr_a: !tt.ptr<f32>, %ptr_b: !tt.ptr<f32>, %ptr_out: !tt.ptr<f32>, %init: f32) {
    // COM: This test verifies that a value is NOT moved if it is used as an
    // COM: scf.for iter_args init operand — the barrier op itself is a user.
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c10_i32 = arith.constant 10 : i32

    // CHECK-LABEL: tt.func @used_as_barrier_operand
    %splat_b = tt.splat %ptr_b : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    // CHECK: %[[B:.*]] = tt.load
    %b = tt.load %splat_b : tensor<256x!tt.ptr<f32>>

    %splat_a = tt.splat %ptr_a : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %init_splat = tt.splat %init : f32 -> tensor<256xf32>

    // COM: %b is passed as the init value for an iter_arg (use at barrier op)
    // CHECK: scf.for
    %result = scf.for %iv = %c0_i32 to %c10_i32 step %c1_i32 iter_args(%acc = %b) -> (tensor<256xf32>) : i32 {
      %a = tt.load %splat_a : tensor<256x!tt.ptr<f32>>
      %sum = arith.addf %acc, %a : tensor<256xf32>
      scf.yield %sum : tensor<256xf32>
    }

    %final = arith.addf %result, %init_splat : tensor<256xf32>
    %splat_out = tt.splat %ptr_out : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    tt.store %splat_out, %final : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}
