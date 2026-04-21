// RUN: triton-opt %s -split-input-file -tritonintelgpu-annotate-cache-control | FileCheck %s

// COM: Test a — load that does NOT feed a dot gets CG (cache = 2).
// COM: Store is intentionally left untouched.

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @streaming_load_gets_cg
  tt.func public @streaming_load_gets_cg(%ptr: tensor<1024x!tt.ptr<f32>>, %out: tensor<1024x!tt.ptr<f32>>) {
    // CHECK: tt.load {{.*}} cacheModifier = cg
    %0 = tt.load %ptr : tensor<1024x!tt.ptr<f32>>
    // CHECK: tt.store
    // CHECK-NOT: cacheModifier = cg
    tt.store %out, %0 : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// COM: Test b — load that directly feeds a tt.dot stays NONE.

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @gemm_operand_load_unchanged
  tt.func public @gemm_operand_load_unchanged(%aptr: tensor<32x32x!tt.ptr<f16>>,
                                              %bptr: tensor<32x32x!tt.ptr<f16>>,
                                              %c: tensor<32x32xf32>) -> tensor<32x32xf32> {
    // CHECK: tt.load
    // CHECK-NOT: cacheModifier = cg
    %a = tt.load %aptr : tensor<32x32x!tt.ptr<f16>>
    // CHECK: tt.load
    // CHECK-NOT: cacheModifier = cg
    %b = tt.load %bptr : tensor<32x32x!tt.ptr<f16>>
    %d = tt.dot %a, %b, %c : tensor<32x32xf16> * tensor<32x32xf16> -> tensor<32x32xf32>
    tt.return %d : tensor<32x32xf32>
  }
}

// -----

// COM: Test c — load that feeds a tt.dot through an scf.for iter_arg stays NONE.

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @gemm_loop_iter_arg_unchanged
  tt.func public @gemm_loop_iter_arg_unchanged(%aptr: tensor<32x32x!tt.ptr<f16>>,
                                               %bptr: tensor<32x32x!tt.ptr<f16>>,
                                               %c: tensor<32x32xf32>,
                                               %lb: index, %ub: index, %step: index)
      -> tensor<32x32xf32> {
    // CHECK: tt.load
    // CHECK-NOT: cacheModifier = cg
    %a = tt.load %aptr : tensor<32x32x!tt.ptr<f16>>
    %res = scf.for %i = %lb to %ub step %step iter_args(%acc = %c) -> tensor<32x32xf32> {
      %b = tt.load %bptr : tensor<32x32x!tt.ptr<f16>>
      %d = tt.dot %a, %b, %acc : tensor<32x32xf16> * tensor<32x32xf16> -> tensor<32x32xf32>
      scf.yield %d : tensor<32x32xf32>
    }
    tt.return %res : tensor<32x32xf32>
  }
}

// -----

// COM: Test d — store of a tt.dot result (GEMM epilogue) stays NONE.

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @gemm_epilogue_store_unchanged
  tt.func public @gemm_epilogue_store_unchanged(%aptr: tensor<32x32x!tt.ptr<f16>>,
                                                %bptr: tensor<32x32x!tt.ptr<f16>>,
                                                %cptr: tensor<32x32x!tt.ptr<f32>>,
                                                %c: tensor<32x32xf32>) {
    %a = tt.load %aptr : tensor<32x32x!tt.ptr<f16>>
    %b = tt.load %bptr : tensor<32x32x!tt.ptr<f16>>
    %d = tt.dot %a, %b, %c : tensor<32x32xf16> * tensor<32x32xf16> -> tensor<32x32xf32>
    // CHECK: tt.store
    // CHECK-NOT: cacheModifier = cg
    tt.store %cptr, %d : tensor<32x32x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// COM: Test e — store of non-dot data is NOT annotated (stores are never
// COM: annotated; see pass docs).

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @streaming_store_unchanged
  tt.func public @streaming_store_unchanged(%ptr: tensor<1024x!tt.ptr<f32>>,
                                            %val: tensor<1024xf32>) {
    // CHECK: tt.store
    // CHECK-NOT: cacheModifier = cg
    tt.store %ptr, %val : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// COM: Test f — pure-input kernel: arg is only loaded (no store), no atomics.
// COM: Load should still get .cg (regression check: heuristic must not over-skip).

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @pure_input_load_gets_cg
  tt.func public @pure_input_load_gets_cg(%ptr: tensor<32x!tt.ptr<f32>>) -> tensor<32xf32> {
    // CHECK: tt.load {{.*}} cacheModifier = cg
    %0 = tt.load %ptr : tensor<32x!tt.ptr<f32>>
    tt.return %0 : tensor<32xf32>
  }
}

// -----

// COM: Test g — read-write arg: the same arg is loaded (via tt.addptr) AND
// COM: stored to (via tt.addptr). Load must NOT get .cg.

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @rw_arg_load_unchanged
  tt.func public @rw_arg_load_unchanged(%base: tensor<32x!tt.ptr<f32>>,
                                        %offs: tensor<32xi32>,
                                        %val: tensor<32xf32>) {
    %p0 = tt.addptr %base, %offs : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    // CHECK: tt.load
    // CHECK-NOT: cacheModifier = cg
    %0 = tt.load %p0 : tensor<32x!tt.ptr<f32>>
    %p1 = tt.addptr %base, %offs : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    tt.store %p1, %val : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// COM: Test h — atomic kernel: function contains tt.atomic_rmw on one arg;
// COM: a separate arg is only loaded. Conservative rule: any atomics in the
// COM: function poison loaded args, so the pure load must NOT get .cg.

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @atomic_kernel_load_unchanged
  tt.func public @atomic_kernel_load_unchanged(%in: tensor<32x!tt.ptr<f32>>,
                                               %acc: tensor<32x!tt.ptr<f32>>,
                                               %v: tensor<32xf32>,
                                               %mask: tensor<32xi1>) -> tensor<32xf32> {
    // CHECK: tt.load
    // CHECK-NOT: cacheModifier = cg
    %0 = tt.load %in : tensor<32x!tt.ptr<f32>>
    %1 = tt.atomic_rmw fadd, acq_rel, gpu, %acc, %v, %mask : (tensor<32x!tt.ptr<f32>>, tensor<32xf32>, tensor<32xi1>) -> tensor<32xf32>
    tt.return %0 : tensor<32xf32>
  }
}

// -----

// COM: Test i — RW arg reached through scf.for iter-arg: arg passed as
// COM: scf.for iter_arg, loaded inside the loop, stored-to elsewhere in the
// COM: same function. Load must NOT get .cg.

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @rw_arg_via_scf_for_unchanged
  tt.func public @rw_arg_via_scf_for_unchanged(%base: tensor<32x!tt.ptr<f32>>,
                                               %offs: tensor<32xi32>,
                                               %val: tensor<32xf32>,
                                               %lb: index, %ub: index, %step: index)
      -> tensor<32xf32> {
    %init = arith.constant dense<0.0> : tensor<32xf32>
    %res = scf.for %i = %lb to %ub step %step iter_args(%p = %base) -> tensor<32x!tt.ptr<f32>> {
      // CHECK: tt.load
      // CHECK-NOT: cacheModifier = cg
      %0 = tt.load %p : tensor<32x!tt.ptr<f32>>
      scf.yield %p : tensor<32x!tt.ptr<f32>>
    }
    %sp = tt.addptr %base, %offs : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    tt.store %sp, %val : tensor<32x!tt.ptr<f32>>
    tt.return %init : tensor<32xf32>
  }
}

// -----

// COM: Test j — mixed kernel: no atomics, two args. DW is RW (loaded and
// COM: stored), X is read-only. Load of X gets .cg; load of DW does not.

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @mixed_args_selective_cg
  tt.func public @mixed_args_selective_cg(%DW: tensor<32x!tt.ptr<f32>>,
                                          %X: tensor<32x!tt.ptr<f32>>,
                                          %offs: tensor<32xi32>,
                                          %val: tensor<32xf32>) -> (tensor<32xf32>, tensor<32xf32>) {
    %pd = tt.addptr %DW, %offs : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    // CHECK: tt.load
    // CHECK-NOT: cacheModifier = cg
    %dw = tt.load %pd : tensor<32x!tt.ptr<f32>>
    // CHECK: tt.load {{.*}} cacheModifier = cg
    %x = tt.load %X : tensor<32x!tt.ptr<f32>>
    %sp = tt.addptr %DW, %offs : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    tt.store %sp, %val : tensor<32x!tt.ptr<f32>>
    tt.return %dw, %x : tensor<32xf32>, tensor<32xf32>
  }
}

// -----

// COM: Test k — root peeling through tt.splat: scalar-ptr arg is splatted
// COM: and used for both load (via addptr chain) and store (via addptr chain),
// COM: making the root arg RW. Load must NOT get .cg.

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @rw_arg_via_splat_unchanged
  tt.func public @rw_arg_via_splat_unchanged(%base: !tt.ptr<f32>,
                                             %offs: tensor<32xi32>,
                                             %val: tensor<32xf32>) -> tensor<32xf32> {
    %splat_l = tt.splat %base : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %lp = tt.addptr %splat_l, %offs : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    // CHECK: tt.load
    // CHECK-NOT: cacheModifier = cg
    %0 = tt.load %lp : tensor<32x!tt.ptr<f32>>
    %splat_s = tt.splat %base : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %sp = tt.addptr %splat_s, %offs : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    tt.store %sp, %val : tensor<32x!tt.ptr<f32>>
    tt.return %0 : tensor<32xf32>
  }
}

// -----

// COM: Test l — user-provided cache modifier on a load is honored as-is.
// COM: The pass must NOT overwrite `ca` with `cg`, even on a pure-input load
// COM: that would otherwise qualify for `.cg` annotation.

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @user_modifier_ca_preserved
  tt.func public @user_modifier_ca_preserved(%ptr: tensor<32x!tt.ptr<f32>>) -> tensor<32xf32> {
    // CHECK: tt.load {{.*}} cacheModifier = ca
    // CHECK-NOT: cacheModifier = cg
    %0 = tt.load %ptr cacheModifier = ca : tensor<32x!tt.ptr<f32>>
    tt.return %0 : tensor<32xf32>
  }
}

// -----

// COM: Test m — forward-flow filter: X is a read-only arg, but its loaded
// COM: value is combined with DW (an RW arg) and stored back to DW. Even
// COM: though X itself is safe in isolation, the loaded data participates in
// COM: a cross-workgroup accumulation on DW. Filter 4 must block .cg on the
// COM: load of X. (Load of DW is also blocked, but by the RW-arg filter.)

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @value_flows_to_rw_store
  tt.func public @value_flows_to_rw_store(%X: tensor<32x!tt.ptr<f32>>,
                                          %DW: tensor<32x!tt.ptr<f32>>,
                                          %offs: tensor<32xi32>) {
    // CHECK: tt.load
    // CHECK-NOT: cacheModifier = cg
    %x = tt.load %X : tensor<32x!tt.ptr<f32>>
    %lp = tt.addptr %DW, %offs : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    // CHECK: tt.load
    // CHECK-NOT: cacheModifier = cg
    %cur = tt.load %lp : tensor<32x!tt.ptr<f32>>
    %sum = arith.addf %x, %cur : tensor<32xf32>
    %sp = tt.addptr %DW, %offs : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    tt.store %sp, %sum : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// COM: Test n — unresolved producer: the load pointer comes from an op the
// COM: pass does not whitelist (arith.select on pointer tensors). Root
// COM: resolution fails, so the load is conservatively treated as unsafe and
// COM: must NOT get .cg.

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @unresolved_producer_unsafe
  tt.func public @unresolved_producer_unsafe(%a: tensor<32x!tt.ptr<f32>>,
                                             %b: tensor<32x!tt.ptr<f32>>,
                                             %cond: i1) -> tensor<32xf32> {
    %p = arith.select %cond, %a, %b : tensor<32x!tt.ptr<f32>>
    // CHECK: tt.load
    // CHECK-NOT: cacheModifier = cg
    %0 = tt.load %p : tensor<32x!tt.ptr<f32>>
    tt.return %0 : tensor<32xf32>
  }
}

// -----

// COM: Test o — unresolved-store broad poison: a store's pointer cannot be
// COM: traced back to a known root (arith.select). Because the pass cannot
// COM: prove which args are written, it poisons every pointer-typed entry-block
// COM: arg, so the load of X (otherwise safe) must NOT get .cg.

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @unresolved_store_poisons_loads
  tt.func public @unresolved_store_poisons_loads(%X: tensor<32x!tt.ptr<f32>>,
                                                 %a: tensor<32x!tt.ptr<f32>>,
                                                 %b: tensor<32x!tt.ptr<f32>>,
                                                 %cond: i1,
                                                 %val: tensor<32xf32>) -> tensor<32xf32> {
    // CHECK: tt.load
    // CHECK-NOT: cacheModifier = cg
    %0 = tt.load %X : tensor<32x!tt.ptr<f32>>
    %sp = arith.select %cond, %a, %b : tensor<32x!tt.ptr<f32>>
    tt.store %sp, %val : tensor<32x!tt.ptr<f32>>
    tt.return %0 : tensor<32xf32>
  }
}

// -----

// COM: Test p — forward-flow through scf.if: loaded value from a safe arg
// COM: passes through an scf.if result and reaches a store on an unsafe (RW)
// COM: arg. The filter must follow yields across scf.if, not just scf.for.

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @value_flows_through_scf_if
  tt.func public @value_flows_through_scf_if(%X: tensor<32x!tt.ptr<f32>>,
                                             %DW: tensor<32x!tt.ptr<f32>>,
                                             %offs: tensor<32xi32>,
                                             %cond: i1) {
    %cst = arith.constant dense<0.0> : tensor<32xf32>
    // CHECK: tt.load
    // CHECK-NOT: cacheModifier = cg
    %x = tt.load %X : tensor<32x!tt.ptr<f32>>
    %res = scf.if %cond -> tensor<32xf32> {
      scf.yield %x : tensor<32xf32>
    } else {
      scf.yield %cst : tensor<32xf32>
    }
    %sp = tt.addptr %DW, %offs : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    tt.store %sp, %res : tensor<32x!tt.ptr<f32>>
    // second store on DW makes DW a RW arg so the above store is on an unsafe
    // ptr; without this the DW arg would only be stored (write-only), not RW.
    %lp = tt.addptr %DW, %offs : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %y = tt.load %lp : tensor<32x!tt.ptr<f32>>
    %sp2 = tt.addptr %DW, %offs : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    tt.store %sp2, %y : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// COM: Test q — atomic_cas in the function. The atomic-poison filter must
// COM: treat atomic_cas the same as atomic_rmw: every pointer arg becomes
// COM: unsafe, so the load of a read-only arg X does NOT get .cg.

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @atomic_cas_kernel_load_unchanged
  tt.func public @atomic_cas_kernel_load_unchanged(%X: tensor<32x!tt.ptr<f32>>,
                                                   %lock: !tt.ptr<i32>,
                                                   %cmp: i32,
                                                   %val: i32) -> tensor<32xf32> {
    // CHECK: tt.load
    // CHECK-NOT: cacheModifier = cg
    %0 = tt.load %X : tensor<32x!tt.ptr<f32>>
    %1 = tt.atomic_cas acq_rel, gpu, %lock, %cmp, %val : (!tt.ptr<i32>, i32, i32) -> i32
    tt.return %0 : tensor<32xf32>
  }
}
