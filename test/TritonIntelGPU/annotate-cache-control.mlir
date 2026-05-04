// RUN: env TRITON_INTEL_DISABLE_ANNOTATE_CACHE_CONTROL=0 triton-opt %s -split-input-file -tritonintelgpu-annotate-cache-control | FileCheck %s

// COM: Test a — load that does NOT feed a dot gets CG (cache = 2).
// COM: Store is intentionally left untouched.

#blocked1d = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @streaming_load_gets_cg
  tt.func public @streaming_load_gets_cg(%ptr: tensor<1024x!tt.ptr<f32>, #blocked1d>, %out: tensor<1024x!tt.ptr<f32>, #blocked1d>) {
    // CHECK: tt.load {{.*}} cacheModifier = cg
    %0 = tt.load %ptr : tensor<1024x!tt.ptr<f32>, #blocked1d>
    // CHECK: tt.store
    // CHECK-NOT: cacheModifier = cg
    tt.store %out, %0 : tensor<1024x!tt.ptr<f32>, #blocked1d>
    tt.return
  }
}

// -----

// COM: Test b — load that directly feeds a tt.dot stays NONE.

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @gemm_operand_load_unchanged
  tt.func public @gemm_operand_load_unchanged(%aptr: tensor<32x32x!tt.ptr<f16>, #dot_a>,
                                              %bptr: tensor<32x32x!tt.ptr<f16>, #dot_b>,
                                              %c: tensor<32x32xf32, #dpas>) -> tensor<32x32xf32, #dpas> {
    // CHECK: tt.load
    // CHECK-NOT: cacheModifier = cg
    %a = tt.load %aptr : tensor<32x32x!tt.ptr<f16>, #dot_a>
    // CHECK: tt.load
    // CHECK-NOT: cacheModifier = cg
    %b = tt.load %bptr : tensor<32x32x!tt.ptr<f16>, #dot_b>
    %d = tt.dot %a, %b, %c : tensor<32x32xf16, #dot_a> * tensor<32x32xf16, #dot_b> -> tensor<32x32xf32, #dpas>
    tt.return %d : tensor<32x32xf32, #dpas>
  }
}

// -----

// COM: Test c — load with #blocked encoding that is converted to #dot_a and
// COM: consumed by tt.dot GETS .cg. The blocked load issues non-overlapping
// COM: subgroup addresses (no cross-subgroup L1 reuse to preserve); the
// COM: layout conversion to dot operand happens in registers after the load.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @blocked_to_dot_via_convert_gets_cg
  tt.func public @blocked_to_dot_via_convert_gets_cg(%aptr: tensor<32x32x!tt.ptr<f16>, #blocked>,
                                                     %bptr: tensor<32x32x!tt.ptr<f16>, #dot_b>,
                                                     %c: tensor<32x32xf32, #dpas>) -> tensor<32x32xf32, #dpas> {
    // CHECK: tt.load {{.*}} cacheModifier = cg
    %a_blk = tt.load %aptr : tensor<32x32x!tt.ptr<f16>, #blocked>
    %a = ttg.convert_layout %a_blk : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #dot_a>
    // CHECK: tt.load
    // CHECK-NOT: cacheModifier = cg
    %b = tt.load %bptr : tensor<32x32x!tt.ptr<f16>, #dot_b>
    %d = tt.dot %a, %b, %c : tensor<32x32xf16, #dot_a> * tensor<32x32xf16, #dot_b> -> tensor<32x32xf32, #dpas>
    tt.return %d : tensor<32x32xf32, #dpas>
  }
}

// -----

// COM: Test d — store of a tt.dot result (GEMM epilogue) stays NONE.

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @gemm_epilogue_store_unchanged
  tt.func public @gemm_epilogue_store_unchanged(%aptr: tensor<32x32x!tt.ptr<f16>, #dot_a>,
                                                %bptr: tensor<32x32x!tt.ptr<f16>, #dot_b>,
                                                %cptr: tensor<32x32x!tt.ptr<f32>, #dpas>,
                                                %c: tensor<32x32xf32, #dpas>) {
    %a = tt.load %aptr : tensor<32x32x!tt.ptr<f16>, #dot_a>
    %b = tt.load %bptr : tensor<32x32x!tt.ptr<f16>, #dot_b>
    %d = tt.dot %a, %b, %c : tensor<32x32xf16, #dot_a> * tensor<32x32xf16, #dot_b> -> tensor<32x32xf32, #dpas>
    // CHECK: tt.store
    // CHECK-NOT: cacheModifier = cg
    tt.store %cptr, %d : tensor<32x32x!tt.ptr<f32>, #dpas>
    tt.return
  }
}

// -----

// COM: Test e — store of non-dot data is NOT annotated (stores are never
// COM: annotated; see pass docs).

#blocked1d = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @streaming_store_unchanged
  tt.func public @streaming_store_unchanged(%ptr: tensor<1024x!tt.ptr<f32>, #blocked1d>,
                                            %val: tensor<1024xf32, #blocked1d>) {
    // CHECK: tt.store
    // CHECK-NOT: cacheModifier = cg
    tt.store %ptr, %val : tensor<1024x!tt.ptr<f32>, #blocked1d>
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

#blocked1d = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @rw_arg_load_unchanged
  tt.func public @rw_arg_load_unchanged(%base: tensor<32x!tt.ptr<f32>, #blocked1d>,
                                        %offs: tensor<32xi32, #blocked1d>,
                                        %val: tensor<32xf32, #blocked1d>) {
    %p0 = tt.addptr %base, %offs : tensor<32x!tt.ptr<f32>, #blocked1d>, tensor<32xi32, #blocked1d>
    // CHECK: tt.load
    // CHECK-NOT: cacheModifier = cg
    %0 = tt.load %p0 : tensor<32x!tt.ptr<f32>, #blocked1d>
    %p1 = tt.addptr %base, %offs : tensor<32x!tt.ptr<f32>, #blocked1d>, tensor<32xi32, #blocked1d>
    tt.store %p1, %val : tensor<32x!tt.ptr<f32>, #blocked1d>
    tt.return
  }
}

// -----

// COM: Test h — atomic kernel: function contains tt.atomic_rmw on one arg;
// COM: a separate arg is only loaded. Cost-model rule: atomics signal complex
// COM: kernels where L1 reuse may matter, so all loads keep default.

#blocked1d = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @atomic_kernel_load_unchanged
  tt.func public @atomic_kernel_load_unchanged(%in: tensor<32x!tt.ptr<f32>, #blocked1d>,
                                               %acc: tensor<32x!tt.ptr<f32>, #blocked1d>,
                                               %v: tensor<32xf32, #blocked1d>,
                                               %mask: tensor<32xi1, #blocked1d>) -> tensor<32xf32, #blocked1d> {
    // CHECK: tt.load
    // CHECK-NOT: cacheModifier = cg
    %0 = tt.load %in : tensor<32x!tt.ptr<f32>, #blocked1d>
    %1 = tt.atomic_rmw fadd, acq_rel, gpu, %acc, %v, %mask : (tensor<32x!tt.ptr<f32>, #blocked1d>, tensor<32xf32, #blocked1d>, tensor<32xi1, #blocked1d>) -> tensor<32xf32, #blocked1d>
    tt.return %0 : tensor<32xf32, #blocked1d>
  }
}

// -----

// COM: Test i — RW arg reached through scf.for iter-arg: arg passed as
// COM: scf.for iter_arg, loaded inside the loop, stored-to elsewhere in the
// COM: same function. Load must NOT get .cg.

#blocked1d = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @rw_arg_via_scf_for_unchanged
  tt.func public @rw_arg_via_scf_for_unchanged(%base: tensor<32x!tt.ptr<f32>, #blocked1d>,
                                               %offs: tensor<32xi32, #blocked1d>,
                                               %val: tensor<32xf32, #blocked1d>,
                                               %lb: index, %ub: index, %step: index)
      -> tensor<32xf32> {
    %init = arith.constant dense<0.0> : tensor<32xf32>
    %res = scf.for %i = %lb to %ub step %step iter_args(%p = %base) -> tensor<32x!tt.ptr<f32>, #blocked1d> {
      // CHECK: tt.load
      // CHECK-NOT: cacheModifier = cg
      %0 = tt.load %p : tensor<32x!tt.ptr<f32>, #blocked1d>
      scf.yield %p : tensor<32x!tt.ptr<f32>, #blocked1d>
    }
    %sp = tt.addptr %base, %offs : tensor<32x!tt.ptr<f32>, #blocked1d>, tensor<32xi32, #blocked1d>
    tt.store %sp, %val : tensor<32x!tt.ptr<f32>, #blocked1d>
    tt.return %init : tensor<32xf32>
  }
}

// -----

// COM: Test j — mixed kernel: no atomics, two args. DW is RW (loaded and
// COM: stored), X is read-only. Load of X gets .cg; load of DW does not.

#blocked1d = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @mixed_args_selective_cg
  tt.func public @mixed_args_selective_cg(%DW: tensor<32x!tt.ptr<f32>, #blocked1d>,
                                          %X: tensor<32x!tt.ptr<f32>, #blocked1d>,
                                          %offs: tensor<32xi32, #blocked1d>,
                                          %val: tensor<32xf32, #blocked1d>) -> (tensor<32xf32, #blocked1d>, tensor<32xf32, #blocked1d>) {
    %pd = tt.addptr %DW, %offs : tensor<32x!tt.ptr<f32>, #blocked1d>, tensor<32xi32, #blocked1d>
    // CHECK: tt.load
    // CHECK-NOT: cacheModifier = cg
    %dw = tt.load %pd : tensor<32x!tt.ptr<f32>, #blocked1d>
    // CHECK: tt.load {{.*}} cacheModifier = cg
    %x = tt.load %X : tensor<32x!tt.ptr<f32>, #blocked1d>
    %sp = tt.addptr %DW, %offs : tensor<32x!tt.ptr<f32>, #blocked1d>, tensor<32xi32, #blocked1d>
    tt.store %sp, %val : tensor<32x!tt.ptr<f32>, #blocked1d>
    tt.return %dw, %x : tensor<32xf32, #blocked1d>, tensor<32xf32, #blocked1d>
  }
}

// -----

// COM: Test k — root peeling through tt.splat: scalar-ptr arg is splatted
// COM: and used for both load (via addptr chain) and store (via addptr chain),
// COM: making the root arg RW. Load must NOT get .cg.

#blocked1d = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @rw_arg_via_splat_unchanged
  tt.func public @rw_arg_via_splat_unchanged(%base: !tt.ptr<f32>,
                                             %offs: tensor<32xi32, #blocked1d>,
                                             %val: tensor<32xf32, #blocked1d>) -> tensor<32xf32, #blocked1d> {
    %splat_l = tt.splat %base : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #blocked1d>
    %lp = tt.addptr %splat_l, %offs : tensor<32x!tt.ptr<f32>, #blocked1d>, tensor<32xi32, #blocked1d>
    // CHECK: tt.load
    // CHECK-NOT: cacheModifier = cg
    %0 = tt.load %lp : tensor<32x!tt.ptr<f32>, #blocked1d>
    %splat_s = tt.splat %base : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #blocked1d>
    %sp = tt.addptr %splat_s, %offs : tensor<32x!tt.ptr<f32>, #blocked1d>, tensor<32xi32, #blocked1d>
    tt.store %sp, %val : tensor<32x!tt.ptr<f32>, #blocked1d>
    tt.return %0 : tensor<32xf32, #blocked1d>
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
// COM: value is combined with DW (an RW arg) and stored back to DW. The
// COM: forward-flow filter blocks .cg on the load of X to preserve L1
// COM: locality for the accumulation pattern. (Load of DW is also blocked,
// COM: but by the RW-arg filter.)

#blocked1d = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @value_flows_to_rw_store
  tt.func public @value_flows_to_rw_store(%X: tensor<32x!tt.ptr<f32>, #blocked1d>,
                                          %DW: tensor<32x!tt.ptr<f32>, #blocked1d>,
                                          %offs: tensor<32xi32, #blocked1d>) {
    // CHECK: tt.load
    // CHECK-NOT: cacheModifier = cg
    %x = tt.load %X : tensor<32x!tt.ptr<f32>, #blocked1d>
    %lp = tt.addptr %DW, %offs : tensor<32x!tt.ptr<f32>, #blocked1d>, tensor<32xi32, #blocked1d>
    // CHECK: tt.load
    // CHECK-NOT: cacheModifier = cg
    %cur = tt.load %lp : tensor<32x!tt.ptr<f32>, #blocked1d>
    %sum = arith.addf %x, %cur : tensor<32xf32, #blocked1d>
    %sp = tt.addptr %DW, %offs : tensor<32x!tt.ptr<f32>, #blocked1d>, tensor<32xi32, #blocked1d>
    tt.store %sp, %sum : tensor<32x!tt.ptr<f32>, #blocked1d>
    tt.return
  }
}

// -----

// COM: Test n — unresolved producer: the load pointer comes from an op the
// COM: pass does not whitelist (arith.select on pointer tensors). Root
// COM: resolution fails, so the load is conservatively skipped.

#blocked1d = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @unresolved_producer_skipped
  tt.func public @unresolved_producer_skipped(%a: tensor<32x!tt.ptr<f32>, #blocked1d>,
                                             %b: tensor<32x!tt.ptr<f32>, #blocked1d>,
                                             %cond: i1) -> tensor<32xf32, #blocked1d> {
    %p = arith.select %cond, %a, %b : tensor<32x!tt.ptr<f32>, #blocked1d>
    // CHECK: tt.load
    // CHECK-NOT: cacheModifier = cg
    %0 = tt.load %p : tensor<32x!tt.ptr<f32>, #blocked1d>
    tt.return %0 : tensor<32xf32, #blocked1d>
  }
}

// -----

// COM: Test o — unresolved-store exclusion: a store's pointer cannot be
// COM: traced back to a known root (arith.select). Because the pass cannot
// COM: prove which args are written, it excludes every pointer-typed
// COM: entry-block arg, so the load of X (otherwise eligible) must NOT get .cg.

#blocked1d = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @unresolved_store_excludes_loads
  tt.func public @unresolved_store_excludes_loads(%X: tensor<32x!tt.ptr<f32>, #blocked1d>,
                                                 %a: tensor<32x!tt.ptr<f32>, #blocked1d>,
                                                 %b: tensor<32x!tt.ptr<f32>, #blocked1d>,
                                                 %cond: i1,
                                                 %val: tensor<32xf32, #blocked1d>) -> tensor<32xf32, #blocked1d> {
    // CHECK: tt.load
    // CHECK-NOT: cacheModifier = cg
    %0 = tt.load %X : tensor<32x!tt.ptr<f32>, #blocked1d>
    %sp = arith.select %cond, %a, %b : tensor<32x!tt.ptr<f32>, #blocked1d>
    tt.store %sp, %val : tensor<32x!tt.ptr<f32>, #blocked1d>
    tt.return %0 : tensor<32xf32, #blocked1d>
  }
}

// -----

// COM: Test p — forward-flow through scf.if: loaded value from a read-only
// COM: arg passes through an scf.if result and reaches a store on an excluded
// COM: (RW) arg. The filter must follow yields across scf.if, not just scf.for.

#blocked1d = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @value_flows_through_scf_if
  tt.func public @value_flows_through_scf_if(%X: tensor<32x!tt.ptr<f32>, #blocked1d>,
                                             %DW: tensor<32x!tt.ptr<f32>, #blocked1d>,
                                             %offs: tensor<32xi32, #blocked1d>,
                                             %cond: i1) {
    %cst = arith.constant dense<0.0> : tensor<32xf32, #blocked1d>
    // CHECK: tt.load
    // CHECK-NOT: cacheModifier = cg
    %x = tt.load %X : tensor<32x!tt.ptr<f32>, #blocked1d>
    %res = scf.if %cond -> tensor<32xf32, #blocked1d> {
      scf.yield %x : tensor<32xf32, #blocked1d>
    } else {
      scf.yield %cst : tensor<32xf32, #blocked1d>
    }
    %sp = tt.addptr %DW, %offs : tensor<32x!tt.ptr<f32>, #blocked1d>, tensor<32xi32, #blocked1d>
    tt.store %sp, %res : tensor<32x!tt.ptr<f32>, #blocked1d>
    // second store on DW makes DW a RW arg so the above store is on an excluded
    // ptr; without this the DW arg would only be stored (write-only), not RW.
    %lp = tt.addptr %DW, %offs : tensor<32x!tt.ptr<f32>, #blocked1d>, tensor<32xi32, #blocked1d>
    %y = tt.load %lp : tensor<32x!tt.ptr<f32>, #blocked1d>
    %sp2 = tt.addptr %DW, %offs : tensor<32x!tt.ptr<f32>, #blocked1d>, tensor<32xi32, #blocked1d>
    tt.store %sp2, %y : tensor<32x!tt.ptr<f32>, #blocked1d>
    tt.return
  }
}

// -----

// COM: Test q — atomic_cas in the function. The atomic cost-model filter
// COM: treats atomic_cas the same as atomic_rmw: every pointer arg is
// COM: excluded, so the load of a read-only arg X does NOT get .cg.

#blocked1d = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @atomic_cas_kernel_load_unchanged
  tt.func public @atomic_cas_kernel_load_unchanged(%X: tensor<32x!tt.ptr<f32>, #blocked1d>,
                                                   %lock: !tt.ptr<i32>,
                                                   %cmp: i32,
                                                   %val: i32) -> tensor<32xf32, #blocked1d> {
    // CHECK: tt.load
    // CHECK-NOT: cacheModifier = cg
    %0 = tt.load %X : tensor<32x!tt.ptr<f32>, #blocked1d>
    %1 = tt.atomic_cas acq_rel, gpu, %lock, %cmp, %val : (!tt.ptr<i32>, i32, i32) -> i32
    tt.return %0 : tensor<32xf32, #blocked1d>
  }
}

// -----

// COM: Test r — scf.while forward flow: a loaded value from a safe arg X is
// COM: threaded through an scf.while before-region arg and after-region yield
// COM: back into the before region, then stored to an RW arg DW. The
// COM: forward-flow filter must propagate through scf.while init, condition,
// COM: and after-region yield so the load of X does NOT get .cg.

#blocked1d = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @value_flows_through_scf_while
  tt.func public @value_flows_through_scf_while(%X: tensor<32x!tt.ptr<f32>, #blocked1d>,
                                                %DW: tensor<32x!tt.ptr<f32>, #blocked1d>,
                                                %offs: tensor<32xi32, #blocked1d>,
                                                %n: i32) {
    // CHECK: tt.load
    // CHECK-NOT: cacheModifier = cg
    %x = tt.load %X : tensor<32x!tt.ptr<f32>, #blocked1d>
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %res:2 = scf.while (%iv = %c0, %acc = %x) : (i32, tensor<32xf32, #blocked1d>) -> (i32, tensor<32xf32, #blocked1d>) {
      %cmp = arith.cmpi slt, %iv, %n : i32
      scf.condition(%cmp) %iv, %acc : i32, tensor<32xf32, #blocked1d>
    } do {
    ^bb0(%iv1: i32, %acc1: tensor<32xf32, #blocked1d>):
      %next = arith.addi %iv1, %c1 : i32
      scf.yield %next, %acc1 : i32, tensor<32xf32, #blocked1d>
    }
    // The while's 2nd result carries %x into this store on the RW arg DW.
    %sp = tt.addptr %DW, %offs : tensor<32x!tt.ptr<f32>, #blocked1d>, tensor<32xi32, #blocked1d>
    tt.store %sp, %res#1 : tensor<32x!tt.ptr<f32>, #blocked1d>
    // Second load/store on DW makes DW an RW arg for the skip-arg analysis.
    %lp = tt.addptr %DW, %offs : tensor<32x!tt.ptr<f32>, #blocked1d>, tensor<32xi32, #blocked1d>
    %y = tt.load %lp : tensor<32x!tt.ptr<f32>, #blocked1d>
    %sp2 = tt.addptr %DW, %offs : tensor<32x!tt.ptr<f32>, #blocked1d>, tensor<32xi32, #blocked1d>
    tt.store %sp2, %y : tensor<32x!tt.ptr<f32>, #blocked1d>
    tt.return
  }
}

// -----

// COM: Test s — dot_op-encoded load whose value is threaded through an scf.while
// COM: before being consumed by tt.dot. The scf.while is incidental — the
// COM: encoding gate already blocks .cg on the load.

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @scf_while_feeds_dot_unchanged
  tt.func public @scf_while_feeds_dot_unchanged(%aptr: tensor<32x32x!tt.ptr<f16>, #dot_a>,
                                                %bptr: tensor<32x32x!tt.ptr<f16>, #dot_b>,
                                                %c: tensor<32x32xf32, #dpas>,
                                                %n: i32)
      -> tensor<32x32xf32, #dpas> {
    // CHECK: tt.load
    // CHECK-NOT: cacheModifier = cg
    %a = tt.load %aptr : tensor<32x32x!tt.ptr<f16>, #dot_a>
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %res:2 = scf.while (%iv = %c0, %acc = %c) : (i32, tensor<32x32xf32, #dpas>) -> (i32, tensor<32x32xf32, #dpas>) {
      %cmp = arith.cmpi slt, %iv, %n : i32
      scf.condition(%cmp) %iv, %acc : i32, tensor<32x32xf32, #dpas>
    } do {
    ^bb0(%iv1: i32, %acc1: tensor<32x32xf32, #dpas>):
      %b = tt.load %bptr : tensor<32x32x!tt.ptr<f16>, #dot_b>
      %d = tt.dot %a, %b, %acc1 : tensor<32x32xf16, #dot_a> * tensor<32x32xf16, #dot_b> -> tensor<32x32xf32, #dpas>
      %next = arith.addi %iv1, %c1 : i32
      scf.yield %next, %d : i32, tensor<32x32xf32, #dpas>
    }
    tt.return %res#1 : tensor<32x32xf32, #dpas>
  }
}

// -----

// COM: Test t — nested slice encodings: load with Slice<Slice<DotOperandEncoding>>
// COM: to exercise the while-loop unwrap in skipForL1Reuse. Two nested SliceEncodingAttr
// COM: layers are unwrapped to reveal the underlying DotOperandEncodingAttr backed by DPAS,
// COM: confirming that .cg annotation is correctly blocked for doubly-sliced dot operands.

#dpas_3d = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4, 2], repCluster = [1, 1, 1], A = [1, 8, 16], B = [1, 16, 16], C = [1, 8, 16]}>
#dot_a_3d = #ttg.dot_op<{opIdx = 0, parent = #dpas_3d, kWidth = 1}>
#slice_2d = #ttg.slice<{dim = 0, parent = #dot_a_3d}>
#slice_1d = #ttg.slice<{dim = 0, parent = #slice_2d}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @nested_slice_over_dot_operand
  tt.func public @nested_slice_over_dot_operand(%p: tensor<16x!tt.ptr<f16>, #slice_1d>) {
    // CHECK: tt.load
    // CHECK-NOT: cacheModifier = cg
    %0 = tt.load %p : tensor<16x!tt.ptr<f16>, #slice_1d>
    tt.return
  }
}
