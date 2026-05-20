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

// COM: Test b — load that directly feeds a tt.dot gets evict_last (spatial known reuse on dot-operand encoding).
// COM: warpsPerCTA = [2, 2]: both A and B operands have warp-broadcast factor 2,
// COM: passing the Phase 3 `factor >= 2` gate. (warpsPerCTA = [4, 1] would
// COM: rejected because the A operand would have broadcast factor 1.)

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @gemm_operand_load_evict_last
  tt.func public @gemm_operand_load_evict_last(%aptr: tensor<32x32x!tt.ptr<f16>, #dot_a>,
                                               %bptr: tensor<32x32x!tt.ptr<f16>, #dot_b>,
                                               %c: tensor<32x32xf32, #dpas>) -> tensor<32x32xf32, #dpas> {
    // CHECK: tt.load
    // CHECK-NOT: cacheModifier = cg
    // CHECK-SAME: evictionPolicy = evict_last
    %a = tt.load %aptr : tensor<32x32x!tt.ptr<f16>, #dot_a>
    // CHECK: tt.load
    // CHECK-NOT: cacheModifier = cg
    // CHECK-SAME: evictionPolicy = evict_last
    %b = tt.load %bptr : tensor<32x32x!tt.ptr<f16>, #dot_b>
    %d = tt.dot %a, %b, %c : tensor<32x32xf16, #dot_a> * tensor<32x32xf16, #dot_b> -> tensor<32x32xf32, #dpas>
    tt.return %d : tensor<32x32xf32, #dpas>
  }
}

// COM: Test b2 — encoded fallback regression guard (API split).
// COM:
// COM: The intended lit case was an encoded non-power-of-2 load
// COM: (e.g. tensor<24x32xf32> with a blocked encoding) where spatial
// COM: anyReuse falls back to true but knownCrossSubgroupReuse returns
// COM: false, so the load gets neither .cg nor evict_last. MLIR's tensor
// COM: verifier rejects non-power-of-2 element counts under blocked
// COM: encodings before AnnotateCacheControl can run, so this case
// COM: cannot be expressed at the lit level. Coverage is moved to the
// COM: analysis-level unit test:
// COM:   ReuseAnalysisTest::Known_FalseFromConservativeFallback_NonPow2
// COM: which is the central regression test for the API split.

// -----

// COM: Test c — load with #blocked encoding that is converted to #dot_a and
// COM: consumed by tt.dot. warpsPerCTA = [1, 4] on the blocked encoding,
// COM: warpsPerCTA = [2, 2] on the dpas. Both loads pass the Phase 3
// COM: factor>=2 gate:
// COM:   - %a_blk: blocked tile per warp = sizePerThread × threadsPerWarp
// COM:     = [1, 16]. With warpsPerCTA = [1, 4] the per-CTA tile is
// COM:     [1, 64], which is *wider than the tensor* (32 cols). Two
// COM:     warps wrap onto each column band → broadcast factor 2 on the
// COM:     LinearLayout (a warp basis is all-zero across out-dims).
// COM:     This is real cross-warp reuse, correctly accepted.
// COM:   - %b: dot_b on dpas#warpsPerCTA=[2,2]; warp basis tiling M is
// COM:     all-zero on B's [K, N] axes. Factor 2.
// COM:
// COM: The Phase 3 gate is *layout-driven*: it inspects the LinearLayout
// COM: directly rather than reading warpsPerCTA heuristically, so
// COM: tile-overflow broadcast (warp tile larger than tensor) is detected
// COM: where a naive Wn-only check would miss it.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @blocked_to_dot_via_convert_warp_broadcast_evict_last
  tt.func public @blocked_to_dot_via_convert_warp_broadcast_evict_last(%aptr: tensor<32x32x!tt.ptr<f16>, #blocked>,
                                                                       %bptr: tensor<32x32x!tt.ptr<f16>, #dot_b>,
                                                                       %c: tensor<32x32xf32, #dpas>) -> tensor<32x32xf32, #dpas> {
    // CHECK: tt.load
    // CHECK-NOT: cacheModifier = cg
    // CHECK-SAME: evictionPolicy = evict_last
    %a_blk = tt.load %aptr : tensor<32x32x!tt.ptr<f16>, #blocked>
    %a = ttg.convert_layout %a_blk : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #dot_a>
    // CHECK: tt.load
    // CHECK-NOT: cacheModifier = cg
    // CHECK-SAME: evictionPolicy = evict_last
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

// COM: Test l2 — user-specified eviction policy (evict_first / evict_last)
// COM: must NOT be overridden. The lowering maps these to precise LSC cache
// COM: modes; stamping .cg here would lose the user's intent.

#blocked1d = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @user_evict_first_preserved
  tt.func public @user_evict_first_preserved(%ptr: tensor<1024x!tt.ptr<f32>, #blocked1d>) -> tensor<1024xf32, #blocked1d> {
    // CHECK: tt.load
    // CHECK-NOT: cacheModifier = cg
    %0 = tt.load %ptr evictionPolicy = evict_first : tensor<1024x!tt.ptr<f32>, #blocked1d>
    tt.return %0 : tensor<1024xf32, #blocked1d>
  }
}

// -----

// COM: Test l3 — evict_last is also honored: pass must not stamp .cg.

#blocked1d = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @user_evict_last_preserved
  tt.func public @user_evict_last_preserved(%ptr: tensor<1024x!tt.ptr<f32>, #blocked1d>) -> tensor<1024xf32, #blocked1d> {
    // CHECK: tt.load
    // CHECK-NOT: cacheModifier = cg
    %0 = tt.load %ptr evictionPolicy = evict_last : tensor<1024x!tt.ptr<f32>, #blocked1d>
    tt.return %0 : tensor<1024xf32, #blocked1d>
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

// -----

// COM: Test u — KV-cache decode regression guard (#6809). Pointer is defined
// COM: outside the scf.for: TemporalReuseAnalysis reports Invariant → reuse on
// COM: the enclosing loop. Load must NOT get .cg even though it is read-only
// COM: and does not feed a dot.

#blocked1d = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @kv_cache_decode_loop_invariant
  tt.func public @kv_cache_decode_loop_invariant(%kv: tensor<32x!tt.ptr<f32>, #blocked1d>,
                                                 %out: tensor<32x!tt.ptr<f32>, #blocked1d>,
                                                 %lb: i32, %ub: i32, %step: i32) {
    scf.for %q = %lb to %ub step %step : i32 {
      // CHECK: tt.load
      // CHECK-NOT: cacheModifier = cg
      %v = tt.load %kv : tensor<32x!tt.ptr<f32>, #blocked1d>
      tt.store %out, %v : tensor<32x!tt.ptr<f32>, #blocked1d>
    }
    tt.return
  }
}

// -----

// COM: Test w — pure streaming in a loop: the pointer advances by exactly
// COM: the tile extent (32) per iteration, so successive tiles are disjoint.
// COM: TemporalReuseAnalysis reports Streaming on the enclosing loop → no
// COM: temporal reuse. Load gets .cg.

#blocked1d = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @streaming_loop_gets_cg
  tt.func public @streaming_loop_gets_cg(%in: !tt.ptr<f32>,
                                         %out: !tt.ptr<f32>,
                                         %lb: i32, %ub: i32) {
    %c32 = arith.constant 32 : i32
    %r = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #blocked1d>
    %s = tt.splat %in : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #blocked1d>
    %sout = tt.splat %out : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #blocked1d>
    %p0 = tt.addptr %s, %r : tensor<32x!tt.ptr<f32>, #blocked1d>, tensor<32xi32, #blocked1d>
    %po0 = tt.addptr %sout, %r : tensor<32x!tt.ptr<f32>, #blocked1d>, tensor<32xi32, #blocked1d>
    scf.for %i = %lb to %ub step %c32 : i32 {
      %iv = tt.splat %i : i32 -> tensor<32xi32, #blocked1d>
      %pp = tt.addptr %p0, %iv : tensor<32x!tt.ptr<f32>, #blocked1d>, tensor<32xi32, #blocked1d>
      %poo = tt.addptr %po0, %iv : tensor<32x!tt.ptr<f32>, #blocked1d>, tensor<32xi32, #blocked1d>
      // CHECK: tt.load {{.*}} cacheModifier = cg
      %v = tt.load %pp : tensor<32x!tt.ptr<f32>, #blocked1d>
      tt.store %poo, %v : tensor<32x!tt.ptr<f32>, #blocked1d>
    }
    tt.return
  }
}

// -----

// COM: Test x — two read-only loads of the same arg at different offsets:
// COM: they alias each other (per AliasAnalysisTest TwoLoadsSameArgDifferentOffsets)
// COM: but no peer has a write effect, so `aliasesWritingPeer` returns false
// COM: for both. Both loads get .cg.

#blocked1d = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @two_readonly_loads_same_arg
  tt.func public @two_readonly_loads_same_arg(%X: tensor<32x!tt.ptr<f32>, #blocked1d>,
                                              %off0: tensor<32xi32, #blocked1d>,
                                              %off1: tensor<32xi32, #blocked1d>)
      -> (tensor<32xf32, #blocked1d>, tensor<32xf32, #blocked1d>) {
    %p0 = tt.addptr %X, %off0 : tensor<32x!tt.ptr<f32>, #blocked1d>, tensor<32xi32, #blocked1d>
    %p1 = tt.addptr %X, %off1 : tensor<32x!tt.ptr<f32>, #blocked1d>, tensor<32xi32, #blocked1d>
    // CHECK: tt.load {{.*}} cacheModifier = cg
    %a = tt.load %p0 : tensor<32x!tt.ptr<f32>, #blocked1d>
    // CHECK: tt.load {{.*}} cacheModifier = cg
    %b = tt.load %p1 : tensor<32x!tt.ptr<f32>, #blocked1d>
    tt.return %a, %b : tensor<32xf32, #blocked1d>, tensor<32xf32, #blocked1d>
  }
}

// -----

// COM: Test y — scale-operand LinearEncodingAttr load. Regression guard for
// COM: deletion of the use-site `tt.dot_scaled` scan in the old pass: after
// COM: deletion, the ONLY mechanism that keeps `.cg` off the scale load is
// COM: SpatialReuseAnalysis structurally catching the scale encoding's
// COM: warp-invariant basis (zero warp dim in the LinearLayout). If this test
// COM: ever regresses, `BlockScaledDPAStoLinearLayout`'s warp basis for the
// COM: scale encoding is wrong — not the pass.
// COM:
// COM: The `warp = [[0, 0], [0, 0]]` basis below encodes "every warp reads
// COM: the same coordinates": P1 reports cross-subgroup reuse and the load
// COM: is left with the default cache modifier.

#ll_scale = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4]], lane = [[1, 0], [2, 0], [4, 0], [8, 0]], warp = [[0, 0], [0, 0]], block = []}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @scale_encoding_load_unchanged
  tt.func public @scale_encoding_load_unchanged(%sptr: tensor<16x8x!tt.ptr<i8>, #ll_scale>)
      -> tensor<16x8xi8, #ll_scale> {
    // CHECK: tt.load
    // CHECK-NOT: cacheModifier = cg
    %s = tt.load %sptr : tensor<16x8x!tt.ptr<i8>, #ll_scale>
    tt.return %s : tensor<16x8xi8, #ll_scale>
  }
}

// -----

// COM: Test h — known cross-subgroup reuse on a load that does NOT
// COM: reach a tt.dot. shouldUseEvictLast rejects (requirement 2).
// COM: Gate 3 still suppresses .cg.

#blocked2d_h = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @known_reuse_no_dot_consumer
  tt.func public @known_reuse_no_dot_consumer(%ptr: tensor<32x32x!tt.ptr<f32>, #blocked2d_h>,
                                              %out: tensor<32x32x!tt.ptr<f32>, #blocked2d_h>) {
    // CHECK: tt.load
    // CHECK-NOT: cacheModifier = cg
    // CHECK-NOT: evictionPolicy = evict_last
    %0 = tt.load %ptr : tensor<32x32x!tt.ptr<f32>, #blocked2d_h>
    tt.store %out, %0 : tensor<32x32x!tt.ptr<f32>, #blocked2d_h>
    tt.return
  }
}

// -----

// COM: Test i — temporal-only proof, no spatial proof, dot-fed.
// COM: shouldUseEvictLast rejects (requirement 1: spatial-only).
// COM: Documents the conservative scope of the first land.

#blocked2d_i = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 4], warpsPerCTA = [2, 2], order = [1, 0]}>
#dpas_i = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot_a_i = #ttg.dot_op<{opIdx = 0, parent = #dpas_i, kWidth = 1}>
#dot_b_i = #ttg.dot_op<{opIdx = 1, parent = #dpas_i, kWidth = 2}>
// COM: B operand is passed as a function arg (not loaded inside the loop) so
// COM: the only in-loop load is the temporal-only candidate. Operand-B loads
// COM: with parent dpas warpsPerCTA=[4,1] would otherwise have spatial known
// COM: reuse on N and get evict_last, which is correct behavior but would
// COM: confuse FileCheck-NOT scoping inside the loop body.
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @temporal_only_dot_load_no_promote
  tt.func public @temporal_only_dot_load_no_promote(%aptr: tensor<32x32x!tt.ptr<f16>, #blocked2d_i>,
                                                    %b: tensor<32x32xf16, #dot_b_i>,
                                                    %c_init: tensor<32x32xf32, #dpas_i>,
                                                    %lb: index, %ub: index, %step: index) -> tensor<32x32xf32, #dpas_i> {
    %result = scf.for %i = %lb to %ub step %step iter_args(%c = %c_init) -> tensor<32x32xf32, #dpas_i> {
      // CHECK: tt.load
      // CHECK-NOT: cacheModifier = cg
      // CHECK-NOT: evictionPolicy = evict_last
      %a_blk = tt.load %aptr : tensor<32x32x!tt.ptr<f16>, #blocked2d_i>
      %a = ttg.convert_layout %a_blk : tensor<32x32xf16, #blocked2d_i> -> tensor<32x32xf16, #dot_a_i>
      %d = tt.dot %a, %b, %c : tensor<32x32xf16, #dot_a_i> * tensor<32x32xf16, #dot_b_i> -> tensor<32x32xf32, #dpas_i>
      scf.yield %d : tensor<32x32xf32, #dpas_i>
    }
    tt.return %result : tensor<32x32xf32, #dpas_i>
  }
}

// -----

// COM: Test j — budget overflow: first dot-fed load fits, second
// COM: exceeds the per-loop running total. First gets evict_last,
// COM: second stays at default (Gate 3 still suppresses .cg).
// COM: warpsPerCTA = [2, 2]: both A and B operands have factor 2, so both
// COM: pass the new factor>=2 gate. The budget gate is what decides which
// COM: gets evict_last (account-as-you-go: A admitted, B exceeds).

#dpas_j = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot_a_j = #ttg.dot_op<{opIdx = 0, parent = #dpas_j, kWidth = 1}>
#dot_b_j = #ttg.dot_op<{opIdx = 1, parent = #dpas_j, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @budget_overflow_first_fits_second_doesnt
  tt.func public @budget_overflow_first_fits_second_doesnt(%aptr: tensor<128x128x!tt.ptr<f16>, #dot_a_j>,
                                                           %bptr: tensor<128x128x!tt.ptr<f16>, #dot_b_j>,
                                                           %c_init: tensor<128x128xf32, #dpas_j>,
                                                           %lb: index, %ub: index, %step: index) -> tensor<128x128xf32, #dpas_j> {
    // CHECK: scf.for
    %r = scf.for %i = %lb to %ub step %step iter_args(%c = %c_init) -> tensor<128x128xf32, #dpas_j> {
      // CHECK: tt.load
      // CHECK-NOT: cacheModifier = cg
      // CHECK-SAME: evictionPolicy = evict_last
      %a = tt.load %aptr : tensor<128x128x!tt.ptr<f16>, #dot_a_j>
      // CHECK: tt.load
      // CHECK-NOT: cacheModifier = cg
      // CHECK-NOT: evictionPolicy = evict_last
      %b = tt.load %bptr : tensor<128x128x!tt.ptr<f16>, #dot_b_j>
      %d = tt.dot %a, %b, %c : tensor<128x128xf16, #dot_a_j> * tensor<128x128xf16, #dot_b_j> -> tensor<128x128xf32, #dpas_j>
      scf.yield %d : tensor<128x128xf32, #dpas_j>
    }
    tt.return %r : tensor<128x128xf32, #dpas_j>
  }
}

// -----

// COM: Test k — single tile exceeds per-load budget cap. Rejected.

#dpas_k = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot_a_k = #ttg.dot_op<{opIdx = 0, parent = #dpas_k, kWidth = 1}>
#dot_b_k = #ttg.dot_op<{opIdx = 1, parent = #dpas_k, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @per_load_budget_exceeded
  tt.func public @per_load_budget_exceeded(%aptr: tensor<128x256x!tt.ptr<f16>, #dot_a_k>,
                                           %bptr: tensor<256x128x!tt.ptr<f16>, #dot_b_k>,
                                           %c: tensor<128x128xf32, #dpas_k>) -> tensor<128x128xf32, #dpas_k> {
    // CHECK: tt.load
    // CHECK-NOT: cacheModifier = cg
    // CHECK-NOT: evictionPolicy = evict_last
    %a = tt.load %aptr : tensor<128x256x!tt.ptr<f16>, #dot_a_k>
    %b = tt.load %bptr : tensor<256x128x!tt.ptr<f16>, #dot_b_k>
    %d = tt.dot %a, %b, %c : tensor<128x256xf16, #dot_a_k> * tensor<256x128xf16, #dot_b_k> -> tensor<128x128xf32, #dpas_k>
    tt.return %d : tensor<128x128xf32, #dpas_k>
  }
}

// -----

// COM: Test l — `ttig.prefetch` on the same pointer as a dot-fed load must
// COM: NOT block evict_last promotion. ttig.prefetch declares
// COM: `MemWrite<L2Cache>` purely to keep the optimizer from CSE/DCE-ing it;
// COM: it does not mutate observable memory. Treating it as a writing peer
// COM: (the prior bug) blocked Phase 2 promotion in the canonical Pipeline
// COM: shape: pre-prefetch + in-loop prefetch + load on the same pointer.

// COM: warpsPerCTA = [2, 2]: both operands have factor 2, satisfying the
// COM: factor>=2 gate. (warpsPerCTA = [4, 1] would reject the A operand.)

#dpas_l = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot_a_l = #ttg.dot_op<{opIdx = 0, parent = #dpas_l, kWidth = 1}>
#dot_b_l = #ttg.dot_op<{opIdx = 1, parent = #dpas_l, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @prefetch_same_ptr_does_not_block_evict_last
  tt.func public @prefetch_same_ptr_does_not_block_evict_last(%aptr: tensor<32x32x!tt.ptr<f16>, #dot_a_l>,
                                                              %bptr: tensor<32x32x!tt.ptr<f16>, #dot_b_l>,
                                                              %c: tensor<32x32xf32, #dpas_l>) -> tensor<32x32xf32, #dpas_l> {
    ttig.prefetch %aptr {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x32x!tt.ptr<f16>, #dot_a_l>
    ttig.prefetch %bptr {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x32x!tt.ptr<f16>, #dot_b_l>
    // CHECK: tt.load
    // CHECK-NOT: cacheModifier = cg
    // CHECK-SAME: evictionPolicy = evict_last
    %a = tt.load %aptr : tensor<32x32x!tt.ptr<f16>, #dot_a_l>
    // CHECK: tt.load
    // CHECK-NOT: cacheModifier = cg
    // CHECK-SAME: evictionPolicy = evict_last
    %b = tt.load %bptr : tensor<32x32x!tt.ptr<f16>, #dot_b_l>
    %d = tt.dot %a, %b, %c : tensor<32x32xf16, #dot_a_l> * tensor<32x32xf16, #dot_b_l> -> tensor<32x32xf32, #dpas_l>
    tt.return %d : tensor<32x32xf32, #dpas_l>
  }
}

// -----

// COM: Test m — Phase 3 broadcast-factor gate. With warpsPerCTA = [4, 1]
// COM: (Wm = 4, Wn = 1), the A operand is loaded into a [M, K] tile that
// COM: warps strictly partition by M (no warp shares any row of A). Even
// COM: though the K axis is "warp-invariant" structurally, the broadcast
// COM: factor is 1 — no real cross-warp reuse — and the new gate must
// COM: REJECT promotion. The B operand has factor 4 (warp basis tiles M
// COM: which doesn't appear on B) and is correctly promoted. This is the
// COM: canonical V.5b over-promotion case.
// COM:
// COM: Reference: .claude/reference/v5b-over-promotion-finding-2026-05-20.md

#dpas_m = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot_a_m = #ttg.dot_op<{opIdx = 0, parent = #dpas_m, kWidth = 1}>
#dot_b_m = #ttg.dot_op<{opIdx = 1, parent = #dpas_m, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @factor_one_rejects_evict_last_factor_n_accepts
  tt.func public @factor_one_rejects_evict_last_factor_n_accepts(%aptr: tensor<32x32x!tt.ptr<f16>, #dot_a_m>,
                                                                 %bptr: tensor<32x32x!tt.ptr<f16>, #dot_b_m>,
                                                                 %c: tensor<32x32xf32, #dpas_m>) -> tensor<32x32xf32, #dpas_m> {
    // A operand: factor = 1 → rejected. Reuse-suspected branch returns
    // false → no annotation set; load stays at default cache + default eviction.
    // CHECK: tt.load
    // CHECK-NOT: cacheModifier = cg
    // CHECK-NOT: evictionPolicy = evict_last
    %a = tt.load %aptr : tensor<32x32x!tt.ptr<f16>, #dot_a_m>
    // B operand: factor = 4 → accepted, promoted to evict_last.
    // CHECK: tt.load
    // CHECK-NOT: cacheModifier = cg
    // CHECK-SAME: evictionPolicy = evict_last
    %b = tt.load %bptr : tensor<32x32x!tt.ptr<f16>, #dot_b_m>
    %d = tt.dot %a, %b, %c : tensor<32x32xf16, #dot_a_m> * tensor<32x32xf16, #dot_b_m> -> tensor<32x32xf32, #dpas_m>
    tt.return %d : tensor<32x32xf32, #dpas_m>
  }
}
