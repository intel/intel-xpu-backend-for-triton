// RUN: env TRITON_INTEL_PREDICATED_STORE=1 triton-opt %s -split-input-file \
// RUN:   --convert-triton-intel-gpu-to-llvm | FileCheck %s --implicit-check-not=nontemporal

// COM: A masked `cg`/`cs` store on a target with predicated-I/O support lowers
// COM: to a two-armed block: a wide store on the fast arm (every lane of the
// COM: vector group in bounds) and per-element `triton_gen.predicated_store` ops
// COM: on the slow arm. Only the fast arm goes through getNonTemporalFlag(),
// COM: while the slow arm carries an explicit cache control, so this is the
// COM: configuration in which the two arms used to disagree: `nontemporal` (IGC
// COM: lowers it to LSC `.uc.uc`, i.e. L1UC_L3UC, which is `cv`'s policy) on the
// COM: fast arm against the requested control on the slow one.
// COM:
// COM: `--implicit-check-not=nontemporal` on the RUN line is the negative
// COM: regression this file carries: no store here may set the flag.

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttig.support_predicated_io} {
  // CHECK-LABEL: store_cg_masked_predicated
  tt.func @store_cg_masked_predicated(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %n: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128xf32, #blocked>
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %2 = tt.addptr %1, %0 : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %3 = tt.splat %n : i32 -> tensor<128xi32, #blocked>
    %4 = arith.cmpi slt, %0, %3 : tensor<128xi32, #blocked>
    tt.store %2, %cst, %4 cacheModifier = cg : tensor<128x!tt.ptr<f32>, #blocked>
    // CHECK: llvm.cond_br
    // COM: Fast arm: one wide unannotated store.
    // CHECK: llvm.store {{.*}} {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<1>
    // COM: Slow arm: one predicated store per element of the group, carrying the
    // COM: control `cg` actually asks for.
    // CHECK-COUNT-4: triton_gen.predicated_store {{.*}} {cache_control = L1UC_L3WB} : (!llvm.ptr<1>, f32, i1) -> ()
    // CHECK: llvm.return
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttig.support_predicated_io} {
  // CHECK-LABEL: store_cs_masked_predicated
  tt.func @store_cs_masked_predicated(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %n: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128xf32, #blocked>
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %2 = tt.addptr %1, %0 : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %3 = tt.splat %n : i32 -> tensor<128xi32, #blocked>
    %4 = arith.cmpi slt, %0, %3 : tensor<128xi32, #blocked>
    tt.store %2, %cst, %4 cacheModifier = cs : tensor<128x!tt.ptr<f32>, #blocked>
    // CHECK: llvm.cond_br
    // COM: Fast arm: one wide unannotated store. Dropping the flag here is an
    // COM: approximation, not an exact lowering -- a plain `llvm.store` has no
    // COM: way to express L1S_L3S. See the NOTE in getNonTemporalFlag().
    // CHECK: llvm.store {{.*}} {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<1>
    // COM: Slow arm: L1S_L3S is emitted exactly.
    // CHECK-COUNT-4: triton_gen.predicated_store {{.*}} {cache_control = L1S_L3S} : (!llvm.ptr<1>, f32, i1) -> ()
    // CHECK: llvm.return
    tt.return
  }
}
