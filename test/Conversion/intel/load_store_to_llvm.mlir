// RUN: env TRITON_INTEL_PREDICATED_LOAD=1 TRITON_INTEL_PREDICATED_STORE=1  triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm | FileCheck %s

// COM: Baseline 128-bit vectorization — ttig.support_256b_load_store is NOT set on the module. See load_store_256b_to_llvm.mlir for the widened variant.

#blocked0 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: global_load_with_attributes
  tt.func @global_load_with_attributes(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
    %3 = tt.splat %1 : i32 -> tensor<256xi32, #blocked0>
    %4 = arith.addi %3, %2 : tensor<256xi32, #blocked0>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %6 = tt.addptr %5, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    %9 = tt.load %6 {isVolatile = true} : tensor<256x!tt.ptr<f32>, #blocked0>
    %10 = tt.load %6 cacheModifier = ca : tensor<256x!tt.ptr<f32>, #blocked0>
    %12 = tt.load %6 cacheModifier = cg : tensor<256x!tt.ptr<f32>, #blocked0>
    %13 = tt.load %6 cacheModifier = wb : tensor<256x!tt.ptr<f32>, #blocked0>
    %14 = tt.load %6 cacheModifier = cs : tensor<256x!tt.ptr<f32>, #blocked0>
    %15 = tt.load %6 cacheModifier = wt : tensor<256x!tt.ptr<f32>, #blocked0>
    %16 = tt.load %6 cacheModifier = cv : tensor<256x!tt.ptr<f32>, #blocked0>
    // CHECK-COUNT-2: llvm.load volatile {{.*}} {alignment = 16 : i64} : !llvm.ptr<1> -> vector<4xi32>
    // CHECK-COUNT-2: llvm.load {{.*}} {alignment = 16 : i64} : !llvm.ptr<1> -> vector<4xi32>
    // CHECK-COUNT-2: llvm.load {{.*}} {alignment = 16 : i64, nontemporal} : !llvm.ptr<1> -> vector<4xi32>
    // CHECK-COUNT-2: llvm.load {{.*}} {alignment = 16 : i64} : !llvm.ptr<1> -> vector<4xi32>
    // CHECK-COUNT-2: llvm.load {{.*}} {alignment = 16 : i64, nontemporal} : !llvm.ptr<1> -> vector<4xi32>
    // CHECK-COUNT-2: llvm.load {{.*}} {alignment = 16 : i64} : !llvm.ptr<1> -> vector<4xi32>
    // CHECK-COUNT-2: llvm.load {{.*}} {alignment = 16 : i64, nontemporal} : !llvm.ptr<1> -> vector<4xi32>
    tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: global_store_with_attributes
  tt.func @global_store_with_attributes(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<256xf32, #blocked0>
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
    %3 = tt.splat %1 : i32 -> tensor<256xi32, #blocked0>
    %4 = arith.addi %3, %2 : tensor<256xi32, #blocked0>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %6 = tt.addptr %5, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    tt.store %6, %cst : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.store %6, %cst cacheModifier = ca : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.store %6, %cst cacheModifier = cg : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.store %6, %cst cacheModifier = wb : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.store %6, %cst cacheModifier = cs : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.store %6, %cst cacheModifier = wt : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.store %6, %cst cacheModifier = cv : tensor<256x!tt.ptr<f32>, #blocked0>
    // CHECK-COUNT-2: llvm.store {{.*}} {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<1>
    // CHECK-COUNT-2: llvm.store {{.*}} {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<1>
    // CHECK-COUNT-2: llvm.store {{.*}} {alignment = 16 : i64, nontemporal} : vector<4xi32>, !llvm.ptr<1>
    // CHECK-COUNT-2: llvm.store {{.*}} {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<1>
    // CHECK-COUNT-2: llvm.store {{.*}} {alignment = 16 : i64, nontemporal} : vector<4xi32>, !llvm.ptr<1>
    // CHECK-COUNT-2: llvm.store {{.*}} {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<1>
    // CHECK-COUNT-2: llvm.store {{.*}} {alignment = 16 : i64, nontemporal} : vector<4xi32>, !llvm.ptr<1>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttig.support_predicated_io} {
  tt.func @load_store_cache_pred(%ptr: tensor<1024x!tt.ptr<f32>, #blocked>, %mask: tensor<1024xi1, #blocked>) {
    // CHECK: triton_gen.predicated_load {{.*}} {cache_control = L1UC_L3C} : (!llvm.ptr<1>, i1, i32) -> i32
    %val = tt.load %ptr, %mask cacheModifier = cg : tensor<1024x!tt.ptr<f32>, #blocked>
    // CHECK: triton_gen.predicated_store {{.*}} {cache_control = L1WT_L3WT} : (!llvm.ptr<1>, i32, i1) -> ()
    tt.store %ptr, %val, %mask cacheModifier = wt : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// COM: evict_first without an explicit cache modifier routes to L1IAR_L3C and
// COM: triggers the `nontemporal` flag on the non-predicated scalar load path.

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttig.support_predicated_io} {
  // CHECK-LABEL: load_evict_first_predicated
  tt.func @load_evict_first_predicated(%ptr: tensor<1024x!tt.ptr<f32>, #blocked>, %mask: tensor<1024xi1, #blocked>) {
    // CHECK: triton_gen.predicated_load {{.*}} {cache_control = L1IAR_L3C} : (!llvm.ptr<1>, i1, i32) -> i32
    %val = tt.load %ptr, %mask evictionPolicy = evict_first : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// COM: evict_last without an explicit cache modifier routes to L1C_L3C on the
// COM: predicated load path and stays temporal on the scalar path.

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttig.support_predicated_io} {
  // CHECK-LABEL: load_evict_last_predicated
  tt.func @load_evict_last_predicated(%ptr: tensor<1024x!tt.ptr<f32>, #blocked>, %mask: tensor<1024xi1, #blocked>) {
    // CHECK: triton_gen.predicated_load {{.*}} {cache_control = L1C_L3C} : (!llvm.ptr<1>, i1, i32) -> i32
    %val = tt.load %ptr, %mask evictionPolicy = evict_last : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// COM: An explicit cache modifier always wins over the eviction policy hint.

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttig.support_predicated_io} {
  // CHECK-LABEL: explicit_cache_wins_over_eviction
  tt.func @explicit_cache_wins_over_eviction(%ptr: tensor<1024x!tt.ptr<f32>, #blocked>, %mask: tensor<1024xi1, #blocked>) {
    // CHECK: triton_gen.predicated_load {{.*}} {cache_control = L1UC_L3C} : (!llvm.ptr<1>, i1, i32) -> i32
    %val = tt.load %ptr, %mask cacheModifier = cg evictionPolicy = evict_last : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// COM: evict_first on the non-predicated scalar load path adds the nontemporal
// COM: flag to the underlying llvm.load, without an explicit cache modifier.

#blocked0 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: load_evict_first_scalar
  tt.func @load_evict_first_scalar(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
    %3 = tt.splat %1 : i32 -> tensor<256xi32, #blocked0>
    %4 = arith.addi %3, %2 : tensor<256xi32, #blocked0>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %6 = tt.addptr %5, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    %7 = tt.load %6 evictionPolicy = evict_first : tensor<256x!tt.ptr<f32>, #blocked0>
    // CHECK-COUNT-2: llvm.load {{.*}} {alignment = 16 : i64, nontemporal} : !llvm.ptr<1> -> vector<4xi32>
    tt.return
  }
}

// -----

// COM: descriptor_load with evict_first routes to L1IAR_L3C on the predicated path.

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttig.support_predicated_io} {
  // CHECK-LABEL: descriptor_load_evict_first_predicated
  tt.func @descriptor_load_evict_first_predicated(%desc: !tt.tensordesc<128xf32>) {
    %c0_i32 = arith.constant 0 : i32
    // CHECK: triton_gen.predicated_load {{.*}} {cache_control = L1IAR_L3C} : (!llvm.ptr<1>, i1, i32) -> i32
    %val = tt.descriptor_load %desc[%c0_i32] evictionPolicy = evict_first : !tt.tensordesc<128xf32> -> tensor<128xf32, #blocked>
    tt.return
  }
}
