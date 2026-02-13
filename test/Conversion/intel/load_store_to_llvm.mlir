// RUN: env TRITON_INTEL_PREDICATED_LOAD=0 TRITON_INTEL_PREDICATED_STORE=0  triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm | FileCheck %s --check-prefixes=CHECK,NO-PREDICATED
// RUN: env TRITON_INTEL_PREDICATED_LOAD=1 TRITON_INTEL_PREDICATED_STORE=1  triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm | FileCheck %s --check-prefixes=CHECK,PREDICATED

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

// This test verifies the vectorization of Load and Store Ops when supplied with cache modifiers.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
// Note, the %n_elements doesn't have a "tt.divisibility" hint, so Triton assumes it's divisibility is 1, this should effect the mask's alignment and further restrict the load/store ops' vector width to be 1.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttig.support_predicated_io"} {
  tt.func @vecadd_masked_vec2(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %n_elements: i32) {
    %c64_i32 = arith.constant 64 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c64_i32 : i32
    %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %3 = tt.splat %1 : i32 -> tensor<64xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<64xi32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked>
    %6 = tt.addptr %5, %4 : tensor<64x!tt.ptr<f32>, #blocked>, tensor<64xi32, #blocked>
    %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked>
    %8 = tt.addptr %7, %4 : tensor<64x!tt.ptr<f32>, #blocked>, tensor<64xi32, #blocked>
    %9 = tt.splat %n_elements : i32 -> tensor<64xi32, #blocked>
    %10 = arith.cmpi "slt", %4, %9 : tensor<64xi32, #blocked>
    // load op has a vector width = 1 due to the %mask's alignment
    // PREDICATED: triton_gen.predicated_load {{.*}} {cache_control = L1UC_L3UC} : (!llvm.ptr<1>, i64, i1, i32) -> i32
    // NO-PREDICATED: llvm.load %{{.*}} {alignment = 4 : i64, nontemporal} : !llvm.ptr<1> -> i32
    %11 = tt.load %6, %10 cacheModifier = cv : tensor<64x!tt.ptr<f32>, #blocked>
    // PREDICATED: triton_gen.predicated_load {{.*}} {cache_control = L1UC_L3C} : (!llvm.ptr<1>, i64, i1, i32) -> i32
    // NO-PREDICATED: llvm.load %{{.*}} {alignment = 4 : i64, nontemporal} : !llvm.ptr<1> -> i32
    %12 = tt.load %8, %10 cacheModifier = cg : tensor<64x!tt.ptr<f32>, #blocked>
    %13 = arith.addf %11, %12 : tensor<64xf32, #blocked>
    %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked>
    %15 = tt.addptr %14, %4 : tensor<64x!tt.ptr<f32>, #blocked>, tensor<64xi32, #blocked>
    // PREDICATED: triton_gen.predicated_store {{.*}} {cache_control = L1WT_L3WT} : (!llvm.ptr<1>, i32, i64, i1) -> ()
    // NO-PREDICATED: llvm.store {{.*}} {alignment = 4 : i64} : i32, !llvm.ptr<1>
    tt.store %15, %13, %10 cacheModifier = wt : tensor<64x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
