// RUN: env TRITON_INTEL_PREDICATED_LOAD=1 TRITON_INTEL_PREDICATED_STORE=1  triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s

// COM: Widened 256-bit vectorization — ttig.support_256b_load_store IS set on every module.
// COM: See load_store_to_llvm.mlir for the baseline 128-bit variant.
// COM: With the 256-bit gate open, an aligned contiguous elementwise load/store that
// COM: would previously emit two 128-bit messages (vector<4xi32>) collapses into a
// COM: single 256-bit message (vector<8xi32>). Sub-32-bit element types (f16, i8) are
// COM: packed into i32 words by the lowering, so all three widened cases expose the
// COM: same vector<8xi32> LLVM shape — the discriminator is the message *count*, not
// COM: the element type.

#blocked0 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttig.support_256b_load_store} {
  // CHECK-LABEL: global_load_256b_f32
  tt.func @global_load_256b_f32(%arg0: !tt.ptr<f32> {tt.divisibility = 32 : i32}) {
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
    %3 = tt.splat %1 : i32 -> tensor<256xi32, #blocked0>
    %4 = arith.addi %3, %2 : tensor<256xi32, #blocked0>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %6 = tt.addptr %5, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    %9 = tt.load %6 : tensor<256x!tt.ptr<f32>, #blocked0>
    // COM: 256 elements / 32 threads = 8 per thread, all contiguous aligned.
    // COM: 8 f32 elements * 32 bits = 256 bits -> one vector<8xi32> load
    // COM: (baseline would be two vector<4xi32> loads).
    // CHECK-COUNT-1: llvm.load {{.*}} {alignment = 32 : i64} : !llvm.ptr<1> -> vector<8xi32>
    // CHECK-NOT:     vector<4xi32>
    tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttig.support_256b_load_store} {
  // CHECK-LABEL: global_store_256b_f32
  tt.func @global_store_256b_f32(%arg0: !tt.ptr<f32> {tt.divisibility = 32 : i32}) {
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
    // COM: Same widening applies to stores. Widened -> single vector<8xi32> store.
    // CHECK-COUNT-1: llvm.store {{.*}} {alignment = 32 : i64} : vector<8xi32>, !llvm.ptr<1>
    // CHECK-NOT:     vector<4xi32>
    tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [16], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttig.support_256b_load_store} {
  // CHECK-LABEL: global_store_256b_f16
  tt.func @global_store_256b_f16(%arg0: !tt.ptr<f16> {tt.divisibility = 32 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<512xf16, #blocked0>
    %c512_i32 = arith.constant 512 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c512_i32 : i32
    %2 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked0>
    %3 = tt.splat %1 : i32 -> tensor<512xi32, #blocked0>
    %4 = arith.addi %3, %2 : tensor<512xi32, #blocked0>
    %5 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<512x!tt.ptr<f16>, #blocked0>
    %6 = tt.addptr %5, %4 : tensor<512x!tt.ptr<f16>, #blocked0>, tensor<512xi32, #blocked0>
    tt.store %6, %cst : tensor<512x!tt.ptr<f16>, #blocked0>
    // COM: 512 elements / 32 threads = 16 per thread. At 16 bits/elem, that is 256 bits/thread.
    // COM: Widened cap produces one 256-bit store packed as vector<8xi32>
    // COM: (baseline would be two vector<4xi32> stores of 8xf16 each).
    // CHECK-COUNT-1: llvm.store {{.*}} {alignment = 32 : i64} : vector<8xi32>, !llvm.ptr<1>
    // CHECK-NOT:     vector<4xi32>
    tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [32], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttig.support_256b_load_store} {
  // CHECK-LABEL: global_store_256b_i8
  tt.func @global_store_256b_i8(%arg0: !tt.ptr<i8> {tt.divisibility = 32 : i32}) {
    %cst = arith.constant dense<0> : tensor<1024xi8, #blocked0>
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked0>
    %3 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked0>
    %4 = arith.addi %3, %2 : tensor<1024xi32, #blocked0>
    %5 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<1024x!tt.ptr<i8>, #blocked0>
    %6 = tt.addptr %5, %4 : tensor<1024x!tt.ptr<i8>, #blocked0>, tensor<1024xi32, #blocked0>
    tt.store %6, %cst : tensor<1024x!tt.ptr<i8>, #blocked0>
    // COM: 1024 elements / 32 threads = 32 per thread. At 8 bits/elem, that is 256 bits/thread.
    // COM: Widened cap produces one 256-bit store packed as vector<8xi32>
    // COM: (baseline would be two vector<4xi32> stores of 16xi8 each).
    // CHECK-COUNT-1: llvm.store {{.*}} {alignment = 32 : i64} : vector<8xi32>, !llvm.ptr<1>
    // CHECK-NOT:     vector<4xi32>
    tt.return
  }
}

// -----

// COM: Sanity-check the gate: a masked store with a runtime-derived mask has
// COM: getMaskAlignment() == 1, which clamps the vectorization factor to 1
// COM: regardless of whether ttig.support_256b_load_store is present. The
// COM: emitted store must therefore be scalar-width (i32), NOT vector<8xi32>.
// COM: This protects against an over-eager widening that ignores mask alignment.

#blocked0 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttig.support_256b_load_store} {
  // CHECK-LABEL: masked_store_fallback
  tt.func @masked_store_fallback(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: tensor<256xi1, #blocked0>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<256xf32, #blocked0>
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
    %3 = tt.splat %1 : i32 -> tensor<256xi32, #blocked0>
    %4 = arith.addi %3, %2 : tensor<256xi32, #blocked0>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %6 = tt.addptr %5, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    tt.store %6, %cst, %arg1 : tensor<256x!tt.ptr<f32>, #blocked0>
    // COM: mask alignment is 1 -> per-element scalar stores (8 scalar stores per thread).
    // CHECK-NOT:   vector<8xi32>
    // CHECK-NOT:   vector<4xi32>
    // CHECK:       llvm.store {{.*}} i32, !llvm.ptr<1>
    tt.return
  }
}
