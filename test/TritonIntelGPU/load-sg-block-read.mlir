// RUN: triton-opt %s --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s

// COM: A single-row tt.load whose lanes hold consecutive 32/64-bit words of a contiguous, word-aligned row under a
// COM: uniform mask becomes sub-group block reads.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.support_predicated_io} {
  // COM: A contiguous, word-aligned row under a uniform mask: one block read
  // COM: under a branch that falls back to `other`.
  // CHECK-LABEL: @row(
  // CHECK: %[[TWO:.*]] = llvm.mlir.constant(2.000000e+00 : f32) : f32
  // CHECK: %[[BC:.*]] = llvm.bitcast %[[TWO]] : f32 to f32
  // CHECK: %[[S:.*]] = llvm.insertvalue %[[BC]], %{{.*}}[0]
  // CHECK: %[[FILL:.*]] = llvm.extractvalue %[[S]][0]
  // CHECK: llvm.cond_br %{{.*}}, ^[[READ:bb[0-9]+]], ^[[END:bb[0-9]+]](%[[FILL]] : f32)
  // CHECK: ^[[READ]]:
  // CHECK: %[[WORD:.*]] = triton_gen.sub_group_block_read %{{.*}} : !llvm.ptr<1> -> i32
  // CHECK: %[[VAL:.*]] = llvm.bitcast %[[WORD]] : i32 to f32
  // CHECK: llvm.br ^[[END]](%[[VAL]] : f32)
  tt.func public @row(%base: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %c: i1) -> tensor<128xf32, #blocked> {
    %r = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %s = tt.splat %base : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %p = tt.addptr %s, %r : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %m = tt.splat %c : i1 -> tensor<128xi1, #blocked>
    %o = arith.constant dense<2.0> : tensor<128xf32, #blocked>
    %x = tt.load %p, %m, %o : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return %x : tensor<128xf32, #blocked>
  }
  // COM: evict_last asks for the default L1C_L3C caching, so it keeps the block read.
  // CHECK-LABEL: @row_evict_last(
  // CHECK: triton_gen.sub_group_block_read
  tt.func public @row_evict_last(%base: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> tensor<128xf32, #blocked> {
    %r = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %s = tt.splat %base : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %p = tt.addptr %s, %r : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %x = tt.load %p {cachePolicy = #tt.cache_policy<cache_modifier = none, eviction_policy = evict_last>} : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return %x : tensor<128xf32, #blocked>
  }
  // COM: Loads that keep the gathers.
  // CHECK-LABEL: @keep_volatile(
  // CHECK-NOT: sub_group_block_read
  // CHECK: llvm.load volatile
  tt.func public @keep_volatile(%base: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> tensor<128xf32, #blocked> {
    %r = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %s = tt.splat %base : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %p = tt.addptr %s, %r : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %x = tt.load %p {isVolatile = true} : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return %x : tensor<128xf32, #blocked>
  }
  // CHECK-LABEL: @keep_cache_modifier(
  // CHECK-NOT: sub_group_block_read
  // CHECK: llvm.load
  tt.func public @keep_cache_modifier(%base: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> tensor<128xf32, #blocked> {
    %r = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %s = tt.splat %base : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %p = tt.addptr %s, %r : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %x = tt.load %p {cachePolicy = #tt.cache_policy<cache_modifier = cg, eviction_policy = evict_normal>} : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return %x : tensor<128xf32, #blocked>
  }
  // CHECK-LABEL: @keep_per_element_mask(
  // CHECK-NOT: sub_group_block_read
  // CHECK: triton_gen.predicated_load
  tt.func public @keep_per_element_mask(%base: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %n: i32) -> tensor<128xf32, #blocked> {
    %r = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %s = tt.splat %base : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %p = tt.addptr %s, %r : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %ns = tt.splat %n : i32 -> tensor<128xi32, #blocked>
    %m = arith.cmpi slt, %r, %ns : tensor<128xi32, #blocked>
    %x = tt.load %p, %m : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return %x : tensor<128xf32, #blocked>
  }
  // CHECK-LABEL: @keep_unaligned(
  // CHECK-NOT: sub_group_block_read
  // CHECK: llvm.load
  tt.func public @keep_unaligned(%base: !tt.ptr<f32>) -> tensor<128xf32, #blocked> {
    %r = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %s = tt.splat %base : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %p = tt.addptr %s, %r : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %x = tt.load %p : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return %x : tensor<128xf32, #blocked>
  }
}
