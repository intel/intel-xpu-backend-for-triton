// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s
// RUN: env TRITON_INTEL_DISABLE_DESCRIPTOR_SG_BLOCK_READ=1 triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s --check-prefix=DISABLED

// COM: A single-row tt.load whose lanes hold consecutive 32/64-bit words of a contiguous, word-aligned row under a
// COM: uniform mask becomes sub-group block reads.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.support_predicated_io} {
  // COM: A contiguous, word-aligned row under a uniform mask: one block read
  // COM: under a branch that falls back to `other`.
  // CHECK-LABEL: @row(
  // CHECK: llvm.mlir.constant(2.000000e+00 : f32) : f32
  // CHECK: llvm.cond_br %{{.*}}, ^[[READ:bb[0-9]+]], ^[[END:bb[0-9]+]](%{{.*}} : f32)
  // CHECK: ^[[READ]]:
  // CHECK: %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[BCAST:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflelj(%{{.*}}, %[[C0]])
  // CHECK: %[[ADDR:.*]] = llvm.inttoptr %[[BCAST]] : i64 to !llvm.ptr<1>
  // CHECK: %[[WORD:.*]] = triton_gen.sub_group_block_read %[[ADDR]] {{.*}}: !llvm.ptr<1> -> i32
  // CHECK: %[[VAL:.*]] = llvm.bitcast %[[WORD]] : i32 to f32
  // CHECK: llvm.br ^[[END]](%[[VAL]] : f32)
  // DISABLED-LABEL: @row(
  // DISABLED-NOT: sub_group_block_read
  // DISABLED: triton_gen.predicated_load
  tt.func public @row(%base: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %c: i1) -> tensor<128xf32, #blocked> {
    %r = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %s = tt.splat %base : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %p = tt.addptr %s, %r : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %m = tt.splat %c : i1 -> tensor<128xi1, #blocked>
    %o = arith.constant dense<2.0> : tensor<128xf32, #blocked>
    %x = tt.load %p, %m, %o : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return %x : tensor<128xf32, #blocked>
  }
  // COM: The load's cache policy carries over to the block read.
  // CHECK-LABEL: @row_cache_control(
  // CHECK: triton_gen.sub_group_block_read %{{.*}} {cache_control = L1UC_L3C} : !llvm.ptr<1> -> i32
  tt.func public @row_cache_control(%base: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> tensor<128xf32, #blocked> {
    %r = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %s = tt.splat %base : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %p = tt.addptr %s, %r : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %x = tt.load %p {cachePolicy = #tt.cache_policy<cache_modifier = cg, eviction_policy = evict_normal>} : tensor<128x!tt.ptr<f32>, #blocked>
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
  // COM: An aligned base with a strided row: the lanes' words are not consecutive.
  // CHECK-LABEL: @keep_strided(
  // CHECK-NOT: sub_group_block_read
  // CHECK: llvm.load
  tt.func public @keep_strided(%base: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> tensor<128xf32, #blocked> {
    %c2 = arith.constant dense<2> : tensor<128xi32, #blocked>
    %r = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %r2 = arith.muli %r, %c2 : tensor<128xi32, #blocked>
    %s = tt.splat %base : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %p = tt.addptr %s, %r2 : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %x = tt.load %p : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return %x : tensor<128xf32, #blocked>
  }
}

// -----

// COM: Two f32 per lane: one 64-bit word, which needs an 8-byte aligned row.
#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.support_predicated_io} {
  // CHECK-LABEL: @row_u64(
  // CHECK: %[[WORD:.*]] = triton_gen.sub_group_block_read %{{.*}} : !llvm.ptr<1> -> i64
  // CHECK: llvm.bitcast %[[WORD]] : i64 to vector<2xf32>
  tt.func public @row_u64(%base: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> tensor<256xf32, #blocked> {
    %r = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %s = tt.splat %base : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %p = tt.addptr %s, %r : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    %x = tt.load %p : tensor<256x!tt.ptr<f32>, #blocked>
    tt.return %x : tensor<256xf32, #blocked>
  }
  // CHECK-LABEL: @keep_u64_4byte_aligned(
  // CHECK-NOT: sub_group_block_read
  // CHECK: llvm.load
  tt.func public @keep_u64_4byte_aligned(%base: !tt.ptr<f32> {tt.divisibility = 4 : i32}) -> tensor<256xf32, #blocked> {
    %r = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %s = tt.splat %base : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %p = tt.addptr %s, %r : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    %x = tt.load %p : tensor<256x!tt.ptr<f32>, #blocked>
    tt.return %x : tensor<256xf32, #blocked>
  }
}

// -----

// COM: A register stepping by one sub-group of words folds into a 2-word read.
#linear = #ttg.linear<{register = [[32]], lane = [[1], [2], [4], [8], [16]], warp = [[64], [128]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.support_predicated_io} {
  // CHECK-LABEL: @row_vec2(
  // CHECK: %[[WORDS:.*]] = triton_gen.sub_group_block_read %{{.*}} : !llvm.ptr<1> -> vector<2xi32>
  // CHECK: llvm.extractelement %[[WORDS]]
  // CHECK: llvm.extractelement %[[WORDS]]
  tt.func public @row_vec2(%base: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> tensor<256xf32, #linear> {
    %r = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #linear>
    %s = tt.splat %base : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #linear>
    %p = tt.addptr %s, %r : tensor<256x!tt.ptr<f32>, #linear>, tensor<256xi32, #linear>
    %x = tt.load %p : tensor<256x!tt.ptr<f32>, #linear>
    tt.return %x : tensor<256xf32, #linear>
  }
}
