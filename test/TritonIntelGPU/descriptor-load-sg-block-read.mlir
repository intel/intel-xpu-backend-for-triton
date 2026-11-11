// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s
// RUN: env TRITON_INTEL_DISABLE_DESCRIPTOR_SG_BLOCK_READ=1 triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s --check-prefix=DISABLED

// A single-row descriptor load whose bounds checks are all block-level, and whose lanes hold consecutive 32/64-bit
// words of the row, becomes one sub-group block read per read group under a single uniform branch.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.support_predicated_io} {
  // CHECK-LABEL: @row_u32
  // DISABLED-LABEL: @row_u32
  tt.func public @row_u32(%base: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %rows: i32, %row: i32) -> tensor<1x128xf32, #blocked> {
    %c0 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c128 = arith.constant 128 : i32
    %c128_i64 = arith.constant 128 : i64
    %desc = tt.make_tensor_descriptor %base, [%rows, %c128], [%c128_i64, %c1_i64] : <f32>, <1x128xf32>
    // CHECK: %[[FILL:.*]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
    // CHECK: llvm.cond_br %{{.*}}, ^[[READ:bb[0-9]+]], ^[[END:bb[0-9]+]](%[[FILL]] : f32)
    // CHECK: ^[[READ]]:
    // CHECK: %[[WORD:.*]] = triton_gen.sub_group_block_read %{{.*}} : !llvm.ptr<1> -> i32
    // CHECK: %[[VAL:.*]] = llvm.bitcast %[[WORD]] : i32 to f32
    // CHECK: llvm.br ^[[END]](%[[VAL]] : f32)
    // CHECK-NOT: triton_gen.predicated_load
    // DISABLED-NOT: triton_gen.sub_group_block_read
    // DISABLED: triton_gen.predicated_load
    %x = tt.descriptor_load %desc[%row, %c0] {ttig.desc_padding = 1 : i32} : !tt.tensordesc<1x128xf32> -> tensor<1x128xf32, #blocked>
    tt.return %x : tensor<1x128xf32, #blocked>
  }
}

// -----

// Two f32 per lane: one 64-bit word.
#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.support_predicated_io} {
  // CHECK-LABEL: @row_u64
  tt.func public @row_u64(%base: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %rows: i32, %row: i32) -> tensor<1x256xf32, #blocked> {
    %c0 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c256 = arith.constant 256 : i32
    %c256_i64 = arith.constant 256 : i64
    %desc = tt.make_tensor_descriptor %base, [%rows, %c256], [%c256_i64, %c1_i64] : <f32>, <1x256xf32>
    // CHECK: %[[WORD:.*]] = triton_gen.sub_group_block_read %{{.*}} : !llvm.ptr<1> -> i64
    // CHECK: llvm.bitcast %[[WORD]] : i64 to vector<2xf32>
    %x = tt.descriptor_load %desc[%row, %c0] {ttig.desc_padding = 1 : i32} : !tt.tensordesc<1x256xf32> -> tensor<1x256xf32, #blocked>
    tt.return %x : tensor<1x256xf32, #blocked>
  }
}

// -----

// A register stepping by one sub-group of words folds into a 2-word read.
#linear = #ttg.linear<{register = [[0, 32]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [[0, 64], [0, 128]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.support_predicated_io} {
  // CHECK-LABEL: @row_vec2
  tt.func public @row_vec2(%base: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %rows: i32, %row: i32) -> tensor<1x256xf32, #linear> {
    %c0 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c256 = arith.constant 256 : i32
    %c256_i64 = arith.constant 256 : i64
    %desc = tt.make_tensor_descriptor %base, [%rows, %c256], [%c256_i64, %c1_i64] : <f32>, <1x256xf32>
    // CHECK: triton_gen.sub_group_block_read %{{.*}} : !llvm.ptr<1> -> vector<2xi32>
    %x = tt.descriptor_load %desc[%row, %c0] {ttig.desc_padding = 1 : i32} : !tt.tensordesc<1x256xf32> -> tensor<1x256xf32, #linear>
    tt.return %x : tensor<1x256xf32, #linear>
  }
}

// -----

// A register stepping past the warps' footprint is a second read group; warp 1 is redundant here.
#linear = #ttg.linear<{register = [[0, 64]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [[0, 32], [0, 0]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.support_predicated_io} {
  // CHECK-LABEL: @row_two_groups
  tt.func public @row_two_groups(%base: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %rows: i32, %row: i32) -> tensor<1x128xf32, #linear> {
    %c0 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c128 = arith.constant 128 : i32
    %c128_i64 = arith.constant 128 : i64
    %desc = tt.make_tensor_descriptor %base, [%rows, %c128], [%c128_i64, %c1_i64] : <f32>, <1x128xf32>
    // CHECK-COUNT-2: triton_gen.sub_group_block_read %{{.*}} : !llvm.ptr<1> -> i32
    %x = tt.descriptor_load %desc[%row, %c0] {ttig.desc_padding = 1 : i32} : !tt.tensordesc<1x128xf32> -> tensor<1x128xf32, #linear>
    tt.return %x : tensor<1x128xf32, #linear>
  }
}

// -----

// Rank-reducing and rank-3 results map their coordinates onto the descriptor's trailing dims.
#blocked1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 32], warpsPerCTA = [1, 1, 4], order = [2, 1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.support_predicated_io} {
  // CHECK-LABEL: @row_rank_reduced
  tt.func public @row_rank_reduced(%base: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %rows: i32, %row: i32) -> tensor<256xf32, #blocked1> {
    %c0 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c256 = arith.constant 256 : i32
    %c256_i64 = arith.constant 256 : i64
    %desc = tt.make_tensor_descriptor %base, [%rows, %c256], [%c256_i64, %c1_i64] : <f32>, <1x256xf32>
    // CHECK: triton_gen.sub_group_block_read %{{.*}} : !llvm.ptr<1> -> i64
    %x = tt.descriptor_load %desc[%row, %c0] {ttig.desc_padding = 1 : i32} : !tt.tensordesc<1x256xf32> -> tensor<256xf32, #blocked1>
    tt.return %x : tensor<256xf32, #blocked1>
  }
  // CHECK-LABEL: @row_rank3
  tt.func public @row_rank3(%base: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %d0: i32, %d1: i32, %i0: i32, %i1: i32) -> tensor<1x1x128xf32, #blocked3> {
    %c0 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c128 = arith.constant 128 : i32
    %c128_i64 = arith.constant 128 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %desc = tt.make_tensor_descriptor %base, [%d0, %d1, %c128], [%c4096_i64, %c128_i64, %c1_i64] : <f32>, <1x1x128xf32>
    // CHECK: triton_gen.sub_group_block_read %{{.*}} : !llvm.ptr<1> -> i32
    %x = tt.descriptor_load %desc[%i0, %i1, %c0] {ttig.desc_padding = 1 : i32} : !tt.tensordesc<1x1x128xf32> -> tensor<1x1x128xf32, #blocked3>
    tt.return %x : tensor<1x1x128xf32, #blocked3>
  }
}

// -----

// NaN padding reaches the out-of-bounds branch.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.support_predicated_io} {
  // CHECK-LABEL: @row_nan_padding
  tt.func public @row_nan_padding(%base: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %rows: i32, %row: i32) -> tensor<1x128xf32, #blocked> {
    %c0 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c128 = arith.constant 128 : i32
    %c128_i64 = arith.constant 128 : i64
    %desc = tt.make_tensor_descriptor %base, [%rows, %c128], [%c128_i64, %c1_i64] {padding = 2 : i32} : <f32>, <1x128xf32>
    // CHECK: %[[NAN:.*]] = llvm.mlir.constant(0x7FC00000 : f32) : f32
    // CHECK: llvm.cond_br %{{.*}}, ^{{bb[0-9]+}}, ^{{bb[0-9]+}}(%[[NAN]] : f32)
    %x = tt.descriptor_load %desc[%row, %c0] {ttig.desc_padding = 2 : i32} : !tt.tensordesc<1x128xf32> -> tensor<1x128xf32, #blocked>
    tt.return %x : tensor<1x128xf32, #blocked>
  }
}

// -----

// Loads that keep the predicated gathers.
#b1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#b2 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#b4 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#cols = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#bcast = #ttg.linear<{register = [[0, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [[0, 32], [0, 64]], block = []}>
#bool32 = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.support_predicated_io} {
  // 16-bit words.
  // CHECK-LABEL: @keep_u16
  // CHECK-NOT: sub_group_block_read
  // CHECK: triton_gen.predicated_load
  tt.func public @keep_u16(%base: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %rows: i32, %row: i32) -> tensor<1x128xbf16, #b1> {
    %c0 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c128 = arith.constant 128 : i32
    %c128_i64 = arith.constant 128 : i64
    %desc = tt.make_tensor_descriptor %base, [%rows, %c128], [%c128_i64, %c1_i64] : <bf16>, <1x128xbf16>
    %x = tt.descriptor_load %desc[%row, %c0] {ttig.desc_padding = 1 : i32} : !tt.tensordesc<1x128xbf16> -> tensor<1x128xbf16, #b1>
    tt.return %x : tensor<1x128xbf16, #b1>
  }
  // 128-bit chunks per lane.
  // CHECK-LABEL: @keep_16_bytes
  // CHECK-NOT: sub_group_block_read
  // CHECK: triton_gen.predicated_load
  tt.func public @keep_16_bytes(%base: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %rows: i32, %row: i32) -> tensor<1x512xf32, #b4> {
    %c0 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c512 = arith.constant 512 : i32
    %c512_i64 = arith.constant 512 : i64
    %desc = tt.make_tensor_descriptor %base, [%rows, %c512], [%c512_i64, %c1_i64] : <f32>, <1x512xf32>
    %x = tt.descriptor_load %desc[%row, %c0] {ttig.desc_padding = 1 : i32} : !tt.tensordesc<1x512xf32> -> tensor<1x512xf32, #b4>
    tt.return %x : tensor<1x512xf32, #b4>
  }
  // A broadcast register: two registers per lane hold the same element.
  // CHECK-LABEL: @keep_broadcast_register
  // CHECK-NOT: sub_group_block_read
  // CHECK: triton_gen.predicated_load
  tt.func public @keep_broadcast_register(%base: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %rows: i32, %row: i32) -> tensor<1x128xf32, #bcast> {
    %c0 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c128 = arith.constant 128 : i32
    %c128_i64 = arith.constant 128 : i64
    %desc = tt.make_tensor_descriptor %base, [%rows, %c128], [%c128_i64, %c1_i64] : <f32>, <1x128xf32>
    %x = tt.descriptor_load %desc[%row, %c0] {ttig.desc_padding = 1 : i32} : !tt.tensordesc<1x128xf32> -> tensor<1x128xf32, #bcast>
    tt.return %x : tensor<1x128xf32, #bcast>
  }
  // Bools, even 32 per lane (one 32-bit word of them).
  // CHECK-LABEL: @keep_bool
  // CHECK-NOT: sub_group_block_read
  // CHECK: triton_gen.predicated_load
  tt.func public @keep_bool(%base: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %rows: i32, %row: i32) -> tensor<1x4096xi1, #bool32> {
    %c0 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c4096 = arith.constant 4096 : i32
    %c4096_i64 = arith.constant 4096 : i64
    %desc = tt.make_tensor_descriptor %base, [%rows, %c4096], [%c4096_i64, %c1_i64] : <i1>, <1x4096xi1>
    %x = tt.descriptor_load %desc[%row, %c0] {ttig.desc_padding = 1 : i32} : !tt.tensordesc<1x4096xi1> -> tensor<1x4096xi1, #bool32>
    tt.return %x : tensor<1x4096xi1, #bool32>
  }
  // Lanes down the rows (16x2 tile): consecutive in the flattened layout, whose minor dim is dim 0, but strided
  // in memory. Constant extents keep both dims block-level.
  // CHECK-LABEL: @keep_multi_row
  // CHECK-NOT: sub_group_block_read
  // CHECK: triton_gen.predicated_load
  tt.func public @keep_multi_row(%base: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> tensor<16x2xf32, #cols> {
    %c0 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64 = arith.constant 64 : i32
    %c64_i64 = arith.constant 64 : i64
    %desc = tt.make_tensor_descriptor %base, [%c64, %c64], [%c64_i64, %c1_i64] : <f32>, <16x2xf32>
    %x = tt.descriptor_load %desc[%c0, %c0] {ttig.desc_padding = 1 : i32} : !tt.tensordesc<16x2xf32> -> tensor<16x2xf32, #cols>
    tt.return %x : tensor<16x2xf32, #cols>
  }
  // The column extent (100) is not a multiple of the block (128): a per-element check.
  // CHECK-LABEL: @keep_per_element
  // CHECK-NOT: sub_group_block_read
  // CHECK: triton_gen.predicated_load
  tt.func public @keep_per_element(%base: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %rows: i32, %row: i32) -> tensor<1x128xf32, #b1> {
    %c0 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c100 = arith.constant 100 : i32
    %c128_i64 = arith.constant 128 : i64
    %desc = tt.make_tensor_descriptor %base, [%rows, %c100], [%c128_i64, %c1_i64] : <f32>, <1x128xf32>
    %x = tt.descriptor_load %desc[%row, %c0] {ttig.desc_padding = 1 : i32} : !tt.tensordesc<1x128xf32> -> tensor<1x128xf32, #b1>
    tt.return %x : tensor<1x128xf32, #b1>
  }
  // A row stride of unknown divisibility proves only 4-byte rows; 64-bit words need 8.
  // CHECK-LABEL: @keep_unaligned
  // CHECK-NOT: sub_group_block_read
  // CHECK: triton_gen.predicated_load
  tt.func public @keep_unaligned(%base: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %rows: i32, %stride: i64, %row: i32) -> tensor<1x256xf32, #b2> {
    %c0 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c256 = arith.constant 256 : i32
    %desc = tt.make_tensor_descriptor %base, [%rows, %c256], [%stride, %c1_i64] : <f32>, <1x256xf32>
    %x = tt.descriptor_load %desc[%row, %c0] {ttig.desc_padding = 1 : i32} : !tt.tensordesc<1x256xf32> -> tensor<1x256xf32, #b2>
    tt.return %x : tensor<1x256xf32, #b2>
  }
  // 12-byte divisibility is at least 8 but only proves 4-byte alignment.
  // CHECK-LABEL: @keep_non_pow2_divisibility
  // CHECK-NOT: sub_group_block_read
  // CHECK: triton_gen.predicated_load
  tt.func public @keep_non_pow2_divisibility(%base: !tt.ptr<f32> {tt.divisibility = 12 : i32}, %rows: i32, %stride: i64 {tt.divisibility = 3 : i32}, %row: i32) -> tensor<1x256xf32, #b2> {
    %c0 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c256 = arith.constant 256 : i32
    %desc = tt.make_tensor_descriptor %base, [%rows, %c256], [%stride, %c1_i64] : <f32>, <1x256xf32>
    %x = tt.descriptor_load %desc[%row, %c0] {ttig.desc_padding = 1 : i32} : !tt.tensordesc<1x256xf32> -> tensor<1x256xf32, #b2>
    tt.return %x : tensor<1x256xf32, #b2>
  }
}
