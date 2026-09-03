// RUN: triton-opt %s -split-input-file -tritonintelgpu-materialize-block-pointer | FileCheck %s

// COM: Test 1: Single-row tile (H == 1) — reshape optimization fires.
// COM: numElements=32, W=32, S=96, f16, numWarps=1.
// COM: H = 32/32 = 1 → the pass should reshape [32] -> [1, 32] and set
// COM: ttig.block_io = "row_major" with ttig.block_io_stride = 96.

#blocked1d = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @test_single_row_h1
  tt.func @test_single_row_h1(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: tensor<32xf16, #blocked1d>) {
    %idx = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #blocked1d>
    %cst32 = arith.constant dense<32> : tensor<32xi32, #blocked1d>
    %cst96 = arith.constant dense<96> : tensor<32xi32, #blocked1d>
    %rem = arith.remui %idx, %cst32 : tensor<32xi32, #blocked1d>
    %div = arith.divui %idx, %cst32 : tensor<32xi32, #blocked1d>
    %mul = arith.muli %div, %cst96 : tensor<32xi32, #blocked1d>
    %off = arith.addi %rem, %mul : tensor<32xi32, #blocked1d>
    %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>, #blocked1d>
    %ptrs = tt.addptr %base, %off : tensor<32x!tt.ptr<f16>, #blocked1d>, tensor<32xi32, #blocked1d>
    // CHECK: tt.reshape
    // CHECK: tt.reshape
    // CHECK: tt.store %{{.*}}, %{{.*}} {ttig.block_io = "row_major", ttig.block_io_stride = 96 : i64} : tensor<1x32x!tt.ptr<f16>
    tt.store %ptrs, %arg1 : tensor<32x!tt.ptr<f16>, #blocked1d>
    tt.return
  }
}

// -----

// COM: Test 2: Multi-row tile (H > 1).
// COM: numElements=1024, W=32, S=96, fp16, numWarps=4, H = 1024/32 = 32.
// COM: Both ptr and val are reshaped then ConvertLayoutOps convert both into the HW delivery encoding
// COM: (sizePerThread=[8,1], threadsPerWarp=[1,32]: lane k owns column k, registers
// COM: stack rows). RemoveLayoutConversions back-propagates the store encoding into
// COM: the pointer arithmetic and eliminates the ptr ConvertLayout.

// CHECK-DAG: [[STOREENC:#[a-z0-9_]+]] = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK-DAG: [[CONSENC:#[a-z0-9_]+]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>

#blocked1d = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // COM: Plain CHECK (not CHECK-LABEL) so the [[STOREENC]]/[[CONSENC]] captures
  // COM: above stay in scope — FileCheck matches CHECK-LABEL blocks independently.
  // CHECK: tt.func @test_multi_row_h32
  tt.func @test_multi_row_h32(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: tensor<1024xf16, #blocked1d>) {
    %idx = tt.make_range {start = 0 : i32, end = 1024 : i32} : tensor<1024xi32, #blocked1d>
    %cst32 = arith.constant dense<32> : tensor<1024xi32, #blocked1d>
    %cst96 = arith.constant dense<96> : tensor<1024xi32, #blocked1d>
    %rem = arith.remui %idx, %cst32 : tensor<1024xi32, #blocked1d>
    %div = arith.divui %idx, %cst32 : tensor<1024xi32, #blocked1d>
    %mul = arith.muli %div, %cst96 : tensor<1024xi32, #blocked1d>
    %off = arith.addi %rem, %mul : tensor<1024xi32, #blocked1d>
    %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked1d>
    %ptrs = tt.addptr %base, %off : tensor<1024x!tt.ptr<f16>, #blocked1d>, tensor<1024xi32, #blocked1d>
    // CHECK: [[PTR2D:%[0-9]+]] = tt.reshape %{{.*}} : tensor<1024x!tt.ptr<f16>, {{.*}}> -> tensor<32x32x!tt.ptr<f16>, [[CONSENC]]>
    // CHECK: [[VAL2D:%[0-9]+]] = tt.reshape %{{.*}} : tensor<1024xf16, {{.*}}> -> tensor<32x32xf16, [[CONSENC]]>
    // CHECK: ttg.convert_layout [[PTR2D]] : tensor<32x32x!tt.ptr<f16>, [[CONSENC]]> -> tensor<32x32x!tt.ptr<f16>, [[STOREENC]]>
    // CHECK: [[CVT:%[0-9]+]] = ttg.convert_layout [[VAL2D]] : tensor<32x32xf16, [[CONSENC]]> -> tensor<32x32xf16, [[STOREENC]]>
    // CHECK: tt.store %{{.*}}, [[CVT]] {ttig.block_io = "row_major", ttig.block_io_stride = 96 : i64} : tensor<32x32x!tt.ptr<f16>, [[STOREENC]]>
    tt.store %ptrs, %arg1 : tensor<1024x!tt.ptr<f16>, #blocked1d>
    tt.return
  }
}

// -----

// COM: Test 3: Masked store with dense<true> constant — should still be reshaped.
// COM: The mask is a direct dense<true> tensor constant (not splat(true)).
// COM: matchPattern/m_One recognizes this as provably all-true.

#blocked1d = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @test_dense_true_mask
  tt.func @test_dense_true_mask(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: tensor<32xf16, #blocked1d>) {
    %idx = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #blocked1d>
    %cst32 = arith.constant dense<32> : tensor<32xi32, #blocked1d>
    %cst96 = arith.constant dense<96> : tensor<32xi32, #blocked1d>
    %rem = arith.remui %idx, %cst32 : tensor<32xi32, #blocked1d>
    %div = arith.divui %idx, %cst32 : tensor<32xi32, #blocked1d>
    %mul = arith.muli %div, %cst96 : tensor<32xi32, #blocked1d>
    %off = arith.addi %rem, %mul : tensor<32xi32, #blocked1d>
    %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>, #blocked1d>
    %ptrs = tt.addptr %base, %off : tensor<32x!tt.ptr<f16>, #blocked1d>, tensor<32xi32, #blocked1d>
    %mask = arith.constant dense<true> : tensor<32xi1, #blocked1d>
    // CHECK: tt.reshape
    // CHECK: tt.reshape
    // CHECK: tt.store %{{.*}}, %{{.*}} {ttig.block_io = "row_major", ttig.block_io_stride = 96 : i64} : tensor<1x32x!tt.ptr<f16>
    tt.store %ptrs, %arg1, %mask : tensor<32x!tt.ptr<f16>, #blocked1d>
    tt.return
  }
}

// -----

// COM: Test 4: Non-canonical index rejection — scaled index.
// COM: The index is 2 * tt.make_range instead of plain tt.make_range.
// COM: The isCanonicalLinearIndex check should reject this because remui's
// COM: LHS is arith.muli, not tt.make_range(0, N).

#blocked1d = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @test_non_canonical_index
  tt.func @test_non_canonical_index(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: tensor<1024xf16, #blocked1d>) {
    %idx = tt.make_range {start = 0 : i32, end = 1024 : i32} : tensor<1024xi32, #blocked1d>
    %cst2 = arith.constant dense<2> : tensor<1024xi32, #blocked1d>
    %scaled_idx = arith.muli %idx, %cst2 : tensor<1024xi32, #blocked1d>
    %cst32 = arith.constant dense<32> : tensor<1024xi32, #blocked1d>
    %cst96 = arith.constant dense<96> : tensor<1024xi32, #blocked1d>
    %rem = arith.remui %scaled_idx, %cst32 : tensor<1024xi32, #blocked1d>
    %div = arith.divui %scaled_idx, %cst32 : tensor<1024xi32, #blocked1d>
    %mul = arith.muli %div, %cst96 : tensor<1024xi32, #blocked1d>
    %off = arith.addi %rem, %mul : tensor<1024xi32, #blocked1d>
    %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked1d>
    %ptrs = tt.addptr %base, %off : tensor<1024x!tt.ptr<f16>, #blocked1d>, tensor<1024xi32, #blocked1d>
    // CHECK-NOT: tt.reshape
    // CHECK: tt.store
    // CHECK-NOT: ttig.block_io
    tt.store %ptrs, %arg1 : tensor<1024x!tt.ptr<f16>, #blocked1d>
    tt.return
  }
}

// -----

// COM: Test 5: Non-trivial mask rejection.
// COM: Same strided offset pattern as Test 1 but the store has a real mask
// COM: argument (not splat(true)). The pass should reject this and leave the
// COM: 1D store unchanged.

#blocked1d = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @test_non_trivial_mask
  tt.func @test_non_trivial_mask(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: tensor<32xf16, #blocked1d>, %mask: tensor<32xi1, #blocked1d>) {
    %idx = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #blocked1d>
    %cst32 = arith.constant dense<32> : tensor<32xi32, #blocked1d>
    %cst96 = arith.constant dense<96> : tensor<32xi32, #blocked1d>
    %rem = arith.remui %idx, %cst32 : tensor<32xi32, #blocked1d>
    %div = arith.divui %idx, %cst32 : tensor<32xi32, #blocked1d>
    %mul = arith.muli %div, %cst96 : tensor<32xi32, #blocked1d>
    %off = arith.addi %rem, %mul : tensor<32xi32, #blocked1d>
    %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>, #blocked1d>
    %ptrs = tt.addptr %base, %off : tensor<32x!tt.ptr<f16>, #blocked1d>, tensor<32xi32, #blocked1d>
    // CHECK-NOT: tt.reshape
    // CHECK: tt.store
    // CHECK-NOT: ttig.block_io
    tt.store %ptrs, %arg1, %mask : tensor<32x!tt.ptr<f16>, #blocked1d>
    tt.return
  }
}

// -----

// COM: Test 6: 1D strided load — gather load with the Inductor offset pattern.
// COM: W=32, S=96, 1024 elements → H=32. The pass reshapes the 1D load to a
// COM: 2D block load with an explicit load encoding, inserts ConvertLayoutOp,
// COM: and reshapes back to 1D.

#blocked1d = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // COM: The pointer and mask must be reshaped without reordering and then
  // COM: physically converted into the load encoding. A relabelling reshape
  // COM: (allow_reorder efficient_layout) moves no data, so the operands would
  // COM: claim the load encoding while still holding 1D-ordered values.
  // CHECK-LABEL: tt.func @test_1d_strided_load
  // CHECK: [[PTR2D:%[0-9]+]] = tt.reshape %{{.*}} : tensor<1024x!tt.ptr<f16>, {{.*}}> -> tensor<32x32x!tt.ptr<f16>, [[CONSENC:#[a-z0-9_]+]]>
  // CHECK: [[PTRCVT:%[0-9]+]] = ttg.convert_layout [[PTR2D]] : tensor<32x32x!tt.ptr<f16>, [[CONSENC]]> -> tensor<32x32x!tt.ptr<f16>, [[LOADENC:#[a-z0-9_]+]]>
  // CHECK: [[MASK2D:%[0-9]+]] = tt.reshape %{{.*}} : tensor<1024xi1, {{.*}}> -> tensor<32x32xi1, [[CONSENC]]>
  // CHECK: [[MASKCVT:%[0-9]+]] = ttg.convert_layout [[MASK2D]] : tensor<32x32xi1, [[CONSENC]]> -> tensor<32x32xi1, [[LOADENC]]>
  // CHECK: [[LOADED:%[0-9]+]] = tt.load [[PTRCVT]], [[MASKCVT]] {ttig.block_io = "row_major", ttig.block_io_stride = 96 : i64} : tensor<32x32x!tt.ptr<f16>, [[LOADENC]]>
  // CHECK: [[CVT:%[0-9]+]] = ttg.convert_layout [[LOADED]] : tensor<32x32xf16, [[LOADENC]]> -> tensor<32x32xf16, [[CONSENC]]>
  // CHECK: tt.reshape [[CVT]] efficient_layout : tensor<32x32xf16, [[CONSENC]]> -> tensor<1024xf16, {{.*}}>
  tt.func @test_1d_strided_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<1024xf16, #blocked1d> {
    %idx = tt.make_range {start = 0 : i32, end = 1024 : i32} : tensor<1024xi32, #blocked1d>
    %c32 = arith.constant dense<32> : tensor<1024xi32, #blocked1d>
    %c96 = arith.constant dense<96> : tensor<1024xi32, #blocked1d>
    %rem = arith.remui %idx, %c32 : tensor<1024xi32, #blocked1d>
    %div = arith.divui %idx, %c32 : tensor<1024xi32, #blocked1d>
    %mul = arith.muli %div, %c96 : tensor<1024xi32, #blocked1d>
    %off = arith.addi %rem, %mul : tensor<1024xi32, #blocked1d>
    %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked1d>
    %ptrs = tt.addptr %base, %off : tensor<1024x!tt.ptr<f16>, #blocked1d>, tensor<1024xi32, #blocked1d>
    %mask = arith.constant dense<true> : tensor<1024xi1, #blocked1d>
    %result = tt.load %ptrs, %mask : tensor<1024x!tt.ptr<f16>, #blocked1d>
    tt.return %result : tensor<1024xf16, #blocked1d>
  }
}

// -----

// COM: Test 7: 1D strided load with H=1 (W=32, 32 elements). With 4 warps,
// COM: per-warp height = 1/4, which does not divide evenly. The pass should
// COM: reject this and leave the 1D load unchanged.

#blocked1d_small = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @test_1d_strided_load_single_row
  // CHECK-NOT: ttig.block_io
  // CHECK: tt.load
  // CHECK-NOT: ttig.block_io
  tt.func @test_1d_strided_load_single_row(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<32xf16, #blocked1d_small> {
    %idx = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #blocked1d_small>
    %c32 = arith.constant dense<32> : tensor<32xi32, #blocked1d_small>
    %c96 = arith.constant dense<96> : tensor<32xi32, #blocked1d_small>
    %rem = arith.remui %idx, %c32 : tensor<32xi32, #blocked1d_small>
    %div = arith.divui %idx, %c32 : tensor<32xi32, #blocked1d_small>
    %mul = arith.muli %div, %c96 : tensor<32xi32, #blocked1d_small>
    %off = arith.addi %rem, %mul : tensor<32xi32, #blocked1d_small>
    %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>, #blocked1d_small>
    %ptrs = tt.addptr %base, %off : tensor<32x!tt.ptr<f16>, #blocked1d_small>, tensor<32xi32, #blocked1d_small>
    %mask = arith.constant dense<true> : tensor<32xi1, #blocked1d_small>
    %result = tt.load %ptrs, %mask : tensor<32x!tt.ptr<f16>, #blocked1d_small>
    tt.return %result : tensor<32xf16, #blocked1d_small>
  }
}

// -----

// COM: Test 8: 1D strided load with W < threadsPerWarp (W=16, tpw=32).
// COM: The pass must bail out — the 2D block load HW delivers the tile into
// COM: the first W lanes and the remaining lanes' data are not a plain
// COM: row/col layout.  Constructing a [1,tpw] load encoding would broadcast
// COM: across lanes and produce an "expensive view" reshape that cannot be
// COM: lowered (make_llir crash). Regression test for issue #6738.

#blocked1d = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @test_1d_strided_load_narrow_width
  // CHECK-NOT: ttig.block_io
  // CHECK: tt.load
  // CHECK-NOT: ttig.block_io
  tt.func @test_1d_strided_load_narrow_width(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<128xf16, #blocked1d> {
    %idx = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #blocked1d>
    %c16 = arith.constant dense<16> : tensor<128xi32, #blocked1d>
    %c96 = arith.constant dense<96> : tensor<128xi32, #blocked1d>
    %rem = arith.remui %idx, %c16 : tensor<128xi32, #blocked1d>
    %div = arith.divui %idx, %c16 : tensor<128xi32, #blocked1d>
    %mul = arith.muli %div, %c96 : tensor<128xi32, #blocked1d>
    %off = arith.addi %rem, %mul : tensor<128xi32, #blocked1d>
    %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x!tt.ptr<f16>, #blocked1d>
    %ptrs = tt.addptr %base, %off : tensor<128x!tt.ptr<f16>, #blocked1d>, tensor<128xi32, #blocked1d>
    %mask = arith.constant dense<true> : tensor<128xi1, #blocked1d>
    %result = tt.load %ptrs, %mask : tensor<128x!tt.ptr<f16>, #blocked1d>
    tt.return %result : tensor<128xf16, #blocked1d>
  }
}

// -----

// COM: Test 9: 1D strided store with W < threadsPerWarp (W=16, tpw=32), H=32.
// COM: The store path carries the same bail-out as the load (issue #6738): a
// COM: [1,tpw] encoding on a dimension of size W < tpw is replicated and the
// COM: reshape cannot be legalized.  Must fall back to the scatter store.

#blocked1d = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @test_1d_strided_store_narrow_width
  // CHECK-NOT: tt.reshape
  // CHECK-NOT: ttig.block_io
  // CHECK: tt.store %{{.*}}, %{{.*}} : tensor<128x!tt.ptr<f16>
  tt.func @test_1d_strided_store_narrow_width(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: tensor<128xf16, #blocked1d>) {
    %idx = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #blocked1d>
    %c16 = arith.constant dense<16> : tensor<128xi32, #blocked1d>
    %c96 = arith.constant dense<96> : tensor<128xi32, #blocked1d>
    %rem = arith.remui %idx, %c16 : tensor<128xi32, #blocked1d>
    %div = arith.divui %idx, %c16 : tensor<128xi32, #blocked1d>
    %mul = arith.muli %div, %c96 : tensor<128xi32, #blocked1d>
    %off = arith.addi %rem, %mul : tensor<128xi32, #blocked1d>
    %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x!tt.ptr<f16>, #blocked1d>
    %ptrs = tt.addptr %base, %off : tensor<128x!tt.ptr<f16>, #blocked1d>, tensor<128xi32, #blocked1d>
    tt.store %ptrs, %arg1 : tensor<128x!tt.ptr<f16>, #blocked1d>
    tt.return
  }
}

// -----

// COM: Test 10: 1D strided store with H < numWarps (perWarpH = 0).
// COM: i8, BLOCK=64, W=64, S=128, numWarps=2, tpw=32.
// COM: H = 64/64 = 1. perWarpH = 1/2 = 0 → HW delivery encoding cannot be
// COM: constructed; reshape must bail out and leave a scatter store.

#blocked_h_lt_nw_store = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @test_1d_strided_store_h_lt_numwarps
  // CHECK-NOT: tt.reshape
  // CHECK-NOT: ttig.block_io
  // CHECK: tt.store %{{.*}}, %{{.*}} : tensor<64x!tt.ptr<i8>
  tt.func @test_1d_strided_store_h_lt_numwarps(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: tensor<64xi8, #blocked_h_lt_nw_store>) {
    %idx  = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32, #blocked_h_lt_nw_store>
    %cW   = arith.constant dense<64>  : tensor<64xi32, #blocked_h_lt_nw_store>
    %cS   = arith.constant dense<128> : tensor<64xi32, #blocked_h_lt_nw_store>
    %rem  = arith.remui %idx, %cW : tensor<64xi32, #blocked_h_lt_nw_store>
    %div  = arith.divui %idx, %cW : tensor<64xi32, #blocked_h_lt_nw_store>
    %mul  = arith.muli  %div, %cS : tensor<64xi32, #blocked_h_lt_nw_store>
    %off  = arith.addi  %rem, %mul : tensor<64xi32, #blocked_h_lt_nw_store>
    %base = tt.splat %arg0 : !tt.ptr<i8> -> tensor<64x!tt.ptr<i8>, #blocked_h_lt_nw_store>
    %ptrs = tt.addptr %base, %off : tensor<64x!tt.ptr<i8>, #blocked_h_lt_nw_store>, tensor<64xi32, #blocked_h_lt_nw_store>
    tt.store %ptrs, %arg1 : tensor<64x!tt.ptr<i8>, #blocked_h_lt_nw_store>
    tt.return
  }
}

// -----

// COM: Test 11: 1D strided load with H < numWarps (perWarpH = 0).
// COM: i8, BLOCK=64, W=64, S=128, numWarps=2, tpw=32.
// COM: H = 64/64 = 1. perWarpH = 1/2 = 0 → HW delivery encoding cannot be
// COM: constructed; reshape must bail out and leave a gather load.

#blocked_h_lt_nw_load = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @test_1d_strided_load_h_lt_numwarps
  // CHECK-NOT: tt.reshape
  // CHECK-NOT: ttig.block_io
  // CHECK: tt.load %{{.*}} : tensor<64x!tt.ptr<i8>
  tt.func @test_1d_strided_load_h_lt_numwarps(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}) -> tensor<64xi8, #blocked_h_lt_nw_load> {
    %idx  = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32, #blocked_h_lt_nw_load>
    %cW   = arith.constant dense<64>  : tensor<64xi32, #blocked_h_lt_nw_load>
    %cS   = arith.constant dense<128> : tensor<64xi32, #blocked_h_lt_nw_load>
    %rem  = arith.remui %idx, %cW : tensor<64xi32, #blocked_h_lt_nw_load>
    %div  = arith.divui %idx, %cW : tensor<64xi32, #blocked_h_lt_nw_load>
    %mul  = arith.muli  %div, %cS : tensor<64xi32, #blocked_h_lt_nw_load>
    %off  = arith.addi  %rem, %mul : tensor<64xi32, #blocked_h_lt_nw_load>
    %base = tt.splat %arg0 : !tt.ptr<i8> -> tensor<64x!tt.ptr<i8>, #blocked_h_lt_nw_load>
    %ptrs = tt.addptr %base, %off : tensor<64x!tt.ptr<i8>, #blocked_h_lt_nw_load>, tensor<64xi32, #blocked_h_lt_nw_load>
    %res  = tt.load %ptrs : tensor<64x!tt.ptr<i8>, #blocked_h_lt_nw_load>
    tt.return %res : tensor<64xi8, #blocked_h_lt_nw_load>
  }
}
