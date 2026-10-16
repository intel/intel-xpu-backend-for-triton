// RUN: triton-opt %s -split-input-file -tritonintelgpu-materialize-block-pointer="enable-descriptor-relayout=true" | FileCheck %s --check-prefixes=CHECK,ON
// RUN: triton-opt %s -split-input-file -tritonintelgpu-materialize-block-pointer | FileCheck %s --check-prefixes=CHECK,OFF
// RUN: env TRITON_INTEL_ENABLE_BLOCK_IO_ALL_LAYOUTS=0 triton-opt %s -split-input-file -tritonintelgpu-materialize-block-pointer="enable-descriptor-relayout=true" | FileCheck %s --check-prefixes=CHECK,OFF
// RUN: triton-opt %s -split-input-file -tritonintelgpu-materialize-block-pointer="enable-descriptor-relayout=true" -tritonintelgpu-remove-layout-conversions | FileCheck %s --check-prefix=RLC

// COM: Issue #8458. With enable-descriptor-relayout, a row-major descriptor load that
// COM: does not feed a dot gets a layout that lowers to 2D block reads. The load is
// COM: marked ttig.block_io_anchor and converted back to its old layout. The option
// COM: off, or TRITON_INTEL_ENABLE_BLOCK_IO_ALL_LAYOUTS=0, leaves the load alone.
// COM: A *_unchanged load keeps its layout, with no anchor and no convert, either way.

// COM: A tt.dot anywhere in the function blocks the relayout, even an unrelated one.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-DAG: #[[$BLOCKED_DOT:.+]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
  // CHECK-LABEL: tt.func public @has_dot_unchanged(
  // RLC-LABEL: tt.func public @has_dot_unchanged(
  tt.func public @has_dot_unchanged(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %w_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptrs: tensor<256x!tt.ptr<f32>, #blocked1>, %nseg: i32, %dot_ptrs: tensor<16x16x!tt.ptr<f32>, #blocked2>) {
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %seg = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %0 = tt.splat %w_ptr : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.addptr %0, %seg : tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %w = tt.load %1 : tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>
    %desc = tt.make_tensor_descriptor %x_ptr, [%nseg, %c256_i32], [%c256_i64, %c1_i64] : <f32>, <16x256xf32>
    // CHECK: %[[LOAD:.*]] = tt.descriptor_load
    // CHECK-NOT: ttig.block_io_anchor
    // CHECK-SAME: !tt.tensordesc<16x256xf32>
    // CHECK-SAME: -> tensor<16x256xf32, #[[$BLOCKED_DOT]]>
    // CHECK-NOT: ttg.convert_layout
    // CHECK: arith.mulf %[[LOAD]],
    %x = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<16x256xf32> -> tensor<16x256xf32, #blocked>
    %2 = tt.expand_dims %w {axis = 1 : i32} : tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xf32, #blocked>
    %3 = tt.broadcast %2 : tensor<16x1xf32, #blocked> -> tensor<16x256xf32, #blocked>
    %4 = arith.mulf %x, %3 : tensor<16x256xf32, #blocked>
    %5 = "tt.reduce"(%4) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %7 = arith.addf %a, %b : f32
      tt.reduce.return %7 : f32
    }) : (tensor<16x256xf32, #blocked>) -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %6 = ttg.convert_layout %5 : tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<256xf32, #blocked1>
    tt.store %out_ptrs, %6 : tensor<256x!tt.ptr<f32>, #blocked1>
    %dot_a = arith.constant dense<1.000000e+00> : tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>
    %dot_b = arith.constant dense<1.000000e+00> : tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked2}>>
    %dot_c = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked2>
    %dot = tt.dot %dot_a, %dot_b, %dot_c : tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>> * tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked2}>> -> tensor<16x16xf32, #blocked2>
    tt.store %dot_ptrs, %dot : tensor<16x16x!tt.ptr<f32>, #blocked2>
    tt.return
  }
}

// -----

// COM: MaterializeBlockPointer tags the load, but a constant height of 2^24 + 1 rows
// COM: does not fit the 2D block message, so the shared lowering predicate refuses it.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-DAG: #[[$BLOCKED_HEIGHT:.+]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
  // CHECK-LABEL: tt.func public @height_overflow_unchanged(
  // RLC-LABEL: tt.func public @height_overflow_unchanged(
  tt.func public @height_overflow_unchanged(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %w_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptrs: tensor<256x!tt.ptr<f32>, #blocked1>) {
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %c16777217_i32 = arith.constant 16777217 : i32
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %seg = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %0 = tt.splat %w_ptr : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.addptr %0, %seg : tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %w = tt.load %1 : tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>
    %desc = tt.make_tensor_descriptor %x_ptr, [%c16777217_i32, %c256_i32], [%c256_i64, %c1_i64] : <f32>, <16x256xf32>
    // CHECK: %[[LOAD:.*]] = tt.descriptor_load
    // CHECK-SAME: ttig.block_io = "row_major"
    // CHECK-NOT: ttig.block_io_anchor
    // CHECK-SAME: !tt.tensordesc<16x256xf32>
    // CHECK-SAME: -> tensor<16x256xf32, #[[$BLOCKED_HEIGHT]]>
    // CHECK-NOT: ttg.convert_layout
    // CHECK: arith.mulf %[[LOAD]],
    %x = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<16x256xf32> -> tensor<16x256xf32, #blocked>
    %2 = tt.expand_dims %w {axis = 1 : i32} : tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xf32, #blocked>
    %3 = tt.broadcast %2 : tensor<16x1xf32, #blocked> -> tensor<16x256xf32, #blocked>
    %4 = arith.mulf %x, %3 : tensor<16x256xf32, #blocked>
    %5 = "tt.reduce"(%4) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %7 = arith.addf %a, %b : f32
      tt.reduce.return %7 : f32
    }) : (tensor<16x256xf32, #blocked>) -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %6 = ttg.convert_layout %5 : tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<256xf32, #blocked1>
    tt.store %out_ptrs, %6 : tensor<256x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

// COM: The same for a constant width of 4194305 f32 (2^24 + 4 bytes), with an
// COM: in-range, 16-byte-aligned pitch.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-DAG: #[[$BLOCKED_WIDTH:.+]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
  // CHECK-LABEL: tt.func public @width_overflow_unchanged(
  // RLC-LABEL: tt.func public @width_overflow_unchanged(
  tt.func public @width_overflow_unchanged(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %w_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptrs: tensor<256x!tt.ptr<f32>, #blocked1>, %nseg: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c4194305_i32 = arith.constant 4194305 : i32
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %seg = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %0 = tt.splat %w_ptr : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.addptr %0, %seg : tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %w = tt.load %1 : tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>
    %desc = tt.make_tensor_descriptor %x_ptr, [%nseg, %c4194305_i32], [%c256_i64, %c1_i64] : <f32>, <16x256xf32>
    // CHECK: %[[LOAD:.*]] = tt.descriptor_load
    // CHECK-SAME: ttig.block_io = "row_major"
    // CHECK-NOT: ttig.block_io_anchor
    // CHECK-SAME: !tt.tensordesc<16x256xf32>
    // CHECK-SAME: -> tensor<16x256xf32, #[[$BLOCKED_WIDTH]]>
    // CHECK-NOT: ttg.convert_layout
    // CHECK: arith.mulf %[[LOAD]],
    %x = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<16x256xf32> -> tensor<16x256xf32, #blocked>
    %2 = tt.expand_dims %w {axis = 1 : i32} : tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xf32, #blocked>
    %3 = tt.broadcast %2 : tensor<16x1xf32, #blocked> -> tensor<16x256xf32, #blocked>
    %4 = arith.mulf %x, %3 : tensor<16x256xf32, #blocked>
    %5 = "tt.reduce"(%4) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %7 = arith.addf %a, %b : f32
      tt.reduce.return %7 : f32
    }) : (tensor<16x256xf32, #blocked>) -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %6 = ttg.convert_layout %5 : tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<256xf32, #blocked1>
    tt.store %out_ptrs, %6 : tensor<256x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

// COM: Reduces on both axes leave no single dim to keep inside a warp.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-DAG: #[[$BLOCKED_AXES:.+]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
  // CHECK-LABEL: tt.func public @conflicting_reduce_axes_unchanged(
  // RLC-LABEL: tt.func public @conflicting_reduce_axes_unchanged(
  tt.func public @conflicting_reduce_axes_unchanged(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %col_ptrs: tensor<256x!tt.ptr<f32>, #blocked1>, %row_ptrs: tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>, %nseg: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %desc = tt.make_tensor_descriptor %x_ptr, [%nseg, %c256_i32], [%c256_i64, %c1_i64] : <f32>, <16x256xf32>
    // CHECK: %[[LOAD:.*]] = tt.descriptor_load
    // CHECK-NOT: ttig.block_io_anchor
    // CHECK-SAME: !tt.tensordesc<16x256xf32>
    // CHECK-SAME: -> tensor<16x256xf32, #[[$BLOCKED_AXES]]>
    // CHECK-NOT: ttg.convert_layout
    // CHECK: "tt.reduce"(%[[LOAD]]) <{axis = 0 : i32}>
    // CHECK: "tt.reduce"(%[[LOAD]]) <{axis = 1 : i32}>
    %x = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<16x256xf32> -> tensor<16x256xf32, #blocked>
    %0 = "tt.reduce"(%x) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %3 = arith.addf %a, %b : f32
      tt.reduce.return %3 : f32
    }) : (tensor<16x256xf32, #blocked>) -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %1 = "tt.reduce"(%x) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %3 = arith.addf %a, %b : f32
      tt.reduce.return %3 : f32
    }) : (tensor<16x256xf32, #blocked>) -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %2 = ttg.convert_layout %0 : tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<256xf32, #blocked1>
    tt.store %col_ptrs, %2 : tensor<256x!tt.ptr<f32>, #blocked1>
    tt.store %row_ptrs, %1 : tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>
    tt.return
  }
}

// -----

// COM: The use analysis does not look through tt.trans, so it refuses.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-DAG: #[[$BLOCKED_TRANS:.+]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
  // CHECK-LABEL: tt.func public @reduce_behind_trans_unchanged(
  // RLC-LABEL: tt.func public @reduce_behind_trans_unchanged(
  tt.func public @reduce_behind_trans_unchanged(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptrs: tensor<256x!tt.ptr<f32>, #blocked1>, %nseg: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %desc = tt.make_tensor_descriptor %x_ptr, [%nseg, %c256_i32], [%c256_i64, %c1_i64] : <f32>, <16x256xf32>
    // CHECK: %[[LOAD:.*]] = tt.descriptor_load
    // CHECK-NOT: ttig.block_io_anchor
    // CHECK-SAME: !tt.tensordesc<16x256xf32>
    // CHECK-SAME: -> tensor<16x256xf32, #[[$BLOCKED_TRANS]]>
    // CHECK-NOT: ttg.convert_layout
    // CHECK: tt.trans %[[LOAD]]
    %x = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<16x256xf32> -> tensor<16x256xf32, #blocked>
    %0 = tt.trans %x {order = array<i32: 1, 0>} : tensor<16x256xf32, #blocked> -> tensor<256x16xf32, #blocked2>
    %1 = "tt.reduce"(%0) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %3 = arith.addf %a, %b : f32
      tt.reduce.return %3 : f32
    }) : (tensor<256x16xf32, #blocked2>) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %2 = ttg.convert_layout %1 : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<256xf32, #blocked1>
    tt.store %out_ptrs, %2 : tensor<256x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

// COM: The load's layout already lowers to 2D block reads.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-DAG: #[[$BLOCKED_BIO:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
  // CHECK-LABEL: tt.func public @already_block_io_unchanged(
  // RLC-LABEL: tt.func public @already_block_io_unchanged(
  tt.func public @already_block_io_unchanged(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %w_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptrs: tensor<256x!tt.ptr<f32>, #blocked1>, %nseg: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %seg = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %0 = tt.splat %w_ptr : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.addptr %0, %seg : tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %w = tt.load %1 : tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>
    %desc = tt.make_tensor_descriptor %x_ptr, [%nseg, %c256_i32], [%c256_i64, %c1_i64] : <f32>, <16x256xf32>
    // CHECK: %[[LOAD:.*]] = tt.descriptor_load
    // CHECK-NOT: ttig.block_io_anchor
    // CHECK-SAME: !tt.tensordesc<16x256xf32>
    // CHECK-SAME: -> tensor<16x256xf32, #[[$BLOCKED_BIO]]>
    // CHECK-NOT: ttg.convert_layout
    // CHECK: arith.mulf %[[LOAD]],
    %x = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<16x256xf32> -> tensor<16x256xf32, #blocked>
    %2 = tt.expand_dims %w {axis = 1 : i32} : tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xf32, #blocked>
    %3 = tt.broadcast %2 : tensor<16x1xf32, #blocked> -> tensor<16x256xf32, #blocked>
    %4 = arith.mulf %x, %3 : tensor<16x256xf32, #blocked>
    %5 = "tt.reduce"(%4) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %7 = arith.addf %a, %b : f32
      tt.reduce.return %7 : f32
    }) : (tensor<16x256xf32, #blocked>) -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %6 = ttg.convert_layout %5 : tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<256xf32, #blocked1>
    tt.store %out_ptrs, %6 : tensor<256x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

// COM: The issue kernel: load, multiply by a row weight, reduce over the rows. The
// COM: weight is a 16-element pointer load, too small to be an RLC anchor, so RLC
// COM: has no competing layout to resolve. RLC carries the anchored layout to the
// COM: reduce: no convert is left between the load and the reduce.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-DAG: #[[$BLOCKED_ROWS:.+]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
  // ON-DAG: #[[$BIO_ROWS:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
  // RLC-DAG: #[[$RLC_BIO_ROWS:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
  // CHECK-LABEL: tt.func public @reduce_rows(
  // RLC-LABEL: tt.func public @reduce_rows(
  tt.func public @reduce_rows(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %w_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptrs: tensor<256x!tt.ptr<f32>, #blocked1>, %nseg: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %seg = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %0 = tt.splat %w_ptr : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.addptr %0, %seg : tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %w = tt.load %1 : tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>
    %desc = tt.make_tensor_descriptor %x_ptr, [%nseg, %c256_i32], [%c256_i64, %c1_i64] : <f32>, <16x256xf32>
    // ON: %[[LOAD:.*]] = tt.descriptor_load
    // ON-SAME: ttig.block_io = "row_major"
    // ON-SAME: ttig.block_io_anchor
    // ON-SAME: -> tensor<16x256xf32, #[[$BIO_ROWS]]>
    // ON-NEXT: %[[CVT:.*]] = ttg.convert_layout %[[LOAD]] : tensor<16x256xf32, #[[$BIO_ROWS]]> -> tensor<16x256xf32, #[[$BLOCKED_ROWS]]>
    // ON: arith.mulf %[[CVT]],
    // OFF: %[[LOAD:.*]] = tt.descriptor_load
    // OFF-NOT: ttig.block_io_anchor
    // OFF-SAME: !tt.tensordesc<16x256xf32>
    // OFF-SAME: -> tensor<16x256xf32, #[[$BLOCKED_ROWS]]>
    // OFF-NOT: ttg.convert_layout
    // OFF: arith.mulf %[[LOAD]],
    // RLC: %[[LOAD:.*]] = tt.descriptor_load
    // RLC-SAME: -> tensor<16x256xf32, #[[$RLC_BIO_ROWS]]>
    // RLC: %[[MUL:.*]] = arith.mulf %[[LOAD]], {{.*}} : tensor<16x256xf32, #[[$RLC_BIO_ROWS]]>
    // RLC: "tt.reduce"(%[[MUL]]) <{axis = 0 : i32}>
    // RLC: }) : (tensor<16x256xf32, #[[$RLC_BIO_ROWS]]>) -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #[[$RLC_BIO_ROWS]]}>>
    %x = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<16x256xf32> -> tensor<16x256xf32, #blocked>
    %2 = tt.expand_dims %w {axis = 1 : i32} : tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xf32, #blocked>
    %3 = tt.broadcast %2 : tensor<16x1xf32, #blocked> -> tensor<16x256xf32, #blocked>
    %4 = arith.mulf %x, %3 : tensor<16x256xf32, #blocked>
    %5 = "tt.reduce"(%4) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %7 = arith.addf %a, %b : f32
      tt.reduce.return %7 : f32
    }) : (tensor<16x256xf32, #blocked>) -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %6 = ttg.convert_layout %5 : tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<256xf32, #blocked1>
    tt.store %out_ptrs, %6 : tensor<256x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

// COM: A reduce over the columns: the warps go along the rows, so each row stays in
// COM: one warp.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-DAG: #[[$BLOCKED_COLS:.+]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
  // ON-DAG: #[[$BIO_COLS:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
  // CHECK-LABEL: tt.func public @reduce_cols(
  // RLC-LABEL: tt.func public @reduce_cols(
  tt.func public @reduce_cols(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptrs: tensor<64x!tt.ptr<f32>, #blocked1>, %nseg: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %desc = tt.make_tensor_descriptor %x_ptr, [%nseg, %c64_i32], [%c64_i64, %c1_i64] : <f32>, <64x64xf32>
    // ON: %[[LOAD:.*]] = tt.descriptor_load
    // ON-SAME: ttig.block_io = "row_major"
    // ON-SAME: ttig.block_io_anchor
    // ON-SAME: -> tensor<64x64xf32, #[[$BIO_COLS]]>
    // ON-NEXT: %[[CVT:.*]] = ttg.convert_layout %[[LOAD]] : tensor<64x64xf32, #[[$BIO_COLS]]> -> tensor<64x64xf32, #[[$BLOCKED_COLS]]>
    // ON: "tt.reduce"(%[[CVT]]) <{axis = 1 : i32}>
    // OFF: %[[LOAD:.*]] = tt.descriptor_load
    // OFF-NOT: ttig.block_io_anchor
    // OFF-SAME: !tt.tensordesc<64x64xf32>
    // OFF-SAME: -> tensor<64x64xf32, #[[$BLOCKED_COLS]]>
    // OFF-NOT: ttg.convert_layout
    // OFF: "tt.reduce"(%[[LOAD]]) <{axis = 1 : i32}>
    %x = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x64xf32> -> tensor<64x64xf32, #blocked>
    %0 = "tt.reduce"(%x) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %2 = arith.addf %a, %b : f32
      tt.reduce.return %2 : f32
    }) : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %1 = ttg.convert_layout %0 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64xf32, #blocked1>
    tt.store %out_ptrs, %1 : tensor<64x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

// COM: Two loads feeding one mulf get the same layout, so RLC removes both converts.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-DAG: #[[$BLOCKED_TWO:.+]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
  // ON-DAG: #[[$BIO_TWO:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
  // RLC-DAG: #[[$RLC_BIO_TWO:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
  // CHECK-LABEL: tt.func public @two_loads_same_consumer(
  // RLC-LABEL: tt.func public @two_loads_same_consumer(
  tt.func public @two_loads_same_consumer(%a_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %b_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptrs: tensor<256x!tt.ptr<f32>, #blocked1>, %nseg: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %desc_a = tt.make_tensor_descriptor %a_ptr, [%nseg, %c256_i32], [%c256_i64, %c1_i64] : <f32>, <16x256xf32>
    %desc_b = tt.make_tensor_descriptor %b_ptr, [%nseg, %c256_i32], [%c256_i64, %c1_i64] : <f32>, <16x256xf32>
    // ON: %[[A:.*]] = tt.descriptor_load
    // ON-SAME: ttig.block_io = "row_major"
    // ON-SAME: ttig.block_io_anchor
    // ON-SAME: -> tensor<16x256xf32, #[[$BIO_TWO]]>
    // ON-NEXT: %[[CVT_A:.*]] = ttg.convert_layout %[[A]] : tensor<16x256xf32, #[[$BIO_TWO]]> -> tensor<16x256xf32, #[[$BLOCKED_TWO]]>
    // ON: %[[B:.*]] = tt.descriptor_load
    // ON-SAME: ttig.block_io = "row_major"
    // ON-SAME: ttig.block_io_anchor
    // ON-SAME: -> tensor<16x256xf32, #[[$BIO_TWO]]>
    // ON-NEXT: %[[CVT_B:.*]] = ttg.convert_layout %[[B]] : tensor<16x256xf32, #[[$BIO_TWO]]> -> tensor<16x256xf32, #[[$BLOCKED_TWO]]>
    // ON: arith.mulf %[[CVT_A]], %[[CVT_B]] : tensor<16x256xf32, #[[$BLOCKED_TWO]]>
    // OFF: %[[A:.*]] = tt.descriptor_load
    // OFF-NOT: ttig.block_io_anchor
    // OFF-SAME: !tt.tensordesc<16x256xf32>
    // OFF-SAME: -> tensor<16x256xf32, #[[$BLOCKED_TWO]]>
    // OFF: %[[B:.*]] = tt.descriptor_load
    // OFF-NOT: ttig.block_io_anchor
    // OFF-SAME: !tt.tensordesc<16x256xf32>
    // OFF-SAME: -> tensor<16x256xf32, #[[$BLOCKED_TWO]]>
    // OFF-NOT: ttg.convert_layout
    // OFF: arith.mulf %[[A]], %[[B]] : tensor<16x256xf32, #[[$BLOCKED_TWO]]>
    // RLC: %[[A:.*]] = tt.descriptor_load
    // RLC-SAME: -> tensor<16x256xf32, #[[$RLC_BIO_TWO]]>
    // RLC: %[[B:.*]] = tt.descriptor_load
    // RLC-SAME: -> tensor<16x256xf32, #[[$RLC_BIO_TWO]]>
    // RLC: %[[MUL:.*]] = arith.mulf %[[A]], %[[B]] : tensor<16x256xf32, #[[$RLC_BIO_TWO]]>
    // RLC: "tt.reduce"(%[[MUL]]) <{axis = 0 : i32}>
    // RLC: }) : (tensor<16x256xf32, #[[$RLC_BIO_TWO]]>) -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #[[$RLC_BIO_TWO]]}>>
    %a = tt.descriptor_load %desc_a[%c0_i32, %c0_i32] : !tt.tensordesc<16x256xf32> -> tensor<16x256xf32, #blocked>
    %b = tt.descriptor_load %desc_b[%c0_i32, %c0_i32] : !tt.tensordesc<16x256xf32> -> tensor<16x256xf32, #blocked>
    %0 = arith.mulf %a, %b : tensor<16x256xf32, #blocked>
    %1 = "tt.reduce"(%0) <{axis = 0 : i32}> ({
    ^bb0(%x: f32, %y: f32):
      %3 = arith.addf %x, %y : f32
      tt.reduce.return %3 : f32
    }) : (tensor<16x256xf32, #blocked>) -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %2 = ttg.convert_layout %1 : tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<256xf32, #blocked1>
    tt.store %out_ptrs, %2 : tensor<256x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}
