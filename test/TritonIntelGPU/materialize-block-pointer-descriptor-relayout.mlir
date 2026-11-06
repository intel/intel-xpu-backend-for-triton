// RUN: triton-opt %s -split-input-file -tritonintelgpu-materialize-block-pointer="enable-descriptor-relayout=true" | FileCheck %s --check-prefixes=CHECK,ON
// RUN: triton-opt %s -split-input-file -tritonintelgpu-materialize-block-pointer | FileCheck %s --check-prefixes=CHECK,OFF
// RUN: env TRITON_INTEL_ENABLE_BLOCK_IO_ALL_LAYOUTS=0 triton-opt %s -split-input-file -tritonintelgpu-materialize-block-pointer="enable-descriptor-relayout=true" | FileCheck %s --check-prefixes=CHECK,OFF
// RUN: triton-opt %s -split-input-file -tritonintelgpu-materialize-block-pointer="enable-descriptor-relayout=true" -tritonintelgpu-remove-layout-conversions | FileCheck %s --check-prefix=RLC
// RUN: triton-opt %s -split-input-file -tritonintelgpu-materialize-block-pointer="enable-descriptor-relayout=true" -tritonintelgpu-remove-layout-conversions -tritonintelgpu-lower-to-2d-block-load | FileCheck %s --check-prefix=LOWER

// COM: Issue #8458. With enable-descriptor-relayout, a row-major descriptor load in a
// COM: function without a dot gets a layout that lowers to 2D block reads. The load is
// COM: marked ttig.block_io_anchor and converted back to its old layout. The option
// COM: off, or TRITON_INTEL_ENABLE_BLOCK_IO_ALL_LAYOUTS=0, leaves the load alone.
// COM: A *_unchanged function gets no anchor either way; the relaid-out cases check the rest.
// COM: The LOWER run adds RLC and LowerTo2DBlockLoad, to check which loads become 2D reads.

// COM: A tt.dot anywhere in the function blocks the relayout, even an unrelated one.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @has_dot_unchanged(
  // CHECK-NOT: ttig.block_io_anchor
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
  // CHECK-LABEL: tt.func public @height_overflow_unchanged(
  // CHECK-NOT: ttig.block_io_anchor
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
  // CHECK-LABEL: tt.func public @width_overflow_unchanged(
  // CHECK-NOT: ttig.block_io_anchor
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
  // CHECK-LABEL: tt.func public @conflicting_reduce_axes_unchanged(
  // CHECK-NOT: ttig.block_io_anchor
  tt.func public @conflicting_reduce_axes_unchanged(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %col_ptrs: tensor<256x!tt.ptr<f32>, #blocked1>, %row_ptrs: tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>, %nseg: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %desc = tt.make_tensor_descriptor %x_ptr, [%nseg, %c256_i32], [%c256_i64, %c1_i64] : <f32>, <16x256xf32>
    %x = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<16x256xf32> -> tensor<16x256xf32, #blocked>
    %1 = "tt.reduce"(%x) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %3 = arith.addf %a, %b : f32
      tt.reduce.return %3 : f32
    }) : (tensor<16x256xf32, #blocked>) -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %0 = "tt.reduce"(%x) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %3 = arith.addf %a, %b : f32
      tt.reduce.return %3 : f32
    }) : (tensor<16x256xf32, #blocked>) -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
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
  // CHECK-LABEL: tt.func public @reduce_behind_trans_unchanged(
  // CHECK-NOT: ttig.block_io_anchor
  tt.func public @reduce_behind_trans_unchanged(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptrs: tensor<256x!tt.ptr<f32>, #blocked1>, %nseg: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %desc = tt.make_tensor_descriptor %x_ptr, [%nseg, %c256_i32], [%c256_i64, %c1_i64] : <f32>, <16x256xf32>
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

// COM: A load whose tile reaches a tt.store is not relaid out: the store keeps its
// COM: own layout, so RLC would leave an SLM convert in front of it.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @ptr_store_unchanged(
  // CHECK-NOT: ttig.block_io_anchor
  tt.func public @ptr_store_unchanged(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptrs: tensor<16x256x!tt.ptr<f32>, #blocked>, %nseg: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %desc = tt.make_tensor_descriptor %x_ptr, [%nseg, %c256_i32], [%c256_i64, %c1_i64] : <f32>, <16x256xf32>
    %x = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<16x256xf32> -> tensor<16x256xf32, #blocked>
    tt.store %out_ptrs, %x : tensor<16x256x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// COM: The same for a tt.descriptor_store.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @desc_store_unchanged(
  // CHECK-NOT: ttig.block_io_anchor
  tt.func public @desc_store_unchanged(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %y_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %nseg: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %desc = tt.make_tensor_descriptor %x_ptr, [%nseg, %c256_i32], [%c256_i64, %c1_i64] : <f32>, <16x256xf32>
    %out_desc = tt.make_tensor_descriptor %y_ptr, [%nseg, %c256_i32], [%c256_i64, %c1_i64] : <f32>, <16x256xf32>
    %x = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<16x256xf32> -> tensor<16x256xf32, #blocked>
    tt.descriptor_store %out_desc[%c0_i32, %c0_i32], %x : !tt.tensordesc<16x256xf32>, tensor<16x256xf32, #blocked>
    tt.return
  }
}

// -----

// COM: The load's layout already lowers to 2D block reads. This does not isolate that
// COM: check: the picked layout is the same one, so the cost check refuses it too.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @already_block_io_unchanged(
  // CHECK-NOT: ttig.block_io_anchor
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

// COM: Only a blocked result is relaid out. This #linear layout equals the 16x256
// COM: #blocked of the other cases, so only the encoding kind refuses it.
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [2, 0], [4, 0], [8, 0]], lane = [[0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], warp = [[0, 128], [1, 0]], block = []}>
#blocked1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @linear_encoding_unchanged(
  // CHECK-NOT: ttig.block_io_anchor
  tt.func public @linear_encoding_unchanged(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptrs: tensor<256x!tt.ptr<f32>, #blocked1>, %nseg: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %desc = tt.make_tensor_descriptor %x_ptr, [%nseg, %c256_i32], [%c256_i64, %c1_i64] : <f32>, <16x256xf32>
    %x = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<16x256xf32> -> tensor<16x256xf32, #linear>
    %0 = "tt.reduce"(%x) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %2 = arith.addf %a, %b : f32
      tt.reduce.return %2 : f32
    }) : (tensor<16x256xf32, #linear>) -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #linear}>>
    %1 = ttg.convert_layout %0 : tensor<256xf32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<256xf32, #blocked1>
    tt.store %out_ptrs, %1 : tensor<256x!tt.ptr<f32>, #blocked1>
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
    // ON-SAME: ttig.desc_padding = 1 : i32
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

// COM: A reduce over the columns: the warps go along the rows, so each warp's
// COM: rows are 8 apart and a 2D block read is 2 rows high. That is 128 messages
// COM: against 32 gathers, so the load is not relaid out.

// COM: A reduce over the columns: the warps go along the rows, so each row stays in
// COM: one warp.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @reduce_cols_unchanged(
  // CHECK-NOT: ttig.block_io_anchor
  // RLC-LABEL: tt.func public @reduce_cols_unchanged(
  tt.func public @reduce_cols_unchanged(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptrs: tensor<64x!tt.ptr<f32>, #blocked1>, %nseg: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %desc = tt.make_tensor_descriptor %x_ptr, [%nseg, %c64_i32], [%c64_i64, %c1_i64] : <f32>, <64x64xf32>
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

// -----

// COM: The tt.dot check is per function: of two functions in one module, only the
// COM: one with the dot keeps its load.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-DAG: #[[$BLOCKED_PER_FUNC:.+]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
  // ON-DAG: #[[$BIO_PER_FUNC:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
  // CHECK-LABEL: tt.func public @dot_function_unchanged(
  // CHECK-NOT: ttig.block_io_anchor
  // RLC-LABEL: tt.func public @dot_function_unchanged(
  tt.func public @dot_function_unchanged(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptrs: tensor<256x!tt.ptr<f32>, #blocked1>, %nseg: i32, %dot_ptrs: tensor<16x16x!tt.ptr<f32>, #blocked2>) {
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %desc = tt.make_tensor_descriptor %x_ptr, [%nseg, %c256_i32], [%c256_i64, %c1_i64] : <f32>, <16x256xf32>
    %x = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<16x256xf32> -> tensor<16x256xf32, #blocked>
    %0 = "tt.reduce"(%x) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %2 = arith.addf %a, %b : f32
      tt.reduce.return %2 : f32
    }) : (tensor<16x256xf32, #blocked>) -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %1 = ttg.convert_layout %0 : tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<256xf32, #blocked1>
    tt.store %out_ptrs, %1 : tensor<256x!tt.ptr<f32>, #blocked1>
    %dot_a = arith.constant dense<1.000000e+00> : tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>
    %dot_b = arith.constant dense<1.000000e+00> : tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked2}>>
    %dot_c = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked2>
    %dot = tt.dot %dot_a, %dot_b, %dot_c : tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>> * tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked2}>> -> tensor<16x16xf32, #blocked2>
    tt.store %dot_ptrs, %dot : tensor<16x16x!tt.ptr<f32>, #blocked2>
    tt.return
  }
  // CHECK-LABEL: tt.func public @dot_free_function(
  tt.func public @dot_free_function(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptrs: tensor<256x!tt.ptr<f32>, #blocked1>, %nseg: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %desc = tt.make_tensor_descriptor %x_ptr, [%nseg, %c256_i32], [%c256_i64, %c1_i64] : <f32>, <16x256xf32>
    // ON: %[[LOAD:.*]] = tt.descriptor_load
    // ON-SAME: ttig.block_io = "row_major"
    // ON-SAME: ttig.block_io_anchor
    // ON-SAME: -> tensor<16x256xf32, #[[$BIO_PER_FUNC]]>
    // ON-NEXT: %[[CVT:.*]] = ttg.convert_layout %[[LOAD]] : tensor<16x256xf32, #[[$BIO_PER_FUNC]]> -> tensor<16x256xf32, #[[$BLOCKED_PER_FUNC]]>
    // ON: "tt.reduce"(%[[CVT]]) <{axis = 0 : i32}>
    // OFF: %[[LOAD:.*]] = tt.descriptor_load
    // OFF-NOT: ttig.block_io_anchor
    // OFF-SAME: !tt.tensordesc<16x256xf32>
    // OFF-SAME: -> tensor<16x256xf32, #[[$BLOCKED_PER_FUNC]]>
    // OFF-NOT: ttg.convert_layout
    // OFF: "tt.reduce"(%[[LOAD]]) <{axis = 0 : i32}>
    %x = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<16x256xf32> -> tensor<16x256xf32, #blocked>
    %0 = "tt.reduce"(%x) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %2 = arith.addf %a, %b : f32
      tt.reduce.return %2 : f32
    }) : (tensor<16x256xf32, #blocked>) -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %1 = ttg.convert_layout %0 : tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<256xf32, #blocked1>
    tt.store %out_ptrs, %1 : tensor<256x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

// COM: The column's sign is not checked: since #8479 was fixed, a negative column
// COM: reads zeros from a 2D block read, so this load is relaid out and lowered to one.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @neg_col_const_relaid_out(
  // LOWER-LABEL: tt.func public @neg_col_const_relaid_out(
  // LOWER: ttig.2d_block_load
  // LOWER: tt.return
  tt.func public @neg_col_const_relaid_out(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptrs: tensor<256x!tt.ptr<f32>, #blocked1>, %nseg: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c-8_i32 = arith.constant -8 : i32
    %c256_i32 = arith.constant 256 : i32
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %desc = tt.make_tensor_descriptor %x_ptr, [%nseg, %c256_i32], [%c256_i64, %c1_i64] : <f32>, <16x256xf32>
    // ON: tt.descriptor_load {{.*}}ttig.block_io_anchor
    // OFF-NOT: ttig.block_io_anchor
    %x = tt.descriptor_load %desc[%c0_i32, %c-8_i32] : !tt.tensordesc<16x256xf32> -> tensor<16x256xf32, #blocked>
    %0 = "tt.reduce"(%x) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %2 = arith.addf %a, %b : f32
      tt.reduce.return %2 : f32
    }) : (tensor<16x256xf32, #blocked>) -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %1 = ttg.convert_layout %0 : tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<256xf32, #blocked1>
    tt.store %out_ptrs, %1 : tensor<256x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

// COM: A descriptor load with fewer than two indices (hand-written IR) is refused.
// COM: Its descriptor is a function argument, so the tag is the one written here.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @no_indices_unchanged(
  // CHECK-NOT: ttig.block_io_anchor
  tt.func public @no_indices_unchanged(%desc: !tt.tensordesc<16x256xf32>, %out_ptrs: tensor<256x!tt.ptr<f32>, #blocked1>) {
    %x = tt.descriptor_load %desc[] {ttig.block_io = "row_major"} : !tt.tensordesc<16x256xf32> -> tensor<16x256xf32, #blocked>
    %0 = "tt.reduce"(%x) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %2 = arith.addf %a, %b : f32
      tt.reduce.return %2 : f32
    }) : (tensor<16x256xf32, #blocked>) -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %1 = ttg.convert_layout %0 : tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<256xf32, #blocked1>
    tt.store %out_ptrs, %1 : tensor<256x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

// COM: With 16 lanes, a reduction whose result is narrowed below 32 bits is not
// COM: relaid out: IGC's vectorizer turns the scalar fptrunc of uniform sub-group
// COM: reduce results into out-of-bounds register writes, and int8 results come out
// COM: wrong. This is the issue kernel's tuned launch, with an f16 output.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [16], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @tpw16_f16_out_unchanged(
  // CHECK-NOT: ttig.block_io_anchor
  tt.func public @tpw16_f16_out_unchanged(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptrs: tensor<128x!tt.ptr<f16>, #blocked1>, %nseg: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i64 = arith.constant 1 : i64
    %c128_i64 = arith.constant 128 : i64
    %desc = tt.make_tensor_descriptor %x_ptr, [%nseg, %c128_i32], [%c128_i64, %c1_i64] : <f32>, <16x128xf32>
    %x = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<16x128xf32> -> tensor<16x128xf32, #blocked>
    %0 = "tt.reduce"(%x) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %3 = arith.addf %a, %b : f32
      tt.reduce.return %3 : f32
    }) : (tensor<16x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %1 = arith.truncf %0 : tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>> to tensor<128xf16, #ttg.slice<{dim = 0, parent = #blocked}>>
    %2 = ttg.convert_layout %1 : tensor<128xf16, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<128xf16, #blocked1>
    tt.store %out_ptrs, %2 : tensor<128x!tt.ptr<f16>, #blocked1>
    tt.return
  }
}

// -----

// COM: The same with an i32 sum truncated to i8.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [16], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @tpw16_i8_out_unchanged(
  // CHECK-NOT: ttig.block_io_anchor
  tt.func public @tpw16_i8_out_unchanged(%x_ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %out_ptrs: tensor<128x!tt.ptr<i8>, #blocked1>, %nseg: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i64 = arith.constant 1 : i64
    %c128_i64 = arith.constant 128 : i64
    %desc = tt.make_tensor_descriptor %x_ptr, [%nseg, %c128_i32], [%c128_i64, %c1_i64] : <i32>, <16x128xi32>
    %x = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<16x128xi32> -> tensor<16x128xi32, #blocked>
    %0 = "tt.reduce"(%x) <{axis = 0 : i32}> ({
    ^bb0(%a: i32, %b: i32):
      %3 = arith.addi %a, %b : i32
      tt.reduce.return %3 : i32
    }) : (tensor<16x128xi32, #blocked>) -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %1 = arith.trunci %0 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> to tensor<128xi8, #ttg.slice<{dim = 0, parent = #blocked}>>
    %2 = ttg.convert_layout %1 : tensor<128xi8, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<128xi8, #blocked1>
    tt.store %out_ptrs, %2 : tensor<128x!tt.ptr<i8>, #blocked1>
    tt.return
  }
}

// -----

// COM: The narrowing may come after a loop: the walk follows the sum through the
// COM: scf.yield to the scf.for result, which is truncated to f16.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @tpw16_loop_f16_out_unchanged(
  // CHECK-NOT: ttig.block_io_anchor
  tt.func public @tpw16_loop_f16_out_unchanged(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptrs: tensor<16x!tt.ptr<f16>, #ttg.slice<{dim = 1, parent = #blocked}>>, %nseg: i32, %ncols: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c1_i64 = arith.constant 1 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %zero = arith.constant dense<0.000000e+00> : tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %desc = tt.make_tensor_descriptor %x_ptr, [%nseg, %c1024_i32], [%c1024_i64, %c1_i64] : <f32>, <16x128xf32>
    %r = scf.for %iv = %c0_i32 to %ncols step %c128_i32 iter_args(%acc = %zero) -> (tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked}>>) : i32 {
      %x = tt.descriptor_load %desc[%c0_i32, %iv] : !tt.tensordesc<16x128xf32> -> tensor<16x128xf32, #blocked>
      %s = "tt.reduce"(%x) <{axis = 1 : i32}> ({
      ^bb0(%a: f32, %b: f32):
        %0 = arith.addf %a, %b : f32
        tt.reduce.return %0 : f32
      }) : (tensor<16x128xf32, #blocked>) -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %sum = arith.addf %acc, %s : tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      scf.yield %sum : tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    }
    %h = arith.truncf %r : tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked}>> to tensor<16xf16, #ttg.slice<{dim = 1, parent = #blocked}>>
    tt.store %out_ptrs, %h : tensor<16x!tt.ptr<f16>, #ttg.slice<{dim = 1, parent = #blocked}>>
    tt.return
  }
}

// -----

// COM: An i1 mask computed from the sum is refused: an i1 layout convert goes through
// COM: shared memory as i8, like the int8 results that came out wrong.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [16], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @tpw16_mask_from_reduce_unchanged(
  // CHECK-NOT: ttig.block_io_anchor
  tt.func public @tpw16_mask_from_reduce_unchanged(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptrs: tensor<128x!tt.ptr<f32>, #blocked1>, %nseg: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i64 = arith.constant 1 : i64
    %c128_i64 = arith.constant 128 : i64
    %desc = tt.make_tensor_descriptor %x_ptr, [%nseg, %c128_i32], [%c128_i64, %c1_i64] : <f32>, <16x128xf32>
    %x = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<16x128xf32> -> tensor<16x128xf32, #blocked>
    %r = "tt.reduce"(%x) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) : (tensor<16x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %z = arith.constant dense<0.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %m = arith.cmpf ogt, %r, %z : tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %mc = ttg.convert_layout %m : tensor<128xi1, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<128xi1, #blocked1>
    %c = ttg.convert_layout %r : tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<128xf32, #blocked1>
    tt.store %out_ptrs, %c, %mc : tensor<128x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

// COM: The walk follows the sum through an scf.if result to the truncation.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [16], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @tpw16_if_f16_out_unchanged(
  // CHECK-NOT: ttig.block_io_anchor
  tt.func public @tpw16_if_f16_out_unchanged(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptrs: tensor<128x!tt.ptr<f16>, #blocked1>, %cond: i1, %nseg: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i64 = arith.constant 1 : i64
    %c128_i64 = arith.constant 128 : i64
    %desc = tt.make_tensor_descriptor %x_ptr, [%nseg, %c128_i32], [%c128_i64, %c1_i64] : <f32>, <16x128xf32>
    %x = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<16x128xf32> -> tensor<16x128xf32, #blocked>
    %r = "tt.reduce"(%x) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) : (tensor<16x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %z = arith.constant dense<0.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %y = scf.if %cond -> (tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>>) {
      scf.yield %r : tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    } else {
      scf.yield %z : tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    }
    %t = arith.truncf %y : tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>> to tensor<128xf16, #ttg.slice<{dim = 0, parent = #blocked}>>
    %c = ttg.convert_layout %t : tensor<128xf16, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<128xf16, #blocked1>
    tt.store %out_ptrs, %c : tensor<128x!tt.ptr<f16>, #blocked1>
    tt.return
  }
}

// -----

// COM: The walk follows the sum into an scf.for as an init value, to the loop result.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [16], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @tpw16_for_init_f16_out_unchanged(
  // CHECK-NOT: ttig.block_io_anchor
  tt.func public @tpw16_for_init_f16_out_unchanged(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptrs: tensor<128x!tt.ptr<f16>, #blocked1>, %n: i32, %nseg: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i64 = arith.constant 1 : i64
    %c128_i64 = arith.constant 128 : i64
    %desc = tt.make_tensor_descriptor %x_ptr, [%nseg, %c128_i32], [%c128_i64, %c1_i64] : <f32>, <16x128xf32>
    %x = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<16x128xf32> -> tensor<16x128xf32, #blocked>
    %r = "tt.reduce"(%x) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) : (tensor<16x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %c1_i32 = arith.constant 1 : i32
    %l = scf.for %i = %c0_i32 to %n step %c1_i32 iter_args(%acc = %r) -> (tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>>) : i32 {
      %d = arith.addf %acc, %acc : tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      scf.yield %d : tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    }
    %t = arith.truncf %l : tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>> to tensor<128xf16, #ttg.slice<{dim = 0, parent = #blocked}>>
    %c = ttg.convert_layout %t : tensor<128xf16, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<128xf16, #blocked1>
    tt.store %out_ptrs, %c : tensor<128x!tt.ptr<f16>, #blocked1>
    tt.return
  }
}

// -----

// COM: A narrowing inside the combine region is refused: the result type does not show it.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [16], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @tpw16_combine_f16_unchanged(
  // CHECK-NOT: ttig.block_io_anchor
  tt.func public @tpw16_combine_f16_unchanged(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptrs: tensor<128x!tt.ptr<f32>, #blocked1>, %nseg: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i64 = arith.constant 1 : i64
    %c128_i64 = arith.constant 128 : i64
    %desc = tt.make_tensor_descriptor %x_ptr, [%nseg, %c128_i32], [%c128_i64, %c1_i64] : <f32>, <16x128xf32>
    %x = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<16x128xf32> -> tensor<16x128xf32, #blocked>
    %r = "tt.reduce"(%x) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %h = arith.truncf %a : f32 to f16
      %e = arith.extf %h : f16 to f32
      %s = arith.addf %e, %b : f32
      tt.reduce.return %s : f32
    }) : (tensor<16x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %c = ttg.convert_layout %r : tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<128xf32, #blocked1>
    tt.store %out_ptrs, %c : tensor<128x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

// COM: Two reductions of the load, and only the second result is narrowed.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [16], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @tpw16_two_reduces_unchanged(
  // CHECK-NOT: ttig.block_io_anchor
  tt.func public @tpw16_two_reduces_unchanged(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptrs: tensor<128x!tt.ptr<f32>, #blocked1>, %out16: tensor<128x!tt.ptr<f16>, #blocked1>, %nseg: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i64 = arith.constant 1 : i64
    %c128_i64 = arith.constant 128 : i64
    %desc = tt.make_tensor_descriptor %x_ptr, [%nseg, %c128_i32], [%c128_i64, %c1_i64] : <f32>, <16x128xf32>
    %x = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<16x128xf32> -> tensor<16x128xf32, #blocked>
    %r = "tt.reduce"(%x) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) : (tensor<16x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %m = "tt.reduce"(%x) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.maxnumf %a, %b : f32
      tt.reduce.return %s : f32
    }) : (tensor<16x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %c = ttg.convert_layout %r : tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<128xf32, #blocked1>
    tt.store %out_ptrs, %c : tensor<128x!tt.ptr<f32>, #blocked1>
    %t = arith.truncf %m : tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>> to tensor<128xf16, #ttg.slice<{dim = 0, parent = #blocked}>>
    %c16 = ttg.convert_layout %t : tensor<128xf16, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<128xf16, #blocked1>
    tt.store %out16, %c16 : tensor<128x!tt.ptr<f16>, #blocked1>
    tt.return
  }
}

// -----

// COM: A user the walk does not understand, tt.print here, refuses even with f32 values.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [16], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @tpw16_unknown_user_unchanged(
  // CHECK-NOT: ttig.block_io_anchor
  tt.func public @tpw16_unknown_user_unchanged(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptrs: tensor<128x!tt.ptr<f32>, #blocked1>, %nseg: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i64 = arith.constant 1 : i64
    %c128_i64 = arith.constant 128 : i64
    %desc = tt.make_tensor_descriptor %x_ptr, [%nseg, %c128_i32], [%c128_i64, %c1_i64] : <f32>, <16x128xf32>
    %x = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<16x128xf32> -> tensor<16x128xf32, #blocked>
    %r = "tt.reduce"(%x) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) : (tensor<16x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    tt.print ": " {hex = false, isSigned = array<i32: 0>} : %r : tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %c = ttg.convert_layout %r : tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<128xf32, #blocked1>
    tt.store %out_ptrs, %c : tensor<128x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

// COM: An fp32 result is not narrowed, so with 16 lanes the load is relaid out.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [16], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-DAG: #[[$BLOCKED_TPW16_F32:.+]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
  // ON-DAG: #[[$BIO_TPW16_F32:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
  // CHECK-LABEL: tt.func public @tpw16_f32_out_relaid_out(
  tt.func public @tpw16_f32_out_relaid_out(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptrs: tensor<128x!tt.ptr<f32>, #blocked1>, %nseg: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i64 = arith.constant 1 : i64
    %c128_i64 = arith.constant 128 : i64
    %desc = tt.make_tensor_descriptor %x_ptr, [%nseg, %c128_i32], [%c128_i64, %c1_i64] : <f32>, <16x128xf32>
    // ON: %[[LOAD:.*]] = tt.descriptor_load
    // ON-SAME: ttig.block_io = "row_major"
    // ON-SAME: ttig.block_io_anchor
    // ON-SAME: -> tensor<16x128xf32, #[[$BIO_TPW16_F32]]>
    // ON-NEXT: %[[CVT:.*]] = ttg.convert_layout %[[LOAD]] : tensor<16x128xf32, #[[$BIO_TPW16_F32]]> -> tensor<16x128xf32, #[[$BLOCKED_TPW16_F32]]>
    // ON: "tt.reduce"(%[[CVT]]) <{axis = 0 : i32}>
    // OFF: %[[LOAD:.*]] = tt.descriptor_load
    // OFF-NOT: ttig.block_io_anchor
    // OFF-SAME: !tt.tensordesc<16x128xf32>
    // OFF-SAME: -> tensor<16x128xf32, #[[$BLOCKED_TPW16_F32]]>
    // OFF-NOT: ttg.convert_layout
    // OFF: "tt.reduce"(%[[LOAD]]) <{axis = 0 : i32}>
    %x = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<16x128xf32> -> tensor<16x128xf32, #blocked>
    %0 = "tt.reduce"(%x) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %2 = arith.addf %a, %b : f32
      tt.reduce.return %2 : f32
    }) : (tensor<16x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %1 = ttg.convert_layout %0 : tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<128xf32, #blocked1>
    tt.store %out_ptrs, %1 : tensor<128x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

// COM: The walk follows the sum through the loop rather than refusing it, so with
// COM: an fp32 result the load in the loop is relaid out.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-DAG: #[[$BLOCKED_TPW16_LOOP_F32:.+]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
  // ON-DAG: #[[$BIO_TPW16_LOOP_F32:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
  // CHECK-LABEL: tt.func public @tpw16_loop_f32_out_relaid_out(
  tt.func public @tpw16_loop_f32_out_relaid_out(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptrs: tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>, %nseg: i32, %ncols: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c1_i64 = arith.constant 1 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %zero = arith.constant dense<0.000000e+00> : tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %desc = tt.make_tensor_descriptor %x_ptr, [%nseg, %c1024_i32], [%c1024_i64, %c1_i64] : <f32>, <16x128xf32>
    // ON: %[[LOAD:.*]] = tt.descriptor_load
    // ON-SAME: ttig.block_io = "row_major"
    // ON-SAME: ttig.block_io_anchor
    // ON-SAME: -> tensor<16x128xf32, #[[$BIO_TPW16_LOOP_F32]]>
    // ON-NEXT: %[[CVT:.*]] = ttg.convert_layout %[[LOAD]] : tensor<16x128xf32, #[[$BIO_TPW16_LOOP_F32]]> -> tensor<16x128xf32, #[[$BLOCKED_TPW16_LOOP_F32]]>
    // ON: "tt.reduce"(%[[CVT]]) <{axis = 1 : i32}>
    // OFF: %[[LOAD:.*]] = tt.descriptor_load
    // OFF-NOT: ttig.block_io_anchor
    // OFF-SAME: !tt.tensordesc<16x128xf32>
    // OFF-SAME: -> tensor<16x128xf32, #[[$BLOCKED_TPW16_LOOP_F32]]>
    // OFF-NOT: ttg.convert_layout
    // OFF: "tt.reduce"(%[[LOAD]]) <{axis = 1 : i32}>
    %r = scf.for %iv = %c0_i32 to %ncols step %c128_i32 iter_args(%acc = %zero) -> (tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked}>>) : i32 {
      %x = tt.descriptor_load %desc[%c0_i32, %iv] : !tt.tensordesc<16x128xf32> -> tensor<16x128xf32, #blocked>
      %s = "tt.reduce"(%x) <{axis = 1 : i32}> ({
      ^bb0(%a: f32, %b: f32):
        %0 = arith.addf %a, %b : f32
        tt.reduce.return %0 : f32
      }) : (tensor<16x128xf32, #blocked>) -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %sum = arith.addf %acc, %s : tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      scf.yield %sum : tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    }
    tt.store %out_ptrs, %r : tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>
    tt.return
  }
}

// -----

// COM: An i32 result is not narrowed either, so with 16 lanes the load is relaid out.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [16], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-DAG: #[[$BLOCKED_TPW16_I32:.+]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
  // ON-DAG: #[[$BIO_TPW16_I32:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
  // CHECK-LABEL: tt.func public @tpw16_i32_out_relaid_out(
  tt.func public @tpw16_i32_out_relaid_out(%x_ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %out_ptrs: tensor<128x!tt.ptr<i32>, #blocked1>, %nseg: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i64 = arith.constant 1 : i64
    %c128_i64 = arith.constant 128 : i64
    %desc = tt.make_tensor_descriptor %x_ptr, [%nseg, %c128_i32], [%c128_i64, %c1_i64] : <i32>, <16x128xi32>
    // ON: %[[LOAD:.*]] = tt.descriptor_load
    // ON-SAME: ttig.block_io = "row_major"
    // ON-SAME: ttig.block_io_anchor
    // ON-SAME: -> tensor<16x128xi32, #[[$BIO_TPW16_I32]]>
    // ON-NEXT: %[[CVT:.*]] = ttg.convert_layout %[[LOAD]] : tensor<16x128xi32, #[[$BIO_TPW16_I32]]> -> tensor<16x128xi32, #[[$BLOCKED_TPW16_I32]]>
    // ON: "tt.reduce"(%[[CVT]]) <{axis = 0 : i32}>
    // OFF: %[[LOAD:.*]] = tt.descriptor_load
    // OFF-NOT: ttig.block_io_anchor
    // OFF-SAME: !tt.tensordesc<16x128xi32>
    // OFF-SAME: -> tensor<16x128xi32, #[[$BLOCKED_TPW16_I32]]>
    // OFF-NOT: ttg.convert_layout
    // OFF: "tt.reduce"(%[[LOAD]]) <{axis = 0 : i32}>
    %x = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<16x128xi32> -> tensor<16x128xi32, #blocked>
    %r = "tt.reduce"(%x) <{axis = 0 : i32}> ({
    ^bb0(%a: i32, %b: i32):
      %s = arith.addi %a, %b : i32
      tt.reduce.return %s : i32
    }) : (tensor<16x128xi32, #blocked>) -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %c = ttg.convert_layout %r : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<128xi32, #blocked1>
    tt.store %out_ptrs, %c : tensor<128x!tt.ptr<i32>, #blocked1>
    tt.return
  }
}

// -----

// COM: With 32 lanes the narrowing does not matter, so this f16 output is relaid out.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-DAG: #[[$BLOCKED_TPW32_F16:.+]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
  // ON-DAG: #[[$BIO_TPW32_F16:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
  // CHECK-LABEL: tt.func public @tpw32_f16_out_relaid_out(
  tt.func public @tpw32_f16_out_relaid_out(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptrs: tensor<256x!tt.ptr<f16>, #blocked1>, %nseg: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %desc = tt.make_tensor_descriptor %x_ptr, [%nseg, %c256_i32], [%c256_i64, %c1_i64] : <f32>, <16x256xf32>
    // ON: %[[LOAD:.*]] = tt.descriptor_load
    // ON-SAME: ttig.block_io = "row_major"
    // ON-SAME: ttig.block_io_anchor
    // ON-SAME: -> tensor<16x256xf32, #[[$BIO_TPW32_F16]]>
    // ON-NEXT: %[[CVT:.*]] = ttg.convert_layout %[[LOAD]] : tensor<16x256xf32, #[[$BIO_TPW32_F16]]> -> tensor<16x256xf32, #[[$BLOCKED_TPW32_F16]]>
    // ON: "tt.reduce"(%[[CVT]]) <{axis = 0 : i32}>
    // OFF: %[[LOAD:.*]] = tt.descriptor_load
    // OFF-NOT: ttig.block_io_anchor
    // OFF-SAME: !tt.tensordesc<16x256xf32>
    // OFF-SAME: -> tensor<16x256xf32, #[[$BLOCKED_TPW32_F16]]>
    // OFF-NOT: ttg.convert_layout
    // OFF: "tt.reduce"(%[[LOAD]]) <{axis = 0 : i32}>
    %x = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<16x256xf32> -> tensor<16x256xf32, #blocked>
    %0 = "tt.reduce"(%x) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %3 = arith.addf %a, %b : f32
      tt.reduce.return %3 : f32
    }) : (tensor<16x256xf32, #blocked>) -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %1 = arith.truncf %0 : tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>> to tensor<256xf16, #ttg.slice<{dim = 0, parent = #blocked}>>
    %2 = ttg.convert_layout %1 : tensor<256xf16, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<256xf16, #blocked1>
    tt.store %out_ptrs, %2 : tensor<256x!tt.ptr<f16>, #blocked1>
    tt.return
  }
}

// -----

// COM: Only a zero-padded descriptor is relaid out: a NaN-padded one exactly one
// COM: element wide returned 0.0 instead of NaN at 16 lanes, for an unknown reason.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @pad_nan_unchanged(
  // CHECK-NOT: ttig.block_io_anchor
  tt.func public @pad_nan_unchanged(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptrs: tensor<256x!tt.ptr<f32>, #blocked1>, %nseg: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %desc = tt.make_tensor_descriptor %x_ptr, [%nseg, %c256_i32], [%c256_i64, %c1_i64] {padding = 2 : i32} : <f32>, <16x256xf32>
    %x = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<16x256xf32> -> tensor<16x256xf32, #blocked>
    %0 = "tt.reduce"(%x) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %2 = arith.addf %a, %b : f32
      tt.reduce.return %2 : f32
    }) : (tensor<16x256xf32, #blocked>) -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %1 = ttg.convert_layout %0 : tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<256xf32, #blocked1>
    tt.store %out_ptrs, %1 : tensor<256x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

// COM: The gates are function-wide: B's descriptor is NaN padded, so neither load is
// COM: relaid out, though the loads meet only through their sums. RLC would otherwise
// COM: move B, and its reduce, into A's layout.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-DAG: #[[$BLOCKED_SIBLING:.+]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
  // ON-DAG: #[[$BIO_SIBLING:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
  // CHECK-LABEL: tt.func public @sibling_pad_nan_unchanged(
  // CHECK-NOT: ttig.block_io_anchor
  // LOWER-LABEL: tt.func public @sibling_pad_nan_unchanged(
  // LOWER-NOT: ttig.2d_block_load
  // LOWER: tt.return
  tt.func public @sibling_pad_nan_unchanged(%a_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %b_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptrs: tensor<256x!tt.ptr<f32>, #blocked1>, %nseg: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c1_i64 = arith.constant 1 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %da = tt.make_tensor_descriptor %a_ptr, [%nseg, %c1024_i32], [%c1024_i64, %c1_i64] : <f32>, <16x256xf32>
    %db = tt.make_tensor_descriptor %b_ptr, [%nseg, %c1024_i32], [%c1024_i64, %c1_i64] {padding = 2 : i32} : <f32>, <16x256xf32>
    %a = tt.descriptor_load %da[%c0_i32, %c0_i32] : !tt.tensordesc<16x256xf32> -> tensor<16x256xf32, #blocked>
    %b = tt.descriptor_load %db[%c0_i32, %c0_i32] : !tt.tensordesc<16x256xf32> -> tensor<16x256xf32, #blocked>
    %ra = "tt.reduce"(%a) <{axis = 0 : i32}> ({
    ^bb0(%x: f32, %y: f32):
      %t = arith.addf %x, %y : f32
      tt.reduce.return %t : f32
    }) : (tensor<16x256xf32, #blocked>) -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %rb = "tt.reduce"(%b) <{axis = 0 : i32}> ({
    ^bb0(%x: f32, %y: f32):
      %t = arith.addf %x, %y : f32
      tt.reduce.return %t : f32
    }) : (tensor<16x256xf32, #blocked>) -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %s = arith.addf %ra, %rb : tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %c = ttg.convert_layout %s : tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<256xf32, #blocked1>
    tt.store %out_ptrs, %c : tensor<256x!tt.ptr<f32>, #blocked1>
    tt.return
  }
  // CHECK-LABEL: tt.func public @sibling_reduces_relaid_out(
  // LOWER-LABEL: tt.func public @sibling_reduces_relaid_out(
  // LOWER-COUNT-2: ttig.2d_block_load
  // LOWER: tt.return
  tt.func public @sibling_reduces_relaid_out(%a_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %b_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptrs: tensor<256x!tt.ptr<f32>, #blocked1>, %nseg: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c1_i64 = arith.constant 1 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %da = tt.make_tensor_descriptor %a_ptr, [%nseg, %c1024_i32], [%c1024_i64, %c1_i64] : <f32>, <16x256xf32>
    %db = tt.make_tensor_descriptor %b_ptr, [%nseg, %c1024_i32], [%c1024_i64, %c1_i64] : <f32>, <16x256xf32>
    // ON-COUNT-2: tt.descriptor_load {{.*}}ttig.block_io_anchor{{.*}} -> tensor<16x256xf32, #[[$BIO_SIBLING]]>
    // OFF-NOT: ttig.block_io_anchor
    // OFF-COUNT-2: tt.descriptor_load {{.*}} -> tensor<16x256xf32, #[[$BLOCKED_SIBLING]]>
    %a = tt.descriptor_load %da[%c0_i32, %c0_i32] : !tt.tensordesc<16x256xf32> -> tensor<16x256xf32, #blocked>
    %b = tt.descriptor_load %db[%c0_i32, %c0_i32] : !tt.tensordesc<16x256xf32> -> tensor<16x256xf32, #blocked>
    %ra = "tt.reduce"(%a) <{axis = 0 : i32}> ({
    ^bb0(%x: f32, %y: f32):
      %t = arith.addf %x, %y : f32
      tt.reduce.return %t : f32
    }) : (tensor<16x256xf32, #blocked>) -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %rb = "tt.reduce"(%b) <{axis = 0 : i32}> ({
    ^bb0(%x: f32, %y: f32):
      %t = arith.addf %x, %y : f32
      tt.reduce.return %t : f32
    }) : (tensor<16x256xf32, #blocked>) -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %s = arith.addf %ra, %rb : tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %c = ttg.convert_layout %s : tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<256xf32, #blocked1>
    tt.store %out_ptrs, %c : tensor<256x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

// COM: The same at 16 lanes: B's sum is narrowed to f16 before it meets A's, so
// COM: neither load is relaid out. With f32 sums both are.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-DAG: #[[$BLOCKED_TPW16_SIBLING:.+]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
  // ON-DAG: #[[$BIO_TPW16_SIBLING:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
  // CHECK-LABEL: tt.func public @tpw16_sibling_narrowed_unchanged(
  // CHECK-NOT: ttig.block_io_anchor
  // LOWER-LABEL: tt.func public @tpw16_sibling_narrowed_unchanged(
  // LOWER-NOT: ttig.2d_block_load
  // LOWER: tt.return
  tt.func public @tpw16_sibling_narrowed_unchanged(%a_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %b_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptrs: tensor<128x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #blocked}>>, %nseg: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i64 = arith.constant 1 : i64
    %c128_i64 = arith.constant 128 : i64
    %da = tt.make_tensor_descriptor %a_ptr, [%nseg, %c128_i32], [%c128_i64, %c1_i64] : <f32>, <16x128xf32>
    %db = tt.make_tensor_descriptor %b_ptr, [%nseg, %c128_i32], [%c128_i64, %c1_i64] : <f32>, <16x128xf32>
    %a = tt.descriptor_load %da[%c0_i32, %c0_i32] : !tt.tensordesc<16x128xf32> -> tensor<16x128xf32, #blocked>
    %b = tt.descriptor_load %db[%c0_i32, %c0_i32] : !tt.tensordesc<16x128xf32> -> tensor<16x128xf32, #blocked>
    %ra = "tt.reduce"(%a) <{axis = 0 : i32}> ({
    ^bb0(%x: f32, %y: f32):
      %t = arith.addf %x, %y : f32
      tt.reduce.return %t : f32
    }) : (tensor<16x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %rb = "tt.reduce"(%b) <{axis = 0 : i32}> ({
    ^bb0(%x: f32, %y: f32):
      %t = arith.addf %x, %y : f32
      tt.reduce.return %t : f32
    }) : (tensor<16x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %hb = arith.truncf %rb : tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>> to tensor<128xf16, #ttg.slice<{dim = 0, parent = #blocked}>>
    %eb = arith.extf %hb : tensor<128xf16, #ttg.slice<{dim = 0, parent = #blocked}>> to tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %s = arith.addf %ra, %eb : tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    tt.store %out_ptrs, %s : tensor<128x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #blocked}>>
    tt.return
  }
  // CHECK-LABEL: tt.func public @tpw16_sibling_reduces_relaid_out(
  // LOWER-LABEL: tt.func public @tpw16_sibling_reduces_relaid_out(
  // LOWER-COUNT-2: ttig.2d_block_load
  // LOWER: tt.return
  tt.func public @tpw16_sibling_reduces_relaid_out(%a_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %b_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptrs: tensor<128x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #blocked}>>, %nseg: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i64 = arith.constant 1 : i64
    %c128_i64 = arith.constant 128 : i64
    %da = tt.make_tensor_descriptor %a_ptr, [%nseg, %c128_i32], [%c128_i64, %c1_i64] : <f32>, <16x128xf32>
    %db = tt.make_tensor_descriptor %b_ptr, [%nseg, %c128_i32], [%c128_i64, %c1_i64] : <f32>, <16x128xf32>
    // ON-COUNT-2: tt.descriptor_load {{.*}}ttig.block_io_anchor{{.*}} -> tensor<16x128xf32, #[[$BIO_TPW16_SIBLING]]>
    // OFF-NOT: ttig.block_io_anchor
    // OFF-COUNT-2: tt.descriptor_load {{.*}} -> tensor<16x128xf32, #[[$BLOCKED_TPW16_SIBLING]]>
    %a = tt.descriptor_load %da[%c0_i32, %c0_i32] : !tt.tensordesc<16x128xf32> -> tensor<16x128xf32, #blocked>
    %b = tt.descriptor_load %db[%c0_i32, %c0_i32] : !tt.tensordesc<16x128xf32> -> tensor<16x128xf32, #blocked>
    %ra = "tt.reduce"(%a) <{axis = 0 : i32}> ({
    ^bb0(%x: f32, %y: f32):
      %t = arith.addf %x, %y : f32
      tt.reduce.return %t : f32
    }) : (tensor<16x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %rb = "tt.reduce"(%b) <{axis = 0 : i32}> ({
    ^bb0(%x: f32, %y: f32):
      %t = arith.addf %x, %y : f32
      tt.reduce.return %t : f32
    }) : (tensor<16x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %s = arith.addf %ra, %rb : tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    tt.store %out_ptrs, %s : tensor<128x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #blocked}>>
    tt.return
  }
}
