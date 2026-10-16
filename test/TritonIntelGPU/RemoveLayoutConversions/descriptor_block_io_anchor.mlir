// RUN: triton-opt %s -split-input-file -tritonintelgpu-remove-layout-conversions | FileCheck %s

// COM: Issue #8458. A `ttig.block_io` load is cheap, so it is not a layout anchor.
// COM: `ttig.block_io_anchor` makes a descriptor load relaid out to `#bio` for 2D block IO
// COM: one, so `#bio` propagates to its consumers instead of being converted back.

// COM: Without the anchor (today), the 2D block IO cost check keeps the load in `#bio`, but
// COM: the convert back to `#blocked` survives, and the mulf and reduce stay in `#blocked`.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#bio = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-DAG: #[[$BLOCKED:.+]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
  // CHECK-DAG: #[[$BIO:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
  // CHECK-LABEL: tt.func public @not_anchored
  tt.func public @not_anchored(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %w_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptrs: tensor<256x!tt.ptr<f32>, #blocked1>, %nseg: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %seg = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %0 = tt.splat %w_ptr : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.addptr %0, %seg : tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %w = tt.load %1 : tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>
    %desc = tt.make_tensor_descriptor %x_ptr, [%nseg, %c256_i32], [%c256_i64, %c1_i64] : <f32>, <16x256xf32>
    // CHECK: %[[LOAD:.*]] = tt.descriptor_load {{.*}} -> tensor<16x256xf32, #[[$BIO]]>
    // CHECK: %[[CVT:.*]] = ttg.convert_layout %[[LOAD]] : tensor<16x256xf32, #[[$BIO]]> -> tensor<16x256xf32, #[[$BLOCKED]]>
    %x = tt.descriptor_load %desc[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<16x256xf32> -> tensor<16x256xf32, #bio>
    %2 = ttg.convert_layout %x : tensor<16x256xf32, #bio> -> tensor<16x256xf32, #blocked>
    %3 = tt.expand_dims %w {axis = 1 : i32} : tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xf32, #blocked>
    %4 = tt.broadcast %3 : tensor<16x1xf32, #blocked> -> tensor<16x256xf32, #blocked>
    // CHECK: %[[MUL:.*]] = arith.mulf %[[CVT]], {{.*}} : tensor<16x256xf32, #[[$BLOCKED]]>
    %5 = arith.mulf %2, %4 : tensor<16x256xf32, #blocked>
    // CHECK: "tt.reduce"(%[[MUL]]) <{axis = 0 : i32}>
    // CHECK: }) : (tensor<16x256xf32, #[[$BLOCKED]]>) -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #[[$BLOCKED]]}>>
    %6 = "tt.reduce"(%5) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %8 = arith.addf %a, %b : f32
      tt.reduce.return %8 : f32
    }) : (tensor<16x256xf32, #blocked>) -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %7 = ttg.convert_layout %6 : tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<256xf32, #blocked1>
    tt.store %out_ptrs, %7 : tensor<256x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

// COM: With the anchor, the mulf and the reduce take `#bio` too: no 16x256 convert remains.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#bio = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-DAG: #[[$BIO:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
  // CHECK-LABEL: tt.func public @anchored
  tt.func public @anchored(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %w_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptrs: tensor<256x!tt.ptr<f32>, #blocked1>, %nseg: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %seg = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %0 = tt.splat %w_ptr : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.addptr %0, %seg : tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %w = tt.load %1 : tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>
    %desc = tt.make_tensor_descriptor %x_ptr, [%nseg, %c256_i32], [%c256_i64, %c1_i64] : <f32>, <16x256xf32>
    // CHECK: %[[LOAD:.*]] = tt.descriptor_load {{.*}} -> tensor<16x256xf32, #[[$BIO]]>
    // CHECK-NOT: ttg.convert_layout {{.*}}16x256xf32
    %x = tt.descriptor_load %desc[%c0_i32, %c0_i32] {ttig.block_io = "row_major", ttig.block_io_anchor} : !tt.tensordesc<16x256xf32> -> tensor<16x256xf32, #bio>
    %2 = ttg.convert_layout %x : tensor<16x256xf32, #bio> -> tensor<16x256xf32, #blocked>
    %3 = tt.expand_dims %w {axis = 1 : i32} : tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xf32, #blocked>
    %4 = tt.broadcast %3 : tensor<16x1xf32, #blocked> -> tensor<16x256xf32, #blocked>
    // CHECK: %[[MUL:.*]] = arith.mulf
    // CHECK-SAME: %[[LOAD]], {{.*}} : tensor<16x256xf32, #[[$BIO]]>
    %5 = arith.mulf %2, %4 : tensor<16x256xf32, #blocked>
    // CHECK: "tt.reduce"(%[[MUL]]) <{axis = 0 : i32}>
    // CHECK: }) : (tensor<16x256xf32, #[[$BIO]]>) -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #[[$BIO]]}>>
    %6 = "tt.reduce"(%5) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %8 = arith.addf %a, %b : f32
      tt.reduce.return %8 : f32
    }) : (tensor<16x256xf32, #blocked>) -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %7 = ttg.convert_layout %6 : tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<256xf32, #blocked1>
    tt.store %out_ptrs, %7 : tensor<256x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}
