// RUN: triton-opt %s -split-input-file -tritonintelgpu-materialize-block-pointer="enable-descriptor-relayout=true" | FileCheck %s

// COM: Issue #8458 stride guard: the relayout is refused when the tiles of programs pid and pid+1 (any grid
// COM: axis) are a power of two above 256 KiB apart, or when that distance is not a compile-time constant.
// COM: Each function is @reduce_rows of materialize-block-pointer-descriptor-relayout.mlir with a new address.


// -----

// COM: Row index pid * 256 at a 1 KiB pitch = 256 KiB, the largest allowed power of two: relaid out.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @index_step_256k(
  // CHECK: %[[LOAD:.*]] = tt.descriptor_load
  // CHECK-SAME: ttig.block_io_anchor
  // CHECK-NEXT: ttg.convert_layout %[[LOAD]]
  tt.func public @index_step_256k(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptrs: tensor<256x!tt.ptr<f32>, #blocked1>, %m: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %pid = tt.get_program_id x : i32
    %desc = tt.make_tensor_descriptor %x_ptr, [%m, %c256_i32], [%c256_i64, %c1_i64] : <f32>, <16x256xf32>
    %row = arith.muli %pid, %c256_i32 : i32
    %x = tt.descriptor_load %desc[%row, %c0_i32] : !tt.tensordesc<16x256xf32> -> tensor<16x256xf32, #blocked>
    %5 = "tt.reduce"(%x) <{axis = 0 : i32}> ({
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

// COM: 528 KiB is not a power of two: relaid out.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @base_step_528k(
  // CHECK: tt.descriptor_load {{.*}}ttig.block_io_anchor
  tt.func public @base_step_528k(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptrs: tensor<256x!tt.ptr<f32>, #blocked1>, %m: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %pid = tt.get_program_id x : i32
    %c135168_i32 = arith.constant 135168 : i32
    %off = arith.muli %pid, %c135168_i32 : i32
    %base = tt.addptr %x_ptr, %off : !tt.ptr<f32>, i32
    %desc = tt.make_tensor_descriptor %base, [%m, %c256_i32], [%c256_i64, %c1_i64] : <f32>, <16x256xf32>
    %row = arith.constant 0 : i32
    %x = tt.descriptor_load %desc[%row, %c0_i32] : !tt.tensordesc<16x256xf32> -> tensor<16x256xf32, #blocked>
    %5 = "tt.reduce"(%x) <{axis = 0 : i32}> ({
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

// COM: A runtime step (pid * %s): refused.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @runtime_step_unchanged(
  // CHECK-NOT: ttig.block_io_anchor
  tt.func public @runtime_step_unchanged(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptrs: tensor<256x!tt.ptr<f32>, #blocked1>, %m: i32, %s: i32 {tt.divisibility = 16 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %pid = tt.get_program_id x : i32
    %off = arith.muli %pid, %s : i32
    %base = tt.addptr %x_ptr, %off : !tt.ptr<f32>, i32
    %desc = tt.make_tensor_descriptor %base, [%m, %c256_i32], [%c256_i64, %c1_i64] : <f32>, <16x256xf32>
    %row = arith.constant 0 : i32
    %x = tt.descriptor_load %desc[%row, %c0_i32] : !tt.tensordesc<16x256xf32> -> tensor<16x256xf32, #blocked>
    %5 = "tt.reduce"(%x) <{axis = 0 : i32}> ({
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

// COM: Base 16 KiB along x, row index 512 KiB along y: refused.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @second_axis_512k_unchanged(
  // CHECK-NOT: ttig.block_io_anchor
  tt.func public @second_axis_512k_unchanged(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptrs: tensor<256x!tt.ptr<f32>, #blocked1>, %m: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %pid = tt.get_program_id x : i32
    %pidy = tt.get_program_id y : i32
    %c4096_i32 = arith.constant 4096 : i32
    %c512_i32 = arith.constant 512 : i32
    %off = arith.muli %pid, %c4096_i32 : i32
    %base = tt.addptr %x_ptr, %off : !tt.ptr<f32>, i32
    %desc = tt.make_tensor_descriptor %base, [%m, %c256_i32], [%c256_i64, %c1_i64] : <f32>, <16x256xf32>
    %row = arith.muli %pidy, %c512_i32 : i32
    %x = tt.descriptor_load %desc[%row, %c0_i32] : !tt.tensordesc<16x256xf32> -> tensor<16x256xf32, #blocked>
    %5 = "tt.reduce"(%x) <{axis = 0 : i32}> ({
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

// COM: Constant row 16 at pitch 256 + pid * 8192: the pitch alone puts tiles 512 KiB apart.
// COM: Any pid-dependent pitch is refused.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @pid_pitch_unchanged(
  // CHECK: tt.descriptor_load
  // CHECK-SAME: ttig.block_io = "row_major"
  // CHECK-NOT: ttig.block_io_anchor
  tt.func public @pid_pitch_unchanged(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptrs: tensor<256x!tt.ptr<f32>, #blocked1>, %m: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    %c256_i32 = arith.constant 256 : i32
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %pid = tt.get_program_id x : i32
    %pid64 = arith.extsi %pid : i32 to i64
    %t = arith.muli %pid64, %c8192_i64 : i64
    %pitch = arith.addi %t, %c256_i64 : i64
    %desc = tt.make_tensor_descriptor %x_ptr, [%m, %c256_i32], [%pitch, %c1_i64] : <f32>, <16x256xf32>
    %x = tt.descriptor_load %desc[%c16_i32, %c0_i32] : !tt.tensordesc<16x256xf32> -> tensor<16x256xf32, #blocked>
    %5 = "tt.reduce"(%x) <{axis = 0 : i32}> ({
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

// COM: Descriptors at x and x + 512 KiB, chosen by pid & 1: each alone has step 0.
// COM: Any choice between definitions is refused.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @pid_select_unchanged(
  // CHECK: tt.descriptor_load
  // CHECK-SAME: ttig.block_io = "row_major"
  // CHECK-NOT: ttig.block_io_anchor
  tt.func public @pid_select_unchanged(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptrs: tensor<256x!tt.ptr<f32>, #blocked1>, %m: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c256_i32 = arith.constant 256 : i32
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %c131072_i32 = arith.constant 131072 : i32
    %pid = tt.get_program_id x : i32
    %y_ptr = tt.addptr %x_ptr, %c131072_i32 : !tt.ptr<f32>, i32
    %da = tt.make_tensor_descriptor %x_ptr, [%m, %c256_i32], [%c256_i64, %c1_i64] : <f32>, <16x256xf32>
    %db = tt.make_tensor_descriptor %y_ptr, [%m, %c256_i32], [%c256_i64, %c1_i64] : <f32>, <16x256xf32>
    %odd = arith.andi %pid, %c1_i32 : i32
    %even = arith.cmpi eq, %odd, %c0_i32 : i32
    %desc = arith.select %even, %da, %db : !tt.tensordesc<16x256xf32>
    %x = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<16x256xf32> -> tensor<16x256xf32, #blocked>
    %5 = "tt.reduce"(%x) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %7 = arith.addf %a, %b : f32
      tt.reduce.return %7 : f32
    }) : (tensor<16x256xf32, #blocked>) -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %6 = ttg.convert_layout %5 : tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<256xf32, #blocked1>
    tt.store %out_ptrs, %6 : tensor<256x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}
