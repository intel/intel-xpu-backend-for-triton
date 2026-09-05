// RUN: triton-opt %s -split-input-file --tritonintelgpu-remove-layout-conversions | FileCheck %s

// COM: NEGATIVE case — reject rematerialization when pointer is loaded and stored with different encodings.
// COM: This prevents a race condition where the same pointer is accessed with different thread-to-address mappings.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_remat_different_encoding
  tt.func @reject_remat_different_encoding(%in_out_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : i32
    %c128 = arith.constant 128 : i32
    %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>

    // Load from in_out_ptr with #blocked encoding
    %ptr_splat = tt.splat %in_out_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %ptrs = tt.addptr %ptr_splat, %range : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %load = tt.load %ptrs : tensor<128x!tt.ptr<f32>, #blocked>

    // Compute something with the loaded data
    %cst = arith.constant dense<2.0> : tensor<128xf32, #blocked>
    %result = arith.mulf %load, %cst : tensor<128xf32, #blocked>

    // Store back to in_out_ptr with #blocked encoding (establishes the encoding for this pointer)
    tt.store %ptrs, %result : tensor<128x!tt.ptr<f32>, #blocked>

    // Convert to #blocked1 for out_ptr store
    // CHECK: ttg.convert_layout
    %converted = ttg.convert_layout %result : tensor<128xf32, #blocked> -> tensor<128xf32, #blocked1>

    // Store to out_ptr with #blocked1 encoding
    %out_ptr_splat = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked1>
    %range1 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked1>
    %out_ptrs = tt.addptr %out_ptr_splat, %range1 : tensor<128x!tt.ptr<f32>, #blocked1>, tensor<128xi32, #blocked1>
    tt.store %out_ptrs, %converted : tensor<128x!tt.ptr<f32>, #blocked1>

    // COM: The convert_layout should NOT be removed because rematerializing
    // COM: the load in #blocked1 would create a race (different encoding = different thread mapping).

    tt.return
  }
}

// -----

// COM: POSITIVE case — allow rematerialization when pointer is loaded and stored with same encoding.
// COM: This is safe because the same encoding means the same thread-to-address mapping.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @allow_remat_same_encoding
  tt.func @allow_remat_same_encoding(%in_out_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : i32
    %c128 = arith.constant 128 : i32
    %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>

    // Load from in_out_ptr with #blocked encoding
    %ptr_splat = tt.splat %in_out_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %ptrs = tt.addptr %ptr_splat, %range : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %load = tt.load %ptrs : tensor<128x!tt.ptr<f32>, #blocked>

    // Compute something with the loaded data
    %cst = arith.constant dense<2.0> : tensor<128xf32, #blocked>
    %result = arith.mulf %load, %cst : tensor<128xf32, #blocked>

    // Store back to in_out_ptr with #blocked encoding
    tt.store %ptrs, %result : tensor<128x!tt.ptr<f32>, #blocked>

    // Use result again for out_ptr (same encoding)
    %out_ptr_splat = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %out_ptrs = tt.addptr %out_ptr_splat, %range : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    tt.store %out_ptrs, %result : tensor<128x!tt.ptr<f32>, #blocked>

    // CHECK-NOT: ttg.convert_layout
    // COM: No convert_layout should appear because both stores use the same encoding,
    // COM: so rematerialization is safe (no race condition).

    tt.return
  }
}
