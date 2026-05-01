// RUN: triton-opt %s -split-input-file --tritonintelgpu-fixup-descriptor-encoding | FileCheck %s

// COM: Test 1 — descriptor_load with a non-2D-block-IO-compatible blocked encoding is
// COM: reassigned to sizePerThread=[1,1] with lanes concentrated along the contiguous
// COM: (innermost) dimension, and a ttg.convert_layout is inserted after the load so
// COM: that downstream users still see the original encoding.
// COM:
// COM: The pass emits the original (OLD) encoding alias first, then the new one.

// CHECK: #[[$OLD_ENC:.+]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK: #[[$NEW_ENC:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @descriptor_load_fixup
  tt.func public @descriptor_load_fixup(%arg0: !tt.ptr<f16>) -> tensor<64x16xf16, #blocked> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c16_i32 = arith.constant 16 : i32
    %c64_i32 = arith.constant 64 : i32
    %c256_i64 = arith.constant 256 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%c64_i32, %c16_i32], [%c256_i64, %c1_i64] : <f16>, <64x16xf16>
    // CHECK: %[[LOAD:.+]] = tt.descriptor_load {{.*}} : !tt.tensordesc<64x16xf16> -> tensor<64x16xf16, #[[$NEW_ENC]]>
    // CHECK-NEXT: %[[CVT:.+]] = ttg.convert_layout %[[LOAD]] : tensor<64x16xf16, #[[$NEW_ENC]]> -> tensor<64x16xf16, #[[$OLD_ENC]]>
    // CHECK-NEXT: tt.return %[[CVT]]
    %load = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x16xf16> -> tensor<64x16xf16, #blocked>
    tt.return %load : tensor<64x16xf16, #blocked>
  }
}

// -----

// COM: Test 2 — descriptor_store with a non-2D-block-IO-compatible blocked encoding:
// COM: a ttg.convert_layout is inserted before the store, converting the input from
// COM: the original encoding to the new 2D-block-IO-compatible encoding.

// CHECK: #[[$OLD_ENC:.+]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK: #[[$NEW_ENC:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @descriptor_store_fixup
  tt.func public @descriptor_store_fixup(%arg0: !tt.ptr<f16>, %arg1: tensor<64x16xf16, #blocked>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c16_i32 = arith.constant 16 : i32
    %c64_i32 = arith.constant 64 : i32
    %c256_i64 = arith.constant 256 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%c64_i32, %c16_i32], [%c256_i64, %c1_i64] : <f16>, <64x16xf16>
    // CHECK: %[[CVT:.+]] = ttg.convert_layout %arg1 : tensor<64x16xf16, #[[$OLD_ENC]]> -> tensor<64x16xf16, #[[$NEW_ENC]]>
    // CHECK-NEXT: tt.descriptor_store {{.*}}, %[[CVT]] : !tt.tensordesc<64x16xf16>, tensor<64x16xf16, #[[$NEW_ENC]]>
    tt.descriptor_store %desc[%c0_i32, %c0_i32], %arg1 : !tt.tensordesc<64x16xf16>, tensor<64x16xf16, #blocked>
    tt.return
  }
}

// -----

// COM: Test 3 — when ttig.support_2d_block_io is absent the pass is a no-op.
// COM: The original blocked encoding is preserved and no ttg.convert_layout is added.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func public @no_block_io_support_noop
  tt.func public @no_block_io_support_noop(%arg0: !tt.ptr<f16>) -> tensor<64x16xf16, #blocked> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c16_i32 = arith.constant 16 : i32
    %c64_i32 = arith.constant 64 : i32
    %c256_i64 = arith.constant 256 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%c64_i32, %c16_i32], [%c256_i64, %c1_i64] : <f16>, <64x16xf16>
    // CHECK: tt.descriptor_load
    // CHECK-NOT: ttg.convert_layout
    %load = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x16xf16> -> tensor<64x16xf16, #blocked>
    tt.return %load : tensor<64x16xf16, #blocked>
  }
}

// -----

// COM: Test 4 — when the descriptor_load already uses the 2D-block-IO-compatible
// COM: encoding (sizePerThread=[1,1], threadsPerWarp=[2,16]) the pass is a no-op:
// COM: no ttg.convert_layout is inserted.

#blocked_compat = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @already_compatible_noop
  tt.func public @already_compatible_noop(%arg0: !tt.ptr<f16>) -> tensor<64x16xf16, #blocked_compat> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c16_i32 = arith.constant 16 : i32
    %c64_i32 = arith.constant 64 : i32
    %c256_i64 = arith.constant 256 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%c64_i32, %c16_i32], [%c256_i64, %c1_i64] : <f16>, <64x16xf16>
    // CHECK: tt.descriptor_load
    // CHECK-NOT: ttg.convert_layout
    %load = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x16xf16> -> tensor<64x16xf16, #blocked_compat>
    tt.return %load : tensor<64x16xf16, #blocked_compat>
  }
}
