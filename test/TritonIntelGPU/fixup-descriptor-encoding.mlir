// RUN: env TRITON_INTEL_FIXUP_DESCRIPTOR_ENCODING=1 triton-opt %s -split-input-file --tritonintelgpu-fixup-descriptor-encoding | FileCheck %s --check-prefixes=CHECK,ENABLED
// RUN: triton-opt %s -split-input-file --tritonintelgpu-fixup-descriptor-encoding | FileCheck %s --check-prefixes=CHECK,DISABLED

// COM: Test 1 — descriptor_load with a non-2D-block-IO-compatible blocked encoding and
// COM: ttig.block_io attribute. When the env var is set, the pass reassigns the encoding
// COM: to sizePerThread=[1,1] and inserts ttg.convert_layout. When unset, pass is no-op.
// COM:
// COM: The pass emits the original (OLD) encoding alias first, then the new one.

// ENABLED: #[[$OLD_ENC:.+]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
// ENABLED: #[[$NEW_ENC:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>

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
    // ENABLED: %[[LOAD:.+]] = tt.descriptor_load {{.*}} {ttig.block_io = "row_major"} : !tt.tensordesc<64x16xf16> -> tensor<64x16xf16, #[[$NEW_ENC]]>
    // ENABLED-NEXT: %[[CVT:.+]] = ttg.convert_layout %[[LOAD]] : tensor<64x16xf16, #[[$NEW_ENC]]> -> tensor<64x16xf16, #[[$OLD_ENC]]>
    // ENABLED-NEXT: tt.return %[[CVT]]
    // DISABLED: %[[LOAD:.+]] = tt.descriptor_load {{.*}} {ttig.block_io = "row_major"} : !tt.tensordesc<64x16xf16> -> tensor<64x16xf16, #blocked>
    // DISABLED-NEXT: tt.return %[[LOAD]]
    %load = tt.descriptor_load %desc[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x16xf16> -> tensor<64x16xf16, #blocked>
    tt.return %load : tensor<64x16xf16, #blocked>
  }
}

// -----

// COM: Test 2 — descriptor_store with a non-2D-block-IO-compatible blocked encoding and
// COM: ttig.block_io attribute. When the env var is set, the pass inserts ttg.convert_layout
// COM: before the store. When unset, pass is no-op.

// ENABLED: #[[$OLD_ENC:.+]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
// ENABLED: #[[$NEW_ENC:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>

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
    // ENABLED: %[[CVT:.+]] = ttg.convert_layout %arg1 : tensor<64x16xf16, #[[$OLD_ENC]]> -> tensor<64x16xf16, #[[$NEW_ENC]]>
    // ENABLED-NEXT: tt.descriptor_store {{.*}}, %[[CVT]] {ttig.block_io = "row_major"} : !tt.tensordesc<64x16xf16>, tensor<64x16xf16, #[[$NEW_ENC]]>
    // DISABLED: tt.descriptor_store {{.*}}, %arg1 {ttig.block_io = "row_major"} : !tt.tensordesc<64x16xf16>, tensor<64x16xf16, #blocked>
    tt.descriptor_store %desc[%c0_i32, %c0_i32], %arg1 {ttig.block_io = "row_major"} : !tt.tensordesc<64x16xf16>, tensor<64x16xf16, #blocked>
    tt.return
  }
}

// -----

// COM: Test 3 — when ttig.support_2d_block_io is absent the pass is a no-op regardless
// COM: of the env var. The original blocked encoding is preserved and no ttg.convert_layout
// COM: is added.

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
    %load = tt.descriptor_load %desc[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x16xf16> -> tensor<64x16xf16, #blocked>
    tt.return %load : tensor<64x16xf16, #blocked>
  }
}

// -----

// COM: Test 4 — when the descriptor_load already uses the 2D-block-IO-compatible
// COM: encoding (sizePerThread=[1,1], threadsPerWarp=[2,16]) with ttig.block_io attribute,
// COM: the pass is a no-op even with env var set: newEnc == oldEnc so no conversion inserted.

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
    %load = tt.descriptor_load %desc[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x16xf16> -> tensor<64x16xf16, #blocked_compat>
    tt.return %load : tensor<64x16xf16, #blocked_compat>
  }
}

// -----

// COM: Test 5 — descriptor_load with non-compatible encoding but NO ttig.block_io attribute.
// COM: Even with env var set, the pass leaves the op untouched (requires ttig.block_io presence).
// COM: This is a regression guard: the pass must be attribute-gated, not just encoding-based.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @no_block_io_attr_noop
  tt.func public @no_block_io_attr_noop(%arg0: !tt.ptr<f16>) -> tensor<64x16xf16, #blocked> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c16_i32 = arith.constant 16 : i32
    %c64_i32 = arith.constant 64 : i32
    %c256_i64 = arith.constant 256 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%c64_i32, %c16_i32], [%c256_i64, %c1_i64] : <f16>, <64x16xf16>
    // CHECK: %[[LOAD:.+]] = tt.descriptor_load {{.*}} : !tt.tensordesc<64x16xf16> -> tensor<64x16xf16, #blocked>
    // CHECK-NEXT: tt.return %[[LOAD]]
    // CHECK-NOT: ttg.convert_layout
    %load = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x16xf16> -> tensor<64x16xf16, #blocked>
    tt.return %load : tensor<64x16xf16, #blocked>
  }
}
