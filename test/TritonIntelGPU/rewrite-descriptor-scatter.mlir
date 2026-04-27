// RUN: triton-opt %s -split-input-file --triton-intel-rewrite-tensor-descriptor-to-pointer | FileCheck %s
//
// Tests for the descriptor_scatter -> descriptor_store optimization in the
// rewrite-tensor-descriptor-to-pointer pass. When x_offsets are provably
// contiguous, the scatter is rewritten to a single descriptor_store op.
// The rewrite is always legal at TTIR level -- downstream passes decide
// whether the descriptor_store becomes a 2D block store or a pointer-based store.

// COM: Case 1 - Static contiguous make_range(0, N): x_offsets = [0, 1, ..., 15].
// COM: Rewritten to a single descriptor_store since the offsets are provably
// COM: contiguous.
module {
  tt.func public @scatter_contiguous_static(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: tensor<16x32xf16>) {
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c256_i64, %c1_i64] {order = array<i32: 0>} : <f16>, <1x32xf16>
    %x_offsets = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    tt.descriptor_scatter %desc[%x_offsets, %c0_i32], %arg1 : !tt.tensordesc<1x32xf16>, tensor<16xi32>, i32, tensor<16x32xf16>
    tt.return
  }
}

// CHECK-LABEL: @scatter_contiguous_static
// CHECK: tt.descriptor_store
// CHECK-NOT: tt.store

// -----

// COM: Case 2 - Static contiguous make_range(0, N) + splat(constant):
// COM: x_offsets = make_range(0, 16) + splat(5). Still provably contiguous,
// COM: rewritten to a single descriptor_store with constant base offset.
module {
  tt.func public @scatter_contiguous_static_offset(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: tensor<16x32xf16>) {
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %c0_i32 = arith.constant 0 : i32
    %c5_i32 = arith.constant 5 : i32
    %c256_i32 = arith.constant 256 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c256_i64, %c1_i64] {order = array<i32: 0>} : <f16>, <1x32xf16>
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %splat = tt.splat %c5_i32 : i32 -> tensor<16xi32>
    %x_offsets = arith.addi %range, %splat : tensor<16xi32>
    tt.descriptor_scatter %desc[%x_offsets, %c0_i32], %arg1 : !tt.tensordesc<1x32xf16>, tensor<16xi32>, i32, tensor<16x32xf16>
    tt.return
  }
}

// CHECK-LABEL: @scatter_contiguous_static_offset
// CHECK: tt.descriptor_store
// CHECK-NOT: tt.store

// -----

// COM: Case 3 - Dynamic contiguous make_range(0, N) + splat(%dynamic_arg):
// COM: x_offsets = make_range(0, 16) + splat(%arg2). Still contiguous since
// COM: make_range provides unit stride and splat adds a uniform offset.
// COM: Rewritten to a single descriptor_store with dynamic base offset.
module {
  tt.func public @scatter_contiguous_dynamic_offset(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: tensor<16x32xf16>, %arg2: i32) {
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c256_i64, %c1_i64] {order = array<i32: 0>} : <f16>, <1x32xf16>
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %splat = tt.splat %arg2 : i32 -> tensor<16xi32>
    %x_offsets = arith.addi %range, %splat : tensor<16xi32>
    tt.descriptor_scatter %desc[%x_offsets, %c0_i32], %arg1 : !tt.tensordesc<1x32xf16>, tensor<16xi32>, i32, tensor<16x32xf16>
    tt.return
  }
}

// CHECK-LABEL: @scatter_contiguous_dynamic_offset
// CHECK: tt.descriptor_store
// CHECK-NOT: tt.store

// -----

// COM: Case 4 - Fallback: non-contiguous offsets. x_offsets is an arbitrary
// COM: argument tensor, not a make_range pattern. Should fall back to tt.store.
module {
  tt.func public @scatter_noncontiguous_fallback(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: tensor<16x32xf16>, %arg2: tensor<16xi32>) {
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c256_i64, %c1_i64] {order = array<i32: 0>} : <f16>, <1x32xf16>
    tt.descriptor_scatter %desc[%arg2, %c0_i32], %arg1 : !tt.tensordesc<1x32xf16>, tensor<16xi32>, i32, tensor<16x32xf16>
    tt.return
  }
}

// CHECK-LABEL: @scatter_noncontiguous_fallback
// CHECK-NOT: tt.descriptor_store
// CHECK: tt.store

// -----

// COM: Case 5 - Multi-range constant offsets: x_offsets = [0,1,2,3, 8,9,10,11].
// COM: Two contiguous sub-ranges of 4 rows each. Rewritten to two
// COM: tt.descriptor_store ops (one per sub-range), with row slices extracted
// COM: from the source tensor via tt.gather.
module {
  tt.func public @scatter_multi_range_constant(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: tensor<8x32xbf16>) {
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c256_i64, %c1_i64] {order = array<i32: 0>} : <bf16>, <1x32xbf16>
    %x_offsets = arith.constant dense<[0, 1, 2, 3, 8, 9, 10, 11]> : tensor<8xi32>
    tt.descriptor_scatter %desc[%x_offsets, %c0_i32], %arg1 : !tt.tensordesc<1x32xbf16>, tensor<8xi32>, i32, tensor<8x32xbf16>
    tt.return
  }
}

// CHECK-LABEL: @scatter_multi_range_constant
// CHECK: tt.descriptor_store
// CHECK: tt.descriptor_store
// CHECK-NOT: tt.store

// -----

// COM: Case 6 - Fallback: multi-range with unequal sub-range sizes.
// COM: x_offsets = [0,1,2,3,4, 8,9,10] has sub-ranges of size 5 and 3.
// COM: All sub-ranges must be equal for the multi-range rewrite to apply.
module {
  tt.func public @scatter_unequal_ranges(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: tensor<8x32xbf16>) {
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c256_i64, %c1_i64] {order = array<i32: 0>} : <bf16>, <1x32xbf16>
    %x_offsets = arith.constant dense<[0, 1, 2, 3, 4, 8, 9, 10]> : tensor<8xi32>
    tt.descriptor_scatter %desc[%x_offsets, %c0_i32], %arg1 : !tt.tensordesc<1x32xbf16>, tensor<8xi32>, i32, tensor<8x32xbf16>
    tt.return
  }
}

// CHECK-LABEL: @scatter_unequal_ranges
// CHECK-NOT: tt.descriptor_store
// CHECK: tt.store
