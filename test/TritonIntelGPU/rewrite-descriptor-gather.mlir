// RUN: env TRITON_INTEL_DISABLE_REWRITE_CONTIGUOUS_GATHER=0 triton-opt %s -split-input-file --triton-intel-rewrite-tensor-descriptor-to-pointer | FileCheck %s
//
// Tests for the descriptor_gather → descriptor_load optimization in the
// rewrite-tensor-descriptor-to-pointer pass. When x_offsets are provably
// contiguous, the gather is rewritten to one or more descriptor_load ops.
// The rewrite is always legal at TTIR level — downstream passes decide
// whether the descriptor_load becomes a 2D block load or a pointer-based load.

// COM: Case 1 - Static contiguous make_range(0, N): x_offsets = [0, 1, ..., 15].
// COM: Rewritten to a single descriptor_load since the offsets are provably
// COM: contiguous.
module {
  tt.func public @gather_contiguous_static(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> (tensor<16x32xf16>) {
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c256_i64, %c1_i64] {order = array<i32: 0>} : <f16>, <tensor<1x32xf16>>
    %x_offsets = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %result = tt.descriptor_gather %desc[%x_offsets, %c0_i32] : (!tt.tensordesc<tensor<1x32xf16>>, tensor<16xi32>, i32) -> tensor<16x32xf16>
    tt.return %result : tensor<16x32xf16>
  }
}

// CHECK-LABEL: @gather_contiguous_static
// CHECK: tt.descriptor_load
// CHECK-SAME: -> tensor<16x32xf16>
// CHECK-NOT: tt.load

// -----

// COM: Case 2 - Static contiguous make_range(0, N) + splat(constant):
// COM: x_offsets = make_range(0, 16) + splat(5). Still provably contiguous,
// COM: rewritten to a single descriptor_load with constant base offset.
module {
  tt.func public @gather_contiguous_static_offset(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> (tensor<16x32xf16>) {
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %c0_i32 = arith.constant 0 : i32
    %c5_i32 = arith.constant 5 : i32
    %c256_i32 = arith.constant 256 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c256_i64, %c1_i64] {order = array<i32: 0>} : <f16>, <tensor<1x32xf16>>
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %splat = tt.splat %c5_i32 : i32 -> tensor<16xi32>
    %x_offsets = arith.addi %range, %splat : tensor<16xi32>
    %result = tt.descriptor_gather %desc[%x_offsets, %c0_i32] : (!tt.tensordesc<tensor<1x32xf16>>, tensor<16xi32>, i32) -> tensor<16x32xf16>
    tt.return %result : tensor<16x32xf16>
  }
}

// CHECK-LABEL: @gather_contiguous_static_offset
// CHECK: tt.descriptor_load
// CHECK-SAME: -> tensor<16x32xf16>
// CHECK-NOT: tt.load

// -----

// COM: Case 3 - Dynamic contiguous make_range(0, N) + splat(%dynamic_arg):
// COM: x_offsets = make_range(0, 16) + splat(%arg1). Still contiguous since
// COM: make_range provides unit stride and splat adds a uniform offset.
// COM: Rewritten to a single descriptor_load with dynamic base offset.
module {
  tt.func public @gather_contiguous_dynamic_offset(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: i32) -> (tensor<16x32xf16>) {
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c256_i64, %c1_i64] {order = array<i32: 0>} : <f16>, <tensor<1x32xf16>>
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %splat = tt.splat %arg1 : i32 -> tensor<16xi32>
    %x_offsets = arith.addi %range, %splat : tensor<16xi32>
    %result = tt.descriptor_gather %desc[%x_offsets, %c0_i32] : (!tt.tensordesc<tensor<1x32xf16>>, tensor<16xi32>, i32) -> tensor<16x32xf16>
    tt.return %result : tensor<16x32xf16>
  }
}

// CHECK-LABEL: @gather_contiguous_dynamic_offset
// CHECK: tt.descriptor_load
// CHECK-SAME: -> tensor<16x32xf16>
// CHECK-NOT: tt.load

// -----

// COM: Case 4 - Multi-range constant offsets: x_offsets = [0,1,2,3, 8,9,10,11].
// COM: Two contiguous sub-ranges of 4 rows each. Rewritten to two
// COM: tt.descriptor_load ops (one per sub-range) concatenated with tt.cat.
module {
  tt.func public @gather_multi_range_constant(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) -> (tensor<8x32xbf16>) {
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c256_i64, %c1_i64] {order = array<i32: 0>} : <bf16>, <tensor<1x32xbf16>>
    %x_offsets = arith.constant dense<[0, 1, 2, 3, 8, 9, 10, 11]> : tensor<8xi32>
    %result = tt.descriptor_gather %desc[%x_offsets, %c0_i32] : (!tt.tensordesc<tensor<1x32xbf16>>, tensor<8xi32>, i32) -> tensor<8x32xbf16>
    tt.return %result : tensor<8x32xbf16>
  }
}

// CHECK-LABEL: @gather_multi_range_constant
// CHECK: tt.descriptor_load
// CHECK: tt.descriptor_load
// CHECK: tt.cat
// CHECK-NOT: tt.load

// -----

// COM: Case 5 - Fallback: non-contiguous offsets. x_offsets is an arbitrary
// COM: argument tensor, not a make_range pattern. Should fall back to tt.load.
module {
  tt.func public @gather_noncontiguous_fallback(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: tensor<16xi32>) -> (tensor<16x32xf16>) {
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c256_i64, %c1_i64] {order = array<i32: 0>} : <f16>, <tensor<1x32xf16>>
    %result = tt.descriptor_gather %desc[%arg1, %c0_i32] : (!tt.tensordesc<tensor<1x32xf16>>, tensor<16xi32>, i32) -> tensor<16x32xf16>
    tt.return %result : tensor<16x32xf16>
  }
}

// CHECK-LABEL: @gather_noncontiguous_fallback
// CHECK-NOT: tt.descriptor_load
// CHECK: tt.load

// -----

// COM: Case 6 - Fallback: multi-range with unequal sub-range sizes.
// COM: x_offsets = [0,1,2,3,4, 8,9,10] has sub-ranges of size 5 and 3.
// COM: tt.cat requires SameTypeOperands, so the pattern cannot apply.
module {
  tt.func public @gather_unequal_ranges(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) -> (tensor<8x32xbf16>) {
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c256_i64, %c1_i64] {order = array<i32: 0>} : <bf16>, <tensor<1x32xbf16>>
    %x_offsets = arith.constant dense<[0, 1, 2, 3, 4, 8, 9, 10]> : tensor<8xi32>
    %result = tt.descriptor_gather %desc[%x_offsets, %c0_i32] : (!tt.tensordesc<tensor<1x32xbf16>>, tensor<8xi32>, i32) -> tensor<8x32xbf16>
    tt.return %result : tensor<8x32xbf16>
  }
}

// CHECK-LABEL: @gather_unequal_ranges
// CHECK-NOT: tt.descriptor_load
// CHECK: tt.load
