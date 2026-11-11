// RUN: triton-opt %s -test-print-stride-info -split-input-file -o %t 2>&1 | FileCheck %s

// CHECK-LABEL: @make_range
tt.func @make_range() {
  // CHECK: tt.make_range {{.*}} => stride = [1]
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  tt.return
}

// -----

// CHECK-LABEL: @constant_ops
tt.func @constant_ops() {
  // CHECK: arith.constant {{.*}} => stride = [0]
  %cst_scalar = arith.constant 42 : i32
  // CHECK: arith.constant {{.*}} => stride = [0]
  %cst_splat_1d = arith.constant dense<512> : tensor<128xi32>
  // CHECK: arith.constant {{.*}} => stride = [0, 0]
  %cst_splat_2d = arith.constant dense<512> : tensor<128x64xi32>
  // CHECK: arith.constant {{.*}} => stride = [0]
  %cst_bool = arith.constant true
  tt.return
}

// -----

// CHECK-LABEL: @splat
tt.func @splat(%arg0: i32) {
  // CHECK: tt.splat {{.*}} => stride = [0]
  %0 = tt.splat %arg0 : i32 -> tensor<128xi32>
  // CHECK: tt.splat {{.*}} => stride = [0, 0]
  %1 = tt.splat %arg0 : i32 -> tensor<128x64xi32>
  tt.return
}

// -----

// CHECK-LABEL: @add_ops
tt.func @add_ops() {
  // CHECK: tt.make_range {{.*}} => stride = [1]
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // CHECK: arith.constant {{.*}} => stride = [0]
  %1 = arith.constant dense<1> : tensor<128xi32>
  // stride(range) + stride(const) = 1 + 0 = 1
  // CHECK: arith.addi {{.*}} => stride = [1]
  %2 = arith.addi %0, %1 : tensor<128xi32>
  // stride(range) + stride(range) = 1 + 1 = 2
  // CHECK: arith.addi {{.*}} => stride = [2]
  %3 = arith.addi %0, %0 : tensor<128xi32>
  tt.return
}

// -----

// CHECK-LABEL: @addptr
tt.func @addptr(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  // CHECK: tt.make_range {{.*}} => stride = [1]
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // CHECK: tt.splat {{.*}} => stride = [0]
  %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
  // stride(splat) + stride(range) = 0 + 1 = 1
  // CHECK: tt.addptr {{.*}} => stride = [1]
  %2 = tt.addptr %1, %0 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
  tt.return
}

// -----

// CHECK-LABEL: @sub_ops
tt.func @sub_ops() {
  // CHECK: tt.make_range {{.*}} => stride = [1]
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // CHECK: arith.constant {{.*}} => stride = [0]
  %1 = arith.constant dense<1> : tensor<128xi32>
  // stride(range) - stride(const) = 1 - 0 = 1
  // CHECK: arith.subi {{.*}} => stride = [1]
  %2 = arith.subi %0, %1 : tensor<128xi32>
  // stride(range) - stride(range) = 1 - 1 = 0
  // CHECK: arith.subi {{.*}} => stride = [0]
  %3 = arith.subi %0, %0 : tensor<128xi32>
  // stride(const) - stride(range) = 0 - 1 => clamped to -1
  // CHECK: arith.subi {{.*}} => stride = [-1]
  %4 = arith.subi %1, %0 : tensor<128xi32>
  tt.return
}

// -----

// CHECK-LABEL: @muli
tt.func @muli(%arg0: i32) {
  // CHECK: tt.make_range {{.*}} => stride = [1]
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // CHECK: arith.constant {{.*}} => stride = [0]
  %cst4 = arith.constant dense<4> : tensor<128xi32>
  // stride(range) * const_value(4) = 1 * 4 = 4
  // CHECK: arith.muli {{.*}} => stride = [4]
  %1 = arith.muli %0, %cst4 : tensor<128xi32>
  // Scalar multiply: stride = 0 (both effectively constant/scalar)
  // CHECK: arith.constant {{.*}} => stride = [0]
  %c128 = arith.constant 128 : i32
  // CHECK: arith.muli {{.*}} => stride = [0]
  %2 = arith.muli %arg0, %c128 : i32
  // negative constant: stride * -2 = -2 => clamped to -1 (unknown)
  // CHECK: arith.constant {{.*}} => stride = [0]
  %neg = arith.constant dense<-2> : tensor<128xi32>
  // CHECK: arith.muli {{.*}} => stride = [-1]
  %3 = arith.muli %0, %neg : tensor<128xi32>
  tt.return
}

// -----

// CHECK-LABEL: @div
tt.func @div() {
  // CHECK: tt.make_range {{.*}} => stride = [1]
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // CHECK: arith.constant {{.*}} => stride = [0]
  %cst4 = arith.constant dense<4> : tensor<128xi32>
  // stride(range) * 4 = 4, then 4 / 4 = 1
  // CHECK: arith.muli {{.*}} => stride = [4]
  %1 = arith.muli %0, %cst4 : tensor<128xi32>
  // stride(4) / const(4) = 1
  // CHECK: arith.divsi {{.*}} => stride = [1]
  %2 = arith.divsi %1, %cst4 : tensor<128xi32>
  // CHECK: arith.divui {{.*}} => stride = [1]
  %3 = arith.divui %1, %cst4 : tensor<128xi32>
  // stride(1) / const(4): not evenly divisible => -1
  // CHECK: arith.divsi {{.*}} => stride = [-1]
  %4 = arith.divsi %0, %cst4 : tensor<128xi32>
  // CHECK: arith.divui {{.*}} => stride = [-1]
  %5 = arith.divui %0, %cst4 : tensor<128xi32>
  // stride(0) / constant = 0
  // CHECK: arith.constant {{.*}} => stride = [0]
  %cst_splat = arith.constant dense<100> : tensor<128xi32>
  // CHECK: arith.divsi {{.*}} => stride = [0]
  %6 = arith.divsi %cst_splat, %cst4 : tensor<128xi32>
  // CHECK: arith.divui {{.*}} => stride = [0]
  %7 = arith.divui %cst_splat, %cst4 : tensor<128xi32>
  // negative divisor => -1 (signed only)
  // CHECK: arith.constant {{.*}} => stride = [0]
  %neg = arith.constant dense<-4> : tensor<128xi32>
  // CHECK: arith.divsi {{.*}} => stride = [-1]
  %8 = arith.divsi %0, %neg : tensor<128xi32>
  // division by zero => -1
  // CHECK: arith.constant {{.*}} => stride = [0]
  %zero = arith.constant dense<0> : tensor<128xi32>
  // CHECK: arith.divsi {{.*}} => stride = [-1]
  %9 = arith.divsi %cst4, %zero : tensor<128xi32>
  // CHECK: arith.divui {{.*}} => stride = [-1]
  %10 = arith.divui %cst4, %zero : tensor<128xi32>
  tt.return
}

// -----

// CHECK-LABEL: @rem
tt.func @rem() {
  // CHECK: tt.make_range {{.*}} => stride = [1]
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // CHECK: arith.constant {{.*}} => stride = [0]
  %cst_mod = arith.constant dense<512> : tensor<128xi32>
  // Range span (127) < modulus (512): no wrap-around, stride preserved.
  // CHECK: arith.remsi {{.*}} => stride = [1]
  %1 = arith.remsi %0, %cst_mod : tensor<128xi32>
  // CHECK: arith.remui {{.*}} => stride = [1]
  %2 = arith.remui %0, %cst_mod : tensor<128xi32>
  // lhs stride == 0 and rhs stride == 0 => stride = 0
  // CHECK: arith.constant {{.*}} => stride = [0]
  %cst_val = arith.constant dense<100> : tensor<128xi32>
  // CHECK: arith.remsi {{.*}} => stride = [0]
  %3 = arith.remsi %cst_val, %cst_mod : tensor<128xi32>
  // CHECK: arith.remui {{.*}} => stride = [0]
  %4 = arith.remui %cst_val, %cst_mod : tensor<128xi32>
  tt.return
}

// -----

// CHECK-LABEL: @rem_stride_wrap
tt.func @rem_stride_wrap(%arg0: i32) {
  // CHECK: tt.make_range {{.*}} => stride = [1]
  %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  // CHECK: arith.constant {{.*}} => stride = [0]
  %c8 = arith.constant dense<8> : tensor<32xi32>
  // CHECK: arith.constant {{.*}} => stride = [0]
  %c64 = arith.constant dense<64> : tensor<32xi32>

  // Range span (31) >= modulus (8): wrap-around breaks stride.
  // CHECK: arith.remsi {{.*}} => stride = [-1]
  %rem_wrap = arith.remsi %range, %c8 : tensor<32xi32>

  // Range span (31) < modulus (64): no wrap-around, stride preserved.
  // CHECK: arith.remsi {{.*}} => stride = [1]
  %rem_nowrap = arith.remsi %range, %c64 : tensor<32xi32>

  // Non-constant rhs (unknown stride): cannot reason about modulus.
  // CHECK: tt.splat {{.*}} => stride = [0]
  %splat_arg = tt.splat %arg0 : i32 -> tensor<32xi32>
  // CHECK: arith.remsi {{.*}} => stride = [-1]
  %rem_nonconst = arith.remsi %range, %splat_arg : tensor<32xi32>

  // Unknown lhs stride: cannot preserve.
  // CHECK: arith.muli {{.*}} => stride = [-1]
  %unknown = arith.muli %range, %splat_arg : tensor<32xi32>
  // CHECK: arith.remsi {{.*}} => stride = [-1]
  %rem_unk_lhs = arith.remsi %unknown, %c8 : tensor<32xi32>

  // Both non-constant strides: unknown.
  // CHECK: tt.make_range {{.*}} => stride = [1]
  %rhs_range = tt.make_range {end = 34 : i32, start = 2 : i32} : tensor<32xi32>
  // CHECK: arith.remsi {{.*}} => stride = [-1]
  %rem_both_range = arith.remsi %range, %rhs_range : tensor<32xi32>

  // Scaled range: gcd(divisibility=2, modulus=64) = 2, maxVal=62 >= 2 => unknown.
  // CHECK: arith.constant {{.*}} => stride = [0]
  %c2 = arith.constant dense<2> : tensor<32xi32>
  // CHECK: arith.muli {{.*}} => stride = [2]
  %scaled = arith.muli %range, %c2 : tensor<32xi32>
  // CHECK: arith.remui {{.*}} => stride = [-1]
  %rem_scaled = arith.remui %scaled, %c64 : tensor<32xi32>

  // Scaled range: maxVal=62 >= modulus=8 => wrap-around.
  // CHECK: arith.remui {{.*}} => stride = [-1]
  %rem_scaled_wrap = arith.remui %scaled, %c8 : tensor<32xi32>

  // Negative modulus: not supported, returns unknown.
  // CHECK: arith.constant {{.*}} => stride = [0]
  %cneg = arith.constant dense<-4> : tensor<32xi32>
  // CHECK: arith.remsi {{.*}} => stride = [-1]
  %rem_neg_mod = arith.remsi %range, %cneg : tensor<32xi32>

  tt.return
}

// -----

// CHECK-LABEL: @expand_dims
tt.func @expand_dims() {
  // CHECK: tt.make_range {{.*}} => stride = [1]
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // Inserts stride 0 at axis 1: [1] -> [1, 0]
  // CHECK: tt.expand_dims {{.*}} => stride = [1, 0]
  %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
  // Inserts stride 0 at axis 0: [1] -> [0, 1]
  // CHECK: tt.expand_dims {{.*}} => stride = [0, 1]
  %2 = tt.expand_dims %0 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
  tt.return
}

// -----

// CHECK-LABEL: @broadcast
tt.func @broadcast() {
  // CHECK: tt.make_range {{.*}} => stride = [1]
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // CHECK: tt.expand_dims {{.*}} => stride = [1, 0]
  %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
  // Broadcast preserves stride: [1, 0] -> [1, 0]
  // CHECK: tt.broadcast {{.*}} => stride = [1, 0]
  %2 = tt.broadcast %1 : tensor<128x1xi32> -> tensor<128x64xi32>
  tt.return
}

// -----

// CHECK-LABEL: @trans
tt.func @trans() {
  // CHECK: tt.make_range {{.*}} => stride = [1]
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // CHECK: tt.expand_dims {{.*}} => stride = [1, 0]
  %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
  // CHECK: tt.broadcast {{.*}} => stride = [1, 0]
  %2 = tt.broadcast %1 : tensor<128x1xi32> -> tensor<128x64xi32>
  // Transpose permutes stride: [1, 0] -> [0, 1]
  // CHECK: tt.trans {{.*}} => stride = [0, 1]
  %3 = tt.trans %2 {order = array<i32: 1, 0>} : tensor<128x64xi32> -> tensor<64x128xi32>
  tt.return
}

// -----

// CHECK-LABEL: @load_pessimistic
tt.func @load_pessimistic(%arg0: tensor<128x!tt.ptr<f32>>) {
  // Load result has unknown stride
  // CHECK: tt.load {{.*}} => stride = [-1]
  %0 = tt.load %arg0 : tensor<128x!tt.ptr<f32>>
  tt.return
}

// -----

// CHECK-LABEL: @passthrough_ext_trunc
tt.func @passthrough_ext_trunc() {
  // CHECK: tt.make_range {{.*}} => stride = [1]
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // extsi passes through stride
  // CHECK: arith.extsi {{.*}} => stride = [1]
  %1 = arith.extsi %0 : tensor<128xi32> to tensor<128xi64>
  // extui passes through stride
  // CHECK: arith.extui {{.*}} => stride = [1]
  %2 = arith.extui %0 : tensor<128xi32> to tensor<128xi64>
  // trunci passes through stride
  // CHECK: arith.trunci {{.*}} => stride = [1]
  %3 = arith.trunci %1 : tensor<128xi64> to tensor<128xi32>
  tt.return
}

// -----

// CHECK-LABEL: @passthrough_bitcast
tt.func @passthrough_bitcast() {
  // CHECK: tt.make_range {{.*}} => stride = [1]
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // Bitcast passes through stride
  // CHECK: tt.bitcast {{.*}} => stride = [1]
  %1 = tt.bitcast %0 : tensor<128xi32> -> tensor<128xf32>
  tt.return
}

// -----

// CHECK-LABEL: @make_tensor_desc_known_strides
tt.func @make_tensor_desc_known_strides(%arg0: !tt.ptr<f16>) {
  %c1_i64 = arith.constant 1 : i64
  %c32_i64 = arith.constant 32 : i64
  %c128_i32 = arith.constant 128 : i32
  %c32_i32 = arith.constant 32 : i32
  // constant stride operands [32, 1] => stride = [32, 1]
  // CHECK: tt.make_tensor_descriptor {{.*}} => stride = [32, 1]
  %0 = tt.make_tensor_descriptor %arg0, [%c128_i32, %c32_i32], [%c32_i64, %c1_i64] : <f16>, <128x32xf16>
  tt.return
}

// -----

// CHECK-LABEL: @make_tensor_desc_unknown_stride
tt.func @make_tensor_desc_unknown_stride(%arg0: !tt.ptr<f16>, %stride: i64) {
  %c128_i32 = arith.constant 128 : i32
  %c32_i32 = arith.constant 32 : i32
  // non-constant stride operand => unknown stride
  // CHECK: tt.make_tensor_descriptor {{.*}} => stride = [-1, -1]
  %0 = tt.make_tensor_descriptor %arg0, [%c128_i32, %c32_i32], [%stride, %stride] : <f16>, <128x32xf16>
  tt.return
}

// -----

// CHECK-LABEL: @descriptor_load
tt.func @descriptor_load(%arg0: !tt.ptr<f16>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c128_i32 = arith.constant 128 : i32
  %c32_i32 = arith.constant 32 : i32
  // CHECK: tt.make_tensor_descriptor {{.*}} => stride = [1, 1]
  %desc = tt.make_tensor_descriptor %arg0, [%c128_i32, %c32_i32], [%c1_i64, %c1_i64] : <f16>, <128x32xf16>
  // descriptor_load always returns pessimistic stride regardless of descriptor
  // CHECK: tt.descriptor_load {{.*}} => stride = [-1, -1]
  %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<128x32xf16> -> tensor<128x32xf16>
  tt.return
}

// -----

// CHECK-LABEL: @ptr_offset_pattern
// This tests a realistic pattern: computing 2D pointer offsets
tt.func @ptr_offset_pattern(%arg1: tensor<128x1xi32>) {
  // CHECK: arith.constant {{.*}} => stride = [0, 0]
  %cst_0 = arith.constant dense<512> : tensor<128x1xi32>
  // CHECK: arith.constant {{.*}} => stride = [0]
  %cst_1 = arith.constant dense<512> : tensor<128xi32>
  // CHECK: arith.constant {{.*}} => stride = [0]
  %c128_i32 = arith.constant 128 : i32
  // A program id is non-negative, so the remainder below keeps its stride.
  // The declined case is in stride-info-assume.mlir.
  // CHECK: tt.get_program_id {{.*}} => stride = [-1]
  %pid = tt.get_program_id x : i32
  // scalar * scalar => stride = [0]
  // CHECK: arith.muli {{.*}} => stride = [0]
  %0 = arith.muli %pid, %c128_i32 : i32
  // Splat scalar => stride = [0]
  // CHECK: tt.splat {{.*}} => stride = [0]
  %1 = tt.splat %0 : i32 -> tensor<128xi32>
  // make_range => stride = [1]
  // CHECK: tt.make_range {{.*}} => stride = [1]
  %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // [0] + [1] = [1]
  // CHECK: arith.addi {{.*}} => stride = [1]
  %3 = arith.addi %1, %2 : tensor<128xi32>
  // rem with constant modulus preserves stride
  // CHECK: arith.remsi {{.*}} => stride = [1]
  %4 = arith.remsi %3, %cst_1 : tensor<128xi32>
  // expand_dims at axis=1: [1] -> [1, 0]
  // CHECK: tt.expand_dims {{.*}} => stride = [1, 0]
  %5 = tt.expand_dims %4 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
  // [1, 0] * splat(512) => stride = [512, 0]
  // CHECK: arith.muli {{.*}} => stride = [512, 0]
  %6 = arith.muli %5, %cst_0 : tensor<128x1xi32>
  // broadcast preserves stride: [512, 0]
  // CHECK: tt.broadcast {{.*}} => stride = [512, 0]
  %7 = tt.broadcast %6 : tensor<128x1xi32> -> tensor<128x64xi32>
  // muli with unknown stride input => [-1, -1]
  // CHECK: arith.muli {{.*}} => stride = [-1, -1]
  %8 = arith.muli %arg1, %cst_0 : tensor<128x1xi32>
  tt.return
}

// -----

// CHECK-LABEL: @add_with_unknown
tt.func @add_with_unknown(%arg0: tensor<128xi32>) {
  // CHECK: tt.make_range {{.*}} => stride = [1]
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // arg0 has unknown stride (-1), so addition gives -1
  // CHECK: arith.addi {{.*}} => stride = [-1]
  %1 = arith.addi %0, %arg0 : tensor<128xi32>
  tt.return
}

// -----

// CHECK-LABEL: @scf_for_iv
tt.func @scf_for_iv() {
  %c0 = arith.constant 0 : i32
  %c10 = arith.constant 10 : i32
  %c1 = arith.constant 1 : i32
  // Induction variable of scf.for should have stride = [0] (scalar)
  // CHECK: scf.for
  // CHECK: } => stride = [0]
  %0 = scf.for %iv = %c0 to %c10 step %c1 iter_args(%acc = %c0) -> (i32) : i32 {
    %1 = arith.addi %acc, %iv : i32
    scf.yield %1 : i32
  }
  tt.return
}

// -----

// CHECK-LABEL: @scf_for_iter_arg
tt.func @scf_for_iter_arg(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  %c0_i32 = arith.constant 0 : i32
  %c10_i32 = arith.constant 10 : i32
  %c1_i32 = arith.constant 1 : i32
  // CHECK: tt.make_range {{.*}} => stride = [1]
  %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // CHECK: tt.splat {{.*}} => stride = [0]
  %ptr_splat = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
  // CHECK: tt.addptr {{.*}} => stride = [1]
  %ptrs = tt.addptr %ptr_splat, %range : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
  // CHECK: arith.constant {{.*}} => stride = [0]
  %cst_step = arith.constant dense<128> : tensor<128xi32>
  // The iter_arg gets joined with all values flowing into it.
  // Initial value has stride [1], loop body adds stride [0], so join = [1].
  // CHECK: scf.for
  %result = scf.for %iv = %c0_i32 to %c10_i32 step %c1_i32 iter_args(%ptr = %ptrs) -> (tensor<128x!tt.ptr<f32>>) : i32 {
    %next = tt.addptr %ptr, %cst_step : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    scf.yield %next : tensor<128x!tt.ptr<f32>>
  }
  // CHECK: } => stride = [1]
  tt.return
}

// -----

// CHECK-LABEL: @scf_if_join
tt.func @scf_if_join(%cond: i1) {
  // CHECK: tt.make_range {{.*}} => stride = [1]
  %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // CHECK: arith.constant {{.*}} => stride = [0]
  %cst = arith.constant dense<42> : tensor<128xi32>
  // Both branches yield stride [1], so join = [1]
  // CHECK: scf.if
  %0 = scf.if %cond -> (tensor<128xi32>) {
    scf.yield %range : tensor<128xi32>
  } else {
    %added = arith.addi %range, %cst : tensor<128xi32>
    scf.yield %added : tensor<128xi32>
  }
  // CHECK: } => stride = [1]
  tt.return
}

// -----

// CHECK-LABEL: @scf_if_diverge
tt.func @scf_if_diverge(%cond: i1) {
  // CHECK: tt.make_range {{.*}} => stride = [1]
  %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // CHECK: arith.constant {{.*}} => stride = [0]
  %cst = arith.constant dense<42> : tensor<128xi32>
  // One branch has stride [1], other has stride [0] => join = [-1]
  // CHECK: scf.if
  %0 = scf.if %cond -> (tensor<128xi32>) {
    scf.yield %range : tensor<128xi32>
  } else {
    scf.yield %cst : tensor<128xi32>
  }
  // CHECK: } => stride = [-1]
  tt.return
}

// -----

// CHECK-LABEL: @index_cast_passthrough
tt.func @index_cast_passthrough() {
  // CHECK: tt.make_range {{.*}} => stride = [1]
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // index_cast passes through stride
  // CHECK: arith.index_cast {{.*}} => stride = [1]
  %1 = arith.index_cast %0 : tensor<128xi32> to tensor<128xindex>
  tt.return
}

// -----

// CHECK-LABEL: @muli_commutative
tt.func @muli_commutative() {
  // CHECK: tt.make_range {{.*}} => stride = [1]
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // CHECK: arith.constant {{.*}} => stride = [0]
  %cst = arith.constant dense<8> : tensor<128xi32>
  // const * range: commutative case
  // CHECK: arith.muli {{.*}} => stride = [8]
  %1 = arith.muli %cst, %0 : tensor<128xi32>
  tt.return
}

// -----

// CHECK-LABEL: @rem_stride_divisibility
// Tests wrap-around detection using gcd(divisibility, modulus).
tt.func @rem_stride_divisibility() {
  // divisibility(start=3) = 1
  // CHECK: tt.make_range {{.*}} => stride = [1]
  %range_3_7 = tt.make_range {end = 7 : i32, start = 3 : i32} : tensor<4xi32>
  // CHECK: arith.constant {{.*}} => stride = [0]
  %c4 = arith.constant dense<4> : tensor<4xi32>
  // CHECK: arith.constant {{.*}} => stride = [0]
  %c6 = arith.constant dense<6> : tensor<4xi32>

  // gcd(1, 4) = 1, maxVal=3 >= 1 => wrap
  // CHECK: arith.remsi {{.*}} => stride = [-1]
  %rem1 = arith.remsi %range_3_7, %c4 : tensor<4xi32>

  // gcd(1, 6) = 1, maxVal=3 >= 1 => wrap ({3,4,5,6}%6 = {3,4,5,0})
  // CHECK: arith.remsi {{.*}} => stride = [-1]
  %rem2 = arith.remsi %range_3_7, %c6 : tensor<4xi32>

  // divisibility(start=16) = 16
  // CHECK: tt.make_range {{.*}} => stride = [1]
  %range_16_20 = tt.make_range {end = 20 : i32, start = 16 : i32} : tensor<4xi32>

  // gcd(16, 4) = 4, maxVal=3 < 4 => no wrap
  // CHECK: arith.remsi {{.*}} => stride = [1]
  %rem3 = arith.remsi %range_16_20, %c4 : tensor<4xi32>

  // divisibility(start=0) = 2^62
  // CHECK: tt.make_range {{.*}} => stride = [1]
  %range_0_4 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  // CHECK: arith.constant {{.*}} => stride = [0]
  %c8 = arith.constant dense<8> : tensor<4xi32>

  // gcd(2^62, 8) = 8, maxVal=3 < 8 => no wrap
  // CHECK: arith.remui {{.*}} => stride = [1]
  %rem4 = arith.remui %range_0_4, %c8 : tensor<4xi32>

  // gcd(2^62, 8) = 8, maxVal=31 >= 8 => wrap
  // CHECK: tt.make_range {{.*}} => stride = [1]
  %range_0_32 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  // CHECK: arith.constant {{.*}} => stride = [0]
  %c8_32 = arith.constant dense<8> : tensor<32xi32>
  // CHECK: arith.remui {{.*}} => stride = [-1]
  %rem5 = arith.remui %range_0_32, %c8_32 : tensor<32xi32>

  tt.return
}

// -----

// CHECK-LABEL: @contiguity_hint
// tt.contiguity hint overrides unknown stride when hint covers full dimension.
tt.func @contiguity_hint() {
  // CHECK: tt.make_range {{.*}} => stride = [1]
  %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  // CHECK: arith.constant {{.*}} => stride = [0]
  %c8 = arith.constant dense<8> : tensor<32xi32>
  // Without hint: wrap-around (span 31 >= modulus 8) => unknown stride.
  // With tt.contiguity hint covering full dim (32 >= 32) => stride = [1].
  // CHECK: arith.remsi {{.*}} => stride = [1]
  %0 = arith.remsi %range, %c8 {tt.contiguity = dense<32> : tensor<1xi32>} : tensor<32xi32>
  tt.return
}

// -----

// CHECK-LABEL: @contiguity_hint_partial
// Hint does not cover full dimension, so stride stays unknown.
tt.func @contiguity_hint_partial() {
  // CHECK: tt.make_range {{.*}} => stride = [1]
  %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  // CHECK: arith.constant {{.*}} => stride = [0]
  %c8 = arith.constant dense<8> : tensor<32xi32>
  // Hint (16) < dim size (32) => not fully contiguous, stride stays unknown.
  // CHECK: arith.remsi {{.*}} => stride = [-1]
  %0 = arith.remsi %range, %c8 {tt.contiguity = dense<16> : tensor<1xi32>} : tensor<32xi32>
  tt.return
}

// -----

// CHECK-LABEL: @poison_init
// Loop fusion seeds a loop-carried pointer with ub.poison and rebuilds it on
// the first iteration (scf.if, or arith.select after canonicalize). The poison
// is never read, so the iter_args take the stride of the rebuilt pointer. The
// loop starts at 1, so a guard on `%iv == 0` does not hold on the first
// iteration (%r).
tt.func @poison_init(%arg0: !tt.ptr<f16>, %n: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant dense<64> : tensor<16x1xi32>
  %poison = ub.poison : tensor<16x16x!tt.ptr<f16>>
  %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %2 = arith.muli %1, %cst : tensor<16x1xi32>
  %3 = tt.broadcast %2 : tensor<16x1xi32> -> tensor<16x16xi32>
  %4 = tt.expand_dims %0 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %5 = tt.broadcast %4 : tensor<1x16xi32> -> tensor<16x16xi32>
  %6 = arith.addi %3, %5 : tensor<16x16xi32>
  %7 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
  %8 = tt.addptr %7, %6 : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
  %9:3 = scf.for %iv = %c1_i32 to %n step %c1_i32 iter_args(%p = %poison, %q = %poison, %r = %poison) -> (tensor<16x16x!tt.ptr<f16>>, tensor<16x16x!tt.ptr<f16>>, tensor<16x16x!tt.ptr<f16>>) : i32 {
    %first = arith.cmpi eq, %iv, %c1_i32 : i32
    %zeroth = arith.cmpi eq, %iv, %c0_i32 : i32
    // CHECK: } => stride = [64, 1]
    %pn = scf.if %first -> (tensor<16x16x!tt.ptr<f16>>) {
      scf.yield %8 : tensor<16x16x!tt.ptr<f16>>
    } else {
      scf.yield %p : tensor<16x16x!tt.ptr<f16>>
    }
    // CHECK: arith.select {{.*}} => stride = [64, 1]
    %qn = arith.select %first, %8, %q : tensor<16x16x!tt.ptr<f16>>
    // CHECK: arith.select {{.*}} => stride = [-1, -1]
    %rn = arith.select %zeroth, %8, %r : tensor<16x16x!tt.ptr<f16>>
    scf.yield %pn, %qn, %rn : tensor<16x16x!tt.ptr<f16>>, tensor<16x16x!tt.ptr<f16>>, tensor<16x16x!tt.ptr<f16>>
  }
  tt.return
}

// -----

// CHECK-LABEL: @select_contiguity_hint
// tl.max_contiguous(tl.where(...)): a tt.contiguity hint names the stride of a
// tensor-condition select.
tt.func @select_contiguity_hint(%m: i32) {
  %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %zero = arith.constant dense<0> : tensor<32xi32>
  %bound = tt.splat %m : i32 -> tensor<32xi32>
  %mask = arith.cmpi slt, %range, %bound : tensor<32xi32>
  // CHECK: arith.select {{.*}} => stride = [-1]
  %0 = arith.select %mask, %range, %zero : tensor<32xi1>, tensor<32xi32>
  // CHECK: arith.select {{.*}} => stride = [1]
  %1 = arith.select %mask, %range, %zero {tt.contiguity = dense<32> : tensor<1xi32>} : tensor<32xi1>, tensor<32xi32>
  tt.return
}

// -----

// CHECK-LABEL: @select_tensor_cond
// Both arms have stride [0, 1]; a tensor condition keeps it only along a dim
// on which the condition is constant over the whole dim.
tt.func @select_tensor_cond(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %m: i32, %c: i1, %lda: i32) {
  %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %2 = tt.broadcast %1 : tensor<1x16xi32> -> tensor<16x16xi32>
  %3 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
  %a = tt.addptr %3, %2 : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
  %4 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
  %b = tt.addptr %4, %2 : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
  %5 = tt.expand_dims %0 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %6 = tt.broadcast %5 : tensor<16x1xi32> -> tensor<16x16xi32>
  %any = arith.cmpi slt, %2, %6 : tensor<16x16xi32>
  %7 = tt.splat %m : i32 -> tensor<16x1xi32>
  %8 = arith.cmpi slt, %5, %7 : tensor<16x1xi32>
  %rows = tt.broadcast %8 : tensor<16x1xi1> -> tensor<16x16xi1>
  %all = tt.splat %c : i1 -> tensor<16x16xi1>
  // CHECK: arith.select {{.*}} => stride = [-1, -1]
  %s0 = arith.select %any, %a, %b : tensor<16x16xi1>, tensor<16x16x!tt.ptr<f16>>
  // CHECK: arith.select {{.*}} => stride = [-1, 1]
  %s1 = arith.select %rows, %a, %b : tensor<16x16xi1>, tensor<16x16x!tt.ptr<f16>>
  // CHECK: arith.select {{.*}} => stride = [0, 1]
  %s2 = arith.select %all, %a, %b : tensor<16x16xi1>, tensor<16x16x!tt.ptr<f16>>
  %c8 = arith.constant dense<8> : tensor<16x16xi32>
  %half = arith.cmpi slt, %2, %c8 : tensor<16x16xi32>
  // CHECK: arith.select {{.*}} => stride = [0, -1]
  %s5 = arith.select %half, %a, %b : tensor<16x16xi1>, tensor<16x16x!tt.ptr<f16>>
  // Arms with runtime pitch %lda: a tensor condition also drops the stride
  // value, a scalar one keeps it.
  %9 = tt.splat %lda : i32 -> tensor<16x1xi32>
  %10 = arith.muli %5, %9 : tensor<16x1xi32>
  %11 = tt.broadcast %10 : tensor<16x1xi32> -> tensor<16x16xi32>
  %12 = arith.addi %11, %2 : tensor<16x16xi32>
  %ra = tt.addptr %3, %12 : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
  %rb = tt.addptr %4, %12 : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
  // CHECK: arith.select {{.*}} => stride = [-1, 1]{{$}}
  %s3 = arith.select %rows, %ra, %rb : tensor<16x16xi1>, tensor<16x16x!tt.ptr<f16>>
  // CHECK: arith.select {{.*}} => stride = [-1, 1], stride_values = {0: {{.*}}}
  %s4 = arith.select %c, %ra, %rb : tensor<16x16x!tt.ptr<f16>>
  tt.return
}

// -----

// CHECK-LABEL: @select_iv_cond
// Both arms advance by 16 per iteration, one tile apart. A condition that
// changes between iterations switches arms (the selected offsets go 0, 0, 32,
// 32, ...), so the select's IV stride is unknown in that loop, also for arms
// that do not depend on it (%s2); an invariant condition keeps it (%s1, and
// the inner loop for %s3).
tt.func @select_iv_cond(%n: i32, %c: i1, %x: tensor<16xi32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %c16_i32 = arith.constant 16 : i32
  %r = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  scf.for %iv = %c0_i32 to %n step %c1_i32 : i32 {
    %0 = arith.muli %iv, %c16_i32 : i32
    %1 = arith.subi %0, %c16_i32 : i32
    %2 = tt.splat %0 : i32 -> tensor<16xi32>
    %3 = tt.splat %1 : i32 -> tensor<16xi32>
    %t = arith.addi %2, %r : tensor<16xi32>
    %f = arith.addi %3, %r : tensor<16xi32>
    %4 = arith.remsi %iv, %c2_i32 : i32
    %even = arith.cmpi eq, %4, %c0_i32 : i32
    // CHECK: arith.select {{.*}} => stride = [1], iv_strides = {scf.for@{{.*}}: [-1]}
    %s0 = arith.select %even, %t, %f : tensor<16xi32>
    // CHECK: arith.select {{.*}} => stride = [1], iv_strides = {scf.for@{{.*}}: [16]}
    %s1 = arith.select %c, %t, %f : tensor<16xi32>
    // CHECK: arith.select {{.*}} => stride = [-1], iv_strides = {scf.for@{{.*}}: [-1]}
    %s2 = arith.select %even, %r, %x : tensor<16xi32>
    scf.for %j = %c0_i32 to %n step %c1_i32 : i32 {
      // CHECK: arith.select {{.*}} => stride = [1], iv_strides = {scf.for@{{[^,]*}}: [-1]}{{$}}
      %s3 = arith.select %even, %t, %f : tensor<16xi32>
    }
  }
  tt.return
}

// -----

// CHECK-LABEL: @select_struct_arms
// At LLVM-lowering time a noinline function's tensor argument is a struct of
// another rank: the select gets the unknown stride of its own rank, which a
// tt.contiguity hint can still refine.
tt.func @select_struct_arms(%sa: !llvm.struct<(ptr<1>, ptr<1>)>, %sb: !llvm.struct<(ptr<1>, ptr<1>)>, %base: !tt.ptr<f16>, %c: i1) {
  %a = builtin.unrealized_conversion_cast %sa : !llvm.struct<(ptr<1>, ptr<1>)> to tensor<2x16x!tt.ptr<f16>>
  %b = builtin.unrealized_conversion_cast %sb : !llvm.struct<(ptr<1>, ptr<1>)> to tensor<2x16x!tt.ptr<f16>>
  %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %2 = tt.broadcast %1 : tensor<1x16xi32> -> tensor<2x16xi32>
  %3 = tt.splat %base : !tt.ptr<f16> -> tensor<2x16x!tt.ptr<f16>>
  %4 = tt.addptr %3, %2 : tensor<2x16x!tt.ptr<f16>>, tensor<2x16xi32>
  // CHECK: arith.select {{.*}} => stride = [-1, -1]
  %5 = arith.select %c, %4, %a : tensor<2x16x!tt.ptr<f16>>
  // CHECK: arith.select {{.*}} => stride = [-1, 1]
  %6 = arith.select %c, %a, %b {tt.contiguity = dense<[1, 16]> : tensor<2xi32>} : tensor<2x16x!tt.ptr<f16>>
  tt.return
}

// -----

// CHECK-LABEL: @dead_zero_init
// Loop fusion also seeds a fused iter_arg with a zero it never reads: the
// guard is an inner counter that starts at 0. The zero's stride is ignored,
// unless a use can read it (%rd, %live) or the guard does not hold on the
// first iteration (%late, %ne, %th). IV columns still join with the init's.
tt.func @dead_zero_init(%n: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %zero = arith.constant dense<0> : tensor<16x16xi32>
  %cst = arith.constant dense<64> : tensor<16x1xi32>
  %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %2 = arith.muli %1, %cst : tensor<16x1xi32>
  %3 = tt.broadcast %2 : tensor<16x1xi32> -> tensor<16x16xi32>
  %4:8 = scf.for %iv = %c0_i32 to %n step %c1_i32 iter_args(%k = %c0_i32, %dif = %zero, %dsel = %zero, %rd = %zero, %late = %zero, %ne = %zero, %th = %zero, %live = %zero) -> (i32, tensor<16x16xi32>, tensor<16x16xi32>, tensor<16x16xi32>, tensor<16x16xi32>, tensor<16x16xi32>, tensor<16x16xi32>, tensor<16x16xi32>) : i32 {
    %first = arith.cmpi eq, %k, %c0_i32 : i32
    %second = arith.cmpi eq, %k, %c1_i32 : i32
    %notfirst = arith.cmpi ne, %k, %c0_i32 : i32
    %ivs = tt.splat %iv : i32 -> tensor<16x16xi32>
    %strided = arith.addi %3, %ivs : tensor<16x16xi32>
    // CHECK: } => stride = [64, 0]
    %qif = scf.if %first -> (tensor<16x16xi32>) {
      scf.yield %3 : tensor<16x16xi32>
    } else {
      scf.yield %dif : tensor<16x16xi32>
    }
    // CHECK: arith.select {{.*}} => stride = [64, 0], iv_strides = {scf.for@{{.*}}: [-1, -1]}
    %qsel = arith.select %first, %strided, %dsel : tensor<16x16xi32>
    // CHECK: arith.select {{.*}} => stride = [-1, 0]
    %qrd = arith.select %first, %rd, %3 : tensor<16x16xi32>
    // CHECK: arith.select {{.*}} => stride = [-1, 0]
    %qlate = arith.select %second, %3, %late : tensor<16x16xi32>
    // CHECK: arith.select {{.*}} => stride = [-1, 0]
    %qne = arith.select %notfirst, %3, %ne : tensor<16x16xi32>
    // CHECK: } => stride = [-1, 0]
    %qth = scf.if %first -> (tensor<16x16xi32>) {
      scf.yield %th : tensor<16x16xi32>
    } else {
      scf.yield %3 : tensor<16x16xi32>
    }
    // CHECK: arith.addi {{.*}} => stride = [-1, 0]
    %r = arith.addi %live, %3 : tensor<16x16xi32>
    // CHECK: arith.select {{.*}} => stride = [-1, 0]
    %qlv = arith.select %first, %3, %live : tensor<16x16xi32>
    %kn = arith.addi %k, %c1_i32 : i32
    scf.yield %kn, %qif, %qsel, %qrd, %qlate, %qne, %qth, %r : i32, tensor<16x16xi32>, tensor<16x16xi32>, tensor<16x16xi32>, tensor<16x16xi32>, tensor<16x16xi32>, tensor<16x16xi32>, tensor<16x16xi32>
  }
  // A zero-trip loop returns the zero inits, so the loop results join them.
  // CHECK-NOT: } => stride = [64, 0]
  tt.return
}
