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
  tt.return
}

// -----

// CHECK-LABEL: @divsi
tt.func @divsi() {
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
  // stride(1) / const(4): not evenly divisible => -1
  // CHECK: arith.divsi {{.*}} => stride = [-1]
  %3 = arith.divsi %0, %cst4 : tensor<128xi32>
  // stride(0) / constant = 0
  // CHECK: arith.constant {{.*}} => stride = [0]
  %cst_splat = arith.constant dense<100> : tensor<128xi32>
  // CHECK: arith.divsi {{.*}} => stride = [0]
  %4 = arith.divsi %cst_splat, %cst4 : tensor<128xi32>
  tt.return
}

// -----

// CHECK-LABEL: @rem
tt.func @rem() {
  // CHECK: tt.make_range {{.*}} => stride = [1]
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // CHECK: arith.constant {{.*}} => stride = [0]
  %cst_mod = arith.constant dense<512> : tensor<128xi32>
  // lhs stride >= 0 (1) and rhs stride == 0 => preserves lhs stride (1)
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

// CHECK-LABEL: @make_tensor_ptr_known_strides
tt.func @make_tensor_ptr_known_strides(%arg0: !tt.ptr<f16>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c32_i64 = arith.constant 32 : i64
  %c128_i64 = arith.constant 128 : i64
  // constant stride operands [32, 1] => stride = [32, 1]
  // CHECK: tt.make_tensor_ptr {{.*}} => stride = [32, 1]
  %0 = tt.make_tensor_ptr %arg0, [%c128_i64, %c32_i64], [%c32_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : !tt.ptr<tensor<128x32xf16>>
  tt.return
}

// -----

// CHECK-LABEL: @make_tensor_ptr_unknown_stride
tt.func @make_tensor_ptr_unknown_stride(%arg0: !tt.ptr<f16>, %stride: i64) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c128_i64 = arith.constant 128 : i64
  %c32_i64 = arith.constant 32 : i64
  // non-constant stride operand => unknown stride
  // CHECK: tt.make_tensor_ptr {{.*}} => stride = [-1, -1]
  %0 = tt.make_tensor_ptr %arg0, [%c128_i64, %c32_i64], [%stride, %stride], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : !tt.ptr<tensor<128x32xf16>>
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
  %0 = tt.make_tensor_descriptor %arg0, [%c128_i32, %c32_i32], [%c32_i64, %c1_i64] : <f16>, <tensor<128x32xf16>>
  tt.return
}

// -----

// CHECK-LABEL: @make_tensor_desc_unknown_stride
tt.func @make_tensor_desc_unknown_stride(%arg0: !tt.ptr<f16>, %stride: i64) {
  %c128_i32 = arith.constant 128 : i32
  %c32_i32 = arith.constant 32 : i32
  // non-constant stride operand => unknown stride
  // CHECK: tt.make_tensor_descriptor {{.*}} => stride = [-1, -1]
  %0 = tt.make_tensor_descriptor %arg0, [%c128_i32, %c32_i32], [%stride, %stride] : <f16>, <tensor<128x32xf16>>
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
  %desc = tt.make_tensor_descriptor %arg0, [%c128_i32, %c32_i32], [%c1_i64, %c1_i64] : <f16>, <tensor<128x32xf16>>
  // descriptor_load always returns pessimistic stride regardless of descriptor
  // CHECK: tt.descriptor_load {{.*}} => stride = [-1, -1]
  %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<128x32xf16>> -> tensor<128x32xf16>
  tt.return
}

// -----

// CHECK-LABEL: @ptr_offset_pattern
// This tests a realistic pattern: computing 2D pointer offsets
tt.func @ptr_offset_pattern(%arg0: i32, %arg1: tensor<128x1xi32>) {
  // CHECK: arith.constant {{.*}} => stride = [0, 0]
  %cst_0 = arith.constant dense<512> : tensor<128x1xi32>
  // CHECK: arith.constant {{.*}} => stride = [0]
  %cst_1 = arith.constant dense<512> : tensor<128xi32>
  // CHECK: arith.constant {{.*}} => stride = [0]
  %c128_i32 = arith.constant 128 : i32
  // scalar * scalar => stride = [0]
  // CHECK: arith.muli {{.*}} => stride = [0]
  %0 = arith.muli %arg0, %c128_i32 : i32
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

// CHECK-LABEL: @divui
tt.func @divui() {
  // CHECK: tt.make_range {{.*}} => stride = [1]
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // CHECK: arith.constant {{.*}} => stride = [0]
  %cst2 = arith.constant dense<2> : tensor<128xi32>
  // stride(range)*2 = 2, then 2 / 2 = 1
  // CHECK: arith.muli {{.*}} => stride = [2]
  %1 = arith.muli %0, %cst2 : tensor<128xi32>
  // CHECK: arith.divui {{.*}} => stride = [1]
  %2 = arith.divui %1, %cst2 : tensor<128xi32>
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

// CHECK-LABEL: @advance_passthrough
tt.func @advance_passthrough(%arg0: !tt.ptr<f16>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c128_i64 = arith.constant 128 : i64
  %c32_i64 = arith.constant 32 : i64
  // CHECK: tt.make_tensor_ptr {{.*}} => stride = [1, 1]
  %ptr = tt.make_tensor_ptr %arg0, [%c128_i64, %c32_i64], [%c1_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : !tt.ptr<tensor<128x32xf16>>
  // advance passes through stride from operand 0
  // CHECK: tt.advance {{.*}} => stride = [1, 1]
  %1 = tt.advance %ptr, [%c0_i32, %c0_i32] : !tt.ptr<tensor<128x32xf16>>
  tt.return
}
