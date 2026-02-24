// RUN: triton-opt %s -split-input-file -triton-intel-fuse-transpose | FileCheck %s

// COM: Block pointer: make_tensor_ptr -> load -> trans -> dot, not in a loop.
tt.func public @fuseTransposeBlockPtr(%arg0: !tt.ptr<f16>, %arg1: tensor<256x128xf16>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c256_i64 = arith.constant 256 : i64
  %c512_i64 = arith.constant 512 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
  %ptr = tt.make_tensor_ptr %arg0, [%c512_i64, %c256_i64], [%c256_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<256x64xf16>>
  %load = tt.load %ptr : !tt.ptr<tensor<256x64xf16>>
  %trans = tt.trans %load {order = array<i32: 1, 0>} : tensor<256x64xf16> -> tensor<64x256xf16>
  %dot = tt.dot %trans, %arg1, %cst, inputPrecision = tf32 : tensor<64x256xf16> * tensor<256x128xf16> -> tensor<64x128xf32>
  tt.return
}
// CHECK-LABEL: fuseTransposeBlockPtr
// CHECK-NOT: tt.trans
// CHECK: [[PTR:%.*]] = tt.make_tensor_ptr %arg0, [%c256_i64, %c512_i64], [%c1_i64, %c256_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x256xf16>>
// CHECK: [[LOAD:%.*]] = tt.load [[PTR]] : !tt.ptr<tensor<64x256xf16>>
// CHECK: tt.dot [[LOAD]], %arg1, {{.*}}, inputPrecision = tf32 : tensor<64x256xf16> * tensor<256x128xf16> -> tensor<64x128xf32>

// -----

// COM: Tensor descriptor: make_tensor_desc -> descriptor_load -> trans -> dot, not in a loop.
tt.func public @fuseTransposeTensorDesc(%arg0: !tt.ptr<f16>, %arg1: tensor<256x128xf16>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c256_i32 = arith.constant 256 : i32
  %c256_i64 = arith.constant 256 : i64
  %c512_i32 = arith.constant 512 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
  %desc = tt.make_tensor_descriptor %arg0, [%c512_i32, %c256_i32], [%c256_i64, %c1_i64] : <f16>, <tensor<256x64xf16>>
  %load = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<256x64xf16>> -> tensor<256x64xf16>
  %trans = tt.trans %load {order = array<i32: 1, 0>} : tensor<256x64xf16> -> tensor<64x256xf16>
  %dot = tt.dot %trans, %arg1, %cst, inputPrecision = tf32 : tensor<64x256xf16> * tensor<256x128xf16> -> tensor<64x128xf32>
  tt.return
}
// CHECK-LABEL: fuseTransposeTensorDesc
// CHECK-NOT: tt.trans
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor %arg0, [%c256_i32, %c512_i32], [%c1_i64, %c256_i64] : <f16>, <tensor<64x256xf16>>
// CHECK: [[LOAD:%.*]] = tt.descriptor_load [[DESC]][%c0_i32, %c0_i32] : !tt.tensordesc<tensor<64x256xf16>> -> tensor<64x256xf16>
// CHECK: tt.dot [[LOAD]], %arg1, {{.*}}, inputPrecision = tf32 : tensor<64x256xf16> * tensor<256x128xf16> -> tensor<64x128xf32>

// -----

// COM: Block pointer in a loop: make_tensor_ptr -> advance -> load -> trans -> dot.
tt.func public @fuseTransposeBlockPtrLoop(%arg0: !tt.ptr<f16>, %arg1: tensor<256x128xf16>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c64_i32 = arith.constant 64 : i32
  %c256_i64 = arith.constant 256 : i64
  %c512_i32 = arith.constant 512 : i32
  %c512_i64 = arith.constant 512 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
  %ptr = tt.make_tensor_ptr %arg0, [%c512_i64, %c256_i64], [%c256_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<256x64xf16>>
  %res:2 = scf.for %iv = %c0_i32 to %c512_i32 step %c64_i32 iter_args(%acc = %cst, %p = %ptr) -> (tensor<64x128xf32>, !tt.ptr<tensor<256x64xf16>>) : i32 {
    %load = tt.load %p : !tt.ptr<tensor<256x64xf16>>
    %trans = tt.trans %load {order = array<i32: 1, 0>} : tensor<256x64xf16> -> tensor<64x256xf16>
    %dot = tt.dot %trans, %arg1, %acc, inputPrecision = tf32 : tensor<64x256xf16> * tensor<256x128xf16> -> tensor<64x128xf32>
    %next = tt.advance %p, [%c64_i32, %c0_i32] : <tensor<256x64xf16>>
    scf.yield %dot, %next : tensor<64x128xf32>, !tt.ptr<tensor<256x64xf16>>
  }
  tt.return
}
// CHECK-LABEL: fuseTransposeBlockPtrLoop
// CHECK-NOT: tt.trans
// CHECK: [[PTR:%.*]] = tt.make_tensor_ptr %arg0, [%c256_i64, %c512_i64], [%c1_i64, %c256_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x256xf16>>
// CHECK: scf.for {{.*}} iter_args({{.*}}, [[LOOP_PTR:%.*]] = [[PTR]])
// CHECK:   [[LOAD:%.*]] = tt.load [[LOOP_PTR]] : !tt.ptr<tensor<64x256xf16>>
// CHECK:   tt.dot [[LOAD]], %arg1, {{.*}}, inputPrecision = tf32 : tensor<64x256xf16> * tensor<256x128xf16> -> tensor<64x128xf32>
// CHECK:   [[ADV:%.*]] = tt.advance [[LOOP_PTR]], [%c0_i32, %c64_i32] : <tensor<64x256xf16>>
// CHECK:   scf.yield {{.*}}, [[ADV]]

// -----

// COM: Tensor descriptor in a loop: make_tensor_desc -> for iter_arg -> descriptor_load -> trans -> dot.
// COM: This is the tensor descriptor analog of fuseTransposeBlockPtrLoop. The descriptor is passed
// COM: as a loop-carried value. fuseMakeTensorDescOp should propagate through the loop (updating the
// COM: for init_arg and block arg types) the same way fuseMakeTensorPtrOp does for block pointers.
tt.func public @fuseTransposeTensorDescLoop(%arg0: !tt.ptr<f16>, %arg1: tensor<256x128xf16>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c64_i32 = arith.constant 64 : i32
  %c256_i32 = arith.constant 256 : i32
  %c256_i64 = arith.constant 256 : i64
  %c512_i32 = arith.constant 512 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
  %desc = tt.make_tensor_descriptor %arg0, [%c512_i32, %c256_i32], [%c256_i64, %c1_i64] : <f16>, <tensor<256x64xf16>>
  %res:2 = scf.for %iv = %c0_i32 to %c512_i32 step %c64_i32 iter_args(%acc = %cst, %d = %desc) -> (tensor<64x128xf32>, !tt.tensordesc<tensor<256x64xf16>>) : i32 {
    %load = tt.descriptor_load %d[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<256x64xf16>> -> tensor<256x64xf16>
    %trans = tt.trans %load {order = array<i32: 1, 0>} : tensor<256x64xf16> -> tensor<64x256xf16>
    %dot = tt.dot %trans, %arg1, %acc, inputPrecision = tf32 : tensor<64x256xf16> * tensor<256x128xf16> -> tensor<64x128xf32>
    scf.yield %dot, %d : tensor<64x128xf32>, !tt.tensordesc<tensor<256x64xf16>>
  }
  tt.return
}
// CHECK-LABEL: fuseTransposeTensorDescLoop
// CHECK-NOT: tt.trans
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor {{.*}} : <f16>, <tensor<64x256xf16>>
// CHECK: scf.for
// CHECK:   [[LOAD:%.*]] = tt.descriptor_load [[DESC]][%c0_i32, %c0_i32] : !tt.tensordesc<tensor<64x256xf16>> -> tensor<64x256xf16>
// CHECK:   tt.dot [[LOAD]], %arg1, {{.*}}, inputPrecision = tf32 : tensor<64x256xf16> * tensor<256x128xf16> -> tensor<64x128xf32>

// -----

// COM: Negative test: trans not feeding a dot op (should not fuse).
tt.func public @noFuseNotDot(%arg0: !tt.ptr<f16>) -> tensor<64x256xf16> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c256_i64 = arith.constant 256 : i64
  %c512_i64 = arith.constant 512 : i64
  %ptr = tt.make_tensor_ptr %arg0, [%c512_i64, %c256_i64], [%c256_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<256x64xf16>>
  %load = tt.load %ptr : !tt.ptr<tensor<256x64xf16>>
  %trans = tt.trans %load {order = array<i32: 1, 0>} : tensor<256x64xf16> -> tensor<64x256xf16>
  tt.return %trans : tensor<64x256xf16>
}
// CHECK-LABEL: noFuseNotDot
// CHECK: tt.load
// CHECK: tt.trans

// -----

// COM: Negative test: trans with non-simple order (not [1, 0]) should not fuse.
tt.func public @noFuseNonSimpleOrder(%arg0: !tt.ptr<f16>, %arg1: tensor<64x2x128xf16>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c32_i64 = arith.constant 32 : i64
  %c256_i64 = arith.constant 256 : i64
  %c512_i64 = arith.constant 512 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<64x2x128xf32>
  %ptr = tt.make_tensor_ptr %arg0, [%c512_i64, %c256_i64, %c32_i64], [%c256_i64, %c32_i64, %c1_i64], [%c0_i32, %c0_i32, %c0_i32] {order = array<i32: 2, 1, 0>} : <tensor<2x64x128xf16>>
  %load = tt.load %ptr : !tt.ptr<tensor<2x64x128xf16>>
  %trans = tt.trans %load {order = array<i32: 1, 0, 2>} : tensor<2x64x128xf16> -> tensor<64x2x128xf16>
  tt.return
}
// CHECK-LABEL: noFuseNonSimpleOrder
// CHECK: tt.load
// CHECK: tt.trans

// -----

// COM: Block pointer in a loop: make_tensor_ptr NOT loop-carried. The ptr is defined
// COM: outside the loop and advanced inside; no iter_arg for the ptr.
tt.func public @fuseTransposeBlockPtrLoopNotCarried(%arg0: !tt.ptr<f16>, %arg1: tensor<256x128xf16>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c64_i32 = arith.constant 64 : i32
  %c256_i64 = arith.constant 256 : i64
  %c512_i32 = arith.constant 512 : i32
  %c512_i64 = arith.constant 512 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
  %ptr = tt.make_tensor_ptr %arg0, [%c512_i64, %c256_i64], [%c256_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<256x64xf16>>
  %res = scf.for %iv = %c0_i32 to %c512_i32 step %c64_i32 iter_args(%acc = %cst) -> (tensor<64x128xf32>) : i32 {
    %adv = tt.advance %ptr, [%c64_i32, %c0_i32] : <tensor<256x64xf16>>
    %load = tt.load %adv : !tt.ptr<tensor<256x64xf16>>
    %trans = tt.trans %load {order = array<i32: 1, 0>} : tensor<256x64xf16> -> tensor<64x256xf16>
    %dot = tt.dot %trans, %arg1, %acc, inputPrecision = tf32 : tensor<64x256xf16> * tensor<256x128xf16> -> tensor<64x128xf32>
    scf.yield %dot : tensor<64x128xf32>
  }
  tt.return
}
// CHECK-LABEL: fuseTransposeBlockPtrLoopNotCarried
// CHECK-NOT: tt.trans
// CHECK: [[PTR:%.*]] = tt.make_tensor_ptr %arg0, [%c256_i64, %c512_i64], [%c1_i64, %c256_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x256xf16>>
// CHECK: scf.for
// CHECK:   [[ADV:%.*]] = tt.advance [[PTR]], [%c0_i32, %c64_i32] : <tensor<64x256xf16>>
// CHECK:   [[LOAD:%.*]] = tt.load [[ADV]] : !tt.ptr<tensor<64x256xf16>>
// CHECK:   tt.dot [[LOAD]], %arg1, {{.*}}, inputPrecision = tf32 : tensor<64x256xf16> * tensor<256x128xf16> -> tensor<64x128xf32>

// -----

// COM: Block pointer: same make_tensor_ptr root used in 2 separate loops.
// COM: The root should be duplicated (one per loop chain).
tt.func public @fuseTransposeBlockPtrTwoLoops(%arg0: i32, %arg1: !tt.ptr<f16>, %arg2: tensor<256x128xf16>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c64_i32 = arith.constant 64 : i32
  %c256_i64 = arith.constant 256 : i64
  %c512_i64 = arith.constant 512 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
  %ptr = tt.make_tensor_ptr %arg1, [%c512_i64, %c256_i64], [%c256_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<256x64xf16>>
  %res1 = scf.for %iv = %c0_i32 to %arg0 step %c64_i32 iter_args(%acc = %cst) -> (tensor<64x128xf32>) : i32 {
    %adv = tt.advance %ptr, [%c64_i32, %c0_i32] : <tensor<256x64xf16>>
    %load = tt.load %adv : !tt.ptr<tensor<256x64xf16>>
    %trans = tt.trans %load {order = array<i32: 1, 0>} : tensor<256x64xf16> -> tensor<64x256xf16>
    %dot = tt.dot %trans, %arg2, %acc, inputPrecision = tf32 : tensor<64x256xf16> * tensor<256x128xf16> -> tensor<64x128xf32>
    scf.yield %dot : tensor<64x128xf32>
  }
  %res2 = scf.for %iv = %c0_i32 to %arg0 step %c64_i32 iter_args(%acc = %cst) -> (tensor<64x128xf32>) : i32 {
    %adv = tt.advance %ptr, [%c64_i32, %c0_i32] : <tensor<256x64xf16>>
    %load = tt.load %adv : !tt.ptr<tensor<256x64xf16>>
    %trans = tt.trans %load {order = array<i32: 1, 0>} : tensor<256x64xf16> -> tensor<64x256xf16>
    %dot = tt.dot %trans, %arg2, %acc, inputPrecision = tf32 : tensor<64x256xf16> * tensor<256x128xf16> -> tensor<64x128xf32>
    scf.yield %dot : tensor<64x128xf32>
  }
  tt.return
}
// CHECK-LABEL: fuseTransposeBlockPtrTwoLoops
// CHECK-NOT: tt.trans
// CHECK-COUNT-2: tt.make_tensor_ptr %arg1, [%c256_i64, %c512_i64], [%c1_i64, %c256_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x256xf16>>
// CHECK: scf.for
// CHECK:   [[ADV1:%.*]] = tt.advance {{.*}}, [%c0_i32, %c64_i32] : <tensor<64x256xf16>>
// CHECK:   [[LOAD1:%.*]] = tt.load [[ADV1]] : !tt.ptr<tensor<64x256xf16>>
// CHECK:   tt.dot [[LOAD1]], %arg2, {{.*}}, inputPrecision = tf32 : tensor<64x256xf16> * tensor<256x128xf16> -> tensor<64x128xf32>
// CHECK: scf.for
// CHECK:   [[ADV2:%.*]] = tt.advance {{.*}}, [%c0_i32, %c64_i32] : <tensor<64x256xf16>>
// CHECK:   [[LOAD2:%.*]] = tt.load [[ADV2]] : !tt.ptr<tensor<64x256xf16>>
// CHECK:   tt.dot [[LOAD2]], %arg2, {{.*}}, inputPrecision = tf32 : tensor<64x256xf16> * tensor<256x128xf16> -> tensor<64x128xf32>

// -----

// COM: Block pointer: 3 loops with overlapping def-use chains.
// COM: Loops 1 and 3 share the same advance (overlapping chains) so cannot be fused.
// COM: Loop 2 has its own advance inside the loop, so it can be fused.
tt.func public @fuseTransposeBlockPtrOverlappingChains(%arg0: i32, %arg1: !tt.ptr<f16>, %arg2: tensor<256x128xf16>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c64_i32 = arith.constant 64 : i32
  %c256_i64 = arith.constant 256 : i64
  %c512_i64 = arith.constant 512 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
  %ptr = tt.make_tensor_ptr %arg1, [%c512_i64, %c256_i64], [%c256_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<256x64xf16>>
  %adv0 = tt.advance %ptr, [%arg0, %c0_i32] : <tensor<256x64xf16>>
  %res1 = scf.for %iv = %c0_i32 to %arg0 step %c64_i32 iter_args(%acc = %cst) -> (tensor<64x128xf32>) : i32 {
    %load = tt.load %adv0 : !tt.ptr<tensor<256x64xf16>>
    %trans = tt.trans %load {order = array<i32: 1, 0>} : tensor<256x64xf16> -> tensor<64x256xf16>
    %dot = tt.dot %trans, %arg2, %acc, inputPrecision = tf32 : tensor<64x256xf16> * tensor<256x128xf16> -> tensor<64x128xf32>
    scf.yield %dot : tensor<64x128xf32>
  }
  %res2 = scf.for %iv = %c0_i32 to %arg0 step %c64_i32 iter_args(%acc = %cst) -> (tensor<64x128xf32>) : i32 {
    %adv = tt.advance %ptr, [%c64_i32, %c0_i32] : <tensor<256x64xf16>>
    %load = tt.load %adv : !tt.ptr<tensor<256x64xf16>>
    %trans = tt.trans %load {order = array<i32: 1, 0>} : tensor<256x64xf16> -> tensor<64x256xf16>
    %dot = tt.dot %trans, %arg2, %acc, inputPrecision = tf32 : tensor<64x256xf16> * tensor<256x128xf16> -> tensor<64x128xf32>
    scf.yield %dot : tensor<64x128xf32>
  }
  %res3 = scf.for %iv = %c0_i32 to %arg0 step %c64_i32 iter_args(%acc = %cst) -> (tensor<64x128xf32>) : i32 {
    %load = tt.load %adv0 : !tt.ptr<tensor<256x64xf16>>
    %trans = tt.trans %load {order = array<i32: 1, 0>} : tensor<256x64xf16> -> tensor<64x256xf16>
    %dot = tt.dot %trans, %arg2, %acc, inputPrecision = tf32 : tensor<64x256xf16> * tensor<256x128xf16> -> tensor<64x128xf32>
    scf.yield %dot : tensor<64x128xf32>
  }
  tt.return
}
// CHECK-LABEL: fuseTransposeBlockPtrOverlappingChains
// CHECK: [[PTR1:%.*]] = tt.make_tensor_ptr %arg1, [%c256_i64, %c512_i64], [%c1_i64, %c256_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x256xf16>>
// CHECK: scf.for
// CHECK:   tt.trans
// CHECK: scf.for
// CHECK-NOT: tt.trans
// CHECK:   [[ADV:%.*]] = tt.advance [[PTR1]], [%c0_i32, %c64_i32] : <tensor<64x256xf16>>
// CHECK:   [[LOAD:%.*]] = tt.load [[ADV]] : !tt.ptr<tensor<64x256xf16>>
// CHECK:   tt.dot [[LOAD]], %arg2, {{.*}}, inputPrecision = tf32 : tensor<64x256xf16> * tensor<256x128xf16> -> tensor<64x128xf32>
// CHECK: scf.for
// CHECK:   tt.trans

// -----

// COM: Tensor descriptor in a loop: descriptor NOT loop-carried. The desc is defined
// COM: outside the loop and used directly inside (no iter_arg for the desc).
tt.func public @fuseTransposeTensorDescNotCarried(%arg0: !tt.ptr<f16>, %arg1: tensor<256x128xf16>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c64_i32 = arith.constant 64 : i32
  %c256_i32 = arith.constant 256 : i32
  %c256_i64 = arith.constant 256 : i64
  %c512_i32 = arith.constant 512 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
  %desc = tt.make_tensor_descriptor %arg0, [%c512_i32, %c256_i32], [%c256_i64, %c1_i64] : <f16>, <tensor<256x64xf16>>
  %res = scf.for %iv = %c0_i32 to %c512_i32 step %c64_i32 iter_args(%acc = %cst) -> (tensor<64x128xf32>) : i32 {
    %load = tt.descriptor_load %desc[%iv, %c0_i32] : !tt.tensordesc<tensor<256x64xf16>> -> tensor<256x64xf16>
    %trans = tt.trans %load {order = array<i32: 1, 0>} : tensor<256x64xf16> -> tensor<64x256xf16>
    %dot = tt.dot %trans, %arg1, %acc, inputPrecision = tf32 : tensor<64x256xf16> * tensor<256x128xf16> -> tensor<64x128xf32>
    scf.yield %dot : tensor<64x128xf32>
  }
  tt.return
}
// CHECK-LABEL: fuseTransposeTensorDescNotCarried
// CHECK-NOT: tt.trans
// CHECK: [[DESC:%.*]] = tt.make_tensor_descriptor %arg0, [%c256_i32, %c512_i32], [%c1_i64, %c256_i64] : <f16>, <tensor<64x256xf16>>
// CHECK: scf.for
// CHECK:   [[LOAD:%.*]] = tt.descriptor_load [[DESC]][%c0_i32, {{.*}}] : !tt.tensordesc<tensor<64x256xf16>> -> tensor<64x256xf16>
// CHECK:   tt.dot [[LOAD]], %arg1, {{.*}}, inputPrecision = tf32 : tensor<64x256xf16> * tensor<256x128xf16> -> tensor<64x128xf32>

// -----

// COM: Tensor descriptor: same make_tensor_descriptor root used in 2 separate loops.
// COM: The root should be duplicated (one per loop chain).
tt.func public @fuseTransposeTensorDescTwoLoops(%arg0: i32, %arg1: !tt.ptr<f16>, %arg2: tensor<256x128xf16>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c64_i32 = arith.constant 64 : i32
  %c256_i32 = arith.constant 256 : i32
  %c256_i64 = arith.constant 256 : i64
  %c512_i32 = arith.constant 512 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
  %desc = tt.make_tensor_descriptor %arg1, [%c512_i32, %c256_i32], [%c256_i64, %c1_i64] : <f16>, <tensor<256x64xf16>>
  %res1 = scf.for %iv = %c0_i32 to %arg0 step %c64_i32 iter_args(%acc = %cst) -> (tensor<64x128xf32>) : i32 {
    %load = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<256x64xf16>> -> tensor<256x64xf16>
    %trans = tt.trans %load {order = array<i32: 1, 0>} : tensor<256x64xf16> -> tensor<64x256xf16>
    %dot = tt.dot %trans, %arg2, %acc, inputPrecision = tf32 : tensor<64x256xf16> * tensor<256x128xf16> -> tensor<64x128xf32>
    scf.yield %dot : tensor<64x128xf32>
  }
  %res2 = scf.for %iv = %c0_i32 to %arg0 step %c64_i32 iter_args(%acc = %cst) -> (tensor<64x128xf32>) : i32 {
    %load = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<256x64xf16>> -> tensor<256x64xf16>
    %trans = tt.trans %load {order = array<i32: 1, 0>} : tensor<256x64xf16> -> tensor<64x256xf16>
    %dot = tt.dot %trans, %arg2, %acc, inputPrecision = tf32 : tensor<64x256xf16> * tensor<256x128xf16> -> tensor<64x128xf32>
    scf.yield %dot : tensor<64x128xf32>
  }
  tt.return
}
// CHECK-LABEL: fuseTransposeTensorDescTwoLoops
// CHECK-NOT: tt.trans
// CHECK-COUNT-2: tt.make_tensor_descriptor %arg1, [%c256_i32, %c512_i32], [%c1_i64, %c256_i64] : <f16>, <tensor<64x256xf16>>
// CHECK: scf.for
// CHECK:   [[LOAD1:%.*]] = tt.descriptor_load {{.*}}[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<64x256xf16>> -> tensor<64x256xf16>
// CHECK:   tt.dot [[LOAD1]], %arg2, {{.*}}, inputPrecision = tf32 : tensor<64x256xf16> * tensor<256x128xf16> -> tensor<64x128xf32>
// CHECK: scf.for
// CHECK:   [[LOAD2:%.*]] = tt.descriptor_load {{.*}}[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<64x256xf16>> -> tensor<64x256xf16>
// CHECK:   tt.dot [[LOAD2]], %arg2, {{.*}}, inputPrecision = tf32 : tensor<64x256xf16> * tensor<256x128xf16> -> tensor<64x128xf32>

// -----

// COM: Negative test: loop result (ptr) used after the loop prevents fusion.
tt.func public @noFuseBlockPtrLoopResultUsed(%arg0: !tt.ptr<f16>, %arg1: tensor<256x128xf16>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c64_i32 = arith.constant 64 : i32
  %c256_i64 = arith.constant 256 : i64
  %c512_i32 = arith.constant 512 : i32
  %c512_i64 = arith.constant 512 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
  %ptr = tt.make_tensor_ptr %arg0, [%c512_i64, %c256_i64], [%c256_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<256x64xf16>>
  %res:2 = scf.for %iv = %c0_i32 to %c512_i32 step %c64_i32 iter_args(%acc = %cst, %p = %ptr) -> (tensor<64x128xf32>, !tt.ptr<tensor<256x64xf16>>) : i32 {
    %adv = tt.advance %p, [%c64_i32, %c0_i32] : <tensor<256x64xf16>>
    %load = tt.load %adv : !tt.ptr<tensor<256x64xf16>>
    %trans = tt.trans %load {order = array<i32: 1, 0>} : tensor<256x64xf16> -> tensor<64x256xf16>
    %dot = tt.dot %trans, %arg1, %acc, inputPrecision = tf32 : tensor<64x256xf16> * tensor<256x128xf16> -> tensor<64x128xf32>
    scf.yield %dot, %adv : tensor<64x128xf32>, !tt.ptr<tensor<256x64xf16>>
  }
  // Use the loop result ptr after the loop — prevents fusion.
  %final = tt.advance %res#1, [%c64_i32, %c64_i32] : <tensor<256x64xf16>>
  tt.return
}
// CHECK-LABEL: noFuseBlockPtrLoopResultUsed
// CHECK: tt.trans

// -----

// COM: Negative test: advance has multiple users (load for trans AND a store).
tt.func public @noFuseBlockPtrMultipleUsers(%arg0: !tt.ptr<f16>, %arg1: tensor<256x128xf16>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c64_i32 = arith.constant 64 : i32
  %c256_i64 = arith.constant 256 : i64
  %c512_i32 = arith.constant 512 : i32
  %c512_i64 = arith.constant 512 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
  %cst_val = arith.constant dense<1.000000e+00> : tensor<256x64xf16>
  %ptr = tt.make_tensor_ptr %arg0, [%c512_i64, %c256_i64], [%c256_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<256x64xf16>>
  %res:2 = scf.for %iv = %c0_i32 to %c512_i32 step %c64_i32 iter_args(%acc = %cst, %p = %ptr) -> (tensor<64x128xf32>, !tt.ptr<tensor<256x64xf16>>) : i32 {
    %adv = tt.advance %p, [%c64_i32, %c0_i32] : <tensor<256x64xf16>>
    %load = tt.load %adv : !tt.ptr<tensor<256x64xf16>>
    %trans = tt.trans %load {order = array<i32: 1, 0>} : tensor<256x64xf16> -> tensor<64x256xf16>
    %dot = tt.dot %trans, %arg1, %acc, inputPrecision = tf32 : tensor<64x256xf16> * tensor<256x128xf16> -> tensor<64x128xf32>
    tt.store %adv, %cst_val : !tt.ptr<tensor<256x64xf16>>
    scf.yield %dot, %adv : tensor<64x128xf32>, !tt.ptr<tensor<256x64xf16>>
  }
  tt.return
}
// CHECK-LABEL: noFuseBlockPtrMultipleUsers
// CHECK: tt.trans

// -----

// COM: Negative test: block ptr comes from an scf.if (current limitation).
tt.func public @noFuseBlockPtrFromIf(%arg0: !tt.ptr<f16>, %arg1: tensor<256x128xf16>, %cond: i1) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c64_i32 = arith.constant 64 : i32
  %c256_i64 = arith.constant 256 : i64
  %c512_i32 = arith.constant 512 : i32
  %c512_i64 = arith.constant 512 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
  %ptrA = tt.make_tensor_ptr %arg0, [%c512_i64, %c256_i64], [%c256_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<256x64xf16>>
  %ptrB = tt.make_tensor_ptr %arg0, [%c512_i64, %c256_i64], [%c256_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<256x64xf16>>
  %advA = tt.advance %ptrA, [%c64_i32, %c0_i32] : <tensor<256x64xf16>>
  %res = scf.for %iv = %c0_i32 to %c512_i32 step %c64_i32 iter_args(%acc = %cst) -> (tensor<64x128xf32>) : i32 {
    %sel = scf.if %cond -> !tt.ptr<tensor<256x64xf16>> {
      scf.yield %advA : !tt.ptr<tensor<256x64xf16>>
    } else {
      scf.yield %ptrB : !tt.ptr<tensor<256x64xf16>>
    }
    %load = tt.load %sel : !tt.ptr<tensor<256x64xf16>>
    %trans = tt.trans %load {order = array<i32: 1, 0>} : tensor<256x64xf16> -> tensor<64x256xf16>
    %dot = tt.dot %trans, %arg1, %acc, inputPrecision = tf32 : tensor<64x256xf16> * tensor<256x128xf16> -> tensor<64x128xf32>
    scf.yield %dot : tensor<64x128xf32>
  }
  tt.return
}
// CHECK-LABEL: noFuseBlockPtrFromIf
// CHECK: tt.trans

// -----

// COM: Negative test: while loop (current limitation — only scf.for is supported).
tt.func public @noFuseBlockPtrWhileLoop(%arg0: !tt.ptr<f16>, %arg1: tensor<256x128xf16>, %cond: i1) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c64_i32 = arith.constant 64 : i32
  %c256_i64 = arith.constant 256 : i64
  %c512_i64 = arith.constant 512 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
  %ptr = tt.make_tensor_ptr %arg0, [%c512_i64, %c256_i64], [%c256_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<256x64xf16>>
  %res:2 = scf.while (%p = %ptr, %i = %c0_i32) : (!tt.ptr<tensor<256x64xf16>>, i32) -> (!tt.ptr<tensor<256x64xf16>>, i32) {
    scf.condition(%cond) %p, %i : !tt.ptr<tensor<256x64xf16>>, i32
  } do {
  ^bb0(%bp: !tt.ptr<tensor<256x64xf16>>, %bi: i32):
    %adv = tt.advance %bp, [%c64_i32, %c0_i32] : <tensor<256x64xf16>>
    %load = tt.load %adv : !tt.ptr<tensor<256x64xf16>>
    %trans = tt.trans %load {order = array<i32: 1, 0>} : tensor<256x64xf16> -> tensor<64x256xf16>
    %dot = tt.dot %trans, %arg1, %cst, inputPrecision = tf32 : tensor<64x256xf16> * tensor<256x128xf16> -> tensor<64x128xf32>
    %ni = arith.addi %bi, %c64_i32 : i32
    scf.yield %adv, %ni : !tt.ptr<tensor<256x64xf16>>, i32
  }
  tt.return
}
// CHECK-LABEL: noFuseBlockPtrWhileLoop
// CHECK: tt.trans

// -----

// COM: Negative test: block ptr yielded by a function call.
tt.func @selectPtr(%cond: i1, %p1: !tt.ptr<tensor<256x64xf16>>, %p2: !tt.ptr<tensor<256x64xf16>>) -> !tt.ptr<tensor<256x64xf16>> attributes {noinline = true} {
  %0 = arith.select %cond, %p1, %p2 : i1, !tt.ptr<tensor<256x64xf16>>
  tt.return %0 : !tt.ptr<tensor<256x64xf16>>
}
tt.func public @noFuseBlockPtrFromCall(%arg0: i32, %arg1: !tt.ptr<f16>, %arg2: tensor<256x128xf16>, %cond: i1) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c64_i32 = arith.constant 64 : i32
  %c256_i64 = arith.constant 256 : i64
  %c512_i64 = arith.constant 512 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
  %ptr = tt.make_tensor_ptr %arg1, [%c512_i64, %c256_i64], [%c256_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<256x64xf16>>
  %res = scf.for %iv = %c0_i32 to %arg0 step %c64_i32 iter_args(%acc = %cst) -> (tensor<64x128xf32>) : i32 {
    %adv1 = tt.advance %ptr, [%iv, %c0_i32] : <tensor<256x64xf16>>
    %adv2 = tt.advance %ptr, [%c0_i32, %iv] : <tensor<256x64xf16>>
    %sel = tt.call @selectPtr(%cond, %adv1, %adv2) : (i1, !tt.ptr<tensor<256x64xf16>>, !tt.ptr<tensor<256x64xf16>>) -> !tt.ptr<tensor<256x64xf16>>
    %load = tt.load %sel : !tt.ptr<tensor<256x64xf16>>
    %trans = tt.trans %load {order = array<i32: 1, 0>} : tensor<256x64xf16> -> tensor<64x256xf16>
    %dot = tt.dot %trans, %arg2, %acc, inputPrecision = tf32 : tensor<64x256xf16> * tensor<256x128xf16> -> tensor<64x128xf32>
    scf.yield %dot : tensor<64x128xf32>
  }
  tt.return
}
// CHECK-LABEL: noFuseBlockPtrFromCall
// CHECK: tt.trans

// -----

// COM: Negative test: tensor descriptor loop result used after the loop prevents fusion.
tt.func public @noFuseTensorDescLoopResultUsed(%arg0: !tt.ptr<f16>, %arg1: tensor<256x128xf16>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c64_i32 = arith.constant 64 : i32
  %c256_i32 = arith.constant 256 : i32
  %c256_i64 = arith.constant 256 : i64
  %c512_i32 = arith.constant 512 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
  %desc = tt.make_tensor_descriptor %arg0, [%c512_i32, %c256_i32], [%c256_i64, %c1_i64] : <f16>, <tensor<256x64xf16>>
  %res:2 = scf.for %iv = %c0_i32 to %c512_i32 step %c64_i32 iter_args(%acc = %cst, %d = %desc) -> (tensor<64x128xf32>, !tt.tensordesc<tensor<256x64xf16>>) : i32 {
    %load = tt.descriptor_load %d[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<256x64xf16>> -> tensor<256x64xf16>
    %trans = tt.trans %load {order = array<i32: 1, 0>} : tensor<256x64xf16> -> tensor<64x256xf16>
    %dot = tt.dot %trans, %arg1, %acc, inputPrecision = tf32 : tensor<64x256xf16> * tensor<256x128xf16> -> tensor<64x128xf32>
    scf.yield %dot, %d : tensor<64x128xf32>, !tt.tensordesc<tensor<256x64xf16>>
  }
  // Use the loop result descriptor after the loop — prevents fusion.
  %final = tt.descriptor_load %res#1[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<256x64xf16>> -> tensor<256x64xf16>
  tt.return
}
// CHECK-LABEL: noFuseTensorDescLoopResultUsed
// COM: Tensor descriptor fusion succeeds because fuseMakeTensorDescOp creates a new
// COM: transposed descriptor independently of the loop structure.
// CHECK-NOT: tt.trans

// -----

// COM: Negative test: descriptor_load result has multiple users (trans chain AND another use).
tt.func public @noFuseTensorDescMultipleUsers(%arg0: !tt.ptr<f16>, %arg1: tensor<256x128xf16>, %arg2: !tt.ptr<f16>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c64_i32 = arith.constant 64 : i32
  %c256_i32 = arith.constant 256 : i32
  %c256_i64 = arith.constant 256 : i64
  %c512_i32 = arith.constant 512 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
  %desc = tt.make_tensor_descriptor %arg0, [%c512_i32, %c256_i32], [%c256_i64, %c1_i64] : <f16>, <tensor<256x64xf16>>
  %storeDesc = tt.make_tensor_descriptor %arg2, [%c512_i32, %c256_i32], [%c256_i64, %c1_i64] : <f16>, <tensor<256x64xf16>>
  %res = scf.for %iv = %c0_i32 to %c512_i32 step %c64_i32 iter_args(%acc = %cst) -> (tensor<64x128xf32>) : i32 {
    %load = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<256x64xf16>> -> tensor<256x64xf16>
    %trans = tt.trans %load {order = array<i32: 1, 0>} : tensor<256x64xf16> -> tensor<64x256xf16>
    %dot = tt.dot %trans, %arg1, %acc, inputPrecision = tf32 : tensor<64x256xf16> * tensor<256x128xf16> -> tensor<64x128xf32>
    // Store the same loaded value — gives descriptor_load an extra user.
    tt.descriptor_store %storeDesc[%c0_i32, %c0_i32], %load : !tt.tensordesc<tensor<256x64xf16>>, tensor<256x64xf16>
    scf.yield %dot : tensor<64x128xf32>
  }
  tt.return
}
// CHECK-LABEL: noFuseTensorDescMultipleUsers
// CHECK: tt.trans

// -----

// COM: Negative test: tensor descriptor comes from an scf.if (current limitation).
tt.func public @noFuseTensorDescFromIf(%arg0: !tt.ptr<f16>, %arg1: tensor<256x128xf16>, %cond: i1) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c64_i32 = arith.constant 64 : i32
  %c256_i32 = arith.constant 256 : i32
  %c256_i64 = arith.constant 256 : i64
  %c512_i32 = arith.constant 512 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
  %descA = tt.make_tensor_descriptor %arg0, [%c512_i32, %c256_i32], [%c256_i64, %c1_i64] : <f16>, <tensor<256x64xf16>>
  %descB = tt.make_tensor_descriptor %arg0, [%c512_i32, %c256_i32], [%c256_i64, %c1_i64] : <f16>, <tensor<256x64xf16>>
  %res = scf.for %iv = %c0_i32 to %c512_i32 step %c64_i32 iter_args(%acc = %cst) -> (tensor<64x128xf32>) : i32 {
    %sel = scf.if %cond -> !tt.tensordesc<tensor<256x64xf16>> {
      scf.yield %descA : !tt.tensordesc<tensor<256x64xf16>>
    } else {
      scf.yield %descB : !tt.tensordesc<tensor<256x64xf16>>
    }
    %load = tt.descriptor_load %sel[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<256x64xf16>> -> tensor<256x64xf16>
    %trans = tt.trans %load {order = array<i32: 1, 0>} : tensor<256x64xf16> -> tensor<64x256xf16>
    %dot = tt.dot %trans, %arg1, %acc, inputPrecision = tf32 : tensor<64x256xf16> * tensor<256x128xf16> -> tensor<64x128xf32>
    scf.yield %dot : tensor<64x128xf32>
  }
  tt.return
}
// CHECK-LABEL: noFuseTensorDescFromIf
// CHECK: tt.trans

// -----

// COM: Negative test: tensor descriptor in a while loop (current limitation).
tt.func public @noFuseTensorDescWhileLoop(%arg0: !tt.ptr<f16>, %arg1: tensor<256x128xf16>, %cond: i1) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c64_i32 = arith.constant 64 : i32
  %c256_i32 = arith.constant 256 : i32
  %c256_i64 = arith.constant 256 : i64
  %c512_i32 = arith.constant 512 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
  %desc = tt.make_tensor_descriptor %arg0, [%c512_i32, %c256_i32], [%c256_i64, %c1_i64] : <f16>, <tensor<256x64xf16>>
  %res:2 = scf.while (%d = %desc, %i = %c0_i32) : (!tt.tensordesc<tensor<256x64xf16>>, i32) -> (!tt.tensordesc<tensor<256x64xf16>>, i32) {
    scf.condition(%cond) %d, %i : !tt.tensordesc<tensor<256x64xf16>>, i32
  } do {
  ^bb0(%bd: !tt.tensordesc<tensor<256x64xf16>>, %bi: i32):
    %load = tt.descriptor_load %bd[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<256x64xf16>> -> tensor<256x64xf16>
    %trans = tt.trans %load {order = array<i32: 1, 0>} : tensor<256x64xf16> -> tensor<64x256xf16>
    %dot = tt.dot %trans, %arg1, %cst, inputPrecision = tf32 : tensor<64x256xf16> * tensor<256x128xf16> -> tensor<64x128xf32>
    %ni = arith.addi %bi, %c64_i32 : i32
    scf.yield %bd, %ni : !tt.tensordesc<tensor<256x64xf16>>, i32
  }
  tt.return
}
// CHECK-LABEL: noFuseTensorDescWhileLoop
// CHECK: tt.trans

// -----

// COM: Negative test: tensor descriptor yielded by a function call.
tt.func @selectDesc(%cond: i1, %d1: !tt.tensordesc<tensor<256x64xf16>>, %d2: !tt.tensordesc<tensor<256x64xf16>>) -> !tt.tensordesc<tensor<256x64xf16>> attributes {noinline = true} {
  %0 = arith.select %cond, %d1, %d2 : i1, !tt.tensordesc<tensor<256x64xf16>>
  tt.return %0 : !tt.tensordesc<tensor<256x64xf16>>
}
tt.func public @noFuseTensorDescFromCall(%arg0: i32, %arg1: !tt.ptr<f16>, %arg2: tensor<256x128xf16>, %cond: i1) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c64_i32 = arith.constant 64 : i32
  %c256_i32 = arith.constant 256 : i32
  %c256_i64 = arith.constant 256 : i64
  %c512_i32 = arith.constant 512 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
  %descA = tt.make_tensor_descriptor %arg1, [%c512_i32, %c256_i32], [%c256_i64, %c1_i64] : <f16>, <tensor<256x64xf16>>
  %descB = tt.make_tensor_descriptor %arg1, [%c512_i32, %c256_i32], [%c256_i64, %c1_i64] : <f16>, <tensor<256x64xf16>>
  %res = scf.for %iv = %c0_i32 to %arg0 step %c64_i32 iter_args(%acc = %cst) -> (tensor<64x128xf32>) : i32 {
    %sel = tt.call @selectDesc(%cond, %descA, %descB) : (i1, !tt.tensordesc<tensor<256x64xf16>>, !tt.tensordesc<tensor<256x64xf16>>) -> !tt.tensordesc<tensor<256x64xf16>>
    %load = tt.descriptor_load %sel[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<256x64xf16>> -> tensor<256x64xf16>
    %trans = tt.trans %load {order = array<i32: 1, 0>} : tensor<256x64xf16> -> tensor<64x256xf16>
    %dot = tt.dot %trans, %arg2, %acc, inputPrecision = tf32 : tensor<64x256xf16> * tensor<256x128xf16> -> tensor<64x128xf32>
    scf.yield %dot : tensor<64x128xf32>
  }
  tt.return
}
// CHECK-LABEL: noFuseTensorDescFromCall
// CHECK: tt.trans
// -----

// COM: Negative test: 3D block pointer with trans [1,0,2] is NOT fused.
// COM: The pass only supports 2D transposes (order [1, 0]).
tt.func public @noFuseTranspose3DBlockPtr(%arg0: !tt.ptr<f16>, %arg1: tensor<4x8x4xf16>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c4_i64 = arith.constant 4 : i64
  %c8_i64 = arith.constant 8 : i64
  %c16_i64 = arith.constant 16 : i64
  %c32_i64 = arith.constant 32 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<4x16x4xf32>
  %ptr = tt.make_tensor_ptr %arg0, [%c16_i64, %c4_i64, %c8_i64], [%c32_i64, %c8_i64, %c1_i64], [%c0_i32, %c0_i32, %c0_i32] {order = array<i32: 2, 1, 0>} : <tensor<16x4x8xf16>>
  %load = tt.load %ptr : !tt.ptr<tensor<16x4x8xf16>>
  %trans = tt.trans %load {order = array<i32: 1, 0, 2>} : tensor<16x4x8xf16> -> tensor<4x16x8xf16>
  %dot = tt.dot %trans, %arg1, %cst, inputPrecision = tf32 : tensor<4x16x8xf16> * tensor<4x8x4xf16> -> tensor<4x16x4xf32>
  tt.return
}
// CHECK-LABEL: noFuseTranspose3DBlockPtr
// CHECK: tt.load
// CHECK: tt.trans

// -----

// COM: Negative test: 3D tensor descriptor with trans [1,0,2] is NOT fused.
// COM: The pass only supports 2D transposes.
tt.func public @noFuseTranspose3DTensorDesc(%arg0: !tt.ptr<f16>, %arg1: tensor<4x8x4xf16>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c4_i32 = arith.constant 4 : i32
  %c8_i32 = arith.constant 8 : i32
  %c8_i64 = arith.constant 8 : i64
  %c16_i32 = arith.constant 16 : i32
  %c32_i64 = arith.constant 32 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<4x16x4xf32>
  %desc = tt.make_tensor_descriptor %arg0, [%c16_i32, %c4_i32, %c8_i32], [%c32_i64, %c8_i64, %c1_i64] : <f16>, <tensor<16x4x8xf16>>
  %load = tt.descriptor_load %desc[%c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<tensor<16x4x8xf16>> -> tensor<16x4x8xf16>
  %trans = tt.trans %load {order = array<i32: 1, 0, 2>} : tensor<16x4x8xf16> -> tensor<4x16x8xf16>
  %dot = tt.dot %trans, %arg1, %cst, inputPrecision = tf32 : tensor<4x16x8xf16> * tensor<4x8x4xf16> -> tensor<4x16x4xf32>
  tt.return
}
// CHECK-LABEL: noFuseTranspose3DTensorDesc
// CHECK: tt.descriptor_load
// CHECK: tt.trans

// -----

// COM: Negative test: 3D block pointer in a loop with trans [1,0,2] is NOT fused.
// COM: The pass only supports 2D transposes.
tt.func public @noFuseTranspose3DBlockPtrLoop(%arg0: !tt.ptr<f16>, %arg1: tensor<4x8x4xf16>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c4_i64 = arith.constant 4 : i64
  %c8_i64 = arith.constant 8 : i64
  %c16_i32 = arith.constant 16 : i32
  %c16_i64 = arith.constant 16 : i64
  %c32_i64 = arith.constant 32 : i64
  %c64_i32 = arith.constant 64 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<4x16x4xf32>
  %ptr = tt.make_tensor_ptr %arg0, [%c16_i64, %c4_i64, %c8_i64], [%c32_i64, %c8_i64, %c1_i64], [%c0_i32, %c0_i32, %c0_i32] {order = array<i32: 2, 1, 0>} : <tensor<16x4x8xf16>>
  %res:2 = scf.for %iv = %c0_i32 to %c64_i32 step %c16_i32 iter_args(%acc = %cst, %p = %ptr) -> (tensor<4x16x4xf32>, !tt.ptr<tensor<16x4x8xf16>>) : i32 {
    %load = tt.load %p : !tt.ptr<tensor<16x4x8xf16>>
    %trans = tt.trans %load {order = array<i32: 1, 0, 2>} : tensor<16x4x8xf16> -> tensor<4x16x8xf16>
    %dot = tt.dot %trans, %arg1, %acc, inputPrecision = tf32 : tensor<4x16x8xf16> * tensor<4x8x4xf16> -> tensor<4x16x4xf32>
    %next = tt.advance %p, [%c16_i32, %c0_i32, %c0_i32] : <tensor<16x4x8xf16>>
    scf.yield %dot, %next : tensor<4x16x4xf32>, !tt.ptr<tensor<16x4x8xf16>>
  }
  tt.return
}
// CHECK-LABEL: noFuseTranspose3DBlockPtrLoop
// CHECK: tt.load
// CHECK: tt.trans

// -----

// COM: Negative test: 3D trans with complex order [2,1,0] should not fuse.
tt.func public @noFuseTranspose3DComplexOrder(%arg0: !tt.ptr<f16>, %arg1: tensor<8x16x4xf16>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c4_i64 = arith.constant 4 : i64
  %c8_i64 = arith.constant 8 : i64
  %c16_i64 = arith.constant 16 : i64
  %c32_i64 = arith.constant 32 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<8x4x4xf32>
  %ptr = tt.make_tensor_ptr %arg0, [%c16_i64, %c4_i64, %c8_i64], [%c32_i64, %c8_i64, %c1_i64], [%c0_i32, %c0_i32, %c0_i32] {order = array<i32: 2, 1, 0>} : <tensor<16x4x8xf16>>
  %load = tt.load %ptr : !tt.ptr<tensor<16x4x8xf16>>
  %trans = tt.trans %load {order = array<i32: 2, 1, 0>} : tensor<16x4x8xf16> -> tensor<8x4x16xf16>
  %dot = tt.dot %trans, %arg1, %cst, inputPrecision = tf32 : tensor<8x4x16xf16> * tensor<8x16x4xf16> -> tensor<8x4x4xf32>
  tt.return
}
// CHECK-LABEL: noFuseTranspose3DComplexOrder
// CHECK: tt.load
// CHECK: tt.trans

// -----

// COM: Negative test: 3D trans with order [0,2,1] (only swapping last two dims) should not fuse.
tt.func public @noFuseTranspose3DDifferentPair(%arg0: !tt.ptr<f16>, %arg1: tensor<4x16x4xf16>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c4_i64 = arith.constant 4 : i64
  %c8_i64 = arith.constant 8 : i64
  %c16_i64 = arith.constant 16 : i64
  %c32_i64 = arith.constant 32 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<4x8x4xf32>
  %ptr = tt.make_tensor_ptr %arg0, [%c4_i64, %c16_i64, %c8_i64], [%c32_i64, %c8_i64, %c1_i64], [%c0_i32, %c0_i32, %c0_i32] {order = array<i32: 2, 1, 0>} : <tensor<4x16x8xf16>>
  %load = tt.load %ptr : !tt.ptr<tensor<4x16x8xf16>>
  %trans = tt.trans %load {order = array<i32: 0, 2, 1>} : tensor<4x16x8xf16> -> tensor<4x8x16xf16>
  %dot = tt.dot %trans, %arg1, %cst, inputPrecision = tf32 : tensor<4x8x16xf16> * tensor<4x16x4xf16> -> tensor<4x8x4xf32>
  tt.return
}
// CHECK-LABEL: noFuseTranspose3DDifferentPair
// CHECK: tt.load
// CHECK: tt.trans
