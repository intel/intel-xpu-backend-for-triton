// RUN: triton-opt %s --triton-intel-rewrite-tensor-descriptor-to-pointer --split-input-file | FileCheck %s
// RUN: env TRITON_INTEL_DISABLE_NATIVE_DESCRIPTOR_SCATTER=1 triton-opt %s --triton-intel-rewrite-tensor-descriptor-to-pointer --split-input-file | FileCheck %s --check-prefix=OFF

// COM: A scatter on a device-side descriptor stays native (later lowered as
// COM: ttig.descriptor_scatter) when its source is a tt.dot result reached only through
// COM: casts and scf.for results, the elements are 16 or 32 bits wide, the last stride is 1
// COM: and the column access is proven 16-byte aligned. Otherwise its makers are expanded
// COM: and the scatter becomes a masked tt.store, as before; OFF expands every scatter.
// COM: x_offsets is an argument everywhere, so no scatter becomes a descriptor store.

// COM: The GEMM epilogue shape: an accumulator loop, a truncf and a scatter at y = 0.
module {
  tt.func public @admit_loop_dot(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %x: tensor<32xi32>, %a: tensor<32x64xbf16>, %b: tensor<64x64xbf16>, %c: tensor<32x64xf32>, %m: i32 {tt.divisibility = 16 : i32}, %s: i64 {tt.divisibility = 16 : i32}, %n: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%m, %m], [%s, %c1_i64] : <bf16>, <1x64xbf16>
    %acc = scf.for %i = %c0_i32 to %n step %c1_i32 iter_args(%k = %c) -> (tensor<32x64xf32>) : i32 {
      %d = tt.dot %a, %b, %k : tensor<32x64xbf16> * tensor<64x64xbf16> -> tensor<32x64xf32>
      scf.yield %d : tensor<32x64xf32>
    }
    %t = arith.truncf %acc : tensor<32x64xf32> to tensor<32x64xbf16>
    tt.descriptor_scatter %desc[%x, %c0_i32], %t : !tt.tensordesc<1x64xbf16>, tensor<32xi32>, i32, tensor<32x64xbf16>
    tt.return
  }
}

// CHECK-LABEL: @admit_loop_dot
// CHECK:       %[[DESC:.*]] = tt.make_tensor_descriptor %arg0, [%arg5, %arg5], [%arg6, %c1_i64] : <bf16>, <1x64xbf16>
// CHECK:       %[[ACC:.*]] = scf.for
// CHECK:       %[[T:.*]] = arith.truncf %[[ACC]]
// CHECK-NEXT:  tt.descriptor_scatter %[[DESC]][%arg1, %c0_i32], %[[T]] : !tt.tensordesc<1x64xbf16>, tensor<32xi32>, i32, tensor<32x64xbf16>
// CHECK-NOT:   tt.store
// OFF-LABEL:   @admit_loop_dot
// OFF-NOT:     tt.make_tensor_descriptor
// OFF:         %[[T:.*]] = arith.truncf
// OFF-NOT:     tt.descriptor_scatter
// OFF:         tt.store %{{[^,]*}}, %[[T]], %{{[^,]*}} : tensor<32x64x!tt.ptr<bf16>>
// OFF-NEXT:    tt.return

// -----

// COM: One function per admission rule; each is admissible but for its rule.
module {
  // COM: An aligned runtime y is kept, here for 32-bit elements stored straight from the dot.
  tt.func public @admit_aligned_y(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %x: tensor<32xi32>, %a: tensor<32x64xbf16>, %b: tensor<64x64xbf16>, %c: tensor<32x64xf32>, %m: i32 {tt.divisibility = 16 : i32}, %s: i64 {tt.divisibility = 16 : i32}) {
    %c128_i32 = arith.constant 128 : i32
    %c1_i64 = arith.constant 1 : i64
    %pid = tt.get_program_id x : i32
    %y = arith.muli %pid, %c128_i32 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%m, %m], [%s, %c1_i64] : <f32>, <1x64xf32>
    %d = tt.dot %a, %b, %c : tensor<32x64xbf16> * tensor<64x64xbf16> -> tensor<32x64xf32>
    tt.descriptor_scatter %desc[%x, %y], %d : !tt.tensordesc<1x64xf32>, tensor<32xi32>, i32, tensor<32x64xf32>
    tt.return
  }
  // COM: y = 1 is not 16-byte aligned ...
  tt.func public @reject_unaligned_y(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %x: tensor<32xi32>, %a: tensor<32x64xbf16>, %b: tensor<64x64xbf16>, %c: tensor<32x64xf32>, %m: i32 {tt.divisibility = 16 : i32}, %s: i64 {tt.divisibility = 16 : i32}) {
    %c1_i32 = arith.constant 1 : i32
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%m, %m], [%s, %c1_i64] : <bf16>, <1x64xbf16>
    %d = tt.dot %a, %b, %c : tensor<32x64xbf16> * tensor<64x64xbf16> -> tensor<32x64xf32>
    %t = arith.truncf %d : tensor<32x64xf32> to tensor<32x64xbf16>
    tt.descriptor_scatter %desc[%x, %c1_i32], %t : !tt.tensordesc<1x64xbf16>, tensor<32xi32>, i32, tensor<32x64xbf16>
    tt.return
  }
  // COM: ... nor is an 8-byte aligned base.
  tt.func public @reject_unaligned_base(%arg0: !tt.ptr<bf16> {tt.divisibility = 8 : i32}, %x: tensor<32xi32>, %a: tensor<32x64xbf16>, %b: tensor<64x64xbf16>, %c: tensor<32x64xf32>, %m: i32 {tt.divisibility = 16 : i32}, %s: i64 {tt.divisibility = 16 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%m, %m], [%s, %c1_i64] : <bf16>, <1x64xbf16>
    %d = tt.dot %a, %b, %c : tensor<32x64xbf16> * tensor<64x64xbf16> -> tensor<32x64xf32>
    %t = arith.truncf %d : tensor<32x64xf32> to tensor<32x64xbf16>
    tt.descriptor_scatter %desc[%x, %c0_i32], %t : !tt.tensordesc<1x64xbf16>, tensor<32xi32>, i32, tensor<32x64xbf16>
    tt.return
  }
  // COM: A source reached from the dot through arithmetic is rejected.
  tt.func public @reject_not_dot_addf(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %x: tensor<32xi32>, %a: tensor<32x64xbf16>, %b: tensor<64x64xbf16>, %c: tensor<32x64xf32>, %m: i32 {tt.divisibility = 16 : i32}, %s: i64 {tt.divisibility = 16 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%m, %m], [%s, %c1_i64] : <bf16>, <1x64xbf16>
    %d = tt.dot %a, %b, %c : tensor<32x64xbf16> * tensor<64x64xbf16> -> tensor<32x64xf32>
    %e = arith.addf %d, %c : tensor<32x64xf32>
    %t = arith.truncf %e : tensor<32x64xf32> to tensor<32x64xbf16>
    tt.descriptor_scatter %desc[%x, %c0_i32], %t : !tt.tensordesc<1x64xbf16>, tensor<32xi32>, i32, tensor<32x64xbf16>
    tt.return
  }
  // COM: So is an scf.for result that the loop yields through arithmetic.
  tt.func public @reject_loop_addf(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %x: tensor<32xi32>, %a: tensor<32x64xbf16>, %b: tensor<64x64xbf16>, %c: tensor<32x64xf32>, %m: i32 {tt.divisibility = 16 : i32}, %s: i64 {tt.divisibility = 16 : i32}, %n: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%m, %m], [%s, %c1_i64] : <bf16>, <1x64xbf16>
    %acc = scf.for %i = %c0_i32 to %n step %c1_i32 iter_args(%k = %c) -> (tensor<32x64xf32>) : i32 {
      %d = tt.dot %a, %b, %k : tensor<32x64xbf16> * tensor<64x64xbf16> -> tensor<32x64xf32>
      %e = arith.addf %d, %c : tensor<32x64xf32>
      scf.yield %e : tensor<32x64xf32>
    }
    %t = arith.truncf %acc : tensor<32x64xf32> to tensor<32x64xbf16>
    tt.descriptor_scatter %desc[%x, %c0_i32], %t : !tt.tensordesc<1x64xbf16>, tensor<32xi32>, i32, tensor<32x64xbf16>
    tt.return
  }
  // COM: 8-bit elements are rejected; the verifier refuses scatters wider than 32 bits.
  tt.func public @reject_width(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %x: tensor<32xi32>, %a: tensor<32x64xi8>, %b: tensor<64x64xi8>, %c: tensor<32x64xi32>, %m: i32 {tt.divisibility = 16 : i32}, %s: i64 {tt.divisibility = 16 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%m, %m], [%s, %c1_i64] : <i8>, <1x64xi8>
    %d = tt.dot %a, %b, %c : tensor<32x64xi8> * tensor<64x64xi8> -> tensor<32x64xi32>
    %t = arith.trunci %d : tensor<32x64xi32> to tensor<32x64xi8>
    tt.descriptor_scatter %desc[%x, %c0_i32], %t : !tt.tensordesc<1x64xi8>, tensor<32xi32>, i32, tensor<32x64xi8>
    tt.return
  }
}

// CHECK-LABEL: @admit_aligned_y
// CHECK:       tt.descriptor_scatter
// CHECK-NOT:   tt.store
// OFF-LABEL:   @admit_aligned_y
// OFF:         tt.store {{.*}} : tensor<32x64x!tt.ptr<f32>>
// CHECK-LABEL: @reject_unaligned_y
// CHECK-NOT:   tt.descriptor_scatter
// CHECK:       tt.store
// CHECK-LABEL: @reject_unaligned_base
// CHECK-NOT:   tt.descriptor_scatter
// CHECK:       tt.store
// CHECK-LABEL: @reject_not_dot_addf
// CHECK-NOT:   tt.descriptor_scatter
// CHECK:       tt.store
// CHECK-LABEL: @reject_loop_addf
// CHECK-NOT:   tt.descriptor_scatter
// CHECK:       tt.store
// CHECK-LABEL: @reject_width
// CHECK-NOT:   tt.descriptor_scatter
// CHECK:       tt.store

// -----

// COM: Makers that reach one op are kept or expanded together.
module {
  // COM: Two makers merged by an scf.if stay native for an admitted scatter ...
  tt.func public @merged_makers(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %x: tensor<32xi32>, %a: tensor<32x64xbf16>, %b: tensor<64x64xbf16>, %c: tensor<32x64xf32>, %m: i32 {tt.divisibility = 16 : i32}, %s: i64 {tt.divisibility = 16 : i32}, %cond: i1) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %d0 = tt.make_tensor_descriptor %arg0, [%m, %m], [%s, %c1_i64] : <bf16>, <1x64xbf16>
    %d1 = tt.make_tensor_descriptor %arg1, [%m, %m], [%s, %c1_i64] : <bf16>, <1x64xbf16>
    %desc = scf.if %cond -> (!tt.tensordesc<1x64xbf16>) {
      scf.yield %d0 : !tt.tensordesc<1x64xbf16>
    } else {
      scf.yield %d1 : !tt.tensordesc<1x64xbf16>
    }
    %d = tt.dot %a, %b, %c : tensor<32x64xbf16> * tensor<64x64xbf16> -> tensor<32x64xf32>
    %t = arith.truncf %d : tensor<32x64xf32> to tensor<32x64xbf16>
    tt.descriptor_scatter %desc[%x, %c0_i32], %t : !tt.tensordesc<1x64xbf16>, tensor<32xi32>, i32, tensor<32x64xbf16>
    tt.return
  }
  // COM: ... unless the second one also feeds a rejected gather.
  tt.func public @merged_makers_gather(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %x: tensor<32xi32>, %a: tensor<32x64xbf16>, %b: tensor<64x64xbf16>, %c: tensor<32x64xf32>, %m: i32 {tt.divisibility = 16 : i32}, %s: i64 {tt.divisibility = 16 : i32}, %cond: i1) -> tensor<32x64xbf16> {
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    %c1_i64 = arith.constant 1 : i64
    %d0 = tt.make_tensor_descriptor %arg0, [%m, %m], [%s, %c1_i64] : <bf16>, <1x64xbf16>
    %d1 = tt.make_tensor_descriptor %arg1, [%m, %m], [%s, %c1_i64] : <bf16>, <1x64xbf16>
    %g = tt.descriptor_gather %d1[%x, %c16_i32] : (!tt.tensordesc<1x64xbf16>, tensor<32xi32>, i32) -> tensor<32x64xbf16>
    %desc = scf.if %cond -> (!tt.tensordesc<1x64xbf16>) {
      scf.yield %d0 : !tt.tensordesc<1x64xbf16>
    } else {
      scf.yield %d1 : !tt.tensordesc<1x64xbf16>
    }
    %d = tt.dot %a, %b, %c : tensor<32x64xbf16> * tensor<64x64xbf16> -> tensor<32x64xf32>
    %t = arith.truncf %d : tensor<32x64xf32> to tensor<32x64xbf16>
    tt.descriptor_scatter %desc[%x, %c0_i32], %t : !tt.tensordesc<1x64xbf16>, tensor<32xi32>, i32, tensor<32x64xbf16>
    tt.return %g : tensor<32x64xbf16>
  }
}

// CHECK-LABEL: @merged_makers(
// CHECK-COUNT-2: tt.make_tensor_descriptor
// CHECK:       tt.descriptor_scatter
// CHECK-NOT:   tt.store
// CHECK-LABEL: @merged_makers_gather
// CHECK-NOT:   tt.make_tensor_descriptor
// CHECK:       tt.load
// CHECK:       tt.store

// -----

// COM: A descriptor carried through scf.for to an admitted scatter keeps its
// COM: descriptor_load native too; OFF expands both.
module {
  tt.func public @shared_load_region(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %x: tensor<32xi32>, %a: tensor<32x64xbf16>, %b: tensor<64x64xbf16>, %c: tensor<32x64xf32>, %m: i32 {tt.divisibility = 16 : i32}, %s: i64 {tt.divisibility = 16 : i32}, %n: i32) -> tensor<1x64xbf16> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%m, %m], [%s, %c1_i64] : <bf16>, <1x64xbf16>
    %r:2 = scf.for %i = %c0_i32 to %n step %c1_i32 iter_args(%dd = %desc, %k = %c) -> (!tt.tensordesc<1x64xbf16>, tensor<32x64xf32>) : i32 {
      %d = tt.dot %a, %b, %k : tensor<32x64xbf16> * tensor<64x64xbf16> -> tensor<32x64xf32>
      scf.yield %dd, %d : !tt.tensordesc<1x64xbf16>, tensor<32x64xf32>
    }
    %t = arith.truncf %r#1 : tensor<32x64xf32> to tensor<32x64xbf16>
    tt.descriptor_scatter %r#0[%x, %c0_i32], %t : !tt.tensordesc<1x64xbf16>, tensor<32xi32>, i32, tensor<32x64xbf16>
    %l = tt.descriptor_load %r#0[%c0_i32, %c0_i32] : !tt.tensordesc<1x64xbf16> -> tensor<1x64xbf16>
    tt.return %l : tensor<1x64xbf16>
  }
}

// CHECK-LABEL: @shared_load_region
// CHECK:       tt.descriptor_scatter
// CHECK:       tt.descriptor_load
// OFF-LABEL:   @shared_load_region
// OFF:         tt.store
// OFF:         tt.load

// -----

// COM: A descriptor_reduce expands its makers, so a scatter sharing one expands too.
module {
  tt.func public @shared_with_reduce(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %x: tensor<32xi32>, %a: tensor<32x64xbf16>, %b: tensor<64x64xbf16>, %c: tensor<32x64xf32>, %m: i32 {tt.divisibility = 16 : i32}, %s: i64 {tt.divisibility = 16 : i32}, %r: tensor<1x64xbf16>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%m, %m], [%s, %c1_i64] : <bf16>, <1x64xbf16>
    %d = tt.dot %a, %b, %c : tensor<32x64xbf16> * tensor<64x64xbf16> -> tensor<32x64xf32>
    %t = arith.truncf %d : tensor<32x64xf32> to tensor<32x64xbf16>
    tt.descriptor_scatter %desc[%x, %c0_i32], %t : !tt.tensordesc<1x64xbf16>, tensor<32xi32>, i32, tensor<32x64xbf16>
    tt.descriptor_reduce add, %desc[%c0_i32, %c0_i32], %r : !tt.tensordesc<1x64xbf16>, tensor<1x64xbf16>
    tt.return
  }
}

// CHECK-LABEL: @shared_with_reduce
// CHECK-NOT:   tt.descriptor_scatter
// CHECK:       tt.store
