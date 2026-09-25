// RUN: triton-opt %s -split-input-file -triton-intel-remove-masks | FileCheck %s

module {
  // COM: A bounds mask on a loop-carried offset that the loop *updates* must be
  // COM: kept. `getFinalValue` resolves the `%offs` iteration argument to its
  // COM: init `tt.make_range(0, 32)`, describing the first iteration only; the
  // COM: loop adds 4096 each time, so the offsets are 0, 4096, 8192, 12288 and
  // COM: the mask is false for every lane of the last three iterations. Judging
  // COM: it against the init value declares it always-true and `dropMask`
  // COM: deletes it (along with `other`), leaving the load reading out of
  // COM: bounds and returning what it found instead of `other` (#8060).
  tt.func public @loop_carried_offset_mask(
      %src: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %dst: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32xf16>
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c128_i32 = arith.constant 128 : i32
    %cst_0 = arith.constant dense<4096> : tensor<32xi32>

    %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %res = scf.for %iv = %c0_i32 to %c128_i32 step %c32_i32
        iter_args(%offs = %range) -> (tensor<32xi32>) : i32 {
      %iv_splat = tt.splat %iv : i32 -> tensor<32xi32>
      %idx = arith.addi %iv_splat, %offs : tensor<32xi32>
      %mask = arith.cmpi slt, %idx, %cst_0 : tensor<32xi32>

      %src_iv = tt.addptr %src, %iv : !tt.ptr<f16>, i32
      %src_splat = tt.splat %src_iv : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>>
      %src_ptr = tt.addptr %src_splat, %offs : tensor<32x!tt.ptr<f16>>, tensor<32xi32>
      %loaded = tt.load %src_ptr, %mask, %cst : tensor<32x!tt.ptr<f16>>

      %dst_iv = tt.addptr %dst, %iv : !tt.ptr<f16>, i32
      %dst_splat = tt.splat %dst_iv : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>>
      %dst_ptr = tt.addptr %dst_splat, %range : tensor<32x!tt.ptr<f16>>, tensor<32xi32>
      tt.store %dst_ptr, %loaded : tensor<32x!tt.ptr<f16>>

      %next_offs = arith.addi %offs, %cst_0 : tensor<32xi32>
      scf.yield %next_offs : tensor<32xi32>
    }
    tt.return
  }

  // CHECK-LABEL: @loop_carried_offset_mask
  // CHECK: scf.for
  // CHECK:   %[[MASK:.*]] = arith.cmpi slt
  // CHECK:   tt.load {{.*}}, %[[MASK]], {{.*}} : tensor<32x!tt.ptr<f16>>
  // CHECK: }
}

// -----

module {
  // COM: Same shape, but the compared-against *bound* is the loop-carried value
  // COM: the loop updates. Resolving it to its init `dense<4096>` proves the
  // COM: mask always-true for the first iteration only.
  tt.func public @loop_carried_bound_mask(
      %src: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32xf16>
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c128_i32 = arith.constant 128 : i32
    %cst_ub = arith.constant dense<4096> : tensor<32xi32>
    %cst_dec = arith.constant dense<4096> : tensor<32xi32>

    %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %src_splat = tt.splat %src : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>>
    %res = scf.for %iv = %c0_i32 to %c128_i32 step %c32_i32
        iter_args(%bound = %cst_ub) -> (tensor<32xi32>) : i32 {
      %iv_splat = tt.splat %iv : i32 -> tensor<32xi32>
      %idx = arith.addi %iv_splat, %range : tensor<32xi32>
      %mask = arith.cmpi slt, %idx, %bound : tensor<32xi32>

      %loaded = tt.load %src_splat, %mask, %cst : tensor<32x!tt.ptr<f16>>

      %next_bound = arith.subi %bound, %cst_dec : tensor<32xi32>
      scf.yield %next_bound : tensor<32xi32>
    }
    tt.return
  }

  // CHECK-LABEL: @loop_carried_bound_mask
  // CHECK: scf.for
  // CHECK:   %[[MASK:.*]] = arith.cmpi slt
  // CHECK:   tt.load {{.*}}, %[[MASK]], {{.*}} : tensor<32x!tt.ptr<f16>>
  // CHECK: }
}
