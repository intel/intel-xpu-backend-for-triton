// RUN: triton-opt %s -triton-intel-remove-masks | FileCheck %s

// COM: Abort regression (#8368): a loop over a type wider than 64 bits whose
// COM: bounds do not fit in int64_t must not reach `APInt::getSExtValue` in
// COM: `collectLoopIVRange`. The bounds are computed (1 << 65, 1 << 64) because
// COM: scf.for's own verifier aborts on a constant bound that does not fit in
// COM: int64_t. The IV has no int64_t range, so the mask is kept.
tt.func public @i128_zero_trip_mask_kept(%arg0: !tt.ptr<f32>) -> tensor<32xf32> {
  %cst = arith.constant dense<0.000000e+00> : tensor<32xf32>
  %one = arith.constant 1 : i128
  %c64 = arith.constant 64 : i128
  %c65 = arith.constant 65 : i128
  %lb = arith.shli %one, %c65 : i128
  %ub = arith.shli %one, %c64 : i128
  %bound = arith.constant dense<1000> : tensor<32xi64>
  %range_i32 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %range = arith.extsi %range_i32 : tensor<32xi32> to tensor<32xi64>
  %base_ptrs = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
  %r = scf.for %iv = %lb to %ub step %one iter_args(%acc = %cst) -> (tensor<32xf32>) : i128 {
    %iv_idx = arith.index_cast %iv : i128 to index
    %iv_i64 = arith.index_cast %iv_idx : index to i64
    %iv_splat = tt.splat %iv_i64 : i64 -> tensor<32xi64>
    %offsets = arith.addi %iv_splat, %range : tensor<32xi64>
    %mask = arith.cmpi slt, %offsets, %bound : tensor<32xi64>
    %ptrs = tt.addptr %base_ptrs, %offsets : tensor<32x!tt.ptr<f32>>, tensor<32xi64>
    %load = tt.load %ptrs, %mask : tensor<32x!tt.ptr<f32>>
    %new = arith.addf %acc, %load : tensor<32xf32>
    scf.yield %new : tensor<32xf32>
  }
  tt.return %r : tensor<32xf32>
}
// CHECK-LABEL: @i128_zero_trip_mask_kept
// CHECK:         scf.for
// CHECK:           %[[MASK:.*]] = arith.cmpi slt
// CHECK-NEXT:      %[[PTRS:.*]] = tt.addptr
// CHECK-NEXT:      %[[LOAD:.*]] = tt.load %[[PTRS]], %[[MASK]] : tensor<32x!tt.ptr<f32>>
// CHECK-NEXT:      arith.addf %{{.*}}, %[[LOAD]]
