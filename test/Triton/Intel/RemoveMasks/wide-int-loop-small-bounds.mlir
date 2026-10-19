// RUN: triton-opt %s -triton-intel-remove-masks | FileCheck %s

// COM: Policy (#8368): collectLoopIVRange returns no range for an IV wider than
// COM: 64 bits, even when its bounds are small, so RemoveMasks keeps the mask.
// COM: `tt.splat` rejects i128, so the IV reaches the mask through
// COM: `arith.index_cast` (peeled by `getFinalValue`).
// COM: Control: with an i64 IV the mask is provably true (IV in [0, 480], lanes
// COM: 0..31, bound 1000), so it is dropped: the load feeding the addf is
// COM: unmasked.
// CHECK-LABEL: @i64_iv_mask_dropped
tt.func public @i64_iv_mask_dropped(%arg0: !tt.ptr<f32>) -> tensor<32xf32> {
  %cst = arith.constant dense<0.000000e+00> : tensor<32xf32>
  %c0 = arith.constant 0 : i64
  %c32 = arith.constant 32 : i64
  %c512 = arith.constant 512 : i64
  %bound = arith.constant dense<1000> : tensor<32xi64>
  %range_i32 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %range = arith.extsi %range_i32 : tensor<32xi32> to tensor<32xi64>
  %base_ptrs = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
  %r = scf.for %iv = %c0 to %c512 step %c32 iter_args(%acc = %cst) -> (tensor<32xf32>) : i64 {
    %iv_idx = arith.index_cast %iv : i64 to index
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
// CHECK:         scf.for
// CHECK:           %[[PTRS:.*]] = tt.addptr
// CHECK-NEXT:      %[[LOAD:.*]] = tt.load %[[PTRS]] : tensor<32x!tt.ptr<f32>>
// CHECK:           arith.addf %{{.*}}, %[[LOAD]]

// COM: Same IR with an i128 IV and bounds: collectLoopIVRange gives no range
// COM: whatever the bounds, so the mask stays.
// CHECK-LABEL: @i128_iv_mask_kept
tt.func public @i128_iv_mask_kept(%arg0: !tt.ptr<f32>) -> tensor<32xf32> {
  %cst = arith.constant dense<0.000000e+00> : tensor<32xf32>
  %c0 = arith.constant 0 : i128
  %c32 = arith.constant 32 : i128
  %c512 = arith.constant 512 : i128
  %bound = arith.constant dense<1000> : tensor<32xi64>
  %range_i32 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %range = arith.extsi %range_i32 : tensor<32xi32> to tensor<32xi64>
  %base_ptrs = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
  %r = scf.for %iv = %c0 to %c512 step %c32 iter_args(%acc = %cst) -> (tensor<32xf32>) : i128 {
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
// CHECK:         scf.for
// CHECK:           %[[MASK:.*]] = arith.cmpi slt
// CHECK-NEXT:      %[[PTRS:.*]] = tt.addptr
// CHECK-NEXT:      %[[LOAD:.*]] = tt.load %[[PTRS]], %[[MASK]] : tensor<32x!tt.ptr<f32>>
// CHECK-NEXT:      arith.addf %{{.*}}, %[[LOAD]]

// COM: i65 is the smallest wide type: no range either, so the mask stays.
// CHECK-LABEL: @i65_iv_mask_kept
tt.func public @i65_iv_mask_kept(%arg0: !tt.ptr<f32>) -> tensor<32xf32> {
  %cst = arith.constant dense<0.000000e+00> : tensor<32xf32>
  %c0 = arith.constant 0 : i65
  %c32 = arith.constant 32 : i65
  %c512 = arith.constant 512 : i65
  %bound = arith.constant dense<1000> : tensor<32xi64>
  %range_i32 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %range = arith.extsi %range_i32 : tensor<32xi32> to tensor<32xi64>
  %base_ptrs = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
  %r = scf.for %iv = %c0 to %c512 step %c32 iter_args(%acc = %cst) -> (tensor<32xf32>) : i65 {
    %iv_idx = arith.index_cast %iv : i65 to index
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
// CHECK:         scf.for
// CHECK:           %[[MASK:.*]] = arith.cmpi slt
// CHECK-NEXT:      %[[PTRS:.*]] = tt.addptr
// CHECK-NEXT:      %[[LOAD:.*]] = tt.load %[[PTRS]], %[[MASK]] : tensor<32x!tt.ptr<f32>>
// CHECK-NEXT:      arith.addf %{{.*}}, %[[LOAD]]
