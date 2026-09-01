// RUN: triton-opt %s -split-input-file -tritonintelgpu-materialize-block-pointer -tritonintelgpu-remove-layout-conversions | FileCheck %s

// COM: Regression test for how RemoveLayoutConversions treats the ConvertLayoutOps
// COM: that reshape1DStridedLoad inserts: one per reshaped operand (pointer, and
// COM: mask when the load is masked) plus one on the result.
// COM:
// COM: The one on the *result* must survive.  The load encoding matches HW
// COM: delivery order and must be anchored; without the anchor fix in
// COM: isExpensiveLoadOrStore that ConvertLayoutOp is eliminated and the load
// COM: encoding is changed, producing incorrect results at runtime.
// COM:
// COM: The ones on the *pointer* and *mask* are expected to fold away.  RLC
// COM: back-propagates the load encoding through the elementwise addptr chain and
// COM: into the mask constant, so the reshapes feed the load directly and no data
// COM: movement remains — which the CHECK-NOT below pins.

#blocked1d = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // COM: The ConvertLayoutOp must survive *between* the load and the reshape back
  // COM: to 1D, and that reshape must keep the result encoding it was created with
  // COM: rather than having the anchored load encoding propagated into it.  Binding
  // COM: the encodings makes this assert that property: an earlier version of these
  // COM: CHECKs only required *some* ttg.convert_layout to appear near the second
  // COM: reshape, which was also satisfied by a convert that layout propagation had
  // COM: inserted after rewriting the reshape's encoding.
  // COM:
  // COM: Note the pointer reshape's operand encoding is *not* the function's 1D
  // COM: encoding: RemoveLayoutConversions back-propagates the load encoding through
  // COM: the elementwise addptr chain, so it becomes a #ttg.linear layout.  Only the
  // COM: final reshape's result is required to match the 1D return type.
  // CHECK-LABEL: tt.func @test_convert_layout_survives
  // CHECK:      [[PTRS:%.*]] = tt.reshape %{{.*}} : tensor<1024x!tt.ptr<f16>, #{{[a-z0-9_]+}}> -> tensor<32x32x!tt.ptr<f16>, #[[BHW:[a-z0-9_]+]]>
  // CHECK-NOT:  ttg.convert_layout
  // CHECK:      [[LOAD:%.*]] = tt.load [[PTRS]], %{{.*}} {ttig.block_io = "row_major", ttig.block_io_stride = 96 : i64} : tensor<32x32x!tt.ptr<f16>, #[[BHW]]>
  // CHECK:      [[CVT:%.*]] = ttg.convert_layout [[LOAD]] : tensor<32x32xf16, #[[BHW]]> -> tensor<32x32xf16, #[[BNAT:[a-z0-9_]+]]>
  // CHECK:      [[RES:%.*]] = tt.reshape [[CVT]] efficient_layout : tensor<32x32xf16, #[[BNAT]]> -> tensor<1024xf16, #[[B1D:[a-z0-9_]+]]>
  // CHECK-NEXT: tt.return [[RES]] : tensor<1024xf16, #[[B1D]]>
  tt.func @test_convert_layout_survives(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<1024xf16, #blocked1d> {
    %idx = tt.make_range {start = 0 : i32, end = 1024 : i32} : tensor<1024xi32, #blocked1d>
    %c32 = arith.constant dense<32> : tensor<1024xi32, #blocked1d>
    %c96 = arith.constant dense<96> : tensor<1024xi32, #blocked1d>
    %rem = arith.remui %idx, %c32 : tensor<1024xi32, #blocked1d>
    %div = arith.divui %idx, %c32 : tensor<1024xi32, #blocked1d>
    %mul = arith.muli %div, %c96 : tensor<1024xi32, #blocked1d>
    %off = arith.addi %rem, %mul : tensor<1024xi32, #blocked1d>
    %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked1d>
    %ptrs = tt.addptr %base, %off : tensor<1024x!tt.ptr<f16>, #blocked1d>, tensor<1024xi32, #blocked1d>
    %mask = arith.constant dense<true> : tensor<1024xi1, #blocked1d>
    %result = tt.load %ptrs, %mask : tensor<1024x!tt.ptr<f16>, #blocked1d>
    tt.return %result : tensor<1024xf16, #blocked1d>
  }
}

// -----

// COM: Same regression test for the store side (H > 1).  reshape1DStridedStore
// COM: converts the value from the natural 2D encoding into the HW delivery
// COM: encoding; if RemoveLayoutConversions folds that ConvertLayoutOp away, the
// COM: store receives the natural encoding again and writes to the wrong
// COM: positions — exactly the silent corruption of #6531/#6634.  The anchor in
// COM: isExpensiveLoadOrStore keys on ttig.block_io_stride, not on op kind, so
// COM: this holds for stores as well as loads.

#blocked1d = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @test_store_convert_layout_survives
  // COM: Two reshapes, then the value-side ConvertLayout (anchored by ttig.block_io_stride)
  // COM: must survive RemoveLayoutConversions. Binding [[CVT]] and verifying it is used as
  // COM: the store's value operand ensures the correct (storeEnc) value reaches the store.
  // CHECK: tt.reshape %{{.*}}
  // CHECK: tt.reshape %{{.*}}
  // CHECK: [[CVT:%[0-9]+]] = ttg.convert_layout
  // CHECK: tt.store %{{.*}}, [[CVT]] {ttig.block_io = "row_major", ttig.block_io_stride = 96 : i64}
  tt.func @test_store_convert_layout_survives(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: tensor<1024xf16, #blocked1d>) {
    %idx = tt.make_range {start = 0 : i32, end = 1024 : i32} : tensor<1024xi32, #blocked1d>
    %c32 = arith.constant dense<32> : tensor<1024xi32, #blocked1d>
    %c96 = arith.constant dense<96> : tensor<1024xi32, #blocked1d>
    %rem = arith.remui %idx, %c32 : tensor<1024xi32, #blocked1d>
    %div = arith.divui %idx, %c32 : tensor<1024xi32, #blocked1d>
    %mul = arith.muli %div, %c96 : tensor<1024xi32, #blocked1d>
    %off = arith.addi %rem, %mul : tensor<1024xi32, #blocked1d>
    %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked1d>
    %ptrs = tt.addptr %base, %off : tensor<1024x!tt.ptr<f16>, #blocked1d>, tensor<1024xi32, #blocked1d>
    tt.store %ptrs, %arg1 : tensor<1024x!tt.ptr<f16>, #blocked1d>
    tt.return
  }
}
