// RUN: triton-opt %s -split-input-file -tritonintelgpu-materialize-block-pointer -tritonintelgpu-remove-layout-conversions | FileCheck %s

// COM: Regression test: verify that RemoveLayoutConversions does NOT remove
// COM: the ConvertLayoutOp inserted by reshape1DStridedLoad.  The load encoding
// COM: matches HW delivery order and must be anchored; without the anchor fix
// COM: in isExpensiveLoadOrStore the ConvertLayoutOp is eliminated and the load
// COM: encoding is changed, producing incorrect results at runtime.

#blocked1d = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @test_convert_layout_survives
  // CHECK: tt.reshape %{{.*}} allow_reorder efficient_layout
  // CHECK: tt.load %{{.*}} {ttig.block_io = "row_major", ttig.block_io_stride = 96 : i64}
  // CHECK: tt.reshape %{{.*}} efficient_layout
  // CHECK: ttg.convert_layout
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
