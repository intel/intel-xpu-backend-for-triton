// RUN: triton-opt %s -split-input-file -tritonintelgpu-insert-global-barrier | FileCheck %s

// A global store followed by a load from the same kernel argument at a
// different layout is a cross-thread read-after-write: a barrier must be
// inserted before the load.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @global_exchange
  tt.func public @global_exchange(%arg0: !tt.ptr<f16>) {
    %cst = arith.constant dense<0.0> : tensor<2x256xf16, #blocked1>
    %mask = arith.constant dense<true> : tensor<2x256xi1, #blocked1>
    %maskld = arith.constant dense<true> : tensor<2x256xi1, #blocked>
    %other = arith.constant dense<0.0> : tensor<2x256xf16, #blocked>
    %off = arith.constant dense<0> : tensor<2x256xi32, #blocked1>
    %off2 = arith.constant dense<0> : tensor<2x256xi32, #blocked>
    %sp = tt.splat %arg0 : !tt.ptr<f16> -> tensor<2x256x!tt.ptr<f16>, #blocked1>
    %pst = tt.addptr %sp, %off : tensor<2x256x!tt.ptr<f16>, #blocked1>, tensor<2x256xi32, #blocked1>
    // CHECK: tt.store
    tt.store %pst, %cst, %mask : tensor<2x256x!tt.ptr<f16>, #blocked1>
    %sp2 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<2x256x!tt.ptr<f16>, #blocked>
    %pld = tt.addptr %sp2, %off2 : tensor<2x256x!tt.ptr<f16>, #blocked>, tensor<2x256xi32, #blocked>
    // CHECK: ttg.barrier all
    // CHECK-NEXT: tt.load
    %v = tt.load %pld, %maskld, %other : tensor<2x256x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

// Same layout on both sides: each thread reads back what it wrote, no
// cross-thread exchange, so no barrier is inserted.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [2, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @same_layout_roundtrip
  // CHECK-NOT: ttg.barrier
  tt.func public @same_layout_roundtrip(%arg0: !tt.ptr<f16>) {
    %cst = arith.constant dense<0.0> : tensor<2x256xf16, #blocked>
    %mask = arith.constant dense<true> : tensor<2x256xi1, #blocked>
    %other = arith.constant dense<0.0> : tensor<2x256xf16, #blocked>
    %off = arith.constant dense<0> : tensor<2x256xi32, #blocked>
    %sp = tt.splat %arg0 : !tt.ptr<f16> -> tensor<2x256x!tt.ptr<f16>, #blocked>
    %p = tt.addptr %sp, %off : tensor<2x256x!tt.ptr<f16>, #blocked>, tensor<2x256xi32, #blocked>
    tt.store %p, %cst, %mask : tensor<2x256x!tt.ptr<f16>, #blocked>
    %v = tt.load %p, %mask, %other : tensor<2x256x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

// A load with no prior store to the same argument (reading an input) must not
// be fenced.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @load_before_store
  // CHECK-NOT: ttg.barrier
  tt.func public @load_before_store(%arg0: !tt.ptr<f16>) {
    %cst = arith.constant dense<0.0> : tensor<2x256xf16, #blocked1>
    %mask = arith.constant dense<true> : tensor<2x256xi1, #blocked1>
    %maskld = arith.constant dense<true> : tensor<2x256xi1, #blocked>
    %other = arith.constant dense<0.0> : tensor<2x256xf16, #blocked>
    %off = arith.constant dense<0> : tensor<2x256xi32, #blocked1>
    %off2 = arith.constant dense<0> : tensor<2x256xi32, #blocked>
    %sp2 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<2x256x!tt.ptr<f16>, #blocked>
    %pld = tt.addptr %sp2, %off2 : tensor<2x256x!tt.ptr<f16>, #blocked>, tensor<2x256xi32, #blocked>
    %v = tt.load %pld, %maskld, %other : tensor<2x256x!tt.ptr<f16>, #blocked>
    %sp = tt.splat %arg0 : !tt.ptr<f16> -> tensor<2x256x!tt.ptr<f16>, #blocked1>
    %pst = tt.addptr %sp, %off : tensor<2x256x!tt.ptr<f16>, #blocked1>, tensor<2x256xi32, #blocked1>
    tt.store %pst, %cst, %mask : tensor<2x256x!tt.ptr<f16>, #blocked1>
    tt.return
  }
}

// -----

// A store/load pair nested in an scf.if region must NOT be fenced: a
// work-group barrier under divergent control flow is undefined behavior.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @exchange_in_divergent_region
  // CHECK-NOT: ttg.barrier
  tt.func public @exchange_in_divergent_region(%arg0: !tt.ptr<f16>, %cond: i1) {
    %cst = arith.constant dense<0.0> : tensor<2x256xf16, #blocked1>
    %mask = arith.constant dense<true> : tensor<2x256xi1, #blocked1>
    %maskld = arith.constant dense<true> : tensor<2x256xi1, #blocked>
    %other = arith.constant dense<0.0> : tensor<2x256xf16, #blocked>
    %off = arith.constant dense<0> : tensor<2x256xi32, #blocked1>
    %off2 = arith.constant dense<0> : tensor<2x256xi32, #blocked>
    scf.if %cond {
      %sp = tt.splat %arg0 : !tt.ptr<f16> -> tensor<2x256x!tt.ptr<f16>, #blocked1>
      %pst = tt.addptr %sp, %off : tensor<2x256x!tt.ptr<f16>, #blocked1>, tensor<2x256xi32, #blocked1>
      tt.store %pst, %cst, %mask : tensor<2x256x!tt.ptr<f16>, #blocked1>
      %sp2 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<2x256x!tt.ptr<f16>, #blocked>
      %pld = tt.addptr %sp2, %off2 : tensor<2x256x!tt.ptr<f16>, #blocked>, tensor<2x256xi32, #blocked>
      %v = tt.load %pld, %maskld, %other : tensor<2x256x!tt.ptr<f16>, #blocked>
    }
    tt.return
  }
}
