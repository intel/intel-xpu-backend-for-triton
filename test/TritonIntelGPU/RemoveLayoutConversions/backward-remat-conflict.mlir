// RUN: env -u TRITON_INTEL_REMOVELAYOUTCONVERSION_SUPPORT_FOR_LOOP triton-opt %s -split-input-file -tritonintelgpu-remove-layout-conversions 2>&1 | FileCheck %s --enable-var-scope
// RUN: env -u TRITON_INTEL_REMOVELAYOUTCONVERSION_SUPPORT_FOR_LOOP triton-opt %s -split-input-file -tritonintelgpu-remove-layout-conversions='max-backward-remat-iterations=10' 2>&1 | FileCheck %s --enable-var-scope
// RUN: env TRITON_INTEL_REMOVELAYOUTCONVERSION_SUPPORT_FOR_LOOP=1 triton-opt %s -split-input-file -tritonintelgpu-remove-layout-conversions 2>&1 | FileCheck %s --enable-var-scope
// RUN: env TRITON_INTEL_REMOVELAYOUTCONVERSION_SUPPORT_FOR_LOOP=1 triton-opt %s -split-input-file -tritonintelgpu-remove-layout-conversions='max-backward-remat-iterations=10' 2>&1 | FileCheck %s --enable-var-scope

// COM: Test for backward rematerialization conflict detection.
// COM: Exercises the fix from upstream Triton PR #9953:
// COM:   "Handle conflicts with existing type during backward remat"
// COM:
// COM: When getConvertBackwardSlice encounters a value that already has the
// COM: desired encoding, it records the layout (for conflict detection) but
// COM: does NOT add the value to the remat slice. Previously, the layout was
// COM: NOT recorded, so if the same value was reached via a different path
// COM: with a conflicting encoding, the conflict went undetected.

// COM: Scenario: Two convert_layout ops on the same value (%add) target
// COM: different encodings (#blocked_b and #blocked_c). Backward remat
// COM: from each convert reaches the same defining chain, but the two
// COM: target encodings conflict. Both convert_layout ops must be preserved.

#blocked_a = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_b = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_c = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [0, 1]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @backward_remat_existing_layout_conflict
  tt.func @backward_remat_existing_layout_conflict(%arg0: !tt.ptr<f32>) -> (tensor<16x16xf32, #blocked_b>, tensor<16x16xf32, #blocked_c>) {
    %cst = arith.constant dense<1.0> : tensor<16x16xf32, #blocked_a>
    %splat = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>, #blocked_a>
    %load = tt.load %splat : tensor<16x16x!tt.ptr<f32>, #blocked_a>
    %add = arith.addf %load, %cst : tensor<16x16xf32, #blocked_a>
    // COM: Both converts target the same source value (%add) but with
    // COM: incompatible target encodings. The pass must preserve at least
    // COM: one convert_layout because they cannot both be rematerialized
    // COM: to different layouts for the same defining chain.
    %convert_to_b = ttg.convert_layout %add : tensor<16x16xf32, #blocked_a> -> tensor<16x16xf32, #blocked_b>
    %convert_to_c = ttg.convert_layout %add : tensor<16x16xf32, #blocked_a> -> tensor<16x16xf32, #blocked_c>
    // CHECK: [[CONVERT_TO_B:%.*]] = ttg.convert_layout
    // CHECK: [[CONVERT_TO_C:%.*]] = ttg.convert_layout
    // CHECK: tt.return [[CONVERT_TO_B]], [[CONVERT_TO_C]]
    tt.return %convert_to_b, %convert_to_c : tensor<16x16xf32, #blocked_b>, tensor<16x16xf32, #blocked_c>
  }
}

// -----

// COM: Variant: the source value already has one of the target encodings.
// COM: This directly tests the fix: when backward remat from convert_to_c
// COM: reaches %val (which already has #blocked_b encoding), the layout is
// COM: recorded as #blocked_b. Later, backward remat from convert_to_b
// COM: would find %val already has #blocked_b (matching), but the layout
// COM: map ensures no conflict since both want #blocked_b. Meanwhile,
// COM: convert_to_c must be preserved because #blocked_c differs from
// COM: #blocked_b.

#blocked_b = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_c = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [0, 1]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @existing_encoding_matches_one_target
  tt.func @existing_encoding_matches_one_target(%arg0: !tt.ptr<f32>) -> (tensor<16x16xf32, #blocked_b>, tensor<16x16xf32, #blocked_c>) {
    // COM: %val is produced with #blocked_b encoding. convert_to_b is
    // COM: identity and should be eliminated. convert_to_c cannot be
    // COM: rematerialized because the source already has #blocked_b,
    // COM: which conflicts with the desired #blocked_c.
    %splat = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>, #blocked_b>
    %val = tt.load %splat : tensor<16x16x!tt.ptr<f32>, #blocked_b>
    %convert_to_b = ttg.convert_layout %val : tensor<16x16xf32, #blocked_b> -> tensor<16x16xf32, #blocked_b>
    %convert_to_c = ttg.convert_layout %val : tensor<16x16xf32, #blocked_b> -> tensor<16x16xf32, #blocked_c>
    // COM: The identity convert (b->b) is eliminated, but the convert to
    // COM: #blocked_c must be preserved since the load anchors #blocked_b.
    // CHECK: %[[LOAD:.*]] = tt.load
    // CHECK: %[[CVT:.*]] = ttg.convert_layout %[[LOAD]]
    // CHECK: tt.return %[[LOAD]], %[[CVT]]
    tt.return %convert_to_b, %convert_to_c : tensor<16x16xf32, #blocked_b>, tensor<16x16xf32, #blocked_c>
  }
}

// -----

// COM: Test that conflict detection works through elementwise ops.
// COM: Both paths share %load as a common ancestor, but need it in
// COM: different layouts. The backward slices from the two converts
// COM: will conflict on %load, so the converts must be preserved.

#blocked_a = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_b = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_c = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [0, 1]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @conflict_through_elementwise
  tt.func @conflict_through_elementwise(%arg0: !tt.ptr<f32>) -> (tensor<16x16xf32, #blocked_b>, tensor<16x16xf32, #blocked_c>) {
    %splat = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>, #blocked_a>
    %load = tt.load %splat : tensor<16x16x!tt.ptr<f32>, #blocked_a>
    %neg = arith.negf %load : tensor<16x16xf32, #blocked_a>
    %abs = math.absf %load : tensor<16x16xf32, #blocked_a>
    %convert_neg = ttg.convert_layout %neg : tensor<16x16xf32, #blocked_a> -> tensor<16x16xf32, #blocked_b>
    %convert_abs = ttg.convert_layout %abs : tensor<16x16xf32, #blocked_a> -> tensor<16x16xf32, #blocked_c>
    // CHECK-COUNT-2: ttg.convert_layout
    // CHECK: tt.return
    tt.return %convert_neg, %convert_abs : tensor<16x16xf32, #blocked_b>, tensor<16x16xf32, #blocked_c>
  }
}

// -----

// COM: Port of upstream Triton PR #10646 for Intel issue #8193.
// COM: Explicitly select XPU for this port and declare the 32-thread warp used
// COM: by the unchanged upstream blocked layouts.
// COM: These output checks pass before the bookkeeping fix; the local
// COM: rewriteSlice assertion in assertions-enabled builds detects stale mappings.
// COM: Both broadcasts are rematerialized in the loop; only the loop-result
// COM: convert remains.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "xpu"} {
  // CHECK-LABEL: tt.func @backward_remat_reuse(
  // CHECK-NOT:     ttg.convert_layout
  // CHECK:         %[[LOOP:.*]]:2 = scf.for
  // CHECK-COUNT-2:   tt.broadcast
  // CHECK-NOT:       ttg.convert_layout
  // CHECK:           scf.yield
  // CHECK-NEXT:    }
  // CHECK-NEXT:    %[[CVT:.*]] = ttg.convert_layout %[[LOOP]]#0
  // CHECK-NEXT:    tt.return %[[CVT]]
  tt.func @backward_remat_reuse(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: i32) -> tensor<64x2xf32, #blocked> {
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<64x2xf32, #blocked>
    %cst_1 = arith.constant dense<1> : tensor<64x2xi32, #blocked>
    %0 = tt.splat %arg1 : i32 -> tensor<64x1xi32, #blocked1>
    %1:2 = scf.for %arg3 = %arg2 to %arg1 step %arg1 iter_args(%arg4 = %cst_0, %arg5 = %cst_0) -> (tensor<64x2xf32, #blocked>, tensor<64x2xf32, #blocked>)  : i32 {
      %2 = tt.broadcast %0 : tensor<64x1xi32, #blocked1> -> tensor<64x2xi32, #blocked1>
      %3 = ttg.convert_layout %2 : tensor<64x2xi32, #blocked1> -> tensor<64x2xi32, #blocked>
      %4 = arith.cmpi slt, %3, %cst_1 : tensor<64x2xi32, #blocked>
      %5 = tt.broadcast %0 : tensor<64x1xi32, #blocked1> -> tensor<64x2xi32, #blocked1>
      %6 = ttg.convert_layout %5 : tensor<64x2xi32, #blocked1> -> tensor<64x2xi32, #blocked>
      %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x2x!tt.ptr<f32>, #blocked>
      %8 = tt.addptr %7, %6 : tensor<64x2x!tt.ptr<f32>, #blocked>, tensor<64x2xi32, #blocked>
      %9 = ttg.convert_layout %8 : tensor<64x2x!tt.ptr<f32>, #blocked> -> tensor<64x2x!tt.ptr<f32>, #blocked1>
      %10 = tt.load %9 : tensor<64x2x!tt.ptr<f32>, #blocked1>
      %11 = ttg.convert_layout %10 : tensor<64x2xf32, #blocked1> -> tensor<64x2xf32, #blocked>
      %12 = arith.select %4, %cst_0, %arg5 : tensor<64x2xi1, #blocked>, tensor<64x2xf32, #blocked>
      %13 = arith.mulf %11, %arg5 : tensor<64x2xf32, #blocked>
      %14 = arith.select %4, %13, %arg4 : tensor<64x2xi1, #blocked>, tensor<64x2xf32, #blocked>
      scf.yield %14, %12 : tensor<64x2xf32, #blocked>, tensor<64x2xf32, #blocked>
    }
    tt.return %1#0 : tensor<64x2xf32, #blocked>
  }
}
