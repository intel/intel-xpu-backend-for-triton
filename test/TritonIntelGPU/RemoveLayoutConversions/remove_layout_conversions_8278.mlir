// RUN: triton-opt %s -split-input-file -tritonintelgpu-remove-layout-conversions 2>&1 | FileCheck %s

// COM: https://github.com/intel/intel-xpu-backend-for-triton/issues/8278
// COM: getConvertBackwardSlice tested canUseResultEncoding's result as a bool, discarding the
// COM: fixed-operand list it returns. A tt.broadcast could then absorb a convert on the
// COM: assumption that its source operand (%s) keeps its current layout, while another path
// COM: through the slice (tt.join/tt.reshape, sharing %s) moved that same operand to a
// COM: different layout. The broadcast was left with mismatched source/result layouts, which
// COM: the verifier rejects. Port of upstream triton-lang/triton#11760's
// COM: @broadcast_absorption_source_layout_conflict, adapted to 16 lanes.

#src = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [16, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#join = #ttg.blocked<{sizePerThread = [1, 2, 2], threadsPerWarp = [16, 1, 1], warpsPerCTA = [1, 1, 1], order = [2, 1, 0]}>
#out = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [16, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-DAG: #[[$SRC_ENC:blocked[0-9]*]] = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [16, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
  // CHECK-DAG: #[[$OUT_ENC:blocked[0-9]*]] = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [16, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
  // CHECK-LABEL: @broadcast_absorption_source_layout_conflict
  // CHECK: %[[SRC:.*]] = tt.expand_dims {{.*}} -> tensor<1x2xi32, #[[$SRC_ENC]]>
  // CHECK-NEXT: %[[B:.*]] = tt.broadcast %[[SRC]] : tensor<1x2xi32, #[[$SRC_ENC]]> -> tensor<2x2xi32, #[[$OUT_ENC]]>
  // CHECK-NEXT: %[[J:.*]] = tt.join %[[SRC]], %[[SRC]] : tensor<1x2xi32, #[[$SRC_ENC]]>
  // CHECK-NEXT: %[[R:.*]] = tt.reshape %[[J]]
  // CHECK-NEXT: %[[A:.*]] = arith.addi %[[B]], %[[R]] : tensor<2x2xi32, #[[$OUT_ENC]]>
  // CHECK-NEXT: %[[C:.*]] = ttg.convert_layout %[[A]] : tensor<2x2xi32, #[[$OUT_ENC]]> -> tensor<2x2xi32, #[[$SRC_ENC]]>
  // CHECK-NEXT: tt.return %[[C]]
  tt.func @broadcast_absorption_source_layout_conflict() -> tensor<2x2xi32, #src> {
    %r = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #ttg.slice<{dim = 0, parent = #src}>>
    %s = tt.expand_dims %r {axis = 0 : i32} : tensor<2xi32, #ttg.slice<{dim = 0, parent = #src}>> -> tensor<1x2xi32, #src>
    %b = tt.broadcast %s : tensor<1x2xi32, #src> -> tensor<2x2xi32, #out>
    %j = tt.join %s, %s : tensor<1x2xi32, #src> -> tensor<1x2x2xi32, #join>
    %v = tt.reshape %j : tensor<1x2x2xi32, #join> -> tensor<2x2xi32, #out>
    %a = arith.addi %b, %v : tensor<2x2xi32, #out>
    %c = ttg.convert_layout %a : tensor<2x2xi32, #out> -> tensor<2x2xi32, #src>
    tt.return %c : tensor<2x2xi32, #src>
  }
}

// -----

// COM: Same bug, reached through canUseResultEncoding's tt.reshape arm instead of tt.broadcast:
// COM: absorbing the reshape's result layout must keep its source layout, which conflicts with
// COM: rematerializing that same source through the transpose branch. Port of upstream
// COM: triton-lang/triton#11758's @reshape_absorption_source_layout_conflict, adapted to 16 lanes.

#src = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#transposed = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
#dst = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 8], warpsPerCTA = [1, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-DAG: #[[$R_SRC_ENC:blocked[0-9]*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
  // CHECK-DAG: #[[$R_T_ENC:blocked[0-9]*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
  // CHECK-DAG: #[[$R_DST_ENC:blocked[0-9]*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 8], warpsPerCTA = [1, 1], order = [0, 1]}>
  // CHECK-LABEL: @reshape_absorption_source_layout_conflict
  // CHECK: %[[SRC:.*]] = tt.broadcast {{.*}} -> tensor<2x4xi32, #[[$R_SRC_ENC]]>
  // CHECK-NEXT: %[[R:.*]] = tt.reshape %[[SRC]] allow_reorder : tensor<2x4xi32, #[[$R_SRC_ENC]]> -> tensor<4x2xi32, #[[$R_T_ENC]]>
  // CHECK-NEXT: %[[T:.*]] = tt.trans %[[SRC]] {{.*}} : tensor<2x4xi32, #[[$R_SRC_ENC]]> -> tensor<4x2xi32, #[[$R_T_ENC]]>
  // CHECK-NEXT: %[[A:.*]] = arith.addi %[[T]], %[[R]] : tensor<4x2xi32, #[[$R_T_ENC]]>
  // CHECK-NEXT: %[[C:.*]] = ttg.convert_layout %[[A]] : tensor<4x2xi32, #[[$R_T_ENC]]> -> tensor<4x2xi32, #[[$R_DST_ENC]]>
  // CHECK-NEXT: tt.return %[[C]]
  tt.func @reshape_absorption_source_layout_conflict() -> tensor<4x2xi32, #dst> {
    %r = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #src}>>
    %e = tt.expand_dims %r {axis = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #src}>> -> tensor<1x4xi32, #src>
    %v = tt.broadcast %e : tensor<1x4xi32, #src> -> tensor<2x4xi32, #src>
    %s = tt.reshape %v allow_reorder : tensor<2x4xi32, #src> -> tensor<4x2xi32, #transposed>
    %t = tt.trans %v {order = array<i32: 1, 0>} : tensor<2x4xi32, #src> -> tensor<4x2xi32, #transposed>
    %a = arith.addi %t, %s : tensor<4x2xi32, #transposed>
    %c = ttg.convert_layout %a : tensor<4x2xi32, #transposed> -> tensor<4x2xi32, #dst>
    tt.return %c : tensor<4x2xi32, #dst>
  }
}
