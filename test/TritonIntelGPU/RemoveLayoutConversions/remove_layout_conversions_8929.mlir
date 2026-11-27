// RUN: triton-opt %s -split-input-file -tritonintelgpu-remove-layout-conversions | FileCheck %s

// COM: https://github.com/intel/intel-xpu-backend-for-triton/issues/8929
// COM: rewriteSlice may reuse only the existing conversions the slice walk accepted.

// CHECK: #[[$DST:.+]] = #ttg.blocked<{sizePerThread = [8]
#src = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [4], order = [0]}>
#dst = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [16], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // COM: %c1 dominates only %e2's use of %x2: the walk accepts it there first, and
  // COM: the later rejection at %e1 must still win.
  // CHECK-LABEL: @accept_then_reject
  // CHECK: %[[S:.+]] = tt.splat %arg0 : f32 -> tensor<32xf32, #[[$DST]]>
  // CHECK: %[[A:.+]] = math.exp %[[S]] :
  // CHECK: %[[X:.+]] = math.exp %[[A]] :
  // CHECK: %[[E1:.+]] = arith.extf %[[X]] :
  // CHECK-NEXT: %[[C1:.+]] = ttg.convert_layout
  // CHECK-NEXT: %[[E2:.+]] = arith.extf %[[X]] :
  // CHECK-NEXT: %[[SUM:.+]] = arith.addf %[[E1]], %[[E2]] :
  // CHECK-NEXT: tt.return %[[C1]], %[[SUM]] :
  tt.func public @accept_then_reject(%s: f32) -> (tensor<32xf32, #dst>, tensor<32xf64, #dst>) {
    %x0 = tt.splat %s : f32 -> tensor<32xf32, #src>
    %x1 = math.exp %x0 : tensor<32xf32, #src>
    %x2 = math.exp %x1 : tensor<32xf32, #src>
    %e1 = arith.extf %x2 : tensor<32xf32, #src> to tensor<32xf64, #src>
    %c1 = ttg.convert_layout %x2 : tensor<32xf32, #src> -> tensor<32xf32, #dst>
    %e2 = arith.extf %x2 : tensor<32xf32, #src> to tensor<32xf64, #src>
    %sum = arith.addf %e1, %e2 : tensor<32xf64, #src>
    %c2 = ttg.convert_layout %sum : tensor<32xf64, #src> -> tensor<32xf64, #dst>
    tt.return %c1, %c2 : tensor<32xf32, #dst>, tensor<32xf64, #dst>
  }
  // COM: Same, with the rejection visited first: the later accept must not override it.
  // CHECK-LABEL: @reject_then_accept
  // CHECK: arith.extf %[[X:.+]] :
  // CHECK: arith.extf %[[X]] :
  tt.func public @reject_then_accept(%s: f32) -> (tensor<32xf32, #dst>, tensor<32xf64, #dst>) {
    %x0 = tt.splat %s : f32 -> tensor<32xf32, #src>
    %x1 = math.exp %x0 : tensor<32xf32, #src>
    %x2 = math.exp %x1 : tensor<32xf32, #src>
    %e1 = arith.extf %x2 : tensor<32xf32, #src> to tensor<32xf64, #src>
    %c1 = ttg.convert_layout %x2 : tensor<32xf32, #src> -> tensor<32xf32, #dst>
    %e2 = arith.extf %x2 : tensor<32xf32, #src> to tensor<32xf64, #src>
    %sum = arith.addf %e2, %e1 : tensor<32xf64, #src>
    %c2 = ttg.convert_layout %sum : tensor<32xf64, #src> -> tensor<32xf64, #dst>
    tt.return %c1, %c2 : tensor<32xf32, #dst>, tensor<32xf64, #dst>
  }
}

// -----

#src = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [4], order = [0]}>
#dst = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [16], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // COM: The hoist's failed inner walk for %e2 rejects %k; that rejection must not
  // COM: leak into the inner walk for %e1, which reuses %k.
  // CHECK-LABEL: @failed_inner_walk_reject
  // CHECK: ttg.convert_layout
  // CHECK: %[[K:.+]] = ttg.convert_layout
  // CHECK-NEXT: arith.extf %[[K]] :
  tt.func public @failed_inner_walk_reject(%arg0: tensor<32xf32, #src>) -> (tensor<32xf32, #dst>, tensor<32xf64, #dst>) {
    %v = arith.negf %arg0 : tensor<32xf32, #src>
    %e2 = arith.extf %v : tensor<32xf32, #src> to tensor<32xf64, #src>
    %k = ttg.convert_layout %v : tensor<32xf32, #src> -> tensor<32xf32, #dst>
    %e1 = arith.extf %v : tensor<32xf32, #src> to tensor<32xf64, #src>
    %w = arith.addf %e1, %e2 : tensor<32xf64, #src>
    %c = ttg.convert_layout %w : tensor<32xf64, #src> -> tensor<32xf64, #dst>
    tt.return %k, %c : tensor<32xf32, #dst>, tensor<32xf64, #dst>
  }
}

// -----

#X = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#Y = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // COM: After accepting the free convert %f for %arg0, the walk must stop there
  // COM: instead of re-walking %f's source, which would fail the slice.
  // CHECK-LABEL: @reuse_free_convert
  // CHECK: %[[F:.+]] = ttg.convert_layout %arg0
  // CHECK-NEXT: %[[E:.+]] = math.exp %[[F]] :
  // CHECK-NEXT: tt.return %[[F]], %[[E]] :
  tt.func public @reuse_free_convert(%arg0: tensor<8x32xf32, #X>) -> (tensor<8x32xf32, #Y>, tensor<8x32xf32, #Y>) {
    %f = ttg.convert_layout %arg0 : tensor<8x32xf32, #X> -> tensor<8x32xf32, #Y>
    %y = math.exp %arg0 : tensor<8x32xf32, #X>
    %c = ttg.convert_layout %y : tensor<8x32xf32, #X> -> tensor<8x32xf32, #Y>
    tt.return %f, %c : tensor<8x32xf32, #Y>, tensor<8x32xf32, #Y>
  }
}

// -----

#E = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#Et = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [4, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func private @producer(f32) -> tensor<16x16xf32, #Et>
  // COM: The slice needs %v in #E, reused through %k, and %k itself in #Et, so the
  // COM: walk must fail and keep %c in either visit order.
  // CHECK-LABEL: @reused_value_seen_first
  // CHECK: %[[W:.+]] = arith.addf
  // CHECK-NEXT: ttg.convert_layout %[[W]] :
  tt.func public @reused_value_seen_first(%s: f32) -> tensor<16x16xf32, #E> {
    %v = tt.call @producer(%s) : (f32) -> tensor<16x16xf32, #Et>
    %k = ttg.convert_layout %v : tensor<16x16xf32, #Et> -> tensor<16x16xf32, #E>
    %t = tt.trans %k {order = array<i32: 1, 0>} : tensor<16x16xf32, #E> -> tensor<16x16xf32, #Et>
    %n = arith.negf %v : tensor<16x16xf32, #Et>
    %w = arith.addf %n, %t : tensor<16x16xf32, #Et>
    %c = ttg.convert_layout %w : tensor<16x16xf32, #Et> -> tensor<16x16xf32, #E>
    tt.return %c : tensor<16x16xf32, #E>
  }
  // CHECK-LABEL: @reuse_accepted_first
  // CHECK: %[[W:.+]] = arith.addf
  // CHECK-NEXT: ttg.convert_layout %[[W]] :
  tt.func public @reuse_accepted_first(%s: f32) -> tensor<16x16xf32, #E> {
    %v = tt.call @producer(%s) : (f32) -> tensor<16x16xf32, #Et>
    %k = ttg.convert_layout %v : tensor<16x16xf32, #Et> -> tensor<16x16xf32, #E>
    %t = tt.trans %k {order = array<i32: 1, 0>} : tensor<16x16xf32, #E> -> tensor<16x16xf32, #Et>
    %n = arith.negf %v : tensor<16x16xf32, #Et>
    %w = arith.addf %t, %n : tensor<16x16xf32, #Et>
    %c = ttg.convert_layout %w : tensor<16x16xf32, #Et> -> tensor<16x16xf32, #E>
    tt.return %c : tensor<16x16xf32, #E>
  }
}
