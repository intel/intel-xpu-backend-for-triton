// RUN: triton-opt %s -split-input-file -tritonintelgpu-remove-layout-conversions -tritonintelgpu-remove-layout-conversions | FileCheck %s

// COM: Issue #8666. Descriptor gathers follow the plain-load anchor rule: one with at least
// COM: numWarps * threadsPerWarp elements (128 here) keeps its layout, a smaller one is rematerialized.

// CHECK-DAG: #[[$A:blocked[0-9]*]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK-DAG: #[[$B:blocked[0-9]*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK-DAG: #[[$C:blocked[0-9]*]] = #ttg.blocked<{sizePerThread = [1, 1, 4]
#A = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#B = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#C = #ttg.blocked<{sizePerThread = [1, 1, 4], threadsPerWarp = [16, 2, 1], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#xa = #ttg.slice<{dim = 1, parent = #A}>
#sc = #ttg.slice<{dim = 2, parent = #C}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @anchored
  // CHECK: ttig.descriptor_gather {{.*}} -> tensor<16x8xf32, #[[$A]]>
  // CHECK-NEXT: math.exp {{.*}} : tensor<16x8xf32, #[[$A]]>
  // CHECK: tt.descriptor_gather {{.*}} -> tensor<16x8xf32, #[[$A]]>
  // CHECK-NEXT: math.exp {{.*}} : tensor<16x8xf32, #[[$A]]>
  tt.func @anchored(%d: !tt.tensordesc<1x8xf32>, %y: i32) -> (tensor<16x8xf32, #B>, tensor<16x8xf32, #B>) {
    %x0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #xa>
    %g0 = ttig.descriptor_gather %d[%x0, %y] : (!tt.tensordesc<1x8xf32>, tensor<16xi32, #xa>, i32) -> tensor<16x8xf32, #A>
    %c0 = ttg.convert_layout %g0 : tensor<16x8xf32, #A> -> tensor<16x8xf32, #B>
    %e0 = math.exp %c0 : tensor<16x8xf32, #B>
    %x1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #xa>
    %g1 = tt.descriptor_gather %d[%x1, %y] : (!tt.tensordesc<1x8xf32>, tensor<16xi32, #xa>, i32) -> tensor<16x8xf32, #A>
    %c1 = ttg.convert_layout %g1 : tensor<16x8xf32, #A> -> tensor<16x8xf32, #B>
    %e1 = math.exp %c1 : tensor<16x8xf32, #B>
    tt.return %e0, %e1 : tensor<16x8xf32, #B>, tensor<16x8xf32, #B>
  }
  // CHECK-LABEL: @cheap
  // CHECK-NOT: ttg.convert_layout
  // CHECK: ttig.descriptor_gather {{.*}} -> tensor<8x8xf32, #[[$B]]>
  // CHECK-NOT: ttg.convert_layout
  // CHECK: tt.descriptor_gather {{.*}} -> tensor<8x8xf32, #[[$B]]>
  // CHECK-NOT: ttg.convert_layout
  tt.func @cheap(%d: !tt.tensordesc<1x8xf32>, %y: i32) -> (tensor<8x8xf32, #B>, tensor<8x8xf32, #B>) {
    %x0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #xa>
    %g0 = ttig.descriptor_gather %d[%x0, %y] : (!tt.tensordesc<1x8xf32>, tensor<8xi32, #xa>, i32) -> tensor<8x8xf32, #A>
    %c0 = ttg.convert_layout %g0 : tensor<8x8xf32, #A> -> tensor<8x8xf32, #B>
    %e0 = math.exp %c0 : tensor<8x8xf32, #B>
    %x1 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #xa>
    %g1 = tt.descriptor_gather %d[%x1, %y] : (!tt.tensordesc<1x8xf32>, tensor<8xi32, #xa>, i32) -> tensor<8x8xf32, #A>
    %c1 = ttg.convert_layout %g1 : tensor<8x8xf32, #A> -> tensor<8x8xf32, #B>
    %e1 = math.exp %c1 : tensor<8x8xf32, #B>
    tt.return %e0, %e1 : tensor<8x8xf32, #B>, tensor<8x8xf32, #B>
  }
  // COM: A single-use expensive gather is relabeled through a shape-op chain, as tt.load is.
  // CHECK-LABEL: @shape_chain
  // CHECK-NOT: ttg.convert_layout
  // CHECK: ttig.descriptor_gather {{.*}} -> tensor<16x8xf32, #ttg.slice<{dim = 2, parent = #[[$C]]}>>
  // CHECK-NOT: ttg.convert_layout
  // CHECK: tt.descriptor_gather {{.*}} -> tensor<16x8xf32, #ttg.slice<{dim = 2, parent = #[[$C]]}>>
  // CHECK-NOT: ttg.convert_layout
  tt.func @shape_chain(%d: !tt.tensordesc<1x8xf32>, %y: i32) -> (tensor<16x8x4xf32, #C>, tensor<16x8x4xf32, #C>) {
    %x0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #xa>
    %g0 = ttig.descriptor_gather %d[%x0, %y] : (!tt.tensordesc<1x8xf32>, tensor<16xi32, #xa>, i32) -> tensor<16x8xf32, #A>
    %c0 = ttg.convert_layout %g0 : tensor<16x8xf32, #A> -> tensor<16x8xf32, #sc>
    %e0 = tt.expand_dims %c0 {axis = 2 : i32} : tensor<16x8xf32, #sc> -> tensor<16x8x1xf32, #C>
    %b0 = tt.broadcast %e0 : tensor<16x8x1xf32, #C> -> tensor<16x8x4xf32, #C>
    %x1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #xa>
    %g1 = tt.descriptor_gather %d[%x1, %y] : (!tt.tensordesc<1x8xf32>, tensor<16xi32, #xa>, i32) -> tensor<16x8xf32, #A>
    %c1 = ttg.convert_layout %g1 : tensor<16x8xf32, #A> -> tensor<16x8xf32, #sc>
    %e1 = tt.expand_dims %c1 {axis = 2 : i32} : tensor<16x8xf32, #sc> -> tensor<16x8x1xf32, #C>
    %b1 = tt.broadcast %e1 : tensor<16x8x1xf32, #C> -> tensor<16x8x4xf32, #C>
    tt.return %b0, %b1 : tensor<16x8x4xf32, #C>, tensor<16x8x4xf32, #C>
  }
}

// -----

// COM: A cheap gather used outside the slice is charged the duplicated-load cost, so RLC
// COM: keeps the 128-unit sub-group shuffle instead of loading the gather twice (4096).
#P = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 1], warpsPerCTA = [1, 16], order = [0, 1]}>
#Q = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 16], order = [0, 1]}>
#xp = #ttg.slice<{dim = 1, parent = #P}>
#xq = #ttg.slice<{dim = 1, parent = #Q}>
module attributes {"ttg.num-warps" = 16 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @cheap_two_uses
  // CHECK-COUNT-2: descriptor_gather
  // CHECK-NOT: descriptor_gather
  tt.func @cheap_two_uses(%d: !tt.tensordesc<1x8xf32>, %y: i32) -> (tensor<16x8xf32, #P>, tensor<16xf32, #xq>, tensor<16x8xf32, #P>, tensor<16xf32, #xq>) {
    %x0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #xp>
    %g0 = ttig.descriptor_gather %d[%x0, %y] : (!tt.tensordesc<1x8xf32>, tensor<16xi32, #xp>, i32) -> tensor<16x8xf32, #P>
    %r0 = "tt.reduce"(%g0) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) : (tensor<16x8xf32, #P>) -> tensor<16xf32, #xp>
    %c0 = ttg.convert_layout %r0 : tensor<16xf32, #xp> -> tensor<16xf32, #xq>
    %x1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #xp>
    %g1 = tt.descriptor_gather %d[%x1, %y] : (!tt.tensordesc<1x8xf32>, tensor<16xi32, #xp>, i32) -> tensor<16x8xf32, #P>
    %r1 = "tt.reduce"(%g1) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) : (tensor<16x8xf32, #P>) -> tensor<16xf32, #xp>
    %c1 = ttg.convert_layout %r1 : tensor<16xf32, #xp> -> tensor<16xf32, #xq>
    tt.return %g0, %c0, %g1, %c1 : tensor<16x8xf32, #P>, tensor<16xf32, #xq>, tensor<16x8xf32, #P>, tensor<16xf32, #xq>
  }
}
