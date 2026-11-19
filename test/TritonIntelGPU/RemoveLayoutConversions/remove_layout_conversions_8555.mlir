// RUN: triton-opt %s -split-input-file -tritonintelgpu-remove-layout-conversions -tritonintelgpu-remove-layout-conversions | FileCheck %s

// The ext/broadcast hoist must keep a convert anchored for a descriptor store;
// otherwise the next RLC run re-layouts the store (issue #8555).

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#coalesced = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>

// Multi-op hoist; the dst is the canonical coalesced store layout (#7104 anchor).
// CHECK: #[[$S1:.+]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK-LABEL: @multi_broadcast_store_anchor
// CHECK: %[[CVT:.+]] = ttg.convert_layout %{{.+}} : tensor<64x128xf16, #{{.+}}> -> tensor<64x128xf16, #[[$S1]]>
// CHECK-NEXT: tt.descriptor_store %{{.+}}, %[[CVT]] : !tt.tensordesc<64x128xf16>, tensor<64x128xf16, #[[$S1]]>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  tt.func @multi_broadcast_store_anchor(%desc: !tt.tensordesc<64x128xf16>, %row: tensor<64x1xf16, #blocked>, %col: tensor<1x128xf16, #blocked>) {
    %c0 = arith.constant 0 : i32
    %r = tt.broadcast %row : tensor<64x1xf16, #blocked> -> tensor<64x128xf16, #blocked>
    %c = tt.broadcast %col : tensor<1x128xf16, #blocked> -> tensor<64x128xf16, #blocked>
    %m = arith.mulf %r, %c : tensor<64x128xf16, #blocked>
    %st = ttg.convert_layout %m : tensor<64x128xf16, #blocked> -> tensor<64x128xf16, #coalesced>
    tt.descriptor_store %desc[%c0, %c0], %st : !tt.tensordesc<64x128xf16>, tensor<64x128xf16, #coalesced>
    tt.return
  }
}

// -----

#rows = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#coalesced = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>

// Single-op hoist; the convert changes the lane-fast dim (#7093 anchor).
// CHECK: #[[$S2:.+]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK-LABEL: @broadcast_store_lane_transpose_anchor
// CHECK: tt.descriptor_store {{.+}} : !tt.tensordesc<64x128xf16>, tensor<64x128xf16, #[[$S2]]>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  tt.func @broadcast_store_lane_transpose_anchor(%desc: !tt.tensordesc<64x128xf16>, %col: tensor<1x128xf16, #rows>) {
    %c0 = arith.constant 0 : i32
    %b = tt.broadcast %col : tensor<1x128xf16, #rows> -> tensor<64x128xf16, #rows>
    %st = ttg.convert_layout %b : tensor<64x128xf16, #rows> -> tensor<64x128xf16, #coalesced>
    tt.descriptor_store %desc[%c0, %c0], %st : !tt.tensordesc<64x128xf16>, tensor<64x128xf16, #coalesced>
    tt.return
  }
}
