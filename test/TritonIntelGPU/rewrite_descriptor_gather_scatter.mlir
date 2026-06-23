// RUN: triton-opt %s -tritonintelgpu-rewrite-descriptor-gather-scatter | FileCheck %s

// CHECK: #[[$ATTR_0:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [8, 4], order = [1, 0]}>
// CHECK: #[[$ATTR_1:.+]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 8], warpsPerCTA = [32, 1], order = [1, 0]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [8, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 8], warpsPerCTA = [32, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32 }{

// CHECK-LABEL: @tt_gather_to_ttig_gather
tt.func @tt_gather_to_ttig_gather(%arg0: !tt.tensordesc<1x128xbf16>, %arg1: tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>, %arg2: i32) {

  // CHECK: %[[CONVERT_LAYOUT_0:.*]] = ttg.convert_layout {{.*}} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #[[$ATTR_0]]}>> -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #[[$ATTR_1]]}>>
  // CHECK: ttig.descriptor_gather {{.*}}{{\[}}%[[CONVERT_LAYOUT_0]], {{.*}}] : (!tt.tensordesc<1x128xbf16>, tensor<32xi32, #ttg.slice<{dim = 1, parent = #[[$ATTR_1]]}>>, i32) -> tensor<32x128xbf16, #[[$ATTR_1]]>
  %0 = tt.descriptor_gather %arg0[%arg1, %arg2] : (!tt.tensordesc<1x128xbf16>, tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32) -> tensor<32x128xbf16, #blocked1>

  // CHECK: %[[CONVERT_LAYOUT_1:.*]] = ttg.convert_layout {{.*}} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #[[$ATTR_0]]}>> -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #[[$ATTR_1]]}>>
  // CHECK: ttig.descriptor_scatter {{.*}}{{\[}}%[[CONVERT_LAYOUT_1]], {{.*}}], {{.*}} : !tt.tensordesc<1x128xbf16>, tensor<32xi32, #ttg.slice<{dim = 1, parent = #[[$ATTR_1]]}>>, i32, tensor<32x128xbf16, #[[$ATTR_1]]>
  tt.descriptor_scatter %arg0[%arg1, %arg2], %0 : !tt.tensordesc<1x128xbf16>, tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, tensor<32x128xbf16, #blocked1>

  tt.return

  // CHECK-NOT: tt.descriptor_gather
  // CHECK-NOT: tt.descriptor_scatter
}
}
