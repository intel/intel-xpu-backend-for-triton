// RUN: triton-opt %s -tritonintelgpu-materialize-block-pointer="enable-descriptor-relayout=true" | FileCheck %s
// RUN: triton-opt %s -tritonintelgpu-materialize-block-pointer | FileCheck %s
// RUN: env TRITON_INTEL_DISABLE_DESCRIPTOR_LOAD_TO_LOAD=0 triton-opt %s -tritonintelgpu-lower-to-2d-block-load -verify-diagnostics

// COM: A descriptor load with fewer than two indices (hand-written IR) is refused.
// COM: Its descriptor is a function argument, so the tag is the one written here.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @no_indices_unchanged(
  // CHECK-NOT: ttig.block_io_anchor
  tt.func public @no_indices_unchanged(%desc: !tt.tensordesc<16x256xf32>, %out_ptrs: tensor<256x!tt.ptr<f32>, #blocked1>) {
    // expected-error @+1 {{'tt.descriptor_load' op expects one index per descriptor dimension}}
    %x = tt.descriptor_load %desc[] {ttig.block_io = "row_major"} : !tt.tensordesc<16x256xf32> -> tensor<16x256xf32, #blocked>
    %0 = "tt.reduce"(%x) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %2 = arith.addf %a, %b : f32
      tt.reduce.return %2 : f32
    }) : (tensor<16x256xf32, #blocked>) -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %1 = ttg.convert_layout %0 : tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<256xf32, #blocked1>
    tt.store %out_ptrs, %1 : tensor<256x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}
