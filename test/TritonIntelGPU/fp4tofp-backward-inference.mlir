// Verify that backward encoding inference through fp4_to_fp succeeds when
// the nibble-selector basis is not at register position 0 (opsPerChannel=8 /
// fp4KPack=2 case, triggering 128x256x128 spill before this fix).
//
// The convert_layout from #blocked7 to #linear1 should be eliminated by
// RemoveLayoutConversions once inference can propagate #linear1 backward
// through fp4_to_fp, moving the layout conversion to the cheaper i8 input.

// RUN: triton-opt %s -split-input-file -tritonintelgpu-remove-layout-conversions | FileCheck %s

#blocked6 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 16], order = [1, 0]}>
#blocked7 = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 16], order = [1, 0]}>
#linear1 = #ttg.linear<{register = [[32, 0], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], lane = [[64, 0], [0, 1], [0, 2], [0, 4]], warp = [[0, 8], [0, 16], [0, 32], [0, 64], [0, 128]], block = []}>

// CHECK-LABEL: @fp4_tofp_no_cvt_on_bf16_result
// After the fix, the convert_layout on the bf16 tensor is eliminated.
// A cheaper convert_layout on the i8 input may appear instead.
// CHECK-NOT: ttg.convert_layout {{.*}} : tensor<128x256xbf16, #blocked7>
module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @fp4_tofp_no_cvt_on_bf16_result(
      %src: tensor<64x256xi8, #blocked6>,
      %acc: tensor<128x256xbf16, #linear1>) -> tensor<128x256xbf16, #linear1> {
    %fp4 = ttg.fp4_to_fp %src {axis = 0 : i32}
        : tensor<64x256xi8, #blocked6> -> tensor<128x256xbf16, #blocked7>
    %cvt = ttg.convert_layout %fp4
        : tensor<128x256xbf16, #blocked7> -> tensor<128x256xbf16, #linear1>
    %result = arith.addf %cvt, %acc : tensor<128x256xbf16, #linear1>
    tt.return %result : tensor<128x256xbf16, #linear1>
  }
}
