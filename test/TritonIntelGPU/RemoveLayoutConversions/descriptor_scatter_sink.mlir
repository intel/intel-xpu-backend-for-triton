// RUN: triton-opt %s -split-input-file -tritonintelgpu-remove-layout-conversions -tritonintelgpu-remove-layout-conversions | FileCheck %s

// COM: Issue #8868. A descriptor scatter drops the convert_layout before its source iff
// COM: getScatterBlockWriteRuns accepts the layout before it; the x offsets then take its dim-1 slice.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [2, 2], A = [16, 16], B = [16, 32], C = [16, 32]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @adopts_dpas
  // CHECK-NEXT: %[[X:.*]] = ttg.convert_layout %arg1 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #mma}>>
  // CHECK-NEXT: ttig.descriptor_scatter %arg0[%[[X]], %arg2], %arg3 : !tt.tensordesc<1x64xbf16>, tensor<32xi32, #ttg.slice<{dim = 1, parent = #mma}>>, i32, tensor<32x64xbf16, #mma>
  // CHECK-NEXT: tt.return
  tt.func @adopts_dpas(%desc: !tt.tensordesc<1x64xbf16>, %x: tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>, %y: i32, %v: tensor<32x64xbf16, #mma>) {
    %c = ttg.convert_layout %v : tensor<32x64xbf16, #mma> -> tensor<32x64xbf16, #blocked>
    ttig.descriptor_scatter %desc[%x, %y], %c : !tt.tensordesc<1x64xbf16>, tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, tensor<32x64xbf16, #blocked>
    tt.return
  }
}

// -----

// COM: With 32 lanes, DPAS lane 16 holds the next row, so the scatter keeps the convert.
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 32, warpsPerCTA = [2, 2], repCluster = [2, 2], A = [16, 16], B = [16, 32], C = [16, 32]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @keeps_tpw32
  // CHECK-NEXT: %[[C:.*]] = ttg.convert_layout %arg3 : tensor<32x64xbf16, #mma> -> tensor<32x64xbf16, #blocked>
  // CHECK-NEXT: ttig.descriptor_scatter %arg0[%arg1, %arg2], %[[C]] : {{.*}}, tensor<32x64xbf16, #blocked>
  tt.func @keeps_tpw32(%desc: !tt.tensordesc<1x64xbf16>, %x: tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>, %y: i32, %v: tensor<32x64xbf16, #mma>) {
    %c = ttg.convert_layout %v : tensor<32x64xbf16, #mma> -> tensor<32x64xbf16, #blocked>
    ttig.descriptor_scatter %desc[%x, %y], %c : !tt.tensordesc<1x64xbf16>, tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, tensor<32x64xbf16, #blocked>
    tt.return
  }
}

// -----

// COM: The helper decides for blocked layouts too: lanes holding column pairs keep the convert.
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @no_dpas
  // CHECK-NEXT: %[[C:.*]] = ttg.convert_layout %arg3 : tensor<32x64xbf16, #blocked1> -> tensor<32x64xbf16, #blocked>
  // CHECK-NEXT: ttig.descriptor_scatter %arg0[%arg1, %arg2], %[[C]] : {{.*}}, tensor<32x64xbf16, #blocked>
  tt.func @no_dpas(%desc: !tt.tensordesc<1x64xbf16>, %x: tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>, %y: i32, %v: tensor<32x64xbf16, #blocked1>) {
    %c = ttg.convert_layout %v : tensor<32x64xbf16, #blocked1> -> tensor<32x64xbf16, #blocked>
    ttig.descriptor_scatter %desc[%x, %y], %c : !tt.tensordesc<1x64xbf16>, tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, tensor<32x64xbf16, #blocked>
    tt.return
  }
}
