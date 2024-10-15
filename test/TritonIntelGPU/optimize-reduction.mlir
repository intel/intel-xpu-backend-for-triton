// RUN: triton-opt %s --split-input-file -tritonintelgpu-optimize-reduction-locality -canonicalize | FileCheck %s

// Test reduction in a single warp (16x16->16).

#mma = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1]}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, triton_gpu.target = "xpu", "triton_gpu.threads-per-warp" = 16 : i32, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block} {

// CHECK-DAG: #[[DPAS:.+]] = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1], A = [8, 8], B = [8, 16], C = [8, 16]}>
// CHECK-DAG: #[[BLOCKED0:.+]] = #triton_gpu.blocked<{sizePerThread = [8, 1, 1, 1, 1], threadsPerWarp = [1, 16, 1, 1, 1], warpsPerCTA = [1, 1, 1, 1, 1], order = [4, 0, 1, 2, 3]}>
// CHECK-DAG: #[[BLOCKED1:.+]] = #triton_gpu.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [1, 1], order = [1, 0]}>

// CHECK:         tt.func @test_single(
// CHECK-SAME:                         %[[VAL_0:.*]]: tensor<16x16xf32, #[[DPAS]]>) -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[DPAS]]}>> {
// CHECK:           %[[VAL_1:.*]] = tt.reshape %[[VAL_0]] {allow_reorder = true} : tensor<16x16xf32, #[[DPAS]]> -> tensor<16x16x1x1x1xf32, #[[BLOCKED0]]>
// CHECK:           %[[VAL_2:.*]] = "tt.reduce"(%[[VAL_1]]) <{axis = 4 : i32}> ({
// CHECK:           ^bb0(%[[VAL_3:.*]]: f32, %[[VAL_4:.*]]: f32):
// CHECK:             %[[VAL_5:.*]] = arith.addf %[[VAL_3]], %[[VAL_4]] : f32
// CHECK:             tt.reduce.return %[[VAL_5]] : f32
// CHECK:           }) : (tensor<16x16x1x1x1xf32, #[[BLOCKED0]]>) -> tensor<16x16x1x1xf32, #triton_gpu.slice<{dim = 4, parent = #[[BLOCKED0]]}>>
// CHECK:           %[[VAL_6:.*]] = "tt.reduce"(%[[VAL_2]]) <{axis = 2 : i32}> ({
// CHECK:           ^bb0(%[[VAL_7:.*]]: f32, %[[VAL_8:.*]]: f32):
// CHECK:             %[[VAL_9:.*]] = arith.addf %[[VAL_7]], %[[VAL_8]] : f32
// CHECK:             tt.reduce.return %[[VAL_9]] : f32
// CHECK:           }) : (tensor<16x16x1x1xf32, #triton_gpu.slice<{dim = 4, parent = #[[BLOCKED0]]}>>) -> tensor<16x16x1xf32, #triton_gpu.slice<{dim = 2, parent = #triton_gpu.slice<{dim = 4, parent = #[[BLOCKED0]]}>}>>
// CHECK:           %[[VAL_10:.*]] = tt.reshape %[[VAL_6]] {allow_reorder = true} : tensor<16x16x1xf32, #triton_gpu.slice<{dim = 2, parent = #triton_gpu.slice<{dim = 4, parent = #[[BLOCKED0]]}>}>> -> tensor<16x16xf32, #[[BLOCKED1]]>
// CHECK:           %[[VAL_11:.*]] = "tt.reduce"(%[[VAL_10]]) <{axis = 1 : i32}> ({
// CHECK:           ^bb0(%[[VAL_12:.*]]: f32, %[[VAL_13:.*]]: f32):
// CHECK:             %[[VAL_14:.*]] = arith.addf %[[VAL_12]], %[[VAL_13]] : f32
// CHECK:             tt.reduce.return %[[VAL_14]] : f32
// CHECK:           }) : (tensor<16x16xf32, #[[BLOCKED1]]>) -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[BLOCKED1]]}>>
// CHECK:           %[[VAL_15:.*]] = triton_gpu.convert_layout %[[VAL_11]] : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[BLOCKED1]]}>> -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[DPAS]]}>>
// CHECK:           tt.return %[[VAL_15]] : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[DPAS]]}>>
// CHECK:         }
  tt.func @test_single(%arg0: tensor<16x16xf32, #mma>) -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>> {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      tt.reduce.return %1 : f32
    }) : (tensor<16x16xf32, #mma>) -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    tt.return %0 : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
  }
}

// -----

// Test reduction in two warps across the non-reduction dimension (32x16->32).

#mma = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [2, 1], repCluster = [1, 1]}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 2 : i32, triton_gpu.target = "xpu", "triton_gpu.threads-per-warp" = 16 : i32, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block} {

// CHECK-DAG: #[[$ATTR_5:.+]] = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [2, 1], repCluster = [1, 1], A = [8, 8], B = [8, 16], C = [8, 16]}>
// CHECK-DAG: #[[$ATTR_3:.+]] = #triton_gpu.blocked<{sizePerThread = [8, 1, 1, 1, 1], threadsPerWarp = [1, 16, 1, 1, 1], warpsPerCTA = [2, 1, 1, 1, 1], order = [4, 0, 1, 2, 3]}>
// CHECK-DAG: #[[$ATTR_4:.+]] = #triton_gpu.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [2, 1], order = [1, 0]}>

// CHECK:         tt.func @test_single_twice(
// CHECK-SAME:                               %[[VAL_0:.*]]: tensor<32x16xf32, #[[$ATTR_5]]>) -> tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_5]]}>> {
// CHECK:           %[[VAL_1:.*]] = tt.reshape %[[VAL_0]] {allow_reorder = true} : tensor<32x16xf32, #[[$ATTR_5]]> -> tensor<32x16x1x1x1xf32, #[[$ATTR_3]]>
// CHECK:           %[[VAL_2:.*]] = "tt.reduce"(%[[VAL_1]]) <{axis = 4 : i32}> ({
// CHECK:           ^bb0(%[[VAL_3:.*]]: f32, %[[VAL_4:.*]]: f32):
// CHECK:             %[[VAL_5:.*]] = arith.addf %[[VAL_3]], %[[VAL_4]] : f32
// CHECK:             tt.reduce.return %[[VAL_5]] : f32
// CHECK:           }) : (tensor<32x16x1x1x1xf32, #[[$ATTR_3]]>) -> tensor<32x16x1x1xf32, #triton_gpu.slice<{dim = 4, parent = #[[$ATTR_3]]}>>
// CHECK:           %[[VAL_6:.*]] = "tt.reduce"(%[[VAL_2]]) <{axis = 2 : i32}> ({
// CHECK:           ^bb0(%[[VAL_7:.*]]: f32, %[[VAL_8:.*]]: f32):
// CHECK:             %[[VAL_9:.*]] = arith.addf %[[VAL_7]], %[[VAL_8]] : f32
// CHECK:             tt.reduce.return %[[VAL_9]] : f32
// CHECK:           }) : (tensor<32x16x1x1xf32, #triton_gpu.slice<{dim = 4, parent = #[[$ATTR_3]]}>>) -> tensor<32x16x1xf32, #triton_gpu.slice<{dim = 2, parent = #triton_gpu.slice<{dim = 4, parent = #[[$ATTR_3]]}>}>>
// CHECK:           %[[VAL_10:.*]] = tt.reshape %[[VAL_6]] {allow_reorder = true} : tensor<32x16x1xf32, #triton_gpu.slice<{dim = 2, parent = #triton_gpu.slice<{dim = 4, parent = #[[$ATTR_3]]}>}>> -> tensor<32x16xf32, #[[$ATTR_4]]>
// CHECK:           %[[VAL_11:.*]] = "tt.reduce"(%[[VAL_10]]) <{axis = 1 : i32}> ({
// CHECK:           ^bb0(%[[VAL_12:.*]]: f32, %[[VAL_13:.*]]: f32):
// CHECK:             %[[VAL_14:.*]] = arith.addf %[[VAL_12]], %[[VAL_13]] : f32
// CHECK:             tt.reduce.return %[[VAL_14]] : f32
// CHECK:           }) : (tensor<32x16xf32, #[[$ATTR_4]]>) -> tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_4]]}>>
// CHECK:           %[[VAL_15:.*]] = triton_gpu.convert_layout %[[VAL_11]] : tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_4]]}>> -> tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_5]]}>>
// CHECK:           tt.return %[[VAL_15]] : tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_5]]}>>
// CHECK:         }
  tt.func @test_single_twice(%arg0: tensor<32x16xf32, #mma>) -> tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>> {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      tt.reduce.return %1 : f32
    }) : (tensor<32x16xf32, #mma>) -> tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    tt.return %0 : tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
  }
}

// -----

// Test reduction in two warps across the reduction dimension (16x32->16).

#mma = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [1, 2], repCluster = [1, 1]}>

// CHECK-DAG: #[[DPAS:.+]] = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [1, 2], repCluster = [1, 1], A = [8, 8], B = [8, 16], C = [8, 16]}>
// CHECK-DAG: #[[BLOCKED0:.+]] = #triton_gpu.blocked<{sizePerThread = [8, 1, 1, 1, 1], threadsPerWarp = [1, 16, 1, 1, 1], warpsPerCTA = [1, 1, 1, 2, 1], order = [4, 0, 1, 2, 3]}>
// CHECK-DAG: #[[BLOCKED1:.+]] = #triton_gpu.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [1, 2], order = [1, 0]}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 2 : i32, triton_gpu.target = "xpu", "triton_gpu.threads-per-warp" = 16 : i32, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block} {

// CHECK:         tt.func @test_two_warps(
// CHECK-SAME:                            %[[VAL_0:.*]]: tensor<16x32xf32, #[[DPAS]]>) -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[DPAS]]}>> {
// CHECK:           %[[VAL_1:.*]] = tt.reshape %[[VAL_0]] {allow_reorder = true} : tensor<16x32xf32, #[[DPAS]]> -> tensor<16x16x1x2x1xf32, #[[BLOCKED0]]>
// CHECK:           %[[VAL_2:.*]] = "tt.reduce"(%[[VAL_1]]) <{axis = 4 : i32}> ({
// CHECK:           ^bb0(%[[VAL_3:.*]]: f32, %[[VAL_4:.*]]: f32):
// CHECK:             %[[VAL_5:.*]] = arith.addf %[[VAL_3]], %[[VAL_4]] : f32
// CHECK:             tt.reduce.return %[[VAL_5]] : f32
// CHECK:           }) : (tensor<16x16x1x2x1xf32, #[[BLOCKED0]]>) -> tensor<16x16x1x2xf32, #triton_gpu.slice<{dim = 4, parent = #[[BLOCKED0]]}>>
// CHECK:           %[[VAL_6:.*]] = "tt.reduce"(%[[VAL_2]]) <{axis = 2 : i32}> ({
// CHECK:           ^bb0(%[[VAL_7:.*]]: f32, %[[VAL_8:.*]]: f32):
// CHECK:             %[[VAL_9:.*]] = arith.addf %[[VAL_7]], %[[VAL_8]] : f32
// CHECK:             tt.reduce.return %[[VAL_9]] : f32
// CHECK:           }) : (tensor<16x16x1x2xf32, #triton_gpu.slice<{dim = 4, parent = #[[BLOCKED0]]}>>) -> tensor<16x16x2xf32, #triton_gpu.slice<{dim = 2, parent = #triton_gpu.slice<{dim = 4, parent = #[[BLOCKED0]]}>}>>
// CHECK:           %[[VAL_10:.*]] = tt.reshape %[[VAL_6]] {allow_reorder = true} : tensor<16x16x2xf32, #triton_gpu.slice<{dim = 2, parent = #triton_gpu.slice<{dim = 4, parent = #[[BLOCKED0]]}>}>> -> tensor<16x32xf32, #[[BLOCKED1]]>
// CHECK:           %[[VAL_11:.*]] = "tt.reduce"(%[[VAL_10]]) <{axis = 1 : i32}> ({
// CHECK:           ^bb0(%[[VAL_12:.*]]: f32, %[[VAL_13:.*]]: f32):
// CHECK:             %[[VAL_14:.*]] = arith.addf %[[VAL_12]], %[[VAL_13]] : f32
// CHECK:             tt.reduce.return %[[VAL_14]] : f32
// CHECK:           }) : (tensor<16x32xf32, #[[BLOCKED1]]>) -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[BLOCKED1]]}>>
// CHECK:           %[[VAL_15:.*]] = triton_gpu.convert_layout %[[VAL_11]] : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[BLOCKED1]]}>> -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[DPAS]]}>>
// CHECK:           tt.return %[[VAL_15]] : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[DPAS]]}>>
// CHECK:         }
  tt.func @test_two_warps(%arg0: tensor<16x32xf32, #mma>) -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>> {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      tt.reduce.return %1 : f32
    }) : (tensor<16x32xf32, #mma>) -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    tt.return %0 : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
  }
}

// -----

// Test reduction in two warps across both dimensions (32x32->32).

#mma = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [1, 1]}>

// CHECK-DAG: #[[DPAS:.+]] = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [1, 1], A = [8, 8], B = [8, 16], C = [8, 16]}>
// CHECK-DAG: #[[BLOCKED0:.+]] = #triton_gpu.blocked<{sizePerThread = [8, 1, 1, 1, 1], threadsPerWarp = [1, 16, 1, 1, 1], warpsPerCTA = [2, 1, 1, 2, 1], order = [4, 0, 1, 2, 3]}>
// CHECK-DAG: #[[BLOCKED1:.+]] = #triton_gpu.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [2, 2], order = [1, 0]}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "xpu", "triton_gpu.threads-per-warp" = 16 : i32, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block} {

// CHECK:         tt.func @test_two_warps_twice(
// CHECK-SAME:                                  %[[VAL_0:.*]]: tensor<32x32xf32, #[[DPAS]]>) -> tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #[[DPAS]]}>> {
// CHECK:           %[[VAL_1:.*]] = tt.reshape %[[VAL_0]] {allow_reorder = true} : tensor<32x32xf32, #[[DPAS]]> -> tensor<32x16x1x2x1xf32, #[[BLOCKED0]]>
// CHECK:           %[[VAL_2:.*]] = "tt.reduce"(%[[VAL_1]]) <{axis = 4 : i32}> ({
// CHECK:           ^bb0(%[[VAL_3:.*]]: f32, %[[VAL_4:.*]]: f32):
// CHECK:             %[[VAL_5:.*]] = arith.addf %[[VAL_3]], %[[VAL_4]] : f32
// CHECK:             tt.reduce.return %[[VAL_5]] : f32
// CHECK:           }) : (tensor<32x16x1x2x1xf32, #[[BLOCKED0]]>) -> tensor<32x16x1x2xf32, #triton_gpu.slice<{dim = 4, parent = #[[BLOCKED0]]}>>
// CHECK:           %[[VAL_6:.*]] = "tt.reduce"(%[[VAL_2]]) <{axis = 2 : i32}> ({
// CHECK:           ^bb0(%[[VAL_7:.*]]: f32, %[[VAL_8:.*]]: f32):
// CHECK:             %[[VAL_9:.*]] = arith.addf %[[VAL_7]], %[[VAL_8]] : f32
// CHECK:             tt.reduce.return %[[VAL_9]] : f32
// CHECK:           }) : (tensor<32x16x1x2xf32, #triton_gpu.slice<{dim = 4, parent = #[[BLOCKED0]]}>>) -> tensor<32x16x2xf32, #triton_gpu.slice<{dim = 2, parent = #triton_gpu.slice<{dim = 4, parent = #[[BLOCKED0]]}>}>>
// CHECK:           %[[VAL_10:.*]] = tt.reshape %[[VAL_6]] {allow_reorder = true} : tensor<32x16x2xf32, #triton_gpu.slice<{dim = 2, parent = #triton_gpu.slice<{dim = 4, parent = #[[BLOCKED0]]}>}>> -> tensor<32x32xf32, #[[BLOCKED1]]>
// CHECK:           %[[VAL_11:.*]] = "tt.reduce"(%[[VAL_10]]) <{axis = 1 : i32}> ({
// CHECK:           ^bb0(%[[VAL_12:.*]]: f32, %[[VAL_13:.*]]: f32):
// CHECK:             %[[VAL_14:.*]] = arith.addf %[[VAL_12]], %[[VAL_13]] : f32
// CHECK:             tt.reduce.return %[[VAL_14]] : f32
// CHECK:           }) : (tensor<32x32xf32, #[[BLOCKED1]]>) -> tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #[[BLOCKED1]]}>>
// CHECK:           %[[VAL_15:.*]] = triton_gpu.convert_layout %[[VAL_11]] : tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #[[BLOCKED1]]}>> -> tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #[[DPAS]]}>>
// CHECK:           tt.return %[[VAL_15]] : tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #[[DPAS]]}>>
// CHECK:         }
  tt.func @test_two_warps_twice(%arg0: tensor<32x32xf32, #mma>) -> tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>> {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      tt.reduce.return %1 : f32
    }) : (tensor<32x32xf32, #mma>) -> tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    tt.return %0 : tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
  }
}

// -----

// Test reduction across 2 warps in the reduction dimension and 4 in the non-reduction dimension with different layout in reduction dimension.

#mma0 = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1]}>
#mma1 = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 2]}>

// CHECK-DAG: #[[DPAS0:.+]] = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 8], B = [8, 16], C = [8, 16]}>
// CHECK-DAG: #[[DPAS1:.+]] = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 2], A = [8, 8], B = [8, 32], C = [8, 32]}>
// CHECK-DAG: #[[BLOCKED0:.+]] = #triton_gpu.blocked<{sizePerThread = [8, 1, 1, 1, 1], threadsPerWarp = [1, 16, 1, 1, 1], warpsPerCTA = [4, 1, 1, 2, 1], order = [4, 0, 1, 2, 3]}>
// CHECK-DAG: #[[BLOCKED1:.+]] = #triton_gpu.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [4, 2], order = [1, 0]}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, triton_gpu.target = "xpu", "triton_gpu.threads-per-warp" = 16 : i32, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block} {

// CHECK:         tt.func @test_0(
// CHECK-SAME:                    %[[VAL_0:.*]]: tensor<64x64xf32, #[[DPAS0]]>) -> tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #[[DPAS0]]}>> {
// CHECK:           %[[VAL_1:.*]] = tt.reshape %[[VAL_0]] {allow_reorder = true} : tensor<64x64xf32, #[[DPAS0]]> -> tensor<64x16x1x2x2xf32, #[[BLOCKED0]]>
// CHECK:           %[[VAL_2:.*]] = "tt.reduce"(%[[VAL_1]]) <{axis = 4 : i32}> ({
// CHECK:           ^bb0(%[[VAL_3:.*]]: f32, %[[VAL_4:.*]]: f32):
// CHECK:             %[[VAL_5:.*]] = arith.maxnumf %[[VAL_3]], %[[VAL_4]] : f32
// CHECK:             tt.reduce.return %[[VAL_5]] : f32
// CHECK:           }) : (tensor<64x16x1x2x2xf32, #[[BLOCKED0]]>) -> tensor<64x16x1x2xf32, #triton_gpu.slice<{dim = 4, parent = #[[BLOCKED0]]}>>
// CHECK:           %[[VAL_6:.*]] = "tt.reduce"(%[[VAL_2]]) <{axis = 2 : i32}> ({
// CHECK:           ^bb0(%[[VAL_7:.*]]: f32, %[[VAL_8:.*]]: f32):
// CHECK:             %[[VAL_9:.*]] = arith.maxnumf %[[VAL_7]], %[[VAL_8]] : f32
// CHECK:             tt.reduce.return %[[VAL_9]] : f32
// CHECK:           }) : (tensor<64x16x1x2xf32, #triton_gpu.slice<{dim = 4, parent = #[[BLOCKED0]]}>>) -> tensor<64x16x2xf32, #triton_gpu.slice<{dim = 2, parent = #triton_gpu.slice<{dim = 4, parent = #[[BLOCKED0]]}>}>>
// CHECK:           %[[VAL_10:.*]] = tt.reshape %[[VAL_6]] {allow_reorder = true} : tensor<64x16x2xf32, #triton_gpu.slice<{dim = 2, parent = #triton_gpu.slice<{dim = 4, parent = #[[BLOCKED0]]}>}>> -> tensor<64x32xf32, #[[BLOCKED1]]>
// CHECK:           %[[VAL_11:.*]] = "tt.reduce"(%[[VAL_10]]) <{axis = 1 : i32}> ({
// CHECK:           ^bb0(%[[VAL_12:.*]]: f32, %[[VAL_13:.*]]: f32):
// CHECK:             %[[VAL_14:.*]] = arith.maxnumf %[[VAL_12]], %[[VAL_13]] : f32
// CHECK:             tt.reduce.return %[[VAL_14]] : f32
// CHECK:           }) : (tensor<64x32xf32, #[[BLOCKED1]]>) -> tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #[[BLOCKED1]]}>>
// CHECK:           %[[VAL_15:.*]] = triton_gpu.convert_layout %[[VAL_11]] : tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #[[BLOCKED1]]}>> -> tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #[[DPAS0]]}>>
// CHECK:           tt.return %[[VAL_15]] : tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #[[DPAS0]]}>>
// CHECK:         }
  tt.func @test_0(%arg0: tensor<64x64xf32, #mma0>) -> tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #mma0}>> {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.maxnumf %arg1, %arg2 : f32
      tt.reduce.return %1 : f32
    }) : (tensor<64x64xf32, #mma0>) -> tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #mma0}>>
    tt.return %0 : tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #mma0}>>
  }

// CHECK:         tt.func @test_1(
// CHECK-SAME:                    %[[VAL_0:.*]]: tensor<64x64xf32, #[[DPAS1]]>) -> tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #[[DPAS1]]}>> {
// CHECK:           %[[VAL_1:.*]] = tt.reshape %[[VAL_0]] {allow_reorder = true} : tensor<64x64xf32, #[[DPAS1]]> -> tensor<64x16x2x2x1xf32, #[[BLOCKED0]]>
// CHECK:           %[[VAL_2:.*]] = "tt.reduce"(%[[VAL_1]]) <{axis = 4 : i32}> ({
// CHECK:           ^bb0(%[[VAL_3:.*]]: f32, %[[VAL_4:.*]]: f32):
// CHECK:             %[[VAL_5:.*]] = arith.maxnumf %[[VAL_3]], %[[VAL_4]] : f32
// CHECK:             tt.reduce.return %[[VAL_5]] : f32
// CHECK:           }) : (tensor<64x16x2x2x1xf32, #[[BLOCKED0]]>) -> tensor<64x16x2x2xf32, #triton_gpu.slice<{dim = 4, parent = #[[BLOCKED0]]}>>
// CHECK:           %[[VAL_6:.*]] = "tt.reduce"(%[[VAL_2]]) <{axis = 2 : i32}> ({
// CHECK:           ^bb0(%[[VAL_7:.*]]: f32, %[[VAL_8:.*]]: f32):
// CHECK:             %[[VAL_9:.*]] = arith.maxnumf %[[VAL_7]], %[[VAL_8]] : f32
// CHECK:             tt.reduce.return %[[VAL_9]] : f32
// CHECK:           }) : (tensor<64x16x2x2xf32, #triton_gpu.slice<{dim = 4, parent = #[[BLOCKED0]]}>>) -> tensor<64x16x2xf32, #triton_gpu.slice<{dim = 2, parent = #triton_gpu.slice<{dim = 4, parent = #[[BLOCKED0]]}>}>>
// CHECK:           %[[VAL_10:.*]] = tt.reshape %[[VAL_6]] {allow_reorder = true} : tensor<64x16x2xf32, #triton_gpu.slice<{dim = 2, parent = #triton_gpu.slice<{dim = 4, parent = #[[BLOCKED0]]}>}>> -> tensor<64x32xf32, #[[BLOCKED1]]>
// CHECK:           %[[VAL_11:.*]] = "tt.reduce"(%[[VAL_10]]) <{axis = 1 : i32}> ({
// CHECK:           ^bb0(%[[VAL_12:.*]]: f32, %[[VAL_13:.*]]: f32):
// CHECK:             %[[VAL_14:.*]] = arith.maxnumf %[[VAL_12]], %[[VAL_13]] : f32
// CHECK:             tt.reduce.return %[[VAL_14]] : f32
// CHECK:           }) : (tensor<64x32xf32, #[[BLOCKED1]]>) -> tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #[[BLOCKED1]]}>>
// CHECK:           %[[VAL_15:.*]] = triton_gpu.convert_layout %[[VAL_11]] : tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #[[BLOCKED1]]}>> -> tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #[[DPAS1]]}>>
// CHECK:           tt.return %[[VAL_15]] : tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #[[DPAS1]]}>>
// CHECK:         }
  tt.func @test_1(%arg0: tensor<64x64xf32, #mma1>) -> tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #mma1}>> {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.maxnumf %arg1, %arg2 : f32
      tt.reduce.return %1 : f32
    }) : (tensor<64x64xf32, #mma1>) -> tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #mma1}>>
    tt.return %0 : tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #mma1}>>
  }
}
