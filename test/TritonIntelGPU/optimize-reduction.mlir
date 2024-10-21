// RUN: triton-opt %s --split-input-file -tritonintelgpu-optimize-reduction-locality | FileCheck %s

// Test reduction in a single warp (16x16->16).

#mma = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [2, 1]}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, triton_gpu.target = "xpu", "triton_gpu.threads-per-warp" = 16 : i32} {

// CHECK-DAG: #[[$ATTR_2:.+]] = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [2, 1], A = [16, 8], B = [8, 16], C = [16, 16]}>
// CHECK-DAG: #[[$ATTR_0:.+]] = #triton_gpu.blocked<{sizePerThread = [16, 1, 1, 1, 1, 1, 1], threadsPerWarp = [1, 1, 1, 16, 1, 1, 1], warpsPerCTA = [1, 1, 1, 1, 1, 1, 1], order = [3, 4, 5, 6, 0, 1, 2]}>
// CHECK-DAG: #[[$ATTR_1:.+]] = #triton_gpu.blocked<{sizePerThread = [1, 1, 1, 16], threadsPerWarp = [16, 1, 1, 1], warpsPerCTA = [1, 1, 1, 1], order = [3, 0, 1, 2]}>
// CHECK-DAG: #[[$ATTR_3:.+]] = #triton_gpu.blocked<{sizePerThread = [1, 1, 1, 16, 1], threadsPerWarp = [16, 1, 1, 1, 1], warpsPerCTA = [1, 1, 1, 1, 1], order = [3, 4, 0, 1, 2]}>
// CHECK-DAG: #[[$ATTR_4:.+]] = #triton_gpu.blocked<{sizePerThread = [16, 1, 1, 1], threadsPerWarp = [1, 1, 1, 16], warpsPerCTA = [1, 1, 1, 1], order = [3, 0, 1, 2]}>

// CHECK:         tt.func @test_single(
// CHECK-SAME:                         %[[VAL_0:.*]]: tensor<16x16xf32, #[[$ATTR_2]]>) -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_2]]}>> {
// CHECK:           %[[VAL_1:.*]] = tt.reshape %[[VAL_0]] {allow_reorder = true, efficient_layout} : tensor<16x16xf32, #[[$ATTR_2]]> -> tensor<16x1x1x16x1x1x1xf32, #[[$ATTR_0]]>
// CHECK:           %[[VAL_2:.*]] = "tt.reduce"(%[[VAL_1]]) <{axis = 6 : i32}> ({
// CHECK:           ^bb0(%[[VAL_3:.*]]: f32, %[[VAL_4:.*]]: f32):
// CHECK:             %[[VAL_5:.*]] = arith.addf %[[VAL_3]], %[[VAL_4]] : f32
// CHECK:             tt.reduce.return %[[VAL_5]] : f32
// CHECK:           }) : (tensor<16x1x1x16x1x1x1xf32, #[[$ATTR_0]]>) -> tensor<16x1x1x16x1x1xf32, #triton_gpu.slice<{dim = 6, parent = #[[$ATTR_0]]}>>
// CHECK:           %[[VAL_6:.*]] = "tt.reduce"(%[[VAL_2]]) <{axis = 4 : i32}> ({
// CHECK:           ^bb0(%[[VAL_7:.*]]: f32, %[[VAL_8:.*]]: f32):
// CHECK:             %[[VAL_9:.*]] = arith.addf %[[VAL_7]], %[[VAL_8]] : f32
// CHECK:             tt.reduce.return %[[VAL_9]] : f32
// CHECK:           }) : (tensor<16x1x1x16x1x1xf32, #triton_gpu.slice<{dim = 6, parent = #[[$ATTR_0]]}>>) -> tensor<16x1x1x16x1xf32, #triton_gpu.slice<{dim = 4, parent = #triton_gpu.slice<{dim = 6, parent = #[[$ATTR_0]]}>}>>
// CHECK:           %[[CONV:.*]] = triton_gpu.convert_layout %[[VAL_6]] : tensor<16x1x1x16x1xf32, #triton_gpu.slice<{dim = 4, parent = #triton_gpu.slice<{dim = 6, parent = #[[$ATTR_0]]}>}>> -> tensor<16x1x1x16x1xf32, #[[$ATTR_3]]>
// CHECK:           %[[VAL_10:.*]] = tt.reshape %[[CONV]] {allow_reorder = true, efficient_layout} : tensor<16x1x1x16x1xf32, #[[$ATTR_3]]> -> tensor<16x1x1x16xf32, #[[$ATTR_1]]>
// CHECK:           %[[VAL_11:.*]] = "tt.reduce"(%[[VAL_10]]) <{axis = 3 : i32}> ({
// CHECK:           ^bb0(%[[VAL_12:.*]]: f32, %[[VAL_13:.*]]: f32):
// CHECK:             %[[VAL_14:.*]] = arith.addf %[[VAL_12]], %[[VAL_13]] : f32
// CHECK:             tt.reduce.return %[[VAL_14]] : f32
// CHECK:           }) : (tensor<16x1x1x16xf32, #[[$ATTR_1]]>) -> tensor<16x1x1xf32, #triton_gpu.slice<{dim = 3, parent = #[[$ATTR_1]]}>>
// CHECK:           %[[VAL_15:.*]] = triton_gpu.convert_layout %[[VAL_11]] : tensor<16x1x1xf32, #triton_gpu.slice<{dim = 3, parent = #[[$ATTR_1]]}>> -> tensor<16x1x1xf32, #triton_gpu.slice<{dim = 3, parent = #[[$ATTR_4]]}>>
// CHECK:           %[[VAL_16:.*]] = tt.reshape %[[VAL_15]] {allow_reorder = true, efficient_layout} : tensor<16x1x1xf32, #triton_gpu.slice<{dim = 3, parent = #[[$ATTR_4]]}>> -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_2]]}>>
// CHECK:           tt.return %[[VAL_16]] : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_2]]}>>
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

#mma = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [2, 1], repCluster = [2, 1]}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 2 : i32, triton_gpu.target = "xpu", "triton_gpu.threads-per-warp" = 16 : i32} {

// CHECK-DAG: #[[$ATTR_5:.+]] = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [2, 1], repCluster = [2, 1], A = [16, 8], B = [8, 16], C = [16, 16]}>
// CHECK-DAG: #[[$ATTR_3:.+]] = #triton_gpu.blocked<{sizePerThread = [16, 1, 1, 1, 1, 1, 1], threadsPerWarp = [1, 1, 1, 16, 1, 1, 1], warpsPerCTA = [1, 1, 2, 1, 1, 1, 1], order = [3, 4, 5, 6, 0, 1, 2]}>
// CHECK-DAG: #[[$ATTR_4:.+]] = #triton_gpu.blocked<{sizePerThread = [1, 1, 1, 16], threadsPerWarp = [16, 1, 1, 1], warpsPerCTA = [1, 1, 2, 1], order = [3, 0, 1, 2]}>
// CHECK-DAG: #[[$BLOCKED:.+]] = #triton_gpu.blocked<{sizePerThread = [1, 1, 1, 16, 1], threadsPerWarp = [16, 1, 1, 1, 1], warpsPerCTA = [1, 1, 2, 1, 1], order = [3, 4, 0, 1, 2]}>
// CHECK-DAG: #[[$BLOCKED1:.+]] = #triton_gpu.blocked<{sizePerThread = [16, 1, 1, 1], threadsPerWarp = [1, 1, 1, 16], warpsPerCTA = [1, 1, 2, 1], order = [3, 0, 1, 2]}>

// CHECK:         tt.func @test_single_twice(
// CHECK-SAME:                               %[[VAL_0:.*]]: tensor<32x16xf32, #[[$ATTR_5]]>) -> tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_5]]}>> {
// CHECK:           %[[VAL_1:.*]] = tt.reshape %[[VAL_0]] {allow_reorder = true, efficient_layout} : tensor<32x16xf32, #[[$ATTR_5]]> -> tensor<16x1x2x16x1x1x1xf32, #[[$ATTR_3]]>
// CHECK:           %[[VAL_2:.*]] = "tt.reduce"(%[[VAL_1]]) <{axis = 6 : i32}> ({
// CHECK:           ^bb0(%[[VAL_3:.*]]: f32, %[[VAL_4:.*]]: f32):
// CHECK:             %[[VAL_5:.*]] = arith.addf %[[VAL_3]], %[[VAL_4]] : f32
// CHECK:             tt.reduce.return %[[VAL_5]] : f32
// CHECK:           }) : (tensor<16x1x2x16x1x1x1xf32, #[[$ATTR_3]]>) -> tensor<16x1x2x16x1x1xf32, #triton_gpu.slice<{dim = 6, parent = #[[$ATTR_3]]}>>
// CHECK:           %[[VAL_6:.*]] = "tt.reduce"(%[[VAL_2]]) <{axis = 4 : i32}> ({
// CHECK:           ^bb0(%[[VAL_7:.*]]: f32, %[[VAL_8:.*]]: f32):
// CHECK:             %[[VAL_9:.*]] = arith.addf %[[VAL_7]], %[[VAL_8]] : f32
// CHECK:             tt.reduce.return %[[VAL_9]] : f32
// CHECK:           }) : (tensor<16x1x2x16x1x1xf32, #triton_gpu.slice<{dim = 6, parent = #[[$ATTR_3]]}>>) -> tensor<16x1x2x16x1xf32, #triton_gpu.slice<{dim = 4, parent = #triton_gpu.slice<{dim = 6, parent = #[[$ATTR_3]]}>}>>
// CHECK:           %[[CONV:.*]] = triton_gpu.convert_layout %[[VAL_6]] : tensor<16x1x2x16x1xf32, #triton_gpu.slice<{dim = 4, parent = #triton_gpu.slice<{dim = 6, parent = #[[$ATTR_3]]}>}>> -> tensor<16x1x2x16x1xf32, #[[$BLOCKED]]>
// CHECK:           %[[VAL_10:.*]] = tt.reshape %[[CONV]] {allow_reorder = true, efficient_layout} : tensor<16x1x2x16x1xf32, #[[$BLOCKED]]> -> tensor<16x1x2x16xf32, #[[$ATTR_4]]>
// CHECK:           %[[VAL_11:.*]] = "tt.reduce"(%[[VAL_10]]) <{axis = 3 : i32}> ({
// CHECK:           ^bb0(%[[VAL_12:.*]]: f32, %[[VAL_13:.*]]: f32):
// CHECK:             %[[VAL_14:.*]] = arith.addf %[[VAL_12]], %[[VAL_13]] : f32
// CHECK:             tt.reduce.return %[[VAL_14]] : f32
// CHECK:           }) : (tensor<16x1x2x16xf32, #[[$ATTR_4]]>) -> tensor<16x1x2xf32, #triton_gpu.slice<{dim = 3, parent = #[[$ATTR_4]]}>>
// CHECK:           %[[VAL_15:.*]] = triton_gpu.convert_layout %[[VAL_11]] : tensor<16x1x2xf32, #triton_gpu.slice<{dim = 3, parent = #[[$ATTR_4]]}>> -> tensor<16x1x2xf32, #triton_gpu.slice<{dim = 3, parent = #[[$BLOCKED1]]}>>
// CHECK:           %[[VAL_16:.*]] = tt.reshape %[[VAL_15]] {allow_reorder = true, efficient_layout} : tensor<16x1x2xf32, #triton_gpu.slice<{dim = 3, parent = #[[$BLOCKED1]]}>> -> tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_5]]}>>
// CHECK:           tt.return %[[VAL_16]] : tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_5]]}>>
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

#mma = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [1, 2], repCluster = [2, 1]}>

// CHECK-DAG: #[[$ATTR_8:.+]] = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [1, 2], repCluster = [2, 1], A = [16, 8], B = [8, 16], C = [16, 16]}>
// CHECK-DAG: #[[$ATTR_6:.+]] = #triton_gpu.blocked<{sizePerThread = [16, 1, 1, 1, 1, 1, 1], threadsPerWarp = [1, 1, 1, 16, 1, 1, 1], warpsPerCTA = [1, 1, 1, 1, 1, 2, 1], order = [3, 4, 5, 6, 0, 1, 2]}>
// CHECK-DAG: #[[$ATTR_7:.+]] = #triton_gpu.blocked<{sizePerThread = [1, 1, 1, 16], threadsPerWarp = [16, 1, 1, 1], warpsPerCTA = [1, 1, 1, 2], order = [3, 0, 1, 2]}>
// CHECK-DAG: #[[$BLOCKED:.+]] = #triton_gpu.blocked<{sizePerThread = [1, 1, 1, 16, 1], threadsPerWarp = [16, 1, 1, 1, 1], warpsPerCTA = [1, 1, 1, 1, 2], order = [3, 4, 0, 1, 2]}>
// CHECK-DAG: #[[$BLOCKED1:.+]] = #triton_gpu.blocked<{sizePerThread = [16, 1, 1, 1], threadsPerWarp = [1, 1, 1, 16], warpsPerCTA = [1, 1, 1, 2], order = [3, 0, 1, 2]}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 2 : i32, triton_gpu.target = "xpu", "triton_gpu.threads-per-warp" = 16 : i32} {

// CHECK-LABEL:   tt.func @test_two_warps_red(
// CHECK-SAME:                                %[[VAL_0:.*]]: tensor<16x32xf32, #[[$ATTR_8]]>) -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_8]]}>> {
// CHECK:           %[[VAL_1:.*]] = tt.reshape %[[VAL_0]] {allow_reorder = true, efficient_layout} : tensor<16x32xf32, #[[$ATTR_8]]> -> tensor<16x1x1x16x1x2x1xf32, #[[$ATTR_6]]>
// CHECK:           %[[VAL_2:.*]] = "tt.reduce"(%[[VAL_1]]) <{axis = 6 : i32}> ({
// CHECK:           ^bb0(%[[VAL_3:.*]]: f32, %[[VAL_4:.*]]: f32):
// CHECK:             %[[VAL_5:.*]] = arith.addf %[[VAL_3]], %[[VAL_4]] : f32
// CHECK:             tt.reduce.return %[[VAL_5]] : f32
// CHECK:           }) : (tensor<16x1x1x16x1x2x1xf32, #[[$ATTR_6]]>) -> tensor<16x1x1x16x1x2xf32, #triton_gpu.slice<{dim = 6, parent = #[[$ATTR_6]]}>>
// CHECK:           %[[VAL_6:.*]] = "tt.reduce"(%[[VAL_2]]) <{axis = 4 : i32}> ({
// CHECK:           ^bb0(%[[VAL_7:.*]]: f32, %[[VAL_8:.*]]: f32):
// CHECK:             %[[VAL_9:.*]] = arith.addf %[[VAL_7]], %[[VAL_8]] : f32
// CHECK:             tt.reduce.return %[[VAL_9]] : f32
// CHECK:           }) : (tensor<16x1x1x16x1x2xf32, #triton_gpu.slice<{dim = 6, parent = #[[$ATTR_6]]}>>) -> tensor<16x1x1x16x2xf32, #triton_gpu.slice<{dim = 4, parent = #triton_gpu.slice<{dim = 6, parent = #[[$ATTR_6]]}>}>>
// CHECK:           %[[CONV:.*]] = triton_gpu.convert_layout %[[VAL_6]] : tensor<16x1x1x16x2xf32, #triton_gpu.slice<{dim = 4, parent = #triton_gpu.slice<{dim = 6, parent = #[[$ATTR_6]]}>}>> -> tensor<16x1x1x16x2xf32, #[[$BLOCKED]]>
// CHECK:           %[[VAL_10:.*]] = tt.reshape %[[CONV]] {allow_reorder = true, efficient_layout} : tensor<16x1x1x16x2xf32, #[[$BLOCKED]]> -> tensor<16x1x1x32xf32, #[[$ATTR_7]]>
// CHECK:           %[[VAL_11:.*]] = "tt.reduce"(%[[VAL_10]]) <{axis = 3 : i32}> ({
// CHECK:           ^bb0(%[[VAL_12:.*]]: f32, %[[VAL_13:.*]]: f32):
// CHECK:             %[[VAL_14:.*]] = arith.addf %[[VAL_12]], %[[VAL_13]] : f32
// CHECK:             tt.reduce.return %[[VAL_14]] : f32
// CHECK:           }) : (tensor<16x1x1x32xf32, #[[$ATTR_7]]>) -> tensor<16x1x1xf32, #triton_gpu.slice<{dim = 3, parent = #[[$ATTR_7]]}>>
// CHECK:           %[[VAL_15:.*]] = triton_gpu.convert_layout %[[VAL_11]] : tensor<16x1x1xf32, #triton_gpu.slice<{dim = 3, parent = #[[$ATTR_7]]}>> -> tensor<16x1x1xf32, #triton_gpu.slice<{dim = 3, parent = #[[$BLOCKED1]]}>>
// CHECK:           %[[VAL_16:.*]] = tt.reshape %[[VAL_15]] {allow_reorder = true, efficient_layout} : tensor<16x1x1xf32, #triton_gpu.slice<{dim = 3, parent = #[[$BLOCKED1]]}>> -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_8]]}>>
// CHECK:           tt.return %[[VAL_16]] : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_8]]}>>
// CHECK:         }
  tt.func @test_two_warps_red(%arg0: tensor<16x32xf32, #mma>) -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>> {
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

#mma = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [2, 1]}>

// CHECK-DAG: #[[$ATTR_9:.+]] = #triton_gpu.blocked<{sizePerThread = [16, 1, 1, 1, 1, 1, 1], threadsPerWarp = [1, 1, 1, 16, 1, 1, 1], warpsPerCTA = [1, 1, 2, 1, 1, 2, 1], order = [3, 4, 5, 6, 0, 1, 2]}>
// CHECK-DAG: #[[$ATTR_10:.+]] = #triton_gpu.blocked<{sizePerThread = [1, 1, 1, 16], threadsPerWarp = [16, 1, 1, 1], warpsPerCTA = [1, 1, 2, 2], order = [3, 0, 1, 2]}>
// CHECK-DAG: #[[$ATTR_11:.+]] = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [2, 1], A = [16, 8], B = [8, 16], C = [16, 16]}>
// CHECK-DAG: #[[$BLOCKED:.+]] = #triton_gpu.blocked<{sizePerThread = [1, 1, 1, 16, 1], threadsPerWarp = [16, 1, 1, 1, 1], warpsPerCTA = [1, 1, 2, 1, 2], order = [3, 4, 0, 1, 2]}>
// CHECK-DAG: #[[$BLOCKED1:.+]] = #triton_gpu.blocked<{sizePerThread = [16, 1, 1, 1], threadsPerWarp = [1, 1, 1, 16], warpsPerCTA = [1, 1, 2, 2], order = [3, 0, 1, 2]}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "xpu", "triton_gpu.threads-per-warp" = 16 : i32} {

// CHECK-LABEL:   tt.func @test_two_warps(
// CHECK-SAME:                            %[[VAL_0:.*]]: tensor<32x32xf32, #[[$ATTR_11]]>) -> tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_11]]}>> {
// CHECK:           %[[VAL_1:.*]] = tt.reshape %[[VAL_0]] {allow_reorder = true, efficient_layout} : tensor<32x32xf32, #[[$ATTR_11]]> -> tensor<16x1x2x16x1x2x1xf32, #[[$ATTR_9]]>
// CHECK:           %[[VAL_2:.*]] = "tt.reduce"(%[[VAL_1]]) <{axis = 6 : i32}> ({
// CHECK:           ^bb0(%[[VAL_3:.*]]: f32, %[[VAL_4:.*]]: f32):
// CHECK:             %[[VAL_5:.*]] = arith.addf %[[VAL_3]], %[[VAL_4]] : f32
// CHECK:             tt.reduce.return %[[VAL_5]] : f32
// CHECK:           }) : (tensor<16x1x2x16x1x2x1xf32, #[[$ATTR_9]]>) -> tensor<16x1x2x16x1x2xf32, #triton_gpu.slice<{dim = 6, parent = #[[$ATTR_9]]}>>
// CHECK:           %[[VAL_6:.*]] = "tt.reduce"(%[[VAL_2]]) <{axis = 4 : i32}> ({
// CHECK:           ^bb0(%[[VAL_7:.*]]: f32, %[[VAL_8:.*]]: f32):
// CHECK:             %[[VAL_9:.*]] = arith.addf %[[VAL_7]], %[[VAL_8]] : f32
// CHECK:             tt.reduce.return %[[VAL_9]] : f32
// CHECK:           }) : (tensor<16x1x2x16x1x2xf32, #triton_gpu.slice<{dim = 6, parent = #[[$ATTR_9]]}>>) -> tensor<16x1x2x16x2xf32, #triton_gpu.slice<{dim = 4, parent = #triton_gpu.slice<{dim = 6, parent = #[[$ATTR_9]]}>}>>
// CHECK:           %[[CONV:.*]] = triton_gpu.convert_layout %[[VAL_6]] : tensor<16x1x2x16x2xf32, #triton_gpu.slice<{dim = 4, parent = #triton_gpu.slice<{dim = 6, parent = #[[$ATTR_9]]}>}>> -> tensor<16x1x2x16x2xf32, #[[$BLOCKED]]>
// CHECK:           %[[VAL_10:.*]] = tt.reshape %[[CONV]] {allow_reorder = true, efficient_layout} : tensor<16x1x2x16x2xf32, #[[$BLOCKED]]> -> tensor<16x1x2x32xf32, #[[$ATTR_10]]>
// CHECK:           %[[VAL_11:.*]] = "tt.reduce"(%[[VAL_10]]) <{axis = 3 : i32}> ({
// CHECK:           ^bb0(%[[VAL_12:.*]]: f32, %[[VAL_13:.*]]: f32):
// CHECK:             %[[VAL_14:.*]] = arith.addf %[[VAL_12]], %[[VAL_13]] : f32
// CHECK:             tt.reduce.return %[[VAL_14]] : f32
// CHECK:           }) : (tensor<16x1x2x32xf32, #[[$ATTR_10]]>) -> tensor<16x1x2xf32, #triton_gpu.slice<{dim = 3, parent = #[[$ATTR_10]]}>>
// CHECK:           %[[VAL_15:.*]] = triton_gpu.convert_layout %[[VAL_11]] : tensor<16x1x2xf32, #triton_gpu.slice<{dim = 3, parent = #[[$ATTR_10]]}>> -> tensor<16x1x2xf32, #triton_gpu.slice<{dim = 3, parent = #[[$BLOCKED1]]}>>
// CHECK:           %[[VAL_16:.*]] = tt.reshape %[[VAL_15]] {allow_reorder = true, efficient_layout} : tensor<16x1x2xf32, #triton_gpu.slice<{dim = 3, parent = #[[$BLOCKED1]]}>> -> tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_11]]}>>
// CHECK:           tt.return %[[VAL_16]] : tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_11]]}>>
// CHECK:         }
  tt.func @test_two_warps(%arg0: tensor<32x32xf32, #mma>) -> tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>> {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      tt.reduce.return %1 : f32
    }) : (tensor<32x32xf32, #mma>) -> tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    tt.return %0 : tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
  }

// CHECK-LABEL:   tt.func @test_two_warps_twice(
// CHECK-SAME:                                  %[[VAL_0:.*]]: tensor<64x32xf32, #[[$ATTR_11]]>) -> tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_11]]}>> {
// CHECK:           %[[VAL_1:.*]] = tt.reshape %[[VAL_0]] {allow_reorder = true, efficient_layout} : tensor<64x32xf32, #[[$ATTR_11]]> -> tensor<16x1x4x16x1x2x1xf32, #[[$ATTR_9]]>
// CHECK:           %[[VAL_2:.*]] = "tt.reduce"(%[[VAL_1]]) <{axis = 6 : i32}> ({
// CHECK:           ^bb0(%[[VAL_3:.*]]: f32, %[[VAL_4:.*]]: f32):
// CHECK:             %[[VAL_5:.*]] = arith.addf %[[VAL_3]], %[[VAL_4]] : f32
// CHECK:             tt.reduce.return %[[VAL_5]] : f32
// CHECK:           }) : (tensor<16x1x4x16x1x2x1xf32, #[[$ATTR_9]]>) -> tensor<16x1x4x16x1x2xf32, #triton_gpu.slice<{dim = 6, parent = #[[$ATTR_9]]}>>
// CHECK:           %[[VAL_6:.*]] = "tt.reduce"(%[[VAL_2]]) <{axis = 4 : i32}> ({
// CHECK:           ^bb0(%[[VAL_7:.*]]: f32, %[[VAL_8:.*]]: f32):
// CHECK:             %[[VAL_9:.*]] = arith.addf %[[VAL_7]], %[[VAL_8]] : f32
// CHECK:             tt.reduce.return %[[VAL_9]] : f32
// CHECK:           }) : (tensor<16x1x4x16x1x2xf32, #triton_gpu.slice<{dim = 6, parent = #[[$ATTR_9]]}>>) -> tensor<16x1x4x16x2xf32, #triton_gpu.slice<{dim = 4, parent = #triton_gpu.slice<{dim = 6, parent = #[[$ATTR_9]]}>}>>
// CHECK:           %[[CONV:.*]] = triton_gpu.convert_layout %[[VAL_6]] : tensor<16x1x4x16x2xf32, #triton_gpu.slice<{dim = 4, parent = #triton_gpu.slice<{dim = 6, parent = #[[$ATTR_9]]}>}>> -> tensor<16x1x4x16x2xf32, #[[$BLOCKED]]>
// CHECK:           %[[VAL_10:.*]] = tt.reshape %[[CONV]] {allow_reorder = true, efficient_layout} : tensor<16x1x4x16x2xf32, #[[$BLOCKED]]> -> tensor<16x1x4x32xf32, #[[$ATTR_10]]>
// CHECK:           %[[VAL_11:.*]] = "tt.reduce"(%[[VAL_10]]) <{axis = 3 : i32}> ({
// CHECK:           ^bb0(%[[VAL_12:.*]]: f32, %[[VAL_13:.*]]: f32):
// CHECK:             %[[VAL_14:.*]] = arith.addf %[[VAL_12]], %[[VAL_13]] : f32
// CHECK:             tt.reduce.return %[[VAL_14]] : f32
// CHECK:           }) : (tensor<16x1x4x32xf32, #[[$ATTR_10]]>) -> tensor<16x1x4xf32, #triton_gpu.slice<{dim = 3, parent = #[[$ATTR_10]]}>>
// CHECK:           %[[VAL_15:.*]] = triton_gpu.convert_layout %[[VAL_11]] : tensor<16x1x4xf32, #triton_gpu.slice<{dim = 3, parent = #[[$ATTR_10]]}>> -> tensor<16x1x4xf32, #triton_gpu.slice<{dim = 3, parent = #[[$BLOCKED1]]}>>
// CHECK:           %[[VAL_16:.*]] = tt.reshape %[[VAL_15]] {allow_reorder = true, efficient_layout} : tensor<16x1x4xf32, #triton_gpu.slice<{dim = 3, parent = #[[$BLOCKED1]]}>> -> tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_11]]}>>
// CHECK:           tt.return %[[VAL_16]] : tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_11]]}>>
// CHECK:         }
  tt.func @test_two_warps_twice(%arg0: tensor<64x32xf32, #mma>) -> tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>> {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      tt.reduce.return %1 : f32
    }) : (tensor<64x32xf32, #mma>) -> tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    tt.return %0 : tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
  }
}

// -----

// Test reduction across 2 warps in the reduction dimension and 4 in the non-reduction dimension.

// CHECK-DAG: #[[$ATTR_14:.+]] = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [2, 2], A = [16, 8], B = [8, 32], C = [16, 32]}>
// CHECK-DAG: #[[$ATTR_12:.+]] = #triton_gpu.blocked<{sizePerThread = [16, 1, 1, 1, 1, 1, 1], threadsPerWarp = [1, 1, 1, 16, 1, 1, 1], warpsPerCTA = [1, 1, 4, 1, 1, 2, 1], order = [3, 4, 5, 6, 0, 1, 2]}>
// CHECK-DAG: #[[$ATTR_13:.+]] = #triton_gpu.blocked<{sizePerThread = [1, 1, 1, 16], threadsPerWarp = [16, 1, 1, 1], warpsPerCTA = [1, 1, 4, 2], order = [3, 0, 1, 2]}>
// CHECK-DAG: #[[$BLOCKED:.+]] = #triton_gpu.blocked<{sizePerThread = [1, 1, 1, 16, 1], threadsPerWarp = [16, 1, 1, 1, 1], warpsPerCTA = [1, 1, 4, 1, 2], order = [3, 4, 0, 1, 2]}>
// CHECK-DAG: #[[$BLOCKED1:.+]] = #triton_gpu.blocked<{sizePerThread = [16, 1, 1, 1], threadsPerWarp = [1, 1, 1, 16], warpsPerCTA = [1, 1, 4, 2], order = [3, 0, 1, 2]}>

#mma = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [2, 2]}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, triton_gpu.target = "xpu", "triton_gpu.threads-per-warp" = 16 : i32} {
// CHECK:         tt.func @test(
// CHECK-SAME:                  %[[VAL_0:.*]]: tensor<64x64xf32, #[[$ATTR_14]]>) -> tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_14]]}>> {
// CHECK:           %[[VAL_1:.*]] = tt.reshape %[[VAL_0]] {allow_reorder = true, efficient_layout} : tensor<64x64xf32, #[[$ATTR_14]]> -> tensor<16x1x4x16x2x2x1xf32, #[[$ATTR_12]]>
// CHECK:           %[[VAL_2:.*]] = "tt.reduce"(%[[VAL_1]]) <{axis = 6 : i32}> ({
// CHECK:           ^bb0(%[[VAL_3:.*]]: f32, %[[VAL_4:.*]]: f32):
// CHECK:             %[[VAL_5:.*]] = arith.maxnumf %[[VAL_3]], %[[VAL_4]] : f32
// CHECK:             tt.reduce.return %[[VAL_5]] : f32
// CHECK:           }) : (tensor<16x1x4x16x2x2x1xf32, #[[$ATTR_12]]>) -> tensor<16x1x4x16x2x2xf32, #triton_gpu.slice<{dim = 6, parent = #[[$ATTR_12]]}>>
// CHECK:           %[[VAL_6:.*]] = "tt.reduce"(%[[VAL_2]]) <{axis = 4 : i32}> ({
// CHECK:           ^bb0(%[[VAL_7:.*]]: f32, %[[VAL_8:.*]]: f32):
// CHECK:             %[[VAL_9:.*]] = arith.maxnumf %[[VAL_7]], %[[VAL_8]] : f32
// CHECK:             tt.reduce.return %[[VAL_9]] : f32
// CHECK:           }) : (tensor<16x1x4x16x2x2xf32, #triton_gpu.slice<{dim = 6, parent = #[[$ATTR_12]]}>>) -> tensor<16x1x4x16x2xf32, #triton_gpu.slice<{dim = 4, parent = #triton_gpu.slice<{dim = 6, parent = #[[$ATTR_12]]}>}>>
// CHECK:           %[[CONV:.*]] = triton_gpu.convert_layout %[[VAL_6]] : tensor<16x1x4x16x2xf32, #triton_gpu.slice<{dim = 4, parent = #triton_gpu.slice<{dim = 6, parent = #[[$ATTR_12]]}>}>> -> tensor<16x1x4x16x2xf32, #[[$BLOCKED]]>
// CHECK:           %[[VAL_10:.*]] = tt.reshape %[[CONV]] {allow_reorder = true, efficient_layout} : tensor<16x1x4x16x2xf32, #[[$BLOCKED]]> -> tensor<16x1x4x32xf32, #[[$ATTR_13]]>
// CHECK:           %[[VAL_11:.*]] = "tt.reduce"(%[[VAL_10]]) <{axis = 3 : i32}> ({
// CHECK:           ^bb0(%[[VAL_12:.*]]: f32, %[[VAL_13:.*]]: f32):
// CHECK:             %[[VAL_14:.*]] = arith.maxnumf %[[VAL_12]], %[[VAL_13]] : f32
// CHECK:             tt.reduce.return %[[VAL_14]] : f32
// CHECK:           }) : (tensor<16x1x4x32xf32, #[[$ATTR_13]]>) -> tensor<16x1x4xf32, #triton_gpu.slice<{dim = 3, parent = #[[$ATTR_13]]}>>
// CHECK:           %[[VAL_15:.*]] = triton_gpu.convert_layout %[[VAL_11]] : tensor<16x1x4xf32, #triton_gpu.slice<{dim = 3, parent = #[[$ATTR_13]]}>> -> tensor<16x1x4xf32, #triton_gpu.slice<{dim = 3, parent = #[[$BLOCKED1]]}>>
// CHECK:           %[[VAL_16:.*]] = tt.reshape %[[VAL_15]] {allow_reorder = true, efficient_layout} : tensor<16x1x4xf32, #triton_gpu.slice<{dim = 3, parent = #[[$BLOCKED1]]}>> -> tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_14]]}>>
// CHECK:           tt.return %[[VAL_16]] : tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_14]]}>>
// CHECK:         }
  tt.func @test(%arg0: tensor<64x64xf32, #mma>) -> tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>> {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.maxnumf %arg1, %arg2 : f32
      tt.reduce.return %1 : f32
    }) : (tensor<64x64xf32, #mma>) -> tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    tt.return %0 : tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
  }

// CHECK:         tt.func @test_repeat_layout(
// CHECK-SAME:                                %[[VAL_0:.*]]: tensor<128x128xf32, #[[$ATTR_14]]>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_14]]}>> {
// CHECK:           %[[VAL_1:.*]] = tt.reshape %[[VAL_0]] {allow_reorder = true, efficient_layout} : tensor<128x128xf32, #[[$ATTR_14]]> -> tensor<16x1x8x16x2x2x2xf32, #[[$ATTR_12]]>
// CHECK:           %[[VAL_2:.*]] = "tt.reduce"(%[[VAL_1]]) <{axis = 6 : i32}> ({
// CHECK:           ^bb0(%[[VAL_3:.*]]: f32, %[[VAL_4:.*]]: f32):
// CHECK:             %[[VAL_5:.*]] = arith.maxnumf %[[VAL_3]], %[[VAL_4]] : f32
// CHECK:             tt.reduce.return %[[VAL_5]] : f32
// CHECK:           }) : (tensor<16x1x8x16x2x2x2xf32, #[[$ATTR_12]]>) -> tensor<16x1x8x16x2x2xf32, #triton_gpu.slice<{dim = 6, parent = #[[$ATTR_12]]}>>
// CHECK:           %[[VAL_6:.*]] = "tt.reduce"(%[[VAL_2]]) <{axis = 4 : i32}> ({
// CHECK:           ^bb0(%[[VAL_7:.*]]: f32, %[[VAL_8:.*]]: f32):
// CHECK:             %[[VAL_9:.*]] = arith.maxnumf %[[VAL_7]], %[[VAL_8]] : f32
// CHECK:             tt.reduce.return %[[VAL_9]] : f32
// CHECK:           }) : (tensor<16x1x8x16x2x2xf32, #triton_gpu.slice<{dim = 6, parent = #[[$ATTR_12]]}>>) -> tensor<16x1x8x16x2xf32, #triton_gpu.slice<{dim = 4, parent = #triton_gpu.slice<{dim = 6, parent = #[[$ATTR_12]]}>}>>
// CHECK:           %[[CONV:.*]] = triton_gpu.convert_layout %[[VAL_6]] : tensor<16x1x8x16x2xf32, #triton_gpu.slice<{dim = 4, parent = #triton_gpu.slice<{dim = 6, parent = #[[$ATTR_12]]}>}>> -> tensor<16x1x8x16x2xf32, #[[$BLOCKED]]>
// CHECK:           %[[VAL_10:.*]] = tt.reshape %[[CONV]] {allow_reorder = true, efficient_layout} : tensor<16x1x8x16x2xf32, #[[$BLOCKED]]> -> tensor<16x1x8x32xf32, #[[$ATTR_13]]>
// CHECK:           %[[VAL_11:.*]] = "tt.reduce"(%[[VAL_10]]) <{axis = 3 : i32}> ({
// CHECK:           ^bb0(%[[VAL_12:.*]]: f32, %[[VAL_13:.*]]: f32):
// CHECK:             %[[VAL_14:.*]] = arith.maxnumf %[[VAL_12]], %[[VAL_13]] : f32
// CHECK:             tt.reduce.return %[[VAL_14]] : f32
// CHECK:           }) : (tensor<16x1x8x32xf32, #[[$ATTR_13]]>) -> tensor<16x1x8xf32, #triton_gpu.slice<{dim = 3, parent = #[[$ATTR_13]]}>>
// CHECK:           %[[VAL_15:.*]] = triton_gpu.convert_layout %[[VAL_11]] : tensor<16x1x8xf32, #triton_gpu.slice<{dim = 3, parent = #[[$ATTR_13]]}>> -> tensor<16x1x8xf32, #triton_gpu.slice<{dim = 3, parent = #[[$BLOCKED1]]}>>
// CHECK:           %[[VAL_16:.*]] = tt.reshape %[[VAL_15]] {allow_reorder = true, efficient_layout} : tensor<16x1x8xf32, #triton_gpu.slice<{dim = 3, parent = #[[$BLOCKED1]]}>> -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_14]]}>>
// CHECK:           tt.return %[[VAL_16]] : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_14]]}>>
// CHECK:         }
  tt.func @test_repeat_layout(%arg0: tensor<128x128xf32, #mma>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>> {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.maxnumf %arg1, %arg2 : f32
      tt.reduce.return %1 : f32
    }) : (tensor<128x128xf32, #mma>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    tt.return %0 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
  }
}

// -----

// Test reduction across 2 warps in the reduction dimension and 4 in the non-reduction dimension with repCluster[0] = 4.

// CHECK-DAG: #[[$DPAS:.+]] = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [4, 2], A = [32, 8], B = [8, 32], C = [32, 32]}>
// CHECK-DAG: #[[$BLOCKED_EW:.+]] = #triton_gpu.blocked<{sizePerThread = [16, 1, 1, 1, 1, 1, 1], threadsPerWarp = [1, 1, 1, 16, 1, 1, 1], warpsPerCTA = [1, 1, 4, 1, 1, 2, 1], order = [3, 4, 5, 6, 0, 1, 2]}>
// CHECK-DAG: #[[$BLOCKED_RED:.+]] = #triton_gpu.blocked<{sizePerThread = [1, 1, 1, 16], threadsPerWarp = [16, 1, 1, 1], warpsPerCTA = [1, 1, 4, 2], order = [3, 0, 1, 2]}>
// CHECK-DAG: #[[$BLOCKED_TRANS:.+]] = #triton_gpu.blocked<{sizePerThread = [1, 1, 1, 16, 1], threadsPerWarp = [16, 1, 1, 1, 1], warpsPerCTA = [1, 1, 4, 1, 2], order = [3, 4, 0, 1, 2]}>
// CHECK-DAG: #[[$BLOCKED_FINAL:.+]] = #triton_gpu.blocked<{sizePerThread = [16, 1, 1, 1], threadsPerWarp = [1, 1, 1, 16], warpsPerCTA = [1, 1, 4, 2], order = [3, 0, 1, 2]}>

#mma = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [4, 2]}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, triton_gpu.target = "xpu", "triton_gpu.threads-per-warp" = 16 : i32} {
// CHECK:         tt.func @test(
// CHECK-SAME:                  %[[VAL_0:.*]]: tensor<128x64xf32, #[[$DPAS]]>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #[[$DPAS]]}>> {
// CHECK:           %[[VAL_1:.*]] = tt.reshape %[[VAL_0]] {allow_reorder = true, efficient_layout} : tensor<128x64xf32, #[[$DPAS]]> -> tensor<16x2x4x16x2x2x1xf32, #[[$BLOCKED_EW]]>
// CHECK:           %[[VAL_2:.*]] = "tt.reduce"(%[[VAL_1]]) <{axis = 6 : i32}> ({
// CHECK:           ^bb0(%[[VAL_3:.*]]: f32, %[[VAL_4:.*]]: f32):
// CHECK:             %[[VAL_5:.*]] = arith.maxnumf %[[VAL_3]], %[[VAL_4]] : f32
// CHECK:             tt.reduce.return %[[VAL_5]] : f32
// CHECK:           }) : (tensor<16x2x4x16x2x2x1xf32, #[[$BLOCKED_EW]]>) -> tensor<16x2x4x16x2x2xf32, #triton_gpu.slice<{dim = 6, parent = #[[$BLOCKED_EW]]}>>
// CHECK:           %[[VAL_6:.*]] = "tt.reduce"(%[[VAL_2]]) <{axis = 4 : i32}> ({
// CHECK:           ^bb0(%[[VAL_7:.*]]: f32, %[[VAL_8:.*]]: f32):
// CHECK:             %[[VAL_9:.*]] = arith.maxnumf %[[VAL_7]], %[[VAL_8]] : f32
// CHECK:             tt.reduce.return %[[VAL_9]] : f32
// CHECK:           }) : (tensor<16x2x4x16x2x2xf32, #triton_gpu.slice<{dim = 6, parent = #[[$BLOCKED_EW]]}>>) -> tensor<16x2x4x16x2xf32, #triton_gpu.slice<{dim = 4, parent = #triton_gpu.slice<{dim = 6, parent = #[[$BLOCKED_EW]]}>}>>
// CHECK:           %[[CONV:.*]] = triton_gpu.convert_layout %[[VAL_6]] : tensor<16x2x4x16x2xf32, #triton_gpu.slice<{dim = 4, parent = #triton_gpu.slice<{dim = 6, parent = #[[$BLOCKED_EW]]}>}>> -> tensor<16x2x4x16x2xf32, #[[$BLOCKED_TRANS]]>
// CHECK:           %[[VAL_10:.*]] = tt.reshape %[[CONV]] {allow_reorder = true, efficient_layout} : tensor<16x2x4x16x2xf32, #[[$BLOCKED_TRANS]]> -> tensor<16x2x4x32xf32, #[[$BLOCKED_RED]]>
// CHECK:           %[[VAL_11:.*]] = "tt.reduce"(%[[VAL_10]]) <{axis = 3 : i32}> ({
// CHECK:           ^bb0(%[[VAL_12:.*]]: f32, %[[VAL_13:.*]]: f32):
// CHECK:             %[[VAL_14:.*]] = arith.maxnumf %[[VAL_12]], %[[VAL_13]] : f32
// CHECK:             tt.reduce.return %[[VAL_14]] : f32
// CHECK:           }) : (tensor<16x2x4x32xf32, #[[$BLOCKED_RED]]>) -> tensor<16x2x4xf32, #triton_gpu.slice<{dim = 3, parent = #[[$BLOCKED_RED]]}>>
// CHECK:           %[[VAL_15:.*]] = triton_gpu.convert_layout %[[VAL_11]] : tensor<16x2x4xf32, #triton_gpu.slice<{dim = 3, parent = #[[$BLOCKED_RED]]}>> -> tensor<16x2x4xf32, #triton_gpu.slice<{dim = 3, parent = #[[$BLOCKED_FINAL]]}>>
// CHECK:           %[[VAL_16:.*]] = tt.reshape %[[VAL_15]] {allow_reorder = true, efficient_layout} : tensor<16x2x4xf32, #triton_gpu.slice<{dim = 3, parent = #[[$BLOCKED_FINAL]]}>> -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #[[$DPAS]]}>>
// CHECK:           tt.return %[[VAL_16]] : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #[[$DPAS]]}>>
// CHECK:         }
  tt.func @test(%arg0: tensor<128x64xf32, #mma>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>> {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.maxnumf %arg1, %arg2 : f32
      tt.reduce.return %1 : f32
    }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    tt.return %0 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
  }

// CHECK:         tt.func @test_repeat_layout(
// CHECK-SAME:                                %[[VAL_0:.*]]: tensor<256x64xf32, #[[$DPAS]]>) -> tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #[[$DPAS]]}>> {
// CHECK:           %[[VAL_1:.*]] = tt.reshape %[[VAL_0]] {allow_reorder = true, efficient_layout} : tensor<256x64xf32, #[[$DPAS]]> -> tensor<16x2x8x16x2x2x1xf32, #[[$BLOCKED_EW]]>
// CHECK:           %[[VAL_2:.*]] = "tt.reduce"(%[[VAL_1]]) <{axis = 6 : i32}> ({
// CHECK:           ^bb0(%[[VAL_3:.*]]: f32, %[[VAL_4:.*]]: f32):
// CHECK:             %[[VAL_5:.*]] = arith.maxnumf %[[VAL_3]], %[[VAL_4]] : f32
// CHECK:             tt.reduce.return %[[VAL_5]] : f32
// CHECK:           }) : (tensor<16x2x8x16x2x2x1xf32, #[[$BLOCKED_EW]]>) -> tensor<16x2x8x16x2x2xf32, #triton_gpu.slice<{dim = 6, parent = #[[$BLOCKED_EW]]}>>
// CHECK:           %[[VAL_6:.*]] = "tt.reduce"(%[[VAL_2]]) <{axis = 4 : i32}> ({
// CHECK:           ^bb0(%[[VAL_7:.*]]: f32, %[[VAL_8:.*]]: f32):
// CHECK:             %[[VAL_9:.*]] = arith.maxnumf %[[VAL_7]], %[[VAL_8]] : f32
// CHECK:             tt.reduce.return %[[VAL_9]] : f32
// CHECK:           }) : (tensor<16x2x8x16x2x2xf32, #triton_gpu.slice<{dim = 6, parent = #[[$BLOCKED_EW]]}>>) -> tensor<16x2x8x16x2xf32, #triton_gpu.slice<{dim = 4, parent = #triton_gpu.slice<{dim = 6, parent = #[[$BLOCKED_EW]]}>}>>
// CHECK:           %[[CONV:.*]] = triton_gpu.convert_layout %[[VAL_6]] : tensor<16x2x8x16x2xf32, #triton_gpu.slice<{dim = 4, parent = #triton_gpu.slice<{dim = 6, parent = #[[$BLOCKED_EW]]}>}>> -> tensor<16x2x8x16x2xf32, #[[$BLOCKED_TRANS]]>
// CHECK:           %[[VAL_10:.*]] = tt.reshape %[[CONV]] {allow_reorder = true, efficient_layout} : tensor<16x2x8x16x2xf32, #[[$BLOCKED_TRANS]]> -> tensor<16x2x8x32xf32, #[[$BLOCKED_RED]]>
// CHECK:           %[[VAL_11:.*]] = "tt.reduce"(%[[VAL_10]]) <{axis = 3 : i32}> ({
// CHECK:           ^bb0(%[[VAL_12:.*]]: f32, %[[VAL_13:.*]]: f32):
// CHECK:             %[[VAL_14:.*]] = arith.maxnumf %[[VAL_12]], %[[VAL_13]] : f32
// CHECK:             tt.reduce.return %[[VAL_14]] : f32
// CHECK:           }) : (tensor<16x2x8x32xf32, #[[$BLOCKED_RED]]>) -> tensor<16x2x8xf32, #triton_gpu.slice<{dim = 3, parent = #[[$BLOCKED_RED]]}>>
// CHECK:           %[[VAL_15:.*]] = triton_gpu.convert_layout %[[VAL_11]] : tensor<16x2x8xf32, #triton_gpu.slice<{dim = 3, parent = #[[$BLOCKED_RED]]}>> -> tensor<16x2x8xf32, #triton_gpu.slice<{dim = 3, parent = #[[$BLOCKED_FINAL]]}>>
// CHECK:           %[[VAL_16:.*]] = tt.reshape %[[VAL_15]] {allow_reorder = true, efficient_layout} : tensor<16x2x8xf32, #triton_gpu.slice<{dim = 3, parent = #[[$BLOCKED_FINAL]]}>> -> tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #[[$DPAS]]}>>
// CHECK:           tt.return %[[VAL_16]] : tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #[[$DPAS]]}>>
// CHECK:         }
  tt.func @test_repeat_layout(%arg0: tensor<256x64xf32, #mma>) -> tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>> {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.maxnumf %arg1, %arg2 : f32
      tt.reduce.return %1 : f32
    }) : (tensor<256x64xf32, #mma>) -> tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    tt.return %0 : tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
  }
}
