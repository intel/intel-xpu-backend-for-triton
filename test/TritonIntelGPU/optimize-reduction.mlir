// RUN: triton-opt %s --split-input-file -tritonintelgpu-optimize-reduction-locality | FileCheck %s

// Test reduction in a single warp (16x16->16).

// CHECK: #[[$ATTR_0:.+]] = #ttg.blocked<{sizePerThread = [1, 8, 1, 2, 1, 1, 1], threadsPerWarp = [16, 1, 1, 1, 1, 1, 1], warpsPerCTA = [1, 1, 1, 1, 1, 1, 1], order = [0, 1, 2, 3, 4, 5, 6]}>
// CHECK: #[[$ATTR_1:.+]] = #ttg.blocked<{sizePerThread = [16, 1, 1, 1, 1], threadsPerWarp = [1, 8, 2, 1, 1], warpsPerCTA = [1, 1, 1, 1, 1], order = [0, 1, 2, 3, 4]}>
// CHECK: #[[$ATTR_2:.+]] = #ttg.blocked<{sizePerThread = [16, 1, 1, 1], threadsPerWarp = [1, 16, 1, 1], warpsPerCTA = [1, 1, 1, 1], order = [0, 1, 2, 3]}>
// CHECK: #[[$ATTR_3:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [0, 1]}>
// CHECK: #[[$ATTR_4:.+]] = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [2, 1], A = [16, 8], B = [8, 16], C = [16, 16]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [2, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {

// CHECK-LABEL:   tt.func @test_single(
// CHECK-SAME:                         %[[VAL_0:.*]]: tensor<16x16xf32, #[[$ATTR_4]]>) -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #[[$ATTR_4]]}>> {
// CHECK:           %[[VAL_1:.*]] = tt.reshape %[[VAL_0]] allow_reorder efficient_layout : tensor<16x16xf32, #[[$ATTR_4]]> -> tensor<16x8x1x2x1x1x1xf32, #[[$ATTR_0]]>
// CHECK:           %[[VAL_2:.*]] = "tt.reduce"(%[[VAL_1]]) <{axis = 2 : i32}> ({
// CHECK:           ^bb0(%[[VAL_3:.*]]: f32, %[[VAL_4:.*]]: f32):
// CHECK:             %[[VAL_5:.*]] = arith.addf %[[VAL_3]], %[[VAL_4]] : f32
// CHECK:             tt.reduce.return %[[VAL_5]] : f32
// CHECK:           }) : (tensor<16x8x1x2x1x1x1xf32, #[[$ATTR_0]]>) -> tensor<16x8x2x1x1x1xf32, #ttg.slice<{dim = 2, parent = #[[$ATTR_0]]}>>
// CHECK:           %[[VAL_6:.*]] = "tt.reduce"(%[[VAL_2]]) <{axis = 4 : i32}> ({
// CHECK:           ^bb0(%[[VAL_7:.*]]: f32, %[[VAL_8:.*]]: f32):
// CHECK:             %[[VAL_9:.*]] = arith.addf %[[VAL_7]], %[[VAL_8]] : f32
// CHECK:             tt.reduce.return %[[VAL_9]] : f32
// CHECK:           }) : (tensor<16x8x2x1x1x1xf32, #ttg.slice<{dim = 2, parent = #[[$ATTR_0]]}>>) -> tensor<16x8x2x1x1xf32, #ttg.slice<{dim = 4, parent = #ttg.slice<{dim = 2, parent = #[[$ATTR_0]]}>}>>
// CHECK:           %[[VAL_10:.*]] = ttg.convert_layout %[[VAL_6]] : tensor<16x8x2x1x1xf32, #ttg.slice<{dim = 4, parent = #ttg.slice<{dim = 2, parent = #[[$ATTR_0]]}>}>> -> tensor<16x8x2x1x1xf32, #[[$ATTR_1]]>
// CHECK:           %[[VAL_11:.*]] = tt.reshape %[[VAL_10]] allow_reorder efficient_layout : tensor<16x8x2x1x1xf32, #[[$ATTR_1]]> -> tensor<16x16x1x1xf32, #[[$ATTR_2]]>
// CHECK:           %[[VAL_12:.*]] = "tt.reduce"(%[[VAL_11]]) <{axis = 0 : i32}> ({
// CHECK:           ^bb0(%[[VAL_13:.*]]: f32, %[[VAL_14:.*]]: f32):
// CHECK:             %[[VAL_15:.*]] = arith.addf %[[VAL_13]], %[[VAL_14]] : f32
// CHECK:             tt.reduce.return %[[VAL_15]] : f32
// CHECK:           }) : (tensor<16x16x1x1xf32, #[[$ATTR_2]]>) -> tensor<16x1x1xf32, #ttg.slice<{dim = 0, parent = #[[$ATTR_2]]}>>
// CHECK:           %[[VAL_16:.*]] = "tt.reduce"(%[[VAL_12]]) <{axis = 1 : i32}> ({
// CHECK:           ^bb0(%[[VAL_17:.*]]: f32, %[[VAL_18:.*]]: f32):
// CHECK:             %[[VAL_19:.*]] = arith.addf %[[VAL_17]], %[[VAL_18]] : f32
// CHECK:             tt.reduce.return %[[VAL_19]] : f32
// CHECK:           }) : (tensor<16x1x1xf32, #ttg.slice<{dim = 0, parent = #[[$ATTR_2]]}>>) -> tensor<16x1xf32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 0, parent = #[[$ATTR_2]]}>}>>
// CHECK:           %[[VAL_20:.*]] = tt.reshape %[[VAL_16]] allow_reorder efficient_layout : tensor<16x1xf32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 0, parent = #[[$ATTR_2]]}>}>> -> tensor<16xf32, #ttg.slice<{dim = 0, parent = #[[$ATTR_3]]}>>
// CHECK:           %[[VAL_21:.*]] = ttg.convert_layout %[[VAL_20]] : tensor<16xf32, #ttg.slice<{dim = 0, parent = #[[$ATTR_3]]}>> -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #[[$ATTR_4]]}>>
// CHECK:           tt.return %[[VAL_21]] : tensor<16xf32, #ttg.slice<{dim = 1, parent = #[[$ATTR_4]]}>>
// CHECK:         }
  tt.func @test_single(%arg0: tensor<16x16xf32, #mma>) -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #mma}>> {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      tt.reduce.return %1 : f32
    }) : (tensor<16x16xf32, #mma>) -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    tt.return %0 : tensor<16xf32, #ttg.slice<{dim = 1, parent = #mma}>>
  }
}

// -----

// Test reduction in two warps across the non-reduction dimension (32x16->32).

// CHECK: #[[$ATTR_5:.+]] = #ttg.blocked<{sizePerThread = [1, 8, 1, 2, 1, 1, 1], threadsPerWarp = [16, 1, 1, 1, 1, 1, 1], warpsPerCTA = [1, 1, 1, 1, 1, 1, 2], order = [0, 1, 2, 3, 4, 5, 6]}>
// CHECK: #[[$ATTR_6:.+]] = #ttg.blocked<{sizePerThread = [16, 1, 1, 1, 1], threadsPerWarp = [1, 8, 2, 1, 1], warpsPerCTA = [1, 1, 1, 1, 2], order = [0, 1, 2, 3, 4]}>
// CHECK: #[[$ATTR_7:.+]] = #ttg.blocked<{sizePerThread = [16, 1, 1, 1], threadsPerWarp = [1, 16, 1, 1], warpsPerCTA = [1, 1, 1, 2], order = [0, 1, 2, 3]}>
// CHECK: #[[$ATTR_8:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 2], order = [0, 1]}>
// CHECK: #[[$ATTR_9:.+]] = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [2, 1], repCluster = [2, 1], A = [16, 8], B = [8, 16], C = [16, 16]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [2, 1], repCluster = [2, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 16 : i32} {

// CHECK-LABEL:   tt.func @test_single_twice(
// CHECK-SAME:                               %[[VAL_0:.*]]: tensor<32x16xf32, #[[$ATTR_9]]>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #[[$ATTR_9]]}>> {
// CHECK:           %[[VAL_1:.*]] = tt.reshape %[[VAL_0]] allow_reorder efficient_layout : tensor<32x16xf32, #[[$ATTR_9]]> -> tensor<16x8x1x2x1x1x2xf32, #[[$ATTR_5]]>
// CHECK:           %[[VAL_2:.*]] = "tt.reduce"(%[[VAL_1]]) <{axis = 2 : i32}> ({
// CHECK:           ^bb0(%[[VAL_3:.*]]: f32, %[[VAL_4:.*]]: f32):
// CHECK:             %[[VAL_5:.*]] = arith.addf %[[VAL_3]], %[[VAL_4]] : f32
// CHECK:             tt.reduce.return %[[VAL_5]] : f32
// CHECK:           }) : (tensor<16x8x1x2x1x1x2xf32, #[[$ATTR_5]]>) -> tensor<16x8x2x1x1x2xf32, #ttg.slice<{dim = 2, parent = #[[$ATTR_5]]}>>
// CHECK:           %[[VAL_6:.*]] = "tt.reduce"(%[[VAL_2]]) <{axis = 4 : i32}> ({
// CHECK:           ^bb0(%[[VAL_7:.*]]: f32, %[[VAL_8:.*]]: f32):
// CHECK:             %[[VAL_9:.*]] = arith.addf %[[VAL_7]], %[[VAL_8]] : f32
// CHECK:             tt.reduce.return %[[VAL_9]] : f32
// CHECK:           }) : (tensor<16x8x2x1x1x2xf32, #ttg.slice<{dim = 2, parent = #[[$ATTR_5]]}>>) -> tensor<16x8x2x1x2xf32, #ttg.slice<{dim = 4, parent = #ttg.slice<{dim = 2, parent = #[[$ATTR_5]]}>}>>
// CHECK:           %[[VAL_10:.*]] = ttg.convert_layout %[[VAL_6]] : tensor<16x8x2x1x2xf32, #ttg.slice<{dim = 4, parent = #ttg.slice<{dim = 2, parent = #[[$ATTR_5]]}>}>> -> tensor<16x8x2x1x2xf32, #[[$ATTR_6]]>
// CHECK:           %[[VAL_11:.*]] = tt.reshape %[[VAL_10]] allow_reorder efficient_layout : tensor<16x8x2x1x2xf32, #[[$ATTR_6]]> -> tensor<16x16x1x2xf32, #[[$ATTR_7]]>
// CHECK:           %[[VAL_12:.*]] = "tt.reduce"(%[[VAL_11]]) <{axis = 0 : i32}> ({
// CHECK:           ^bb0(%[[VAL_13:.*]]: f32, %[[VAL_14:.*]]: f32):
// CHECK:             %[[VAL_15:.*]] = arith.addf %[[VAL_13]], %[[VAL_14]] : f32
// CHECK:             tt.reduce.return %[[VAL_15]] : f32
// CHECK:           }) : (tensor<16x16x1x2xf32, #[[$ATTR_7]]>) -> tensor<16x1x2xf32, #ttg.slice<{dim = 0, parent = #[[$ATTR_7]]}>>
// CHECK:           %[[VAL_16:.*]] = "tt.reduce"(%[[VAL_12]]) <{axis = 1 : i32}> ({
// CHECK:           ^bb0(%[[VAL_17:.*]]: f32, %[[VAL_18:.*]]: f32):
// CHECK:             %[[VAL_19:.*]] = arith.addf %[[VAL_17]], %[[VAL_18]] : f32
// CHECK:             tt.reduce.return %[[VAL_19]] : f32
// CHECK:           }) : (tensor<16x1x2xf32, #ttg.slice<{dim = 0, parent = #[[$ATTR_7]]}>>) -> tensor<16x2xf32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 0, parent = #[[$ATTR_7]]}>}>>
// CHECK:           %[[VAL_20:.*]] = tt.reshape %[[VAL_16]] allow_reorder efficient_layout : tensor<16x2xf32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 0, parent = #[[$ATTR_7]]}>}>> -> tensor<32xf32, #ttg.slice<{dim = 0, parent = #[[$ATTR_8]]}>>
// CHECK:           %[[VAL_21:.*]] = ttg.convert_layout %[[VAL_20]] : tensor<32xf32, #ttg.slice<{dim = 0, parent = #[[$ATTR_8]]}>> -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #[[$ATTR_9]]}>>
// CHECK:           tt.return %[[VAL_21]] : tensor<32xf32, #ttg.slice<{dim = 1, parent = #[[$ATTR_9]]}>>
// CHECK:         }
  tt.func @test_single_twice(%arg0: tensor<32x16xf32, #mma>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #mma}>> {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      tt.reduce.return %1 : f32
    }) : (tensor<32x16xf32, #mma>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    tt.return %0 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #mma}>>
  }
}

// -----

// Test reduction in two warps across the reduction dimension (16x32->16).

// CHECK: #[[$ATTR_10:.+]] = #ttg.blocked<{sizePerThread = [1, 8, 1, 2, 1, 1, 1], threadsPerWarp = [16, 1, 1, 1, 1, 1, 1], warpsPerCTA = [1, 1, 1, 1, 2, 1, 1], order = [0, 1, 2, 3, 4, 5, 6]}>
// CHECK: #[[$ATTR_11:.+]] = #ttg.blocked<{sizePerThread = [16, 1, 1, 1, 1], threadsPerWarp = [1, 8, 2, 1, 1], warpsPerCTA = [1, 1, 1, 2, 1], order = [0, 1, 2, 3, 4]}>
// CHECK: #[[$ATTR_12:.+]] = #ttg.blocked<{sizePerThread = [16, 1, 1, 1], threadsPerWarp = [1, 16, 1, 1], warpsPerCTA = [1, 1, 2, 1], order = [0, 1, 2, 3]}>
// CHECK: #[[$ATTR_13:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 1], order = [0, 1]}>
// CHECK: #[[$ATTR_14:.+]] = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [1, 2], repCluster = [2, 1], A = [16, 8], B = [8, 16], C = [16, 16]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [1, 2], repCluster = [2, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 16 : i32} {

// CHECK-LABEL:   tt.func @test_two_warps_red(
// CHECK-SAME:                                %[[VAL_0:.*]]: tensor<16x32xf32, #[[$ATTR_14]]>) -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #[[$ATTR_14]]}>> {
// CHECK:           %[[VAL_1:.*]] = tt.reshape %[[VAL_0]] allow_reorder efficient_layout : tensor<16x32xf32, #[[$ATTR_14]]> -> tensor<16x8x1x2x2x1x1xf32, #[[$ATTR_10]]>
// CHECK:           %[[VAL_2:.*]] = "tt.reduce"(%[[VAL_1]]) <{axis = 2 : i32}> ({
// CHECK:           ^bb0(%[[VAL_3:.*]]: f32, %[[VAL_4:.*]]: f32):
// CHECK:             %[[VAL_5:.*]] = arith.addf %[[VAL_3]], %[[VAL_4]] : f32
// CHECK:             tt.reduce.return %[[VAL_5]] : f32
// CHECK:           }) : (tensor<16x8x1x2x2x1x1xf32, #[[$ATTR_10]]>) -> tensor<16x8x2x2x1x1xf32, #ttg.slice<{dim = 2, parent = #[[$ATTR_10]]}>>
// CHECK:           %[[VAL_6:.*]] = "tt.reduce"(%[[VAL_2]]) <{axis = 4 : i32}> ({
// CHECK:           ^bb0(%[[VAL_7:.*]]: f32, %[[VAL_8:.*]]: f32):
// CHECK:             %[[VAL_9:.*]] = arith.addf %[[VAL_7]], %[[VAL_8]] : f32
// CHECK:             tt.reduce.return %[[VAL_9]] : f32
// CHECK:           }) : (tensor<16x8x2x2x1x1xf32, #ttg.slice<{dim = 2, parent = #[[$ATTR_10]]}>>) -> tensor<16x8x2x2x1xf32, #ttg.slice<{dim = 4, parent = #ttg.slice<{dim = 2, parent = #[[$ATTR_10]]}>}>>
// CHECK:           %[[VAL_10:.*]] = ttg.convert_layout %[[VAL_6]] : tensor<16x8x2x2x1xf32, #ttg.slice<{dim = 4, parent = #ttg.slice<{dim = 2, parent = #[[$ATTR_10]]}>}>> -> tensor<16x8x2x2x1xf32, #[[$ATTR_11]]>
// CHECK:           %[[VAL_11:.*]] = tt.reshape %[[VAL_10]] allow_reorder efficient_layout : tensor<16x8x2x2x1xf32, #[[$ATTR_11]]> -> tensor<16x16x2x1xf32, #[[$ATTR_12]]>
// CHECK:           %[[VAL_12:.*]] = "tt.reduce"(%[[VAL_11]]) <{axis = 0 : i32}> ({
// CHECK:           ^bb0(%[[VAL_13:.*]]: f32, %[[VAL_14:.*]]: f32):
// CHECK:             %[[VAL_15:.*]] = arith.addf %[[VAL_13]], %[[VAL_14]] : f32
// CHECK:             tt.reduce.return %[[VAL_15]] : f32
// CHECK:           }) : (tensor<16x16x2x1xf32, #[[$ATTR_12]]>) -> tensor<16x2x1xf32, #ttg.slice<{dim = 0, parent = #[[$ATTR_12]]}>>
// CHECK:           %[[VAL_16:.*]] = "tt.reduce"(%[[VAL_12]]) <{axis = 1 : i32}> ({
// CHECK:           ^bb0(%[[VAL_17:.*]]: f32, %[[VAL_18:.*]]: f32):
// CHECK:             %[[VAL_19:.*]] = arith.addf %[[VAL_17]], %[[VAL_18]] : f32
// CHECK:             tt.reduce.return %[[VAL_19]] : f32
// CHECK:           }) : (tensor<16x2x1xf32, #ttg.slice<{dim = 0, parent = #[[$ATTR_12]]}>>) -> tensor<16x1xf32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 0, parent = #[[$ATTR_12]]}>}>>
// CHECK:           %[[VAL_20:.*]] = tt.reshape %[[VAL_16]] allow_reorder efficient_layout : tensor<16x1xf32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 0, parent = #[[$ATTR_12]]}>}>> -> tensor<16xf32, #ttg.slice<{dim = 0, parent = #[[$ATTR_13]]}>>
// CHECK:           %[[VAL_21:.*]] = ttg.convert_layout %[[VAL_20]] : tensor<16xf32, #ttg.slice<{dim = 0, parent = #[[$ATTR_13]]}>> -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #[[$ATTR_14]]}>>
// CHECK:           tt.return %[[VAL_21]] : tensor<16xf32, #ttg.slice<{dim = 1, parent = #[[$ATTR_14]]}>>
// CHECK:         }
  tt.func @test_two_warps_red(%arg0: tensor<16x32xf32, #mma>) -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #mma}>> {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      tt.reduce.return %1 : f32
    }) : (tensor<16x32xf32, #mma>) -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    tt.return %0 : tensor<16xf32, #ttg.slice<{dim = 1, parent = #mma}>>
  }
}

// -----

// Test reduction in two warps across both dimensions (32x32->32).

// CHECK: #[[$ATTR_15:.+]] = #ttg.blocked<{sizePerThread = [1, 8, 1, 2, 1, 1, 1], threadsPerWarp = [16, 1, 1, 1, 1, 1, 1], warpsPerCTA = [1, 1, 1, 1, 2, 1, 2], order = [0, 1, 2, 3, 4, 5, 6]}>
// CHECK: #[[$ATTR_16:.+]] = #ttg.blocked<{sizePerThread = [16, 1, 1, 1, 1], threadsPerWarp = [1, 8, 2, 1, 1], warpsPerCTA = [1, 1, 1, 2, 2], order = [0, 1, 2, 3, 4]}>
// CHECK: #[[$ATTR_17:.+]] = #ttg.blocked<{sizePerThread = [16, 1, 1, 1], threadsPerWarp = [1, 16, 1, 1], warpsPerCTA = [1, 1, 2, 2], order = [0, 1, 2, 3]}>
// CHECK: #[[$ATTR_18:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [0, 1]}>
// CHECK: #[[$ATTR_19:.+]] = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [2, 1], A = [16, 8], B = [8, 16], C = [16, 16]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [2, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {

// CHECK-LABEL:   tt.func @test_two_warps(
// CHECK-SAME:                            %[[VAL_0:.*]]: tensor<32x32xf32, #[[$ATTR_19]]>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #[[$ATTR_19]]}>> {
// CHECK:           %[[VAL_1:.*]] = tt.reshape %[[VAL_0]] allow_reorder efficient_layout : tensor<32x32xf32, #[[$ATTR_19]]> -> tensor<16x8x1x2x2x1x2xf32, #[[$ATTR_15]]>
// CHECK:           %[[VAL_2:.*]] = "tt.reduce"(%[[VAL_1]]) <{axis = 2 : i32}> ({
// CHECK:           ^bb0(%[[VAL_3:.*]]: f32, %[[VAL_4:.*]]: f32):
// CHECK:             %[[VAL_5:.*]] = arith.addf %[[VAL_3]], %[[VAL_4]] : f32
// CHECK:             tt.reduce.return %[[VAL_5]] : f32
// CHECK:           }) : (tensor<16x8x1x2x2x1x2xf32, #[[$ATTR_15]]>) -> tensor<16x8x2x2x1x2xf32, #ttg.slice<{dim = 2, parent = #[[$ATTR_15]]}>>
// CHECK:           %[[VAL_6:.*]] = "tt.reduce"(%[[VAL_2]]) <{axis = 4 : i32}> ({
// CHECK:           ^bb0(%[[VAL_7:.*]]: f32, %[[VAL_8:.*]]: f32):
// CHECK:             %[[VAL_9:.*]] = arith.addf %[[VAL_7]], %[[VAL_8]] : f32
// CHECK:             tt.reduce.return %[[VAL_9]] : f32
// CHECK:           }) : (tensor<16x8x2x2x1x2xf32, #ttg.slice<{dim = 2, parent = #[[$ATTR_15]]}>>) -> tensor<16x8x2x2x2xf32, #ttg.slice<{dim = 4, parent = #ttg.slice<{dim = 2, parent = #[[$ATTR_15]]}>}>>
// CHECK:           %[[VAL_10:.*]] = ttg.convert_layout %[[VAL_6]] : tensor<16x8x2x2x2xf32, #ttg.slice<{dim = 4, parent = #ttg.slice<{dim = 2, parent = #[[$ATTR_15]]}>}>> -> tensor<16x8x2x2x2xf32, #[[$ATTR_16]]>
// CHECK:           %[[VAL_11:.*]] = tt.reshape %[[VAL_10]] allow_reorder efficient_layout : tensor<16x8x2x2x2xf32, #[[$ATTR_16]]> -> tensor<16x16x2x2xf32, #[[$ATTR_17]]>
// CHECK:           %[[VAL_12:.*]] = "tt.reduce"(%[[VAL_11]]) <{axis = 0 : i32}> ({
// CHECK:           ^bb0(%[[VAL_13:.*]]: f32, %[[VAL_14:.*]]: f32):
// CHECK:             %[[VAL_15:.*]] = arith.addf %[[VAL_13]], %[[VAL_14]] : f32
// CHECK:             tt.reduce.return %[[VAL_15]] : f32
// CHECK:           }) : (tensor<16x16x2x2xf32, #[[$ATTR_17]]>) -> tensor<16x2x2xf32, #ttg.slice<{dim = 0, parent = #[[$ATTR_17]]}>>
// CHECK:           %[[VAL_16:.*]] = "tt.reduce"(%[[VAL_12]]) <{axis = 1 : i32}> ({
// CHECK:           ^bb0(%[[VAL_17:.*]]: f32, %[[VAL_18:.*]]: f32):
// CHECK:             %[[VAL_19:.*]] = arith.addf %[[VAL_17]], %[[VAL_18]] : f32
// CHECK:             tt.reduce.return %[[VAL_19]] : f32
// CHECK:           }) : (tensor<16x2x2xf32, #ttg.slice<{dim = 0, parent = #[[$ATTR_17]]}>>) -> tensor<16x2xf32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 0, parent = #[[$ATTR_17]]}>}>>
// CHECK:           %[[VAL_20:.*]] = tt.reshape %[[VAL_16]] allow_reorder efficient_layout : tensor<16x2xf32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 0, parent = #[[$ATTR_17]]}>}>> -> tensor<32xf32, #ttg.slice<{dim = 0, parent = #[[$ATTR_18]]}>>
// CHECK:           %[[VAL_21:.*]] = ttg.convert_layout %[[VAL_20]] : tensor<32xf32, #ttg.slice<{dim = 0, parent = #[[$ATTR_18]]}>> -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #[[$ATTR_19]]}>>
// CHECK:           tt.return %[[VAL_21]] : tensor<32xf32, #ttg.slice<{dim = 1, parent = #[[$ATTR_19]]}>>
// CHECK:         }
  tt.func @test_two_warps(%arg0: tensor<32x32xf32, #mma>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #mma}>> {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      tt.reduce.return %1 : f32
    }) : (tensor<32x32xf32, #mma>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    tt.return %0 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #mma}>>
  }
}

// -----

// Test reduction across 2 warps in the reduction dimension and 4 in the non-reduction dimension.

// CHECK: #[[$ATTR_20:.+]] = #ttg.blocked<{sizePerThread = [1, 8, 2, 2, 1, 1, 1], threadsPerWarp = [16, 1, 1, 1, 1, 1, 1], warpsPerCTA = [1, 1, 1, 1, 2, 1, 4], order = [0, 1, 2, 3, 4, 5, 6]}>
// CHECK: #[[$ATTR_21:.+]] = #ttg.blocked<{sizePerThread = [16, 1, 1, 1, 1], threadsPerWarp = [1, 8, 2, 1, 1], warpsPerCTA = [1, 1, 1, 2, 4], order = [0, 1, 2, 3, 4]}>
// CHECK: #[[$ATTR_22:.+]] = #ttg.blocked<{sizePerThread = [16, 1, 1, 1], threadsPerWarp = [1, 16, 1, 1], warpsPerCTA = [1, 1, 2, 4], order = [0, 1, 2, 3]}>
// CHECK: #[[$ATTR_23:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [0, 1]}>
// CHECK: #[[$ATTR_24:.+]] = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [2, 2], A = [16, 8], B = [8, 32], C = [16, 32]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [2, 2]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
// CHECK-LABEL:   tt.func @test(
// CHECK-SAME:                  %[[VAL_0:.*]]: tensor<64x64xf32, #[[$ATTR_24]]>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #[[$ATTR_24]]}>> {
// CHECK:           %[[VAL_1:.*]] = tt.reshape %[[VAL_0]] allow_reorder efficient_layout : tensor<64x64xf32, #[[$ATTR_24]]> -> tensor<16x8x2x2x2x1x4xf32, #[[$ATTR_20]]>
// CHECK:           %[[VAL_2:.*]] = "tt.reduce"(%[[VAL_1]]) <{axis = 2 : i32}> ({
// CHECK:           ^bb0(%[[VAL_3:.*]]: f32, %[[VAL_4:.*]]: f32):
// CHECK:             %[[VAL_5:.*]] = arith.maxnumf %[[VAL_3]], %[[VAL_4]] : f32
// CHECK:             tt.reduce.return %[[VAL_5]] : f32
// CHECK:           }) : (tensor<16x8x2x2x2x1x4xf32, #[[$ATTR_20]]>) -> tensor<16x8x2x2x1x4xf32, #ttg.slice<{dim = 2, parent = #[[$ATTR_20]]}>>
// CHECK:           %[[VAL_6:.*]] = "tt.reduce"(%[[VAL_2]]) <{axis = 4 : i32}> ({
// CHECK:           ^bb0(%[[VAL_7:.*]]: f32, %[[VAL_8:.*]]: f32):
// CHECK:             %[[VAL_9:.*]] = arith.maxnumf %[[VAL_7]], %[[VAL_8]] : f32
// CHECK:             tt.reduce.return %[[VAL_9]] : f32
// CHECK:           }) : (tensor<16x8x2x2x1x4xf32, #ttg.slice<{dim = 2, parent = #[[$ATTR_20]]}>>) -> tensor<16x8x2x2x4xf32, #ttg.slice<{dim = 4, parent = #ttg.slice<{dim = 2, parent = #[[$ATTR_20]]}>}>>
// CHECK:           %[[VAL_10:.*]] = ttg.convert_layout %[[VAL_6]] : tensor<16x8x2x2x4xf32, #ttg.slice<{dim = 4, parent = #ttg.slice<{dim = 2, parent = #[[$ATTR_20]]}>}>> -> tensor<16x8x2x2x4xf32, #[[$ATTR_21]]>
// CHECK:           %[[VAL_11:.*]] = tt.reshape %[[VAL_10]] allow_reorder efficient_layout : tensor<16x8x2x2x4xf32, #[[$ATTR_21]]> -> tensor<16x16x2x4xf32, #[[$ATTR_22]]>
// CHECK:           %[[VAL_12:.*]] = "tt.reduce"(%[[VAL_11]]) <{axis = 0 : i32}> ({
// CHECK:           ^bb0(%[[VAL_13:.*]]: f32, %[[VAL_14:.*]]: f32):
// CHECK:             %[[VAL_15:.*]] = arith.maxnumf %[[VAL_13]], %[[VAL_14]] : f32
// CHECK:             tt.reduce.return %[[VAL_15]] : f32
// CHECK:           }) : (tensor<16x16x2x4xf32, #[[$ATTR_22]]>) -> tensor<16x2x4xf32, #ttg.slice<{dim = 0, parent = #[[$ATTR_22]]}>>
// CHECK:           %[[VAL_16:.*]] = "tt.reduce"(%[[VAL_12]]) <{axis = 1 : i32}> ({
// CHECK:           ^bb0(%[[VAL_17:.*]]: f32, %[[VAL_18:.*]]: f32):
// CHECK:             %[[VAL_19:.*]] = arith.maxnumf %[[VAL_17]], %[[VAL_18]] : f32
// CHECK:             tt.reduce.return %[[VAL_19]] : f32
// CHECK:           }) : (tensor<16x2x4xf32, #ttg.slice<{dim = 0, parent = #[[$ATTR_22]]}>>) -> tensor<16x4xf32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 0, parent = #[[$ATTR_22]]}>}>>
// CHECK:           %[[VAL_20:.*]] = tt.reshape %[[VAL_16]] allow_reorder efficient_layout : tensor<16x4xf32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 0, parent = #[[$ATTR_22]]}>}>> -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #[[$ATTR_23]]}>>
// CHECK:           %[[VAL_21:.*]] = ttg.convert_layout %[[VAL_20]] : tensor<64xf32, #ttg.slice<{dim = 0, parent = #[[$ATTR_23]]}>> -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #[[$ATTR_24]]}>>
// CHECK:           tt.return %[[VAL_21]] : tensor<64xf32, #ttg.slice<{dim = 1, parent = #[[$ATTR_24]]}>>
// CHECK:         }
  tt.func @test(%arg0: tensor<64x64xf32, #mma>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>> {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.maxnumf %arg1, %arg2 : f32
      tt.reduce.return %1 : f32
    }) : (tensor<64x64xf32, #mma>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    tt.return %0 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
  }
}

// -----

// Test reduction across 2 warps in the reduction dimension and 4 in the non-reduction dimension with repCluster[0] = 4.

// CHECK: #[[$ATTR_25:.+]] = #ttg.blocked<{sizePerThread = [1, 8, 2, 4, 1, 1, 1], threadsPerWarp = [16, 1, 1, 1, 1, 1, 1], warpsPerCTA = [1, 1, 1, 1, 2, 1, 4], order = [0, 1, 2, 3, 4, 5, 6]}>
// CHECK: #[[$ATTR_26:.+]] = #ttg.blocked<{sizePerThread = [16, 2, 1, 1, 1], threadsPerWarp = [1, 4, 4, 1, 1], warpsPerCTA = [1, 1, 1, 2, 4], order = [0, 1, 2, 3, 4]}>
// CHECK: #[[$ATTR_27:.+]] = #ttg.blocked<{sizePerThread = [16, 2, 1, 1], threadsPerWarp = [1, 16, 1, 1], warpsPerCTA = [1, 1, 2, 4], order = [0, 1, 2, 3]}>
// CHECK: #[[$ATTR_28:.+]] = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [0, 1]}>
// CHECK: #[[$ATTR_29:.+]] = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [4, 2], A = [32, 8], B = [8, 32], C = [32, 32]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [4, 2]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
// CHECK-LABEL:   tt.func @test(
// CHECK-SAME:                  %[[VAL_0:.*]]: tensor<128x64xf32, #[[$ATTR_29]]>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #[[$ATTR_29]]}>> {
// CHECK:           %[[VAL_1:.*]] = tt.reshape %[[VAL_0]] allow_reorder efficient_layout : tensor<128x64xf32, #[[$ATTR_29]]> -> tensor<16x8x2x4x2x1x4xf32, #[[$ATTR_25]]>
// CHECK:           %[[VAL_2:.*]] = "tt.reduce"(%[[VAL_1]]) <{axis = 2 : i32}> ({
// CHECK:           ^bb0(%[[VAL_3:.*]]: f32, %[[VAL_4:.*]]: f32):
// CHECK:             %[[VAL_5:.*]] = arith.maxnumf %[[VAL_3]], %[[VAL_4]] : f32
// CHECK:             tt.reduce.return %[[VAL_5]] : f32
// CHECK:           }) : (tensor<16x8x2x4x2x1x4xf32, #[[$ATTR_25]]>) -> tensor<16x8x4x2x1x4xf32, #ttg.slice<{dim = 2, parent = #[[$ATTR_25]]}>>
// CHECK:           %[[VAL_6:.*]] = "tt.reduce"(%[[VAL_2]]) <{axis = 4 : i32}> ({
// CHECK:           ^bb0(%[[VAL_7:.*]]: f32, %[[VAL_8:.*]]: f32):
// CHECK:             %[[VAL_9:.*]] = arith.maxnumf %[[VAL_7]], %[[VAL_8]] : f32
// CHECK:             tt.reduce.return %[[VAL_9]] : f32
// CHECK:           }) : (tensor<16x8x4x2x1x4xf32, #ttg.slice<{dim = 2, parent = #[[$ATTR_25]]}>>) -> tensor<16x8x4x2x4xf32, #ttg.slice<{dim = 4, parent = #ttg.slice<{dim = 2, parent = #[[$ATTR_25]]}>}>>
// CHECK:           %[[VAL_10:.*]] = ttg.convert_layout %[[VAL_6]] : tensor<16x8x4x2x4xf32, #ttg.slice<{dim = 4, parent = #ttg.slice<{dim = 2, parent = #[[$ATTR_25]]}>}>> -> tensor<16x8x4x2x4xf32, #[[$ATTR_26]]>
// CHECK:           %[[VAL_11:.*]] = tt.reshape %[[VAL_10]] allow_reorder efficient_layout : tensor<16x8x4x2x4xf32, #[[$ATTR_26]]> -> tensor<16x32x2x4xf32, #[[$ATTR_27]]>
// CHECK:           %[[VAL_12:.*]] = "tt.reduce"(%[[VAL_11]]) <{axis = 0 : i32}> ({
// CHECK:           ^bb0(%[[VAL_13:.*]]: f32, %[[VAL_14:.*]]: f32):
// CHECK:             %[[VAL_15:.*]] = arith.maxnumf %[[VAL_13]], %[[VAL_14]] : f32
// CHECK:             tt.reduce.return %[[VAL_15]] : f32
// CHECK:           }) : (tensor<16x32x2x4xf32, #[[$ATTR_27]]>) -> tensor<32x2x4xf32, #ttg.slice<{dim = 0, parent = #[[$ATTR_27]]}>>
// CHECK:           %[[VAL_16:.*]] = "tt.reduce"(%[[VAL_12]]) <{axis = 1 : i32}> ({
// CHECK:           ^bb0(%[[VAL_17:.*]]: f32, %[[VAL_18:.*]]: f32):
// CHECK:             %[[VAL_19:.*]] = arith.maxnumf %[[VAL_17]], %[[VAL_18]] : f32
// CHECK:             tt.reduce.return %[[VAL_19]] : f32
// CHECK:           }) : (tensor<32x2x4xf32, #ttg.slice<{dim = 0, parent = #[[$ATTR_27]]}>>) -> tensor<32x4xf32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 0, parent = #[[$ATTR_27]]}>}>>
// CHECK:           %[[VAL_20:.*]] = tt.reshape %[[VAL_16]] allow_reorder efficient_layout : tensor<32x4xf32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 0, parent = #[[$ATTR_27]]}>}>> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #[[$ATTR_28]]}>>
// CHECK:           %[[VAL_21:.*]] = ttg.convert_layout %[[VAL_20]] : tensor<128xf32, #ttg.slice<{dim = 0, parent = #[[$ATTR_28]]}>> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #[[$ATTR_29]]}>>
// CHECK:           tt.return %[[VAL_21]] : tensor<128xf32, #ttg.slice<{dim = 1, parent = #[[$ATTR_29]]}>>
// CHECK:         }
  tt.func @test(%arg0: tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.maxnumf %arg1, %arg2 : f32
      tt.reduce.return %1 : f32
    }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    tt.return %0 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
  }
}

// -----

// Test invalid reduction in two warps across the reduction dimension (16x16->16).
// The number of elements in the reduction dimension is not enough to be covered by the encoding.

// CHECK: #[[$ATTR_31:.+]] = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [1, 2], repCluster = [2, 1], A = [16, 8], B = [8, 16], C = [16, 16]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [1, 2], repCluster = [2, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 16 : i32} {

// CHECK-LABEL:   tt.func @test_invalid_two_warps_red(
// CHECK-SAME:                                %[[VAL_0:.*]]: tensor<16x16xf32, #[[$ATTR_31]]>) -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #[[$ATTR_31]]}>> {
// CHECK:           %[[VAL_1:.*]] = "tt.reduce"(%[[VAL_0]]) <{axis = 1 : i32}> ({
// CHECK:           ^bb0(%[[VAL_2:.*]]: f32, %[[VAL_3:.*]]: f32):
// CHECK:             %[[VAL_4:.*]] = arith.addf %[[VAL_2]], %[[VAL_3]] : f32
// CHECK:             tt.reduce.return %[[VAL_4]] : f32
// CHECK:           }) : (tensor<16x16xf32, #[[$ATTR_31]]>) -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #[[$ATTR_31]]}>>
// CHECK:           tt.return %[[VAL_1]] : tensor<16xf32, #ttg.slice<{dim = 1, parent = #[[$ATTR_31]]}>>
// CHECK:         }
  tt.func @test_invalid_two_warps_red(%arg0: tensor<16x16xf32, #mma>) -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #mma}>> {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      tt.reduce.return %1 : f32
    }) : (tensor<16x16xf32, #mma>) -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    tt.return %0 : tensor<16xf32, #ttg.slice<{dim = 1, parent = #mma}>>
  }
}
