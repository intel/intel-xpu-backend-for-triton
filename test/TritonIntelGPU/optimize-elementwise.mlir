// RUN: triton-opt %s --split-input-file -tritonintelgpu-optimize-elementwise-parallelism | FileCheck %s

// CHECK: #[[$ATTR_0:.+]] = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [1], order = [0]}>
// CHECK: #[[$ATTR_1:.+]] = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [2, 2], A = [16, 16], B = [16, 32], C = [16, 32]}>

#mma = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [2, 2], A = [16, 16], B = [16, 32], C = [16, 32]}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
// CHECK-LABEL:   tt.func @test_dpas(
// CHECK-SAME:                       %[[VAL_0:.*]]: tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>>,
// CHECK-SAME:                       %[[VAL_1:.*]]: tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>>)
  tt.func @test_dpas(%arg0: tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>, %arg1: tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>) -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>> {
// CHECK:           %[[VAL_2:.*]] = triton_gpu.convert_layout %[[VAL_0]] : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>> -> tensor<16xf32, #[[$ATTR_0]]>
// CHECK:           %[[VAL_3:.*]] = triton_gpu.convert_layout %[[VAL_1]] : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>> -> tensor<16xf32, #[[$ATTR_0]]>
// CHECK:           %[[VAL_4:.*]] = arith.addf %[[VAL_2]], %[[VAL_3]] : tensor<16xf32, #[[$ATTR_0]]>
    %0 = arith.addf %arg0, %arg1 : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
// CHECK:           %[[VAL_5:.*]] = triton_gpu.convert_layout %[[VAL_4]] : tensor<16xf32, #[[$ATTR_0]]> -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>>
// CHECK:           tt.return %[[VAL_5]] : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>>
    tt.return %0 : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
  }
}

// -----

// CHECK: #[[$ATTR_0:.+]] = #triton_gpu.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [0, 1]}>
// CHECK: #[[$ATTR_1:.+]] = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [1], order = [0]}>

#blocked = #triton_gpu.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [0, 1]}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
// CHECK-LABEL:   tt.func @test_blocked(
// CHECK-SAME:                          %[[VAL_0:.*]]: tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_0]]}>>,
// CHECK-SAME:                          %[[VAL_1:.*]]: tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_0]]}>>)
  tt.func @test_blocked(%arg0: tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>, %arg1: tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> {
// CHECK:           %[[VAL_2:.*]] = triton_gpu.convert_layout %[[VAL_0]] : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_0]]}>> -> tensor<16xf32, #[[$ATTR_1]]>
// CHECK:           %[[VAL_3:.*]] = triton_gpu.convert_layout %[[VAL_1]] : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_0]]}>> -> tensor<16xf32, #[[$ATTR_1]]>
// CHECK:           %[[VAL_4:.*]] = arith.addf %[[VAL_2]], %[[VAL_3]] : tensor<16xf32, #[[$ATTR_1]]>
    %0 = arith.addf %arg0, %arg1 : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
// CHECK:           %[[VAL_5:.*]] = triton_gpu.convert_layout %[[VAL_4]] : tensor<16xf32, #[[$ATTR_1]]> -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_0]]}>>
// CHECK:           tt.return %[[VAL_5]] : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_0]]}>>
    tt.return %0 : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
  }
}

// -----

// CHECK: #[[$ATTR_0:.+]] = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [0, 1]}>
// CHECK: #[[$ATTR_1:.+]] = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [1], order = [0]}>

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [0, 1]}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
// CHECK-LABEL:   tt.func @test_blocked_repeat(
// CHECK-SAME:                                 %[[VAL_0:.*]]: tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_0]]}>>,
// CHECK-SAME:                                 %[[VAL_1:.*]]: tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_0]]}>>)
  tt.func @test_blocked_repeat(%arg0: tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>, %arg1: tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> {
// CHECK:           %[[VAL_2:.*]] = triton_gpu.convert_layout %[[VAL_0]] : tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_0]]}>> -> tensor<64xf32, #[[$ATTR_1]]>
// CHECK:           %[[VAL_3:.*]] = triton_gpu.convert_layout %[[VAL_1]] : tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_0]]}>> -> tensor<64xf32, #[[$ATTR_1]]>
// CHECK:           %[[VAL_4:.*]] = arith.addf %[[VAL_2]], %[[VAL_3]] : tensor<64xf32, #[[$ATTR_1]]>
    %0 = arith.addf %arg0, %arg1 : tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
// CHECK:           %[[VAL_5:.*]] = triton_gpu.convert_layout %[[VAL_4]] : tensor<64xf32, #[[$ATTR_1]]> -> tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_0]]}>>
// CHECK:           tt.return %[[VAL_5]] : tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_0]]}>>
    tt.return %0 : tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
  }
}

// -----

// CHECK: #[[$ATTR_0:.+]] = #triton_gpu.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 1], order = [0, 1]}>
// CHECK: #[[$ATTR_1:.+]] = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [2], order = [0]}>

#blocked = #triton_gpu.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 1], order = [0, 1]}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 2 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
// CHECK-LABEL:   tt.func @test_blocked_multi_warp(
// CHECK-SAME:                                     %[[VAL_0:.*]]: tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_0]]}>>,
// CHECK-SAME:                                     %[[VAL_1:.*]]: tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_0]]}>>) -> tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_0]]}>> {
  tt.func @test_blocked_multi_warp(%arg0: tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>, %arg1: tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> {
// CHECK:           %[[VAL_2:.*]] = triton_gpu.convert_layout %[[VAL_0]] : tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_0]]}>> -> tensor<32xf32, #[[$ATTR_1]]>
// CHECK:           %[[VAL_3:.*]] = triton_gpu.convert_layout %[[VAL_1]] : tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_0]]}>> -> tensor<32xf32, #[[$ATTR_1]]>
// CHECK:           %[[VAL_4:.*]] = arith.addf %[[VAL_2]], %[[VAL_3]] : tensor<32xf32, #[[$ATTR_1]]>
    %0 = arith.addf %arg0, %arg1 : tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
// CHECK:           %[[VAL_5:.*]] = triton_gpu.convert_layout %[[VAL_4]] : tensor<32xf32, #[[$ATTR_1]]> -> tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_0]]}>>
// CHECK:           tt.return %[[VAL_5]] : tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_0]]}>>
    tt.return %0 : tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
  }
}

// -----

// CHECK: #[[$ATTR_0:.+]] = #triton_gpu.blocked<{sizePerThread = [32, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [0, 1]}>
// CHECK: #[[$ATTR_1:.+]] = #triton_gpu.blocked<{sizePerThread = [2], threadsPerWarp = [16], warpsPerCTA = [4], order = [0]}>

#blocked = #triton_gpu.blocked<{sizePerThread = [32, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [0, 1]}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
// CHECK-LABEL:   tt.func @test_blocked_multi_warp_double_stride(
// CHECK-SAME:                                                   %[[VAL_0:.*]]: tensor<128xf16, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_0]]}>>,
// CHECK-SAME:                                                   %[[VAL_1:.*]]: tensor<128xf16, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_0]]}>>) -> tensor<128xf16, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_0]]}>> {
  tt.func @test_blocked_multi_warp_double_stride(%arg0: tensor<128xf16, #triton_gpu.slice<{dim = 1, parent = #blocked}>>, %arg1: tensor<128xf16, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<128xf16, #triton_gpu.slice<{dim = 1, parent = #blocked}>> {
// CHECK:           %[[VAL_2:.*]] = triton_gpu.convert_layout %[[VAL_0]] : tensor<128xf16, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_0]]}>> -> tensor<128xf16, #[[$ATTR_1]]>
// CHECK:           %[[VAL_3:.*]] = triton_gpu.convert_layout %[[VAL_1]] : tensor<128xf16, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_0]]}>> -> tensor<128xf16, #[[$ATTR_1]]>
// CHECK:           %[[VAL_4:.*]] = arith.addf %[[VAL_2]], %[[VAL_3]] : tensor<128xf16, #[[$ATTR_1]]>
    %0 = arith.addf %arg0, %arg1 : tensor<128xf16, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
// CHECK:           %[[VAL_5:.*]] = triton_gpu.convert_layout %[[VAL_4]] : tensor<128xf16, #[[$ATTR_1]]> -> tensor<128xf16, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_0]]}>>
// CHECK:           tt.return %[[VAL_5]] : tensor<128xf16, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_0]]}>>
    tt.return %0 : tensor<128xf16, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
  }
}

// -----

// CHECK: #[[$ATTR_0:.+]] = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [8], order = [0]}>
// CHECK: #[[$ATTR_1:.+]] = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 1], repCluster = [2, 2], A = [16, 16], B = [16, 32], C = [16, 32]}>

#mma = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 1], repCluster = [2, 2], A = [16, 16], B = [16, 32], C = [16, 32]}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
// CHECK-LABEL:   tt.func @test_mma_multi_warp_double_stride(
// CHECK-SAME:                                               %[[VAL_0:.*]]: tensor<128xf16, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>>,
// CHECK-SAME:                                               %[[VAL_1:.*]]: tensor<128xf16, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>>) -> tensor<128xf16, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>> {
  tt.func @test_mma_multi_warp_double_stride(%arg0: tensor<128xf16, #triton_gpu.slice<{dim = 1, parent = #mma}>>, %arg1: tensor<128xf16, #triton_gpu.slice<{dim = 1, parent = #mma}>>) -> tensor<128xf16, #triton_gpu.slice<{dim = 1, parent = #mma}>> {
// CHECK:           %[[VAL_2:.*]] = triton_gpu.convert_layout %[[VAL_0]] : tensor<128xf16, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>> -> tensor<128xf16, #[[$ATTR_0]]>
// CHECK:           %[[VAL_3:.*]] = triton_gpu.convert_layout %[[VAL_1]] : tensor<128xf16, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>> -> tensor<128xf16, #[[$ATTR_0]]>
// CHECK:           %[[VAL_4:.*]] = arith.addf %[[VAL_2]], %[[VAL_3]] : tensor<128xf16, #[[$ATTR_0]]>
    %0 = arith.addf %arg0, %arg1 : tensor<128xf16, #triton_gpu.slice<{dim = 1, parent = #mma}>>
// CHECK:           %[[VAL_5:.*]] = triton_gpu.convert_layout %[[VAL_4]] : tensor<128xf16, #[[$ATTR_0]]> -> tensor<128xf16, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>>
// CHECK:           tt.return %[[VAL_5]] : tensor<128xf16, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>>
    tt.return %0 : tensor<128xf16, #triton_gpu.slice<{dim = 1, parent = #mma}>>
  }
}

// -----

// CHECK: #[[$ATTR_0:.+]] = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [2], order = [0]}>
// CHECK: #[[$ATTR_1:.+]] = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [2, 1], repCluster = [2, 2], A = [16, 16], B = [16, 32], C = [16, 32]}>

#mma = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [2, 1], repCluster = [2, 2], A = [16, 16], B = [16, 32], C = [16, 32]}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 2 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
// CHECK-LABEL:   tt.func @test_mma_multi_warp_double_stride_repeat(
// CHECK-SAME:                                                      %[[VAL_0:.*]]: tensor<128xf16, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>>,
// CHECK-SAME:                                                      %[[VAL_1:.*]]: tensor<128xf16, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>>) -> tensor<128xf16, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>> {
  tt.func @test_mma_multi_warp_double_stride_repeat(%arg0: tensor<128xf16, #triton_gpu.slice<{dim = 1, parent = #mma}>>, %arg1: tensor<128xf16, #triton_gpu.slice<{dim = 1, parent = #mma}>>) -> tensor<128xf16, #triton_gpu.slice<{dim = 1, parent = #mma}>> {
// CHECK:           %[[VAL_2:.*]] = triton_gpu.convert_layout %[[VAL_0]] : tensor<128xf16, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>> -> tensor<128xf16, #[[$ATTR_0]]>
// CHECK:           %[[VAL_3:.*]] = triton_gpu.convert_layout %[[VAL_1]] : tensor<128xf16, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>> -> tensor<128xf16, #[[$ATTR_0]]>
// CHECK:           %[[VAL_4:.*]] = arith.addf %[[VAL_2]], %[[VAL_3]] : tensor<128xf16, #[[$ATTR_0]]>
    %0 = arith.addf %arg0, %arg1 : tensor<128xf16, #triton_gpu.slice<{dim = 1, parent = #mma}>>
// CHECK:           %[[VAL_5:.*]] = triton_gpu.convert_layout %[[VAL_4]] : tensor<128xf16, #[[$ATTR_0]]> -> tensor<128xf16, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>>
// CHECK:           tt.return %[[VAL_5]] : tensor<128xf16, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>>
    tt.return %0 : tensor<128xf16, #triton_gpu.slice<{dim = 1, parent = #mma}>>
  }
}
