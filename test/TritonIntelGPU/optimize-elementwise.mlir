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

// -----

// CHECK: #[[$ATTR_0:.+]] = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [1], order = [0]}>
// CHECK: #[[$ATTR_1:.+]] = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [2, 2], A = [16, 16], B = [16, 32], C = [16, 32]}>

#mma = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [2, 2], A = [16, 16], B = [16, 32], C = [16, 32]}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
// CHECK-LABEL:   tt.func @test_multi_user(
// CHECK-SAME:                             %[[VAL_0:.*]]: tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>>, %[[VAL_1:.*]]: tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>>, %[[VAL_2:.*]]: tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>>)
  tt.func @test_multi_user(%arg0: tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>, %arg1: tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>, %arg2: tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>) -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>> {
// CHECK:           %[[VAL_3:.*]] = triton_gpu.convert_layout %[[VAL_0]] : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>> -> tensor<16xf32, #[[$ATTR_0]]>
// CHECK:           %[[VAL_4:.*]] = triton_gpu.convert_layout %[[VAL_1]] : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>> -> tensor<16xf32, #[[$ATTR_0]]>
// CHECK:           %[[VAL_5:.*]] = arith.addf %[[VAL_3]], %[[VAL_4]] : tensor<16xf32, #[[$ATTR_0]]>
    %0 = arith.addf %arg0, %arg1 : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
// CHECK:           %[[VAL_6:.*]] = triton_gpu.convert_layout %[[VAL_5]] : tensor<16xf32, #[[$ATTR_0]]> -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>>
// CHECK:           %[[VAL_7:.*]] = triton_gpu.convert_layout %[[VAL_0]] : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>> -> tensor<16xf32, #[[$ATTR_0]]>
// CHECK:           %[[VAL_8:.*]] = triton_gpu.convert_layout %[[VAL_2]] : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>> -> tensor<16xf32, #[[$ATTR_0]]>
// CHECK:           %[[VAL_9:.*]] = arith.addf %[[VAL_7]], %[[VAL_8]] : tensor<16xf32, #[[$ATTR_0]]>
    %1 = arith.addf %arg0, %arg2 : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
// CHECK:           %[[VAL_10:.*]] = triton_gpu.convert_layout %[[VAL_9]] : tensor<16xf32, #[[$ATTR_0]]> -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>>
// CHECK:           %[[VAL_11:.*]] = triton_gpu.convert_layout %[[VAL_6]] : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>> -> tensor<16xf32, #[[$ATTR_0]]>
// CHECK:           %[[VAL_12:.*]] = triton_gpu.convert_layout %[[VAL_10]] : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>> -> tensor<16xf32, #[[$ATTR_0]]>
// CHECK:           %[[VAL_13:.*]] = arith.addf %[[VAL_11]], %[[VAL_12]] : tensor<16xf32, #[[$ATTR_0]]>
    %2 = arith.addf %0, %1 : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
// CHECK:           %[[VAL_14:.*]] = triton_gpu.convert_layout %[[VAL_13]] : tensor<16xf32, #[[$ATTR_0]]> -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>>
// CHECK:           tt.return %[[VAL_14]] : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>>
    tt.return %2 : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
  }
}

// -----

// CHECK: #[[$ATTR_0:.+]] = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [1], order = [0]}>
// CHECK: #[[$ATTR_1:.+]] = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [2, 2], A = [16, 16], B = [16, 32], C = [16, 32]}>

#mma = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [2, 2], A = [16, 16], B = [16, 32], C = [16, 32]}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {

// CHECK-LABEL:   tt.func @test_basic_loop(
// CHECK-SAME:                             %[[VAL_0:.*]]: tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>>, %[[VAL_1:.*]]: tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>>,
// CHECK-SAME:                             %[[VAL_2:.*]]: index, %[[VAL_3:.*]]: index, %[[VAL_4:.*]]: index
  tt.func @test_basic_loop(%arg0: tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>, %arg1: tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>, %arg2: index, %arg3: index, %arg4: index) -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>> {
// CHECK:           %[[VAL_5:.*]] = triton_gpu.convert_layout %[[VAL_0]] : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>> -> tensor<16xf32, #[[$ATTR_0]]>
// CHECK:           %[[VAL_6:.*]] = scf.for %[[VAL_7:.*]] = %[[VAL_2]] to %[[VAL_3]] step %[[VAL_4]] iter_args(%[[VAL_8:.*]] = %[[VAL_5]]) -> (tensor<16xf32, #[[$ATTR_0]]>) {
    %0 = scf.for %arg5 = %arg2 to %arg3 step %arg4 iter_args(%arg6 = %arg0) -> (tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>) {
// CHECK:             %[[VAL_9:.*]] = triton_gpu.convert_layout %[[VAL_8]] : tensor<16xf32, #[[$ATTR_0]]> -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>>
// CHECK:             %[[VAL_10:.*]] = triton_gpu.convert_layout %[[VAL_1]] : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>> -> tensor<16xf32, #[[$ATTR_0]]>
// CHECK:             %[[VAL_11:.*]] = triton_gpu.convert_layout %[[VAL_9]] : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>> -> tensor<16xf32, #[[$ATTR_0]]>
// CHECK:             %[[VAL_12:.*]] = arith.addf %[[VAL_10]], %[[VAL_11]] : tensor<16xf32, #[[$ATTR_0]]>
      %1 = arith.addf %arg1, %arg6 : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
// CHECK:             %[[VAL_13:.*]] = triton_gpu.convert_layout %[[VAL_12]] : tensor<16xf32, #[[$ATTR_0]]> -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>>
// CHECK:             %[[VAL_14:.*]] = triton_gpu.convert_layout %[[VAL_13]] : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>> -> tensor<16xf32, #[[$ATTR_0]]>
// CHECK:             scf.yield %[[VAL_14]] : tensor<16xf32, #[[$ATTR_0]]>
      scf.yield %1 : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    }
// CHECK:           %[[VAL_15:.*]] = triton_gpu.convert_layout %[[VAL_6]] : tensor<16xf32, #[[$ATTR_0]]> -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>>
// CHECK:           tt.return %[[VAL_15]] : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>>
    tt.return %0 : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
  }
}

// -----

// CHECK: #[[$ATTR_0:.+]] = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [1], order = [0]}>
// CHECK: #[[$ATTR_1:.+]] = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [2, 2], A = [16, 16], B = [16, 32], C = [16, 32]}>

#mma = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [2, 2], A = [16, 16], B = [16, 32], C = [16, 32]}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {

// CHECK-LABEL:   tt.func @test_advanced_loop(
// CHECK-SAME:                                %[[VAL_0:.*]]: tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>>, %[[VAL_1:.*]]: tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>>,
// CHECK-SAME:                                %[[VAL_2:.*]]: index, %[[VAL_3:.*]]: index, %[[VAL_4:.*]]: index,
// CHECK-SAME:                                %[[VAL_5:.*]]: tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>>
  tt.func @test_advanced_loop(%arg0: tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>, %arg1: tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>, %arg2: index, %arg3: index, %arg4: index, %arg5: tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>) -> (tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>, tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>) {
// CHECK:           %[[VAL_6:.*]] = triton_gpu.convert_layout %[[VAL_0]] : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>> -> tensor<16xf32, #[[$ATTR_0]]>
// CHECK:           %[[VAL_7:.*]] = triton_gpu.convert_layout %[[VAL_5]] : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>> -> tensor<16xf32, #[[$ATTR_0]]>
// CHECK:           %[[VAL_8:.*]]:2 = scf.for %[[VAL_9:.*]] = %[[VAL_2]] to %[[VAL_3]] step %[[VAL_4]] iter_args(%[[VAL_10:.*]] = %[[VAL_6]], %[[VAL_11:.*]] = %[[VAL_7]]) -> (tensor<16xf32, #[[$ATTR_0]]>, tensor<16xf32, #[[$ATTR_0]]>) {
    %0:2 = scf.for %arg6 = %arg2 to %arg3 step %arg4 iter_args(%arg7 = %arg0, %arg8 = %arg5) -> (tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>, tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>) {
// CHECK:             %[[VAL_12:.*]] = triton_gpu.convert_layout %[[VAL_10]] : tensor<16xf32, #[[$ATTR_0]]> -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>>
// CHECK:             %[[VAL_13:.*]] = triton_gpu.convert_layout %[[VAL_11]] : tensor<16xf32, #[[$ATTR_0]]> -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>>
// CHECK:             %[[VAL_14:.*]] = triton_gpu.convert_layout %[[VAL_1]] : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>> -> tensor<16xf32, #[[$ATTR_0]]>
// CHECK:             %[[VAL_15:.*]] = triton_gpu.convert_layout %[[VAL_12]] : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>> -> tensor<16xf32, #[[$ATTR_0]]>
// CHECK:             %[[VAL_16:.*]] = arith.addf %[[VAL_14]], %[[VAL_15]] : tensor<16xf32, #[[$ATTR_0]]>
      %1 = arith.addf %arg1, %arg7 : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
// CHECK:             %[[VAL_17:.*]] = triton_gpu.convert_layout %[[VAL_16]] : tensor<16xf32, #[[$ATTR_0]]> -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>>
// CHECK:             %[[VAL_18:.*]] = triton_gpu.convert_layout %[[VAL_17]] : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>> -> tensor<16xf32, #[[$ATTR_0]]>
// CHECK:             %[[VAL_19:.*]] = triton_gpu.convert_layout %[[VAL_13]] : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>> -> tensor<16xf32, #[[$ATTR_0]]>
// CHECK:             %[[VAL_20:.*]] = arith.addf %[[VAL_18]], %[[VAL_19]] : tensor<16xf32, #[[$ATTR_0]]>
// CHECK:             %[[VAL_21:.*]] = triton_gpu.convert_layout %[[VAL_20]] : tensor<16xf32, #[[$ATTR_0]]> -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>>
// CHECK:             %[[VAL_22:.*]] = triton_gpu.convert_layout %[[VAL_17]] : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>> -> tensor<16xf32, #[[$ATTR_0]]>
// CHECK:             %[[VAL_23:.*]] = triton_gpu.convert_layout %[[VAL_21]] : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>> -> tensor<16xf32, #[[$ATTR_0]]>
// CHECK:             scf.yield %[[VAL_22]], %[[VAL_23]] : tensor<16xf32, #[[$ATTR_0]]>, tensor<16xf32, #[[$ATTR_0]]>
      %2 = arith.addf %1, %arg8 : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
      scf.yield %1, %2 : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>, tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    }
// CHECK:           }
// CHECK:           %[[VAL_24:.*]] = triton_gpu.convert_layout %[[VAL_25:.*]]#0 : tensor<16xf32, #[[$ATTR_0]]> -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>>
// CHECK:           %[[VAL_26:.*]] = triton_gpu.convert_layout %[[VAL_25]]#1 : tensor<16xf32, #[[$ATTR_0]]> -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>>
// CHECK:           tt.return %[[VAL_24]], %[[VAL_26]] : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>>, tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>>
    tt.return %0#0, %0#1 : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>, tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
  }
}
