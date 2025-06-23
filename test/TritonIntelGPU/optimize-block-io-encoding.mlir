// RUN: triton-opt %s -split-input-file --tritonintelgpu-optimize-block-io-encoding | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 16], warpsPerCTA = [8, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 4], warpsPerCTA = [32, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 16], warpsPerCTA = [16, 2], order = [1, 0]}>
// CHECK: #mma = #ttig.subgroup_2d_block<{warpsPerCTA = [8, 4], instrShape = [8, 16], numBlocks=2, order=[1, 0], kWidth=1, threadsPerWarp=16}>
// CHECK: #mma1 = #ttig.subgroup_2d_block<{warpsPerCTA = [8, 4], instrShape = [16, 16], numBlocks=2, order=[0, 1], kWidth=2, threadsPerWarp=16}>
// CHECK: #mma2 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 4], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 4], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_dpas, ttig.support_sg_2d_block} {
  tt.func public @matmul_kernel_with_block_pointers(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>) {
    %c4_i32 = arith.constant 4 : i32
    %c256_i32 = arith.constant 256 : i32
    %c1024_i64 = arith.constant 1024 : i64
    %c5120_i64 = arith.constant 5120 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %c4096_i64 = arith.constant 4096 : i64
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %c5120_i32 = arith.constant 5120 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #blocked>
    %0 = tt.get_program_id x : i32
    %1 = arith.divsi %0, %c64_i32 : i32
    %2 = arith.muli %1, %c4_i32 : i32
    %3 = arith.subi %c4_i32, %2 : i32
    %4 = arith.minsi %3, %c4_i32 : i32
    %5 = arith.remsi %0, %4 : i32
    %6 = arith.addi %2, %5 : i32
    %7 = arith.remsi %0, %c64_i32 : i32
    %8 = arith.divsi %7, %4 : i32
    %9 = arith.muli %6, %c256_i32 : i32
    // CHECK: %[[MAKE_TENSOR_PTR_A:.*]] = tt.make_tensor_ptr {{.*}} : <tensor<256x32xf16, #mma>>
    %10 = tt.make_tensor_ptr %arg0, [%c1024_i64, %c5120_i64], [%c5120_i64, %c1_i64], [%9, %c0_i32] {order = array<i32: 1, 0>} : <tensor<256x32xf16, #blocked1>>
    %11 = arith.muli %8, %c256_i32 : i32
    // CHECK: %[[MAKE_TENSOR_PTR_B:.*]] = tt.make_tensor_ptr {{.*}} : <tensor<32x256xf16, #mma1>>
    %12 = tt.make_tensor_ptr %arg1, [%c5120_i64, %c4096_i64], [%c4096_i64, %c1_i64], [%c0_i32, %11] {order = array<i32: 1, 0>} : <tensor<32x256xf16, #blocked2>>
    // CHECK: scf.for {{.*}} iter_args({{.*}} = {{.*}}, %[[ARG5:.*]] = %[[MAKE_TENSOR_PTR_A]], %[[ARG6:.*]] = %[[MAKE_TENSOR_PTR_B]])
    %13:3 = scf.for %arg3 = %c0_i32 to %c5120_i32 step %c32_i32 iter_args(%arg4 = %cst, %arg5 = %10, %arg6 = %12) -> (tensor<256x256xf32, #blocked>, !tt.ptr<tensor<256x32xf16, #blocked1>>, !tt.ptr<tensor<32x256xf16, #blocked2>>)  : i32 {
      %17 = tt.load %arg5 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<256x32xf16, #blocked1>>
      // CHECK: %[[A_LOAD:.*]] = tt.load %[[ARG5]] {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<256x32xf16, #mma>>
      // CHECK: {{.*}} = ttg.convert_layout %[[A_LOAD]] : tensor<256x32xf16, #mma> -> tensor<256x32xf16, #blocked1>
      %18 = tt.load %arg6 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<32x256xf16, #blocked2>>
      // CHECK: %[[B_LOAD:.*]] = tt.load %[[ARG6]] {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<32x256xf16, #mma1>>
      // CHECK: {{.*}} = ttg.convert_layout %[[B_LOAD]] : tensor<32x256xf16, #mma1> -> tensor<32x256xf16, #blocked2>
      %19 = ttg.convert_layout %17 : tensor<256x32xf16, #blocked1> -> tensor<256x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %20 = ttg.convert_layout %18 : tensor<32x256xf16, #blocked2> -> tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %21 = ttg.convert_layout %arg4 : tensor<256x256xf32, #blocked> -> tensor<256x256xf32, #mma>
      %22 = ttg.convert_layout %19 : tensor<256x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> -> tensor<256x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %23 = ttg.convert_layout %20 : tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      // CHECK: tt.dot {{.*}} : tensor<256x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma2, kWidth = 1}>> * tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma2, kWidth = 2}>> -> tensor<256x256xf32, #mma2>
      %24 = tt.dot %22, %23, %21, inputPrecision = tf32 : tensor<256x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<256x256xf32, #mma>
      %25 = ttg.convert_layout %24 : tensor<256x256xf32, #mma> -> tensor<256x256xf32, #blocked>
      // CHECK: %[[ADVANCE_A:.*]] = tt.advance {{.*}} : <tensor<256x32xf16, #mma>>
      %26 = tt.advance %arg5, [%c0_i32, %c32_i32] : <tensor<256x32xf16, #blocked1>>
      // CHECK: %[[ADVANCE_B:.*]] = tt.advance {{.*}} : <tensor<32x256xf16, #mma1>>
      %27 = tt.advance %arg6, [%c32_i32, %c0_i32] : <tensor<32x256xf16, #blocked2>>
      // CHECK: scf.yield {{.*}}, %[[ADVANCE_A]], %[[ADVANCE_B]]
      scf.yield %25, %26, %27 : tensor<256x256xf32, #blocked>, !tt.ptr<tensor<256x32xf16, #blocked1>>, !tt.ptr<tensor<32x256xf16, #blocked2>>
    }
    %14 = tt.make_tensor_ptr %arg2, [%c1024_i64, %c4096_i64], [%c4096_i64, %c1_i64], [%9, %11] {order = array<i32: 1, 0>} : <tensor<256x256xf16, #blocked2>>
    %15 = arith.truncf %13#0 : tensor<256x256xf32, #blocked> to tensor<256x256xf16, #blocked>
    %16 = ttg.convert_layout %15 : tensor<256x256xf16, #blocked> -> tensor<256x256xf16, #blocked2>
    tt.store %14, %16 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<256x256xf16, #blocked2>>
    tt.return
  }
}
