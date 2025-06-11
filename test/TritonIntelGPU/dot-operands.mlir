// RUN: triton-opt %s -split-input-file -tritonintelgpu-optimize-dot-operands -canonicalize | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [8, 4], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [16, 0], [0, 16], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0]], warp = [[32, 0], [64, 0], [0, 0], [0, 0], [0, 0]], block = []}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 4], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_sg_2d_block} {
  // COM: tt.load -> tt.trans -> tt.dot chain, not in a loop.
  tt.func public @fuseLoadWithTrans1(%arg0: !tt.ptr<tensor<256x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>>, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<tensor<256x256xf32, #blocked>>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c256_i32 = arith.constant 256 : i32
    %c1024_i64 = arith.constant 1024 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %0 = tt.make_tensor_ptr %arg1, [%c1024_i64, %c1_i64], [%c1_i64, %c1024_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<256x32xbf16, #linear>>
    %1 = tt.load %arg0 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<256x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>>
    %2 = tt.advance %0, [%c256_i32, %c0_i32] : <tensor<256x32xbf16, #linear>>
    %3 = tt.load %2 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "column_major"} : !tt.ptr<tensor<256x32xbf16, #linear>>
    %4 = tt.trans %3 {order = array<i32: 1, 0>} : tensor<256x32xbf16, #linear> -> tensor<32x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %5 = tt.dot %1, %4, %cst, inputPrecision = tf32 : tensor<256x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<256x256xf32, #mma>
    %6 = ttg.convert_layout %5 : tensor<256x256xf32, #mma> -> tensor<256x256xf32, #blocked>
    tt.store %arg2, %6 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<256x256xf32, #blocked>>
    tt.return
  }
  // CHECK-LABEL: fuseLoadWithTrans1
  // CHECK-NOT: tt.trans
  // CHECK: [[PTR:%.*]] = tt.make_tensor_ptr %arg1, [%c1_i64, %c1024_i64], [%c1024_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
  // CHECK: [[ADV:%.*]] = tt.advance [[PTR]], [%c0_i32, %c256_i32] : <tensor<32x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
  // CHECK: [[LOAD_B:%.*]] = tt.load [[ADV]] {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<32x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
  // CHECK: tt.dot {{.*}}, [[LOAD_B]], {{.*}}, inputPrecision = tf32 : tensor<256x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<256x256xf32, #mma>
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [8, 4], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [16, 0], [0, 16], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0]], warp = [[32, 0], [64, 0], [0, 0], [0, 0], [0, 0]], block = []}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 4], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_sg_2d_block} {
  // COM: tt.load -> tt.trans -> tt.dot chain, in a loop.
  // COM: where the 'make_tensor_ptr' result is not loop carried.
  tt.func public @fuseLoadWithTrans2(%arg0: !tt.ptr<tensor<256x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>>, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<tensor<256x256xf32, #blocked>>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c32_i32 = arith.constant 32 : i32
    %c256_i32 = arith.constant 256 : i32
    %c1024_i64 = arith.constant 1024 : i64
    %c1024_i32 = arith.constant 1024 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %0 = tt.make_tensor_ptr %arg1, [%c1024_i64, %c1_i64], [%c1_i64, %c1024_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<256x32xbf16, #linear>>
    %res:2 = scf.for %arg3 = %c0_i32 to %c1024_i32 step %c32_i32 iter_args(%arg4 = %cst, %arg5 = %c0_i32) -> (tensor<256x256xf32, #mma>, i32) : i32 {
      %1 = tt.load %arg0 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<256x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>>
      %2 = tt.advance %0, [%c256_i32, %c0_i32] : <tensor<256x32xbf16, #linear>>
      %3 = tt.load %2 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "column_major"} : !tt.ptr<tensor<256x32xbf16, #linear>>
      %4 = tt.trans %3 {order = array<i32: 1, 0>} : tensor<256x32xbf16, #linear> -> tensor<32x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %5 = tt.dot %1, %4, %arg4, inputPrecision = tf32 : tensor<256x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<256x256xf32, #mma>
      %6 = arith.addi %arg5, %c32_i32 : i32
      scf.yield %5, %6 : tensor<256x256xf32, #mma>, i32
    }
    %6 = ttg.convert_layout %res#0 : tensor<256x256xf32, #mma> -> tensor<256x256xf32, #blocked>
    tt.store %arg2, %6 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<256x256xf32, #blocked>>
    tt.return
  }
  // CHECK-LABEL: fuseLoadWithTrans2
  // CHECK-NOT: tt.trans
  // CHECK: [[PTR:%.*]] = tt.make_tensor_ptr %arg1, [%c1_i64, %c1024_i64], [%c1024_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
  // CHECK: scf.for {{.*}}
  // CHECK:   [[ADV:%.*]] = tt.advance [[PTR]], [%c0_i32, %c256_i32] : <tensor<32x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
  // CHECK:   [[LOAD_B:%.*]] = tt.load [[ADV]] {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<32x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
  // CHECK:   tt.dot {{.*}}, [[LOAD_B]], {{.*}}, inputPrecision = tf32 : tensor<256x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<256x256xf32, #mma>
  // CHECK:   scf.yield
}
