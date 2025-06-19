// RUN: triton-opt %s -split-input-file -tritonintelgpu-optimize-dot-operands | FileCheck %s

#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [16, 0], [0, 16], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0]], warp = [[32, 0], [64, 0], [0, 0], [0, 0], [0, 0]], block = []}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 4], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32} {
  // COM: tt.load -> tt.trans -> tt.dot chain, not in a loop.
  tt.func public @fuseLoadWithTrans1(%arg0: !tt.ptr<tensor<256x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>>, %arg1: !tt.ptr<bf16>) {
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

#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [16, 0], [0, 16], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0]], warp = [[32, 0], [64, 0], [0, 0], [0, 0], [0, 0]], block = []}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 4], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32} {
  // COM: tt.load -> tt.trans -> tt.dot chain, in a loop.
  // COM: where the 'make_tensor_ptr' result is not loop carried.
  tt.func public @fuseLoadWithTrans2(%arg0: !tt.ptr<tensor<256x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>>, %arg1: !tt.ptr<bf16>) {
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

// -----

#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [16, 0], [0, 16], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0]], warp = [[32, 0], [64, 0], [0, 0], [0, 0], [0, 0]], block = []}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 4], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32} {
  // COM: tt.load -> tt.trans -> tt.dot chain, in a loop.
  // COM: where the 'make_tensor_ptr' result is loop carried.
  tt.func public @fuseLoadWithTrans3(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2: !tt.ptr<f32>) {
    %c4_i32 = arith.constant 4 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c256_i32 = arith.constant 256 : i32
    %c16_i32 = arith.constant 16 : i32
    %c1_i64 = arith.constant 1 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %0 = tt.get_program_id x : i32
    %1 = arith.divsi %0, %c16_i32 : i32
    %2 = arith.muli %1, %c4_i32 : i32
    %3 = arith.subi %c4_i32, %2 : i32
    %4 = arith.minsi %3, %c4_i32 : i32
    %5 = arith.remsi %0, %c16_i32 : i32
    %6 = arith.remsi %5, %4 : i32
    %7 = arith.addi %2, %6 : i32
    %8 = arith.divsi %5, %4 : i32
    %9 = tt.make_tensor_ptr %arg0, [%c1024_i64, %c1024_i64], [%c1024_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<256x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>>
    %10 = tt.make_tensor_ptr %arg1, [%c1024_i64, %c1_i64], [%c1_i64, %c1024_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<256x32xbf16, #linear>>
    %13:3 = scf.for %arg3 = %c0_i32 to %c1024_i32 step %c32_i32 iter_args(%arg4 = %cst, %arg5 = %c0_i32, %arg6 = %10) -> (tensor<256x256xf32, #mma>, i32, !tt.ptr<tensor<256x32xbf16, #linear>>) : i32 {
      %17 = tt.advance %9, [%c256_i32, %arg5] : <tensor<256x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>>
      %18 = tt.load %17 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<256x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>>
      %19 = tt.advance %arg6, [%c16_i32, %arg5] : <tensor<256x32xbf16, #linear>>
      %20 = tt.load %19 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "column_major"} : !tt.ptr<tensor<256x32xbf16, #linear>>
      %21 = tt.trans %20 {order = array<i32: 1, 0>} : tensor<256x32xbf16, #linear> -> tensor<32x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %22 = tt.dot %18, %21, %arg4, inputPrecision = tf32 : tensor<256x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<256x256xf32, #mma>
      %23 = arith.addi %arg5, %c32_i32 : i32
      scf.yield %22, %23, %19 : tensor<256x256xf32, #mma>, i32, !tt.ptr<tensor<256x32xbf16, #linear>>
    }
    tt.return
  }
  // CHECK-LABEL: fuseLoadWithTrans3
  // CHECK-NOT: tt.trans
  // CHECK  [[IDX1:%.*]] = arith.muli
  // CHECK: [[PTR:%.*]] = tt.make_tensor_ptr %arg1, [%c1_i64, %c1024_i64], [%c1024_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
  // CHECK: scf.for {{.*}} iter_args({{.*}}, [[ARG5:%.*]] = {{.*}}, [[ARG6:%.*]] = [[PTR]])
  // CHECK:   [[ADV:%.*]] = tt.advance [[ARG6]], [[[ARG5]], %c16_i32] : <tensor<32x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
  // CHECK:   [[LOAD_B:%.*]] = tt.load [[ADV]] {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<32x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>
  // CHECK:   tt.dot {{.*}}, [[LOAD_B]], {{.*}}, inputPrecision = tf32 : tensor<256x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<256x256xf32, #mma>
  // CHECK:   scf.yield {{.*}}, {{.*}}, [[ADV]]
}

// -----

#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [16, 0], [0, 16], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0]], warp = [[32, 0], [64, 0], [0, 0], [0, 0], [0, 0]], block = []}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 4], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32} {
  // COM: Ensure load is not fused with transpose if the loop result corresponding to the pointer used by the load
  // COM: that 'feeds' the transpose operation is used.
  tt.func public @doNotFuseLoadWithTrans1(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2: !tt.ptr<f32>) {
    %c4_i32 = arith.constant 4 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c256_i32 = arith.constant 256 : i32
    %c16_i32 = arith.constant 16 : i32
    %c1_i64 = arith.constant 1 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %0 = tt.get_program_id x : i32
    %1 = arith.divsi %0, %c16_i32 : i32
    %2 = arith.muli %1, %c4_i32 : i32
    %3 = arith.subi %c4_i32, %2 : i32
    %4 = arith.minsi %3, %c4_i32 : i32
    %5 = arith.remsi %0, %c16_i32 : i32
    %6 = arith.remsi %5, %4 : i32
    %7 = arith.addi %2, %6 : i32
    %8 = arith.divsi %5, %4 : i32
    %9 = tt.make_tensor_ptr %arg0, [%c1024_i64, %c1024_i64], [%c1024_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<256x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>>
    %10 = tt.make_tensor_ptr %arg1, [%c1024_i64, %c1_i64], [%c1_i64, %c1024_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<256x32xbf16, #linear>>
    %13:3 = scf.for %arg3 = %c0_i32 to %c1024_i32 step %c32_i32 iter_args(%arg4 = %cst, %arg5 = %c0_i32, %arg6 = %10) -> (tensor<256x256xf32, #mma>, i32, !tt.ptr<tensor<256x32xbf16, #linear>>) : i32 {
      %17 = tt.advance %9, [%c256_i32, %arg5] : <tensor<256x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>>
      %18 = tt.load %17 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<256x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>>
      %19 = tt.advance %arg6, [%c16_i32, %arg5] : <tensor<256x32xbf16, #linear>>
      %20 = tt.load %19 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "column_major"} : !tt.ptr<tensor<256x32xbf16, #linear>>
      %21 = tt.trans %20 {order = array<i32: 1, 0>} : tensor<256x32xbf16, #linear> -> tensor<32x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %22 = tt.dot %18, %21, %arg4, inputPrecision = tf32 : tensor<256x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<256x256xf32, #mma>
      %23 = arith.addi %arg5, %c32_i32 : i32
      scf.yield %22, %23, %19 : tensor<256x256xf32, #mma>, i32, !tt.ptr<tensor<256x32xbf16, #linear>>
    }
    %15 = tt.advance %13#2, [%c16_i32, %c16_i32] : <tensor<256x32xbf16, #linear>>
    tt.return
  }
  // CHECK-LABEL: doNotFuseLoadWithTrans1
  // CHECK: tt.trans
}

// -----

#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [16, 0], [0, 16], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0]], warp = [[32, 0], [64, 0], [0, 0], [0, 0], [0, 0]], block = []}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 4], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32} {
  // COM: Ensure load is not fused with transpose if there are multiple users of an operation in the def-use chain containing the load + transpose.
  // COM: In this case `%19` is used by the load that feeds the transpose and by a store operation.
  tt.func public @doNotFuseLoadWithTrans2(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2: !tt.ptr<f32>) {
    %c4_i32 = arith.constant 4 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c256_i32 = arith.constant 256 : i32
    %c16_i32 = arith.constant 16 : i32
    %c1_i64 = arith.constant 1 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %cst_1 = arith.constant dense<1.000000e+00> : tensor<256x32xbf16, #linear>
    %0 = tt.get_program_id x : i32
    %1 = arith.divsi %0, %c16_i32 : i32
    %2 = arith.muli %1, %c4_i32 : i32
    %3 = arith.subi %c4_i32, %2 : i32
    %4 = arith.minsi %3, %c4_i32 : i32
    %5 = arith.remsi %0, %c16_i32 : i32
    %6 = arith.remsi %5, %4 : i32
    %7 = arith.addi %2, %6 : i32
    %8 = arith.divsi %5, %4 : i32
    %9 = tt.make_tensor_ptr %arg0, [%c1024_i64, %c1024_i64], [%c1024_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<256x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>>
    %10 = tt.make_tensor_ptr %arg1, [%c1024_i64, %c1_i64], [%c1_i64, %c1024_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<256x32xbf16, #linear>>
    %13:3 = scf.for %arg3 = %c0_i32 to %c1024_i32 step %c32_i32 iter_args(%arg4 = %cst, %arg5 = %c0_i32, %arg6 = %10) -> (tensor<256x256xf32, #mma>, i32, !tt.ptr<tensor<256x32xbf16, #linear>>) : i32 {
      %17 = tt.advance %9, [%c256_i32, %arg5] : <tensor<256x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>>
      %18 = tt.load %17 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<256x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>>
      %19 = tt.advance %arg6, [%c16_i32, %arg5] : <tensor<256x32xbf16, #linear>>
      %20 = tt.load %19 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "column_major"} : !tt.ptr<tensor<256x32xbf16, #linear>>
      %21 = tt.trans %20 {order = array<i32: 1, 0>} : tensor<256x32xbf16, #linear> -> tensor<32x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %22 = tt.dot %18, %21, %arg4, inputPrecision = tf32 : tensor<256x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<256x256xf32, #mma>
      tt.store %19, %cst_1 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<256x32xbf16, #linear>>
      %23 = arith.addi %arg5, %c32_i32 : i32
      scf.yield %22, %23, %19 : tensor<256x256xf32, #mma>, i32, !tt.ptr<tensor<256x32xbf16, #linear>>
    }
    tt.return
  }
  // CHECK-LABEL: doNotFuseLoadWithTrans2
  // CHECK: tt.trans
}

// -----

#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [16, 0], [0, 16], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0]], warp = [[32, 0], [64, 0], [0, 0], [0, 0], [0, 0]], block = []}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 4], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32} {
  // COM: tt.load -> tt.trans -> tt.dot chain, in a loop.
  // COM: where the 'make_tensor_ptr' result is not loop carried.
  tt.func public @doNotFuseLoadWithTrans3(%arg0: !tt.ptr<tensor<256x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>>, %arg1: !tt.ptr<bf16>, %cond: i1) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c32_i32 = arith.constant 32 : i32
    %c256_i32 = arith.constant 256 : i32
    %c1024_i64 = arith.constant 1024 : i64
    %c1024_i32 = arith.constant 1024 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %0 = tt.make_tensor_ptr %arg1, [%c1024_i64, %c1_i64], [%c1_i64, %c1024_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<256x32xbf16, #linear>>
    %a = tt.make_tensor_ptr %arg1, [%c1024_i64, %c1_i64], [%c1_i64, %c1024_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<256x32xbf16, #linear>>

    %res:2 = scf.for %arg3 = %c0_i32 to %c1024_i32 step %c32_i32 iter_args(%arg4 = %cst, %arg5 = %c0_i32) -> (tensor<256x256xf32, #mma>, i32) : i32 {
      %1 = tt.load %arg0 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<256x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>>
      %2 = tt.advance %0, [%c256_i32, %c0_i32] : <tensor<256x32xbf16, #linear>>
      %3 = scf.if %cond -> !tt.ptr<tensor<256x32xbf16, #linear>> {
        scf.yield %2 : !tt.ptr<tensor<256x32xbf16, #linear>>
      } else {
        scf.yield %a : !tt.ptr<tensor<256x32xbf16, #linear>>
      }
      %a4 = tt.load %3 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "column_major"} : !tt.ptr<tensor<256x32xbf16, #linear>>
      %b4 = tt.load %2 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "column_major"} : !tt.ptr<tensor<256x32xbf16, #linear>>
      %5 = tt.trans %b4 {order = array<i32: 1, 0>} : tensor<256x32xbf16, #linear> -> tensor<32x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %6 = tt.dot %1, %5, %arg4, inputPrecision = tf32 : tensor<256x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<256x256xf32, #mma>
      %7 = arith.addi %arg5, %c32_i32 : i32
      scf.yield %6, %7 : tensor<256x256xf32, #mma>, i32
    }
    tt.return
  }
  // CHECK-LABEL: doNotFuseLoadWithTrans3
  // CHECK: tt.trans
}
