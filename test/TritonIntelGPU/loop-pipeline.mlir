// RUN: triton-opt %s -split-input-file -tritonintelgpu-pipeline="num-stages=3" | FileCheck %s

// CHECK: #[[$BLOCK_0:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [1, 0]}>
// CHECK: #[[$BLOCK_1:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
// CHECK: #[[$DPAS:.+]] = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_sg_2d_block"} {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    // CHECK-LABEL:   tt.func public @matmul_kernel
    %c8_i32 = arith.constant 8 : i32
    %c64_i32 = arith.constant 64 : i32
    %c256_i32 = arith.constant 256 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<32> : tensor<64x32xi32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x32xf16, #blocked>
    %c63_i32 = arith.constant 63 : i32
    %c255_i32 = arith.constant 255 : i32
    %c31_i32 = arith.constant 31 : i32
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<64x256xf32, #dpas>
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg3, %c63_i32 : i32
    %2 = arith.divsi %1, %c64_i32 : i32
    %3 = arith.addi %arg4, %c255_i32 : i32
    %4 = arith.divsi %3, %c256_i32 : i32
    %5 = arith.muli %4, %c8_i32 : i32
    %6 = arith.divsi %0, %5 : i32
    %7 = arith.muli %6, %c8_i32 : i32
    %8 = arith.subi %2, %7 : i32
    %9 = arith.minsi %8, %c8_i32 : i32
    %10 = arith.remsi %0, %9 : i32
    %11 = arith.addi %7, %10 : i32
    %12 = arith.remsi %0, %5 : i32
    %13 = arith.divsi %12, %9 : i32
    %14 = arith.muli %11, %c64_i32 : i32
    %15 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %16 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %17 = tt.splat %14 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %18 = tt.splat %14 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %19 = arith.addi %17, %15 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %20 = arith.addi %18, %16 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %21 = tt.splat %arg3 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %22 = arith.remsi %19, %21 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %23 = arith.muli %13, %c256_i32 : i32
    %24 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %25 = tt.splat %23 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %26 = arith.addi %25, %24 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %27 = tt.splat %arg4 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %28 = arith.remsi %26, %27 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %29 = tt.expand_dims %22 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %30 = tt.splat %arg6 : i32 -> tensor<64x1xi32, #blocked>
    %31 = arith.muli %29, %30 : tensor<64x1xi32, #blocked>
    %32 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %33 = tt.expand_dims %32 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %34 = tt.broadcast %31 : tensor<64x1xi32, #blocked> -> tensor<64x32xi32, #blocked>
    %35 = tt.broadcast %33 : tensor<1x32xi32, #blocked> -> tensor<64x32xi32, #blocked>
    %36 = arith.addi %34, %35 : tensor<64x32xi32, #blocked>
    %37 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x32x!tt.ptr<f16>, #blocked>
    %38 = tt.addptr %37, %36 : tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<64x32xi32, #blocked>
    %39 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %40 = tt.expand_dims %39 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xi32, #blocked1>
    %41 = tt.splat %arg7 : i32 -> tensor<32x1xi32, #blocked1>
    %42 = arith.muli %40, %41 : tensor<32x1xi32, #blocked1>
    %43 = tt.expand_dims %28 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x256xi32, #blocked1>
    %44 = tt.broadcast %42 : tensor<32x1xi32, #blocked1> -> tensor<32x256xi32, #blocked1>
    %45 = tt.broadcast %43 : tensor<1x256xi32, #blocked1> -> tensor<32x256xi32, #blocked1>
    %46 = arith.addi %44, %45 : tensor<32x256xi32, #blocked1>
    %47 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<32x256x!tt.ptr<f16>, #blocked1>
    %48 = tt.addptr %47, %46 : tensor<32x256x!tt.ptr<f16>, #blocked1>, tensor<32x256xi32, #blocked1>
    %49 = arith.addi %arg5, %c31_i32 : i32
    %50 = arith.divsi %49, %c32_i32 : i32
    %51 = arith.muli %arg7, %c32_i32 : i32
    %52 = tt.splat %51 : i32 -> tensor<32x256xi32, #blocked1>
    // COM: There are 3 stages in loop pipelining, the first 2 prefetching stages are before the loop and the last one is inside the loop.
    // CHECK: %[[LOAD_MASK:.*]] = arith.cmpi slt, {{.*}} : tensor<1x32xi32, #[[$BLOCK_0]]>
    // CHECK-NEXT: %[[LOAD_MASK_2D:.*]] = tt.broadcast %[[LOAD_MASK]] : tensor<1x32xi1, #[[$BLOCK_0]]> -> tensor<64x32xi1, #[[$BLOCK_0]]>
    // CHECK-NEXT: %[[LOOP_MASK:.*]] = tt.splat {{.*}} : i1 -> tensor<64x32xi1, #[[$BLOCK_0]]>
    // CHECK-NEXT: %[[PREFETCH_MASK:.*]] = arith.andi %[[LOOP_MASK]], %[[LOAD_MASK_2D]] : tensor<64x32xi1, #[[$BLOCK_0]]>
    // CHECK-NEXT: ttig.prefetch {{.*}}, %[[PREFETCH_MASK]] {{.*}}ttig.block_io = "row_major"{{.*}} : tensor<64x32x!tt.ptr<f16>, #[[$BLOCK_0]]>
    // CHECK: ttig.prefetch {{.*}} : tensor<32x256x!tt.ptr<f16>, #[[$BLOCK_1]]>
    // CHECK: ttig.prefetch {{.*}} : tensor<64x32x!tt.ptr<f16>, #[[$BLOCK_0]]>
    // CHECK: ttig.prefetch {{.*}} : tensor<32x256x!tt.ptr<f16>, #[[$BLOCK_1]]>
    // CHECK: scf.for %[[VAL_92:.*]] = {{.*}} to {{.*}} step {{.*}} iter_args(%[[VAL_93:.*]] = {{.*}}, %[[VAL_94:.*]] = {{.*}}, %[[VAL_95:.*]] = {{.*}}, %[[VAL_96:.*]] = {{.*}}, %[[ARG_13:.*]] = {{.*}}, %[[ARG_14:.*]] = {{.*}}, %[[VAL_97:.*]] = {{.*}}, %[[ARG_16:.*]] = {{.*}}, %[[ARG_17:.*]] = {{.*}}) ->
    // CHECK-SAME: (tensor<64x256xf32, #[[$DPAS]]>, tensor<64x32x!tt.ptr<f16>, #[[$BLOCK_0]]>, tensor<32x256x!tt.ptr<f16>, #[[$BLOCK_1]]>, tensor<64x32x!tt.ptr<f16>, #[[$BLOCK_0]]>, tensor<64x32xi1, #[[$BLOCK_0]]>, tensor<64x32xi1, #[[$BLOCK_0]]>, tensor<32x256x!tt.ptr<f16>, #[[$BLOCK_1]]>, i32, i32)  : i32 {
    // CHECK:   %[[VAL_106:.*]] = tt.addptr %[[VAL_94]], {{.*}} : tensor<64x32x!tt.ptr<f16>, #[[$BLOCK_0]]>, tensor<64x32xi32, #[[$BLOCK_0]]>
    // CHECK:   %[[VAL_107:.*]] = tt.addptr %[[VAL_95]], {{.*}} : tensor<32x256x!tt.ptr<f16>, #[[$BLOCK_1]]>, tensor<32x256xi32, #[[$BLOCK_1]]>
    // CHECK:   %[[VAL_108:.*]] = arith.subi {{.*}} : i32
    // CHECK:   %[[VAL_109:.*]] = tt.splat %[[VAL_108]] : i32 -> tensor<1x32xi32, #blocked>
    // CHECK-NEXT: %[[LOAD_MASK:.*]] = arith.cmpi slt, {{.*}}, %[[VAL_109]] : tensor<1x32xi32, #blocked>
    // CHECK-NEXT: %[[LOAD_MASK_2D_1:.*]] = tt.broadcast %[[LOAD_MASK]] : tensor<1x32xi1, #blocked> -> tensor<64x32xi1, #blocked>
    // CHECK-NEXT: %[[LOOP_MASK:.*]] = tt.splat {{.*}} : i1 -> tensor<64x32xi1, #[[$BLOCK_0]]>
    // CHECK-NEXT: %[[PREFETCH_MASK:.*]] = arith.andi %[[LOOP_MASK]], %[[LOAD_MASK_2D_1]] : tensor<64x32xi1, #blocked>
    // CHECK-NEXT: ttig.prefetch %[[VAL_106]], %[[PREFETCH_MASK]] {{.*}} : tensor<64x32x!tt.ptr<f16>, #[[$BLOCK_0]]>
    // CHECK:   %[[PREFETCH_MASK:.*]] = tt.splat {{.*}} : i1 -> tensor<32x256xi1, #[[$BLOCK_1]]>
    // CHECK-NEXT: ttig.prefetch %[[VAL_107]], %[[PREFETCH_MASK]] {{.*}} : tensor<32x256x!tt.ptr<f16>, #[[$BLOCK_1]]>
    // CHECK:   %[[VAL_116:.*]] = tt.load %[[VAL_96]], {{.*}}, {{.*}} : tensor<64x32x!tt.ptr<f16>, #[[$BLOCK_0]]>
    // CHECK:   %[[VAL_120:.*]] = tt.load %[[VAL_97]] {{.*}} : tensor<32x256x!tt.ptr<f16>, #[[$BLOCK_1]]>
    // CHECK:   %[[VAL_121:.*]] = ttg.convert_layout %[[VAL_116]] : tensor<64x32xf16, #[[$BLOCK_0]]> -> tensor<64x32xf16, #{{.*}}<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>
    // CHECK:   %[[VAL_122:.*]] = ttg.convert_layout %[[VAL_120]] : tensor<32x256xf16, #[[$BLOCK_1]]> -> tensor<32x256xf16, #{{.*}}<{opIdx = 1, parent = #[[$DPAS]], kWidth = 2}>>
    // CHECK:   %[[VAL_123:.*]] = tt.dot %[[VAL_121]], %[[VAL_122]], %[[VAL_93]], inputPrecision = tf32 : tensor<64x32xf16, #{{.*}}<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>> * tensor<32x256xf16, #{{.*}}<{opIdx = 1, parent = #[[$DPAS]], kWidth = 2}>> -> tensor<64x256xf32, #[[$DPAS]]>
    // CHECK:   scf.yield %[[VAL_123]], %[[VAL_106]], %[[VAL_107]], %[[VAL_94]], %[[ARG_14]], %[[LOAD_MASK_2D_1]], %[[VAL_95]], %[[ARG_17]], %[[VAL_108]] :
    // CHECK-SAME: tensor<64x256xf32, #[[$DPAS]]>, tensor<64x32x!tt.ptr<f16>, #[[$BLOCK_0]]>, tensor<32x256x!tt.ptr<f16>, #[[$BLOCK_1]]>, tensor<64x32x!tt.ptr<f16>, #[[$BLOCK_0]]>, tensor<64x32xi1, #[[$BLOCK_0]]>, tensor<64x32xi1, #[[$BLOCK_0]]>, tensor<32x256x!tt.ptr<f16>, #[[$BLOCK_1]]>, i32, i32
    %53:3 = scf.for %arg9 = %c0_i32 to %50 step %c1_i32 iter_args(%arg10 = %cst_2, %arg11 = %38, %arg12 = %48) -> (tensor<64x256xf32, #dpas>, tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<32x256x!tt.ptr<f16>, #blocked1>)  : i32 {
      %72 = arith.muli %arg9, %c32_i32 : i32
      %73 = arith.subi %arg5, %72 : i32
      %74 = tt.splat %73 : i32 -> tensor<1x32xi32, #blocked>
      %75 = arith.cmpi slt, %33, %74 : tensor<1x32xi32, #blocked>
      %76 = tt.broadcast %75 : tensor<1x32xi1, #blocked> -> tensor<64x32xi1, #blocked>
      %77 = tt.load %arg11, %76, %cst_0 {ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #blocked>
      %78 = tt.splat %73 : i32 -> tensor<32x1xi32, #blocked1>
      %79 = arith.cmpi slt, %40, %78 : tensor<32x1xi32, #blocked1>
      %80 = tt.broadcast %79 : tensor<32x1xi1, #blocked1> -> tensor<32x256xi1, #blocked1>
      %81 = tt.load %arg12 {ttig.block_io = "row_major"} : tensor<32x256x!tt.ptr<f16>, #blocked1>
      %82 = ttg.convert_layout %77 : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #dot0>
      %83 = ttg.convert_layout %81 : tensor<32x256xf16, #blocked1> -> tensor<32x256xf16, #dot1>
      %84 = tt.dot %82, %83, %arg10, inputPrecision = tf32 : tensor<64x32xf16, #dot0> * tensor<32x256xf16, #dot1> -> tensor<64x256xf32, #dpas>
      %85 = tt.addptr %arg11, %cst : tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<64x32xi32, #blocked>
      %86 = tt.addptr %arg12, %52 : tensor<32x256x!tt.ptr<f16>, #blocked1>, tensor<32x256xi32, #blocked1>
      scf.yield %84, %85, %86 : tensor<64x256xf32, #dpas>, tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<32x256x!tt.ptr<f16>, #blocked1>
    }
    tt.return
  }
}

// -----

// COM: Test that loads for a tt.dot operation are prefetched when the loaded value feeds the tt.dot operation directly.
// CHECK: #[[$BLOCK:.+]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [8, 4], order = [1, 0]}>
// CHECK: #[[$DPAS:.+]] = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [8, 4], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>

module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_sg_2d_block"} {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}) {
    // CHECK-LABEL:   tt.func public @matmul_kernel
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #dpas>
    %c127_i32 = arith.constant 127 : i32
    %c255_i32 = arith.constant 255 : i32
    %c64_i32 = arith.constant 64 : i32
    %c256_i32 = arith.constant 256 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c128_i32 = arith.constant 128 : i32
    %c4_i32 = arith.constant 4 : i32
    %cst_0 = arith.constant dense<0> : tensor<1x256xi64, #blocked>
    %cst_1 = arith.constant dense<0> : tensor<128x1xi64, #blocked>
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg3, %c127_i32 : i32
    %2 = arith.divsi %1, %c128_i32 : i32
    %3 = arith.addi %arg4, %c255_i32 : i32
    %4 = arith.divsi %3, %c256_i32 : i32
    %5 = arith.muli %4, %c4_i32 : i32
    %6 = arith.divsi %0, %5 : i32
    %7 = arith.muli %6, %c4_i32 : i32
    %8 = arith.subi %2, %7 : i32
    %9 = arith.minsi %8, %c4_i32 : i32
    %10 = arith.remsi %0, %9 : i32
    %11 = arith.addi %7, %10 : i32
    %12 = arith.remsi %0, %5 : i32
    %13 = arith.divsi %12, %9 : i32
    %14 = arith.muli %11, %c128_i32 : i32
    %15 = arith.extsi %arg3 : i32 to i64
    %16 = arith.extsi %arg5 : i32 to i64
    %17 = arith.extsi %arg6 : i32 to i64
    %18 = tt.make_tensor_ptr %arg0, [%15, %16], [%17, %c1_i64], [%14, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x64xf16, #dot0>>
    %19 = arith.muli %13, %c256_i32 : i32
    %20 = arith.extsi %arg4 : i32 to i64
    %21 = arith.extsi %arg7 : i32 to i64
    %22 = tt.make_tensor_ptr %arg1, [%16, %20], [%21, %c1_i64], [%c0_i32, %19] {order = array<i32: 1, 0>} : <tensor<64x256xf16, #dot1>>

    // CHECK:      ttig.prefetch {{.*}} : !tt.ptr<tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>>
    // CHECK:      ttig.prefetch {{.*}} : !tt.ptr<tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$DPAS]], kWidth = 2}>>>
    // CHECK:      ttig.prefetch {{.*}} : !tt.ptr<tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>>
    // CHECK:      ttig.prefetch {{.*}} : !tt.ptr<tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$DPAS]], kWidth = 2}>>>
    // CHECK:      scf.for %[[IV:.*]] = {{.*}} to {{.*}} step {{.*}} iter_args({{.*}}) -> (tensor<128x256xf32, #mma>, !tt.ptr<tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>>, !tt.ptr<tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>, !tt.ptr<tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>>, !tt.ptr<tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>)
    // CHECK:        ttig.prefetch {{.*}} : !tt.ptr<tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>>
    // CHECK:        ttig.prefetch {{.*}} : !tt.ptr<tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$DPAS]], kWidth = 2}>>
    // CHECK:        tt.dot {{.*}} : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>> * tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$DPAS]], kWidth = 2}>> -> tensor<128x256xf32, #[[$DPAS]]>
    // CHECK-NEXT:   scf.yield
    %23:3 = scf.for %arg9 = %c0_i32 to %arg5 step %c64_i32 iter_args(%arg10 = %cst, %arg11 = %18, %arg12 = %22) -> (tensor<128x256xf32, #dpas>, !tt.ptr<tensor<128x64xf16, #dot0>>, !tt.ptr<tensor<64x256xf16, #dot1>>)  : i32 {
      %56 = tt.load %arg11 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<128x64xf16, #dot0>>
      %57 = tt.load %arg12 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<64x256xf16, #dot1>>
      %58 = tt.dot %56, %57, %arg10, inputPrecision = tf32 : tensor<128x64xf16, #dot0> * tensor<64x256xf16, #dot1> -> tensor<128x256xf32, #dpas>
      %59 = tt.advance %arg11, [%c0_i32, %c64_i32] : <tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>
      %60 = tt.advance %arg12, [%c64_i32, %c0_i32] : <tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>>
      scf.yield %58, %59, %60 : tensor<128x256xf32, #dpas>, !tt.ptr<tensor<128x64xf16, #dot0>>, !tt.ptr<tensor<64x256xf16, #dot1>>
    }
    tt.return
  }
}

// -----

// COM: Test that dependency between AdvanceOp and none-tensor load are not triggering a pipeline schedule order error.
// CHECK-NOT:     error: operation scheduled before its operands
// CHECK: #[[$BLOCK:.+]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [8, 4], order = [1, 0]}>
// CHECK: #[[$DPAS:.+]] = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [8, 4], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>

module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_sg_2d_block"} {
  tt.func public @matmul_kernel_dep(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<i32> {tt.divisibility = 16 : i32}) {
    // CHECK-LABEL:   tt.func public @matmul_kernel_dep
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #dpas>
    %c127_i32 = arith.constant 127 : i32
    %c255_i32 = arith.constant 255 : i32
    %c64_i32 = arith.constant 64 : i32
    %c256_i32 = arith.constant 256 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c128_i32 = arith.constant 128 : i32
    %c4_i32 = arith.constant 4 : i32
    %cst_0 = arith.constant dense<0> : tensor<1x256xi64, #blocked>
    %cst_1 = arith.constant dense<0> : tensor<128x1xi64, #blocked>
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg3, %c127_i32 : i32
    %2 = arith.divsi %1, %c128_i32 : i32
    %3 = arith.addi %arg4, %c255_i32 : i32
    %4 = arith.divsi %3, %c256_i32 : i32
    %5 = arith.muli %4, %c4_i32 : i32
    %6 = arith.divsi %0, %5 : i32
    %7 = arith.muli %6, %c4_i32 : i32
    %8 = arith.subi %2, %7 : i32
    %9 = arith.minsi %8, %c4_i32 : i32
    %10 = arith.remsi %0, %9 : i32
    %11 = arith.addi %7, %10 : i32
    %12 = arith.remsi %0, %5 : i32
    %13 = arith.divsi %12, %9 : i32
    %14 = arith.muli %11, %c128_i32 : i32
    %15 = arith.extsi %arg3 : i32 to i64
    %16 = arith.extsi %arg5 : i32 to i64
    %17 = arith.extsi %arg6 : i32 to i64
    %18 = tt.make_tensor_ptr %arg0, [%15, %16], [%17, %c1_i64], [%14, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x64xf16, #dot0>>
    %19 = arith.muli %13, %c256_i32 : i32
    %20 = arith.extsi %arg4 : i32 to i64
    %21 = arith.extsi %arg7 : i32 to i64
    %25 = tt.addptr %arg8, %0 : !tt.ptr<i32>, i32
    %26 = tt.load %25 : !tt.ptr<i32>
    %22 = tt.make_tensor_ptr %arg1, [%16, %20], [%21, %c1_i64], [%c0_i32, %19] {order = array<i32: 1, 0>} : <tensor<64x256xf16, #dot1>>

    // CHECK:      ttig.prefetch {{.*}} : !tt.ptr<tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>>
    // CHECK:      ttig.prefetch {{.*}} : !tt.ptr<tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$DPAS]], kWidth = 2}>>>
    // CHECK:      ttig.prefetch {{.*}} : !tt.ptr<tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>>
    // CHECK:      ttig.prefetch {{.*}} : !tt.ptr<tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$DPAS]], kWidth = 2}>>>
    // CHECK:      scf.for %[[IV:.*]] = {{.*}} to {{.*}} step {{.*}} iter_args({{.*}}) -> (tensor<128x256xf32, #mma>, !tt.ptr<tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>>, !tt.ptr<tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>, !tt.ptr<tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>>, !tt.ptr<tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>)
    // CHECK:        ttig.prefetch {{.*}} : !tt.ptr<tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>>
    // CHECK:        ttig.prefetch {{.*}} : !tt.ptr<tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$DPAS]], kWidth = 2}>>
    // CHECK:        tt.dot {{.*}} : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>> * tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$DPAS]], kWidth = 2}>> -> tensor<128x256xf32, #[[$DPAS]]>
    // CHECK-NEXT:   scf.yield
    %23:3 = scf.for %arg9 = %c0_i32 to %arg5 step %c64_i32 iter_args(%arg10 = %cst, %arg11 = %18, %arg12 = %22) -> (tensor<128x256xf32, #dpas>, !tt.ptr<tensor<128x64xf16, #dot0>>, !tt.ptr<tensor<64x256xf16, #dot1>>)  : i32 {
      %56 = tt.load %arg11 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<128x64xf16, #dot0>>
      %57 = tt.load %arg12 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<64x256xf16, #dot1>>
      %58 = tt.dot %56, %57, %arg10, inputPrecision = tf32 : tensor<128x64xf16, #dot0> * tensor<64x256xf16, #dot1> -> tensor<128x256xf32, #dpas>
      %102 = tt.addptr %arg8, %c4_i32 : !tt.ptr<i32>, i32
      %100 = arith.addi %c0_i32, %c4_i32 : i32
      %101 = arith.cmpi slt, %100, %26 : i32
      %103 = tt.load %102, %101 evictionPolicy = evict_last : !tt.ptr<i32>
      %59 = tt.advance %arg11, [%c0_i32, %103] : <tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>
      %60 = tt.advance %arg12, [%103, %c0_i32] : <tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>>
      scf.yield %58, %59, %60 : tensor<128x256xf32, #dpas>, !tt.ptr<tensor<128x64xf16, #dot0>>, !tt.ptr<tensor<64x256xf16, #dot1>>
    }
    tt.return
  }
}

// -----

// COM: Reproducer for issue #3810.

// CHECK: #[[BLOCKED1:.+]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [4, 1, 4], threadsPerWarp = [1, 2, 16], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [4, 4, 1], threadsPerWarp = [1, 16, 2], warpsPerCTA = [4, 1, 1], order = [1, 2, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 8, 2], threadsPerWarp = [4, 8, 1], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_dpas, ttig.support_sg_2d_block, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel_descriptor_persistent(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) {
    // CHECK: tt.func public @matmul_kernel_descriptor_persistent([[PARAM_0:%.*]]: !tt.ptr<f16> {{.*}}, [[PARAM_1:%.*]]: !tt.ptr<f16> {{.*}}, [[PARAM_2:%.*]]: !tt.ptr<f16> {{.*}}, [[PARAM_3:%.*]]: i32 {{.*}}, [[PARAM_4:%.*]]: i32 {{.*}}, [[PARAM_5:%.*]]: i32 {{.*}})
    %c448_i32 = arith.constant 448 : i32
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c127_i32 = arith.constant 127 : i32
    %c63_i32 = arith.constant 63 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg3, %c127_i32 : i32
    %2 = arith.divsi %1, %c128_i32 : i32
    %3 = arith.addi %arg4, %c127_i32 : i32
    %4 = arith.divsi %3, %c128_i32 : i32
    %5 = arith.addi %arg5, %c63_i32 : i32
    %6 = arith.divsi %5, %c64_i32 : i32
    %7 = arith.muli %2, %4 : i32
    %8 = arith.extsi %arg5 : i32 to i64
    %9 = arith.extsi %arg4 : i32 to i64
    %10 = arith.subi %0, %c448_i32 : i32
    %11 = arith.muli %4, %c8_i32 : i32
    %12 = arith.extsi %arg3 : i32 to i64
    // CHECK:  scf.for %[[OUTER_IV:.*]] = {{.*}} to {{.*}} step {{.*}} iter_args({{.*}}) -> (i32)
    // CHECK:      [[PTR1:%.*]] = tt.make_tensor_ptr [[PARAM_0]], {{.*}} : <tensor<128x64xf16, #[[BLOCKED1]]>>
    // CHECK:      ttig.prefetch [[PTR1]], {{.*}} {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>, ttig.block_io = "row_major"} : !tt.ptr<tensor<128x64xf16, #[[BLOCKED1]]>>
    // CHECK:      [[PTR2:%.*]] = tt.make_tensor_ptr [[PARAM_1]], {{.*}} : <tensor<128x64xf16, #[[BLOCKED1]]>>
    // CHECK:      ttig.prefetch [[PTR2]], {{.*}} {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>, ttig.block_io = "row_major"} : !tt.ptr<tensor<128x64xf16, #[[BLOCKED1]]>>
    // CHECK:      [[PTR3:%.*]] = tt.make_tensor_ptr [[PARAM_0]], {{.*}} : <tensor<128x64xf16, #[[BLOCKED1]]>>
    // CHECK:      ttig.prefetch [[PTR3]], {{.*}} {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>, ttig.block_io = "row_major"} : !tt.ptr<tensor<128x64xf16, #[[BLOCKED1]]>>
    // CHECK:      [[PTR4:%.*]] = tt.make_tensor_ptr [[PARAM_1]], {{.*}} : <tensor<128x64xf16, #[[BLOCKED1]]>>
    // CHECK:      ttig.prefetch [[PTR4]], {{.*}} {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>, ttig.block_io = "row_major"} : !tt.ptr<tensor<128x64xf16, #[[BLOCKED1]]>>
    // CHECK-NEXT: scf.for %[[INNER_IV:.*]] = {{.*}} to {{.*}} step {{.*}} iter_args({{.*}}) -> (tensor<128x128xf32, #blocked>, !tt.ptr<tensor<128x64xf16, #blocked1>>, !tt.ptr<tensor<128x64xf16, #blocked1>>, !tt.ptr<tensor<128x64xf16, #blocked1>>, !tt.ptr<tensor<128x64xf16, #[[BLOCKED1]]>>)
    // CHECK:        [[PTR5:%.*]] = tt.make_tensor_ptr [[PARAM_0]], {{.*}} : <tensor<128x64xf16, #[[BLOCKED1]]>>
    // CHECK:        ttig.prefetch [[PTR5]], {{.*}} {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>, ttig.block_io = "row_major"} : !tt.ptr<tensor<128x64xf16, #[[BLOCKED1]]>>
    // CHECK:        [[PTR6:%.*]] = tt.make_tensor_ptr [[PARAM_1]], {{.*}} : <tensor<128x64xf16, #[[BLOCKED1]]>>
    // CHECK:        ttig.prefetch [[PTR6]], {{.*}} {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>, ttig.block_io = "row_major"} : !tt.ptr<tensor<128x64xf16, #[[BLOCKED1]]>>
    %13 = scf.for %arg6 = %0 to %7 step %c448_i32 iter_args(%arg7 = %10) -> (i32)  : i32 {
      %14 = arith.divsi %arg6, %11 : i32
      %15 = arith.muli %14, %c8_i32 : i32
      %16 = arith.subi %2, %15 : i32
      %17 = arith.minsi %16, %c8_i32 : i32
      %18 = arith.remsi %arg6, %17 : i32
      %19 = arith.addi %15, %18 : i32
      %20 = arith.remsi %arg6, %11 : i32
      %21 = arith.divsi %20, %17 : i32
      %22 = arith.muli %19, %c128_i32 : i32
      %23 = arith.muli %21, %c128_i32 : i32
      %24 = scf.for %arg8 = %c0_i32 to %6 step %c1_i32 iter_args(%arg9 = %cst) -> (tensor<128x128xf32, #blocked>)  : i32 {
        %44 = arith.muli %arg8, %c64_i32 : i32
        %45 = tt.make_tensor_ptr %arg0, [%12, %8], [%8, %c1_i64], [%22, %44] {order = array<i32: 1, 0>} : <tensor<128x64xf16, #blocked1>>
        %46 = tt.load %45 {ttig.block_io = "row_major"} : !tt.ptr<tensor<128x64xf16, #blocked1>>
        %47 = tt.make_tensor_ptr %arg1, [%9, %8], [%8, %c1_i64], [%23, %44] {order = array<i32: 1, 0>} : <tensor<128x64xf16, #blocked1>>
        %48 = tt.load %47 {ttig.block_io = "row_major"} : !tt.ptr<tensor<128x64xf16, #blocked1>>
        %49 = tt.trans %48 {order = array<i32: 1, 0>} : tensor<128x64xf16, #blocked1> -> tensor<64x128xf16, #blocked2>
        %50 = tt.fp_to_fp %46 : tensor<128x64xf16, #blocked1> -> tensor<128x64xf32, #blocked1>
        %51 = ttg.convert_layout %50 : tensor<128x64xf32, #blocked1> -> tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
        %52 = tt.fp_to_fp %49 : tensor<64x128xf16, #blocked2> -> tensor<64x128xf32, #blocked2>
        %53 = ttg.convert_layout %52 : tensor<64x128xf32, #blocked2> -> tensor<64x128xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
        %54 = tt.dot %51, %53, %arg9, inputPrecision = tf32 : tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf32, #blocked>
        scf.yield %54 : tensor<128x128xf32, #blocked>
      }
      %25 = arith.addi %arg7, %c448_i32 : i32
      %26 = arith.divsi %25, %11 : i32
      %27 = arith.muli %26, %c8_i32 : i32
      %28 = arith.subi %2, %27 : i32
      %29 = arith.minsi %28, %c8_i32 : i32
      %30 = arith.remsi %25, %29 : i32
      %31 = arith.addi %27, %30 : i32
      %32 = arith.remsi %25, %11 : i32
      %33 = arith.divsi %32, %29 : i32
      %34 = arith.muli %31, %c128_i32 : i32
      %35 = arith.muli %33, %c128_i32 : i32
      %36 = tt.reshape %24 : tensor<128x128xf32, #blocked> -> tensor<128x2x64xf32, #blocked3>
      %37 = tt.trans %36 {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked3> -> tensor<128x64x2xf32, #blocked4>
      %38 = ttg.convert_layout %37 : tensor<128x64x2xf32, #blocked4> -> tensor<128x64x2xf32, #blocked5>
      %outLHS, %outRHS = tt.split %38 : tensor<128x64x2xf32, #blocked5> -> tensor<128x64xf32, #blocked1>
      %39 = arith.truncf %outLHS : tensor<128x64xf32, #blocked1> to tensor<128x64xf16, #blocked1>
      %40 = tt.make_tensor_ptr %arg2, [%12, %9], [%9, %c1_i64], [%34, %35] {order = array<i32: 1, 0>} : <tensor<128x64xf16, #blocked1>>
      tt.store %40, %39 : !tt.ptr<tensor<128x64xf16, #blocked1>>
      %41 = arith.truncf %outRHS : tensor<128x64xf32, #blocked1> to tensor<128x64xf16, #blocked1>
      %42 = arith.addi %35, %c64_i32 : i32
      %43 = tt.make_tensor_ptr %arg2, [%12, %9], [%9, %c1_i64], [%34, %42] {order = array<i32: 1, 0>} : <tensor<128x64xf16, #blocked1>>
      tt.store %43, %41 : !tt.ptr<tensor<128x64xf16, #blocked1>>
      scf.yield %25 : i32
    } {tt.flatten}
    tt.return
  }
}

// -----

// COM: Ensure prefetch operations aren't generated for 3D loads.
#linear = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 0, 16], [0, 0, 32], [0, 64, 0]], lane = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8]], warp = [[0, 0, 0], [0, 0, 0], [0, 32, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 16, 0], [0, 0, 16], [0, 0, 32], [0, 128, 0]], lane = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0]], warp = [[0, 32, 0], [0, 64, 0], [0, 0, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [16, 0], [0, 16], [0, 32], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0]], warp = [[32, 0], [64, 0], [0, 0]], block = []}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [2, 4], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
module attributes {ttig.support_dpas, ttig.support_sg_2d_block, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: batched_gemm_3d_tma_kernel
  tt.func public @batched_gemm_3d_tma_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32, %arg6: i32 {tt.divisibility = 16 : i32}) {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %0 = tt.get_program_id x : i32
    %16 = arith.extsi %arg6 : i32 to i64
    %24 = arith.extsi %arg3 : i32 to i64
    %26:1 = scf.for %arg7 = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%arg8 = %c1_i32) -> (i32)  : i32 {
      // CHECK-NOT: prefetch
      %27 = arith.cmpi eq, %arg8, %c1_i32 : i32
      %29 = arith.select %27, %c0_i32, %arg8 : i32
      %33 = tt.make_tensor_ptr %arg0, [%24, %24, %16], [%16, %16, %c1_i64], [%arg8, %arg8, %29] {order = array<i32: 1, 0>} : <tensor<1x128x64xf16, #linear>>
      %34 = tt.load %33 {ttig.block_io = "row_major"} : !tt.ptr<tensor<1x128x64xf16, #linear>>
      %35 = tt.reshape %34 : tensor<1x128x64xf16, #linear> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %36 = tt.make_tensor_ptr %arg1, [%24, %16, %16], [%16, %16, %c1_i64], [%arg8, %arg8, %29] {order = array<i32: 1, 0>} : <tensor<1x256x64xf16, #linear1>>
      %37 = tt.load %36 {ttig.block_io = "row_major"} : !tt.ptr<tensor<1x256x64xf16, #linear1>>
      %38 = tt.reshape %37 : tensor<1x256x64xf16, #linear1> -> tensor<256x64xf16, #linear2>
      %39 = tt.trans %38 {order = array<i32: 1, 0>} : tensor<256x64xf16, #linear2> -> tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %40 = tt.dot %35, %39, %cst : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x256xf32, #mma>
      scf.yield %29 : i32
    }
    tt.return
  }
}
