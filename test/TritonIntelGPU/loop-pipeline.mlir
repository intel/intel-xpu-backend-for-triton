// RUN: triton-opt %s -split-input-file -tritonintelgpu-pipeline="num-stages=3" | FileCheck %s

// CHECK: #[[$BLOCK_0:.+]] = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [1, 0]}>
// CHECK: #[[$BLOCK_1:.+]] = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
// CHECK: #[[$DPAS:.+]] = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4], A = [8, 16], B = [16, 16], C = [8, 16]}>
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4], A = [8, 16], B = [16, 16], C = [8, 16]}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    // CHECK-LABEL:   tt.func public @matmul_kernel
    %c8_i32 = arith.constant 8 : i32
    %c64_i32 = arith.constant 64 : i32
    %c256_i32 = arith.constant 256 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<32> : tensor<64x32xi32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x32xf16, #blocked>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<32x256xf16, #blocked1>
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
    %15 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %16 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %17 = tt.splat %14 : i32 -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %18 = tt.splat %14 : i32 -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %19 = arith.addi %17, %15 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %20 = arith.addi %18, %16 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %21 = tt.splat %arg3 : i32 -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %22 = arith.remsi %19, %21 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %23 = arith.muli %13, %c256_i32 : i32
    %24 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %25 = tt.splat %23 : i32 -> tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %26 = arith.addi %25, %24 : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %27 = tt.splat %arg4 : i32 -> tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %28 = arith.remsi %26, %27 : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %29 = tt.expand_dims %22 {axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %30 = tt.splat %arg6 : i32 -> tensor<64x1xi32, #blocked>
    %31 = arith.muli %29, %30 : tensor<64x1xi32, #blocked>
    %32 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %33 = tt.expand_dims %32 {axis = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %34 = tt.broadcast %31 : tensor<64x1xi32, #blocked> -> tensor<64x32xi32, #blocked>
    %35 = tt.broadcast %33 : tensor<1x32xi32, #blocked> -> tensor<64x32xi32, #blocked>
    %36 = arith.addi %34, %35 : tensor<64x32xi32, #blocked>
    %37 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x32x!tt.ptr<f16>, #blocked>
    %38 = tt.addptr %37, %36 : tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<64x32xi32, #blocked>
    %39 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %40 = tt.expand_dims %39 {axis = 1 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xi32, #blocked1>
    %41 = tt.splat %arg7 : i32 -> tensor<32x1xi32, #blocked1>
    %42 = arith.muli %40, %41 : tensor<32x1xi32, #blocked1>
    %43 = tt.expand_dims %28 {axis = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x256xi32, #blocked1>
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
    // CHECK: triton_intel_gpu.prefetch {{.*}} : tensor<64x32x!tt.ptr<f16>, #[[$BLOCK_0]]>
    // CHECK-NEXT: triton_intel_gpu.prefetch {{.*}} : tensor<32x256x!tt.ptr<f16>, #[[$BLOCK_1]]>
    // CHECK: triton_intel_gpu.prefetch {{.*}} : tensor<64x32x!tt.ptr<f16>, #[[$BLOCK_0]]>
    // CHECK-NEXT: triton_intel_gpu.prefetch {{.*}} : tensor<32x256x!tt.ptr<f16>, #[[$BLOCK_1]]>
    // CHECK: scf.for %[[VAL_92:.*]] = {{.*}} to {{.*}} step {{.*}} iter_args(%[[VAL_93:.*]] = {{.*}}, %[[VAL_94:.*]] = {{.*}}, %[[VAL_95:.*]] = {{.*}}, %[[VAL_96:.*]] = {{.*}}, %[[VAL_97:.*]] = {{.*}}) -> (tensor<64x256xf32, #[[$DPAS]]>, tensor<64x32x!tt.ptr<f16>, #[[$BLOCK_0]]>, tensor<32x256x!tt.ptr<f16>, #[[$BLOCK_1]]>, tensor<64x32x!tt.ptr<f16>, #[[$BLOCK_0]]>, tensor<32x256x!tt.ptr<f16>, #[[$BLOCK_1]]>)  : i32 {
    // CHECK:   %[[VAL_106:.*]] = tt.addptr %[[VAL_94]], {{.*}} : tensor<64x32x!tt.ptr<f16>, #[[$BLOCK_0]]>, tensor<64x32xi32, #[[$BLOCK_0]]>
    // CHECK:   %[[VAL_107:.*]] = tt.addptr %[[VAL_95]], {{.*}} : tensor<32x256x!tt.ptr<f16>, #[[$BLOCK_1]]>, tensor<32x256xi32, #[[$BLOCK_1]]>
    // CHECK:   triton_intel_gpu.prefetch %[[VAL_106]] {{.*}} : tensor<64x32x!tt.ptr<f16>, #[[$BLOCK_0]]>
    // CHECK:   triton_intel_gpu.prefetch %[[VAL_107]] {{.*}} : tensor<32x256x!tt.ptr<f16>, #[[$BLOCK_1]]>
    // CHECK:   %[[VAL_116:.*]] = tt.load %[[VAL_96]], {{.*}}, {{.*}} : tensor<64x32x!tt.ptr<f16>, #[[$BLOCK_0]]>
    // CHECK:   %[[VAL_120:.*]] = tt.load %[[VAL_97]], {{.*}}, {{.*}} : tensor<32x256x!tt.ptr<f16>, #[[$BLOCK_1]]>
    // CHECK:   %[[VAL_121:.*]] = triton_gpu.convert_layout %[[VAL_116]] : tensor<64x32xf16, #[[$BLOCK_0]]> -> tensor<64x32xf16, #{{.*}}<{opIdx = 0, parent = #[[$DPAS]]}>>
    // CHECK:   %[[VAL_122:.*]] = triton_gpu.convert_layout %[[VAL_120]] : tensor<32x256xf16, #[[$BLOCK_1]]> -> tensor<32x256xf16, #{{.*}}<{opIdx = 1, parent = #[[$DPAS]]}>>
    // CHECK:   %[[VAL_123:.*]] = tt.dot %[[VAL_121]], %[[VAL_122]], %[[VAL_93]], inputPrecision = tf32 : tensor<64x32xf16, #{{.*}}<{opIdx = 0, parent = #[[$DPAS]]}>> * tensor<32x256xf16, #{{.*}}<{opIdx = 1, parent = #[[$DPAS]]}>> -> tensor<64x256xf32, #[[$DPAS]]>
    // CHECK:   scf.yield %[[VAL_123]], %[[VAL_106]], %[[VAL_107]], %[[VAL_94]], %[[VAL_95]] : tensor<64x256xf32, #[[$DPAS]]>, tensor<64x32x!tt.ptr<f16>, #[[$BLOCK_0]]>, tensor<32x256x!tt.ptr<f16>, #[[$BLOCK_1]]>, tensor<64x32x!tt.ptr<f16>, #[[$BLOCK_0]]>, tensor<32x256x!tt.ptr<f16>, #[[$BLOCK_1]]>
    %53:3 = scf.for %arg9 = %c0_i32 to %50 step %c1_i32 iter_args(%arg10 = %cst_2, %arg11 = %38, %arg12 = %48) -> (tensor<64x256xf32, #dpas>, tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<32x256x!tt.ptr<f16>, #blocked1>)  : i32 {
      %72 = arith.muli %arg9, %c32_i32 : i32
      %73 = arith.subi %arg5, %72 : i32
      %74 = tt.splat %73 : i32 -> tensor<1x32xi32, #blocked>
      %75 = arith.cmpi slt, %33, %74 : tensor<1x32xi32, #blocked>
      %76 = tt.broadcast %75 : tensor<1x32xi1, #blocked> -> tensor<64x32xi1, #blocked>
      %77 = tt.load %arg11, %76, %cst_0 : tensor<64x32x!tt.ptr<f16>, #blocked>
      %78 = tt.splat %73 : i32 -> tensor<32x1xi32, #blocked1>
      %79 = arith.cmpi slt, %40, %78 : tensor<32x1xi32, #blocked1>
      %80 = tt.broadcast %79 : tensor<32x1xi1, #blocked1> -> tensor<32x256xi1, #blocked1>
      %81 = tt.load %arg12, %80, %cst_1 : tensor<32x256x!tt.ptr<f16>, #blocked1>
      %82 = triton_gpu.convert_layout %77 : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas}>>
      %83 = triton_gpu.convert_layout %81 : tensor<32x256xf16, #blocked1> -> tensor<32x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #dpas}>>
      %84 = tt.dot %82, %83, %arg10, inputPrecision = tf32 : tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas}>> * tensor<32x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #dpas}>> -> tensor<64x256xf32, #dpas>
      %85 = tt.addptr %arg11, %cst : tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<64x32xi32, #blocked>
      %86 = tt.addptr %arg12, %52 : tensor<32x256x!tt.ptr<f16>, #blocked1>, tensor<32x256xi32, #blocked1>
      scf.yield %84, %85, %86 : tensor<64x256xf32, #dpas>, tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<32x256x!tt.ptr<f16>, #blocked1>
    }
    %54 = arith.truncf %53#0 : tensor<64x256xf32, #dpas> to tensor<64x256xf16, #dpas>
    %55 = triton_gpu.convert_layout %54 : tensor<64x256xf16, #dpas> -> tensor<64x256xf16, #blocked1>
    %56 = tt.expand_dims %20 {axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
    %57 = tt.splat %arg8 : i32 -> tensor<64x1xi32, #blocked1>
    %58 = arith.muli %57, %56 : tensor<64x1xi32, #blocked1>
    %59 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #blocked1>
    %60 = tt.addptr %59, %58 : tensor<64x1x!tt.ptr<f16>, #blocked1>, tensor<64x1xi32, #blocked1>
    %61 = tt.expand_dims %26 {axis = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x256xi32, #blocked1>
    %62 = tt.broadcast %60 : tensor<64x1x!tt.ptr<f16>, #blocked1> -> tensor<64x256x!tt.ptr<f16>, #blocked1>
    %63 = tt.broadcast %61 : tensor<1x256xi32, #blocked1> -> tensor<64x256xi32, #blocked1>
    %64 = tt.addptr %62, %63 : tensor<64x256x!tt.ptr<f16>, #blocked1>, tensor<64x256xi32, #blocked1>
    %65 = tt.splat %arg3 : i32 -> tensor<64x1xi32, #blocked1>
    %66 = arith.cmpi slt, %56, %65 : tensor<64x1xi32, #blocked1>
    %67 = tt.splat %arg4 : i32 -> tensor<1x256xi32, #blocked1>
    %68 = arith.cmpi slt, %61, %67 : tensor<1x256xi32, #blocked1>
    %69 = tt.broadcast %66 : tensor<64x1xi1, #blocked1> -> tensor<64x256xi1, #blocked1>
    %70 = tt.broadcast %68 : tensor<1x256xi1, #blocked1> -> tensor<64x256xi1, #blocked1>
    %71 = arith.andi %69, %70 : tensor<64x256xi1, #blocked1>
    tt.store %64, %55, %71 : tensor<64x256x!tt.ptr<f16>, #blocked1>
    tt.return
  }
}

// -----

// COM: Test that loads for a tt.dot operation are prefetched when the loaded value feeds the tt.dot operation directly.
// CHECK: #[[$BLOCK:.+]] = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [8, 4], order = [1, 0]}>
// CHECK: #[[$DPAS:.+]] = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], A = [8, 16], B = [16, 16], C = [8, 16]}>

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [8, 4], order = [1, 0]}>
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], A = [8, 16], B = [16, 16], C = [8, 16]}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 32 : i32, triton_gpu.target = "xpu:DEVICE_ARCH.PVC", "triton_gpu.threads-per-warp" = 16 : i32} {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) {
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
    %18 = tt.make_tensor_ptr %arg0, [%15, %16], [%17, %c1_i64], [%14, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas}>>>
    %19 = arith.muli %13, %c256_i32 : i32
    %20 = arith.extsi %arg4 : i32 to i64
    %21 = arith.extsi %arg7 : i32 to i64
    %22 = tt.make_tensor_ptr %arg1, [%16, %20], [%21, %c1_i64], [%c0_i32, %19] {order = array<i32: 1, 0>} : <tensor<64x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #dpas}>>>

    // CHECK:      triton_intel_gpu.prefetch {{.*}} : !tt.ptr<tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #[[$DPAS]]}>>>
    // CHECK-NEXT: triton_intel_gpu.prefetch {{.*}} : !tt.ptr<tensor<64x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[$DPAS]]}>>>
    // CHECK:      triton_intel_gpu.prefetch {{.*}} : !tt.ptr<tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #[[$DPAS]]}>>>
    // CHECK-NEXT: triton_intel_gpu.prefetch {{.*}} : !tt.ptr<tensor<64x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[$DPAS]]}>>>
    // CHECK:      scf.for %[[IV:.*]] = {{.*}} to {{.*}} step {{.*}} iter_args({{.*}}) -> (tensor<128x256xf32, #mma>, !tt.ptr<tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>>>, !tt.ptr<tensor<64x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma}>>>, !tt.ptr<tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>>>, !tt.ptr<tensor<64x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma}>>>)
    // CHECK:        triton_intel_gpu.prefetch {{.*}} : !tt.ptr<tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #[[$DPAS]]}>>>
    // CHECK-NEXT:   triton_intel_gpu.prefetch {{.*}} : !tt.ptr<tensor<64x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[$DPAS]]}>>
    // CHECK:        tt.dot {{.*}} : tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #[[$DPAS]]}>> * tensor<64x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[$DPAS]]}>> -> tensor<128x256xf32, #[[$DPAS]]>
    // CHECK-NEXT:   scf.yield
    %23:3 = scf.for %arg9 = %c0_i32 to %arg5 step %c64_i32 iter_args(%arg10 = %cst, %arg11 = %18, %arg12 = %22) -> (tensor<128x256xf32, #dpas>, !tt.ptr<tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas}>>>, !tt.ptr<tensor<64x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #dpas}>>>)  : i32 {
      %56 = tt.load %arg11 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas}>>>
      %57 = tt.load %arg12 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #dpas}>>>
      %58 = tt.dot %56, %57, %arg10, inputPrecision = tf32 : tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas}>> * tensor<64x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #dpas}>> -> tensor<128x256xf32, #dpas>
      %59 = tt.advance %arg11, [%c0_i32, %c64_i32] : <tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas}>>>
      %60 = tt.advance %arg12, [%c64_i32, %c0_i32] : <tensor<64x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #dpas}>>>
      scf.yield %58, %59, %60 : tensor<128x256xf32, #dpas>, !tt.ptr<tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas}>>>, !tt.ptr<tensor<64x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #dpas}>>>
    }
    %24 = arith.extsi %arg8 : i32 to i64
    %25 = arith.extsi %14 : i32 to i64
    %26 = arith.extsi %19 : i32 to i64
    %27 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<128x256x!tt.ptr<f32>, #blocked>
    %28 = tt.splat %25 : i64 -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %29 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %30 = arith.extsi %29 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %31 = arith.addi %28, %30 : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %32 = tt.expand_dims %31 {axis = 1 : i32} : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi64, #blocked>
    %33 = tt.splat %24 : i64 -> tensor<128x1xi64, #blocked>
    %34 = arith.muli %32, %33 : tensor<128x1xi64, #blocked>
    %35 = tt.broadcast %34 : tensor<128x1xi64, #blocked> -> tensor<128x256xi64, #blocked>
    %36 = tt.addptr %27, %35 : tensor<128x256x!tt.ptr<f32>, #blocked>, tensor<128x256xi64, #blocked>
    %37 = tt.splat %26 : i64 -> tensor<256xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %38 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %39 = arith.extsi %38 : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> to tensor<256xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %40 = arith.addi %37, %39 : tensor<256xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %41 = tt.expand_dims %40 {axis = 0 : i32} : tensor<256xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi64, #blocked>
    %42 = tt.broadcast %41 : tensor<1x256xi64, #blocked> -> tensor<128x256xi64, #blocked>
    %43 = tt.addptr %36, %42 : tensor<128x256x!tt.ptr<f32>, #blocked>, tensor<128x256xi64, #blocked>
    %44 = arith.cmpi sge, %32, %cst_1 : tensor<128x1xi64, #blocked>
    %45 = tt.splat %15 : i64 -> tensor<128x1xi64, #blocked>
    %46 = arith.cmpi slt, %32, %45 : tensor<128x1xi64, #blocked>
    %47 = arith.andi %44, %46 : tensor<128x1xi1, #blocked>
    %48 = tt.broadcast %47 : tensor<128x1xi1, #blocked> -> tensor<128x256xi1, #blocked>
    %49 = arith.cmpi sge, %41, %cst_0 : tensor<1x256xi64, #blocked>
    %50 = tt.splat %20 : i64 -> tensor<1x256xi64, #blocked>
    %51 = arith.cmpi slt, %41, %50 : tensor<1x256xi64, #blocked>
    %52 = arith.andi %49, %51 : tensor<1x256xi1, #blocked>
    %53 = tt.broadcast %52 : tensor<1x256xi1, #blocked> -> tensor<128x256xi1, #blocked>
    %54 = arith.andi %48, %53 : tensor<128x256xi1, #blocked>
    %55 = triton_gpu.convert_layout %23#0 : tensor<128x256xf32, #dpas> -> tensor<128x256xf32, #blocked>
    tt.store %43, %55, %54 : tensor<128x256x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
