// Run: triton-opt %s -split-input-file --tritonintelgpu-rewrite-tensor-pointer=device-architecture=PVC | FileCheck %s

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>
#mma = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], A = [8, 16], B = [16, 16], C = [8, 16]}>
module attributes {"triton_gpu.compute-capability" = 2 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
  tt.func public @matmul_no_scf_with_advance_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
  // CHECK-LABEL:   tt.func public @matmul_no_scf_with_advance_kernel(
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
// CHECK: #[[BLOCKED:.+]] = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 16], order = [1, 0]}>
// CHECK: #[[DPAS:.+]] = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [16, 4], A = [8, 16], B = [16, 16], C = [8, 16]}>
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 16], order = [1, 0]}>
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [16, 4], A = [8, 16], B = [16, 16], C = [8, 16]}>
module attributes {"triton_gpu.compute-capability" = 0 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 64 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
  tt.func public @matmul_kernel_with_block_pointers(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c4_i32 = arith.constant 4 : i32
    %c256_i32 = arith.constant 256 : i32
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c-64_i32 = arith.constant -64 : i32
    %c-32_i32 = arith.constant -32 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %0 = arith.extsi %arg3 : i32 to i64
    %1 = arith.extsi %arg5 : i32 to i64
    %2 = arith.extsi %arg6 : i32 to i64
    %3 = tt.make_tensor_ptr %arg0, [%0, %1], [%2, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>>>
    %4 = arith.extsi %arg4 : i32 to i64
    %5 = arith.extsi %arg7 : i32 to i64
    %6 = tt.make_tensor_ptr %arg1, [%1, %4], [%5, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma}>>>
    %7 = tt.advance %3, [%c64_i32, %c-32_i32] : <tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>>>
    %8 = tt.advance %7, [%c-64_i32, %c32_i32] : <tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>>>
    %9 = tt.load %8 {boundaryCheck = array<i32: 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, padding = 1 : i32} : !tt.ptr<tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>>>
    %10 = tt.load %6 {boundaryCheck = array<i32: 0>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, padding = 1 : i32} : !tt.ptr<tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma}>>>
    %11 = tt.dot %9, %10, %cst, inputPrecision = tf32 : tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>> * tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma}>> -> tensor<64x64xf32, #mma>
    %12 = triton_gpu.convert_layout %11 : tensor<64x64xf32, #mma> -> tensor<64x64xf32, #blocked>
    %13 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %14 = tt.expand_dims %13 {axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %15 = tt.splat %arg8 : i32 -> tensor<64x1xi32, #blocked>
    %16 = arith.muli %14, %15 : tensor<64x1xi32, #blocked>
    %17 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #blocked>
    %18 = tt.addptr %17, %16 : tensor<64x1x!tt.ptr<f32>, #blocked>, tensor<64x1xi32, #blocked>
    %19 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %20 = tt.expand_dims %19 {axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %21 = tt.broadcast %18 : tensor<64x1x!tt.ptr<f32>, #blocked> -> tensor<64x64x!tt.ptr<f32>, #blocked>
    %22 = tt.broadcast %20 : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked>
    %23 = tt.addptr %21, %22 : tensor<64x64x!tt.ptr<f32>, #blocked>, tensor<64x64xi32, #blocked>
    tt.store %23, %12 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32} : tensor<64x64x!tt.ptr<f32>, #blocked>
    %c255_i32 = arith.constant 255 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #dpas>
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg3, %c255_i32 : i32
    %2 = arith.divsi %1, %c256_i32 : i32
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
    %14 = arith.muli %11, %c256_i32 : i32
    %15 = arith.extsi %arg3 : i32 to i64
    %16 = arith.extsi %arg5 : i32 to i64
    %17 = arith.extsi %arg6 : i32 to i64
    // CHECK:  tt.make_tensor_ptr {{.*}}, {{\[}}{{.*}}, {{.*}}], {{\[}}{{.*}}, {{.*}}], {{\[}}{{.*}}, {{.*}}] {order = array<i32: 1, 0>} : <tensor<256x32xf16, {{.*}}<{opIdx = 0, parent = #[[DPAS]]}>>>
    %18 = tt.make_tensor_ptr %arg0, [%15, %16], [%17, %c1_i64], [%14, %c0_i32] {order = array<i32: 1, 0>} : <tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas}>>>
    %19 = arith.muli %13, %c256_i32 : i32
    %20 = arith.extsi %arg4 : i32 to i64
    %21 = arith.extsi %arg7 : i32 to i64
    // CHECK:  tt.make_tensor_ptr {{.*}}, {{\[}}{{.*}}, {{.*}}], {{\[}}{{.*}}, {{.*}}], {{\[}}{{.*}}, {{.*}}] {order = array<i32: 1, 0>} : <tensor<32x256xf16, {{.*}}<{opIdx = 1, parent = #[[DPAS]]}>>>
    %22 = tt.make_tensor_ptr %arg1, [%16, %20], [%21, %c1_i64], [%c0_i32, %19] {order = array<i32: 1, 0>} : <tensor<32x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #dpas}>>>
    %23:3 = scf.for %arg9 = %c0_i32 to %arg5 step %c32_i32 iter_args(%arg10 = %cst, %arg11 = %18, %arg12 = %22) -> (tensor<256x256xf32, #dpas>, !tt.ptr<tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas}>>>, !tt.ptr<tensor<32x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #dpas}>>>)  : i32 {
      // CHECK:  tt.load {{.*}} {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<256x32xf16, {{.*}}<{opIdx = 0, parent = #[[DPAS]]}>>>
      // CHECK:  tt.load {{.*}} {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<32x256xf16, {{.*}}<{opIdx = 1, parent = #[[DPAS]]}>>>
      // CHECK:  tt.fp_to_fp {{.*}} : tensor<256x32xf16, {{.*}}<{opIdx = 0, parent = #[[DPAS]]}>> -> tensor<256x32xf32, {{.*}}<{opIdx = 0, parent = #[[DPAS]]}>>
      // CHECK:  tt.fp_to_fp {{.*}} : tensor<32x256xf16, {{.*}}<{opIdx = 1, parent = #[[DPAS]]}>> -> tensor<32x256xf32, {{.*}}<{opIdx = 1, parent = #[[DPAS]]}>>
      // CHECK:  tt.dot {{.*}}, {{.*}}, {{.*}}, inputPrecision = tf32 : tensor<256x32xf32, {{.*}}<{opIdx = 0, parent = #[[DPAS]]}>> * tensor<32x256xf32, {{.*}}<{opIdx = 1, parent = #[[DPAS]]}>> -> tensor<256x256xf32, #[[DPAS]]>
      // CHECK:  tt.advance {{.*}}, {{\[}}{{.*}}, {{.*}}] : <tensor<256x32xf16, {{.*}}<{opIdx = 0, parent = #[[DPAS]]}>>>
      // CHECK:  tt.advance {{.*}}, {{\[}}{{.*}}, {{.*}}] : <tensor<32x256xf16, {{.*}}<{opIdx = 1, parent = #[[DPAS]]}>>>
      %28 = tt.load %arg11 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas}>>>
      %29 = tt.load %arg12 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<32x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #dpas}>>>
      %30 = tt.fp_to_fp %28 : tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas}>> -> tensor<256x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas}>>
      %31 = tt.fp_to_fp %29 : tensor<32x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #dpas}>> -> tensor<32x256xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #dpas}>>
      %32 = tt.dot %30, %31, %arg10, inputPrecision = tf32 : tensor<256x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas}>> * tensor<32x256xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #dpas}>> -> tensor<256x256xf32, #dpas>
      %33 = tt.advance %arg11, [%c0_i32, %c32_i32] : <tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas}>>>
      %34 = tt.advance %arg12, [%c32_i32, %c0_i32] : <tensor<32x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #dpas}>>>
      scf.yield %32, %33, %34 : tensor<256x256xf32, #dpas>, !tt.ptr<tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas}>>>, !tt.ptr<tensor<32x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #dpas}>>>
    }
    // CHECK:  arith.truncf {{.*}}#0 : tensor<256x256xf32, #[[DPAS]]> to tensor<256x256xf16, #[[DPAS]]>
    // CHECK:  triton_gpu.convert_layout {{.*}} : tensor<256x256xf16, #[[DPAS]]> -> tensor<256x256xf16, #[[BLOCKED]]>
    // CHECK:  arith.extsi {{.*}} : i32 to i64
    // CHECK:  arith.extsi {{.*}} : i32 to i64
    // CHECK:  arith.extsi {{.*}} : i32 to i64
    // CHECK:  tt.splat {{.*}} : !tt.ptr<f16> -> tensor<256x256x!tt.ptr<f16>, #[[BLOCKED]]>
    // CHECK:  tt.splat {{.*}} : i64 -> tensor<256xi64, {{.*}}<{dim = 1, parent = #[[BLOCKED]]}>>
    // CHECK:  tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, {{.*}}<{dim = 1, parent = #[[BLOCKED]]}>>
    // CHECK:  arith.extsi {{.*}} : tensor<256xi32, {{.*}}<{dim = 1, parent = #[[BLOCKED]]}>> to tensor<256xi64, {{.*}}<{dim = 1, parent = #[[BLOCKED]]}>>
    // CHECK:  arith.addi {{.*}}, {{.*}} : tensor<256xi64, {{.*}}<{dim = 1, parent = #[[BLOCKED]]}>>
    // CHECK:  tt.expand_dims {{.*}} {axis = 1 : i32} : tensor<256xi64, {{.*}}<{dim = 1, parent = #[[BLOCKED]]}>> -> tensor<256x1xi64, #[[BLOCKED]]>
    // CHECK:  tt.splat {{.*}} : i64 -> tensor<256x1xi64, #[[BLOCKED]]>
    // CHECK:  arith.muli {{.*}}, {{.*}} : tensor<256x1xi64, #[[BLOCKED]]>
    // CHECK:  tt.broadcast {{.*}} : tensor<256x1xi64, #[[BLOCKED]]> -> tensor<256x256xi64, #[[BLOCKED]]>
    // CHECK:  tt.addptr {{.*}}, {{.*}} : tensor<256x256x!tt.ptr<f16>, #[[BLOCKED]]>, tensor<256x256xi64, #[[BLOCKED]]>
    // CHECK:  tt.splat {{.*}} : i64 -> tensor<256xi64, {{.*}}<{dim = 0, parent = #[[BLOCKED]]}>>
    // CHECK:  tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, {{.*}}<{dim = 0, parent = #[[BLOCKED]]}>>
    // CHECK:  arith.extsi {{.*}} : tensor<256xi32, {{.*}}<{dim = 0, parent = #[[BLOCKED]]}>> to tensor<256xi64, {{.*}}<{dim = 0, parent = #[[BLOCKED]]}>>
    // CHECK:  arith.addi {{.*}}, {{.*}} : tensor<256xi64, {{.*}}<{dim = 0, parent = #[[BLOCKED]]}>>
    // CHECK:  tt.expand_dims {{.*}} {axis = 0 : i32} : tensor<256xi64, {{.*}}<{dim = 0, parent = #[[BLOCKED]]}>> -> tensor<1x256xi64, #[[BLOCKED]]>
    // CHECK:  tt.splat {{.*}} : i64 -> tensor<1x256xi64, #[[BLOCKED]]>
    // CHECK:  arith.muli {{.*}}, {{.*}} : tensor<1x256xi64, #[[BLOCKED]]>
    // CHECK:  tt.broadcast {{.*}} : tensor<1x256xi64, #[[BLOCKED]]> -> tensor<256x256xi64, #[[BLOCKED]]>
    // CHECK:  tt.addptr {{.*}}, {{.*}} : tensor<256x256x!tt.ptr<f16>, #[[BLOCKED]]>, tensor<256x256xi64, #[[BLOCKED]]>
    // CHECK:  arith.constant 0 : i64
    // CHECK:  tt.splat {{.*}} : i64 -> tensor<256x1xi64, #[[BLOCKED]]>
    // CHECK:  arith.cmpi sge, {{.*}}, {{.*}} : tensor<256x1xi64, #[[BLOCKED]]>
    // CHECK:  tt.splat {{.*}} : i64 -> tensor<256x1xi64, #[[BLOCKED]]>
    // CHECK:  arith.cmpi slt, {{.*}}, {{.*}} : tensor<256x1xi64, #[[BLOCKED]]>
    // CHECK:  arith.andi {{.*}}, {{.*}} : tensor<256x1xi1, #[[BLOCKED]]>
    // CHECK:  tt.broadcast {{.*}} : tensor<256x1xi1, #[[BLOCKED]]> -> tensor<256x256xi1, #[[BLOCKED]]>
    // CHECK:  arith.constant 0 : i64
    // CHECK:  tt.splat {{.*}} : i64 -> tensor<1x256xi64, #[[BLOCKED]]>
    // CHECK:  arith.cmpi sge, {{.*}}, {{.*}} : tensor<1x256xi64, #[[BLOCKED]]>
    // CHECK:  tt.splat {{.*}} : i64 -> tensor<1x256xi64, #[[BLOCKED]]>
    // CHECK:  arith.cmpi slt, {{.*}}, {{.*}} : tensor<1x256xi64, #[[BLOCKED]]>
    // CHECK:  arith.andi {{.*}}, {{.*}} : tensor<1x256xi1, #[[BLOCKED]]>
    // CHECK:  tt.broadcast {{.*}} : tensor<1x256xi1, #[[BLOCKED]]> -> tensor<256x256xi1, #[[BLOCKED]]>
    // CHECK:  arith.andi {{.*}}, {{.*}} : tensor<256x256xi1, #[[BLOCKED]]>
    // CHECK:  tt.store {{.*}}, {{.*}}, {{.*}} : tensor<256x256x!tt.ptr<f16>, #[[BLOCKED]]>
    %24 = arith.truncf %23#0 : tensor<256x256xf32, #dpas> to tensor<256x256xf16, #dpas>
    %25 = triton_gpu.convert_layout %24 : tensor<256x256xf16, #dpas> -> tensor<256x256xf16, #blocked>
    %26 = arith.extsi %arg8 : i32 to i64
    %27 = tt.make_tensor_ptr %arg2, [%15, %20], [%26, %c1_i64], [%14, %19] {order = array<i32: 1, 0>} : <tensor<256x256xf16, #blocked>>
    tt.store %27, %25 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<256x256xf16, #blocked>>
    tt.return
  }
}