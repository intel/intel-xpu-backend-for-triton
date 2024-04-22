// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s

// CHECK: llvm.func spir_funccc @llvm.genx.GenISA.LSC2DBlockRead.v8i32(i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1, i32) -> vector<8xi32>
// CHECK: llvm.func spir_funccc @llvm.genx.GenISA.LSC2DBlockRead.v8i16(i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1, i32) -> vector<8xi16>
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], A = [8, 16], B = [16, 16], C = [8, 16]}>
module attributes {"triton_gpu.compute-capability" = 2 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
  tt.func public @matmul_no_scf_with_advance_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #dpas>
    %c32_i32 = arith.constant 32 : i32
    %c-64_i32 = arith.constant -64 : i32
    %c-32_i32 = arith.constant -32 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %0 = arith.extsi %arg3 : i32 to i64
    %1 = arith.extsi %arg5 : i32 to i64
    %2 = arith.extsi %arg6 : i32 to i64
    %3 = tt.make_tensor_ptr %arg0, [%0, %1], [%2, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas}>>>
    %4 = arith.extsi %arg4 : i32 to i64
    %5 = arith.extsi %arg7 : i32 to i64
    %6 = tt.make_tensor_ptr %arg1, [%1, %4], [%5, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #dpas}>>>
    %7 = tt.advance %3, [%c64_i32, %c-32_i32] : <tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas}>>>
    %8 = tt.advance %7, [%c-64_i32, %c32_i32] : <tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas}>>>
    // CHECK-COUNT-2: llvm.call @llvm.genx.GenISA.LSC2DBlockRead.v8i16({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}) : (i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1, i32) -> vector<8xi16>
    %9 = tt.load %8 {boundaryCheck = array<i32: 1>, padding = 1 : i32} : !tt.ptr<tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas}>>>
    // CHECK-COUNT-2: llvm.call @llvm.genx.GenISA.LSC2DBlockRead.v8i32({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}) : (i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1, i32) -> vector<8xi32>
    %10 = tt.load %6 {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #dpas}>>>
    %11 = tt.dot %9, %10, %cst, inputPrecision = tf32 : tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas}>> * tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #dpas}>> -> tensor<64x64xf32, #dpas>
    %12 = triton_gpu.convert_layout %11 : tensor<64x64xf32, #dpas> -> tensor<64x64xf32, #blocked>
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
    tt.store %23, %12 : tensor<64x64x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
