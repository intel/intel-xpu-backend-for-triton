// RUN: triton-opt %s --convert-triton-intel-gpu-to-llvm | FileCheck %s --implicit-check-not=llvm.inline_asm

// CHECK-DAG: llvm.func spir_funccc @llvm.genx.GenISA.LSC2DBlockWrite.v8i16(i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1, i32, vector<8xi16>)
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #triton_gpu.dot_op<{opIdx = 0, parent = #dpas, kWidth=2}>
#dot1 = #triton_gpu.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
  tt.func public @matmul_no_scf_with_advance_kernel(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #dpas>
    %c32_i32 = arith.constant 32 : i32
    %c-64_i32 = arith.constant -64 : i32
    %c-32_i32 = arith.constant -32 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %3 = tt.make_tensor_ptr %arg0, [%arg3, %arg5], [%arg6, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x32xf16, #dot0>>
    %6 = tt.make_tensor_ptr %arg1, [%arg3, %arg4], [%arg7, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x64xf16, #dot1>>
    %7 = tt.advance %3, [%c64_i32, %c-32_i32] : <tensor<64x32xf16, #dot0>>
    %8 = tt.advance %7, [%c-64_i32, %c32_i32] : <tensor<64x32xf16, #dot0>>
    %9 = tt.load %8 {boundaryCheck = array<i32: 1>, padding = 1 : i32} : !tt.ptr<tensor<64x32xf16, #dot0>>
    %10 = tt.load %6 {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<32x64xf16, #dot1>>
    %11 = tt.dot %9, %10, %cst, inputPrecision = tf32 : tensor<64x32xf16, #dot0> * tensor<32x64xf16, #dot1> -> tensor<64x64xf32, #dpas>
    %12 = arith.truncf %11#0 : tensor<64x64xf32, #dpas> to tensor<64x64xf16, #dpas>
    %13 = tt.make_tensor_ptr %arg2, [%arg3, %arg5], [%arg6, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x64xf16, #dpas>>
    // The next two lines is used to start checking constant related to the BlockStore.
    // CHECK-COUNT-6: llvm.call spir_funccc @_Z12get_local_idj
    // CHECK-COUNT-39: llvm.extractvalue
    // Next constant must be equal to warpsPerCTA[0]
    // CHECK: %[[CST_4:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK: %[[VAL_0:.*]] = llvm.urem %{{[0-9]+}}, %[[CST_4]] : i32
    // Next constant must be equal to warpsPerCTA[1]
    // CHECK: %[[CST_2:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK: %[[VAL_1:.*]] = llvm.urem %{{[0-9]+}}, %[[CST_2]] : i32
    // Next constant must is elemsPerInstr[0]
    // CHECK: %[[CST_8:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK: llvm.mul %[[VAL_0]], %[[CST_8]] : i32
    // Next constant must is elemsPerInstr[1]
    // CHECK: %[[CST_16:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK: llvm.mul %[[VAL_1]], %[[CST_16]] : i32
    // CHECK: llvm.mlir.undef : vector<8xf16>
    // CHECK-COUNT-8: llvm.insertelement %{{[0-9]+}}, %{{[0-9]+}}{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK: llvm.call spir_funccc @llvm.genx.GenISA.LSC2DBlockWrite.v8i16{{.*}}
    // CHECK: llvm.mlir.undef : vector<8xf16>
    // CHECK-COUNT-8: llvm.insertelement %{{[0-9]+}}, %{{[0-9]+}}{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK: llvm.call spir_funccc @llvm.genx.GenISA.LSC2DBlockWrite.v8i16{{.*}}
    // CHECK: llvm.mlir.undef : vector<8xf16>
    // CHECK-COUNT-8: llvm.insertelement %{{[0-9]+}}, %{{[0-9]+}}{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK: llvm.call spir_funccc @llvm.genx.GenISA.LSC2DBlockWrite.v8i16{{.*}}
    // CHECK: llvm.mlir.undef : vector<8xf16>
    // CHECK-COUNT-8: llvm.insertelement %{{[0-9]+}}, %{{[0-9]+}}{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK: llvm.call spir_funccc @llvm.genx.GenISA.LSC2DBlockWrite.v8i16{{.*}}
    tt.store %13, %12 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x64xf16, #dpas>>
    tt.return
  }
}
