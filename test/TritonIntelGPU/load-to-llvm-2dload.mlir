// RUN: triton-opt %s --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm --canonicalize | FileCheck %s --implicit-check-not=llvm.inline_asm

// CHECK: llvm.func spir_funccc @_Z38intel_sub_group_f16_f16_matrix_mad_k16Dv8_sDv8_iDv8_f(vector<8xi16>, vector<8xi32>, vector<8xf32>) -> vector<8xf32> attributes {passthrough = ["convergent"]}
// CHECK: llvm.func spir_funccc @llvm.genx.GenISA.LSC2DBlockRead.v8i32(i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1, i32) -> vector<8xi32>
// CHECK: llvm.func spir_funccc @llvm.genx.GenISA.LSC2DBlockRead.v8i16(i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1, i32) -> vector<8xi16>
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], A = [8, 16], B = [16, 16], C = [8, 16]}>
module attributes {"triton_gpu.compute-capability" = 2 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
  tt.func public @matmul_no_scf_with_advance_kernel(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f32>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
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
    // CHECK: %[[A:.*]] = llvm.call @llvm.genx.GenISA.LSC2DBlockRead.v8i16{{.*}} -> vector<8xi16>
    // CHECK: %[[castA:.*]] = llvm.bitcast %[[A]] : vector<8xi16> to vector<8xf16>
    // CHECK: llvm.call @llvm.genx.GenISA.LSC2DBlockRead.v8i16{{.*}} -> vector<8xi16>
    // CHECK-NEXT: llvm.bitcast {{.*}} : vector<8xi16> to vector<8xf16>
    // CHECK: llvm.call @llvm.genx.GenISA.LSC2DBlockRead.v8i16{{.*}} -> vector<8xi16>
    // CHECK-NEXT: llvm.bitcast {{.*}} : vector<8xi16> to vector<8xf16>
    // CHECK: llvm.call @llvm.genx.GenISA.LSC2DBlockRead.v8i16{{.*}} -> vector<8xi16>
    // CHECK-NEXT: llvm.bitcast {{.*}} : vector<8xi16> to vector<8xf16>
    // CHECK: %[[castA_0:.*]] = llvm.extractelement %[[castA]]{{.*}} : vector<8xf16>
    // CHECK: %[[castA_1:.*]] = llvm.extractelement %[[castA]]{{.*}} : vector<8xf16>
    // CHECK: %[[castA_2:.*]] = llvm.extractelement %[[castA]]{{.*}} : vector<8xf16>
    // CHECK: %[[castA_3:.*]] = llvm.extractelement %[[castA]]{{.*}} : vector<8xf16>
    // CHECK: %[[castA_4:.*]] = llvm.extractelement %[[castA]]{{.*}} : vector<8xf16>
    // CHECK: %[[castA_5:.*]] = llvm.extractelement %[[castA]]{{.*}} : vector<8xf16>
    // CHECK: %[[castA_6:.*]] = llvm.extractelement %[[castA]]{{.*}} : vector<8xf16>
    // CHECK: %[[castA_7:.*]] = llvm.extractelement %[[castA]]{{.*}} : vector<8xf16>
    // CHECK: %[[B:.*]] = llvm.call @llvm.genx.GenISA.LSC2DBlockRead.v8i32{{.*}} -> vector<8xi32>
    // CHECK: %[[castB:.*]] = llvm.bitcast %[[B]] : vector<8xi32> to vector<16xf16>
    // CHECK: llvm.call @llvm.genx.GenISA.LSC2DBlockRead.v8i32{{.*}} -> vector<8xi32>
    // CHECK-NEXT: llvm.bitcast {{.*}} : vector<8xi32> to vector<16xf16>
    // CHECK: llvm.call @llvm.genx.GenISA.LSC2DBlockRead.v8i32{{.*}} -> vector<8xi32>
    // CHECK-NEXT: llvm.bitcast {{.*}} : vector<8xi32> to vector<16xf16>
    // CHECK: llvm.call @llvm.genx.GenISA.LSC2DBlockRead.v8i32{{.*}} -> vector<8xi32>
    // CHECK-NEXT: llvm.bitcast {{.*}} : vector<8xi32> to vector<16xf16>
    // CHECK: %[[castB_00:.*]] = llvm.extractelement %[[castB]]{{.*}} : vector<16xf16>
    // CHECK: %[[castB_01:.*]] = llvm.extractelement %[[castB]]{{.*}} : vector<16xf16>
    // CHECK: %[[castB_02:.*]] = llvm.extractelement %[[castB]]{{.*}} : vector<16xf16>
    // CHECK: %[[castB_03:.*]] = llvm.extractelement %[[castB]]{{.*}} : vector<16xf16>
    // CHECK: %[[castB_04:.*]] = llvm.extractelement %[[castB]]{{.*}} : vector<16xf16>
    // CHECK: %[[castB_05:.*]] = llvm.extractelement %[[castB]]{{.*}} : vector<16xf16>
    // CHECK: %[[castB_06:.*]] = llvm.extractelement %[[castB]]{{.*}} : vector<16xf16>
    // CHECK: %[[castB_07:.*]] = llvm.extractelement %[[castB]]{{.*}} : vector<16xf16>
    // CHECK: %[[castB_08:.*]] = llvm.extractelement %[[castB]]{{.*}} : vector<16xf16>
    // CHECK: %[[castB_09:.*]] = llvm.extractelement %[[castB]]{{.*}} : vector<16xf16>
    // CHECK: %[[castB_10:.*]] = llvm.extractelement %[[castB]]{{.*}} : vector<16xf16>
    // CHECK: %[[castB_11:.*]] = llvm.extractelement %[[castB]]{{.*}} : vector<16xf16>
    // CHECK: %[[castB_12:.*]] = llvm.extractelement %[[castB]]{{.*}} : vector<16xf16>
    // CHECK: %[[castB_13:.*]] = llvm.extractelement %[[castB]]{{.*}} : vector<16xf16>
    // CHECK: %[[castB_14:.*]] = llvm.extractelement %[[castB]]{{.*}} : vector<16xf16>
    // CHECK: %[[castB_15:.*]] = llvm.extractelement %[[castB]]{{.*}} : vector<16xf16>
    // CHECK: %[[vecA_0:.*]] = llvm.mlir.undef : vector<8xf16>
    // CHECK: %[[vecA_1:.*]] = llvm.insertelement %[[castA_0]], %[[vecA_0]]{{.*}} : vector<8xf16>
    // CHECK: %[[vecA_2:.*]] = llvm.insertelement %[[castA_1]], %[[vecA_1]]{{.*}} : vector<8xf16>
    // CHECK: %[[vecA_3:.*]] = llvm.insertelement %[[castA_2]], %[[vecA_2]]{{.*}} : vector<8xf16>
    // CHECK: %[[vecA_4:.*]] = llvm.insertelement %[[castA_3]], %[[vecA_3]]{{.*}} : vector<8xf16>
    // CHECK: %[[vecA_5:.*]] = llvm.insertelement %[[castA_4]], %[[vecA_4]]{{.*}} : vector<8xf16>
    // CHECK: %[[vecA_6:.*]] = llvm.insertelement %[[castA_5]], %[[vecA_5]]{{.*}} : vector<8xf16>
    // CHECK: %[[vecA_7:.*]] = llvm.insertelement %[[castA_6]], %[[vecA_6]]{{.*}} : vector<8xf16>
    // CHECK: %[[vecA_8:.*]] = llvm.insertelement %[[castA_7]], %[[vecA_7]]{{.*}} : vector<8xf16>
    // CHECK: %[[vecA:.*]] = llvm.bitcast %[[vecA_8]] : vector<8xf16> to vector<8xi16>
    // CHECK: %[[vecB_00:.*]] = llvm.mlir.undef : vector<16xf16>
    // CHECK: %[[vecB_01:.*]] = llvm.insertelement %[[castB_00]], %[[vecB_00]]{{.*}} : vector<16xf16>
    // CHECK: %[[vecB_02:.*]] = llvm.insertelement %[[castB_01]], %[[vecB_01]]{{.*}} : vector<16xf16>
    // CHECK: %[[vecB_03:.*]] = llvm.insertelement %[[castB_02]], %[[vecB_02]]{{.*}} : vector<16xf16>
    // CHECK: %[[vecB_04:.*]] = llvm.insertelement %[[castB_03]], %[[vecB_03]]{{.*}} : vector<16xf16>
    // CHECK: %[[vecB_05:.*]] = llvm.insertelement %[[castB_04]], %[[vecB_04]]{{.*}} : vector<16xf16>
    // CHECK: %[[vecB_06:.*]] = llvm.insertelement %[[castB_05]], %[[vecB_05]]{{.*}} : vector<16xf16>
    // CHECK: %[[vecB_07:.*]] = llvm.insertelement %[[castB_06]], %[[vecB_06]]{{.*}} : vector<16xf16>
    // CHECK: %[[vecB_08:.*]] = llvm.insertelement %[[castB_07]], %[[vecB_07]]{{.*}} : vector<16xf16>
    // CHECK: %[[vecB_09:.*]] = llvm.insertelement %[[castB_08]], %[[vecB_08]]{{.*}} : vector<16xf16>
    // CHECK: %[[vecB_10:.*]] = llvm.insertelement %[[castB_09]], %[[vecB_09]]{{.*}} : vector<16xf16>
    // CHECK: %[[vecB_11:.*]] = llvm.insertelement %[[castB_10]], %[[vecB_10]]{{.*}} : vector<16xf16>
    // CHECK: %[[vecB_12:.*]] = llvm.insertelement %[[castB_11]], %[[vecB_11]]{{.*}} : vector<16xf16>
    // CHECK: %[[vecB_13:.*]] = llvm.insertelement %[[castB_12]], %[[vecB_12]]{{.*}} : vector<16xf16>
    // CHECK: %[[vecB_14:.*]] = llvm.insertelement %[[castB_13]], %[[vecB_13]]{{.*}} : vector<16xf16>
    // CHECK: %[[vecB_15:.*]] = llvm.insertelement %[[castB_14]], %[[vecB_14]]{{.*}} : vector<16xf16>
    // CHECK: %[[vecB_16:.*]] = llvm.insertelement %[[castB_15]], %[[vecB_15]]{{.*}} : vector<16xf16>
    // CHECK: %[[vecB:.*]] = llvm.bitcast %[[vecB_16]] : vector<16xf16> to vector<8xi32>
    // CHECK: llvm.call @_Z38intel_sub_group_f16_f16_matrix_mad_k16Dv8_sDv8_iDv8_f(%[[vecA]], %[[vecB]], %{{.*}}) {passthrough = ["convergent"]} : (vector<8xi16>, vector<8xi32>, vector<8xf32>) -> vector<8xf32>
    // CHECK-COUNT-7: llvm.call @_Z38intel_sub_group_f16_f16_matrix_mad_k16Dv8_sDv8_iDv8_f{{.*}} -> vector<8xf32>
    %9 = tt.load %8 {boundaryCheck = array<i32: 1>, padding = 1 : i32} : !tt.ptr<tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas}>>>
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
