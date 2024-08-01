// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s

#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [32, 1], repCluster = [1, 2], A = [8, 16], B = [16, 32], C = [8, 32]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 32 : i32, triton_gpu.target = "xpu:DEVICE_ARCH.PVC", "triton_gpu.threads-per-warp" = 16 : i32, triton_intel_gpu.min_sg_size = 16 : i32, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block} {
  // CHECK-LABEL: convert_dpas_to_dot_rep_cluster_1_2
  // CHECK-SAME:  %[[VAL_0:.*]]: !llvm.struct<({{.*}})>) attributes {triton_gen.intel_reqd_sub_group_size = [16 : i32], triton_gen.max_work_group_size = [512 : i32, 1 : i32, 1 : i32]} {
  tt.func public @convert_dpas_to_dot_rep_cluster_1_2(%arg: tensor<1024x32xf16, #dpas>) {
    // COM: The repetitions order of dot layout and dpas layout are same when the GEMM tiling is clustered as repCluster [1, 2].
    // CHECK:           %[[VAL_81:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[REP_0:.*]] = llvm.insertelement {{.*}}, {{.*}}{{\[}}%[[VAL_81]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_98:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[REP_1:.*]] = llvm.insertelement {{.*}}, {{.*}}{{\[}}%[[VAL_98]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_115:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[REP_2:.*]] = llvm.insertelement {{.*}}, {{.*}}{{\[}}%[[VAL_115]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_132:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[REP_3:.*]] = llvm.insertelement {{.*}}, {{.*}}{{\[}}%[[VAL_132]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_149:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[REP_4:.*]] = llvm.insertelement {{.*}}, {{.*}}{{\[}}%[[VAL_149]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_166:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[REP_5:.*]] = llvm.insertelement {{.*}}, {{.*}}{{\[}}%[[VAL_166]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_183:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[REP_6:.*]] = llvm.insertelement {{.*}}, {{.*}}{{\[}}%[[VAL_183]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_200:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[REP_7:.*]] = llvm.insertelement {{.*}}, {{.*}}{{\[}}%[[VAL_200]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_216:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[VAL_217:.*]] = llvm.extractelement %[[REP_0]]{{\[}}%[[VAL_216]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_232:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[VAL_233:.*]] = llvm.extractelement %[[REP_1]]{{\[}}%[[VAL_232]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_248:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[VAL_249:.*]] = llvm.extractelement %[[REP_2]]{{\[}}%[[VAL_248]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_264:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[VAL_265:.*]] = llvm.extractelement %[[REP_3]]{{\[}}%[[VAL_264]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_280:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[VAL_281:.*]] = llvm.extractelement %[[REP_4]]{{\[}}%[[VAL_280]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_296:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[VAL_297:.*]] = llvm.extractelement %[[REP_5]]{{\[}}%[[VAL_296]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_312:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[VAL_313:.*]] = llvm.extractelement %[[REP_6]]{{\[}}%[[VAL_312]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_328:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[VAL_329:.*]] = llvm.extractelement %[[REP_7]]{{\[}}%[[VAL_328]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_338:.*]] = llvm.insertvalue %[[VAL_217]], {{.*}}[7]
    // CHECK:           %[[VAL_346:.*]] = llvm.insertvalue %[[VAL_233]], {{.*}}[15]
    // CHECK:           %[[VAL_354:.*]] = llvm.insertvalue %[[VAL_249]], {{.*}}[23]
    // CHECK:           %[[VAL_362:.*]] = llvm.insertvalue %[[VAL_265]], {{.*}}[31]
    // CHECK:           %[[VAL_370:.*]] = llvm.insertvalue %[[VAL_281]], {{.*}}[39]
    // CHECK:           %[[VAL_378:.*]] = llvm.insertvalue %[[VAL_297]], {{.*}}[47]
    // CHECK:           %[[VAL_386:.*]] = llvm.insertvalue %[[VAL_313]], {{.*}}[55]
    // CHECK:           %[[VAL_394:.*]] = llvm.insertvalue %[[VAL_329]], {{.*}}[63]
    %108 = triton_gpu.convert_layout %arg : tensor<1024x32xf16, #dpas> -> tensor<1024x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas, kWidth = 2}>>
    tt.return
  }
}

// -----

#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [32, 1], repCluster = [2, 2], A = [8, 16], B = [16, 32], C = [8, 32]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 32 : i32, triton_gpu.target = "xpu:DEVICE_ARCH.PVC", "triton_gpu.threads-per-warp" = 16 : i32, triton_intel_gpu.min_sg_size = 16 : i32, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block} {
  // CHECK-LABEL: convert_dpas_to_dot_rep_cluster_2_2
  // CHECK-SAME:  %[[VAL_0:.*]]: !llvm.struct<({{.*}})>) attributes {triton_gen.intel_reqd_sub_group_size = [16 : i32], triton_gen.max_work_group_size = [512 : i32, 1 : i32, 1 : i32]} {
  tt.func public @convert_dpas_to_dot_rep_cluster_2_2(%arg: tensor<1024x32xf16, #dpas>) {
    // COM: The repetitions order of dpas layout when the GEMM tiling is clustered as repCluster [2, 2]:
    // COM:   - 0, 1, 2, 3, 4, 5, 6, 7.
    // COM: The repetitions order of dot layout when the GEMM tiling is clustered as repCluster [2, 2]:
    // COM:   - 0, 2, 1, 3, 4, 6, 5, 7.
    // CHECK:           %[[VAL_81:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[REP_0:.*]] = llvm.insertelement {{.*}}, {{.*}}{{\[}}%[[VAL_81]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_98:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[REP_1:.*]] = llvm.insertelement {{.*}}, {{.*}}{{\[}}%[[VAL_98]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_115:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[REP_2:.*]] = llvm.insertelement {{.*}}, {{.*}}{{\[}}%[[VAL_115]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_132:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[REP_3:.*]] = llvm.insertelement {{.*}}, {{.*}}{{\[}}%[[VAL_132]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_149:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[REP_4:.*]] = llvm.insertelement {{.*}}, {{.*}}{{\[}}%[[VAL_149]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_166:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[REP_5:.*]] = llvm.insertelement {{.*}}, {{.*}}{{\[}}%[[VAL_166]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_183:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[REP_6:.*]] = llvm.insertelement {{.*}}, {{.*}}{{\[}}%[[VAL_183]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_200:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[REP_7:.*]] = llvm.insertelement {{.*}}, {{.*}}{{\[}}%[[VAL_200]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_216:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[VAL_217:.*]] = llvm.extractelement %[[REP_0]]{{\[}}%[[VAL_216]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_232:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[VAL_233:.*]] = llvm.extractelement %[[REP_2]]{{\[}}%[[VAL_232]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_248:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[VAL_249:.*]] = llvm.extractelement %[[REP_1]]{{\[}}%[[VAL_248]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_264:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[VAL_265:.*]] = llvm.extractelement %[[REP_3]]{{\[}}%[[VAL_264]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_280:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[VAL_281:.*]] = llvm.extractelement %[[REP_4]]{{\[}}%[[VAL_280]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_296:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[VAL_297:.*]] = llvm.extractelement %[[REP_6]]{{\[}}%[[VAL_296]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_312:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[VAL_313:.*]] = llvm.extractelement %[[REP_5]]{{\[}}%[[VAL_312]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_328:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[VAL_329:.*]] = llvm.extractelement %[[REP_7]]{{\[}}%[[VAL_328]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_338:.*]] = llvm.insertvalue %[[VAL_217]], {{.*}}[7]
    // CHECK:           %[[VAL_346:.*]] = llvm.insertvalue %[[VAL_233]], {{.*}}[15]
    // CHECK:           %[[VAL_354:.*]] = llvm.insertvalue %[[VAL_249]], {{.*}}[23]
    // CHECK:           %[[VAL_362:.*]] = llvm.insertvalue %[[VAL_265]], {{.*}}[31]
    // CHECK:           %[[VAL_370:.*]] = llvm.insertvalue %[[VAL_281]], {{.*}}[39]
    // CHECK:           %[[VAL_378:.*]] = llvm.insertvalue %[[VAL_297]], {{.*}}[47]
    // CHECK:           %[[VAL_386:.*]] = llvm.insertvalue %[[VAL_313]], {{.*}}[55]
    // CHECK:           %[[VAL_394:.*]] = llvm.insertvalue %[[VAL_329]], {{.*}}[63]
    %108 = triton_gpu.convert_layout %arg : tensor<1024x32xf16, #dpas> -> tensor<1024x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas, kWidth = 2}>>
    tt.return
  }
}

// -----

#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [32, 1], repCluster = [4, 2], A = [8, 16], B = [16, 32], C = [8, 32]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 32 : i32, triton_gpu.target = "xpu:DEVICE_ARCH.PVC", "triton_gpu.threads-per-warp" = 16 : i32, triton_intel_gpu.min_sg_size = 16 : i32, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block} {
  // CHECK-LABEL: convert_dpas_to_dot_rep_cluster_4_2
  // CHECK-SAME:  %[[VAL_0:.*]]: !llvm.struct<({{.*}})>) attributes {triton_gen.intel_reqd_sub_group_size = [16 : i32], triton_gen.max_work_group_size = [512 : i32, 1 : i32, 1 : i32]} {
  tt.func public @convert_dpas_to_dot_rep_cluster_4_2(%arg: tensor<1024x32xf16, #dpas>) {
    // COM: The repetitions order of dpas layout when the GEMM tiling is clustered as repCluster [4, 2]:
    // COM:   - 0, 1, 2, 3, 4, 5, 6, 7.
    // COM: The repetitions order of dot layout when the GEMM tiling is clustered as repCluster [4, 2]:
    // COM:   - 0, 2, 4, 6, 1, 3, 5, 7.
    // CHECK:           %[[VAL_81:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[REP_0:.*]] = llvm.insertelement {{.*}}, {{.*}}{{\[}}%[[VAL_81]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_98:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[REP_1:.*]] = llvm.insertelement {{.*}}, {{.*}}{{\[}}%[[VAL_98]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_115:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[REP_2:.*]] = llvm.insertelement {{.*}}, {{.*}}{{\[}}%[[VAL_115]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_132:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[REP_3:.*]] = llvm.insertelement {{.*}}, {{.*}}{{\[}}%[[VAL_132]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_149:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[REP_4:.*]] = llvm.insertelement {{.*}}, {{.*}}{{\[}}%[[VAL_149]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_166:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[REP_5:.*]] = llvm.insertelement {{.*}}, {{.*}}{{\[}}%[[VAL_166]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_183:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[REP_6:.*]] = llvm.insertelement {{.*}}, {{.*}}{{\[}}%[[VAL_183]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_200:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[REP_7:.*]] = llvm.insertelement {{.*}}, {{.*}}{{\[}}%[[VAL_200]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_216:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[VAL_217:.*]] = llvm.extractelement %[[REP_0]]{{\[}}%[[VAL_216]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_232:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[VAL_233:.*]] = llvm.extractelement %[[REP_2]]{{\[}}%[[VAL_232]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_248:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[VAL_249:.*]] = llvm.extractelement %[[REP_4]]{{\[}}%[[VAL_248]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_264:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[VAL_265:.*]] = llvm.extractelement %[[REP_6]]{{\[}}%[[VAL_264]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_280:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[VAL_281:.*]] = llvm.extractelement %[[REP_1]]{{\[}}%[[VAL_280]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_296:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[VAL_297:.*]] = llvm.extractelement %[[REP_3]]{{\[}}%[[VAL_296]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_312:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[VAL_313:.*]] = llvm.extractelement %[[REP_5]]{{\[}}%[[VAL_312]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_328:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[VAL_329:.*]] = llvm.extractelement %[[REP_7]]{{\[}}%[[VAL_328]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_338:.*]] = llvm.insertvalue %[[VAL_217]], {{.*}}[7]
    // CHECK:           %[[VAL_346:.*]] = llvm.insertvalue %[[VAL_233]], {{.*}}[15]
    // CHECK:           %[[VAL_354:.*]] = llvm.insertvalue %[[VAL_249]], {{.*}}[23]
    // CHECK:           %[[VAL_362:.*]] = llvm.insertvalue %[[VAL_265]], {{.*}}[31]
    // CHECK:           %[[VAL_370:.*]] = llvm.insertvalue %[[VAL_281]], {{.*}}[39]
    // CHECK:           %[[VAL_378:.*]] = llvm.insertvalue %[[VAL_297]], {{.*}}[47]
    // CHECK:           %[[VAL_386:.*]] = llvm.insertvalue %[[VAL_313]], {{.*}}[55]
    // CHECK:           %[[VAL_394:.*]] = llvm.insertvalue %[[VAL_329]], {{.*}}[63]
    %108 = triton_gpu.convert_layout %arg : tensor<1024x32xf16, #dpas> -> tensor<1024x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas, kWidth = 2}>>
    tt.return
  }
}
