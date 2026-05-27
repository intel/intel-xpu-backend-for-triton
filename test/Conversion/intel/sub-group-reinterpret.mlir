// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm --convert-tritongen-to-llvm | FileCheck %s

// 32x32xf8E5M2 mma -> dot_a layout conversion via sub-group bitcast shuffle.
// The conversion reinterprets packed f8E5M2 elements by calling the
// GenISA_SubgroupBitcastShuffle intrinsic.

#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 4, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1], A = [8, 32], B = [32, 16], C = [8, 16]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>

// CHECK: llvm.func spir_funccc @llvm.genx.GenISA.SubgroupBitcastShuffle.v8i16.v16i8(vector<16xi8>) -> vector<8xi16>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @test_reinterpret(
  // CHECK-SAME:      %[[ARG0:.*]]: !llvm.struct<(i8, i8, {{.*}})>,
  tt.func @test_reinterpret(%arg0: tensor<32x32xf8E5M2, #mma>) -> tensor<32x32xf8E5M2, #dot_a> {
    // CHECK:           %[[DPAS_D_0:.*]] = llvm.extractvalue %[[ARG0]][0] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_1:.*]] = llvm.extractvalue %[[ARG0]][1] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_2:.*]] = llvm.extractvalue %[[ARG0]][2] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_3:.*]] = llvm.extractvalue %[[ARG0]][3] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_4:.*]] = llvm.extractvalue %[[ARG0]][4] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_5:.*]] = llvm.extractvalue %[[ARG0]][5] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_6:.*]] = llvm.extractvalue %[[ARG0]][6] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_7:.*]] = llvm.extractvalue %[[ARG0]][7] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_8:.*]] = llvm.extractvalue %[[ARG0]][8] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_9:.*]] = llvm.extractvalue %[[ARG0]][9] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_10:.*]] = llvm.extractvalue %[[ARG0]][10] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_11:.*]] = llvm.extractvalue %[[ARG0]][11] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_12:.*]] = llvm.extractvalue %[[ARG0]][12] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_13:.*]] = llvm.extractvalue %[[ARG0]][13] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_14:.*]] = llvm.extractvalue %[[ARG0]][14] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_15:.*]] = llvm.extractvalue %[[ARG0]][15] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_16:.*]] = llvm.extractvalue %[[ARG0]][16] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_17:.*]] = llvm.extractvalue %[[ARG0]][17] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_18:.*]] = llvm.extractvalue %[[ARG0]][18] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_19:.*]] = llvm.extractvalue %[[ARG0]][19] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_20:.*]] = llvm.extractvalue %[[ARG0]][20] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_21:.*]] = llvm.extractvalue %[[ARG0]][21] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_22:.*]] = llvm.extractvalue %[[ARG0]][22] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_23:.*]] = llvm.extractvalue %[[ARG0]][23] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_24:.*]] = llvm.extractvalue %[[ARG0]][24] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_25:.*]] = llvm.extractvalue %[[ARG0]][25] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_26:.*]] = llvm.extractvalue %[[ARG0]][26] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_27:.*]] = llvm.extractvalue %[[ARG0]][27] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_28:.*]] = llvm.extractvalue %[[ARG0]][28] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_29:.*]] = llvm.extractvalue %[[ARG0]][29] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_30:.*]] = llvm.extractvalue %[[ARG0]][30] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_31:.*]] = llvm.extractvalue %[[ARG0]][31] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_32:.*]] = llvm.extractvalue %[[ARG0]][32] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_33:.*]] = llvm.extractvalue %[[ARG0]][33] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_34:.*]] = llvm.extractvalue %[[ARG0]][34] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_35:.*]] = llvm.extractvalue %[[ARG0]][35] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_36:.*]] = llvm.extractvalue %[[ARG0]][36] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_37:.*]] = llvm.extractvalue %[[ARG0]][37] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_38:.*]] = llvm.extractvalue %[[ARG0]][38] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_39:.*]] = llvm.extractvalue %[[ARG0]][39] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_40:.*]] = llvm.extractvalue %[[ARG0]][40] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_41:.*]] = llvm.extractvalue %[[ARG0]][41] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_42:.*]] = llvm.extractvalue %[[ARG0]][42] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_43:.*]] = llvm.extractvalue %[[ARG0]][43] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_44:.*]] = llvm.extractvalue %[[ARG0]][44] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_45:.*]] = llvm.extractvalue %[[ARG0]][45] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_46:.*]] = llvm.extractvalue %[[ARG0]][46] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_47:.*]] = llvm.extractvalue %[[ARG0]][47] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_48:.*]] = llvm.extractvalue %[[ARG0]][48] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_49:.*]] = llvm.extractvalue %[[ARG0]][49] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_50:.*]] = llvm.extractvalue %[[ARG0]][50] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_51:.*]] = llvm.extractvalue %[[ARG0]][51] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_52:.*]] = llvm.extractvalue %[[ARG0]][52] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_53:.*]] = llvm.extractvalue %[[ARG0]][53] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_54:.*]] = llvm.extractvalue %[[ARG0]][54] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_55:.*]] = llvm.extractvalue %[[ARG0]][55] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_56:.*]] = llvm.extractvalue %[[ARG0]][56] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_57:.*]] = llvm.extractvalue %[[ARG0]][57] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_58:.*]] = llvm.extractvalue %[[ARG0]][58] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_59:.*]] = llvm.extractvalue %[[ARG0]][59] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_60:.*]] = llvm.extractvalue %[[ARG0]][60] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_61:.*]] = llvm.extractvalue %[[ARG0]][61] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_62:.*]] = llvm.extractvalue %[[ARG0]][62] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK:           %[[DPAS_D_63:.*]] = llvm.extractvalue %[[ARG0]][63] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>

    // COM: Two 2x DPAS D packed to 1x DPAS A of fp8 type.
    // COM: The order is a0, a8, a1, a9, a2, a10, a3, a11, a4, a12, a5, a13, a6, a14, a7, a15.
    // CHECK:           %[[DPAS_A:.*]] = llvm.mlir.undef : vector<16xi8>
    // CHECK:           %[[MLIR_1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[DPAS_A_0:.*]] = llvm.insertelement %[[DPAS_D_0]], %[[DPAS_A]]{{\[}}%[[MLIR_1]] : i32] : vector<16xi8>
    // CHECK:           %[[MLIR_2:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:           %[[DPAS_A_1:.*]] = llvm.insertelement %[[DPAS_D_8]], %[[DPAS_A_0]]{{\[}}%[[MLIR_2]] : i32] : vector<16xi8>
    // CHECK:           %[[MLIR_3:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK:           %[[DPAS_A_2:.*]] = llvm.insertelement %[[DPAS_D_1]], %[[DPAS_A_1]]{{\[}}%[[MLIR_3]] : i32] : vector<16xi8>
    // CHECK:           %[[MLIR_4:.*]] = llvm.mlir.constant(3 : i32) : i32
    // CHECK:           %[[DPAS_A_3:.*]] = llvm.insertelement %[[DPAS_D_9]], %[[DPAS_A_2]]{{\[}}%[[MLIR_4]] : i32] : vector<16xi8>
    // CHECK:           %[[MLIR_5:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK:           %[[DPAS_A_4:.*]] = llvm.insertelement %[[DPAS_D_2]], %[[DPAS_A_3]]{{\[}}%[[MLIR_5]] : i32] : vector<16xi8>
    // CHECK:           %[[MLIR_6:.*]] = llvm.mlir.constant(5 : i32) : i32
    // CHECK:           %[[DPAS_A_5:.*]] = llvm.insertelement %[[DPAS_D_10]], %[[DPAS_A_4]]{{\[}}%[[MLIR_6]] : i32] : vector<16xi8>
    // CHECK:           %[[MLIR_7:.*]] = llvm.mlir.constant(6 : i32) : i32
    // CHECK:           %[[DPAS_A_6:.*]] = llvm.insertelement %[[DPAS_D_3]], %[[DPAS_A_5]]{{\[}}%[[MLIR_7]] : i32] : vector<16xi8>
    // CHECK:           %[[MLIR_8:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[DPAS_A_7:.*]] = llvm.insertelement %[[DPAS_D_11]], %[[DPAS_A_6]]{{\[}}%[[MLIR_8]] : i32] : vector<16xi8>
    // CHECK:           %[[MLIR_9:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK:           %[[DPAS_A_8:.*]] = llvm.insertelement %[[DPAS_D_4]], %[[DPAS_A_7]]{{\[}}%[[MLIR_9]] : i32] : vector<16xi8>
    // CHECK:           %[[MLIR_10:.*]] = llvm.mlir.constant(9 : i32) : i32
    // CHECK:           %[[DPAS_A_9:.*]] = llvm.insertelement %[[DPAS_D_12]], %[[DPAS_A_8]]{{\[}}%[[MLIR_10]] : i32] : vector<16xi8>
    // CHECK:           %[[MLIR_11:.*]] = llvm.mlir.constant(10 : i32) : i32
    // CHECK:           %[[DPAS_A_10:.*]] = llvm.insertelement %[[DPAS_D_5]], %[[DPAS_A_9]]{{\[}}%[[MLIR_11]] : i32] : vector<16xi8>
    // CHECK:           %[[MLIR_12:.*]] = llvm.mlir.constant(11 : i32) : i32
    // CHECK:           %[[DPAS_A_11:.*]] = llvm.insertelement %[[DPAS_D_13]], %[[DPAS_A_10]]{{\[}}%[[MLIR_12]] : i32] : vector<16xi8>
    // CHECK:           %[[MLIR_13:.*]] = llvm.mlir.constant(12 : i32) : i32
    // CHECK:           %[[DPAS_A_12:.*]] = llvm.insertelement %[[DPAS_D_6]], %[[DPAS_A_11]]{{\[}}%[[MLIR_13]] : i32] : vector<16xi8>
    // CHECK:           %[[MLIR_14:.*]] = llvm.mlir.constant(13 : i32) : i32
    // CHECK:           %[[DPAS_A_13:.*]] = llvm.insertelement %[[DPAS_D_14]], %[[DPAS_A_12]]{{\[}}%[[MLIR_14]] : i32] : vector<16xi8>
    // CHECK:           %[[MLIR_15:.*]] = llvm.mlir.constant(14 : i32) : i32
    // CHECK:           %[[DPAS_A_14:.*]] = llvm.insertelement %[[DPAS_D_7]], %[[DPAS_A_13]]{{\[}}%[[MLIR_15]] : i32] : vector<16xi8>
    // CHECK:           %[[MLIR_16:.*]] = llvm.mlir.constant(15 : i32) : i32
    // CHECK:           %[[DPAS_A_15:.*]] = llvm.insertelement %[[DPAS_D_15]], %[[DPAS_A_14]]{{\[}}%[[MLIR_16]] : i32] : vector<16xi8>
    // CHECK:           %[[PACKED_DPAS_A:.*]] = llvm.call spir_funccc @llvm.genx.GenISA.SubgroupBitcastShuffle.v8i16.v16i8(%[[DPAS_A_15]])
    // CHECK:           %[[UNPACKED_DPAS_A:.*]] = llvm.bitcast %[[PACKED_DPAS_A]] : vector<8xi16> to vector<16xi8>

    // COM: The 2nd DPAS operands shuffle.
    // COM: The base is a16 instead of a0, and the order is a16, a24, a17, a25, a18, a26, a19, a27, a20, a28, a21, a29, a22, a30, a23, a31.
    // CHECK:           %[[DPAS_A:.*]] = llvm.mlir.undef : vector<16xi8>
    // CHECK:           %[[MLIR_34:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[DPAS_A_16:.*]] = llvm.insertelement %[[DPAS_D_16]], %[[DPAS_A]]{{\[}}%[[MLIR_34]] : i32] : vector<16xi8>
    // CHECK:           %[[MLIR_35:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:           %[[DPAS_A_17:.*]] = llvm.insertelement %[[DPAS_D_24]], %[[DPAS_A_16]]{{\[}}%[[MLIR_35]] : i32] : vector<16xi8>
    // CHECK:           %[[MLIR_36:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK:           %[[DPAS_A_18:.*]] = llvm.insertelement %[[DPAS_D_17]], %[[DPAS_A_17]]{{\[}}%[[MLIR_36]] : i32] : vector<16xi8>
    // CHECK:           %[[MLIR_37:.*]] = llvm.mlir.constant(3 : i32) : i32
    // CHECK:           %[[DPAS_A_19:.*]] = llvm.insertelement %[[DPAS_D_25]], %[[DPAS_A_18]]{{\[}}%[[MLIR_37]] : i32] : vector<16xi8>
    // CHECK:           %[[MLIR_38:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK:           %[[DPAS_A_20:.*]] = llvm.insertelement %[[DPAS_D_18]], %[[DPAS_A_19]]{{\[}}%[[MLIR_38]] : i32] : vector<16xi8>
    // CHECK:           %[[MLIR_39:.*]] = llvm.mlir.constant(5 : i32) : i32
    // CHECK:           %[[DPAS_A_21:.*]] = llvm.insertelement %[[DPAS_D_26]], %[[DPAS_A_20]]{{\[}}%[[MLIR_39]] : i32] : vector<16xi8>
    // CHECK:           %[[MLIR_40:.*]] = llvm.mlir.constant(6 : i32) : i32
    // CHECK:           %[[DPAS_A_22:.*]] = llvm.insertelement %[[DPAS_D_19]], %[[DPAS_A_21]]{{\[}}%[[MLIR_40]] : i32] : vector<16xi8>
    // CHECK:           %[[MLIR_41:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[DPAS_A_23:.*]] = llvm.insertelement %[[DPAS_D_27]], %[[DPAS_A_22]]{{\[}}%[[MLIR_41]] : i32] : vector<16xi8>
    // CHECK:           %[[MLIR_42:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK:           %[[DPAS_A_24:.*]] = llvm.insertelement %[[DPAS_D_20]], %[[DPAS_A_23]]{{\[}}%[[MLIR_42]] : i32] : vector<16xi8>
    // CHECK:           %[[MLIR_43:.*]] = llvm.mlir.constant(9 : i32) : i32
    // CHECK:           %[[DPAS_A_25:.*]] = llvm.insertelement %[[DPAS_D_28]], %[[DPAS_A_24]]{{\[}}%[[MLIR_43]] : i32] : vector<16xi8>
    // CHECK:           %[[MLIR_44:.*]] = llvm.mlir.constant(10 : i32) : i32
    // CHECK:           %[[DPAS_A_26:.*]] = llvm.insertelement %[[DPAS_D_21]], %[[DPAS_A_25]]{{\[}}%[[MLIR_44]] : i32] : vector<16xi8>
    // CHECK:           %[[MLIR_45:.*]] = llvm.mlir.constant(11 : i32) : i32
    // CHECK:           %[[DPAS_A_27:.*]] = llvm.insertelement %[[DPAS_D_29]], %[[DPAS_A_26]]{{\[}}%[[MLIR_45]] : i32] : vector<16xi8>
    // CHECK:           %[[MLIR_46:.*]] = llvm.mlir.constant(12 : i32) : i32
    // CHECK:           %[[DPAS_A_28:.*]] = llvm.insertelement %[[DPAS_D_22]], %[[DPAS_A_27]]{{\[}}%[[MLIR_46]] : i32] : vector<16xi8>
    // CHECK:           %[[MLIR_47:.*]] = llvm.mlir.constant(13 : i32) : i32
    // CHECK:           %[[DPAS_A_29:.*]] = llvm.insertelement %[[DPAS_D_30]], %[[DPAS_A_28]]{{\[}}%[[MLIR_47]] : i32] : vector<16xi8>
    // CHECK:           %[[MLIR_48:.*]] = llvm.mlir.constant(14 : i32) : i32
    // CHECK:           %[[DPAS_A_30:.*]] = llvm.insertelement %[[DPAS_D_23]], %[[DPAS_A_29]]{{\[}}%[[MLIR_48]] : i32] : vector<16xi8>
    // CHECK:           %[[MLIR_49:.*]] = llvm.mlir.constant(15 : i32) : i32
    // CHECK:           %[[DPAS_A_31:.*]] = llvm.insertelement %[[DPAS_D_31]], %[[DPAS_A_30]]{{\[}}%[[MLIR_49]] : i32] : vector<16xi8>
    // CHECK:           %[[PACKED_DPAS_A:.*]] = llvm.call spir_funccc @llvm.genx.GenISA.SubgroupBitcastShuffle.v8i16.v16i8(%[[DPAS_A_31]]) {convergent, function_type = !llvm.func<vector<8xi16> (vector<16xi8>)>, linkage = #llvm.linkage<external>, no_unwind, sym_name = "llvm.genx.GenISA.SubgroupBitcastShuffle.v8i16.v16i8", visibility_ = 0 : i64, will_return} : (vector<16xi8>) -> vector<8xi16>
    // CHECK:           %[[UNPACKED_DPAS_A:.*]] = llvm.bitcast %[[PACKED_DPAS_A]] : vector<8xi16> to vector<16xi8>

    // COM: The remianing 2 DPAS D to DPAS A shuffle.
    // CHECK-COUNT-2: llvm.call spir_funccc @llvm.genx.GenISA.SubgroupBitcastShuffle.v8i16.v16i8(
    %0 = ttg.convert_layout %arg0 : tensor<32x32xf8E5M2, #mma> -> tensor<32x32xf8E5M2, #dot_a>
    tt.return %0 : tensor<32x32xf8E5M2, #dot_a>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [2, 1], A = [16, 8], B = [8, 16], C = [16, 16]}>

module attributes {"ttg.num-warps" = 1 : i32, ttg.shared = 1280 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK:   llvm.func spir_funccc @llvm.genx.GenISA.SubgroupBitcastShuffle.v1i64.v4i16
  // CHECK-LABEL:   llvm.func spir_kernelcc @test_reinterpret(
  tt.func public @test_reinterpret(%arg0: tensor<16x16xf32, #mma>, %arg1: tensor<16x16xf16, #mma>)  -> (tensor<16x16xf32, #blocked>, tensor<16x16xf16, #blocked>) {
    // CHECK-COUNT-4:  llvm.call spir_funccc @llvm.genx.GenISA.SubgroupBitcastShuffle.v1i64.v4i16
    %1 = ttg.convert_layout %arg1 : tensor<16x16xf16, #mma> -> tensor<16x16xf16, #blocked>
    // COM: This should be converted to a call to GenISA.SubgroupBitcastShuffle, but IGC currently don't support bitcast >= 128 bits.
    // CHECK-NOT: llvm.call spir_funccc llvm.genx.GenISA.SubgroupBitcastShuffle
    %0 = ttg.convert_layout %arg0 {allocation.offset = 0 : i32} : tensor<16x16xf32, #mma> -> tensor<16x16xf32, #blocked>
    tt.return %0, %1 : tensor<16x16xf32, #blocked>, tensor<16x16xf16, #blocked>
  }
}
