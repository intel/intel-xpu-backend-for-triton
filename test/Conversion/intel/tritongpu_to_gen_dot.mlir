// RUN: triton-opt %s -split-input-file --allocate-shared-memory  --convert-triton-intel-gpu-to-llvm --cse -canonicalize | FileCheck %s --implicit-check-not=llvm.inline_asm --check-prefixes=CHECK,NO-AGGRESSIVE-REUSE
// RUN: env TRITON_INTEL_AGGRESSIVE_DPAS_REUSE=1 triton-opt %s -split-input-file --allocate-shared-memory  --convert-triton-intel-gpu-to-llvm --cse -canonicalize | FileCheck %s --implicit-check-not=llvm.inline_asm  --check-prefixes=CHECK,AGGRESSIVE-REUSE

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#dpas, kWidth=1}>
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#dpas, kWidth=2}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK:  llvm.func spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fi(i32, vector<8xi16>, vector<8xi32>, vector<8xf32>, i32) -> vector<8xf32> attributes {convergent, memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, no_unwind, will_return}
  // CHECK-LABEL: dot_f32_f16_f16_f32_1
  tt.func @dot_f32_f16_f16_f32_1(%a: tensor<8x16xf16, #dot_operand_a>, %b: tensor<16x16xf16, #dot_operand_b>, %c: tensor<8x16xf32, #dpas>) {
    // CHECK: llvm.call spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fi(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}} : (i32, vector<8xi16>, vector<8xi32>, vector<8xf32>, i32) -> vector<8xf32>
    %0 = tt.dot %a, %b, %c, inputPrecision = tf32 : tensor<8x16xf16, #dot_operand_a> * tensor<16x16xf16, #dot_operand_b> -> tensor<8x16xf32, #dpas>
    tt.return
  }
}

// -----

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#dpas, kWidth=1}>
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#dpas, kWidth=2}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK: llvm.func spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fi(i32, vector<8xi16>, vector<8xi32>, vector<8xf32>, i32) -> vector<8xf32> attributes {convergent, memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, no_unwind, will_return}
  // CHECK-LABEL: dot_f32_f16_f16_f32_2
  tt.func @dot_f32_f16_f16_f32_2(%a: tensor<16x16xf16, #dot_operand_a>, %b: tensor<16x16xf16, #dot_operand_b>, %c: tensor<16x16xf32, #dpas>) {
    // COM: 2 repetitions along axis for M.
    // CHECK-COUNT-2: llvm.call spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fi(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}} : (i32, vector<8xi16>, vector<8xi32>, vector<8xf32>, i32) -> vector<8xf32>
    %0 = tt.dot %a, %b, %c, inputPrecision = tf32 : tensor<16x16xf16, #dot_operand_a> * tensor<16x16xf16, #dot_operand_b> -> tensor<16x16xf32, #dpas>
    tt.return
  }
}

// -----

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 4, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#dpas, kWidth=2}>
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#dpas, kWidth=4}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK:   llvm.func spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iS0_i(i32, vector<8xi16>, vector<8xi32>, vector<8xi32>, i32) -> vector<8xi32> attributes {convergent, memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, no_unwind, will_return}
  // CHECK-LABEL: dot_i32_i8_i8_i32_1
  tt.func @dot_i32_i8_i8_i32_1(%a: tensor<8x32xi8, #dot_operand_a>, %b: tensor<32x16xi8, #dot_operand_b>, %c: tensor<8x16xi32, #dpas>) {
    // CHECK: llvm.call spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iS0_i(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}} : (i32, vector<8xi16>, vector<8xi32>, vector<8xi32>, i32) -> vector<8xi32>
    %0 = tt.dot %a, %b, %c, inputPrecision = tf32 : tensor<8x32xi8, #dot_operand_a> * tensor<32x16xi8, #dot_operand_b> -> tensor<8x16xi32, #dpas>
    tt.return
  }
}

// -----

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 4, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#dpas, kWidth=2}>
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#dpas, kWidth=4}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK: llvm.func spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iS0_i(i32, vector<8xi16>, vector<8xi32>, vector<8xi32>, i32) -> vector<8xi32> attributes {convergent, memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, no_unwind, will_return}
  // CHECK-LABEL: dot_i32_i8_i8_i32_2
  tt.func @dot_i32_i8_i8_i32_2(%a: tensor<8x64xi8, #dot_operand_a>, %b: tensor<64x16xi8, #dot_operand_b>, %c: tensor<8x16xi32, #dpas>) {
    // COM: 2 repetition along axis for K.
    // CHECK-COUNT2: llvm.call spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iS0_i(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}} : (i32, vector<8xi16>, vector<8xi32>, vector<8xi32>, i32) -> vector<8xi32>
    %0 = tt.dot %a, %b, %c, inputPrecision = tf32 : tensor<8x64xi8, #dot_operand_a> * tensor<64x16xi8, #dot_operand_b> -> tensor<8x16xi32, #dpas>
    tt.return
  }
}

// -----

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#dpas, kWidth=1}>
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#dpas, kWidth=1}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @dot_f32_tf32_tf32_f32_1(
  // CHECK-SAME:    %[[A:.*]]: !llvm.struct<(f32, f32, f32, f32)>, %[[B:.*]]: !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>,
  // CHECK-SAME:    %[[C:.*]]: !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>) attributes {intel_reqd_sub_group_size = 32 : i32, reqd_work_group_size = array<i32: 32, 1, 1>} {
  tt.func @dot_f32_tf32_tf32_f32_1(%a: tensor<8x8xf32, #dot_operand_a>, %b: tensor<8x16xf32, #dot_operand_b>, %c: tensor<8x16xf32, #dpas>) {
    // COM: To simplify, only check RTNE and its usage for the last element of A, B, C
    // CHECK: %[[A_LAST_VAL:.*]] = llvm.extractvalue %[[A]][3]
    // CHECK: %[[A_RTNE_VAL:.*]] = llvm.call spir_funccc @_Z25__spirv_RoundFToTF32INTELf(%[[A_LAST_VAL]])
    // CHECK: %[[A_0:.*]] = llvm.insertelement %[[A_RTNE_VAL]], %{{.*}}{{\[}}%{{.*}} : i32] : vector<4xf32>
    // CHECK: %[[B_LAST_VAL:.*]] = llvm.extractvalue %[[B]][7]
    // CHECK: %[[B_RTNE_VAL:.*]] = llvm.call spir_funccc @_Z25__spirv_RoundFToTF32INTELf(%[[B_LAST_VAL]])
    // CHECK: %[[B_0:.*]] = llvm.insertelement %[[B_RTNE_VAL]], %{{.*}}{{\[}}%{{.*}} : i32] : vector<8xf32>
    // CHECK: %[[C_LAST_VAL:.*]] = llvm.extractvalue %[[C]][7]
    // CHECK: %[[C_0:.*]] = llvm.insertelement %[[C_LAST_VAL]], %{{.*}}{{\[}}%{{.*}} : i32] : vector<8xf32>
    // CHECK: llvm.call spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv4_fDv8_fS0_i(%{{.*}}, %[[A_0]], %[[B_0]], %[[C_0]], %{{.*}}} : (i32, vector<4xf32>, vector<8xf32>, vector<8xf32>, i32) -> vector<8xf32>
    %0 = tt.dot %a, %b, %c, inputPrecision = tf32 : tensor<8x8xf32, #dot_operand_a> * tensor<8x16xf32, #dot_operand_b> -> tensor<8x16xf32, #dpas>
    tt.return
  }
}

// -----

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#dpas, kWidth=1}>
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#dpas, kWidth=1}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: dot_f32_tf32_tf32_f32_2
  tt.func @dot_f32_tf32_tf32_f32_2(%a: tensor<8x8xf32, #dot_operand_a>, %b: tensor<8x32xf32, #dot_operand_b>, %c: tensor<8x32xf32, #dpas>) {
    // COM: 2 repetitions along axis for N.
    // CHECK-COUNT-2: llvm.call spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv4_fDv8_fS0_i(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}} : (i32, vector<4xf32>, vector<8xf32>, vector<8xf32>, i32) -> vector<8xf32>
    %0 = tt.dot %a, %b, %c, inputPrecision = tf32 : tensor<8x8xf32, #dot_operand_a> * tensor<8x32xf32, #dot_operand_b> -> tensor<8x32xf32, #dpas>
    tt.return
  }
}

// -----

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 2]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#dpas, kWidth=1}>
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#dpas, kWidth=2}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK: llvm.func spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fi(i32, vector<8xi16>, vector<8xi32>, vector<8xf32>, i32) -> vector<8xf32> attributes {convergent, memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, no_unwind, will_return}
  // CHECK-LABEL: llvm.func spir_kernelcc @dot_rep_cluster_4_2(
  // CHECK-SAME:    %[[A:.*]]: !llvm.struct<(f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16)>, %[[B:.*]]: !llvm.struct<(f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16)>,
  // CHECK-SAME:    %[[C:.*]]: !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>) attributes {intel_reqd_sub_group_size = 16 : i32, reqd_work_group_size = array<i32: 16, 1, 1>} {
  tt.func @dot_rep_cluster_4_2(%a: tensor<32x32xf16, #dot_operand_a>, %b: tensor<32x32xf16, #dot_operand_b>, %c: tensor<32x32xf32, #dpas>) {
    // CHECK:           %[[VAL_3:.*]] = llvm.mlir.undef : vector<8xf32>
    // CHECK:           %[[CST_15:.*]] = llvm.mlir.constant(15 : i32) : i32
    // CHECK:           %[[CST_14:.*]] = llvm.mlir.constant(14 : i32) : i32
    // CHECK:           %[[CST_13:.*]] = llvm.mlir.constant(13 : i32) : i32
    // CHECK:           %[[CST_12:.*]] = llvm.mlir.constant(12 : i32) : i32
    // CHECK:           %[[CST_11:.*]] = llvm.mlir.constant(11 : i32) : i32
    // CHECK:           %[[CST_10:.*]] = llvm.mlir.constant(10 : i32) : i32
    // CHECK:           %[[CST_9:.*]] = llvm.mlir.constant(9 : i32) : i32
    // CHECK:           %[[CST_8:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK:           %[[VAL_12:.*]] = llvm.mlir.undef : vector<16xf16>
    // CHECK:           %[[CST_7:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[CST_6:.*]] = llvm.mlir.constant(6 : i32) : i32
    // CHECK:           %[[CST_5:.*]] = llvm.mlir.constant(5 : i32) : i32
    // CHECK:           %[[CST_4:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK:           %[[CST_3:.*]] = llvm.mlir.constant(3 : i32) : i32
    // CHECK:           %[[CST_2:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK:           %[[CST_1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:           %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_21:.*]] = llvm.mlir.undef : vector<8xf16>

    // COM: The shape of Operand A replica is [4, 2]
    // COM: The replica order are [0, 4]
    // COM:                       [1, 5]
    // COM:                       [2, 6]
    // COM:                       [3, 7]

    // CHECK:           %[[VAL_22:.*]] = llvm.extractvalue %[[A]][0]
    // CHECK:           %[[VAL_23:.*]] = llvm.extractvalue %[[A]][1]
    // CHECK:           %[[VAL_24:.*]] = llvm.extractvalue %[[A]][2]
    // CHECK:           %[[VAL_25:.*]] = llvm.extractvalue %[[A]][3]
    // CHECK:           %[[VAL_26:.*]] = llvm.extractvalue %[[A]][4]
    // CHECK:           %[[VAL_27:.*]] = llvm.extractvalue %[[A]][5]
    // CHECK:           %[[VAL_28:.*]] = llvm.extractvalue %[[A]][6]
    // CHECK:           %[[VAL_29:.*]] = llvm.extractvalue %[[A]][7]
    // CHECK:           %[[VAL_30:.*]] = llvm.extractvalue %[[A]][8]
    // CHECK:           %[[VAL_31:.*]] = llvm.extractvalue %[[A]][9]
    // CHECK:           %[[VAL_32:.*]] = llvm.extractvalue %[[A]][10]
    // CHECK:           %[[VAL_33:.*]] = llvm.extractvalue %[[A]][11]
    // CHECK:           %[[VAL_34:.*]] = llvm.extractvalue %[[A]][12]
    // CHECK:           %[[VAL_35:.*]] = llvm.extractvalue %[[A]][13]
    // CHECK:           %[[VAL_36:.*]] = llvm.extractvalue %[[A]][14]
    // CHECK:           %[[VAL_37:.*]] = llvm.extractvalue %[[A]][15]
    // CHECK:           %[[VAL_38:.*]] = llvm.extractvalue %[[A]][16]
    // CHECK:           %[[VAL_39:.*]] = llvm.extractvalue %[[A]][17]
    // CHECK:           %[[VAL_40:.*]] = llvm.extractvalue %[[A]][18]
    // CHECK:           %[[VAL_41:.*]] = llvm.extractvalue %[[A]][19]
    // CHECK:           %[[VAL_42:.*]] = llvm.extractvalue %[[A]][20]
    // CHECK:           %[[VAL_43:.*]] = llvm.extractvalue %[[A]][21]
    // CHECK:           %[[VAL_44:.*]] = llvm.extractvalue %[[A]][22]
    // CHECK:           %[[VAL_45:.*]] = llvm.extractvalue %[[A]][23]
    // CHECK:           %[[VAL_46:.*]] = llvm.extractvalue %[[A]][24]
    // CHECK:           %[[VAL_47:.*]] = llvm.extractvalue %[[A]][25]
    // CHECK:           %[[VAL_48:.*]] = llvm.extractvalue %[[A]][26]
    // CHECK:           %[[VAL_49:.*]] = llvm.extractvalue %[[A]][27]
    // CHECK:           %[[VAL_50:.*]] = llvm.extractvalue %[[A]][28]
    // CHECK:           %[[VAL_51:.*]] = llvm.extractvalue %[[A]][29]
    // CHECK:           %[[VAL_52:.*]] = llvm.extractvalue %[[A]][30]
    // CHECK:           %[[VAL_53:.*]] = llvm.extractvalue %[[A]][31]
    // CHECK:           %[[VAL_54:.*]] = llvm.extractvalue %[[A]][32]
    // CHECK:           %[[VAL_55:.*]] = llvm.extractvalue %[[A]][33]
    // CHECK:           %[[VAL_56:.*]] = llvm.extractvalue %[[A]][34]
    // CHECK:           %[[VAL_57:.*]] = llvm.extractvalue %[[A]][35]
    // CHECK:           %[[VAL_58:.*]] = llvm.extractvalue %[[A]][36]
    // CHECK:           %[[VAL_59:.*]] = llvm.extractvalue %[[A]][37]
    // CHECK:           %[[VAL_60:.*]] = llvm.extractvalue %[[A]][38]
    // CHECK:           %[[VAL_61:.*]] = llvm.extractvalue %[[A]][39]
    // CHECK:           %[[VAL_62:.*]] = llvm.extractvalue %[[A]][40]
    // CHECK:           %[[VAL_63:.*]] = llvm.extractvalue %[[A]][41]
    // CHECK:           %[[VAL_64:.*]] = llvm.extractvalue %[[A]][42]
    // CHECK:           %[[VAL_65:.*]] = llvm.extractvalue %[[A]][43]
    // CHECK:           %[[VAL_66:.*]] = llvm.extractvalue %[[A]][44]
    // CHECK:           %[[VAL_67:.*]] = llvm.extractvalue %[[A]][45]
    // CHECK:           %[[VAL_68:.*]] = llvm.extractvalue %[[A]][46]
    // CHECK:           %[[VAL_69:.*]] = llvm.extractvalue %[[A]][47]
    // CHECK:           %[[VAL_70:.*]] = llvm.extractvalue %[[A]][48]
    // CHECK:           %[[VAL_71:.*]] = llvm.extractvalue %[[A]][49]
    // CHECK:           %[[VAL_72:.*]] = llvm.extractvalue %[[A]][50]
    // CHECK:           %[[VAL_73:.*]] = llvm.extractvalue %[[A]][51]
    // CHECK:           %[[VAL_74:.*]] = llvm.extractvalue %[[A]][52]
    // CHECK:           %[[VAL_75:.*]] = llvm.extractvalue %[[A]][53]
    // CHECK:           %[[VAL_76:.*]] = llvm.extractvalue %[[A]][54]
    // CHECK:           %[[VAL_77:.*]] = llvm.extractvalue %[[A]][55]
    // CHECK:           %[[VAL_78:.*]] = llvm.extractvalue %[[A]][56]
    // CHECK:           %[[VAL_79:.*]] = llvm.extractvalue %[[A]][57]
    // CHECK:           %[[VAL_80:.*]] = llvm.extractvalue %[[A]][58]
    // CHECK:           %[[VAL_81:.*]] = llvm.extractvalue %[[A]][59]
    // CHECK:           %[[VAL_82:.*]] = llvm.extractvalue %[[A]][60]
    // CHECK:           %[[VAL_83:.*]] = llvm.extractvalue %[[A]][61]
    // CHECK:           %[[VAL_84:.*]] = llvm.extractvalue %[[A]][62]
    // CHECK:           %[[VAL_85:.*]] = llvm.extractvalue %[[A]][63]
    // CHECK:           %[[VAL_86:.*]] = llvm.insertelement %[[VAL_22]], %[[VAL_21]]{{\[}}%[[CST_0]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_87:.*]] = llvm.insertelement %[[VAL_23]], %[[VAL_86]]{{\[}}%[[CST_1]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_88:.*]] = llvm.insertelement %[[VAL_24]], %[[VAL_87]]{{\[}}%[[CST_2]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_89:.*]] = llvm.insertelement %[[VAL_25]], %[[VAL_88]]{{\[}}%[[CST_3]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_90:.*]] = llvm.insertelement %[[VAL_26]], %[[VAL_89]]{{\[}}%[[CST_4]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_91:.*]] = llvm.insertelement %[[VAL_27]], %[[VAL_90]]{{\[}}%[[CST_5]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_92:.*]] = llvm.insertelement %[[VAL_28]], %[[VAL_91]]{{\[}}%[[CST_6]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_93:.*]] = llvm.insertelement %[[VAL_29]], %[[VAL_92]]{{\[}}%[[CST_7]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_95:.*]] = llvm.insertelement %[[VAL_30]], %[[VAL_21]]{{\[}}%[[CST_0]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_96:.*]] = llvm.insertelement %[[VAL_31]], %[[VAL_95]]{{\[}}%[[CST_1]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_97:.*]] = llvm.insertelement %[[VAL_32]], %[[VAL_96]]{{\[}}%[[CST_2]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_98:.*]] = llvm.insertelement %[[VAL_33]], %[[VAL_97]]{{\[}}%[[CST_3]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_99:.*]] = llvm.insertelement %[[VAL_34]], %[[VAL_98]]{{\[}}%[[CST_4]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_100:.*]] = llvm.insertelement %[[VAL_35]], %[[VAL_99]]{{\[}}%[[CST_5]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_101:.*]] = llvm.insertelement %[[VAL_36]], %[[VAL_100]]{{\[}}%[[CST_6]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_102:.*]] = llvm.insertelement %[[VAL_37]], %[[VAL_101]]{{\[}}%[[CST_7]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_104:.*]] = llvm.insertelement %[[VAL_38]], %[[VAL_21]]{{\[}}%[[CST_0]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_105:.*]] = llvm.insertelement %[[VAL_39]], %[[VAL_104]]{{\[}}%[[CST_1]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_106:.*]] = llvm.insertelement %[[VAL_40]], %[[VAL_105]]{{\[}}%[[CST_2]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_107:.*]] = llvm.insertelement %[[VAL_41]], %[[VAL_106]]{{\[}}%[[CST_3]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_108:.*]] = llvm.insertelement %[[VAL_42]], %[[VAL_107]]{{\[}}%[[CST_4]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_109:.*]] = llvm.insertelement %[[VAL_43]], %[[VAL_108]]{{\[}}%[[CST_5]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_110:.*]] = llvm.insertelement %[[VAL_44]], %[[VAL_109]]{{\[}}%[[CST_6]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_111:.*]] = llvm.insertelement %[[VAL_45]], %[[VAL_110]]{{\[}}%[[CST_7]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_113:.*]] = llvm.insertelement %[[VAL_46]], %[[VAL_21]]{{\[}}%[[CST_0]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_114:.*]] = llvm.insertelement %[[VAL_47]], %[[VAL_113]]{{\[}}%[[CST_1]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_115:.*]] = llvm.insertelement %[[VAL_48]], %[[VAL_114]]{{\[}}%[[CST_2]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_116:.*]] = llvm.insertelement %[[VAL_49]], %[[VAL_115]]{{\[}}%[[CST_3]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_117:.*]] = llvm.insertelement %[[VAL_50]], %[[VAL_116]]{{\[}}%[[CST_4]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_118:.*]] = llvm.insertelement %[[VAL_51]], %[[VAL_117]]{{\[}}%[[CST_5]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_119:.*]] = llvm.insertelement %[[VAL_52]], %[[VAL_118]]{{\[}}%[[CST_6]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_120:.*]] = llvm.insertelement %[[VAL_53]], %[[VAL_119]]{{\[}}%[[CST_7]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_122:.*]] = llvm.insertelement %[[VAL_54]], %[[VAL_21]]{{\[}}%[[CST_0]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_123:.*]] = llvm.insertelement %[[VAL_55]], %[[VAL_122]]{{\[}}%[[CST_1]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_124:.*]] = llvm.insertelement %[[VAL_56]], %[[VAL_123]]{{\[}}%[[CST_2]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_125:.*]] = llvm.insertelement %[[VAL_57]], %[[VAL_124]]{{\[}}%[[CST_3]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_126:.*]] = llvm.insertelement %[[VAL_58]], %[[VAL_125]]{{\[}}%[[CST_4]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_127:.*]] = llvm.insertelement %[[VAL_59]], %[[VAL_126]]{{\[}}%[[CST_5]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_128:.*]] = llvm.insertelement %[[VAL_60]], %[[VAL_127]]{{\[}}%[[CST_6]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_129:.*]] = llvm.insertelement %[[VAL_61]], %[[VAL_128]]{{\[}}%[[CST_7]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_131:.*]] = llvm.insertelement %[[VAL_62]], %[[VAL_21]]{{\[}}%[[CST_0]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_132:.*]] = llvm.insertelement %[[VAL_63]], %[[VAL_131]]{{\[}}%[[CST_1]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_133:.*]] = llvm.insertelement %[[VAL_64]], %[[VAL_132]]{{\[}}%[[CST_2]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_134:.*]] = llvm.insertelement %[[VAL_65]], %[[VAL_133]]{{\[}}%[[CST_3]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_135:.*]] = llvm.insertelement %[[VAL_66]], %[[VAL_134]]{{\[}}%[[CST_4]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_136:.*]] = llvm.insertelement %[[VAL_67]], %[[VAL_135]]{{\[}}%[[CST_5]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_137:.*]] = llvm.insertelement %[[VAL_68]], %[[VAL_136]]{{\[}}%[[CST_6]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_138:.*]] = llvm.insertelement %[[VAL_69]], %[[VAL_137]]{{\[}}%[[CST_7]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_140:.*]] = llvm.insertelement %[[VAL_70]], %[[VAL_21]]{{\[}}%[[CST_0]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_141:.*]] = llvm.insertelement %[[VAL_71]], %[[VAL_140]]{{\[}}%[[CST_1]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_142:.*]] = llvm.insertelement %[[VAL_72]], %[[VAL_141]]{{\[}}%[[CST_2]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_143:.*]] = llvm.insertelement %[[VAL_73]], %[[VAL_142]]{{\[}}%[[CST_3]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_144:.*]] = llvm.insertelement %[[VAL_74]], %[[VAL_143]]{{\[}}%[[CST_4]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_145:.*]] = llvm.insertelement %[[VAL_75]], %[[VAL_144]]{{\[}}%[[CST_5]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_146:.*]] = llvm.insertelement %[[VAL_76]], %[[VAL_145]]{{\[}}%[[CST_6]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_147:.*]] = llvm.insertelement %[[VAL_77]], %[[VAL_146]]{{\[}}%[[CST_7]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_149:.*]] = llvm.insertelement %[[VAL_78]], %[[VAL_21]]{{\[}}%[[CST_0]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_150:.*]] = llvm.insertelement %[[VAL_79]], %[[VAL_149]]{{\[}}%[[CST_1]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_151:.*]] = llvm.insertelement %[[VAL_80]], %[[VAL_150]]{{\[}}%[[CST_2]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_152:.*]] = llvm.insertelement %[[VAL_81]], %[[VAL_151]]{{\[}}%[[CST_3]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_153:.*]] = llvm.insertelement %[[VAL_82]], %[[VAL_152]]{{\[}}%[[CST_4]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_154:.*]] = llvm.insertelement %[[VAL_83]], %[[VAL_153]]{{\[}}%[[CST_5]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_155:.*]] = llvm.insertelement %[[VAL_84]], %[[VAL_154]]{{\[}}%[[CST_6]] : i32] : vector<8xf16>
    // CHECK:           %[[VAL_156:.*]] = llvm.insertelement %[[VAL_85]], %[[VAL_155]]{{\[}}%[[CST_7]] : i32] : vector<8xf16>

    // COM: The shape of Operand B replica is [2, 2]
    // COM: The replica order are [0, 1]
    // COM:                       [2, 3]

    // CHECK:           %[[VAL_158:.*]] = llvm.extractvalue %[[B]][0]
    // CHECK:           %[[VAL_159:.*]] = llvm.extractvalue %[[B]][1]
    // CHECK:           %[[VAL_160:.*]] = llvm.extractvalue %[[B]][2]
    // CHECK:           %[[VAL_161:.*]] = llvm.extractvalue %[[B]][3]
    // CHECK:           %[[VAL_162:.*]] = llvm.extractvalue %[[B]][4]
    // CHECK:           %[[VAL_163:.*]] = llvm.extractvalue %[[B]][5]
    // CHECK:           %[[VAL_164:.*]] = llvm.extractvalue %[[B]][6]
    // CHECK:           %[[VAL_165:.*]] = llvm.extractvalue %[[B]][7]
    // CHECK:           %[[VAL_166:.*]] = llvm.extractvalue %[[B]][8]
    // CHECK:           %[[VAL_167:.*]] = llvm.extractvalue %[[B]][9]
    // CHECK:           %[[VAL_168:.*]] = llvm.extractvalue %[[B]][10]
    // CHECK:           %[[VAL_169:.*]] = llvm.extractvalue %[[B]][11]
    // CHECK:           %[[VAL_170:.*]] = llvm.extractvalue %[[B]][12]
    // CHECK:           %[[VAL_171:.*]] = llvm.extractvalue %[[B]][13]
    // CHECK:           %[[VAL_172:.*]] = llvm.extractvalue %[[B]][14]
    // CHECK:           %[[VAL_173:.*]] = llvm.extractvalue %[[B]][15]
    // CHECK:           %[[VAL_174:.*]] = llvm.extractvalue %[[B]][16]
    // CHECK:           %[[VAL_175:.*]] = llvm.extractvalue %[[B]][17]
    // CHECK:           %[[VAL_176:.*]] = llvm.extractvalue %[[B]][18]
    // CHECK:           %[[VAL_177:.*]] = llvm.extractvalue %[[B]][19]
    // CHECK:           %[[VAL_178:.*]] = llvm.extractvalue %[[B]][20]
    // CHECK:           %[[VAL_179:.*]] = llvm.extractvalue %[[B]][21]
    // CHECK:           %[[VAL_180:.*]] = llvm.extractvalue %[[B]][22]
    // CHECK:           %[[VAL_181:.*]] = llvm.extractvalue %[[B]][23]
    // CHECK:           %[[VAL_182:.*]] = llvm.extractvalue %[[B]][24]
    // CHECK:           %[[VAL_183:.*]] = llvm.extractvalue %[[B]][25]
    // CHECK:           %[[VAL_184:.*]] = llvm.extractvalue %[[B]][26]
    // CHECK:           %[[VAL_185:.*]] = llvm.extractvalue %[[B]][27]
    // CHECK:           %[[VAL_186:.*]] = llvm.extractvalue %[[B]][28]
    // CHECK:           %[[VAL_187:.*]] = llvm.extractvalue %[[B]][29]
    // CHECK:           %[[VAL_188:.*]] = llvm.extractvalue %[[B]][30]
    // CHECK:           %[[VAL_189:.*]] = llvm.extractvalue %[[B]][31]
    // CHECK:           %[[VAL_190:.*]] = llvm.extractvalue %[[B]][32]
    // CHECK:           %[[VAL_191:.*]] = llvm.extractvalue %[[B]][33]
    // CHECK:           %[[VAL_192:.*]] = llvm.extractvalue %[[B]][34]
    // CHECK:           %[[VAL_193:.*]] = llvm.extractvalue %[[B]][35]
    // CHECK:           %[[VAL_194:.*]] = llvm.extractvalue %[[B]][36]
    // CHECK:           %[[VAL_195:.*]] = llvm.extractvalue %[[B]][37]
    // CHECK:           %[[VAL_196:.*]] = llvm.extractvalue %[[B]][38]
    // CHECK:           %[[VAL_197:.*]] = llvm.extractvalue %[[B]][39]
    // CHECK:           %[[VAL_198:.*]] = llvm.extractvalue %[[B]][40]
    // CHECK:           %[[VAL_199:.*]] = llvm.extractvalue %[[B]][41]
    // CHECK:           %[[VAL_200:.*]] = llvm.extractvalue %[[B]][42]
    // CHECK:           %[[VAL_201:.*]] = llvm.extractvalue %[[B]][43]
    // CHECK:           %[[VAL_202:.*]] = llvm.extractvalue %[[B]][44]
    // CHECK:           %[[VAL_203:.*]] = llvm.extractvalue %[[B]][45]
    // CHECK:           %[[VAL_204:.*]] = llvm.extractvalue %[[B]][46]
    // CHECK:           %[[VAL_205:.*]] = llvm.extractvalue %[[B]][47]
    // CHECK:           %[[VAL_206:.*]] = llvm.extractvalue %[[B]][48]
    // CHECK:           %[[VAL_207:.*]] = llvm.extractvalue %[[B]][49]
    // CHECK:           %[[VAL_208:.*]] = llvm.extractvalue %[[B]][50]
    // CHECK:           %[[VAL_209:.*]] = llvm.extractvalue %[[B]][51]
    // CHECK:           %[[VAL_210:.*]] = llvm.extractvalue %[[B]][52]
    // CHECK:           %[[VAL_211:.*]] = llvm.extractvalue %[[B]][53]
    // CHECK:           %[[VAL_212:.*]] = llvm.extractvalue %[[B]][54]
    // CHECK:           %[[VAL_213:.*]] = llvm.extractvalue %[[B]][55]
    // CHECK:           %[[VAL_214:.*]] = llvm.extractvalue %[[B]][56]
    // CHECK:           %[[VAL_215:.*]] = llvm.extractvalue %[[B]][57]
    // CHECK:           %[[VAL_216:.*]] = llvm.extractvalue %[[B]][58]
    // CHECK:           %[[VAL_217:.*]] = llvm.extractvalue %[[B]][59]
    // CHECK:           %[[VAL_218:.*]] = llvm.extractvalue %[[B]][60]
    // CHECK:           %[[VAL_219:.*]] = llvm.extractvalue %[[B]][61]
    // CHECK:           %[[VAL_220:.*]] = llvm.extractvalue %[[B]][62]
    // CHECK:           %[[VAL_221:.*]] = llvm.extractvalue %[[B]][63]
    // CHECK:           %[[VAL_222:.*]] = llvm.insertelement %[[VAL_158]], %[[VAL_12]]{{\[}}%[[CST_0]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_223:.*]] = llvm.insertelement %[[VAL_159]], %[[VAL_222]]{{\[}}%[[CST_1]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_224:.*]] = llvm.insertelement %[[VAL_160]], %[[VAL_223]]{{\[}}%[[CST_2]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_225:.*]] = llvm.insertelement %[[VAL_161]], %[[VAL_224]]{{\[}}%[[CST_3]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_226:.*]] = llvm.insertelement %[[VAL_162]], %[[VAL_225]]{{\[}}%[[CST_4]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_227:.*]] = llvm.insertelement %[[VAL_163]], %[[VAL_226]]{{\[}}%[[CST_5]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_228:.*]] = llvm.insertelement %[[VAL_164]], %[[VAL_227]]{{\[}}%[[CST_6]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_229:.*]] = llvm.insertelement %[[VAL_165]], %[[VAL_228]]{{\[}}%[[CST_7]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_230:.*]] = llvm.insertelement %[[VAL_166]], %[[VAL_229]]{{\[}}%[[CST_8]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_231:.*]] = llvm.insertelement %[[VAL_167]], %[[VAL_230]]{{\[}}%[[CST_9]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_232:.*]] = llvm.insertelement %[[VAL_168]], %[[VAL_231]]{{\[}}%[[CST_10]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_233:.*]] = llvm.insertelement %[[VAL_169]], %[[VAL_232]]{{\[}}%[[CST_11]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_234:.*]] = llvm.insertelement %[[VAL_170]], %[[VAL_233]]{{\[}}%[[CST_12]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_235:.*]] = llvm.insertelement %[[VAL_171]], %[[VAL_234]]{{\[}}%[[CST_13]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_236:.*]] = llvm.insertelement %[[VAL_172]], %[[VAL_235]]{{\[}}%[[CST_14]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_237:.*]] = llvm.insertelement %[[VAL_173]], %[[VAL_236]]{{\[}}%[[CST_15]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_239:.*]] = llvm.insertelement %[[VAL_174]], %[[VAL_12]]{{\[}}%[[CST_0]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_240:.*]] = llvm.insertelement %[[VAL_175]], %[[VAL_239]]{{\[}}%[[CST_1]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_241:.*]] = llvm.insertelement %[[VAL_176]], %[[VAL_240]]{{\[}}%[[CST_2]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_242:.*]] = llvm.insertelement %[[VAL_177]], %[[VAL_241]]{{\[}}%[[CST_3]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_243:.*]] = llvm.insertelement %[[VAL_178]], %[[VAL_242]]{{\[}}%[[CST_4]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_244:.*]] = llvm.insertelement %[[VAL_179]], %[[VAL_243]]{{\[}}%[[CST_5]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_245:.*]] = llvm.insertelement %[[VAL_180]], %[[VAL_244]]{{\[}}%[[CST_6]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_246:.*]] = llvm.insertelement %[[VAL_181]], %[[VAL_245]]{{\[}}%[[CST_7]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_247:.*]] = llvm.insertelement %[[VAL_182]], %[[VAL_246]]{{\[}}%[[CST_8]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_248:.*]] = llvm.insertelement %[[VAL_183]], %[[VAL_247]]{{\[}}%[[CST_9]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_249:.*]] = llvm.insertelement %[[VAL_184]], %[[VAL_248]]{{\[}}%[[CST_10]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_250:.*]] = llvm.insertelement %[[VAL_185]], %[[VAL_249]]{{\[}}%[[CST_11]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_251:.*]] = llvm.insertelement %[[VAL_186]], %[[VAL_250]]{{\[}}%[[CST_12]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_252:.*]] = llvm.insertelement %[[VAL_187]], %[[VAL_251]]{{\[}}%[[CST_13]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_253:.*]] = llvm.insertelement %[[VAL_188]], %[[VAL_252]]{{\[}}%[[CST_14]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_254:.*]] = llvm.insertelement %[[VAL_189]], %[[VAL_253]]{{\[}}%[[CST_15]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_256:.*]] = llvm.insertelement %[[VAL_190]], %[[VAL_12]]{{\[}}%[[CST_0]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_257:.*]] = llvm.insertelement %[[VAL_191]], %[[VAL_256]]{{\[}}%[[CST_1]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_258:.*]] = llvm.insertelement %[[VAL_192]], %[[VAL_257]]{{\[}}%[[CST_2]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_259:.*]] = llvm.insertelement %[[VAL_193]], %[[VAL_258]]{{\[}}%[[CST_3]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_260:.*]] = llvm.insertelement %[[VAL_194]], %[[VAL_259]]{{\[}}%[[CST_4]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_261:.*]] = llvm.insertelement %[[VAL_195]], %[[VAL_260]]{{\[}}%[[CST_5]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_262:.*]] = llvm.insertelement %[[VAL_196]], %[[VAL_261]]{{\[}}%[[CST_6]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_263:.*]] = llvm.insertelement %[[VAL_197]], %[[VAL_262]]{{\[}}%[[CST_7]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_264:.*]] = llvm.insertelement %[[VAL_198]], %[[VAL_263]]{{\[}}%[[CST_8]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_265:.*]] = llvm.insertelement %[[VAL_199]], %[[VAL_264]]{{\[}}%[[CST_9]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_266:.*]] = llvm.insertelement %[[VAL_200]], %[[VAL_265]]{{\[}}%[[CST_10]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_267:.*]] = llvm.insertelement %[[VAL_201]], %[[VAL_266]]{{\[}}%[[CST_11]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_268:.*]] = llvm.insertelement %[[VAL_202]], %[[VAL_267]]{{\[}}%[[CST_12]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_269:.*]] = llvm.insertelement %[[VAL_203]], %[[VAL_268]]{{\[}}%[[CST_13]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_270:.*]] = llvm.insertelement %[[VAL_204]], %[[VAL_269]]{{\[}}%[[CST_14]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_271:.*]] = llvm.insertelement %[[VAL_205]], %[[VAL_270]]{{\[}}%[[CST_15]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_273:.*]] = llvm.insertelement %[[VAL_206]], %[[VAL_12]]{{\[}}%[[CST_0]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_274:.*]] = llvm.insertelement %[[VAL_207]], %[[VAL_273]]{{\[}}%[[CST_1]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_275:.*]] = llvm.insertelement %[[VAL_208]], %[[VAL_274]]{{\[}}%[[CST_2]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_276:.*]] = llvm.insertelement %[[VAL_209]], %[[VAL_275]]{{\[}}%[[CST_3]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_277:.*]] = llvm.insertelement %[[VAL_210]], %[[VAL_276]]{{\[}}%[[CST_4]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_278:.*]] = llvm.insertelement %[[VAL_211]], %[[VAL_277]]{{\[}}%[[CST_5]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_279:.*]] = llvm.insertelement %[[VAL_212]], %[[VAL_278]]{{\[}}%[[CST_6]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_280:.*]] = llvm.insertelement %[[VAL_213]], %[[VAL_279]]{{\[}}%[[CST_7]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_281:.*]] = llvm.insertelement %[[VAL_214]], %[[VAL_280]]{{\[}}%[[CST_8]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_282:.*]] = llvm.insertelement %[[VAL_215]], %[[VAL_281]]{{\[}}%[[CST_9]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_283:.*]] = llvm.insertelement %[[VAL_216]], %[[VAL_282]]{{\[}}%[[CST_10]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_284:.*]] = llvm.insertelement %[[VAL_217]], %[[VAL_283]]{{\[}}%[[CST_11]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_285:.*]] = llvm.insertelement %[[VAL_218]], %[[VAL_284]]{{\[}}%[[CST_12]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_286:.*]] = llvm.insertelement %[[VAL_219]], %[[VAL_285]]{{\[}}%[[CST_13]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_287:.*]] = llvm.insertelement %[[VAL_220]], %[[VAL_286]]{{\[}}%[[CST_14]] : i32] : vector<16xf16>
    // CHECK:           %[[VAL_288:.*]] = llvm.insertelement %[[VAL_221]], %[[VAL_287]]{{\[}}%[[CST_15]] : i32] : vector<16xf16>

    // COM: The shape of Operand C replica is [4, 2]
    // COM: The replica order are [0, 1]
    // COM:                       [2, 3]
    // COM:                       [4, 5]
    // COM:                       [6, 7]

    // CHECK:           %[[VAL_290:.*]] = llvm.extractvalue %[[C]][0]
    // CHECK:           %[[VAL_291:.*]] = llvm.extractvalue %[[C]][1]
    // CHECK:           %[[VAL_292:.*]] = llvm.extractvalue %[[C]][2]
    // CHECK:           %[[VAL_293:.*]] = llvm.extractvalue %[[C]][3]
    // CHECK:           %[[VAL_294:.*]] = llvm.extractvalue %[[C]][4]
    // CHECK:           %[[VAL_295:.*]] = llvm.extractvalue %[[C]][5]
    // CHECK:           %[[VAL_296:.*]] = llvm.extractvalue %[[C]][6]
    // CHECK:           %[[VAL_297:.*]] = llvm.extractvalue %[[C]][7]
    // CHECK:           %[[VAL_298:.*]] = llvm.extractvalue %[[C]][8]
    // CHECK:           %[[VAL_299:.*]] = llvm.extractvalue %[[C]][9]
    // CHECK:           %[[VAL_300:.*]] = llvm.extractvalue %[[C]][10]
    // CHECK:           %[[VAL_301:.*]] = llvm.extractvalue %[[C]][11]
    // CHECK:           %[[VAL_302:.*]] = llvm.extractvalue %[[C]][12]
    // CHECK:           %[[VAL_303:.*]] = llvm.extractvalue %[[C]][13]
    // CHECK:           %[[VAL_304:.*]] = llvm.extractvalue %[[C]][14]
    // CHECK:           %[[VAL_305:.*]] = llvm.extractvalue %[[C]][15]
    // CHECK:           %[[VAL_306:.*]] = llvm.extractvalue %[[C]][16]
    // CHECK:           %[[VAL_307:.*]] = llvm.extractvalue %[[C]][17]
    // CHECK:           %[[VAL_308:.*]] = llvm.extractvalue %[[C]][18]
    // CHECK:           %[[VAL_309:.*]] = llvm.extractvalue %[[C]][19]
    // CHECK:           %[[VAL_310:.*]] = llvm.extractvalue %[[C]][20]
    // CHECK:           %[[VAL_311:.*]] = llvm.extractvalue %[[C]][21]
    // CHECK:           %[[VAL_312:.*]] = llvm.extractvalue %[[C]][22]
    // CHECK:           %[[VAL_313:.*]] = llvm.extractvalue %[[C]][23]
    // CHECK:           %[[VAL_314:.*]] = llvm.extractvalue %[[C]][24]
    // CHECK:           %[[VAL_315:.*]] = llvm.extractvalue %[[C]][25]
    // CHECK:           %[[VAL_316:.*]] = llvm.extractvalue %[[C]][26]
    // CHECK:           %[[VAL_317:.*]] = llvm.extractvalue %[[C]][27]
    // CHECK:           %[[VAL_318:.*]] = llvm.extractvalue %[[C]][28]
    // CHECK:           %[[VAL_319:.*]] = llvm.extractvalue %[[C]][29]
    // CHECK:           %[[VAL_320:.*]] = llvm.extractvalue %[[C]][30]
    // CHECK:           %[[VAL_321:.*]] = llvm.extractvalue %[[C]][31]
    // CHECK:           %[[VAL_322:.*]] = llvm.extractvalue %[[C]][32]
    // CHECK:           %[[VAL_323:.*]] = llvm.extractvalue %[[C]][33]
    // CHECK:           %[[VAL_324:.*]] = llvm.extractvalue %[[C]][34]
    // CHECK:           %[[VAL_325:.*]] = llvm.extractvalue %[[C]][35]
    // CHECK:           %[[VAL_326:.*]] = llvm.extractvalue %[[C]][36]
    // CHECK:           %[[VAL_327:.*]] = llvm.extractvalue %[[C]][37]
    // CHECK:           %[[VAL_328:.*]] = llvm.extractvalue %[[C]][38]
    // CHECK:           %[[VAL_329:.*]] = llvm.extractvalue %[[C]][39]
    // CHECK:           %[[VAL_330:.*]] = llvm.extractvalue %[[C]][40]
    // CHECK:           %[[VAL_331:.*]] = llvm.extractvalue %[[C]][41]
    // CHECK:           %[[VAL_332:.*]] = llvm.extractvalue %[[C]][42]
    // CHECK:           %[[VAL_333:.*]] = llvm.extractvalue %[[C]][43]
    // CHECK:           %[[VAL_334:.*]] = llvm.extractvalue %[[C]][44]
    // CHECK:           %[[VAL_335:.*]] = llvm.extractvalue %[[C]][45]
    // CHECK:           %[[VAL_336:.*]] = llvm.extractvalue %[[C]][46]
    // CHECK:           %[[VAL_337:.*]] = llvm.extractvalue %[[C]][47]
    // CHECK:           %[[VAL_338:.*]] = llvm.extractvalue %[[C]][48]
    // CHECK:           %[[VAL_339:.*]] = llvm.extractvalue %[[C]][49]
    // CHECK:           %[[VAL_340:.*]] = llvm.extractvalue %[[C]][50]
    // CHECK:           %[[VAL_341:.*]] = llvm.extractvalue %[[C]][51]
    // CHECK:           %[[VAL_342:.*]] = llvm.extractvalue %[[C]][52]
    // CHECK:           %[[VAL_343:.*]] = llvm.extractvalue %[[C]][53]
    // CHECK:           %[[VAL_344:.*]] = llvm.extractvalue %[[C]][54]
    // CHECK:           %[[VAL_345:.*]] = llvm.extractvalue %[[C]][55]
    // CHECK:           %[[VAL_346:.*]] = llvm.extractvalue %[[C]][56]
    // CHECK:           %[[VAL_347:.*]] = llvm.extractvalue %[[C]][57]
    // CHECK:           %[[VAL_348:.*]] = llvm.extractvalue %[[C]][58]
    // CHECK:           %[[VAL_349:.*]] = llvm.extractvalue %[[C]][59]
    // CHECK:           %[[VAL_350:.*]] = llvm.extractvalue %[[C]][60]
    // CHECK:           %[[VAL_351:.*]] = llvm.extractvalue %[[C]][61]
    // CHECK:           %[[VAL_352:.*]] = llvm.extractvalue %[[C]][62]
    // CHECK:           %[[VAL_353:.*]] = llvm.extractvalue %[[C]][63]
    // CHECK:           %[[VAL_354:.*]] = llvm.insertelement %[[VAL_290]], %[[VAL_3]]{{\[}}%[[CST_0]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_355:.*]] = llvm.insertelement %[[VAL_291]], %[[VAL_354]]{{\[}}%[[CST_1]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_356:.*]] = llvm.insertelement %[[VAL_292]], %[[VAL_355]]{{\[}}%[[CST_2]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_357:.*]] = llvm.insertelement %[[VAL_293]], %[[VAL_356]]{{\[}}%[[CST_3]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_358:.*]] = llvm.insertelement %[[VAL_294]], %[[VAL_357]]{{\[}}%[[CST_4]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_359:.*]] = llvm.insertelement %[[VAL_295]], %[[VAL_358]]{{\[}}%[[CST_5]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_360:.*]] = llvm.insertelement %[[VAL_296]], %[[VAL_359]]{{\[}}%[[CST_6]] : i32] : vector<8xf32>
    // CHECK:           %[[C_0_0:.*]] = llvm.insertelement %[[VAL_297]], %[[VAL_360]]{{\[}}%[[CST_7]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_362:.*]] = llvm.insertelement %[[VAL_298]], %[[VAL_3]]{{\[}}%[[CST_0]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_363:.*]] = llvm.insertelement %[[VAL_299]], %[[VAL_362]]{{\[}}%[[CST_1]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_364:.*]] = llvm.insertelement %[[VAL_300]], %[[VAL_363]]{{\[}}%[[CST_2]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_365:.*]] = llvm.insertelement %[[VAL_301]], %[[VAL_364]]{{\[}}%[[CST_3]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_366:.*]] = llvm.insertelement %[[VAL_302]], %[[VAL_365]]{{\[}}%[[CST_4]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_367:.*]] = llvm.insertelement %[[VAL_303]], %[[VAL_366]]{{\[}}%[[CST_5]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_368:.*]] = llvm.insertelement %[[VAL_304]], %[[VAL_367]]{{\[}}%[[CST_6]] : i32] : vector<8xf32>
    // CHECK:           %[[C_0_1:.*]] = llvm.insertelement %[[VAL_305]], %[[VAL_368]]{{\[}}%[[CST_7]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_370:.*]] = llvm.insertelement %[[VAL_306]], %[[VAL_3]]{{\[}}%[[CST_0]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_371:.*]] = llvm.insertelement %[[VAL_307]], %[[VAL_370]]{{\[}}%[[CST_1]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_372:.*]] = llvm.insertelement %[[VAL_308]], %[[VAL_371]]{{\[}}%[[CST_2]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_373:.*]] = llvm.insertelement %[[VAL_309]], %[[VAL_372]]{{\[}}%[[CST_3]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_374:.*]] = llvm.insertelement %[[VAL_310]], %[[VAL_373]]{{\[}}%[[CST_4]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_375:.*]] = llvm.insertelement %[[VAL_311]], %[[VAL_374]]{{\[}}%[[CST_5]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_376:.*]] = llvm.insertelement %[[VAL_312]], %[[VAL_375]]{{\[}}%[[CST_6]] : i32] : vector<8xf32>
    // CHECK:           %[[C_1_0:.*]] = llvm.insertelement %[[VAL_313]], %[[VAL_376]]{{\[}}%[[CST_7]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_378:.*]] = llvm.insertelement %[[VAL_314]], %[[VAL_3]]{{\[}}%[[CST_0]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_379:.*]] = llvm.insertelement %[[VAL_315]], %[[VAL_378]]{{\[}}%[[CST_1]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_380:.*]] = llvm.insertelement %[[VAL_316]], %[[VAL_379]]{{\[}}%[[CST_2]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_381:.*]] = llvm.insertelement %[[VAL_317]], %[[VAL_380]]{{\[}}%[[CST_3]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_382:.*]] = llvm.insertelement %[[VAL_318]], %[[VAL_381]]{{\[}}%[[CST_4]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_383:.*]] = llvm.insertelement %[[VAL_319]], %[[VAL_382]]{{\[}}%[[CST_5]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_384:.*]] = llvm.insertelement %[[VAL_320]], %[[VAL_383]]{{\[}}%[[CST_6]] : i32] : vector<8xf32>
    // CHECK:           %[[C_1_1:.*]] = llvm.insertelement %[[VAL_321]], %[[VAL_384]]{{\[}}%[[CST_7]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_386:.*]] = llvm.insertelement %[[VAL_322]], %[[VAL_3]]{{\[}}%[[CST_0]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_387:.*]] = llvm.insertelement %[[VAL_323]], %[[VAL_386]]{{\[}}%[[CST_1]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_388:.*]] = llvm.insertelement %[[VAL_324]], %[[VAL_387]]{{\[}}%[[CST_2]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_389:.*]] = llvm.insertelement %[[VAL_325]], %[[VAL_388]]{{\[}}%[[CST_3]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_390:.*]] = llvm.insertelement %[[VAL_326]], %[[VAL_389]]{{\[}}%[[CST_4]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_391:.*]] = llvm.insertelement %[[VAL_327]], %[[VAL_390]]{{\[}}%[[CST_5]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_392:.*]] = llvm.insertelement %[[VAL_328]], %[[VAL_391]]{{\[}}%[[CST_6]] : i32] : vector<8xf32>
    // CHECK:           %[[C_2_0:.*]] = llvm.insertelement %[[VAL_329]], %[[VAL_392]]{{\[}}%[[CST_7]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_394:.*]] = llvm.insertelement %[[VAL_330]], %[[VAL_3]]{{\[}}%[[CST_0]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_395:.*]] = llvm.insertelement %[[VAL_331]], %[[VAL_394]]{{\[}}%[[CST_1]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_396:.*]] = llvm.insertelement %[[VAL_332]], %[[VAL_395]]{{\[}}%[[CST_2]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_397:.*]] = llvm.insertelement %[[VAL_333]], %[[VAL_396]]{{\[}}%[[CST_3]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_398:.*]] = llvm.insertelement %[[VAL_334]], %[[VAL_397]]{{\[}}%[[CST_4]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_399:.*]] = llvm.insertelement %[[VAL_335]], %[[VAL_398]]{{\[}}%[[CST_5]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_400:.*]] = llvm.insertelement %[[VAL_336]], %[[VAL_399]]{{\[}}%[[CST_6]] : i32] : vector<8xf32>
    // CHECK:           %[[C_2_1:.*]] = llvm.insertelement %[[VAL_337]], %[[VAL_400]]{{\[}}%[[CST_7]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_402:.*]] = llvm.insertelement %[[VAL_338]], %[[VAL_3]]{{\[}}%[[CST_0]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_403:.*]] = llvm.insertelement %[[VAL_339]], %[[VAL_402]]{{\[}}%[[CST_1]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_404:.*]] = llvm.insertelement %[[VAL_340]], %[[VAL_403]]{{\[}}%[[CST_2]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_405:.*]] = llvm.insertelement %[[VAL_341]], %[[VAL_404]]{{\[}}%[[CST_3]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_406:.*]] = llvm.insertelement %[[VAL_342]], %[[VAL_405]]{{\[}}%[[CST_4]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_407:.*]] = llvm.insertelement %[[VAL_343]], %[[VAL_406]]{{\[}}%[[CST_5]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_408:.*]] = llvm.insertelement %[[VAL_344]], %[[VAL_407]]{{\[}}%[[CST_6]] : i32] : vector<8xf32>
    // CHECK:           %[[C_3_0:.*]] = llvm.insertelement %[[VAL_345]], %[[VAL_408]]{{\[}}%[[CST_7]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_410:.*]] = llvm.insertelement %[[VAL_346]], %[[VAL_3]]{{\[}}%[[CST_0]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_411:.*]] = llvm.insertelement %[[VAL_347]], %[[VAL_410]]{{\[}}%[[CST_1]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_412:.*]] = llvm.insertelement %[[VAL_348]], %[[VAL_411]]{{\[}}%[[CST_2]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_413:.*]] = llvm.insertelement %[[VAL_349]], %[[VAL_412]]{{\[}}%[[CST_3]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_414:.*]] = llvm.insertelement %[[VAL_350]], %[[VAL_413]]{{\[}}%[[CST_4]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_415:.*]] = llvm.insertelement %[[VAL_351]], %[[VAL_414]]{{\[}}%[[CST_5]] : i32] : vector<8xf32>
    // CHECK:           %[[VAL_416:.*]] = llvm.insertelement %[[VAL_352]], %[[VAL_415]]{{\[}}%[[CST_6]] : i32] : vector<8xf32>
    // CHECK:           %[[C_3_1:.*]] = llvm.insertelement %[[VAL_353]], %[[VAL_416]]{{\[}}%[[CST_7]] : i32] : vector<8xf32>

    // COM: Total 16 dpas ops unrolled.
    // CHECK:           %[[B_0_0:.*]] = llvm.bitcast %[[VAL_237]] : vector<16xf16> to vector<8xi32>
    // CHECK:           %[[A_0_0:.*]] = llvm.bitcast %[[VAL_93]] : vector<8xf16> to vector<8xi16>
    // CHECK:           %[[C_0_0_0:.*]] = llvm.call spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fi(%{{.*}}, %[[A_0_0]], %[[B_0_0]], %[[C_0_0]], %{{.*}})
    // CHECK:           %[[A_1_0:.*]] = llvm.bitcast %[[VAL_102]] : vector<8xf16> to vector<8xi16>
    // CHECK:           %[[C_1_0_0:.*]] = llvm.call spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fi(%{{.*}}, %[[A_1_0]], %[[B_0_0]], %[[C_1_0]], %{{.*}})
    // CHECK:           %[[A_2_0:.*]] = llvm.bitcast %[[VAL_111]] : vector<8xf16> to vector<8xi16>
    // CHECK:           %[[C_2_0_0:.*]] = llvm.call spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fi(%{{.*}}, %[[A_2_0]], %[[B_0_0]], %[[C_2_0]], %{{.*}})
    // CHECK:           %[[A_3_0:.*]] = llvm.bitcast %[[VAL_120]] : vector<8xf16> to vector<8xi16>
    // CHECK:           %[[C_3_0_0:.*]] = llvm.call spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fi(%{{.*}}, %[[A_3_0]], %[[B_0_0]], %[[C_3_0]], %{{.*}})
    // CHECK:           %[[B_0_1:.*]] = llvm.bitcast %[[VAL_254]] : vector<16xf16> to vector<8xi32>

    // NO-AGGRESSIVE-REUSE:           %[[C_0_1_0:.*]] = llvm.call spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fi(%{{.*}}, %[[A_0_0]], %[[B_0_1]], %[[C_0_1]], %{{.*}})
    // NO-AGGRESSIVE-REUSE:           %[[C_1_1_0:.*]] = llvm.call spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fi(%{{.*}}, %[[A_1_0]], %[[B_0_1]], %[[C_1_1]], %{{.*}})
    // NO-AGGRESSIVE-REUSE:           %[[C_2_1_0:.*]] = llvm.call spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fi(%{{.*}}, %[[A_2_0]], %[[B_0_1]], %[[C_2_1]], %{{.*}})
    // NO-AGGRESSIVE-REUSE:           %[[C_3_1_0:.*]] = llvm.call spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fi(%{{.*}}, %[[A_3_0]], %[[B_0_1]], %[[C_3_1]], %{{.*}})
    // AGGRESSIVE-REUSE:              %[[C_3_1_0:.*]] = llvm.call spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fi(%{{.*}}, %[[A_3_0]], %[[B_0_1]], %[[C_3_1]], %{{.*}})
    // AGGRESSIVE-REUSE:              %[[C_2_1_0:.*]] = llvm.call spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fi(%{{.*}}, %[[A_2_0]], %[[B_0_1]], %[[C_2_1]], %{{.*}})
    // AGGRESSIVE-REUSE:              %[[C_1_1_0:.*]] = llvm.call spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fi(%{{.*}}, %[[A_1_0]], %[[B_0_1]], %[[C_1_1]], %{{.*}})
    // AGGRESSIVE-REUSE:              %[[C_0_1_0:.*]] = llvm.call spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fi(%{{.*}}, %[[A_0_0]], %[[B_0_1]], %[[C_0_1]], %{{.*}})

    // CHECK:           %[[B_1_0:.*]] = llvm.bitcast %[[VAL_271]] : vector<16xf16> to vector<8xi32>
    // CHECK:           %[[A_0_1:.*]] = llvm.bitcast %[[VAL_129]] : vector<8xf16> to vector<8xi16>
    // CHECK:           {{.*}} = llvm.call spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fi(%{{.*}}, %[[A_0_1]], %[[B_1_0]], %[[C_0_0_0]], %{{.*}})
    // CHECK:           %[[A_1_1:.*]] = llvm.bitcast %[[VAL_138]] : vector<8xf16> to vector<8xi16>
    // CHECK:           {{.*}} = llvm.call spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fi(%{{.*}}, %[[A_1_1]], %[[B_1_0]], %[[C_1_0_0]], %{{.*}})
    // CHECK:           %[[A_2_1:.*]] = llvm.bitcast %[[VAL_147]] : vector<8xf16> to vector<8xi16>
    // CHECK:           {{.*}} = llvm.call spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fi(%{{.*}}, %[[A_2_1]], %[[B_1_0]], %[[C_2_0_0]], %{{.*}})
    // CHECK:           %[[A_3_1:.*]] = llvm.bitcast %[[VAL_156]] : vector<8xf16> to vector<8xi16>
    // CHECK:           {{.*}} = llvm.call spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fi(%{{.*}}, %[[A_3_1]], %[[B_1_0]], %[[C_3_0_0]], %{{.*}})
    // CHECK:           %[[B_1_1:.*]] = llvm.bitcast %[[VAL_288]] : vector<16xf16> to vector<8xi32>

    // NO-AGGRESSIVE-REUSE:           {{.*}} = llvm.call spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fi(%{{.*}}, %[[A_0_1]], %[[B_1_1]], %[[C_0_1_0]], %{{.*}})
    // NO-AGGRESSIVE-REUSE:           {{.*}} = llvm.call spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fi(%{{.*}}, %[[A_1_1]], %[[B_1_1]], %[[C_1_1_0]], %{{.*}})
    // NO-AGGRESSIVE-REUSE:           {{.*}} = llvm.call spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fi(%{{.*}}, %[[A_2_1]], %[[B_1_1]], %[[C_2_1_0]], %{{.*}})
    // NO-AGGRESSIVE-REUSE:           {{.*}} = llvm.call spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fi(%{{.*}}, %[[A_3_1]], %[[B_1_1]], %[[C_3_1_0]], %{{.*}})
    // AGGRESSIVE-REUSE:              {{.*}} = llvm.call spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fi(%{{.*}}, %[[A_3_1]], %[[B_1_1]], %[[C_3_1_0]], %{{.*}})
    // AGGRESSIVE-REUSE:              {{.*}} = llvm.call spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fi(%{{.*}}, %[[A_2_1]], %[[B_1_1]], %[[C_2_1_0]], %{{.*}})
    // AGGRESSIVE-REUSE:              {{.*}} = llvm.call spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fi(%{{.*}}, %[[A_1_1]], %[[B_1_1]], %[[C_1_1_0]], %{{.*}})
    // AGGRESSIVE-REUSE:              {{.*}} = llvm.call spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fi(%{{.*}}, %[[A_0_1]], %[[B_1_1]], %[[C_0_1_0]], %{{.*}})

    %0 = tt.dot %a, %b, %c, inputPrecision = tf32 : tensor<32x32xf16, #dot_operand_a> * tensor<32x32xf16, #dot_operand_b> -> tensor<32x32xf32, #dpas>
    tt.return
  }
}
