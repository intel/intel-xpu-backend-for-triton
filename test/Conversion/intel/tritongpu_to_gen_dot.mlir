// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory  --convert-triton-intel-gpu-to-llvm | FileCheck %s --implicit-check-not=llvm.inline_asm

#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#dpas, kWidth=2}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#dpas, kWidth=2}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: dot_f32_f16_f16_f32_1
  tt.func @dot_f32_f16_f16_f32_1(%a: tensor<8x16xf16, #dot_operand_a>, %b: tensor<16x16xf16, #dot_operand_b>, %c: tensor<8x16xf32, #dpas>) {
    // CHECK: llvm.call @_Z38intel_sub_group_f16_f16_matrix_mad_k16Dv8_sDv8_iDv8_f(%{{.*}}, %{{.*}}, %{{.*}}) {passthrough = ["convergent"]} : (vector<8xi16>, vector<8xi32>, vector<8xf32>) -> vector<8xf32>
    %0 = tt.dot %a, %b, %c, inputPrecision = tf32 : tensor<8x16xf16, #dot_operand_a> * tensor<16x16xf16, #dot_operand_b> -> tensor<8x16xf32, #dpas>
    tt.return
  }
}

// -----

#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#dpas, kWidth=2}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#dpas, kWidth=2}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: dot_f32_f16_f16_f32_2
  tt.func @dot_f32_f16_f16_f32_2(%a: tensor<16x16xf16, #dot_operand_a>, %b: tensor<16x16xf16, #dot_operand_b>, %c: tensor<16x16xf32, #dpas>) {
    // COM: 2 repetitions along axis for M.
    // CHECK-COUNT-2: llvm.call @_Z38intel_sub_group_f16_f16_matrix_mad_k16Dv8_sDv8_iDv8_f(%{{.*}}, %{{.*}}, %{{.*}}) {passthrough = ["convergent"]} : (vector<8xi16>, vector<8xi32>, vector<8xf32>) -> vector<8xf32>
    %0 = tt.dot %a, %b, %c, inputPrecision = tf32 : tensor<16x16xf16, #dot_operand_a> * tensor<16x16xf16, #dot_operand_b> -> tensor<16x16xf32, #dpas>
    tt.return
  }
}

// -----

#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 4, threadsPerWarp = 16, warpsPerCTA = [1, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#dpas, kWidth=4}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#dpas, kWidth=4}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: dot_i32_i8_i8_i32_1
  tt.func @dot_i32_i8_i8_i32_1(%a: tensor<8x32xi8, #dot_operand_a>, %b: tensor<32x16xi8, #dot_operand_b>, %c: tensor<8x16xi32, #dpas>) {
    // CHECK: llvm.call @_Z36intel_sub_group_i8_i8_matrix_mad_k32Dv8_sDv8_iS0_(%{{.*}}, %{{.*}}, %{{.*}}) {passthrough = ["convergent"]} : (vector<8xi16>, vector<8xi32>, vector<8xi32>) -> vector<8xi32>
    %0 = tt.dot %a, %b, %c, inputPrecision = tf32 : tensor<8x32xi8, #dot_operand_a> * tensor<32x16xi8, #dot_operand_b> -> tensor<8x16xi32, #dpas>
    tt.return
  }
}

// -----

#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 4, threadsPerWarp = 16, warpsPerCTA = [1, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#dpas, kWidth=4}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#dpas, kWidth=4}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: dot_i32_i8_i8_i32_2
  tt.func @dot_i32_i8_i8_i32_2(%a: tensor<8x64xi8, #dot_operand_a>, %b: tensor<64x16xi8, #dot_operand_b>, %c: tensor<8x16xi32, #dpas>) {
    // COM: 2 repetition along axis for K.
    // CHECK: llvm.call @_Z36intel_sub_group_i8_i8_matrix_mad_k32Dv8_sDv8_iS0_(%{{.*}}, %{{.*}}, %{{.*}}) {passthrough = ["convergent"]} : (vector<8xi16>, vector<8xi32>, vector<8xi32>) -> vector<8xi32>
    %0 = tt.dot %a, %b, %c, inputPrecision = tf32 : tensor<8x64xi8, #dot_operand_a> * tensor<64x16xi8, #dot_operand_b> -> tensor<8x16xi32, #dpas>
    tt.return
  }
}

// -----

#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [1, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#dpas, kWidth=1}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#dpas, kWidth=1}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: dot_f32_tf32_tf32_f32_1
  tt.func @dot_f32_tf32_tf32_f32_1(%a: tensor<8x8xf32, #dot_operand_a>, %b: tensor<8x16xf32, #dot_operand_b>, %c: tensor<8x16xf32, #dpas>) {
    // CHECK: llvm.call @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v4i32.v8i32
    %0 = tt.dot %a, %b, %c, inputPrecision = tf32 : tensor<8x8xf32, #dot_operand_a> * tensor<8x16xf32, #dot_operand_b> -> tensor<8x16xf32, #dpas>
    tt.return
  }
}

// -----

#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [1, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#dpas, kWidth=1}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#dpas, kWidth=1}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: dot_f32_tf32_tf32_f32_2
  tt.func @dot_f32_tf32_tf32_f32_2(%a: tensor<8x8xf32, #dot_operand_a>, %b: tensor<8x32xf32, #dot_operand_b>, %c: tensor<8x32xf32, #dpas>) {
    // COM: 2 repetitions along axis for N.
    // CHECK-COUNT-2: llvm.call @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v4i32.v8i32
    %0 = tt.dot %a, %b, %c, inputPrecision = tf32 : tensor<8x8xf32, #dot_operand_a> * tensor<8x32xf32, #dot_operand_b> -> tensor<8x32xf32, #dpas>
    tt.return
  }
}
