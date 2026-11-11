// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm --cse -canonicalize | FileCheck %s --check-prefix=LAZY
// RUN: env TRITON_INTEL_DISABLE_LAZY_FP8_DPAS_OPERANDS=1 triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm --cse -canonicalize | FileCheck %s --check-prefix=EAGER

// An fp8 dot operand extended to fp16 for DPAS is converted one DPAS operand
// at a time, one DPAS call ahead of its first use, instead of in full ahead of
// the dot, when the dot has at least 4 DPAS calls per K slice and its operands
// and accumulator take at least 512 bytes per lane. The fp8e4m3 conversion is
// one fmul per element.

// Converted lazily: the dot is large enough to spill with all of B converted up front.

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#dpas, kWidth=1}>
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#dpas, kWidth=2}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // A 32x64 fp16, B 64x32 fp8, C 32x32 fp32: 768 bytes per lane. Each K slice of
  // B is converted just ahead of its DPAS calls, so conversions follow DPAS.
  // LAZY-LABEL: llvm.func spir_kernelcc @dot_fp8_b
  // LAZY:       llvm.fmul
  // LAZY:       triton_gen.dpas
  // LAZY:       llvm.fmul
  // LAZY:       triton_gen.dpas
  // LAZY:       llvm.return
  // EAGER-LABEL: llvm.func spir_kernelcc @dot_fp8_b
  // EAGER:       triton_gen.dpas
  // EAGER-NOT:   llvm.fmul
  // EAGER:       llvm.return
  tt.func public @dot_fp8_b(%a: tensor<32x64xf16, #dot_operand_a>, %b: tensor<64x32xf8E4M3FN, #dot_operand_b>, %c: tensor<32x32xf32, #dpas>, %out: !tt.ptr<f32>) {
    %b16 = tt.fp_to_fp %b : tensor<64x32xf8E4M3FN, #dot_operand_b> -> tensor<64x32xf16, #dot_operand_b>
    %d = tt.dot %a, %b16, %c, inputPrecision = tf32 : tensor<32x64xf16, #dot_operand_a> * tensor<64x32xf16, #dot_operand_b> -> tensor<32x32xf32, #dpas>
    %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #dpas>
    tt.store %ptrs, %d : tensor<32x32x!tt.ptr<f32>, #dpas>
    tt.return
  }
}
// -----

// A dot with 1 DPAS call per K slice (A 8x64, B 64x16, C 8x16) keeps the conversion up front.

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#dpas, kWidth=1}>
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#dpas, kWidth=2}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // LAZY-LABEL: llvm.func spir_kernelcc @dot_fp8_b_small
  // LAZY:       llvm.fmul
  // LAZY:       triton_gen.dpas
  // LAZY-NOT:   llvm.fmul
  // LAZY:       llvm.return
  tt.func public @dot_fp8_b_small(%a: tensor<8x64xf16, #dot_operand_a>, %b: tensor<64x16xf8E4M3FN, #dot_operand_b>, %c: tensor<8x16xf32, #dpas>, %out: !tt.ptr<f32>) {
    %b16 = tt.fp_to_fp %b : tensor<64x16xf8E4M3FN, #dot_operand_b> -> tensor<64x16xf16, #dot_operand_b>
    %d = tt.dot %a, %b16, %c, inputPrecision = tf32 : tensor<8x64xf16, #dot_operand_a> * tensor<64x16xf16, #dot_operand_b> -> tensor<8x16xf32, #dpas>
    %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<8x16x!tt.ptr<f32>, #dpas>
    tt.store %ptrs, %d : tensor<8x16x!tt.ptr<f32>, #dpas>
    tt.return
  }
}
// -----

// A conversion with another user is left to the `tt.fp_to_fp` lowering.

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#dpas, kWidth=1}>
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#dpas, kWidth=2}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // LAZY-LABEL: llvm.func spir_kernelcc @dot_fp8_b_two_uses
  // LAZY:       llvm.fmul
  // LAZY:       triton_gen.dpas
  // LAZY-NOT:   llvm.fmul
  // LAZY:       llvm.return
  tt.func public @dot_fp8_b_two_uses(%a: tensor<32x64xf16, #dot_operand_a>, %b: tensor<64x32xf8E4M3FN, #dot_operand_b>, %c: tensor<32x32xf32, #dpas>, %out: !tt.ptr<f32>) {
    %b16 = tt.fp_to_fp %b : tensor<64x32xf8E4M3FN, #dot_operand_b> -> tensor<64x32xf16, #dot_operand_b>
    %d = tt.dot %a, %b16, %c, inputPrecision = tf32 : tensor<32x64xf16, #dot_operand_a> * tensor<64x32xf16, #dot_operand_b> -> tensor<32x32xf32, #dpas>
    %e = tt.dot %a, %b16, %d, inputPrecision = tf32 : tensor<32x64xf16, #dot_operand_a> * tensor<64x32xf16, #dot_operand_b> -> tensor<32x32xf32, #dpas>
    %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #dpas>
    tt.store %ptrs, %e : tensor<32x32x!tt.ptr<f32>, #dpas>
    tt.return
  }
}
// -----

// A conversion in another block than its dot (e.g. hoisted out of the dot's loop, like the Q operand of attention) stays where it is.

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#dpas, kWidth=1}>
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#dpas, kWidth=2}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // LAZY-LABEL: llvm.func spir_kernelcc @dot_fp8_b_other_block
  // LAZY:       llvm.fmul
  // LAZY:       llvm.br
  // LAZY-NOT:   llvm.fmul
  // LAZY:       triton_gen.dpas
  // LAZY-NOT:   llvm.fmul
  // LAZY:       llvm.return
  tt.func public @dot_fp8_b_other_block(%a: tensor<32x64xf16, #dot_operand_a>, %b: tensor<64x32xf8E4M3FN, #dot_operand_b>, %c: tensor<32x32xf32, #dpas>, %out: !tt.ptr<f32>) {
    %b16 = tt.fp_to_fp %b : tensor<64x32xf8E4M3FN, #dot_operand_b> -> tensor<64x32xf16, #dot_operand_b>
    cf.br ^bb1
  ^bb1:
    %d = tt.dot %a, %b16, %c, inputPrecision = tf32 : tensor<32x64xf16, #dot_operand_a> * tensor<64x32xf16, #dot_operand_b> -> tensor<32x32xf32, #dpas>
    %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #dpas>
    tt.store %ptrs, %d : tensor<32x32x!tt.ptr<f32>, #dpas>
    tt.return
  }
}

// -----

// A dot with fewer than 4 DPAS calls per K slice (A 8x256, B 256x32: 1 x 2)
// keeps the conversion up front, although it takes 1344 bytes per lane.

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#dpas, kWidth=1}>
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#dpas, kWidth=2}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // LAZY-LABEL: llvm.func spir_kernelcc @dot_fp8_b_few_dpas_per_k
  // LAZY:       llvm.fmul
  // LAZY:       triton_gen.dpas
  // LAZY-NOT:   llvm.fmul
  // LAZY:       llvm.return
  tt.func public @dot_fp8_b_few_dpas_per_k(%a: tensor<8x256xf16, #dot_operand_a>, %b: tensor<256x32xf8E4M3FN, #dot_operand_b>, %c: tensor<8x32xf32, #dpas>, %out: !tt.ptr<f32>) {
    %b16 = tt.fp_to_fp %b : tensor<256x32xf8E4M3FN, #dot_operand_b> -> tensor<256x32xf16, #dot_operand_b>
    %d = tt.dot %a, %b16, %c, inputPrecision = tf32 : tensor<8x256xf16, #dot_operand_a> * tensor<256x32xf16, #dot_operand_b> -> tensor<8x32xf32, #dpas>
    %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<8x32x!tt.ptr<f32>, #dpas>
    tt.store %ptrs, %d : tensor<8x32x!tt.ptr<f32>, #dpas>
    tt.return
  }
}

// -----

// Scalar conversions are lowered as before (no crash in the lazy-operand check).
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // LAZY-LABEL: llvm.func spir_kernelcc @scalar_fp_to_fp
  // LAZY:       llvm.return
  // EAGER-LABEL: llvm.func spir_kernelcc @scalar_fp_to_fp
  // EAGER:       llvm.return
  tt.func public @scalar_fp_to_fp(%p8: !tt.ptr<f8E4M3FN>, %pbf: !tt.ptr<bf16>, %out: !tt.ptr<f16>) {
    %x = tt.load %p8 : !tt.ptr<f8E4M3FN>
    %x16 = tt.fp_to_fp %x : f8E4M3FN -> f16
    %y = tt.load %pbf : !tt.ptr<bf16>
    %y16 = tt.fp_to_fp %y, rounding = rtne : bf16 -> f16
    %s = arith.addf %x16, %y16 : f16
    %ptrs = tt.splat %out : !tt.ptr<f16> -> tensor<16x!tt.ptr<f16>, #blocked>
    %v = tt.splat %s : f16 -> tensor<16xf16, #blocked>
    tt.store %ptrs, %v : tensor<16x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

// A dot with 4 DPAS calls per K slice but below 512 bytes per lane (A 32x32,
// B 32x16, C 32x16: 320 bytes, K = 32, like the P * V dot of decode attention)
// keeps the conversion up front.

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#dpas, kWidth=1}>
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#dpas, kWidth=2}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // LAZY-LABEL: llvm.func spir_kernelcc @dot_fp8_b_under_bytes
  // LAZY:       llvm.fmul
  // LAZY:       triton_gen.dpas
  // LAZY-NOT:   llvm.fmul
  // LAZY:       llvm.return
  tt.func public @dot_fp8_b_under_bytes(%a: tensor<32x32xf16, #dot_operand_a>, %b: tensor<32x16xf8E4M3FN, #dot_operand_b>, %c: tensor<32x16xf32, #dpas>, %out: !tt.ptr<f32>) {
    %b16 = tt.fp_to_fp %b : tensor<32x16xf8E4M3FN, #dot_operand_b> -> tensor<32x16xf16, #dot_operand_b>
    %d = tt.dot %a, %b16, %c, inputPrecision = tf32 : tensor<32x32xf16, #dot_operand_a> * tensor<32x16xf16, #dot_operand_b> -> tensor<32x16xf32, #dpas>
    %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<32x16x!tt.ptr<f32>, #dpas>
    tt.store %ptrs, %d : tensor<32x16x!tt.ptr<f32>, #dpas>
    tt.return
  }
}

// -----

// fp8 A and B with repCluster = [4, 2] (8 DPAS calls per K slice, 768 bytes
// per lane): both operands are converted per DPAS operand. A DPAS operand
// split with the other operand's repCluster would not match the DPAS operand
// types.

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 2]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#dpas, kWidth=1}>
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#dpas, kWidth=2}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // LAZY-LABEL: llvm.func spir_kernelcc @dot_fp8_ab_rep_cluster
  // LAZY:       llvm.fmul
  // LAZY:       triton_gen.dpas
  // LAZY:       llvm.fmul
  // LAZY:       triton_gen.dpas
  // LAZY:       llvm.return
  // EAGER-LABEL: llvm.func spir_kernelcc @dot_fp8_ab_rep_cluster
  // EAGER:       triton_gen.dpas
  // EAGER-NOT:   llvm.fmul
  // EAGER:       llvm.return
  tt.func public @dot_fp8_ab_rep_cluster(%a: tensor<32x64xf8E4M3FN, #dot_operand_a>, %b: tensor<64x32xf8E4M3FN, #dot_operand_b>, %c: tensor<32x32xf32, #dpas>, %out: !tt.ptr<f32>) {
    %a16 = tt.fp_to_fp %a : tensor<32x64xf8E4M3FN, #dot_operand_a> -> tensor<32x64xf16, #dot_operand_a>
    %b16 = tt.fp_to_fp %b : tensor<64x32xf8E4M3FN, #dot_operand_b> -> tensor<64x32xf16, #dot_operand_b>
    %d = tt.dot %a16, %b16, %c, inputPrecision = tf32 : tensor<32x64xf16, #dot_operand_a> * tensor<64x32xf16, #dot_operand_b> -> tensor<32x32xf32, #dpas>
    %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #dpas>
    tt.store %ptrs, %d : tensor<32x32x!tt.ptr<f32>, #dpas>
    tt.return
  }
}

// -----

// fp8 A only with repCluster = [4, 2] (8 DPAS calls per K slice, 768 bytes per
// lane): A is converted per DPAS operand. B needs no conversion, so an fmul
// after the first DPAS call can only come from A.

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 2]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#dpas, kWidth=1}>
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#dpas, kWidth=2}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // LAZY-LABEL: llvm.func spir_kernelcc @dot_fp8_a_rep_cluster
  // LAZY:       llvm.fmul
  // LAZY:       triton_gen.dpas
  // LAZY:       llvm.fmul
  // LAZY:       triton_gen.dpas
  // LAZY:       llvm.return
  // EAGER-LABEL: llvm.func spir_kernelcc @dot_fp8_a_rep_cluster
  // EAGER:       triton_gen.dpas
  // EAGER-NOT:   llvm.fmul
  // EAGER:       llvm.return
  tt.func public @dot_fp8_a_rep_cluster(%a: tensor<32x64xf8E4M3FN, #dot_operand_a>, %b: tensor<64x32xf16, #dot_operand_b>, %c: tensor<32x32xf32, #dpas>, %out: !tt.ptr<f32>) {
    %a16 = tt.fp_to_fp %a : tensor<32x64xf8E4M3FN, #dot_operand_a> -> tensor<32x64xf16, #dot_operand_a>
    %d = tt.dot %a16, %b, %c, inputPrecision = tf32 : tensor<32x64xf16, #dot_operand_a> * tensor<64x32xf16, #dot_operand_b> -> tensor<32x32xf32, #dpas>
    %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #dpas>
    tt.store %ptrs, %d : tensor<32x32x!tt.ptr<f32>, #dpas>
    tt.return
  }
}
