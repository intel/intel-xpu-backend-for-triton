// RUN: triton-opt %s -split-input-file --tritonintelgpu-lower-to-2d-block-load --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm --canonicalize | FileCheck %s

// COM: With TRITON_INTEL_FAST_MATH=1 (the ttig.fast_math module attribute) every floating-point
// COM: instruction later gets `fast`, and the rebias multiply by 256.0 would be reassociated with
// COM: a multiply of the converted value: `x.to(f16) * s` becomes `x * (256 * s)`, which overflows
// COM: fp16 for s >= 256. The product goes through an inline-asm mov, which neither LLVM nor IGC
// COM: can look through, before it is used. Without the attribute the sequence is the one in
// COM: fp8e4m3_to_fp16.mlir, fp8e4m3_to_fp16_pairs.mlir and fp8e4m3_to_fp16_block_load.mlir.
#blocked = #ttg.blocked<{sizePerThread = [16], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.min_sg_size = 16 : i32, ttig.target_arch = "spir64", ttig.fast_math} {
  // CHECK-LABEL: @convert_fp8e4m3_to_fp16_fast_math
  tt.func public @convert_fp8e4m3_to_fp16_fast_math(%src: tensor<16xf8E4M3FN, #blocked>) -> tensor<16xf16, #blocked> {
    // CHECK-DAG: [[C256:%.*]] = llvm.mlir.constant(2.560000e+02 : f16) : f16
    // CHECK: [[X:%.*]] = llvm.bitcast {{%.*}} : i16 to f16
    // CHECK-NEXT: [[M:%.*]] = llvm.fmul [[X]], [[C256]] : f16
    // CHECK-NEXT: [[MI:%.*]] = llvm.bitcast [[M]] : f16 to i16
    // CHECK-NEXT: [[MOV:%.*]] = llvm.inline_asm asm_dialect = att operand_attrs = [] "mov (M1, 32) $0(0,0)<1> $1(0,0)<1;1,0>", "=rw,rw" [[MI]] : (i16) -> i16
    // CHECK-NEXT: [[H:%.*]] = llvm.bitcast [[MOV]] : i16 to f16
    // CHECK-NEXT: llvm.intr.fabs([[H]])
    // CHECK-COUNT-15: llvm.inline_asm
    // CHECK-NOT: llvm.inline_asm
    %dst = tt.fp_to_fp %src : tensor<16xf8E4M3FN, #blocked> -> tensor<16xf16, #blocked>
    tt.return %dst : tensor<16xf16, #blocked>
  }
}
// -----

#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.target_arch = "spir64", ttig.fast_math} {
  // CHECK-LABEL: @convert_fp8e4m3_to_fp16_pairs_fast_math
  tt.func public @convert_fp8e4m3_to_fp16_pairs_fast_math(%src: tensor<32x16xf8E4M3FN, #dot1>) -> tensor<32x16xf16, #dot1> {
    // CHECK: [[M:%.*]] = llvm.fmul {{%.*}} : vector<2xf16>
    // CHECK-NEXT: [[MI:%.*]] = llvm.bitcast [[M]] : vector<2xf16> to i32
    // CHECK-NEXT: [[MOV:%.*]] = llvm.inline_asm asm_dialect = att operand_attrs = [] "mov (M1, 16) $0(0,0)<1> $1(0,0)<1;1,0>", "=rw,rw" [[MI]] : (i32) -> i32
    // CHECK-NEXT: llvm.bitcast [[MOV]] : i32 to vector<2xf16>
    %dst = tt.fp_to_fp %src : tensor<32x16xf8E4M3FN, #dot1> -> tensor<32x16xf16, #dot1>
    tt.return %dst : tensor<32x16xf16, #dot1>
  }
}

// -----

// COM: An operand of a dot that lowers to DPAS feeds no multiply that could be reassociated with
// COM: the rebias, and keeps the faster sequence.
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_subgroup_matrix_multiply_accumulate, ttig.target_arch = "spir64", ttig.fast_math} {
  // CHECK-LABEL: @fp8e4m3_to_fp16_dpas_operand_fast_math
  tt.func public @fp8e4m3_to_fp16_dpas_operand_fast_math(%a: tensor<8x16xf16, #dot0>, %src: tensor<16x16xf8E4M3FN, #dot1>, %c: tensor<8x16xf32, #mma>) -> tensor<8x16xf32, #mma> {
    // CHECK-NOT: llvm.inline_asm
    // CHECK: triton_gen.dpas
    %b = tt.fp_to_fp %src : tensor<16x16xf8E4M3FN, #dot1> -> tensor<16x16xf16, #dot1>
    %d = tt.dot %a, %b, %c : tensor<8x16xf16, #dot0> * tensor<16x16xf16, #dot1> -> tensor<8x16xf32, #mma>
    tt.return %d : tensor<8x16xf32, #mma>
  }
}

// -----

// COM: A dot with a blocked layout lowers to multiply-adds, so its operand is upcast with the
// COM: barrier.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
#dota = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotb = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.min_sg_size = 16 : i32, ttig.target_arch = "spir64", ttig.fast_math} {
  // CHECK-LABEL: @fp8e4m3_to_fp16_fma_operand_fast_math
  tt.func public @fp8e4m3_to_fp16_fma_operand_fast_math(%src: tensor<32x16xf8E4M3FN, #dota>, %b: tensor<16x32xf16, #dotb>, %c: tensor<32x32xf16, #blocked>) -> tensor<32x32xf16, #blocked> {
    // CHECK: [[M:%.*]] = llvm.fmul {{%.*}} : vector<2xf16>
    // CHECK-NEXT: [[MI:%.*]] = llvm.bitcast [[M]] : vector<2xf16> to i32
    // CHECK-NEXT: llvm.inline_asm {{.*}} [[MI]] : (i32) -> i32
    %a = tt.fp_to_fp %src : tensor<32x16xf8E4M3FN, #dota> -> tensor<32x16xf16, #dota>
    %d = tt.dot %a, %b, %c : tensor<32x16xf16, #dota> * tensor<16x32xf16, #dotb> -> tensor<32x32xf16, #blocked>
    tt.return %d : tensor<32x32xf16, #blocked>
  }
}

// -----

// COM: Bytes of 32-bit block loads are read from the loaded dwords; with a multiply of the result,
// COM: that sequence gets the barrier too.
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.target_arch = "spir64", ttig.fast_math} {
  // CHECK-LABEL: @fp8_upcast_from_block_load_times_scale_fast_math
  tt.func public @fp8_upcast_from_block_load_times_scale_fast_math(%arg0: !tt.ptr<f8E4M3FN>, %arg1: i32 {tt.divisibility = 64 : i32}, %arg2: i32 {tt.divisibility = 64 : i32}, %s: f16) -> tensor<64x16xf16, #dot1> {
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c64_i32 = arith.constant 64 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%c64_i32, %c64_i32], [%c64_i64, %c1_i64] : <f8E4M3FN>, <16x64xf8E4M3FN>
    // CHECK: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 32, {{.*}} transpose = true
    %x = tt.descriptor_load %desc[%arg1, %arg2] {ttig.block_io = "column_major"} : !tt.tensordesc<16x64xf8E4M3FN> -> tensor<64x16xf8E4M3FN, #dot1>
    // CHECK: [[T:%.*]] = llvm.trunc {{%.*}} : i32 to i16
    // CHECK: [[M:%.*]] = llvm.fmul {{%.*}} : f16
    // CHECK-NEXT: [[MI:%.*]] = llvm.bitcast [[M]] : f16 to i16
    // CHECK-NEXT: llvm.inline_asm {{.*}} [[MI]] : (i16) -> i16
    %y = tt.fp_to_fp %x : tensor<64x16xf8E4M3FN, #dot1> -> tensor<64x16xf16, #dot1>
    %ss = tt.splat %s : f16 -> tensor<64x16xf16, #dot1>
    %z = arith.mulf %y, %ss : tensor<64x16xf16, #dot1>
    tt.return %z : tensor<64x16xf16, #dot1>
  }
}

// -----

// COM: A result that reaches DPAS through a layout conversion and shared memory keeps the faster
// COM: sequence, also with a dealloc of the buffer (Gluon's _keep_alive); one that reaches a
// COM: multiply through a layout conversion gets the barrier.
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_subgroup_matrix_multiply_accumulate, ttig.target_arch = "spir64", ttig.fast_math} {
  // CHECK-LABEL: @fp8_upcast_forwarded_to_dpas_fast_math
  tt.func public @fp8_upcast_forwarded_to_dpas_fast_math(%a: tensor<8x16xf16, #dot0>, %src: tensor<16x16xf8E4M3FN, #blocked>, %c: tensor<8x16xf32, #mma>) -> tensor<8x16xf32, #mma> {
    // CHECK-NOT: llvm.inline_asm
    // CHECK: triton_gen.dpas
    %b = tt.fp_to_fp %src : tensor<16x16xf8E4M3FN, #blocked> -> tensor<16x16xf16, #blocked>
    %cvt = ttg.convert_layout %b : tensor<16x16xf16, #blocked> -> tensor<16x16xf16, #blocked>
    %buf = ttg.local_alloc %cvt : (tensor<16x16xf16, #blocked>) -> !ttg.memdesc<16x16xf16, #shared, #smem>
    %bd = ttg.local_load %buf : !ttg.memdesc<16x16xf16, #shared, #smem> -> tensor<16x16xf16, #dot1>
    %d = tt.dot %a, %bd, %c : tensor<8x16xf16, #dot0> * tensor<16x16xf16, #dot1> -> tensor<8x16xf32, #mma>
    ttg.local_dealloc %buf : !ttg.memdesc<16x16xf16, #shared, #smem>
    tt.return %d : tensor<8x16xf32, #mma>
  }

  // CHECK-LABEL: @fp8_upcast_forwarded_to_multiply_fast_math
  tt.func public @fp8_upcast_forwarded_to_multiply_fast_math(%src: tensor<16x16xf8E4M3FN, #blocked>, %s: f16) -> tensor<16x16xf16, #blocked> {
    // CHECK: [[M:%.*]] = llvm.fmul {{%.*}} : f16
    // CHECK-NEXT: [[MI:%.*]] = llvm.bitcast [[M]] : f16 to i16
    // CHECK-NEXT: llvm.inline_asm {{.*}} [[MI]] : (i16) -> i16
    %b = tt.fp_to_fp %src : tensor<16x16xf8E4M3FN, #blocked> -> tensor<16x16xf16, #blocked>
    %cvt = ttg.convert_layout %b : tensor<16x16xf16, #blocked> -> tensor<16x16xf16, #blocked>
    %ss = tt.splat %s : f16 -> tensor<16x16xf16, #blocked>
    %z = arith.mulf %cvt, %ss : tensor<16x16xf16, #blocked>
    tt.return %z : tensor<16x16xf16, #blocked>
  }

  // COM: A result that reaches both a DPAS dot and a multiply gets the barrier.
  // CHECK-LABEL: @fp8_upcast_to_dpas_and_multiply_fast_math
  tt.func public @fp8_upcast_to_dpas_and_multiply_fast_math(%a: tensor<8x16xf16, #dot0>, %src: tensor<16x16xf8E4M3FN, #blocked>, %c: tensor<8x16xf32, #mma>, %s: f16) -> (tensor<8x16xf32, #mma>, tensor<16x16xf16, #blocked>) {
    // CHECK: [[M:%.*]] = llvm.fmul {{%.*}} : f16
    // CHECK-NEXT: [[MI:%.*]] = llvm.bitcast [[M]] : f16 to i16
    // CHECK-NEXT: llvm.inline_asm {{.*}} [[MI]] : (i16) -> i16
    %b = tt.fp_to_fp %src : tensor<16x16xf8E4M3FN, #blocked> -> tensor<16x16xf16, #blocked>
    %buf = ttg.local_alloc %b : (tensor<16x16xf16, #blocked>) -> !ttg.memdesc<16x16xf16, #shared, #smem>
    %bd = ttg.local_load %buf : !ttg.memdesc<16x16xf16, #shared, #smem> -> tensor<16x16xf16, #dot1>
    %d = tt.dot %a, %bd, %c : tensor<8x16xf16, #dot0> * tensor<16x16xf16, #dot1> -> tensor<8x16xf32, #mma>
    %ss = tt.splat %s : f16 -> tensor<16x16xf16, #blocked>
    %z = arith.mulf %b, %ss : tensor<16x16xf16, #blocked>
    tt.return %d, %z : tensor<8x16xf32, #mma>, tensor<16x16xf16, #blocked>
  }

  // COM: An upcast to f32 goes through f16, but feeds f32 multiplies, where 256 * s does not overflow.
  // CHECK-LABEL: @fp8_upcast_to_f32_times_scale_fast_math
  tt.func public @fp8_upcast_to_f32_times_scale_fast_math(%src: tensor<16x16xf8E4M3FN, #blocked>, %s: f32) -> tensor<16x16xf32, #blocked> {
    // CHECK-NOT: llvm.inline_asm
    // CHECK: llvm.return
    %b = tt.fp_to_fp %src : tensor<16x16xf8E4M3FN, #blocked> -> tensor<16x16xf32, #blocked>
    %ss = tt.splat %s : f32 -> tensor<16x16xf32, #blocked>
    %z = arith.mulf %b, %ss : tensor<16x16xf32, #blocked>
    tt.return %z : tensor<16x16xf32, #blocked>
  }
}

// -----

// COM: The LTS integer-domain sequence gets the barrier too: the integer operations after its multiply can be
// COM: folded away for inputs that LLVM knows to be positive and never reserved.
#blocked = #ttg.blocked<{sizePerThread = [16], threadsPerWarp = [16], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.is_lts, ttig.min_sg_size = 16 : i32, ttig.target_arch = "spir64", ttig.fast_math} {
  // CHECK-LABEL: @fp8_upcast_lts_times_scale_fast_math
  tt.func public @fp8_upcast_lts_times_scale_fast_math(%src: tensor<16xf8E4M3FN, #blocked>, %s: f16) -> tensor<16xf16, #blocked> {
    // CHECK: [[M:%.*]] = llvm.fmul {{%.*}} : vector<2xf16>
    // CHECK-NEXT: [[MI:%.*]] = llvm.bitcast [[M]] : vector<2xf16> to i32
    // CHECK-NEXT: llvm.inline_asm {{.*}} [[MI]] : (i32) -> i32
    %b = tt.fp_to_fp %src : tensor<16xf8E4M3FN, #blocked> -> tensor<16xf16, #blocked>
    %ss = tt.splat %s : f16 -> tensor<16xf16, #blocked>
    %z = arith.mulf %b, %ss : tensor<16xf16, #blocked>
    tt.return %z : tensor<16xf16, #blocked>
  }
}
