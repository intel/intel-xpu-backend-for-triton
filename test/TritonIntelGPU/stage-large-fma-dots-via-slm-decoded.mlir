// RUN: triton-opt %s -split-input-file --tritonintelgpu-stage-large-fma-dots-via-slm | FileCheck %s
// RUN: triton-opt %s -split-input-file --tritonintelgpu-stage-large-fma-dots-via-slm="stage-decoded=false" | FileCheck %s --check-prefix=FP8
// RUN: triton-opt %s -split-input-file --tritonintelgpu-stage-large-fma-dots-via-slm="slm-size=131072" | FileCheck %s --check-prefix=SLM128K
// RUN: triton-opt %s -split-input-file --tritonintelgpu-stage-large-fma-dots-via-slm="slm-size=77824" | FileCheck %s --check-prefix=SLM76K

// fp8 operands that AccelerateMatmul decoded to fp16 before the layout
// conversion (an fp32 dot: the fp16 -> fp32 cast is after the conversion).
// The decoded fp16 values (16 KB + 32 KB) fit in the 56 KB cap, so they are
// staged and only fp16 -> fp32 is replayed per K tile. With
// stage-decoded=false the fp8 values are staged and the decode is replayed per
// K tile. Either way the K tile is 8, the largest that keeps a tile's operand
// values within the pass's per-thread register budget.

#blocked  = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#dot0     = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dot1     = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @fp8_decoded_staged
  // CHECK-DAG: [[AD:%.*]] = tt.fp_to_fp {{%[0-9]+}} : tensor<64x128xf8E4M3FN, #{{.*}}> -> tensor<64x128xf16, #{{.*}}>
  // CHECK-DAG: [[BD:%.*]] = tt.fp_to_fp {{%[0-9]+}} : tensor<128x128xf8E5M2, #{{.*}}> -> tensor<128x128xf16, #{{.*}}>
  // CHECK-DAG: ttg.local_alloc [[AD]] : (tensor<64x128xf16
  // CHECK-DAG: ttg.local_alloc [[BD]] : (tensor<128x128xf16
  // CHECK-NOT: ttg.convert_layout
  // CHECK: ttg.local_load {{.*}} -> tensor<64x8xf16, #ttg.dot_op<{opIdx = 0
  // CHECK: ttg.local_load {{.*}} -> tensor<8x128xf16, #ttg.dot_op<{opIdx = 1
  // CHECK-NOT: f8E
  // CHECK: tt.fp_to_fp {{.*}}xf16, #ttg.dot_op<{opIdx = 0{{.*}}xf32
  // CHECK: tt.fp_to_fp {{.*}}xf16, #ttg.dot_op<{opIdx = 1{{.*}}xf32
  // CHECK: tt.dot
  // CHECK-NOT: f8E
  // CHECK: tt.return

  // FP8-LABEL: @fp8_decoded_staged
  // FP8-DAG: ttg.local_alloc {{%[0-9]+}} : (tensor<64x128xf8E4M3FN
  // FP8-DAG: ttg.local_alloc {{%[0-9]+}} : (tensor<128x128xf8E5M2
  // FP8: ttg.local_load {{.*}} -> tensor<64x8xf8E4M3FN, #ttg.dot_op<{opIdx = 0
  // FP8: ttg.local_load {{.*}} -> tensor<8x128xf8E5M2, #ttg.dot_op<{opIdx = 1
  // FP8: tt.fp_to_fp {{.*}}xf8E4M3FN, #ttg.dot_op<{opIdx = 0{{.*}}xf16
  // FP8: tt.fp_to_fp {{.*}}xf16, #ttg.dot_op<{opIdx = 0{{.*}}xf32
  // FP8: tt.fp_to_fp {{.*}}xf8E5M2, #ttg.dot_op<{opIdx = 1{{.*}}xf16
  // FP8: tt.fp_to_fp {{.*}}xf16, #ttg.dot_op<{opIdx = 1{{.*}}xf32
  // FP8: tt.dot
  tt.func public @fp8_decoded_staged(%parg0: tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked>,
                                     %parg1: tensor<128x128x!tt.ptr<f8E5M2>, #blocked1>) -> tensor<64x128xf32, #blocked> {
    %arg0 = tt.load %parg0 : tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked>
    %arg1 = tt.load %parg1 : tensor<128x128x!tt.ptr<f8E5M2>, #blocked1>
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked>
    %ad = tt.fp_to_fp %arg0 : tensor<64x128xf8E4M3FN, #blocked> -> tensor<64x128xf16, #blocked>
    %ac = ttg.convert_layout %ad : tensor<64x128xf16, #blocked> -> tensor<64x128xf16, #dot0>
    %a = tt.fp_to_fp %ac : tensor<64x128xf16, #dot0> -> tensor<64x128xf32, #dot0>
    %bd = tt.fp_to_fp %arg1 : tensor<128x128xf8E5M2, #blocked1> -> tensor<128x128xf16, #blocked1>
    %bc = ttg.convert_layout %bd : tensor<128x128xf16, #blocked1> -> tensor<128x128xf16, #dot1>
    %b = tt.fp_to_fp %bc : tensor<128x128xf16, #dot1> -> tensor<128x128xf32, #dot1>
    %r = tt.dot %a, %b, %cst {inputPrecision = 2 : i32, maxNumImpreciseAcc = 0 : i32}
         : tensor<64x128xf32, #dot0> * tensor<128x128xf32, #dot1> -> tensor<64x128xf32, #blocked>
    tt.return %r : tensor<64x128xf32, #blocked>
  }
}

// -----

// Decoded A and B (32 KB + 32 KB) don't fit in the default 56 KB cap. A has
// more replicated copies (16 per element, B has 8), so only A is staged
// decoded (32 KB + 16 KB). With a 128 KB target both are staged decoded.

#blocked  = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#dot0     = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dot1     = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @fp8_decoded_partial
  // CHECK-DAG: [[AD:%.*]] = tt.fp_to_fp {{%[0-9]+}} : tensor<128x128xf8E4M3FN, #{{.*}}> -> tensor<128x128xbf16, #{{.*}}>
  // CHECK-DAG: ttg.local_alloc [[AD]] : (tensor<128x128xbf16
  // CHECK-DAG: ttg.local_alloc {{%[0-9]+}} : (tensor<128x128xf8E4M3FN
  // CHECK: ttg.local_load {{.*}} -> tensor<128x8xbf16, #ttg.dot_op<{opIdx = 0
  // CHECK: ttg.local_load {{.*}} -> tensor<8x128xf8E4M3FN, #ttg.dot_op<{opIdx = 1
  // CHECK-NOT: tt.fp_to_fp {{.*}}xf8E4M3FN, #ttg.dot_op<{opIdx = 0
  // CHECK: tt.fp_to_fp {{.*}}xf8E4M3FN, #ttg.dot_op<{opIdx = 1{{.*}}xbf16
  // CHECK: tt.dot

  // SLM128K-LABEL: @fp8_decoded_partial
  // SLM128K-DAG: ttg.local_alloc {{.*}} : (tensor<128x128xbf16, #{{.*}}>) -> !ttg.memdesc<128x128xbf16, #{{.*}}, #smem>
  // SLM128K-DAG: ttg.local_alloc {{.*}} : (tensor<128x128xbf16, #{{.*}}>) -> !ttg.memdesc<128x128xbf16, #{{.*}}, #smem>
  // SLM128K-NOT: f8E4M3FN, #ttg.dot_op
  // SLM128K: tt.return
  tt.func public @fp8_decoded_partial(%parg0: tensor<128x128x!tt.ptr<f8E4M3FN>, #blocked>,
                                      %parg1: tensor<128x128x!tt.ptr<f8E4M3FN>, #blocked1>) -> tensor<128x128xbf16, #blocked> {
    %arg0 = tt.load %parg0 : tensor<128x128x!tt.ptr<f8E4M3FN>, #blocked>
    %arg1 = tt.load %parg1 : tensor<128x128x!tt.ptr<f8E4M3FN>, #blocked1>
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xbf16, #blocked>
    %ad = tt.fp_to_fp %arg0 : tensor<128x128xf8E4M3FN, #blocked> -> tensor<128x128xbf16, #blocked>
    %a = ttg.convert_layout %ad : tensor<128x128xbf16, #blocked> -> tensor<128x128xbf16, #dot0>
    %bd = tt.fp_to_fp %arg1 : tensor<128x128xf8E4M3FN, #blocked1> -> tensor<128x128xbf16, #blocked1>
    %b = ttg.convert_layout %bd : tensor<128x128xbf16, #blocked1> -> tensor<128x128xbf16, #dot1>
    %r = tt.dot %a, %b, %cst {inputPrecision = 2 : i32, maxNumImpreciseAcc = 0 : i32}
         : tensor<128x128xbf16, #dot0> * tensor<128x128xbf16, #dot1> -> tensor<128x128xbf16, #blocked>
    tt.return %r : tensor<128x128xbf16, #blocked>
  }
}

// -----

// With native fp8 conversion, fp8 -> fp16 is one builtin: the fp8 values are
// staged and the decode replayed. fp8 -> bf16 is still emulated, so a bf16
// decode is staged decoded.

#blocked  = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#dot0     = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dot1     = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_f8_conversion} {
  // CHECK-LABEL: @native_f8_conversion
  // CHECK-DAG: ttg.local_alloc {{%[0-9]+}} : (tensor<64x128xf8E4M3FN
  // CHECK-DAG: ttg.local_alloc {{.*}} : (tensor<128x128xbf16
  // CHECK: tt.fp_to_fp {{.*}}xf8E4M3FN, #ttg.dot_op<{opIdx = 0{{.*}}xf16
  // CHECK-NOT: f8E5M2, #ttg.dot_op
  // CHECK: tt.return
  tt.func public @native_f8_conversion(%parg0: tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked>,
                                       %parg1: tensor<128x128x!tt.ptr<f8E5M2>, #blocked1>) -> tensor<64x128xf32, #blocked> {
    %arg0 = tt.load %parg0 : tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked>
    %arg1 = tt.load %parg1 : tensor<128x128x!tt.ptr<f8E5M2>, #blocked1>
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked>
    %ad = tt.fp_to_fp %arg0 : tensor<64x128xf8E4M3FN, #blocked> -> tensor<64x128xf16, #blocked>
    %ac = ttg.convert_layout %ad : tensor<64x128xf16, #blocked> -> tensor<64x128xf16, #dot0>
    %a = tt.fp_to_fp %ac : tensor<64x128xf16, #dot0> -> tensor<64x128xf32, #dot0>
    %bd = tt.fp_to_fp %arg1 : tensor<128x128xf8E5M2, #blocked1> -> tensor<128x128xbf16, #blocked1>
    %bc = ttg.convert_layout %bd : tensor<128x128xbf16, #blocked1> -> tensor<128x128xbf16, #dot1>
    %b = arith.extf %bc : tensor<128x128xbf16, #dot1> to tensor<128x128xf32, #dot1>
    %r = tt.dot %a, %b, %cst {inputPrecision = 2 : i32, maxNumImpreciseAcc = 0 : i32}
         : tensor<64x128xf32, #dot0> * tensor<128x128xf32, #dot1> -> tensor<64x128xf32, #blocked>
    tt.return %r : tensor<64x128xf32, #blocked>
  }
}

// -----

// A 16-bit cast before the conversion that is not an fp8 decode (here
// fp16 -> fp32 written by the user) is not a decode: the smallest value is
// staged, as before.

#blocked  = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#dot0     = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dot1     = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @f16_cast_before_conversion
  // CHECK-DAG: ttg.local_alloc {{%[0-9]+}} : (tensor<64x128xf16
  // CHECK-DAG: ttg.local_alloc {{%[0-9]+}} : (tensor<128x64xf16
  // CHECK: tt.fp_to_fp {{.*}}xf16, #ttg.dot_op<{opIdx = 0{{.*}}xf32
  tt.func public @f16_cast_before_conversion(%parg0: tensor<64x128x!tt.ptr<f16>, #blocked>,
                                             %parg1: tensor<128x64x!tt.ptr<f16>, #blocked1>) -> tensor<64x64xf32, #blocked> {
    %arg0 = tt.load %parg0 : tensor<64x128x!tt.ptr<f16>, #blocked>
    %arg1 = tt.load %parg1 : tensor<128x64x!tt.ptr<f16>, #blocked1>
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %ad = tt.fp_to_fp %arg0 : tensor<64x128xf16, #blocked> -> tensor<64x128xf32, #blocked>
    %a = ttg.convert_layout %ad : tensor<64x128xf32, #blocked> -> tensor<64x128xf32, #dot0>
    %bd = tt.fp_to_fp %arg1 : tensor<128x64xf16, #blocked1> -> tensor<128x64xf32, #blocked1>
    %b = ttg.convert_layout %bd : tensor<128x64xf32, #blocked1> -> tensor<128x64xf32, #dot1>
    %r = tt.dot %a, %b, %cst {inputPrecision = 2 : i32, maxNumImpreciseAcc = 0 : i32}
         : tensor<64x128xf32, #dot0> * tensor<128x64xf32, #dot1> -> tensor<64x64xf32, #blocked>
    tt.return %r : tensor<64x64xf32, #blocked>
  }
}

// -----

// The fp8 decode is not the only cast before the conversion (fp8 -> fp16 ->
// fp32): staging the conversion's source would stage fp32. The fp8 value is
// staged and both casts are replayed.

#blocked  = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#dot0     = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dot1     = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @fp8_decode_widen_before_conversion
  // CHECK-NOT: ttg.local_alloc {{.*}} : (tensor<{{[0-9x]+}}xf{{(16|32)}}
  // CHECK-DAG: ttg.local_alloc {{%[0-9]+}} : (tensor<64x128xf8E4M3FN
  // CHECK-DAG: ttg.local_alloc {{%[0-9]+}} : (tensor<128x64xf8E5M2
  // CHECK: tt.fp_to_fp {{.*}}xf8E4M3FN, #ttg.dot_op<{opIdx = 0{{.*}}xf16
  // CHECK: tt.fp_to_fp {{.*}}xf16, #ttg.dot_op<{opIdx = 0{{.*}}xf32
  // CHECK: tt.dot
  tt.func public @fp8_decode_widen_before_conversion(%parg0: tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked>,
                                                     %parg1: tensor<128x64x!tt.ptr<f8E5M2>, #blocked1>) -> tensor<64x64xf32, #blocked> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %a8 = tt.load %parg0 : tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked>
    %b8 = tt.load %parg1 : tensor<128x64x!tt.ptr<f8E5M2>, #blocked1>
    %a16 = tt.fp_to_fp %a8 : tensor<64x128xf8E4M3FN, #blocked> -> tensor<64x128xf16, #blocked>
    %a32 = tt.fp_to_fp %a16 : tensor<64x128xf16, #blocked> -> tensor<64x128xf32, #blocked>
    %a = ttg.convert_layout %a32 : tensor<64x128xf32, #blocked> -> tensor<64x128xf32, #dot0>
    %b16 = tt.fp_to_fp %b8 : tensor<128x64xf8E5M2, #blocked1> -> tensor<128x64xf16, #blocked1>
    %b32 = tt.fp_to_fp %b16 : tensor<128x64xf16, #blocked1> -> tensor<128x64xf32, #blocked1>
    %b = ttg.convert_layout %b32 : tensor<128x64xf32, #blocked1> -> tensor<128x64xf32, #dot1>
    %r = tt.dot %a, %b, %cst {inputPrecision = 2 : i32, maxNumImpreciseAcc = 0 : i32}
         : tensor<64x128xf32, #dot0> * tensor<128x64xf32, #dot1> -> tensor<64x64xf32, #blocked>
    tt.return %r : tensor<64x64xf32, #blocked>
  }
}

// -----

// An fp8 cast before the conversion that does not produce a 16-bit type
// (fp8 -> fp32) is not staged decoded either: the fp8 value is staged and
// the cast is replayed.

#blocked  = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#dot0     = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dot1     = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @fp8_to_f32_before_conversion
  // CHECK-NOT: ttg.local_alloc {{.*}} : (tensor<{{[0-9x]+}}xf{{(16|32)}}
  // CHECK-DAG: ttg.local_alloc {{%[0-9]+}} : (tensor<64x128xf8E4M3FN
  // CHECK-DAG: ttg.local_alloc {{%[0-9]+}} : (tensor<128x64xf8E5M2
  // CHECK: tt.fp_to_fp {{.*}}xf8E4M3FN, #ttg.dot_op<{opIdx = 0{{.*}}xf32
  // CHECK: tt.dot
  tt.func public @fp8_to_f32_before_conversion(%parg0: tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked>,
                                               %parg1: tensor<128x64x!tt.ptr<f8E5M2>, #blocked1>) -> tensor<64x64xf32, #blocked> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %a8 = tt.load %parg0 : tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked>
    %b8 = tt.load %parg1 : tensor<128x64x!tt.ptr<f8E5M2>, #blocked1>
    %a32 = tt.fp_to_fp %a8 : tensor<64x128xf8E4M3FN, #blocked> -> tensor<64x128xf32, #blocked>
    %a = ttg.convert_layout %a32 : tensor<64x128xf32, #blocked> -> tensor<64x128xf32, #dot0>
    %b32 = tt.fp_to_fp %b8 : tensor<128x64xf8E5M2, #blocked1> -> tensor<128x64xf32, #blocked1>
    %b = ttg.convert_layout %b32 : tensor<128x64xf32, #blocked1> -> tensor<128x64xf32, #dot1>
    %r = tt.dot %a, %b, %cst {inputPrecision = 2 : i32, maxNumImpreciseAcc = 0 : i32}
         : tensor<64x128xf32, #dot0> * tensor<128x64xf32, #dot1> -> tensor<64x64xf32, #blocked>
    tt.return %r : tensor<64x64xf32, #blocked>
  }
}

// -----

// Operands with different element counts: A (256x128) has 8 copies per element
// and 2048 elements per thread, B (128x32) has 16 copies per element and 512
// per thread. A has more copies in total, so decoding it saves more decodes and
// is tried first. Staging takes 36 KB as is, 40 KB with B decoded, 68 KB with
// A decoded and 72 KB with both. If A's option does not fit, B's is tried.
//   default (56 KB):   B decoded, A fp8
//   slm-size=77824 (68 KB): A decoded, B fp8
//   slm-size=131072 (120 KB): both decoded

#blocked  = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#dot0     = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dot1     = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @copies_in_total
  // CHECK-DAG: ttg.local_alloc {{%[0-9]+}} : (tensor<256x128xf8E4M3FN
  // CHECK-DAG: ttg.local_alloc {{%[0-9]+}} : (tensor<128x32xf16
  // CHECK: tt.return

  // SLM76K-LABEL: @copies_in_total
  // SLM76K-DAG: ttg.local_alloc {{%[0-9]+}} : (tensor<256x128xf16
  // SLM76K-DAG: ttg.local_alloc {{%[0-9]+}} : (tensor<128x32xf8E5M2
  // SLM76K: tt.return

  // SLM128K-LABEL: @copies_in_total
  // SLM128K-DAG: ttg.local_alloc {{%[0-9]+}} : (tensor<256x128xf16
  // SLM128K-DAG: ttg.local_alloc {{%[0-9]+}} : (tensor<128x32xf16
  // SLM128K: tt.return

  // FP8-LABEL: @copies_in_total
  // FP8-DAG: ttg.local_alloc {{%[0-9]+}} : (tensor<256x128xf8E4M3FN
  // FP8-DAG: ttg.local_alloc {{%[0-9]+}} : (tensor<128x32xf8E5M2
  // FP8: tt.return
  tt.func public @copies_in_total(%parg0: tensor<256x128x!tt.ptr<f8E4M3FN>, #blocked>,
                                  %parg1: tensor<128x32x!tt.ptr<f8E5M2>, #blocked1>) -> tensor<256x32xf32, #blocked> {
    %cst = arith.constant dense<0.000000e+00> : tensor<256x32xf32, #blocked>
    %a8 = tt.load %parg0 : tensor<256x128x!tt.ptr<f8E4M3FN>, #blocked>
    %b8 = tt.load %parg1 : tensor<128x32x!tt.ptr<f8E5M2>, #blocked1>
    %ad = tt.fp_to_fp %a8 : tensor<256x128xf8E4M3FN, #blocked> -> tensor<256x128xf16, #blocked>
    %ac = ttg.convert_layout %ad : tensor<256x128xf16, #blocked> -> tensor<256x128xf16, #dot0>
    %a = tt.fp_to_fp %ac : tensor<256x128xf16, #dot0> -> tensor<256x128xf32, #dot0>
    %bd = tt.fp_to_fp %b8 : tensor<128x32xf8E5M2, #blocked1> -> tensor<128x32xf16, #blocked1>
    %bc = ttg.convert_layout %bd : tensor<128x32xf16, #blocked1> -> tensor<128x32xf16, #dot1>
    %b = tt.fp_to_fp %bc : tensor<128x32xf16, #dot1> -> tensor<128x32xf32, #dot1>
    %r = tt.dot %a, %b, %cst {inputPrecision = 2 : i32, maxNumImpreciseAcc = 0 : i32}
         : tensor<256x128xf32, #dot0> * tensor<128x32xf32, #dot1> -> tensor<256x32xf32, #blocked>
    tt.return %r : tensor<256x32xf32, #blocked>
  }
}

// -----

// Two chained dots that each stage 48 KB as they are, within the per-dot limit
// the pass has always used. Both are staged on any target, as before, even
// though they stage 96 KB together. The second dot's A operand is the result
// of the first; staging the first replaces it, so the second dot's operand
// chain has to be found again afterwards.
// The K tile follows the per-thread register budget: 16 for the fp16 dot,
// 8 for the fp32 one.

#blocked  = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#dot0     = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dot1     = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @chained_dots_floor
  // CHECK-COUNT-2: ttg.local_alloc
  // CHECK-COUNT-7: tt.dot {{.*}} : tensor<64x16xf16
  // CHECK: [[Z:%.*]] = tt.dot {{.*}} : tensor<64x16xf16
  // CHECK-DAG: ttg.local_alloc [[Z]] : (tensor<64x128xf32
  // CHECK-DAG: ttg.local_alloc {{%[0-9]+}} : (tensor<128x32xf32
  // CHECK-COUNT-16: tt.dot {{.*}} : tensor<64x8xf32
  // CHECK: tt.return

  // SLM128K-LABEL: @chained_dots_floor
  // SLM128K-COUNT-4: ttg.local_alloc
  // SLM128K: tt.return
  tt.func public @chained_dots_floor(%parg0: tensor<64x128x!tt.ptr<f16>, #blocked>,
                                     %parg1: tensor<128x128x!tt.ptr<f16>, #blocked1>,
                                     %parg2: tensor<128x32x!tt.ptr<f32>, #blocked1>) -> tensor<64x32xf32, #blocked> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked>
    %cst1 = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #blocked>
    %x = tt.load %parg0 : tensor<64x128x!tt.ptr<f16>, #blocked>
    %y = tt.load %parg1 : tensor<128x128x!tt.ptr<f16>, #blocked1>
    %w = tt.load %parg2 : tensor<128x32x!tt.ptr<f32>, #blocked1>
    %xa = ttg.convert_layout %x : tensor<64x128xf16, #blocked> -> tensor<64x128xf16, #dot0>
    %yb = ttg.convert_layout %y : tensor<128x128xf16, #blocked1> -> tensor<128x128xf16, #dot1>
    %z = tt.dot %xa, %yb, %cst {inputPrecision = 2 : i32, maxNumImpreciseAcc = 0 : i32}
         : tensor<64x128xf16, #dot0> * tensor<128x128xf16, #dot1> -> tensor<64x128xf32, #blocked>
    %za = ttg.convert_layout %z : tensor<64x128xf32, #blocked> -> tensor<64x128xf32, #dot0>
    %wb = ttg.convert_layout %w : tensor<128x32xf32, #blocked1> -> tensor<128x32xf32, #dot1>
    %r = tt.dot %za, %wb, %cst1 {inputPrecision = 2 : i32, maxNumImpreciseAcc = 0 : i32}
         : tensor<64x128xf32, #dot0> * tensor<128x32xf32, #dot1> -> tensor<64x32xf32, #blocked>
    tt.return %r : tensor<64x32xf32, #blocked>
  }
}

// -----

// Two chained dots that each stage 96 KB as they are, beyond the per-dot
// limit. With a 128 KB target, a dot may use more only while all the dots
// staged in the function fit in 120 KB together, since a later pass can make
// their buffers live at the same time: only the first dot is staged.

#blocked  = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#dot0     = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dot1     = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @chained_dots_over_budget
  // CHECK-NOT: ttg.local_alloc
  // CHECK: tt.return

  // SLM128K-LABEL: @chained_dots_over_budget
  // SLM128K-COUNT-2: ttg.local_alloc
  // SLM128K-COUNT-16: tt.dot {{.*}} : tensor<64x8xf32
  // SLM128K-NOT: ttg.local_alloc
  // SLM128K: tt.dot {{.*}} : tensor<64x128xf32, #ttg.dot_op<{opIdx = 0
  // SLM128K: tt.return
  tt.func public @chained_dots_over_budget(%parg0: tensor<64x128x!tt.ptr<f32>, #blocked>,
                                           %parg1: tensor<128x128x!tt.ptr<f32>, #blocked1>,
                                           %parg2: tensor<128x128x!tt.ptr<f32>, #blocked1>) -> tensor<64x128xf32, #blocked> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked>
    %x = tt.load %parg0 : tensor<64x128x!tt.ptr<f32>, #blocked>
    %y = tt.load %parg1 : tensor<128x128x!tt.ptr<f32>, #blocked1>
    %w = tt.load %parg2 : tensor<128x128x!tt.ptr<f32>, #blocked1>
    %xa = ttg.convert_layout %x : tensor<64x128xf32, #blocked> -> tensor<64x128xf32, #dot0>
    %yb = ttg.convert_layout %y : tensor<128x128xf32, #blocked1> -> tensor<128x128xf32, #dot1>
    %z = tt.dot %xa, %yb, %cst {inputPrecision = 2 : i32, maxNumImpreciseAcc = 0 : i32}
         : tensor<64x128xf32, #dot0> * tensor<128x128xf32, #dot1> -> tensor<64x128xf32, #blocked>
    %za = ttg.convert_layout %z : tensor<64x128xf32, #blocked> -> tensor<64x128xf32, #dot0>
    %wb = ttg.convert_layout %w : tensor<128x128xf32, #blocked1> -> tensor<128x128xf32, #dot1>
    %r = tt.dot %za, %wb, %cst {inputPrecision = 2 : i32, maxNumImpreciseAcc = 0 : i32}
         : tensor<64x128xf32, #dot0> * tensor<128x128xf32, #dot1> -> tensor<64x128xf32, #blocked>
    tt.return %r : tensor<64x128xf32, #blocked>
  }
}

// -----

// Two fp8 dots in one function. Each stages 24 KB as fp8 (within the per-dot
// limit, so both are staged), 32 KB with A decoded and 48 KB with both decoded.
// With the default 56 KB, decoding is limited by the total of the function:
// the first dot gets A decoded (32 + 24 = 56 KB) and the second stays fp8.
// With a 128 KB target both dots are staged fully decoded (96 KB).

#blocked  = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#dot0     = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dot1     = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @two_fp8_dots_budget
  // CHECK: ttg.local_alloc {{%[0-9]+}} : (tensor<64x128xf16
  // CHECK: ttg.local_alloc {{%[0-9]+}} : (tensor<128x128xf8E5M2
  // CHECK: ttg.local_alloc {{%[0-9]+}} : (tensor<64x128xf8E4M3FN
  // CHECK: ttg.local_alloc {{%[0-9]+}} : (tensor<128x128xf8E5M2
  // CHECK: tt.return

  // FP8-LABEL: @two_fp8_dots_budget
  // FP8: ttg.local_alloc {{%[0-9]+}} : (tensor<64x128xf8E4M3FN
  // FP8: ttg.local_alloc {{%[0-9]+}} : (tensor<128x128xf8E5M2
  // FP8: ttg.local_alloc {{%[0-9]+}} : (tensor<64x128xf8E4M3FN
  // FP8: ttg.local_alloc {{%[0-9]+}} : (tensor<128x128xf8E5M2
  // FP8: tt.return

  // SLM128K-LABEL: @two_fp8_dots_budget
  // SLM128K: ttg.local_alloc {{%[0-9]+}} : (tensor<64x128xf16
  // SLM128K: ttg.local_alloc {{%[0-9]+}} : (tensor<128x128xf16
  // SLM128K: ttg.local_alloc {{%[0-9]+}} : (tensor<64x128xf16
  // SLM128K: ttg.local_alloc {{%[0-9]+}} : (tensor<128x128xf16
  // SLM128K-NOT: xf8E{{[A-Z0-9]+}}, #ttg.dot_op
  // SLM128K: tt.return
  tt.func public @two_fp8_dots_budget(%parg0: tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked>,
                                      %parg1: tensor<128x128x!tt.ptr<f8E5M2>, #blocked1>,
                                      %parg2: tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked>,
                                      %parg3: tensor<128x128x!tt.ptr<f8E5M2>, #blocked1>) -> tensor<64x128xf32, #blocked> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked>
    %a0 = tt.load %parg0 : tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked>
    %b0 = tt.load %parg1 : tensor<128x128x!tt.ptr<f8E5M2>, #blocked1>
    %a1 = tt.load %parg2 : tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked>
    %b1 = tt.load %parg3 : tensor<128x128x!tt.ptr<f8E5M2>, #blocked1>
    %ad0 = tt.fp_to_fp %a0 : tensor<64x128xf8E4M3FN, #blocked> -> tensor<64x128xf16, #blocked>
    %ac0 = ttg.convert_layout %ad0 : tensor<64x128xf16, #blocked> -> tensor<64x128xf16, #dot0>
    %aw0 = tt.fp_to_fp %ac0 : tensor<64x128xf16, #dot0> -> tensor<64x128xf32, #dot0>
    %bd0 = tt.fp_to_fp %b0 : tensor<128x128xf8E5M2, #blocked1> -> tensor<128x128xf16, #blocked1>
    %bc0 = ttg.convert_layout %bd0 : tensor<128x128xf16, #blocked1> -> tensor<128x128xf16, #dot1>
    %bw0 = tt.fp_to_fp %bc0 : tensor<128x128xf16, #dot1> -> tensor<128x128xf32, #dot1>
    %d0 = tt.dot %aw0, %bw0, %cst {inputPrecision = 2 : i32, maxNumImpreciseAcc = 0 : i32}
         : tensor<64x128xf32, #dot0> * tensor<128x128xf32, #dot1> -> tensor<64x128xf32, #blocked>
    %ad1 = tt.fp_to_fp %a1 : tensor<64x128xf8E4M3FN, #blocked> -> tensor<64x128xf16, #blocked>
    %ac1 = ttg.convert_layout %ad1 : tensor<64x128xf16, #blocked> -> tensor<64x128xf16, #dot0>
    %aw1 = tt.fp_to_fp %ac1 : tensor<64x128xf16, #dot0> -> tensor<64x128xf32, #dot0>
    %bd1 = tt.fp_to_fp %b1 : tensor<128x128xf8E5M2, #blocked1> -> tensor<128x128xf16, #blocked1>
    %bc1 = ttg.convert_layout %bd1 : tensor<128x128xf16, #blocked1> -> tensor<128x128xf16, #dot1>
    %bw1 = tt.fp_to_fp %bc1 : tensor<128x128xf16, #dot1> -> tensor<128x128xf32, #dot1>
    %d1 = tt.dot %aw1, %bw1, %d0 {inputPrecision = 2 : i32, maxNumImpreciseAcc = 0 : i32}
         : tensor<64x128xf32, #dot0> * tensor<128x128xf32, #dot1> -> tensor<64x128xf32, #blocked>
    tt.return %d1 : tensor<64x128xf32, #blocked>
  }
}
