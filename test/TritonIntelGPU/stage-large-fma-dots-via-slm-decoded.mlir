// RUN: triton-opt %s -split-input-file --tritonintelgpu-stage-large-fma-dots-via-slm | FileCheck %s
// RUN: triton-opt %s -split-input-file --tritonintelgpu-stage-large-fma-dots-via-slm="stage-decoded=false" | FileCheck %s --check-prefix=FP8
// RUN: triton-opt %s -split-input-file --tritonintelgpu-stage-large-fma-dots-via-slm="slm-size=131072" | FileCheck %s --check-prefix=SLM128K

// fp8 operands that AccelerateMatmul decoded to fp16 before the layout
// conversion (an fp32 dot: the fp16 -> fp32 cast is after the conversion).
// The decoded fp16 values (16 KB + 32 KB) fit in the 56 KB cap, so they are
// staged and only fp16 -> fp32 is replayed per K tile. With stage-decoded=false
// the fp8 values are staged and the decode is replayed per K tile.

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
  // CHECK: ttg.local_load {{.*}} -> tensor<64x{{[0-9]+}}xf16, #ttg.dot_op<{opIdx = 0
  // CHECK: ttg.local_load {{.*}} -> tensor<{{[0-9]+}}x128xf16, #ttg.dot_op<{opIdx = 1
  // CHECK-NOT: f8E
  // CHECK: tt.fp_to_fp {{.*}}xf16, #ttg.dot_op<{opIdx = 0{{.*}}xf32
  // CHECK: tt.fp_to_fp {{.*}}xf16, #ttg.dot_op<{opIdx = 1{{.*}}xf32
  // CHECK: tt.dot
  // CHECK-NOT: f8E
  // CHECK: tt.return

  // FP8-LABEL: @fp8_decoded_staged
  // FP8-DAG: ttg.local_alloc {{%[0-9]+}} : (tensor<64x128xf8E4M3FN
  // FP8-DAG: ttg.local_alloc {{%[0-9]+}} : (tensor<128x128xf8E5M2
  // FP8: ttg.local_load {{.*}}xf8E4M3FN, #ttg.dot_op<{opIdx = 0
  // FP8: ttg.local_load {{.*}}xf8E5M2, #ttg.dot_op<{opIdx = 1
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
  // CHECK: ttg.local_load {{.*}}xbf16, #ttg.dot_op<{opIdx = 0
  // CHECK: ttg.local_load {{.*}}xf8E4M3FN, #ttg.dot_op<{opIdx = 1
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

// A chain of dots: the second dot's A operand is the result of the first. Each
// stages 96 KB, so they are staged only with a 128 KB target. Staging the first
// dot replaces its result, so the second dot's operand chain has to be found
// after that, on the dots that replace it.

#blocked  = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#dot0     = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dot1     = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @chained_dots
  // CHECK-NOT: ttg.local_alloc
  // CHECK: tt.return

  // SLM128K-LABEL: @chained_dots
  // SLM128K: [[A0:%.*]] = ttg.local_alloc {{%[0-9]+}} : (tensor<64x128xf32
  // SLM128K: [[B0:%.*]] = ttg.local_alloc {{%[0-9]+}} : (tensor<128x128xf32
  // SLM128K: [[C0:%.*]] = tt.dot {{.*}} -> tensor<64x128xf32, #blocked>
  // SLM128K: [[C1:%.*]] = tt.dot {{.*}} -> tensor<64x128xf32, #blocked>
  // SLM128K: [[C2:%.*]] = tt.dot {{.*}} -> tensor<64x128xf32, #blocked>
  // SLM128K: [[C3:%.*]] = tt.dot {{.*}} -> tensor<64x128xf32, #blocked>
  // SLM128K: ttg.local_alloc [[C3]] : (tensor<64x128xf32
  // SLM128K: tt.dot
  // SLM128K: tt.return
  tt.func public @chained_dots(%parg0: tensor<64x128x!tt.ptr<f32>, #blocked>,
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
