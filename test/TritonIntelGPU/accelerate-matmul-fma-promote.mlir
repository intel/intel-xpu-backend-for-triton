// RUN: triton-opt %s -split-input-file --tritonintelgpu-accelerate-matmul | FileCheck %s

// -----

// On the FMA path, an fp8 operand replicated at least 16 times in the dot
// operand layout is decoded before the layout conversion, only to a 16-bit
// type: an fp32 promotion is finished in the dot operand layout. Here B is
// replicated 16 times and A only 4 times, so for fp16 and fp32 A keeps its cast
// in the dot operand layout; an fp8 -> bf16 decode is moved for both.
// CHECK-DAG: #[[$BL:.+]] = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 4]
// CHECK-DAG: #[[$SRC:.+]] = #ttg.blocked<{sizePerThread = [1, 16]
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.min_sg_size" = 16 : i32} {
  // CHECK-LABEL: tt.func public @fp8_to_bf16_fma
  tt.func public @fp8_to_bf16_fma(
    %arg0: tensor<128x64xf8E4M3FN, #blocked1>,
    %arg1: tensor<64x128xf8E4M3FN, #blocked1>,
    %arg2: tensor<128x128x!tt.ptr<bf16>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xbf16, #blocked>
    // CHECK-DAG: [[AD:%.*]] = tt.fp_to_fp %arg0 : tensor<128x64xf8E4M3FN, #[[$SRC]]> -> tensor<128x64xbf16, #[[$SRC]]>
    // CHECK-DAG: [[AC:%.*]] = ttg.convert_layout [[AD]] : tensor<128x64xbf16, #[[$SRC]]> -> tensor<128x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #[[$BL]]}>>
    // CHECK-DAG: [[BD:%.*]] = tt.fp_to_fp %arg1 : tensor<64x128xf8E4M3FN, #[[$SRC]]> -> tensor<64x128xbf16, #[[$SRC]]>
    // CHECK-DAG: [[BC:%.*]] = ttg.convert_layout [[BD]] : tensor<64x128xbf16, #[[$SRC]]> -> tensor<64x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #[[$BL]]}>>
    // CHECK: tt.dot [[AC]], [[BC]], {{.*}} -> tensor<128x128xbf16, #[[$BL]]>
    %a = ttg.convert_layout %arg0 : tensor<128x64xf8E4M3FN, #blocked1> -> tensor<128x64xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %b = ttg.convert_layout %arg1 : tensor<64x128xf8E4M3FN, #blocked1> -> tensor<64x128xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %result = tt.dot %a, %b, %cst : tensor<128x64xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xbf16, #blocked>
    tt.store %arg2, %result : tensor<128x128x!tt.ptr<bf16>, #blocked>
    tt.return
  }

  // CHECK-LABEL: tt.func public @fp8_to_f16_fma
  tt.func public @fp8_to_f16_fma(
    %arg0: tensor<128x64xf8E4M3FN, #blocked1>,
    %arg1: tensor<64x128xf8E5M2, #blocked1>,
    %arg2: tensor<128x128x!tt.ptr<f16>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #blocked>
    // CHECK-DAG: [[A8:%.*]] = ttg.convert_layout %arg0 : tensor<128x64xf8E4M3FN, #[[$SRC]]> -> tensor<128x64xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #[[$BL]]}>>
    // CHECK-DAG: [[A:%.*]] = tt.fp_to_fp [[A8]] : tensor<128x64xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #[[$BL]]}>> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$BL]]}>>
    // CHECK-DAG: [[BD:%.*]] = tt.fp_to_fp %arg1 : tensor<64x128xf8E5M2, #[[$SRC]]> -> tensor<64x128xf16, #[[$SRC]]>
    // CHECK-DAG: [[BC:%.*]] = ttg.convert_layout [[BD]] : tensor<64x128xf16, #[[$SRC]]> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$BL]]}>>
    // CHECK: tt.dot [[A]], [[BC]], {{.*}} -> tensor<128x128xf16, #[[$BL]]>
    %a = ttg.convert_layout %arg0 : tensor<128x64xf8E4M3FN, #blocked1> -> tensor<128x64xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %b = ttg.convert_layout %arg1 : tensor<64x128xf8E5M2, #blocked1> -> tensor<64x128xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %result = tt.dot %a, %b, %cst : tensor<128x64xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf16, #blocked>
    tt.store %arg2, %result : tensor<128x128x!tt.ptr<f16>, #blocked>
    tt.return
  }

  // CHECK-LABEL: tt.func public @fp8_to_f32_fma
  tt.func public @fp8_to_f32_fma(
    %arg0: tensor<128x64xf8E4M3FN, #blocked1>,
    %arg1: tensor<64x128xf8E5M2, #blocked1>,
    %arg2: tensor<128x128x!tt.ptr<f32>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    // CHECK-DAG: [[A8:%.*]] = ttg.convert_layout %arg0 : tensor<128x64xf8E4M3FN, #[[$SRC]]> -> tensor<128x64xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #[[$BL]]}>>
    // CHECK-DAG: [[A:%.*]] = tt.fp_to_fp [[A8]] : tensor<128x64xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #[[$BL]]}>> -> tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #[[$BL]]}>>
    // CHECK-DAG: [[BD:%.*]] = tt.fp_to_fp %arg1 : tensor<64x128xf8E5M2, #[[$SRC]]> -> tensor<64x128xf16, #[[$SRC]]>
    // CHECK-DAG: [[BC:%.*]] = ttg.convert_layout [[BD]] : tensor<64x128xf16, #[[$SRC]]> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$BL]]}>>
    // CHECK-DAG: [[B:%.*]] = tt.fp_to_fp [[BC]] : tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$BL]]}>> -> tensor<64x128xf32, #ttg.dot_op<{opIdx = 1, parent = #[[$BL]]}>>
    // CHECK: tt.dot [[A]], [[B]], {{.*}} -> tensor<128x128xf32, #[[$BL]]>
    %a = ttg.convert_layout %arg0 : tensor<128x64xf8E4M3FN, #blocked1> -> tensor<128x64xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %b = ttg.convert_layout %arg1 : tensor<64x128xf8E5M2, #blocked1> -> tensor<64x128xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %result = tt.dot %a, %b, %cst : tensor<128x64xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf32, #blocked>
    tt.store %arg2, %result : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// With one thread along M, A is replicated 32 times and B not at all.
// CHECK-DAG: #[[$BL:.+]] = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 16]
// CHECK-DAG: #[[$SRC:.+]] = #ttg.blocked<{sizePerThread = [1, 16]
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 16], warpsPerCTA = [1, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [2, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.min_sg_size" = 16 : i32} {
  // CHECK-LABEL: tt.func public @fp8_to_f16_fma_a_replicated
  tt.func public @fp8_to_f16_fma_a_replicated(
    %arg0: tensor<128x64xf8E5M2, #blocked1>,
    %arg1: tensor<64x128xf8E5M2, #blocked1>,
    %arg2: tensor<128x128x!tt.ptr<f16>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #blocked>
    // CHECK-DAG: [[AD:%.*]] = tt.fp_to_fp %arg0 : tensor<128x64xf8E5M2, #[[$SRC]]> -> tensor<128x64xf16, #[[$SRC]]>
    // CHECK-DAG: [[AC:%.*]] = ttg.convert_layout [[AD]] : tensor<128x64xf16, #[[$SRC]]> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$BL]]}>>
    // CHECK-DAG: [[B8:%.*]] = ttg.convert_layout %arg1 : tensor<64x128xf8E5M2, #[[$SRC]]> -> tensor<64x128xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #[[$BL]]}>>
    // CHECK-DAG: [[B:%.*]] = tt.fp_to_fp [[B8]] : tensor<64x128xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #[[$BL]]}>> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$BL]]}>>
    // CHECK: tt.dot [[AC]], [[B]], {{.*}} -> tensor<128x128xf16, #[[$BL]]>
    %a = ttg.convert_layout %arg0 : tensor<128x64xf8E5M2, #blocked1> -> tensor<128x64xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %b = ttg.convert_layout %arg1 : tensor<64x128xf8E5M2, #blocked1> -> tensor<64x128xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %result = tt.dot %a, %b, %cst : tensor<128x64xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf16, #blocked>
    tt.store %arg2, %result : tensor<128x128x!tt.ptr<f16>, #blocked>
    tt.return
  }

  // CHECK-LABEL: tt.func public @fp8_to_f32_fma_a_replicated
  tt.func public @fp8_to_f32_fma_a_replicated(
    %arg0: tensor<128x64xf8E4M3FN, #blocked1>,
    %arg1: tensor<64x128xf8E5M2, #blocked1>,
    %arg2: tensor<128x128x!tt.ptr<f32>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    // CHECK-DAG: [[AD:%.*]] = tt.fp_to_fp %arg0 : tensor<128x64xf8E4M3FN, #[[$SRC]]> -> tensor<128x64xf16, #[[$SRC]]>
    // CHECK-DAG: [[AC:%.*]] = ttg.convert_layout [[AD]] : tensor<128x64xf16, #[[$SRC]]> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$BL]]}>>
    // CHECK-DAG: [[A:%.*]] = tt.fp_to_fp [[AC]] : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$BL]]}>> -> tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #[[$BL]]}>>
    // CHECK-DAG: [[B8:%.*]] = ttg.convert_layout %arg1 : tensor<64x128xf8E5M2, #[[$SRC]]> -> tensor<64x128xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #[[$BL]]}>>
    // CHECK-DAG: [[B:%.*]] = tt.fp_to_fp [[B8]] : tensor<64x128xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #[[$BL]]}>> -> tensor<64x128xf32, #ttg.dot_op<{opIdx = 1, parent = #[[$BL]]}>>
    // CHECK: tt.dot [[A]], [[B]], {{.*}} -> tensor<128x128xf32, #[[$BL]]>
    %a = ttg.convert_layout %arg0 : tensor<128x64xf8E4M3FN, #blocked1> -> tensor<128x64xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %b = ttg.convert_layout %arg1 : tensor<64x128xf8E5M2, #blocked1> -> tensor<64x128xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %result = tt.dot %a, %b, %cst : tensor<128x64xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf32, #blocked>
    tt.store %arg2, %result : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// Other operand types keep their cast in the dot operand layout, even when
// replicated: their cast is cheap.
// CHECK-DAG: #[[$BL:.+]] = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 4]
// CHECK-DAG: #[[$SRC:.+]] = #ttg.blocked<{sizePerThread = [1, 16]
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.min_sg_size" = 16 : i32} {
  // CHECK-LABEL: tt.func public @i8_to_i32_fma
  tt.func public @i8_to_i32_fma(
    %arg0: tensor<128x64xi8, #blocked1>,
    %arg1: tensor<64x128xi8, #blocked1>,
    %arg2: tensor<128x128x!tt.ptr<i32>, #blocked>) {
    %cst = arith.constant dense<0> : tensor<128x128xi32, #blocked>
    // CHECK-DAG: [[A8:%.*]] = ttg.convert_layout %arg0 : tensor<128x64xi8, #[[$SRC]]> -> tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #[[$BL]]}>>
    // CHECK-DAG: [[A:%.*]] = arith.extsi [[A8]] : tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #[[$BL]]}>> to tensor<128x64xi32, #ttg.dot_op<{opIdx = 0, parent = #[[$BL]]}>>
    // CHECK-DAG: [[B8:%.*]] = ttg.convert_layout %arg1 : tensor<64x128xi8, #[[$SRC]]> -> tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #[[$BL]]}>>
    // CHECK-DAG: [[B:%.*]] = arith.extsi [[B8]] : tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #[[$BL]]}>> to tensor<64x128xi32, #ttg.dot_op<{opIdx = 1, parent = #[[$BL]]}>>
    // CHECK: tt.dot [[A]], [[B]], {{.*}} -> tensor<128x128xi32, #[[$BL]]>
    %a = ttg.convert_layout %arg0 : tensor<128x64xi8, #blocked1> -> tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %b = ttg.convert_layout %arg1 : tensor<64x128xi8, #blocked1> -> tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %result = tt.dot %a, %b, %cst : tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xi32, #blocked>
    tt.store %arg2, %result : tensor<128x128x!tt.ptr<i32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: tt.func public @f16_to_f32_fma
  tt.func public @f16_to_f32_fma(
    %arg0: tensor<128x64xf16, #blocked1>,
    %arg1: tensor<64x128xf16, #blocked1>,
    %arg2: tensor<128x128x!tt.ptr<f32>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    // CHECK-DAG: [[A8:%.*]] = ttg.convert_layout %arg0 : tensor<128x64xf16, #[[$SRC]]> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$BL]]}>>
    // CHECK-DAG: [[A:%.*]] = tt.fp_to_fp [[A8]] : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$BL]]}>> -> tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #[[$BL]]}>>
    // CHECK-DAG: [[B8:%.*]] = ttg.convert_layout %arg1 : tensor<64x128xf16, #[[$SRC]]> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$BL]]}>>
    // CHECK-DAG: [[B:%.*]] = tt.fp_to_fp [[B8]] : tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$BL]]}>> -> tensor<64x128xf32, #ttg.dot_op<{opIdx = 1, parent = #[[$BL]]}>>
    // CHECK: tt.dot [[A]], [[B]], {{.*}} -> tensor<128x128xf32, #[[$BL]]>
    %a = ttg.convert_layout %arg0 : tensor<128x64xf16, #blocked1> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %b = ttg.convert_layout %arg1 : tensor<64x128xf16, #blocked1> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %result = tt.dot %a, %b, %cst : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf32, #blocked>
    tt.store %arg2, %result : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// With native fp8 conversion (ttig.support_f8_conversion), fp8 -> fp16 is a
// single instruction, so even a replicated A keeps an fp16 or fp32 promotion in
// the dot operand layout; fp8 -> bf16 is still emulated and is moved.
// CHECK-DAG: #[[$BL:.+]] = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 16]
// CHECK-DAG: #[[$SRC:.+]] = #ttg.blocked<{sizePerThread = [1, 16]
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 16], warpsPerCTA = [1, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [2, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.min_sg_size" = 16 : i32, "ttig.support_f8_conversion"} {
  // CHECK-LABEL: tt.func public @native_f8_conversion_f16
  tt.func public @native_f8_conversion_f16(
    %arg0: tensor<128x64xf8E4M3FN, #blocked1>,
    %arg1: tensor<64x128xf8E4M3FN, #blocked1>,
    %arg2: tensor<128x128x!tt.ptr<f16>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #blocked>
    // CHECK-DAG: [[A8:%.*]] = ttg.convert_layout %arg0 : tensor<128x64xf8E4M3FN, #[[$SRC]]> -> tensor<128x64xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #[[$BL]]}>>
    // CHECK-DAG: tt.fp_to_fp [[A8]] : tensor<128x64xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #[[$BL]]}>> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$BL]]}>>
    // CHECK-NOT: tt.fp_to_fp %arg0
    %a = ttg.convert_layout %arg0 : tensor<128x64xf8E4M3FN, #blocked1> -> tensor<128x64xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %b = ttg.convert_layout %arg1 : tensor<64x128xf8E4M3FN, #blocked1> -> tensor<64x128xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %result = tt.dot %a, %b, %cst : tensor<128x64xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf16, #blocked>
    tt.store %arg2, %result : tensor<128x128x!tt.ptr<f16>, #blocked>
    tt.return
  }

  // CHECK-LABEL: tt.func public @native_f8_conversion_f32
  tt.func public @native_f8_conversion_f32(
    %arg0: tensor<128x64xf8E4M3FN, #blocked1>,
    %arg1: tensor<64x128xf8E4M3FN, #blocked1>,
    %arg2: tensor<128x128x!tt.ptr<f32>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    // CHECK-DAG: [[A8:%.*]] = ttg.convert_layout %arg0 : tensor<128x64xf8E4M3FN, #[[$SRC]]> -> tensor<128x64xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #[[$BL]]}>>
    // CHECK-DAG: tt.fp_to_fp [[A8]] : tensor<128x64xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #[[$BL]]}>> -> tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #[[$BL]]}>>
    // CHECK-NOT: tt.fp_to_fp %arg0
    %a = ttg.convert_layout %arg0 : tensor<128x64xf8E4M3FN, #blocked1> -> tensor<128x64xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %b = ttg.convert_layout %arg1 : tensor<64x128xf8E4M3FN, #blocked1> -> tensor<64x128xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %result = tt.dot %a, %b, %cst : tensor<128x64xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf32, #blocked>
    tt.store %arg2, %result : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: tt.func public @native_f8_conversion_bf16
  tt.func public @native_f8_conversion_bf16(
    %arg0: tensor<128x64xf8E4M3FN, #blocked1>,
    %arg1: tensor<64x128xf8E4M3FN, #blocked1>,
    %arg2: tensor<128x128x!tt.ptr<bf16>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xbf16, #blocked>
    // CHECK-DAG: [[AD:%.*]] = tt.fp_to_fp %arg0 : tensor<128x64xf8E4M3FN, #[[$SRC]]> -> tensor<128x64xbf16, #[[$SRC]]>
    // CHECK-DAG: ttg.convert_layout [[AD]] : tensor<128x64xbf16, #[[$SRC]]> -> tensor<128x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #[[$BL]]}>>
    %a = ttg.convert_layout %arg0 : tensor<128x64xf8E4M3FN, #blocked1> -> tensor<128x64xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %b = ttg.convert_layout %arg1 : tensor<64x128xf8E4M3FN, #blocked1> -> tensor<64x128xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %result = tt.dot %a, %b, %cst : tensor<128x64xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xbf16, #blocked>
    tt.store %arg2, %result : tensor<128x128x!tt.ptr<bf16>, #blocked>
    tt.return
  }
}
